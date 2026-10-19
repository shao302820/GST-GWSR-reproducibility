"""Portable current workflow. Training runs only after an explicit stage command."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/gwsr/exact'))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold, train_test_split
import archive_engine
import grid_engine
import paper_gp
import preparation
import select_alpha_tau_exact as association

SEEDS = (42, 123, 456)
K_VALUES = tuple(range(20, 61, 5))
ALPHAS = (0., .25, .5, .75, 1.)
TAUS = (.5, .6, .7, .8)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False, encoding='utf-8-sig')


def fresh(path):
    path.mkdir(parents=True, exist_ok=False)


def split_data(path):
    data = pd.read_csv(path)
    assert len(data) == 1024
    assert np.isfinite(data[['lon', 'lat', 'heat_flow', *association.base.FEATURES]]).all().all()
    assert (data.heat_flow > 0).all()
    data['row_id'] = np.arange(len(data))
    train, test = train_test_split(np.arange(len(data)), test_size=.2, random_state=42)
    return data.iloc[train].copy().reset_index(drop=True), data.iloc[test].copy().reset_index(drop=True)


def fold_data(train, k):
    for fold, (fit, validation) in enumerate(KFold(5, shuffle=True, random_state=42).split(train), 1):
        raw = train.iloc[fit].copy()
        held = train.iloc[validation].copy().reset_index(drop=True)
        prepared = preparation.prepare_fold(raw, held.drop(columns=['heat_flow']), k)
        yield fold, prepared, held.heat_flow.to_numpy()


def independent_point(prepared, task_id, seed, k):
    i, scaled, expression, local_r2, info = paper_gp.predict_one_point(
        task_id, prepared['xfit'], prepared['yfit'], prepared['xquery'][task_id],
        prepared['distances'][task_id], k, seed)
    if info['failure_reason']:
        raise RuntimeError(info['failure_reason'])
    pred = float(paper_gp.inverse_transform(np.array([scaled]), prepared['ys'])[0])
    return dict(task_id=i, prediction=pred, expression=expression,
                local_fit_R2=local_r2, **info)


def seed_frame(prepared, observed, records):
    records = sorted(records, key=lambda row: row['task_id'])
    assert [r['task_id'] for r in records] == list(range(len(observed)))
    frame = prepared['query'][['row_id', 'lon', 'lat']].copy()
    frame['q_obs'] = observed
    frame['q_pred'] = [r['prediction'] for r in records]
    frame['expression_exact'] = [r['expression_exact'] for r in records]
    assert np.isfinite(frame.q_pred).all()
    return frame, records


def ensemble(frames):
    assert len(frames) == 3
    result = frames[0][['row_id', 'lon', 'lat', 'q_obs']].copy()
    for seed, frame in zip(SEEDS, frames):
        np.testing.assert_array_equal(result.row_id, frame.row_id)
        np.testing.assert_array_equal(result.q_obs, frame.q_obs)
        result[f'q_pred_seed{seed}'] = frame.q_pred.to_numpy()
    result['q_pred'] = np.mean([f.q_pred.to_numpy() for f in frames], axis=0)
    return result


def k_selection(train, target, pool):
    fresh(target)
    summary = []
    for k in K_VALUES:
        folds = []
        for fold, prepared, observed in fold_data(train, k):
            write_json(target / f'fold{fold}_preprocessing.json', prepared['preprocessing'])
            frames = []
            for seed in SEEDS:
                rows = pool(delayed(independent_point)(prepared, i, seed, k) for i in range(len(observed)))
                frame, rows = seed_frame(prepared, observed, rows)
                write_csv(target / f'k{k}/fold{fold}_seed{seed}.csv', frame)
                write_json(target / f'k{k}/fold{fold}_seed{seed}_histories.json', rows)
                frames.append(frame)
            mean = ensemble(frames)
            write_csv(target / f'k{k}/ensemble_fold{fold}.csv', mean)
            folds.append(mean)
        pooled = pd.concat(folds, ignore_index=True)
        assert len(pooled) == pooled.row_id.nunique() == 819
        write_csv(target / f'ensemble_oof_k{k}.csv', pooled)
        summary.append(dict(k=k, **association.metrics(pooled.q_obs, pooled.q_pred)))
        print('k_complete', k, summary[-1], flush=True)
    selected = sorted(summary, key=lambda r: (r['RMSE'], r['k']))[0]
    write_csv(target / 'cv_ensemble_k_summary.csv', pd.DataFrame(summary))
    write_json(target / 'selected_k.json', dict(selected_k=selected['k'], selected=selected,
        selection_rule='minimum pooled819 original-scale three-seed prediction-mean ensemble RMSE; ties smaller k'))
    return selected['k']


def plan(prepared, alpha, tau):
    sg, sx = association.similarity_matrices(prepared)
    rows = association.build_order(alpha * sg + (1 - alpha) * sx,
        prepared['query'][association.base.COORDS].to_numpy(), tau)
    groups = [[r for r in rows if r['chain_id'] == cid] for cid in range(24)]
    assert all(groups)
    return groups


def chain_batch(pool, prepared, observed, groups, arm, seed, directory, stem):
    start = time.perf_counter()
    chunks = pool(delayed(grid_engine.run_chain)(prepared, group, seed, arm) for group in groups)
    seconds = time.perf_counter() - start
    rows = [r for chunk in chunks for r in chunk['rows']]
    frame, rows = seed_frame(prepared, observed, rows)
    for r in rows:
        assert r['generations_executed'] == (5 if r['transferred'] else 20)
        assert r['stop_reason'] == 'fixed_budget'
    for key in ('transferred', 'generations_executed', 'best_generation', 'best_objective',
                'explicit_objective_evaluations', 'search_seconds', 'archive_management_seconds',
                'incoming_duplicate_fill', 'parent', 'chain_id', 'order'):
        frame[key] = [r[key] for r in rows]
    meta = dict(arm=arm, seed=seed, n=len(observed), **association.metrics(frame.q_obs, frame.q_pred),
        batch_wall_seconds=seconds, points5=sum(r['transferred'] for r in rows),
        points20=sum(not r['transferred'] for r in rows),
        explicit_objective_evaluations=sum(r['explicit_objective_evaluations'] for r in rows),
        archive_management_seconds=sum(r['archive_management_seconds'] for r in rows))
    write_csv(directory / (stem + '.csv'), frame)
    write_json(directory / (stem + '_histories.json'), rows)
    write_json(directory / (stem + '.json'), meta)
    print('batch_complete', directory.name, stem, flush=True)
    return frame, meta


def gst_selection(train, target, pool, k):
    fresh(target)
    candidates = []
    arms = ('independent20',) + tuple(f'alpha_{a:g}' for a in ALPHAS)
    for tau in TAUS:
        stage = target / f'tau{round(tau * 10):02d}'
        fresh(stage)
        ensembles = {arm: [] for arm in arms}
        metadata = []
        for fold, prepared, observed in fold_data(train, k):
            write_json(stage / f'fold{fold}_preprocessing.json', prepared['preprocessing'])
            plans = {arm: plan(prepared, .75 if arm == 'independent20' else float(arm[6:]), tau) for arm in arms}
            per_arm = {arm: [] for arm in arms}
            for si, seed in enumerate(SEEDS):
                offset = ((fold - 1) * 3 + si) % len(arms)
                for arm in arms[offset:] + arms[:offset]:
                    frame, info = chain_batch(pool, prepared, observed, plans[arm],
                        'independent20' if arm == 'independent20' else 'elite_archive', seed,
                        stage / arm, f'fold{fold}_seed{seed}')
                    info.update(arm=arm, fold=fold, tau=tau)
                    metadata.append(info)
                    per_arm[arm].append(frame)
            for arm in arms:
                mean = ensemble(per_arm[arm])
                write_csv(stage / arm / f'ensemble_fold{fold}.csv', mean)
                ensembles[arm].append(mean)
        summary = []
        for arm in arms:
            pooled = pd.concat(ensembles[arm], ignore_index=True)
            assert len(pooled) == pooled.row_id.nunique() == 819
            write_csv(stage / arm / 'ensemble_oof.csv', pooled)
            row = dict(arm=arm, tau=tau, alpha=None if arm == 'independent20' else float(arm[6:]),
                       **association.metrics(pooled.q_obs, pooled.q_pred))
            summary.append(row)
            if arm != 'independent20':
                candidates.append(row)
        write_csv(stage / 'cv_summary.csv', pd.DataFrame(summary))
        write_csv(stage / 'cv_seed_fold_metrics.csv', pd.DataFrame(metadata))
        write_json(stage / 'completion.json', dict(completed_batches=90, ensemble_folds=30,
                   task_histories=14742, tau=tau))
    assert len(candidates) == 20
    selected = sorted(candidates, key=lambda r: (r['RMSE'], r['alpha'], r['tau']))[0]
    write_csv(target / 'cv_candidate_summary.csv', pd.DataFrame(candidates))
    write_json(target / 'selected_alpha_tau.json', dict(k=k, alpha=selected['alpha'], tau=selected['tau'],
        selected_cv=selected, candidate_count=20,
        rule='minimum pooled819 original-scale prediction-mean ensemble RMSE; exact ties alpha then tau'))
    return selected


def outer_test(train, held, target, pool, selected):
    fresh(target)
    k, alpha, tau = int(selected['k']), float(selected['alpha']), float(selected['tau'])
    write_json(target / 'parameter_lock.json', selected)
    prepared = preparation.prepare_fold(train, held.drop(columns=['heat_flow']), k)
    groups = plan(prepared, alpha, tau)
    write_json(target / 'training_preprocessing.json', prepared['preprocessing'])
    schedule = ((42, 'independent20'), (42, 'elite_archive'), (123, 'elite_archive'),
                (123, 'independent20'), (456, 'independent20'), (456, 'elite_archive'))
    frames, metadata = {}, []
    for seed, arm in schedule:
        frame, info = chain_batch(pool, prepared, held.heat_flow.to_numpy(), groups, arm, seed,
                                 target / arm, f'seed{seed}')
        frames[arm, seed] = frame
        metadata.append(info)
    summary = []
    for arm in ('independent20', 'elite_archive'):
        mean = ensemble([frames[arm, seed] for seed in SEEDS])
        write_csv(target / arm / 'ensemble.csv', mean)
        summary.append(dict(arm=arm, **association.metrics(mean.q_obs, mean.q_pred),
            batch_wall_seconds=sum(m['batch_wall_seconds'] for m in metadata if m['arm'] == arm)))
    write_csv(target / 'test_seed_metrics.csv', pd.DataFrame(metadata))
    write_csv(target / 'test_ensemble_metrics.csv', pd.DataFrame(summary))
    write_json(target / 'completion.json', dict(completed_batches=6, ensembles=2, task_histories=1230))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('k-cv', 'gst-cv', 'outer', 'all'))
    parser.add_argument('--input', type=Path, default=ROOT / 'data/processed/heat_flow_stations.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'run_outputs/current')
    parser.add_argument('--selection-root', type=Path, help='directory containing k_selection/ and gst_cv/ for standalone stages')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    import importlib.metadata as md
    assert md.version('gplearn') == '0.4.3'
    train, held = split_data(args.input)
    selections = args.selection_root or args.output
    args.output.mkdir(parents=True, exist_ok=True)
    # Each stage refuses to replace an existing directory; recorded results are read-only inputs.
    with parallel_config(backend='loky', inner_max_num_threads=1):
        with Parallel(n_jobs=args.workers) as pool:
            pool(delayed(grid_engine.worker_identity)() for _ in range(args.workers * 3))
            if args.stage in ('k-cv', 'all'):
                k_selection(train, args.output / 'k_selection', pool)
            if args.stage in ('gst-cv', 'all'):
                k = json.loads((selections / 'k_selection/selected_k.json').read_text(encoding='utf-8-sig'))['selected_k']
                gst_selection(train, args.output / 'gst_cv', pool, k)
            if args.stage in ('outer', 'all'):
                selected = json.loads((selections / 'gst_cv/selected_alpha_tau.json').read_text(encoding='utf-8-sig'))
                outer_test(train, held, args.output / 'final_outer', pool, selected)


if __name__ == '__main__':
    main()
