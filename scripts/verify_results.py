"""Read-only verification of current recorded predictions, histories and selection."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results'
SEEDS=(42,123,456)


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def metrics(frame):
    y,p=frame.q_obs.to_numpy(),frame.q_pred.to_numpy()
    return dict(R2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),
                RMSE=float(np.sqrt(np.mean((y-p)**2))),MAE=float(np.mean(np.abs(y-p))))


def compare_metrics(frame,row):
    for k,v in metrics(frame).items():np.testing.assert_allclose(v,row[k],rtol=1e-12,atol=1e-12)


def check_ensemble(directory,fold,n):
    stem=f'fold{fold}_seed' if fold else 'seed'
    frames=[pd.read_csv(directory/f'{stem}{s}.csv') for s in SEEDS]
    result=pd.read_csv(directory/(f'ensemble_fold{fold}.csv' if fold else 'ensemble.csv'))
    assert len(result)==result.row_id.nunique()==n
    for f in frames:
        np.testing.assert_array_equal(f.row_id,result.row_id)
        np.testing.assert_array_equal(f.q_obs,result.q_obs)
    np.testing.assert_allclose(result.q_pred,np.mean([f.q_pred.to_numpy() for f in frames],axis=0),rtol=0,atol=1e-12)
    return result


def histories(path):
    rows=read(path)
    for r in rows:
        transfer=r['transferred'];g=5 if transfer else 20
        assert not r['failed'] and r['generations_executed']==g and r['stop_reason']=='fixed_budget'
        assert r['search_entry_normalizations']==1 and abs(r['weights_sum_after']-1)<1e-14
        assert [t['generation'] for t in r['trace']]==(list(range(6)) if transfer else list(range(1,21)))
        np.testing.assert_array_equal(r['best_so_far_objectives'],np.minimum.accumulate(r['generation_min_objectives']))
        assert r['explicit_objective_evaluations']==(6000 if transfer else 20000)
        if transfer:assert r['incoming_unique_archive_candidates']+r['incoming_duplicate_fill']==800
        if r['archive_enabled']:
            assert all(1<=t['archive']['size']<=1000 for t in r['trace'])
    return len(rows)


def main():
    provenance=read(RESULTS/'source_provenance.json')
    assert hashlib.sha256((ROOT/'data/processed/heat_flow_stations.csv').read_bytes()).hexdigest()==provenance['station_sha256']
    for relative,entry in provenance['copied_artifacts'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==entry['sha256'],relative
    candidates=[];total_batches=total_ensembles=total_histories=0
    for code in ('05','06','07','08'):
        stage=RESULTS/'gst_cv'/f'tau{code}'
        summary=pd.read_csv(stage/'cv_summary.csv')
        batches=ensembles=count=0
        for arm in ('independent20','alpha_0','alpha_0.25','alpha_0.5','alpha_0.75','alpha_1'):
            folds=[]
            for fold in range(1,6):
                folds.append(check_ensemble(stage/arm,fold,164 if fold<5 else 163));ensembles+=1
                for seed in SEEDS:
                    csv=stage/arm/f'fold{fold}_seed{seed}.csv'
                    frame=pd.read_csv(csv)
                    meta=read(csv.with_suffix('.json'))
                    compare_metrics(frame,meta)
                    count+=histories(csv.with_name(csv.stem+'_histories.json'));batches+=1
                    if arm=='independent20':
                        reference=pd.read_csv(RESULTS/'k_selection/seed_folds'/f'cv_k55_seed{seed}_fold{fold}.csv').set_index('row_id').loc[frame.row_id]
                        np.testing.assert_allclose(frame.q_pred,reference.q_pred,rtol=0,atol=1e-10)
                        assert frame.expression_exact.tolist()==reference.expression_exact.tolist()
            pooled=pd.concat(folds,ignore_index=True)
            assert len(pooled)==pooled.row_id.nunique()==819
            compare_metrics(pooled,summary[summary.arm==arm].iloc[0])
        assert (batches,ensembles,count)==(90,30,14742)
        total_batches+=batches;total_ensembles+=ensembles;total_histories+=count
        candidates.extend(summary[summary.arm!='independent20'][['alpha','tau','R2','RMSE','MAE']].to_dict('records'))
    assert (total_batches,total_ensembles,total_histories)==(360,120,58968)
    best=sorted(candidates,key=lambda r:(r['RMSE'],r['alpha'],r['tau']))[0]
    assert len(candidates)==20 and (best['alpha'],best['tau'])==(.75,.5)
    locked=read(RESULTS/'gst_cv/selected_alpha_tau.json')
    assert (locked['k'],locked['alpha'],locked['tau'])==(55,.75,.5)
    ksummary=pd.read_csv(RESULTS/'k_selection/cv_ensemble_k_summary.csv')
    assert set(ksummary.k)==set(range(20,61,5))
    for _,row in ksummary.iterrows():
        frame=pd.read_csv(RESULTS/'k_selection'/f'ensemble_oof_k{int(row.k)}.csv')
        assert len(frame)==frame.row_id.nunique()==819
        compare_metrics(frame,row)
        # Verify all nine k candidates against their three original seed-fold predictions.
        for fold in range(1,6):
            seed_frames=[pd.read_csv(RESULTS/'k_selection/seed_folds'/f'cv_k{int(row.k)}_seed{s}_fold{fold}.csv') for s in SEEDS]
            aligned=frame.set_index('row_id').loc[seed_frames[0].row_id]
            np.testing.assert_allclose(aligned.q_pred,np.mean([f.q_pred.to_numpy() for f in seed_frames],axis=0),rtol=0,atol=1e-12)
    assert int(ksummary.sort_values(['RMSE','k']).iloc[0].k)==55
    final=RESULTS/'final_outer';n_histories=0
    lock_sha=hashlib.sha256((final/'parameter_lock.json').read_bytes()).hexdigest()
    final_summary=pd.read_csv(final/'test_ensemble_metrics.csv')
    for arm in ('independent20','elite_archive'):
        frame=check_ensemble(final/arm,None,205)
        compare_metrics(frame,final_summary[final_summary.arm==arm].iloc[0])
        for seed in SEEDS:
            metadata=read(final/arm/f'seed{seed}.json')
            assert metadata['parameter_lock_sha256']==lock_sha
            path=final/arm/f'seed{seed}_histories.json';n_histories+=histories(path)
            rows=read(path)
            assert sum(r['transferred'] for r in rows)==(181 if arm=='elite_archive' else 0)
    assert n_histories==1230
    all_preds=pd.read_csv(RESULTS/'model_comparison/predictions.csv')
    model_metrics=pd.read_csv(RESULTS/'model_comparison/metrics.csv')
    mapping={'OLS':'ols_log','GWR':'gwr','MGWR':'mgwr','GWRF':'gwrf','RF':'random_forest',
             'GBR':'gradient_boosting','XGBoost':'xgboost','SVR-RBF':'svr_rbf','GWSR':'GWSR','GST-GWSR':'GST-GWSR'}
    for name,key in mapping.items():
        frame=pd.DataFrame(dict(q_obs=all_preds.q_obs,q_pred=all_preds[f'q_pred__{key}']))
        compare_metrics(frame,model_metrics[model_metrics.model==name].iloc[0])
    print(json.dumps(dict(passed=True,k_candidates=9,gst_combinations=20,cv_batches=total_batches,
        cv_ensemble_folds=total_ensembles,cv_histories=total_histories,outer_histories=n_histories,
        outer_models=10,selected=dict(k=55,alpha=.75,tau=.5),training_performed=False)))


if __name__=='__main__':main()
