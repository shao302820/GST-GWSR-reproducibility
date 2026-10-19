"""Fixed5/20 archive GP with historical-GWSR search-entry normalization.

The control follows the frozen engine, including initialization RNG, target
reevaluation and generation counting. No archive operation executes programs.
"""
from __future__ import annotations
import copy
from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np
import select_alpha_tau_exact as legacy
from gplearn._program import _Program

POPULATION=1000
ARCHIVE_CAPACITY=1000
ARMS=('prefix_transfer','elite_archive')


def signature(program):
    tokens=[]
    for item in program.program:
        if hasattr(item,'arity'):
            tokens.append(('fn',item.name,str(item.arity)))
        elif isinstance(item,(int,np.integer)):
            tokens.append(('x',str(int(item))))
        else:
            tokens.append(('c',float(item).hex()))
    return tuple(tokens)


class EliteArchive:
    def __init__(self,capacity=ARCHIVE_CAPACITY):
        if capacity<1:raise ValueError('Archive capacity must be positive')
        self.capacity=capacity
        self.entries={}
        self.seconds=0.
        self.seen_candidates=0

    def update(self,population,generation):
        start=time.perf_counter()
        merged=dict(self.entries)
        for p in population:
            self.seen_candidates+=1
            value=float(p.fitness_)
            if not np.isfinite(value):continue
            key=signature(p)
            previous=merged.get(key)
            if previous is None or value<previous[0]:
                # New candidates are copied only if they survive the capacity cut.
                merged[key]=(value,p,int(generation),False)
        ranked=sorted(merged,key=lambda k:(merged[k][0],k))[:self.capacity]
        updated={}
        for key in ranked:
            value,p,g,owned=merged[key]
            updated[key]=(value,p if owned else copy.deepcopy(p),g,True)
        self.entries=updated
        self.seconds+=time.perf_counter()-start

    def pool(self):
        return [entry[1] for entry in self.entries.values()]

    def snapshot(self):
        scores=[e[0] for e in self.entries.values()]
        return dict(size=len(scores),best_score=scores[0] if scores else None,
                    worst_score=scores[-1] if scores else None,
                    management_seconds=self.seconds,seen_candidates=self.seen_candidates)


def take800(pool):
    if not pool:raise RuntimeError('No finite archived source candidate')
    unique=min(len(pool),800)
    selected=pool[:800]
    if len(selected)<800:
        selected=selected+[pool[i%len(pool)] for i in range(800-len(selected))]
    return selected,dict(incoming_unique_archive_candidates=unique,
                         incoming_duplicate_fill=800-unique)


def parameters(seed,generations=20):
    p=legacy.base.build_sr_model(seed).get_params().copy()
    functions=[legacy._function_map[n] for n in p['function_set']]
    arities={}
    for fn in functions:arities.setdefault(fn.arity,[]).append(fn)
    p.update(generations=int(generations),population_size=1000,function_set=functions,
             arities=arities,_metric=legacy._fitness_map['mse'],_transformer=None,max_samples=1.,
             method_probs=np.cumsum([p['p_crossover'],p['p_subtree_mutation'],p['p_hoist_mutation'],p['p_point_mutation']]))
    return p


@contextmanager
def count_executes(counter):
    execute=_Program.execute
    def counted(self,X):
        if len(X) >= 20:
            counter['training_execute_calls']+=1
            counter['training_row_evaluations']+=len(X)
        return execute(self,X)
    _Program.execute=counted
    try:yield
    finally:_Program.execute=execute


def search_population(X,y,w,seed,generations,initial=None,archive_enabled=False):
    params=parameters(seed,generations)
    rng=np.random.RandomState(seed)
    penalty=float(params['parsimony_coefficient'])
    parents=None if initial is None else [copy.deepcopy(p) for p in initial]
    archive=EliteArchive() if archive_enabled else None
    best,best_j,best_generation=None,float('inf'),None
    trace=[];objective_calls=0;invalid=0
    start=time.perf_counter()
    def evaluate(population,generation):
        nonlocal best,best_j,best_generation,objective_calls,invalid
        values=[]
        for p in population:
            j=legacy.objective(p,X,y,w,penalty)
            objective_calls+=1;invalid+=int(not np.isfinite(j))
            p.fitness_,p.raw_fitness_=j,j
            values.append(j)
            if j<best_j:
                best,best_j,best_generation=copy.deepcopy(p),float(j),generation
        if archive is not None:
            archive.update(population,generation)
            if archive.entries and archive.snapshot()['best_score']!=best_j:
                raise RuntimeError('Archive lost the best-so-far source expression')
        trace.append(dict(generation=generation,generation_min=float(min(values)),
                          best_objective=best_j,explicit_objective_evaluations=objective_calls,
                          elapsed_seconds=time.perf_counter()-start,
                          archive=archive.snapshot() if archive is not None else None))
        return bool(np.isfinite(values).any())
    if parents is not None:evaluate(parents,0)
    for gen in range(int(generations)):
        seeds=rng.randint(np.iinfo(np.int32).max,size=1000)
        population=legacy._parallel_evolve(1000,parents,X,y,w,seeds,params)
        if not evaluate(population,gen+1):raise RuntimeError('No finite local objective in a GP generation')
        parents=population
    if best is None:raise RuntimeError('No valid local symbolic expression')
    outgoing=archive.pool() if archive is not None else parents
    info=dict(generations_executed=int(generations),best_generation=best_generation,best_objective=best_j,
              stop_reason='fixed_budget',trace=trace,
              best_so_far_objectives=[r['best_objective'] for r in trace],
              generation_min_objectives=[r['generation_min'] for r in trace],
              explicit_objective_evaluations=objective_calls,invalid_objectives=invalid,
              archive_enabled=archive_enabled,outgoing_candidate_count=len(outgoing),
              archive_management_seconds=archive.seconds if archive is not None else 0.)
    return best,parents,outgoing,info


def run_task(prepared,tid,seed,arm,source_pool=None):
    if arm not in ARMS:raise ValueError('Unknown experiment arm')
    start,cpu_start=time.perf_counter(),time.process_time()
    positions=prepared['positions'][tid]
    X,y,w=prepared['xfit'][positions],prepared['yfit'][positions],prepared['weights'][tid]
    # Match paper_gp.search exactly: copy and re-normalize once at task entry.
    # This same immutable local weight array is used for fresh initialization,
    # inherited-population reevaluation, every generation and prediction guards.
    X,y,w=np.asarray(X),np.asarray(y),np.asarray(w,dtype=float).copy()
    if (not np.isfinite(X).all() or not np.isfinite(y).all()
            or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0):
        raise ValueError('Invalid local training data or weights.')
    weights_sum_before=float(w.sum())
    w/=w.sum()
    weights_sum_after=float(w.sum())
    counter=dict(training_execute_calls=0,training_row_evaluations=0)
    transferred=source_pool is not None
    generations=5 if transferred else 20
    selection=dict(incoming_unique_archive_candidates=None,incoming_duplicate_fill=0)
    with count_executes(counter):
        initial=None
        if transferred:
            fresh_params=parameters(seed+100000+tid)
            fresh_rng=np.random.RandomState(seed+200000+tid)
            fresh=legacy._parallel_evolve(1000,None,X,y,w,
                  fresh_rng.randint(np.iinfo(np.int32).max,size=1000),fresh_params)
            if arm=='elite_archive':source,selection=take800(source_pool)
            else:
                if len(source_pool)!=1000:raise RuntimeError('Control final population size changed')
                source=source_pool[:800]
            initial=[copy.deepcopy(p) for p in source]+[copy.deepcopy(p) for p in fresh[:200]]
        # Generation0 in the trace is the legacy inherited initialization, outside5 loops.
        pre_search_seconds=time.perf_counter()-start
        best,final_population,outgoing,info=search_population(X,y,w,seed+tid,generations,initial,arm=='elite_archive')
        init_seconds=pre_search_seconds+info['trace'][0]['elapsed_seconds']
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        scaled=float(best.execute(prepared['xquery'][tid].reshape(1,-1))[0])
    protected=not np.isfinite(scaled) or abs(scaled)>=5.
    if protected:scaled=float(np.average(y,weights=w))
    log_pred=float(prepared['ys'].inverse_transform(np.asarray([[scaled]]))[0,0])
    pred=float(np.exp(np.clip(log_pred,-5.,10.)))
    exact=legacy.paper_gp.exact_expression(best)
    if not np.isfinite(pred):raise RuntimeError('Nonfinite protected prediction')
    # No source/archive scoring is hidden outside this task's timer.
    return dict(task_id=int(tid),prediction=pred,expression_exact=exact,
                expression=f'CLIPPED({exact[:80]})' if protected else exact,
                protected=bool(protected),failed=False,transferred=transferred,
                initialization_seconds=init_seconds,search_seconds=time.perf_counter()-start,
                search_cpu_seconds=time.process_time()-cpu_start,**info,**counter,**selection,
                search_entry_normalizations=1,weights_sum_before=weights_sum_before,
                weights_sum_after=weights_sum_after,population=outgoing)


def run_chain(prepared,rows,seed,arm):
    start,cpu_start=time.perf_counter(),time.process_time()
    pools,records={},[]
    for row in rows:
        tid,parent=int(row['query_id']),int(row['parent'])
        pool=pools.get(parent) if bool(row['eligible']) else None
        record=run_task(prepared,tid,seed,arm,pool)
        pools[tid]=record.pop('population')
        record.update(parent=parent,similarity=float(row['score']) if np.isfinite(row['score']) else None,
                      eligible=bool(row['eligible']),chain_id=int(row['chain_id']),order=int(row['order']))
        records.append(record)
    return dict(chain_id=int(rows[0]['chain_id']),rows=records,
                chain_seconds=time.perf_counter()-start,chain_cpu_seconds=time.process_time()-cpu_start)


def worker_identity():
    import os
    time.sleep(.1)
    return dict(pid=os.getpid(),full=20,short=5,population=1000,engine=str(Path(__file__).resolve()))
