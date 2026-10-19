"""Dispatch unchanged archive GP or independent20 on identical chain blocks."""
import time
import archive_engine as core
legacy=core.legacy


def run_chain(prepared,rows,seed,arm):
    if arm!='independent20':
        return core.run_chain(prepared,rows,seed,'elite_archive')
    start,cpu=time.perf_counter(),time.process_time()
    records=[]
    for row in rows:
        tid=int(row['query_id'])
        record=core.run_task(prepared,tid,seed,'prefix_transfer',None)
        record.pop('population')
        record.update(parent=-1,similarity=None,eligible=False,
                      chain_id=int(row['chain_id']),order=int(row['order']))
        records.append(record)
    return dict(chain_id=int(rows[0]['chain_id']),rows=records,
                chain_seconds=time.perf_counter()-start,chain_cpu_seconds=time.process_time()-cpu)


def worker_identity():
    return core.worker_identity()
