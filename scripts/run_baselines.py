"""Run the eight external baselines using their recorded tuning implementations."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from baselines.core import BenchmarkConfig, load_dataset
from baselines.global_models import GLOBAL_MODEL_RUNNERS
from baselines.spatial_models import SPATIAL_MODEL_RUNNERS


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [safe(v) for v in value]
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'data/processed/heat_flow_stations.csv')
    parser.add_argument('--output',type=Path,default=ROOT/'run_outputs/external_baselines')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=BenchmarkConfig(station_file=args.input,n_jobs=args.workers)
    data=load_dataset(config)
    frames={}
    runners={**GLOBAL_MODEL_RUNNERS,**SPATIAL_MODEL_RUNNERS}
    for model,runner in runners.items():
        seeds=(42,123,456) if model in ('random_forest','gradient_boosting','xgboost','gwrf') else (42,)
        predictions=[]
        for seed in seeds:
            result=runner(data,replace(config,model_seed=seed))
            directory=args.output/model/f'seed{seed}';directory.mkdir(parents=True)
            pred=np.asarray(result.prediction_heat_flow)
            predictions.append(pred)
            pd.DataFrame(dict(row_id=data.test_original_indices,q_obs=data.y_test,q_pred=pred)).to_csv(directory/'predictions.csv',index=False)
            (directory/'metadata.json').write_text(json.dumps(safe(dict(selected_parameters=result.selected_parameters,
                complexity=result.complexity,efficiency=result.efficiency,notes=result.notes)),indent=2,ensure_ascii=False),encoding='utf-8')
            result.tuning_results.to_csv(directory/'tuning_results.csv',index=False)
            print('completed',model,seed,flush=True)
        frames[f'q_pred__{model}']=np.mean(predictions,axis=0)
    pd.DataFrame(dict(row_id=data.test_original_indices,lon=data.coordinates_test[:,0],lat=data.coordinates_test[:,1],
        q_obs=data.y_test,**frames)).to_csv(args.output/'ensemble_predictions.csv',index=False)


if __name__=='__main__':main()
