"""Check scientific preparation and planning against the recorded205 run, without GP."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/gwsr/exact'))
import preparation
import select_alpha_tau_exact as association


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=pd.read_csv(ROOT/'data/processed/heat_flow_stations.csv')
        source['row_id']=np.arange(len(source))
        fit,test=train_test_split(np.arange(1024),test_size=.2,random_state=42)
        cls.prep=preparation.prepare_fold(source.iloc[fit].reset_index(drop=True),
            source.iloc[test].drop(columns=['heat_flow']).reset_index(drop=True))
        cls.reference=np.load(ROOT/'results/final_outer/test_task_information.npz')

    def test_fold_fitted_transformations_match(self):
        recorded=json.loads((ROOT/'results/final_outer/training_preprocessing.json').read_text(encoding='utf-8-sig'))
        self.assertEqual(self.prep['preprocessing'],recorded)
        self.assertEqual(len(self.prep['train']),815)
        self.assertNotIn('heat_flow',self.prep['query'])

    def test_target_point_descriptors_and_weights_match(self):
        np.testing.assert_array_equal(self.prep['descriptors'],self.reference['descriptors'])
        np.testing.assert_array_equal(self.prep['positions'],self.reference['fitting_positions'])
        np.testing.assert_array_equal(self.prep['weights'],self.reference['fitting_weights'])
        self.assertEqual(self.prep['descriptors'].shape,(205,8))

    def test_haversine_against_independent_implementation(self):
        from sklearn.metrics.pairwise import haversine_distances
        query=self.prep['query'][['lat','lon']].to_numpy()
        fit=self.prep['train'][['lat','lon']].to_numpy()
        np.testing.assert_allclose(self.prep['distances'],
            haversine_distances(np.radians(query),np.radians(fit))*6371.0088,rtol=1e-12,atol=1e-9)

    def test_similarities_and_full_plan_match(self):
        geographic,feature=association.similarity_matrices(self.prep)
        np.testing.assert_array_equal(geographic,self.reference['geographic_similarity'])
        np.testing.assert_array_equal(feature,self.reference['feature_similarity'])
        plan=pd.DataFrame(association.build_order(.75*geographic+.25*feature,
            self.prep['query'][['lon','lat']].to_numpy(),.5))
        reference=pd.read_csv(ROOT/'results/final_outer/planned_search_branches.csv')
        for field in ('chain_id','order','query_id','parent','is_anchor','eligible'):
            np.testing.assert_array_equal(plan[field],reference[field])
        np.testing.assert_allclose(plan.score,reference.score,rtol=0,atol=1e-15,equal_nan=True)
        self.assertEqual(int(plan.eligible.sum()),181)

    def test_ensemble_averages_before_metrics(self):
        spec=importlib.util.spec_from_file_location('current_workflow',ROOT/'scripts/run_experiment.py')
        workflow=importlib.util.module_from_spec(spec);spec.loader.exec_module(workflow)
        frames=[pd.read_csv(ROOT/'results/final_outer/elite_archive'/f'seed{s}.csv') for s in (42,123,456)]
        mean=workflow.ensemble(frames)
        reference=pd.read_csv(ROOT/'results/final_outer/elite_archive/ensemble.csv')
        np.testing.assert_allclose(mean.q_pred,reference.q_pred,rtol=0,atol=1e-12)


if __name__=='__main__':unittest.main()
