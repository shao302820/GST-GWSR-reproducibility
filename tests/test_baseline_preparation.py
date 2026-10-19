"""External-baseline split and spatial tuning blocks match recorded inputs."""
from pathlib import Path
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from baselines.core import BenchmarkConfig, load_dataset


class BaselinePreparationTests(unittest.TestCase):
    def test_retained_training_rows_and_spatial_blocks(self):
        data=load_dataset(BenchmarkConfig(ROOT/'data/processed/heat_flow_stations.csv'))
        reference=pd.read_csv(ROOT/'results/external_baselines/training_spatial_cv_assignments.csv')
        np.testing.assert_array_equal(data.train_original_indices,reference.original_row_index)
        np.testing.assert_array_equal(data.spatial_block_labels,reference.spatial_cv_block)
        predictions=pd.read_csv(ROOT/'results/external_baselines/ensemble_predictions.csv')
        np.testing.assert_array_equal(data.test_original_indices,predictions.row_id)
        np.testing.assert_array_equal(data.y_test,predictions.q_obs)
        self.assertEqual(data.x_train.shape,(815,8))
        self.assertEqual(data.x_test.shape,(205,8))
        self.assertEqual(len(data.cv_splits),5)


if __name__=='__main__':unittest.main()
