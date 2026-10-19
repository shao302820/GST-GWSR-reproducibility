"""Model settings, bisquare weights and inverse transform for the current study."""
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import StandardScaler
REPO_ROOT = Path(__file__).resolve().parents[3]
STATION_FILE = Path(os.environ.get("GST_GWSR_STATION_FILE", str(REPO_ROOT / "data/processed/heat_flow_stations.csv")))
FEATURES = ["topo", "bouguer", "sediment", "moho", "lab", "cpd", "lithos_km", "volcano"]
TARGET = "heat_flow"
COORDS = ["lon", "lat"]

def iqr_bounds(values: pd.Series, multiplier: float = 3.0) -> Tuple[float, float, float, float, float]:
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return q1, q3, iqr, lower, upper


def get_local_sample(
    center_idx: int,
    d_row: np.ndarray,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    k: int,
    min_samples: int = 20,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k_use = max(k, min_samples)
    k_use = min(k_use, len(d_row))
    sorted_idx = np.argsort(d_row)
    if center_idx >= 0:
        sorted_idx = sorted_idx[sorted_idx != center_idx]

    local_idx = sorted_idx[:k_use]
    x_local = x_ref[local_idx]
    y_local = y_ref[local_idx]
    d_local = d_row[local_idx]
    h = d_row[sorted_idx[k_use - 1]] + 1e-8
    weights = np.where(d_local < h, (1.0 - (d_local / h) ** 2) ** 2, 0.0)
    weights = weights / weights.sum()
    return x_local, y_local, weights


def build_sr_model(seed: int) -> SymbolicRegressor:
    return SymbolicRegressor(
        population_size=1000,
        generations=20,
        tournament_size=20,
        init_depth=(2, 5),
        init_method="half and half",
        function_set=["add", "sub", "mul", "div", "sqrt", "log", "abs", "neg"],
        metric="mse",
        parsimony_coefficient=0.01,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=0.8,
        feature_names=FEATURES,
        warm_start=False,
        n_jobs=1,
        verbose=0,
        random_state=seed,
    )


def inverse_transform(y_scaled: np.ndarray, scaler_y: StandardScaler) -> np.ndarray:
    y_log = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    return np.exp(np.clip(y_log, -5, 10))
