"""Portable data preparation for the recorded external baseline implementations."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Sequence, Tuple
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class BenchmarkConfig:
    station_file: Path
    model_seed: int = 42
    split_seed: int = 42
    test_size: float = .2
    iqr_multiplier: float = 3.
    spatial_cv_blocks: int = 5
    n_jobs: int = 12
    smoke_test: bool = False

@dataclass
class BenchmarkDataset:
    run_config: Any
    experiment_data: Any
    x_train_raw: np.ndarray
    x_test_raw: np.ndarray
    x_train: np.ndarray
    x_test: np.ndarray
    y_train_log: np.ndarray
    y_train_scaled: np.ndarray
    y_test: np.ndarray
    coordinates_train: np.ndarray
    coordinates_test: np.ndarray
    train_original_indices: np.ndarray
    test_original_indices: np.ndarray
    spatial_block_labels: np.ndarray
    cv_splits: List[Tuple[np.ndarray, np.ndarray]]
    protocol_fingerprint: str
    split_sha256: str
    smoke_train_subset: bool

    @property
    def feature_names(self) -> List[str]:
        return list(self.run_config.features)

    def inverse_target(self, scaled_log_values: Sequence[float]) -> np.ndarray:
        log_values = self.experiment_data.scaler_y.inverse_transform(
            np.asarray(scaled_log_values, dtype=float).reshape(-1, 1)
        ).ravel()
        return np.exp(np.clip(log_values, -5.0, 10.0))


@dataclass
class CallTiming:
    wall_seconds: float
    cpu_seconds_main_process: float
    peak_tracemalloc_mb: float

    def to_record(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_wall_seconds": self.wall_seconds,
            f"{prefix}_cpu_seconds_main_process": (
                self.cpu_seconds_main_process
            ),
            f"{prefix}_peak_tracemalloc_mb": self.peak_tracemalloc_mb,
        }


def measure_call(function: Callable[[], Any]) -> Tuple[Any, CallTiming]:
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        result = function()
    finally:
        cpu_seconds = time.process_time() - cpu_start
        wall_seconds = time.perf_counter() - wall_start
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, CallTiming(
        wall_seconds=float(wall_seconds),
        cpu_seconds_main_process=float(cpu_seconds),
        peak_tracemalloc_mb=float(peak_bytes / (1024.0**2)),
    )


def _canonical_spatial_labels(
    coordinates: np.ndarray,
    *,
    blocks: int,
    seed: int,
) -> np.ndarray:
    raw = KMeans(
        n_clusters=blocks,
        random_state=seed,
        n_init=20,
    ).fit_predict(coordinates)
    centroids = {
        int(label): tuple(
            np.mean(coordinates[raw == label], axis=0).tolist()
        )
        for label in np.unique(raw)
    }
    ordered = sorted(centroids, key=lambda label: centroids[label])
    mapping = {old: new for new, old in enumerate(ordered)}
    labels = np.asarray([mapping[int(value)] for value in raw], dtype=int)
    for label in range(blocks):
        if not np.any(labels == label):
            raise RuntimeError("Spatial CV produced an empty block.")
    return labels

def load_dataset(config):
    features = ['topo', 'bouguer', 'sediment', 'moho', 'lab', 'cpd', 'lithos_km', 'volcano']
    data = pd.read_csv(config.station_file)
    train_ids, test_ids = train_test_split(np.arange(len(data)), test_size=config.test_size, random_state=config.split_seed)
    raw, test = data.iloc[train_ids].copy(), data.iloc[test_ids].copy()
    q1, q3 = raw.heat_flow.quantile([.25, .75])
    keep = raw.heat_flow.between(q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1))
    train = raw.loc[keep].copy()
    xs, ys = StandardScaler(), StandardScaler()
    x_train = xs.fit_transform(train[features].to_numpy(dtype=float))
    x_test = xs.transform(test[features].to_numpy(dtype=float))
    y_log = np.log(train.heat_flow.to_numpy(dtype=float))
    y_scaled = ys.fit_transform(y_log.reshape(-1, 1)).ravel()
    coords = train[['lon', 'lat']].to_numpy(dtype=float)
    labels = _canonical_spatial_labels(coords, blocks=config.spatial_cv_blocks, seed=config.split_seed)
    return BenchmarkDataset(run_config=SimpleNamespace(features=features), experiment_data=SimpleNamespace(scaler_y=ys),
        x_train_raw=train[features].to_numpy(dtype=float), x_test_raw=test[features].to_numpy(dtype=float),
        x_train=x_train, x_test=x_test, y_train_log=y_log, y_train_scaled=y_scaled,
        y_test=test.heat_flow.to_numpy(dtype=float), coordinates_train=coords,
        coordinates_test=test[['lon', 'lat']].to_numpy(dtype=float), train_original_indices=train.index.to_numpy(),
        test_original_indices=test_ids, spatial_block_labels=labels,
        cv_splits=[(np.flatnonzero(labels != b), np.flatnonzero(labels == b)) for b in range(config.spatial_cv_blocks)],
        protocol_fingerprint='current_external_baselines', split_sha256='', smoke_train_subset=False)
