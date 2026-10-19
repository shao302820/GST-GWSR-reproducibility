from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from .core import BenchmarkConfig, BenchmarkDataset, measure_call
from .model_types import ModelRun


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _adaptive_bisquare_weights(
    distances: np.ndarray,
    bandwidth_neighbors: float,
) -> np.ndarray:
    distances = np.asarray(distances, dtype=float)
    neighbor_count = int(
        np.clip(
            round(float(bandwidth_neighbors)),
            2,
            len(distances),
        )
    )
    ordered = np.partition(distances, neighbor_count - 1)
    threshold = max(float(ordered[neighbor_count - 1]), 1e-12)
    threshold *= 1.0000001
    ratios = distances / threshold
    weights = np.where(
        ratios < 1.0,
        (1.0 - ratios**2) ** 2,
        0.0,
    )
    if not np.any(weights > 0):
        weights[int(np.argmin(distances))] = 1.0
    return weights


def run_gwr(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    y = dataset.y_train_scaled.reshape(-1, 1)
    minimum_bw = max(20, 2 * (dataset.x_train.shape[1] + 1) + 2)
    maximum_bw = max(minimum_bw + 1, len(dataset.x_train) - 1)
    selector = Sel_BW(
        dataset.coordinates_train,
        y,
        dataset.x_train,
        kernel="bisquare",
        fixed=False,
        constant=True,
        spherical=False,
        n_jobs=config.n_jobs,
    )
    bandwidth, tuning_timing = measure_call(
        lambda: selector.search(
            search_method="golden_section",
            criterion="AICc",
            bw_min=minimum_bw,
            bw_max=maximum_bw,
            tol=1e-5 if config.smoke_test else 1e-6,
            max_iter=60 if config.smoke_test else 200,
        )
    )
    model = GWR(
        dataset.coordinates_train,
        y,
        dataset.x_train,
        bandwidth,
        kernel="bisquare",
        fixed=False,
        constant=True,
        spherical=False,
        hat_matrix=False,
        n_jobs=config.n_jobs,
    )
    results, fit_timing = measure_call(model.fit)

    def predict() -> np.ndarray:
        output = model.predict(
            dataset.coordinates_test,
            dataset.x_test,
            exog_scale=results.scale,
            exog_resid=results.resid_response,
        )
        return np.asarray(output.predictions, dtype=float).reshape(-1)

    prediction_scaled, prediction_timing = measure_call(predict)
    prediction = dataset.inverse_target(prediction_scaled)
    total_wall = (
        tuning_timing.wall_seconds
        + fit_timing.wall_seconds
        + prediction_timing.wall_seconds
    )
    tuning_frame = pd.DataFrame(
        [
            {
                "model": "gwr",
                "selection_method": "training-only AICc",
                "adaptive_bandwidth_neighbors": float(bandwidth),
                "criterion_value_AICc": _finite_float(results.aicc),
            }
        ]
    )
    return ModelRun(
        model_id="gwr",
        display_name="经典地理加权回归（GWR）",
        prediction_scaled_log=prediction_scaled,
        prediction_heat_flow=prediction,
        selected_parameters={
            "adaptive_bandwidth_neighbors": float(bandwidth),
            "kernel": "bisquare",
            "fixed": False,
            "constant": True,
            "spherical": False,
            "bandwidth_selection": "training-only AICc",
        },
        complexity={
            "local_coefficient_count": int(
                results.params.shape[0] * results.params.shape[1]
            ),
            "coefficient_surfaces": int(results.params.shape[1]),
            "effective_number_of_parameters": _finite_float(results.ENP),
        },
        efficiency={
            **tuning_timing.to_record("tuning"),
            **fit_timing.to_record("final_fit"),
            **prediction_timing.to_record("prediction"),
            "total_wall_seconds": float(total_wall),
            "search_candidate_count": None,
            "spatial_cv_fit_count": 0,
            "parallel_workers": config.n_jobs,
            "memory_measurement_scope": (
                "tracemalloc main process only; native and worker memory excluded"
            ),
            "time_measurement_scope": (
                "current run: training-only bandwidth search + final fit + "
                "outer-test prediction"
            ),
        },
        tuning_results=tuning_frame,
        diagnostics={
            "training_AICc": _finite_float(results.aicc),
            "training_R2_scaled_log": _finite_float(results.R2),
            "training_adjusted_R2_scaled_log": _finite_float(
                results.adj_R2
            ),
            "sigma2": _finite_float(results.sigma2),
        },
        notes={
            "test_target_used_for_tuning": False,
            "predictive_features": dataset.feature_names,
            "coordinates_as_predictors": False,
            "coordinate_role": "adaptive spatial kernel only",
            "distance_metric": (
                "Euclidean longitude/latitude degrees, matching GST-GWSR"
            ),
        },
        model_object=model,
    )


def _mgwr_out_of_sample_prediction(
    *,
    coordinates_train: np.ndarray,
    coordinates_test: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    parameters_train: np.ndarray,
    bandwidths: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, float]]:
    train_design = np.column_stack(
        [np.ones(len(x_train), dtype=float), x_train]
    )
    test_design = np.column_stack(
        [np.ones(len(x_test), dtype=float), x_test]
    )
    fitted_components = train_design * parameters_train
    fitted_train = np.sum(fitted_components, axis=1)
    residual = y_train - fitted_train
    partial_targets = fitted_components + residual[:, None]

    predictions = np.zeros(len(test_design), dtype=float)
    for test_index, coordinate in enumerate(coordinates_test):
        distances = np.linalg.norm(
            coordinates_train - coordinate[None, :],
            axis=1,
        )
        local_coefficients = np.zeros(train_design.shape[1], dtype=float)
        for column in range(train_design.shape[1]):
            weights = _adaptive_bisquare_weights(
                distances,
                bandwidths[column],
            )
            predictor = train_design[:, column]
            denominator = float(np.sum(weights * predictor * predictor))
            if denominator <= 1e-12:
                local_coefficients[column] = 0.0
            else:
                local_coefficients[column] = float(
                    np.sum(
                        weights
                        * predictor
                        * partial_targets[:, column]
                    )
                    / denominator
                )
        predictions[test_index] = float(
            np.dot(test_design[test_index], local_coefficients)
        )
    reconstruction_rmse = float(
        np.sqrt(mean_squared_error(y_train, fitted_train))
    )
    return predictions, {
        "training_backfit_reconstruction_RMSE_scaled_log": (
            reconstruction_rmse
        )
    }


def run_mgwr(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    y = dataset.y_train_scaled.reshape(-1, 1)
    coefficient_count = dataset.x_train.shape[1] + 1
    minimum_bw = max(20, 2 * coefficient_count + 2)
    maximum_bw = max(minimum_bw + 1, len(dataset.x_train) - 1)
    selector = Sel_BW(
        dataset.coordinates_train,
        y,
        dataset.x_train,
        kernel="bisquare",
        fixed=False,
        multi=True,
        constant=True,
        spherical=False,
        n_jobs=config.n_jobs,
    )
    bandwidths, tuning_timing = measure_call(
        lambda: selector.search(
            search_method="golden_section",
            criterion="AICc",
            init_multi=None,
            tol=1e-5 if config.smoke_test else 1e-6,
            max_iter=40 if config.smoke_test else 100,
            tol_multi=5e-3 if config.smoke_test else 1e-4,
            max_iter_multi=8 if config.smoke_test else 100,
            multi_bw_min=[minimum_bw] * coefficient_count,
            multi_bw_max=[maximum_bw] * coefficient_count,
            bws_same_times=3 if config.smoke_test else 5,
            verbose=False,
        )
    )
    model = MGWR(
        dataset.coordinates_train,
        y,
        dataset.x_train,
        selector,
        kernel="bisquare",
        fixed=False,
        constant=True,
        spherical=False,
        hat_matrix=False,
        n_jobs=config.n_jobs,
    )
    results, fit_timing = measure_call(model.fit)

    prediction_output, prediction_timing = measure_call(
        lambda: _mgwr_out_of_sample_prediction(
            coordinates_train=dataset.coordinates_train,
            coordinates_test=dataset.coordinates_test,
            x_train=dataset.x_train,
            x_test=dataset.x_test,
            y_train=dataset.y_train_scaled,
            parameters_train=np.asarray(results.params, dtype=float),
            bandwidths=np.asarray(bandwidths, dtype=float).reshape(-1),
        )
    )
    prediction_scaled, prediction_diagnostics = prediction_output
    prediction = dataset.inverse_target(prediction_scaled)
    bandwidth_values = np.asarray(bandwidths, dtype=float).reshape(-1)
    names = ["Intercept"] + dataset.feature_names
    total_wall = (
        tuning_timing.wall_seconds
        + fit_timing.wall_seconds
        + prediction_timing.wall_seconds
    )
    tuning_frame = pd.DataFrame(
        {
            "model": "mgwr",
            "coefficient": names,
            "adaptive_bandwidth_neighbors": bandwidth_values,
            "selection_method": "training-only multiscale AICc",
        }
    )
    return ModelRun(
        model_id="mgwr",
        display_name="多尺度地理加权回归（MGWR）",
        prediction_scaled_log=prediction_scaled,
        prediction_heat_flow=prediction,
        selected_parameters={
            "bandwidths_by_coefficient": {
                name: float(value)
                for name, value in zip(names, bandwidth_values)
            },
            "kernel": "bisquare",
            "fixed": False,
            "constant": True,
            "spherical": False,
            "bandwidth_selection": (
                "training-only multiscale AICc backfitting"
            ),
        },
        complexity={
            "local_coefficient_count": int(
                results.params.shape[0] * results.params.shape[1]
            ),
            "coefficient_surfaces": int(results.params.shape[1]),
            "effective_number_of_parameters": _finite_float(results.ENP),
        },
        efficiency={
            **tuning_timing.to_record("tuning"),
            **fit_timing.to_record("final_fit"),
            **prediction_timing.to_record("prediction"),
            "total_wall_seconds": float(total_wall),
            "search_candidate_count": None,
            "spatial_cv_fit_count": 0,
            "parallel_workers": config.n_jobs,
            "memory_measurement_scope": (
                "tracemalloc main process only; native and worker memory excluded"
            ),
            "time_measurement_scope": (
                "current run: multiscale bandwidth search + final fit + "
                "training-only backfitting extrapolation to outer test"
            ),
        },
        tuning_results=tuning_frame,
        diagnostics={
            "training_AICc": _finite_float(results.aicc),
            "training_R2_scaled_log": _finite_float(results.R2),
            "training_adjusted_R2_scaled_log": _finite_float(
                results.adj_R2
            ),
            **prediction_diagnostics,
        },
        notes={
            "test_target_used_for_tuning": False,
            "predictive_features": dataset.feature_names,
            "coordinates_as_predictors": False,
            "coordinate_role": (
                "coefficient-specific adaptive spatial kernels"
            ),
            "distance_metric": (
                "Euclidean longitude/latitude degrees, matching GST-GWSR"
            ),
            "out_of_sample_prediction": (
                "mgwr 2.2.1 has no native out-of-sample MGWR.predict; "
                "predictions use training-only final backfitting partial "
                "targets and each selected coefficient bandwidth"
            ),
        },
        model_object=model,
    )


def _gaussian_local_weights(distances: np.ndarray) -> np.ndarray:
    bandwidth = max(float(np.max(distances)), 1e-12)
    weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
    return weights / max(float(np.sum(weights)), 1e-12)


def _fit_local_rf_prediction(
    *,
    target_position: int | None,
    eligible_train_positions: np.ndarray | None,
    target_x: np.ndarray,
    target_coordinate: np.ndarray,
    x_train: np.ndarray,
    y_train: np.ndarray,
    coordinates_train: np.ndarray,
    k_neighbors: int,
    max_features: str | float,
    min_samples_leaf: int,
    n_estimators: int,
    seed: int,
) -> Dict[str, Any]:
    distances = np.linalg.norm(
        coordinates_train - target_coordinate[None, :],
        axis=1,
    )
    if eligible_train_positions is not None:
        eligible = np.zeros(len(distances), dtype=bool)
        eligible[
            np.asarray(eligible_train_positions, dtype=int)
        ] = True
        distances[~eligible] = np.inf
    if target_position is not None:
        distances[int(target_position)] = np.inf
    finite_positions = np.flatnonzero(np.isfinite(distances))
    k_use = min(int(k_neighbors), len(finite_positions))
    local_positions = np.argsort(distances)[:k_use]
    local_distances = distances[local_positions]
    weights = _gaussian_local_weights(local_distances)
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        bootstrap=True,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(
        x_train[local_positions],
        y_train[local_positions],
        sample_weight=weights,
    )
    prediction = float(model.predict(target_x.reshape(1, -1))[0])
    node_count = int(
        sum(tree.tree_.node_count for tree in model.estimators_)
    )
    mean_depth = float(
        np.mean([tree.tree_.max_depth for tree in model.estimators_])
    )
    return {
        "prediction": prediction,
        "feature_importances": model.feature_importances_.copy(),
        "node_count": node_count,
        "mean_depth": mean_depth,
        "local_k": k_use,
    }


def _gwrf_anchor_positions(
    dataset: BenchmarkDataset,
    *,
    limit: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions: List[int] = []
    blocks = np.unique(dataset.spatial_block_labels)
    per_block = max(1, int(math.ceil(limit / len(blocks))))
    for block in blocks:
        candidates = np.flatnonzero(dataset.spatial_block_labels == block)
        take = min(per_block, len(candidates))
        positions.extend(
            rng.choice(candidates, size=take, replace=False).tolist()
        )
    return np.asarray(sorted(positions[:limit]), dtype=int)


def run_gwrf(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    anchor_limit = min(
        len(dataset.x_train),
        30 if config.smoke_test else 120,
    )
    anchors = _gwrf_anchor_positions(
        dataset,
        limit=anchor_limit,
        seed=config.model_seed,
    )
    k_values = (
        [min(25, len(dataset.x_train) - 1), min(50, len(dataset.x_train) - 1)]
        if config.smoke_test
        else [25, 50, 100]
    )
    max_features_values: List[str | float] = (
        ["sqrt"]
        if config.smoke_test
        else ["sqrt", 0.7]
    )
    leaf_values = [1] if config.smoke_test else [1, 2]
    candidates = [
        {
            "k_neighbors": int(k),
            "max_features": max_features,
            "min_samples_leaf": int(leaf),
        }
        for k in k_values
        for max_features in max_features_values
        for leaf in leaf_values
        if k >= 5
    ]
    tuning_tree_count = 40 if config.smoke_test else 80

    def tune_all() -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            with parallel_backend("loky", inner_max_num_threads=1):
                outputs = Parallel(
                    n_jobs=config.n_jobs,
                    verbose=0,
                )(
                    delayed(_fit_local_rf_prediction)(
                        target_position=int(position),
                        eligible_train_positions=np.flatnonzero(
                            dataset.spatial_block_labels
                            != dataset.spatial_block_labels[int(position)]
                        ),
                        target_x=dataset.x_train_raw[int(position)],
                        target_coordinate=dataset.coordinates_train[
                            int(position)
                        ],
                        x_train=dataset.x_train_raw,
                        y_train=dataset.y_train_log,
                        coordinates_train=dataset.coordinates_train,
                        k_neighbors=candidate["k_neighbors"],
                        max_features=candidate["max_features"],
                        min_samples_leaf=candidate["min_samples_leaf"],
                        n_estimators=tuning_tree_count,
                        seed=(
                            config.model_seed
                            + 100_000 * candidate_index
                            + int(position)
                        ),
                    )
                    for position in anchors
                )
            predictions = np.asarray(
                [item["prediction"] for item in outputs],
                dtype=float,
            )
            observed = dataset.y_train_log[anchors]
            rows.append(
                {
                    **candidate,
                    "anchor_count": len(anchors),
                    "tuning_tree_count": tuning_tree_count,
                    "training_spatial_anchor_RMSE_log_heat_flow": float(
                        np.sqrt(mean_squared_error(observed, predictions))
                    ),
                }
            )
        return rows

    tuning_rows, tuning_timing = measure_call(tune_all)
    tuning_frame = pd.DataFrame(tuning_rows).sort_values(
        [
            "training_spatial_anchor_RMSE_log_heat_flow",
            "k_neighbors",
            "min_samples_leaf",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    best = tuning_frame.iloc[0].to_dict()
    final_tree_count = 100 if config.smoke_test else 300

    def fit_predict_all() -> List[Dict[str, Any]]:
        with parallel_backend("loky", inner_max_num_threads=1):
            return Parallel(n_jobs=config.n_jobs, verbose=0)(
                delayed(_fit_local_rf_prediction)(
                    target_position=None,
                    eligible_train_positions=None,
                    target_x=dataset.x_test_raw[test_index],
                    target_coordinate=dataset.coordinates_test[test_index],
                    x_train=dataset.x_train_raw,
                    y_train=dataset.y_train_log,
                    coordinates_train=dataset.coordinates_train,
                    k_neighbors=int(best["k_neighbors"]),
                    max_features=best["max_features"],
                    min_samples_leaf=int(best["min_samples_leaf"]),
                    n_estimators=final_tree_count,
                    seed=config.model_seed + 1_000_000 + test_index,
                )
                for test_index in range(len(dataset.x_test))
            )

    outputs, final_timing = measure_call(fit_predict_all)
    prediction_log = np.asarray(
        [item["prediction"] for item in outputs],
        dtype=float,
    )
    prediction_scaled = (
        dataset.experiment_data.scaler_y.transform(
            prediction_log.reshape(-1, 1)
        ).ravel()
    )
    prediction = dataset.inverse_target(prediction_scaled)
    importance = np.mean(
        np.asarray(
            [item["feature_importances"] for item in outputs],
            dtype=float,
        ),
        axis=0,
    )
    total_nodes = int(sum(item["node_count"] for item in outputs))
    mean_depth = float(
        np.mean([item["mean_depth"] for item in outputs])
    )
    total_wall = tuning_timing.wall_seconds + final_timing.wall_seconds
    tuning_frame.insert(0, "model", "gwrf")
    return ModelRun(
        model_id="gwrf",
        display_name="地理加权随机森林（GWRF）",
        prediction_scaled_log=prediction_scaled,
        prediction_heat_flow=prediction,
        selected_parameters={
            "k_neighbors": int(best["k_neighbors"]),
            "max_features": best["max_features"],
            "min_samples_leaf": int(best["min_samples_leaf"]),
            "n_estimators": final_tree_count,
            "kernel": "Gaussian",
            "bandwidth_selection": (
                "training-only common spatial-block CV on stratified "
                "validation anchors"
            ),
        },
        complexity={
            "local_model_count": len(outputs),
            "trees_per_local_model": final_tree_count,
            "total_tree_count": int(len(outputs) * final_tree_count),
            "total_tree_nodes": total_nodes,
            "mean_tree_depth": mean_depth,
            "mean_feature_importance": {
                name: float(value)
                for name, value in zip(dataset.feature_names, importance)
            },
        },
        efficiency={
            **tuning_timing.to_record("tuning"),
            **final_timing.to_record("local_fit_and_prediction"),
            "final_fit_wall_seconds": final_timing.wall_seconds,
            "prediction_wall_seconds": 0.0,
            "total_wall_seconds": float(total_wall),
            "search_candidate_count": len(candidates),
            "spatial_cv_fit_count": int(
                len(candidates) * len(anchors)
            ),
            "parallel_workers": config.n_jobs,
            "memory_measurement_scope": (
                "tracemalloc main process only; native and worker memory excluded"
            ),
            "time_measurement_scope": (
                "current run: training-only local-anchor tuning + one local "
                "forest fit/prediction per outer-test point"
            ),
        },
        tuning_results=tuning_frame,
        diagnostics={
            "best_training_spatial_anchor_RMSE_log_heat_flow": float(
                best["training_spatial_anchor_RMSE_log_heat_flow"]
            ),
            "training_anchor_count": int(len(anchors)),
            "spatial_cv_blocks": int(config.spatial_cv_blocks),
        },
        notes={
            "test_target_used_for_tuning": False,
            "predictive_features": dataset.feature_names,
            "coordinates_as_predictors": False,
            "coordinate_role": "local-neighborhood and Gaussian weights",
            "distance_metric": (
                "Euclidean longitude/latitude degrees, matching GST-GWSR"
            ),
            "implementation_scope": (
                "auditable local geographically weighted random forest; "
                "not a claim of reproducing a specific third-party GRF package"
            ),
            "tuning_protocol": (
                "Each tuning anchor is predicted after excluding its entire "
                "common spatial CV block from the eligible local training "
                "neighborhood."
            ),
        },
        model_object=None,
    )


SPATIAL_MODEL_RUNNERS = {
    "gwr": run_gwr,
    "mgwr": run_mgwr,
    "gwrf": run_gwrf,
}
