from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .core import BenchmarkConfig, BenchmarkDataset, measure_call
from .model_types import ModelRun


def _clean_params(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        str(key): (
            value.item() if isinstance(value, np.generic) else value
        )
        for key, value in params.items()
    }


def _grid_search(
    *,
    model_id: str,
    display_name: str,
    estimator: Any,
    parameter_grid: Dict[str, Iterable[Any]],
    final_parameter_overrides: Dict[str, Any],
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> Tuple[ModelRun, Any]:
    # Every inner-CV training fold fits its own feature and target scalers.
    # This prevents validation-block statistics (especially target mean/std)
    # from leaking into hyperparameter selection.
    wrapped_estimator = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("x_scaler", StandardScaler()),
                ("model", estimator),
            ]
        ),
        transformer=StandardScaler(),
    )
    prefixed_grid = {
        f"regressor__model__{key}": values
        for key, values in parameter_grid.items()
    }
    search = GridSearchCV(
        estimator=wrapped_estimator,
        param_grid=prefixed_grid,
        scoring="neg_root_mean_squared_error",
        cv=dataset.cv_splits,
        refit=False,
        n_jobs=config.n_jobs,
        pre_dispatch=max(1, 2 * abs(config.n_jobs)),
        error_score="raise",
        return_train_score=False,
    )
    _, tuning_timing = measure_call(
        lambda: search.fit(dataset.x_train_raw, dataset.y_train_log)
    )
    best_index = int(np.argmax(search.cv_results_["mean_test_score"]))
    prefixed_best = _clean_params(
        search.cv_results_["params"][best_index]
    )
    best_parameters = {
        key.removeprefix("regressor__model__"): value
        for key, value in prefixed_best.items()
    }
    final_parameters = {
        **best_parameters,
        **final_parameter_overrides,
    }
    final_model = clone(wrapped_estimator).set_params(
        **{
            f"regressor__model__{key}": value
            for key, value in final_parameters.items()
        }
    )
    _, fit_timing = measure_call(
        lambda: final_model.fit(
            dataset.x_train_raw,
            dataset.y_train_log,
        )
    )
    prediction_log, prediction_timing = measure_call(
        lambda: np.asarray(
            final_model.predict(dataset.x_test_raw),
            dtype=float,
        )
    )
    prediction_scaled = (
        dataset.experiment_data.scaler_y.transform(
            prediction_log.reshape(-1, 1)
        ).ravel()
    )
    prediction = dataset.inverse_target(prediction_scaled)
    fitted_estimator = final_model.regressor_.named_steps["model"]

    tuning_frame = pd.DataFrame(search.cv_results_)[
        [
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score",
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
        ]
    ].copy()
    tuning_frame["params"] = tuning_frame["params"].map(
        lambda values: {
            key.removeprefix("regressor__model__"): value
            for key, value in values.items()
        }
    )
    tuning_frame["mean_spatial_cv_RMSE_log_heat_flow"] = -tuning_frame[
        "mean_test_score"
    ]
    tuning_frame.insert(0, "model", model_id)

    complexity: Dict[str, Any] = {}
    if isinstance(fitted_estimator, RandomForestRegressor):
        complexity = {
            "tree_count": len(fitted_estimator.estimators_),
            "total_tree_nodes": int(
                sum(
                    tree.tree_.node_count
                    for tree in fitted_estimator.estimators_
                )
            ),
            "mean_tree_depth": float(
                np.mean(
                    [
                        tree.tree_.max_depth
                        for tree in fitted_estimator.estimators_
                    ]
                )
            ),
        }
    elif isinstance(fitted_estimator, GradientBoostingRegressor):
        trees = fitted_estimator.estimators_.ravel().tolist()
        complexity = {
            "tree_count": len(trees),
            "total_tree_nodes": int(
                sum(tree.tree_.node_count for tree in trees)
            ),
            "mean_tree_depth": float(
                np.mean([tree.tree_.max_depth for tree in trees])
            ),
        }
    elif isinstance(fitted_estimator, SVR):
        complexity = {
            "support_vector_count": int(len(fitted_estimator.support_)),
            "support_vector_fraction": float(
                len(fitted_estimator.support_) / len(dataset.x_train)
            ),
        }
    elif isinstance(fitted_estimator, XGBRegressor):
        booster = fitted_estimator.get_booster()
        complexity = {
            "tree_count": int(booster.num_boosted_rounds()),
            "feature_count": int(fitted_estimator.n_features_in_),
        }

    total_wall = (
        tuning_timing.wall_seconds
        + fit_timing.wall_seconds
        + prediction_timing.wall_seconds
    )
    efficiency = {
        **tuning_timing.to_record("tuning"),
        **fit_timing.to_record("final_fit"),
        **prediction_timing.to_record("prediction"),
        "total_wall_seconds": float(total_wall),
        "search_candidate_count": len(ParameterGrid(parameter_grid)),
        "spatial_cv_fit_count": int(
            len(ParameterGrid(parameter_grid)) * len(dataset.cv_splits)
        ),
        "parallel_workers": config.n_jobs,
        "memory_measurement_scope": (
            "tracemalloc main process only; native and worker memory excluded"
        ),
        "time_measurement_scope": (
            "current run: spatial tuning + final fit + outer-test prediction"
        ),
    }
    run = ModelRun(
        model_id=model_id,
        display_name=display_name,
        prediction_scaled_log=prediction_scaled,
        prediction_heat_flow=prediction,
        selected_parameters=final_parameters,
        complexity=complexity,
        efficiency=efficiency,
        tuning_results=tuning_frame,
        diagnostics={
            "best_spatial_cv_RMSE_log_heat_flow": float(
                -search.cv_results_["mean_test_score"][best_index]
            ),
            "best_spatial_cv_score_std": float(
                search.cv_results_["std_test_score"][best_index]
            ),
        },
        notes={
            "test_target_used_for_tuning": False,
            "predictive_features": dataset.feature_names,
            "coordinates_as_predictors": False,
            "inner_fold_scaling": (
                "Feature StandardScaler and log-target StandardScaler are "
                "fitted separately inside every spatial CV training fold."
            ),
        },
        model_object=final_model,
    )
    return run, final_model


def run_ols(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    model = LinearRegression(fit_intercept=True)
    _, fit_timing = measure_call(
        lambda: model.fit(dataset.x_train, dataset.y_train_scaled)
    )
    prediction_scaled, prediction_timing = measure_call(
        lambda: np.asarray(model.predict(dataset.x_test), dtype=float)
    )
    prediction = dataset.inverse_target(prediction_scaled)
    coefficients = {
        name: float(value)
        for name, value in zip(dataset.feature_names, model.coef_)
    }
    return ModelRun(
        model_id="ols_log",
        display_name="OLS（统一对数目标）",
        prediction_scaled_log=prediction_scaled,
        prediction_heat_flow=prediction,
        selected_parameters={"fit_intercept": True},
        complexity={
            "coefficient_count": int(len(model.coef_) + 1),
            "intercept": float(model.intercept_),
            "coefficients": coefficients,
        },
        efficiency={
            **fit_timing.to_record("final_fit"),
            **prediction_timing.to_record("prediction"),
            "tuning_wall_seconds": 0.0,
            "total_wall_seconds": float(
                fit_timing.wall_seconds + prediction_timing.wall_seconds
            ),
            "search_candidate_count": 0,
            "spatial_cv_fit_count": 0,
            "parallel_workers": 1,
            "memory_measurement_scope": (
                "tracemalloc main process only; native memory excluded"
            ),
            "time_measurement_scope": (
                "current run: final fit + outer-test prediction"
            ),
        },
        diagnostics={},
        notes={
            "test_target_used_for_tuning": False,
            "predictive_features": dataset.feature_names,
            "coordinates_as_predictors": False,
        },
        model_object=model,
    )


def run_random_forest(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    estimator = RandomForestRegressor(
        n_estimators=120 if config.smoke_test else 250,
        random_state=config.model_seed,
        n_jobs=1,
        bootstrap=True,
    )
    grid = (
        {
            "max_features": ["sqrt", 1.0],
            "min_samples_leaf": [1, 2],
            "max_depth": [None],
        }
        if config.smoke_test
        else {
            "max_features": ["sqrt", 0.7, 1.0],
            "min_samples_leaf": [1, 2, 4],
            "max_depth": [None, 12],
        }
    )
    run, _ = _grid_search(
        model_id="random_forest",
        display_name="全局随机森林",
        estimator=estimator,
        parameter_grid=grid,
        final_parameter_overrides={
            "n_estimators": 200 if config.smoke_test else 500,
            "n_jobs": config.n_jobs,
        },
        dataset=dataset,
        config=config,
    )
    return run


def run_gradient_boosting(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    estimator = GradientBoostingRegressor(
        random_state=config.model_seed,
        loss="squared_error",
    )
    grid = (
        {
            "n_estimators": [100],
            "learning_rate": [0.05],
            "max_depth": [2, 3],
            "min_samples_leaf": [3],
        }
        if config.smoke_test
        else {
            "n_estimators": [150, 300],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3],
            "min_samples_leaf": [3, 6],
        }
    )
    run, _ = _grid_search(
        model_id="gradient_boosting",
        display_name="梯度提升回归树",
        estimator=estimator,
        parameter_grid=grid,
        final_parameter_overrides={},
        dataset=dataset,
        config=config,
    )
    return run


def run_xgboost(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    estimator = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=config.model_seed,
        n_jobs=1,
        reg_lambda=1.0,
        verbosity=0,
    )
    grid = (
        {
            "n_estimators": [120],
            "learning_rate": [0.05],
            "max_depth": [2, 3],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }
        if config.smoke_test
        else {
            "n_estimators": [300, 600],
            "learning_rate": [0.03, 0.05],
            "max_depth": [2, 3],
            "subsample": [0.8],
            "colsample_bytree": [0.8, 1.0],
        }
    )
    run, _ = _grid_search(
        model_id="xgboost",
        display_name="XGBoost",
        estimator=estimator,
        parameter_grid=grid,
        final_parameter_overrides={"n_jobs": config.n_jobs},
        dataset=dataset,
        config=config,
    )
    return run


def run_svr(
    dataset: BenchmarkDataset,
    config: BenchmarkConfig,
) -> ModelRun:
    estimator = SVR(kernel="rbf")
    grid = (
        {
            "C": [1.0, 10.0],
            "gamma": ["scale"],
            "epsilon": [0.05],
        }
        if config.smoke_test
        else {
            "C": [1.0, 10.0, 100.0],
            "gamma": ["scale", 0.03, 0.10],
            "epsilon": [0.05, 0.10],
        }
    )
    run, _ = _grid_search(
        model_id="svr_rbf",
        display_name="支持向量回归（RBF）",
        estimator=estimator,
        parameter_grid=grid,
        final_parameter_overrides={},
        dataset=dataset,
        config=config,
    )
    return run


GLOBAL_MODEL_RUNNERS = {
    "ols_log": run_ols,
    "random_forest": run_random_forest,
    "gradient_boosting": run_gradient_boosting,
    "xgboost": run_xgboost,
    "svr_rbf": run_svr,
}
