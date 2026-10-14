"""Closed, original-unit magnitude fits on already-admitted numeric matrices."""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting

from ._magnitude_estimators import build_estimator
from .legacy_adapter import forbid_fitting

__all__ = ["fitted_magnitude_recipe", "fit_magnitude_predictions"]


def fitted_magnitude_recipe(kind: str) -> dict:
    """Return a detached JSON policy; accepting caller overrides is unsupported."""
    if kind not in ("history_ridge", "history_hgb"):
        raise ValueError("unknown fitted magnitude model")
    ridge = kind == "history_ridge"
    params = ({"alpha": 1.0, "fit_intercept": True, "copy_X": True,
               "max_iter": None, "tol": 0.0001, "solver": "svd", "positive": False,
               "random_state": 0} if ridge else {
        "loss": "squared_error", "quantile": None, "learning_rate": 0.1,
        "max_iter": 100, "max_leaf_nodes": 7, "max_depth": None,
        "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0,
        "max_bins": 255, "categorical_features": None, "monotonic_cst": None,
        "interaction_cst": None, "warm_start": False, "early_stopping": False,
        "scoring": "loss", "validation_fraction": 0.1, "n_iter_no_change": 10,
        "tol": 1e-7, "verbose": 0, "random_state": 0})
    preprocessing = ({
        "imputer": {"strategy": "median", "keep_empty_features": True},
        "missing_indicator": {"features": "all", "sparse": False},
        "scaler": {"with_mean": True, "with_std": True}, "output_features": 22,
    } if ridge else {"missing_values": "native", "scaling": "none",
                    "all_empty_training_columns": "constant_zero_train_and_predict"})
    return {"schema_version": "magnitude_fitted_recipe.v1", "model": kind,
        "features": ["n_prior", "mean_prior_move", "mean_prior_abs_move",
                     "ema2_prior_move", "ema2_prior_abs_move", "ema4_prior_move",
                     "ema4_prior_abs_move", "ema8_prior_move", "ema8_prior_abs_move",
                     "ema12_prior_move", "ema12_prior_abs_move"],
        "estimator": {"class": "Ridge" if ridge else "HistGradientBoostingRegressor",
                      "params": params}, "preprocessing": preprocessing,
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1}


def fit_magnitude_predictions(kind: str, train_features, train_labels, test_features) -> list[float]:
    """Fit fresh training-only preprocessing/model state and return detached floats."""
    forbid_fitting("engine.v2.models.training.magnitude.fit_magnitude_predictions")
    forbid_v2_fitting("engine.v2.models.training.magnitude.fit_magnitude_predictions")
    recipe = fitted_magnitude_recipe(kind)
    train, labels, test = (np.asarray(value, dtype=float)
                           for value in (train_features, train_labels, test_features))
    for matrix in (train, test):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 11
                or np.isinf(matrix).any()):
            raise ValueError("magnitude features require nonempty eleven-column matrices without infinity")
    if labels.shape != (len(train),) or not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("magnitude labels require one finite nonnegative value per training row")
    if kind == "history_hgb":
        # sklearn 1.9 binning cannot accept an entirely missing training column.
        empty = np.isnan(train).all(axis=0)
        train = np.where(empty, 0.0, train)
        test = np.where(empty, 0.0, test)
    model = build_estimator(recipe)
    with threadpool_limits(limits=1):
        model.fit(train, labels)
        predictions = np.asarray(model.predict(test), dtype=float)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError("magnitude predictions require one finite value per evaluation row")
    return np.maximum(predictions, 0.0).tolist()
