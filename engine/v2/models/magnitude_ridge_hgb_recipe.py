"""Closed variable-width Ridge/HGB metadata, without training or source admission."""
from __future__ import annotations

import json
import math
import re

from engine.v2.foundation import canonical_json

__all__ = ["ridge_hgb_recipe", "validate_ridge_hgb_recipe"]

_RIDGE_ALPHA_BOUNDS = (1e-6, 1000.0)
_HGB_LEARNING_RATE_BOUNDS = (0.0, 1.0)
_HGB_MAX_LEAF_NODES_V1_BOUNDS = (7, 15)
_HGB_MAX_LEAF_NODES_V2_BOUNDS = (2, 64)
_HGB_MIN_SAMPLES_LEAF_BOUNDS = (1, 1000)
_HGB_L2_REGULARIZATION_BOUNDS = (0.0, 1000.0)


def _bounded_number(value, bounds, *, integer=False) -> bool:
    lower, upper = bounds
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not lower <= value <= upper:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _bounded_alpha(value) -> bool:
    return _bounded_number(value, _RIDGE_ALPHA_BOUNDS)


def _ridge_parameters(alpha, max_leaf_nodes, early):
    if (not _bounded_alpha(alpha) or max_leaf_nodes is not None
            or any(value is not None for value in early)):
        raise ValueError("Ridge requires declared alpha and no HGB parameter")
    return {"alpha": float(alpha), "fit_intercept": True, "copy_X": True,
            "max_iter": None, "tol": 0.0001, "solver": "svd", "positive": False,
            "random_state": 0}


def _legacy_hgb_parameters(alpha, max_leaf_nodes):
    if (type(max_leaf_nodes) is not int
            or max_leaf_nodes not in _HGB_MAX_LEAF_NODES_V1_BOUNDS or alpha is not None):
        raise ValueError("HGB requires declared integer leaf count and no Ridge parameter")
    return {"loss": "squared_error", "quantile": None, "learning_rate": 0.1,
        "max_iter": 100, "max_leaf_nodes": max_leaf_nodes, "max_depth": None,
        "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0,
        "max_bins": 255, "categorical_features": None, "monotonic_cst": None,
        "interaction_cst": None, "warm_start": False, "early_stopping": False,
        "scoring": "loss", "validation_fraction": 0.1, "n_iter_no_change": 10,
        "tol": 1e-7, "verbose": 0, "random_state": 0}


def _check_early_parameters(alpha, max_leaf_nodes, early):
    if not all(value is not None for value in early):
        raise ValueError("HGB early-stopping parameters require all three declared values")
    if alpha is not None:
        raise ValueError("HGB early-stopping recipe takes no Ridge alpha parameter")
    lower, upper = _HGB_MAX_LEAF_NODES_V2_BOUNDS
    if not _bounded_number(max_leaf_nodes, _HGB_MAX_LEAF_NODES_V2_BOUNDS, integer=True):
        raise ValueError(f"HGB max_leaf_nodes must be an integer within [{lower}, {upper}]")
    if (not _bounded_number(early[0], _HGB_LEARNING_RATE_BOUNDS) or not early[0] > 0):
        raise ValueError("HGB learning_rate must be finite and within declared bounds")
    if not _bounded_number(early[1], _HGB_MIN_SAMPLES_LEAF_BOUNDS, integer=True):
        raise ValueError("HGB min_samples_leaf must be an integer within declared bounds")
    if not _bounded_number(early[2], _HGB_L2_REGULARIZATION_BOUNDS):
        raise ValueError("HGB l2_regularization must be finite and within declared bounds")


def _early_stopping_hgb_parameters(alpha, max_leaf_nodes, learning_rate, min_samples_leaf,
                                   l2_regularization):
    early = (learning_rate, min_samples_leaf, l2_regularization)
    _check_early_parameters(alpha, max_leaf_nodes, early)
    return {"loss": "squared_error", "quantile": None, "learning_rate": float(learning_rate),
        "max_iter": 500, "max_leaf_nodes": max_leaf_nodes, "max_depth": None,
        "min_samples_leaf": int(min_samples_leaf), "l2_regularization": float(l2_regularization),
        "max_features": 1.0, "max_bins": 255, "categorical_features": None,
        "monotonic_cst": None, "interaction_cst": None, "warm_start": False,
        "early_stopping": True, "scoring": "loss", "validation_fraction": 0.1,
        "n_iter_no_change": 10, "tol": 1e-7, "verbose": 0, "random_state": 0}


def _parameters(model, alpha, max_leaf_nodes, learning_rate=None, min_samples_leaf=None,
                l2_regularization=None):
    early = (learning_rate, min_samples_leaf, l2_regularization)
    if model == "history_ridge":
        return _ridge_parameters(alpha, max_leaf_nodes, early)
    if model != "history_hgb":
        raise ValueError("unknown variable-width magnitude model")
    if all(value is None for value in early):
        return _legacy_hgb_parameters(alpha, max_leaf_nodes)
    return _early_stopping_hgb_parameters(alpha, max_leaf_nodes, learning_rate,
                                          min_samples_leaf, l2_regularization)


def ridge_hgb_recipe(*, model, columns, feature_dictionary_hash, alpha=None, max_leaf_nodes=None,
                     learning_rate=None, min_samples_leaf=None, l2_regularization=None) -> dict:
    """Bind a caller-admitted ordered subset; dictionary membership stays external.

    The v2 HGB early-stopping validation holdout is a random 10% of training rows,
    not a time-ordered holdout; it only selects the tree count.
    """
    if (not isinstance(columns, (list, tuple)) or not columns
            or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                   for name in columns) or len(set(columns)) != len(columns)):
        raise ValueError("magnitude columns must be ordered unique feature identifiers")
    if (not isinstance(feature_dictionary_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", feature_dictionary_hash)):
        raise ValueError("magnitude requires a canonical feature dictionary hash")
    params = _parameters(model, alpha, max_leaf_nodes, learning_rate, min_samples_leaf,
                         l2_regularization)
    ridge = model == "history_ridge"
    version = ("magnitude_ridge_hgb_recipe.v2"
               if model == "history_hgb" and all(value is not None for value in (
                   learning_rate, min_samples_leaf, l2_regularization))
               else "magnitude_ridge_hgb_recipe.v1")
    return {
        "schema_version": version, "model": model,
        "features": list(columns), "feature_dictionary_hash": feature_dictionary_hash,
        "estimator": {"class": "Ridge" if ridge else "HistGradientBoostingRegressor",
                      "params": params},
        "preprocessing": {
            "imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True},
            "output_features": 2 * len(columns),
        } if ridge else {"missing_values": "native", "scaling": "none",
                        "all_empty_training_columns": "constant_zero_train_and_predict"},
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1,
    }


def validate_ridge_hgb_recipe(recipe) -> dict:
    """Detach the canonical policy; refuse unknown fields or altered fixed knobs."""
    try:
        json.dumps(recipe, allow_nan=False)  # canonical_json alone stringifies arbitrary objects.
        params = recipe["estimator"]["params"]
        early = ({"learning_rate": params.get("learning_rate"),
                  "min_samples_leaf": params.get("min_samples_leaf"),
                  "l2_regularization": params.get("l2_regularization")}
                 if recipe["schema_version"] == "magnitude_ridge_hgb_recipe.v2" else {})
        expected = ridge_hgb_recipe(model=recipe["model"], columns=recipe["features"],
            feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params.get("alpha"), max_leaf_nodes=params.get("max_leaf_nodes"), **early)
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed variable-width magnitude recipe")
