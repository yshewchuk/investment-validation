"""Closed variable-width Ridge/HGB metadata, without training or source admission."""
from __future__ import annotations

import json
import math
import re

from engine.v2.foundation import canonical_json

__all__ = ["ridge_hgb_recipe", "validate_ridge_hgb_recipe"]

_RIDGE_ALPHA_BOUNDS = (1e-6, 1000.0)


def _bounded_alpha(value) -> bool:
    lower, upper = _RIDGE_ALPHA_BOUNDS
    if type(value) not in (int, float) or not lower <= value <= upper:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _parameters(model, alpha, max_leaf_nodes):
    if model == "history_ridge":
        if not _bounded_alpha(alpha) or max_leaf_nodes is not None:
            raise ValueError("Ridge requires declared alpha and no HGB parameter")
        return {"alpha": float(alpha), "fit_intercept": True, "copy_X": True,
                "max_iter": None, "tol": 0.0001, "solver": "svd", "positive": False,
                "random_state": 0}
    if model != "history_hgb":
        raise ValueError("unknown variable-width magnitude model")
    if type(max_leaf_nodes) is not int or max_leaf_nodes not in (7, 15) or alpha is not None:
        raise ValueError("HGB requires declared integer leaf count and no Ridge parameter")
    return {"loss": "squared_error", "quantile": None, "learning_rate": 0.1,
        "max_iter": 100, "max_leaf_nodes": max_leaf_nodes, "max_depth": None,
        "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0,
        "max_bins": 255, "categorical_features": None, "monotonic_cst": None,
        "interaction_cst": None, "warm_start": False, "early_stopping": False,
        "scoring": "loss", "validation_fraction": 0.1, "n_iter_no_change": 10,
        "tol": 1e-7, "verbose": 0, "random_state": 0}


def ridge_hgb_recipe(*, model, columns, feature_dictionary_hash, alpha=None, max_leaf_nodes=None) -> dict:
    """Bind a caller-admitted ordered subset; dictionary membership stays external."""
    if (not isinstance(columns, (list, tuple)) or not columns
            or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                   for name in columns) or len(set(columns)) != len(columns)):
        raise ValueError("magnitude columns must be ordered unique feature identifiers")
    if (not isinstance(feature_dictionary_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", feature_dictionary_hash)):
        raise ValueError("magnitude requires a canonical feature dictionary hash")
    params = _parameters(model, alpha, max_leaf_nodes)
    ridge = model == "history_ridge"
    return {
        "schema_version": "magnitude_ridge_hgb_recipe.v1", "model": model,
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
        expected = ridge_hgb_recipe(model=recipe["model"], columns=recipe["features"],
            feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params.get("alpha"), max_leaf_nodes=params.get("max_leaf_nodes"))
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed variable-width magnitude recipe")
