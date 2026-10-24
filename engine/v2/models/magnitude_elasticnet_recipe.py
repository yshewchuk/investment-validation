"""Closed ElasticNet metadata; no training, sklearn, data reads or admission."""
from __future__ import annotations

import json
import re

from engine.v2.foundation import canonical_json

__all__ = ["elasticnet_recipe", "validate_elasticnet_recipe"]


def elasticnet_recipe(*, columns, feature_dictionary_hash, alpha, l1_ratio) -> dict:
    """Bind a caller-admitted ordered subset; its dictionary membership is external."""
    if (not isinstance(columns, (list, tuple)) or not columns
            or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                   for name in columns) or len(set(columns)) != len(columns)):
        raise ValueError("ElasticNet columns must be ordered unique feature identifiers")
    if (not isinstance(feature_dictionary_hash, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", feature_dictionary_hash)):
        raise ValueError("ElasticNet requires a canonical feature dictionary hash")
    if (type(alpha) not in (int, float) or alpha not in (0.01, 0.1, 1.0)
            or type(l1_ratio) not in (int, float) or l1_ratio not in (0.25, 0.75)):
        raise ValueError("ElasticNet parameters must belong to the declared six combinations")
    return {
        "schema_version": "magnitude_elasticnet_recipe.v1", "model": "elasticnet",
        "features": list(columns), "feature_dictionary_hash": feature_dictionary_hash,
        "estimator": {"class": "ElasticNet", "params": {
            "alpha": float(alpha), "l1_ratio": float(l1_ratio), "fit_intercept": True,
            "precompute": False, "max_iter": 10000, "copy_X": True, "tol": 0.0001,
            "warm_start": False, "positive": False, "random_state": 0, "selection": "cyclic",
        }},
        "preprocessing": {
            "imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True},
            "output_features": 2 * len(columns),
        },
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1,
        "nonconvergence": "variant_failure_no_retry",
    }


def validate_elasticnet_recipe(recipe) -> dict:
    """Return detached canonical metadata; refuse changed policy or extra fields."""
    try:
        json.dumps(recipe, allow_nan=False)  # Reject objects canonical_json would stringify.
        params = recipe["estimator"]["params"]
        expected = elasticnet_recipe(columns=recipe["features"],
            feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params["alpha"], l1_ratio=params["l1_ratio"])
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed ElasticNet recipe")
