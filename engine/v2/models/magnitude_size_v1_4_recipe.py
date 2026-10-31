"""Closed metadata for the causal SIZE v1.4 OLS+MLP blend; not source admission or fitting."""
from __future__ import annotations

import copy

__all__ = ["FEATURES", "size_v1_4_recipe"]

FEATURES = (
    "has_implied_quote",
    "mean_prior_abs_move",
    "abs_dist_ema",
    "ema12r_abs",
    "mean_prior_move",
    "signed_streak",
    "dist_high",
    "dist_ema",
    "spy_vol20",
    "spy_dd252",
    "mean_prior_or_implied",
    "or_implied",
    "or_rvol30",
    "mcap_log",
)

_INVALID_RECIPE = "invalid closed SIZE v1.4 recipe"


def size_v1_4_recipe(native_recipe: dict, component_parameters: dict) -> dict:
    """Return detached closed SIZE v1.4 blend metadata for the declared feature order."""
    try:
        features_match = (
            type(native_recipe) is dict
            and type(component_parameters) is dict
            and list(native_recipe.get("features", [])) == list(FEATURES)
        )
    except TypeError:
        features_match = False
    if not features_match:
        raise ValueError(_INVALID_RECIPE)
    return {
        "schema_version": "size_causal_comparator.v1",
        "native": copy.deepcopy(native_recipe),
        "component_parameters": copy.deepcopy(component_parameters),
        "features": list(FEATURES),
        "minimum_prior": 4,
        "minimum_training_rows": 500,
        "masks": {
            "or_implied": [0, 60],
            "or_rvol30": [0, 700],
            "training_target": [0, 200],
        },
        "prediction_projection": "none",
        "missing": "abstain",
        "quote_value": "stored_without_reconstruction",
        "refit": "fresh_per_outer_horizon",
        "history_mean": "stored_mean_else_qualified_training_mean",
        "production_reproduction": False,
    }
