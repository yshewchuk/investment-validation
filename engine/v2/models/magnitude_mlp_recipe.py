"""Closed standalone neural profiles; no training imports or source admission."""
import json
import re

from engine.v2.foundation import canonical_json

__all__ = ["mlp_recipe", "validate_mlp_recipe"]

_PROFILES = {"relu32_l2_0p01": ([32], 0.01),
             "relu64x32_l2_0p01": ([64, 32], 0.01),
             "relu64x32_l2_0p1": ([64, 32], 0.1)}


def mlp_recipe(*, columns, feature_dictionary_hash, profile):
    """Bind one fixed profile and ordered caller-admitted feature subset."""
    if (not isinstance(columns, (list, tuple)) or not columns
            or any(type(name) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*", name)
                   for name in columns) or len(set(columns)) != len(columns)):
        raise ValueError("MLP columns must be ordered unique feature identifiers")
    if (type(feature_dictionary_hash) is not str
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", feature_dictionary_hash)
            or type(profile) is not str or profile not in _PROFILES):
        raise ValueError("MLP requires a canonical dictionary hash and declared profile")
    layers, alpha = _PROFILES[profile]
    return {
        "schema_version": "magnitude_mlp_recipe.v1", "model": "mlp", "profile": profile,
        "features": list(columns), "feature_dictionary_hash": feature_dictionary_hash,
        "estimator": {"class": "MLPRegressor", "params": {
            "hidden_layer_sizes": list(layers), "activation": "relu", "solver": "adam",
            "loss": "squared_error", "alpha": alpha, "batch_size": "auto",
            "learning_rate": "constant", "learning_rate_init": 0.001, "power_t": 0.5,
            "max_iter": 400, "shuffle": True, "random_state": 20260829, "tol": 0.0001,
            "verbose": False, "warm_start": False, "momentum": 0.9,
            "nesterovs_momentum": True, "early_stopping": False, "validation_fraction": 0.1,
            "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-8,
            "n_iter_no_change": 15, "max_fun": 15000}},
        "preprocessing": {
            "variance_filter": {"threshold": 0.0},
            "imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True},
            "union_features": 2 * len(columns), "output_features": "training_nonconstant"},
        "target_transform": {"class": "StandardScaler", "with_mean": True, "with_std": True},
        "prediction_projection": "clip_zero", "fit_scope": "fold_and_horizon",
        "numerical_threads": 1, "component_fits": 1,
        "nonconvergence": "variant_failure_no_retry"}


def validate_mlp_recipe(recipe):
    """Detach canonical metadata; reject undeclared parameters and policy drift."""
    try:
        json.dumps(recipe, allow_nan=False)
        expected = mlp_recipe(columns=recipe["features"],
            feature_dictionary_hash=recipe["feature_dictionary_hash"], profile=recipe["profile"])
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed MLP recipe")
