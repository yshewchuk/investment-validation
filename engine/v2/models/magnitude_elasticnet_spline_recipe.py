"""Closed spline-arm constants; no sklearn, no data reads."""
import json

from engine.v2.foundation import canonical_json
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe

# Removed for quantile ties at k=6 on the stored v11 panel (design 6(c)): implied_move, tail_share_gt10, imp_chg5, imp_chg10, iv30_chg20; their raw and log columns stay in the feature set.
SPLINE_COLUMNS = (
    "mean_prior_abs_move", "ema2_prior_abs_move", "ema4_prior_abs_move",
    "ema8_prior_abs_move", "ema12_prior_abs_move",
    "iv10", "iv30", "exern_iv10", "exern_iv30", "fwd90_30",
    "fexern90_30", "rvol30", "spy_vol5", "spy_vol20", "spy_vol60",
    "season_heat_mean_abs_5", "season_heat_mean_abs_20",
    "season_heat_ratio_20_252", "tail_max_abs", "tail_std_abs",
    "tail_p75_abs", "tail_p90_abs", "tail_max4_abs",
    "tail_std_over_mean", "tail_max_over_mean", "iv10_30", "iv_rv",
    "ex_ratio", "iv30_chg5",
    "iv10_chg5", "own_rvol5", "own_rvol10", "own_rvol60", "own_absret1",
)

__all__ = ["SPLINE_COLUMNS", "elasticnet_spline_recipe", "validate_elasticnet_spline_recipe"]


def elasticnet_spline_recipe(*, columns, feature_dictionary_hash, alpha, l1_ratio,
                             n_knots) -> dict:
    """Bind the closed ElasticNet policy plus the closed spline transform."""
    if type(n_knots) is not int or not 4 <= n_knots <= 8:
        raise ValueError("spline n_knots must be an integer in 4..8")
    if any(name not in columns for name in SPLINE_COLUMNS):
        raise ValueError("spline columns must be a subset of the feature columns")
    base = elasticnet_recipe(columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                             alpha=alpha, l1_ratio=l1_ratio)
    preprocessing = {
        "imputer": base["preprocessing"]["imputer"],
        "missing_indicator": base["preprocessing"]["missing_indicator"],
        "scaler": base["preprocessing"]["scaler"],
        "spline": {"columns": list(SPLINE_COLUMNS), "n_knots": n_knots, "degree": 3,
                   "knot_rule": "quantile", "extrapolation": "constant",
                   "include_bias": False, "input": "imputed", "output": "append"},
        "output_features": base["preprocessing"]["output_features"]
                            + len(SPLINE_COLUMNS) * (n_knots + 1),
    }
    return {**base, "schema_version": "magnitude_elasticnet_spline_recipe.v1",
            "estimator": {**base["estimator"],
                          "params": {**base["estimator"]["params"], "precompute": True}},
            "preprocessing": preprocessing}


def validate_elasticnet_spline_recipe(recipe) -> dict:
    """Return detached canonical metadata; refuse changed policy or extra fields."""
    try:
        json.dumps(recipe, allow_nan=False)  # Reject objects canonical_json would stringify.
        params = recipe["estimator"]["params"]
        expected = elasticnet_spline_recipe(columns=recipe["features"],
            feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params["alpha"], l1_ratio=params["l1_ratio"],
            n_knots=recipe["preprocessing"]["spline"]["n_knots"])
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed ElasticNet spline recipe")
