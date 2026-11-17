"""Closed signed spline-arm constants; no sklearn, no data reads."""
import json

from engine.v2.foundation import canonical_json
from engine.v2.models.magnitude_elasticnet_spline_recipe import elasticnet_spline_recipe

__all__ = ["regression_elasticnet_spline_recipe",
           "validate_regression_elasticnet_spline_recipe"]


def regression_elasticnet_spline_recipe(*, target_domain, columns, feature_dictionary_hash,
                                        alpha, l1_ratio, n_knots) -> dict:
    """Signed twin of the nonnegative spline recipe; the spline block is unchanged."""
    if type(target_domain) is not str or target_domain != "signed":
        raise ValueError("regression spline recipe requires the signed target domain")
    base = elasticnet_spline_recipe(columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                                    alpha=alpha, l1_ratio=l1_ratio, n_knots=n_knots)
    return {**base, "schema_version": "regression_elasticnet_spline_recipe.v1",
            "target_domain": "signed", "prediction_projection": "none"}


def validate_regression_elasticnet_spline_recipe(recipe) -> dict:
    """Return detached canonical metadata; refuse changed policy or extra fields."""
    try:
        json.dumps(recipe, allow_nan=False)  # Reject objects canonical_json would stringify.
        params = recipe["estimator"]["params"]
        expected = regression_elasticnet_spline_recipe(target_domain=recipe["target_domain"],
            columns=recipe["features"], feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params["alpha"], l1_ratio=params["l1_ratio"],
            n_knots=recipe["preprocessing"]["spline"]["n_knots"])
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed regression ElasticNet spline recipe")
