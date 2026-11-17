"""Closed signed/nonnegative MLP constants; no sklearn, no training imports."""
import json

from engine.v2.foundation import canonical_json
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe

__all__ = ["regression_mlp_recipe", "validate_regression_mlp_recipe"]


def regression_mlp_recipe(*, target_domain, columns, feature_dictionary_hash, profile) -> dict:
    """Signed/nonnegative twin of the magnitude MLP recipe; the neural block is unchanged."""
    if type(target_domain) is not str or target_domain not in ("signed", "nonnegative"):
        raise ValueError("regression MLP recipe requires a signed or nonnegative target domain")
    base = mlp_recipe(columns=columns, feature_dictionary_hash=feature_dictionary_hash,
                      profile=profile)
    return {**base, "schema_version": "regression_mlp_recipe.v1",
            "target_domain": target_domain,
            "prediction_projection": "none" if target_domain == "signed" else "clip_zero"}


def validate_regression_mlp_recipe(recipe) -> dict:
    """Return detached canonical metadata; refuse changed policy or extra fields."""
    try:
        json.dumps(recipe, allow_nan=False)  # Reject objects canonical_json would stringify.
        expected = regression_mlp_recipe(
            target_domain=recipe["target_domain"], columns=recipe["features"],
            feature_dictionary_hash=recipe["feature_dictionary_hash"],
            profile=recipe["profile"])
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed regression MLP recipe")
