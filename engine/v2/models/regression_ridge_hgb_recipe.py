"""Closed domain-aware regression metadata; no source admission or fitting."""
import json

from engine.v2.foundation import canonical_json

from .magnitude_ridge_hgb_recipe import ridge_hgb_recipe


def regression_recipe(*, target_domain, model, columns, feature_dictionary_hash,
                      alpha=None, max_leaf_nodes=None):
    """Bind signed/nonnegative outcomes explicitly, retaining the closed estimators."""
    if type(target_domain) is not str or target_domain not in ("signed", "nonnegative"):
        raise ValueError("regression requires a signed or nonnegative target domain")
    recipe = ridge_hgb_recipe(model=model, columns=columns,
        feature_dictionary_hash=feature_dictionary_hash, alpha=alpha, max_leaf_nodes=max_leaf_nodes)
    return {**recipe, "schema_version": "regression_ridge_hgb_recipe.v1",
        "target_domain": target_domain,
        "prediction_projection": "none" if target_domain == "signed" else "clip_zero"}


def validate_regression_recipe(recipe):
    """Return detached canonical policy; domain and projection cannot drift apart."""
    try:
        json.dumps(recipe, allow_nan=False)
        params = recipe["estimator"]["params"]
        expected = regression_recipe(target_domain=recipe["target_domain"], model=recipe["model"],
            columns=recipe["features"], feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params.get("alpha"), max_leaf_nodes=params.get("max_leaf_nodes"))
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed domain-aware regression recipe")
