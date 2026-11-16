"""Closed domain-aware regression metadata; no source admission or fitting."""
import json

from engine.v2.foundation import canonical_json

from .magnitude_ridge_hgb_recipe import ridge_hgb_recipe


def regression_recipe(*, target_domain, model, columns, feature_dictionary_hash,
                      alpha=None, max_leaf_nodes=None, learning_rate=None,
                      min_samples_leaf=None, l2_regularization=None, loss=None,
                      max_iter=None, quantile=None):
    """Bind signed/nonnegative outcomes explicitly, retaining the closed estimators."""
    if type(target_domain) is not str or target_domain not in ("signed", "nonnegative"):
        raise ValueError("regression requires a signed or nonnegative target domain")
    if target_domain == "signed" and loss in ("gamma", "poisson"):
        raise ValueError("signed regression HGB refuses gamma and poisson losses")
    recipe = ridge_hgb_recipe(model=model, columns=columns,
        feature_dictionary_hash=feature_dictionary_hash, alpha=alpha, max_leaf_nodes=max_leaf_nodes,
        learning_rate=learning_rate, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization, loss=loss, max_iter=max_iter, quantile=quantile)
    return {**recipe, "schema_version": "regression_ridge_hgb_recipe.v1",
        "target_domain": target_domain,
        "prediction_projection": "none" if target_domain == "signed" else "clip_zero"}


def validate_regression_recipe(recipe):
    """Return detached canonical policy; domain and projection cannot drift apart."""
    try:
        json.dumps(recipe, allow_nan=False)
        params = recipe["estimator"]["params"]
        optional = {"loss": params.get("loss"), "max_iter": params.get("max_iter"),
                    "quantile": params.get("quantile")}
        if recipe["model"] == "history_hgb" and params.get("early_stopping") is True:
            optional.update(learning_rate=params.get("learning_rate"),
                min_samples_leaf=params.get("min_samples_leaf"),
                l2_regularization=params.get("l2_regularization"))
        expected = regression_recipe(target_domain=recipe["target_domain"], model=recipe["model"],
            columns=recipe["features"], feature_dictionary_hash=recipe["feature_dictionary_hash"],
            alpha=params.get("alpha"), max_leaf_nodes=params.get("max_leaf_nodes"), **optional)
        if canonical_json(recipe) == canonical_json(expected):
            return expected
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError, RecursionError):
        pass
    raise ValueError("invalid closed domain-aware regression recipe")
