"""Guarded numeric regression dispatch; no source reads, selection or persistence."""
from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting

from .legacy_adapter import forbid_fitting


def fit_regression_candidate(recipe, train_features, train_labels, prediction_features, *, columns):
    """Delegate one fit unchanged; the closed family validator owns policy checks."""
    forbid_fitting("fit_regression_candidate")
    forbid_v2_fitting("fit_regression_candidate")
    if type(recipe) is not dict:
        raise ValueError("regression candidate requires a supported recipe schema/model pair")
    schema, model = recipe.get("schema_version"), recipe.get("model")
    if type(schema) is not str or type(model) is not str:
        raise ValueError("regression candidate requires a supported recipe schema/model pair")
    if schema == "regression_elasticnet_recipe.v1" and model == "elasticnet":
        from .magnitude_elasticnet import fit_regression_elasticnet_predictions as fit
    elif schema == "regression_ridge_hgb_recipe.v1" and model in ("history_ridge", "history_hgb"):
        from .magnitude_ridge_hgb import fit_regression_predictions as fit
    else:
        raise ValueError("regression candidate requires a supported recipe schema/model pair")
    return fit(recipe, train_features, train_labels, prediction_features, columns=columns)
