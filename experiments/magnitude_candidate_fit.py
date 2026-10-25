"""One numeric family fit, without source admission, orchestration or selection."""
from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.models.no_fit import forbid_fitting

__all__ = ["fit_candidate"]


def fit_candidate(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Preserve the delegated family's result and native errors; never retry."""
    forbid_legacy_fitting("fit_candidate")
    forbid_fitting("fit_candidate")
    if type(recipe) is not dict:
        raise ValueError("candidate requires a supported recipe schema/model pair")
    schema, model = recipe.get("schema_version"), recipe.get("model")
    if type(schema) is not str or type(model) is not str:
        raise ValueError("candidate requires a supported recipe schema/model pair")
    if schema == "magnitude_elasticnet_recipe.v1" and model == "elasticnet":
        from engine.v2.models.training.magnitude_elasticnet import fit_elasticnet_predictions as fit
    elif schema == "magnitude_ridge_hgb_recipe.v1" and model in ("history_ridge", "history_hgb"):
        from engine.v2.models.training.magnitude_ridge_hgb import fit_ridge_hgb_predictions as fit
    else:
        raise ValueError("candidate requires a supported recipe schema/model pair")
    return fit(recipe, train_features, train_labels, prediction_features, columns=columns)
