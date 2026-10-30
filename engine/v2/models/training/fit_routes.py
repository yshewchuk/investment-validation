"""Closed numeric execution routes; no registration, source or outcome admission."""
from engine.v2.models.no_fit import forbid_fitting

from .legacy_adapter import forbid_fitting as forbid_legacy_fitting


def fit_execution(route, recipe, train_features, train_labels, prediction_features, *, columns):
    """Delegate one closed route, preserving results/errors without retry or I/O.

    Route names are internal numeric choices, not new persisted execution schemas.
    Original magnitude returns predictions only, matching the candidate envelope.
    """
    forbid_legacy_fitting("fit_execution")
    forbid_fitting("fit_execution")
    if type(route) is not str or route not in ("magnitude", "magnitude_candidate", "regression_candidate"):
        raise ValueError("unsupported numeric fit execution route")
    if route == "magnitude":
        if type(recipe) is not dict or recipe.get("schema_version") != "magnitude_fitted_recipe.v1":
            raise ValueError("original magnitude route requires its admitted recipe schema")
        if not isinstance(columns, (list, tuple)) or list(columns) != recipe.get("features"):
            raise ValueError("magnitude matrix columns must exactly match the ordered recipe subset")
        from .magnitude import fit_magnitude_predictions
        return {"predictions": fit_magnitude_predictions(recipe, train_features, train_labels, prediction_features)}
    if route == "magnitude_candidate":
        from .magnitude_candidate import fit_candidate as fit
    else:
        from .regression_candidate import fit_regression_candidate as fit
    return fit(recipe, train_features, train_labels, prediction_features, columns=columns)
