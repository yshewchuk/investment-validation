"""Closed numeric execution routes; no registration, source or outcome admission."""
from datetime import date

from engine.v2.foundation import content_hash
from engine.v2.models.no_fit import forbid_fitting

from .legacy_adapter import forbid_fitting as forbid_legacy_fitting


def check_fitting_allowed(path):
    """Check both existing switches before an orchestrator inspects numeric inputs."""
    forbid_legacy_fitting(path)
    forbid_fitting(path)


def _event_date_metadata(recipe, event_dates, row_count):
    if not recipe.get("requires_event_dates", False) and event_dates is None:
        return {}
    if recipe.get("requires_event_dates", False) is not True:
        raise ValueError("MLP event dates require requires_event_dates=True")
    if not isinstance(event_dates, (list, tuple)):
        raise ValueError("MLP event dates must be a list or tuple")
    if len(event_dates) != row_count:
        raise ValueError("MLP event dates must align with training rows")
    for value in event_dates:
        if not isinstance(value, str):
            raise ValueError("MLP event dates must be canonical ISO date strings")
        try:
            canonical = date.fromisoformat(value).isoformat() == value
        except ValueError:
            canonical = False
        if not canonical:
            raise ValueError("MLP event dates must be canonical ISO date strings")
    return {"training_event_dates_id": content_hash(list(event_dates))}


def fit_execution(route, recipe, train_features, train_labels, prediction_features, *, columns,
                  event_dates=None):
    """Delegate one closed route, preserving results/errors without retry or I/O.

    Route names are internal numeric choices, not new persisted execution schemas.
    Original magnitude returns predictions only, matching the candidate envelope.
    """
    check_fitting_allowed("fit_execution")
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
        options = {"columns": columns}
        if event_dates is not None:
            options["event_dates"] = event_dates
        return fit(recipe, train_features, train_labels, prediction_features, **options)
    from .regression_candidate import fit_regression_candidate as fit
    return fit(recipe, train_features, train_labels, prediction_features, columns=columns)
