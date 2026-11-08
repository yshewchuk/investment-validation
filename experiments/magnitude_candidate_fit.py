"""Lazy compatibility boundary for shared fixed-nonnegative candidate fitting."""
from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.no_fit import forbid_fitting

__all__ = ["fit_candidate"]


def _require_gamma_floor(recipe) -> None:
    estimator = recipe.get("estimator") if type(recipe) is dict else None
    params = estimator.get("params") if type(estimator) is dict else None
    if type(params) is not dict or params.get("loss") != "gamma":
        return
    if recipe.get("target_floor_pct") is None:
        from engine.v2.ops.errors import fail
        raise fail("INVALID_EXPERIMENT_SPEC",
                   "gamma HGB recipes require target_floor_pct")


def fit_candidate(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Preserve guard/envelope refusals before importing training; never retry."""
    forbid_legacy_fitting("fit_candidate")
    forbid_fitting("fit_candidate")
    _require_gamma_floor(recipe)
    magnitude_candidate_family(recipe)
    from engine.v2.models.training.magnitude_candidate import FitInputs, fit_candidate as fit
    inputs = FitInputs(train_features, train_labels, prediction_features, columns)
    return fit(recipe, inputs)
