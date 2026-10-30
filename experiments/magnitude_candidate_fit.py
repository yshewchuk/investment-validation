"""Lazy compatibility boundary for shared fixed-nonnegative candidate fitting."""
from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.no_fit import forbid_fitting

__all__ = ["fit_candidate"]


def fit_candidate(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Preserve guard/envelope refusals before importing training; never retry."""
    forbid_legacy_fitting("fit_candidate")
    forbid_fitting("fit_candidate")
    magnitude_candidate_family(recipe)
    from engine.v2.models.training.magnitude_candidate import fit_candidate as fit
    return fit(recipe, train_features, train_labels, prediction_features, columns=columns)
