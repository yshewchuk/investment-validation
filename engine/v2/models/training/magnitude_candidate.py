"""Shared fixed-nonnegative numeric dispatch without source admission or selection."""
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.no_fit import forbid_fitting

from .legacy_adapter import forbid_fitting as forbid_legacy_fitting


def fit_candidate(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Preserve delegated results and native errors; never retry."""
    forbid_legacy_fitting("fit_candidate")
    forbid_fitting("fit_candidate")
    if magnitude_candidate_family(recipe) == "elasticnet":
        from .magnitude_elasticnet import fit_elasticnet_predictions as fit
    else:
        from .magnitude_ridge_hgb import fit_ridge_hgb_predictions as fit
    return fit(recipe, train_features, train_labels, prediction_features, columns=columns)
