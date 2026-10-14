"""Closed, original-unit magnitude fits on already-admitted numeric matrices."""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting

from ._magnitude_estimators import build_estimator
from .legacy_adapter import forbid_fitting

__all__ = ["fit_magnitude_predictions"]


def fit_magnitude_predictions(recipe: dict, train_features, train_labels, test_features) -> list[float]:
    """Consume an admitted recipe; fit training-only state and return detached floats."""
    forbid_fitting("engine.v2.models.training.magnitude.fit_magnitude_predictions")
    forbid_v2_fitting("engine.v2.models.training.magnitude.fit_magnitude_predictions")
    kind = recipe["model"]
    if kind not in ("history_ridge", "history_hgb"):
        raise ValueError("unknown fitted magnitude model")
    train, labels, test = (np.asarray(value, dtype=float)
                           for value in (train_features, train_labels, test_features))
    for matrix in (train, test):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != 11
                or np.isinf(matrix).any()):
            raise ValueError("magnitude features require nonempty eleven-column matrices without infinity")
    if labels.shape != (len(train),) or not np.isfinite(labels).all() or (labels < 0).any():
        raise ValueError("magnitude labels require one finite nonnegative value per training row")
    if kind == "history_hgb":
        # sklearn 1.9 binning cannot accept an entirely missing training column.
        empty = np.isnan(train).all(axis=0)
        train = np.where(empty, 0.0, train)
        test = np.where(empty, 0.0, test)
    model = build_estimator(recipe)
    with threadpool_limits(limits=1):
        model.fit(train, labels)
        predictions = np.asarray(model.predict(test), dtype=float)
    if predictions.shape != (len(test),) or not np.isfinite(predictions).all():
        raise ValueError("magnitude predictions require one finite value per evaluation row")
    return np.maximum(predictions, 0.0).tolist()
