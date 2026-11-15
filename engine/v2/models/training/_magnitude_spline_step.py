"""Raw passthrough plus a checked quantile B-spline basis."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import SplineTransformer

from engine.v2.data.errors import fail

__all__ = ["distinct_knots", "spline_step"]


def distinct_knots(column, n_knots):
    """Count distinct knots using SplineTransformer's quantile rule."""
    return len(np.unique(np.percentile(column, np.linspace(0, 100, n_knots))))


class _CheckedSpline(SplineTransformer):
    """Refuse training columns that cannot supply distinct quantile knots."""

    def fit(self, X, y=None, sample_weight=None):
        X = np.asarray(X, dtype=float)
        for index in range(X.shape[1]):
            if distinct_knots(X[:, index], self.n_knots) < self.n_knots:
                raise fail("EXPERIMENT_VARIANT_FAILED",
                           f"spline column {index} has fewer than {self.n_knots} "
                           "distinct quantile knots")
        return super().fit(X, y, sample_weight)

    def transform(self, X):
        out = super().transform(X)
        if not np.isfinite(out).all():
            raise fail("EXPERIMENT_VARIANT_FAILED", "spline basis is not finite")
        return out


def spline_step(feature_names, spline_columns, n_knots):
    """Pass imputed features through and append a checked spline basis."""
    position = {name: index for index, name in enumerate(feature_names)}
    missing = [name for name in spline_columns if name not in position]
    if missing:
        raise fail("CONTRACT_MISMATCH",
                   "spline columns missing from imputer output: " + ", ".join(missing))
    return ColumnTransformer([
        ("raw", "passthrough", list(range(len(feature_names)))),
        ("spline", _CheckedSpline(n_knots=n_knots, degree=3, knots="quantile",
                                  extrapolation="constant", include_bias=False),
         [position[name] for name in spline_columns]),
    ], n_jobs=1)
