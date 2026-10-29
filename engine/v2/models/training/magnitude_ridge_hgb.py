"""One source-free variable-width Ridge/HGB fit on explicitly selected matrices."""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_ridge_hgb_recipe import validate_ridge_hgb_recipe
from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting
from engine.v2.models.regression_ridge_hgb_recipe import validate_regression_recipe

from ._magnitude_estimators import build_estimator
from .legacy_adapter import forbid_fitting

__all__ = ["fit_ridge_hgb_predictions", "fit_regression_predictions"]


def _matrices(train_features, train_labels, prediction_features, width, *, nonnegative=True):
    try:
        train, labels, prediction = (np.array(value, dtype=float, copy=True)
            for value in (train_features, train_labels, prediction_features))
    except (TypeError, OverflowError) as exc:
        raise ValueError("magnitude inputs must be convertible to floating-point values") from exc
    for matrix in (train, prediction):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != width
                or np.isinf(matrix).any()):
            raise ValueError("magnitude requires nonempty selected-column matrices without infinity")
    if (labels.shape != (len(train),) or not np.isfinite(labels).all()
            or (nonnegative and (labels < 0).any())):
        domain = "nonnegative " if nonnegative else ""
        raise ValueError(f"magnitude labels require one finite {domain}value per training row")
    return train, labels, prediction


def fit_ridge_hgb_predictions(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Caller owns source/fold admission and truthful columns; no selection or I/O."""
    forbid_fitting("engine.v2.models.training.magnitude_ridge_hgb.fit_ridge_hgb_predictions")
    forbid_v2_fitting("engine.v2.models.training.magnitude_ridge_hgb.fit_ridge_hgb_predictions")
    recipe = validate_ridge_hgb_recipe(recipe)
    return _fit(recipe, train_features, train_labels, prediction_features, columns=columns,
                nonnegative=True, result_schema="magnitude_ridge_hgb_fit.v1")


def fit_regression_predictions(recipe, train_features, train_labels, prediction_features, *, columns):
    """Fit an explicit domain-aware recipe; no target identity or admission is inferred."""
    forbid_fitting("engine.v2.models.training.magnitude_ridge_hgb.fit_regression_predictions")
    forbid_v2_fitting("engine.v2.models.training.magnitude_ridge_hgb.fit_regression_predictions")
    recipe = validate_regression_recipe(recipe)
    return _fit(recipe, train_features, train_labels, prediction_features, columns=columns,
                nonnegative=recipe["target_domain"] == "nonnegative",
                result_schema="regression_ridge_hgb_fit.v1")


def _fit(recipe, train_features, train_labels, prediction_features, *, columns, nonnegative, result_schema):
    if not isinstance(columns, (list, tuple)) or list(columns) != recipe["features"]:
        raise ValueError("magnitude matrix columns must exactly match the ordered recipe subset")
    train, labels, prediction = _matrices(train_features, train_labels, prediction_features, len(columns),
                                        nonnegative=nonnegative)
    if recipe["model"] == "history_hgb":
        # Freeze only training-all-null columns; partial NaNs retain native routing.
        empty = np.isnan(train).all(axis=0)
        train[:, empty] = 0.0
        prediction[:, empty] = 0.0
    model = build_estimator(recipe)
    with threadpool_limits(limits=1):
        model.fit(train, labels)
        predictions = np.asarray(model.predict(prediction), dtype=float)
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("magnitude predictions require one finite value per prediction row")
    return {"schema_version": result_schema, "recipe_id": content_hash(recipe),
            "predictions": (np.maximum(predictions, 0.0) if nonnegative else predictions).tolist()}
