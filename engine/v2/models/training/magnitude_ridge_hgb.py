"""One source-free variable-width Ridge/HGB fit on explicitly selected matrices."""
from __future__ import annotations

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_ridge_hgb_recipe import validate_ridge_hgb_recipe
from engine.v2.models.regression_ridge_hgb_recipe import validate_regression_recipe

from ._magnitude_estimators import build_estimator
from .fit_inputs import FitInputs, check_fitting_allowed

__all__ = ["fit_ridge_hgb_predictions", "fit_regression_predictions"]


def _matrices(inputs, width, row_error, *, nonnegative=True):
    try:
        train, labels, prediction = (np.array(value, dtype=float, copy=True)
            for value in (inputs.train_features, inputs.train_labels, inputs.prediction_features))
    except (TypeError, OverflowError) as exc:
        raise ValueError("magnitude inputs must be convertible to floating-point values") from exc
    for matrix in (train, prediction):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != width
                or np.isinf(matrix).any()):
            raise ValueError("magnitude requires nonempty selected-column matrices without infinity")
    if labels.ndim > 0:
        inputs.validate_row_count(len(train), len(labels), row_error)
    if (labels.shape != (len(train),) or not np.isfinite(labels).all()
            or (nonnegative and (labels < 0).any())):
        domain = "nonnegative " if nonnegative else ""
        raise ValueError(f"magnitude labels require one finite {domain}value per training row")
    return train, labels, prediction


def fit_ridge_hgb_predictions(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Caller owns source/fold admission and truthful columns; no selection or I/O."""
    check_fitting_allowed("engine.v2.models.training.magnitude_ridge_hgb.fit_ridge_hgb_predictions")
    recipe = validate_ridge_hgb_recipe(recipe)
    inputs = FitInputs.from_recipe(
        recipe, train_features, train_labels, prediction_features, columns,
        column_error="magnitude matrix columns must exactly match the ordered recipe subset",
        guard_path="engine.v2.models.training.magnitude_ridge_hgb.fit_ridge_hgb_predictions")
    return _fit(recipe, inputs, nonnegative=True, result_schema="magnitude_ridge_hgb_fit.v1",
                row_error="magnitude labels require one finite nonnegative value per training row")


def fit_regression_predictions(recipe, train_features, train_labels, prediction_features, *, columns):
    """Fit an explicit domain-aware recipe; no target identity or admission is inferred."""
    check_fitting_allowed("engine.v2.models.training.magnitude_ridge_hgb.fit_regression_predictions")
    recipe = validate_regression_recipe(recipe)
    domain = "nonnegative " if recipe["target_domain"] == "nonnegative" else ""
    inputs = FitInputs.from_recipe(
        recipe, train_features, train_labels, prediction_features, columns,
        column_error="magnitude matrix columns must exactly match the ordered recipe subset",
        guard_path="engine.v2.models.training.magnitude_ridge_hgb.fit_regression_predictions")
    return _fit(recipe, inputs,
                nonnegative=recipe["target_domain"] == "nonnegative",
                result_schema="regression_ridge_hgb_fit.v1",
                row_error=f"magnitude labels require one finite {domain}value per training row")


def _fit(recipe, inputs, *, nonnegative, result_schema, row_error):
    train, labels, prediction = _matrices(inputs, len(inputs.columns), row_error,
                                        nonnegative=nonnegative)
    if recipe["model"] == "history_hgb":
        # Freeze only training-all-null columns; partial NaNs retain native routing.
        empty = np.isnan(train).all(axis=0)
        train[:, empty] = 0.0
        prediction[:, empty] = 0.0
    floor = recipe.get("target_floor_pct")
    if floor is not None:
        labels = np.maximum(labels, floor)
    model = build_estimator(recipe)
    with threadpool_limits(limits=1):
        model.fit(train, labels)
        predictions = np.asarray(model.predict(prediction), dtype=float)
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("magnitude predictions require one finite value per prediction row")
    return {"schema_version": result_schema, "recipe_id": content_hash(recipe),
            "predictions": (np.maximum(predictions, 0.0) if nonnegative else predictions).tolist()}
