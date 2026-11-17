"""One fresh ElasticNet fit on caller-admitted, explicitly selected matrices."""
from __future__ import annotations

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash

from . import _magnitude_elasticnet_spline_fit as spline_fit
from ._magnitude_elasticnet_estimator import ConvergenceWarning, build_estimator
from .fit_inputs import FitInputs, check_fitting_allowed

__all__ = ["fit_elasticnet_predictions", "fit_regression_elasticnet_predictions"]


def _matrices(inputs, width, row_error, *, nonnegative=True):
    train, labels, prediction = (np.array(value, dtype=float, copy=True)
        for value in (inputs.train_features, inputs.train_labels, inputs.prediction_features))
    for matrix in (train, prediction):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != width
                or np.isinf(matrix).any()):
            raise ValueError("ElasticNet requires nonempty selected-column matrices without infinity")
    if labels.ndim > 0:
        inputs.validate_row_count(len(train), len(labels), row_error)
    if (labels.shape != (len(train),) or not np.isfinite(labels).all()
            or (nonnegative and (labels < 0).any())):
        domain = "nonnegative " if nonnegative else ""
        raise ValueError(f"ElasticNet labels require one finite {domain}value per training row")
    return train, labels, prediction


def _coefficient_evidence(model, columns, recipe):
    coefficients = np.asarray(model.coef_, dtype=float)
    expected = spline_fit.expected_width(recipe, columns)
    if (coefficients.shape != (expected,) or not np.isfinite(coefficients).all()
            or not np.isfinite(model.intercept_) or not np.isfinite(model.dual_gap_)):
        raise ValueError("ElasticNet produced nonfinite fit evidence")
    pairs = spline_fit.coefficient_pairs(recipe, columns)
    entries = [{"column": column, "kind": kind, "coefficient": float(coefficient)}
        for (kind, column), coefficient in zip(pairs, coefficients)]
    selected = {entry["column"] for entry in entries if entry["coefficient"] != 0.0}
    return {"coefficients": entries,
        "coefficient_scale": ("standardized_inputs_log1p_target"
                              if spline_fit.is_log1p(recipe) else "standardized_inputs"),
        "selected_columns": [column for column in columns if column in selected],
        "intercept": float(model.intercept_), "converged": True,
        "n_iter": int(model.n_iter_), "dual_gap": float(model.dual_gap_)}


def fit_elasticnet_predictions(recipe, train_features, train_labels, prediction_features,
                               *, columns) -> dict:
    """No fold selection here: the caller supplies qualified training and a fixed recipe."""
    check_fitting_allowed("engine.v2.models.training.magnitude_elasticnet.fit_elasticnet_predictions")
    recipe = spline_fit.select_validator(recipe)(recipe)
    inputs = FitInputs.from_recipe(
        recipe, train_features, train_labels, prediction_features, columns,
        column_error="ElasticNet matrix columns must exactly match the ordered recipe subset",
        guard_path="engine.v2.models.training.magnitude_elasticnet.fit_elasticnet_predictions")
    return _fit(recipe, inputs, nonnegative=True, result_schema="magnitude_elasticnet_fit.v1",
                row_error="ElasticNet labels require one finite nonnegative value per training row")


def fit_regression_elasticnet_predictions(recipe, train_features, train_labels,
                                          prediction_features, *, columns):
    """Fit an explicit domain-aware recipe without inferring source or target admission."""
    check_fitting_allowed("engine.v2.models.training.magnitude_elasticnet.fit_regression_elasticnet_predictions")
    recipe = spline_fit.select_regression_validator(recipe)(recipe)
    domain = "nonnegative " if recipe["target_domain"] == "nonnegative" else ""
    inputs = FitInputs.from_recipe(
        recipe, train_features, train_labels, prediction_features, columns,
        column_error="ElasticNet matrix columns must exactly match the ordered recipe subset",
        guard_path="engine.v2.models.training.magnitude_elasticnet.fit_regression_elasticnet_predictions")
    return _fit(recipe, inputs,
                nonnegative=recipe["target_domain"] == "nonnegative",
                result_schema="regression_elasticnet_fit.v1",
                row_error=f"ElasticNet labels require one finite {domain}value per training row")


def _fit(recipe, inputs, *, nonnegative, result_schema, row_error):
    train, labels, prediction = _matrices(inputs, len(inputs.columns), row_error,
                                        nonnegative=nonnegative)
    model = build_estimator(recipe)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        elapsed = spline_fit.timed_fit(model, train, labels)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise spline_fit.convergence_error(recipe)
        inner = model.regressor_ if spline_fit.is_log1p(recipe) else model
        evidence = _coefficient_evidence(inner[-1], inputs.columns, recipe)
        predictions = np.asarray(model.predict(prediction), dtype=float)
    if spline_fit.is_spline(recipe):
        spline_fit.log_spline_fit(recipe, elapsed)
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("ElasticNet predictions require one finite value per prediction row")
    result = {"schema_version": result_schema, "recipe_id": content_hash(recipe),
        "predictions": (np.maximum(predictions, 0.0) if nonnegative else predictions).tolist(), **evidence}
    if spline_fit.is_log1p(recipe):
        result["target_transform"] = "log1p"
    return result
