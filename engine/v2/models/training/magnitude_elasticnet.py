"""One fresh ElasticNet fit on caller-admitted, explicitly selected matrices."""
from __future__ import annotations

import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_elasticnet_recipe import (
    validate_elasticnet_recipe,
    validate_regression_elasticnet_recipe,
)
from engine.v2.models.no_fit import forbid_fitting as forbid_v2_fitting

from ._magnitude_elasticnet_estimator import ConvergenceWarning, build_estimator
from .legacy_adapter import forbid_fitting

__all__ = ["fit_elasticnet_predictions", "fit_regression_elasticnet_predictions"]


def _matrices(train_features, train_labels, prediction_features, width, *, nonnegative=True):
    train, labels, prediction = (np.array(value, dtype=float, copy=True)
        for value in (train_features, train_labels, prediction_features))
    for matrix in (train, prediction):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != width
                or np.isinf(matrix).any()):
            raise ValueError("ElasticNet requires nonempty selected-column matrices without infinity")
    if (labels.shape != (len(train),) or not np.isfinite(labels).all()
            or (nonnegative and (labels < 0).any())):
        domain = "nonnegative " if nonnegative else ""
        raise ValueError(f"ElasticNet labels require one finite {domain}value per training row")
    return train, labels, prediction


def _coefficient_evidence(model, columns):
    coefficients = np.asarray(model.coef_, dtype=float)
    if (coefficients.shape != (2 * len(columns),) or not np.isfinite(coefficients).all()
            or not np.isfinite(model.intercept_) or not np.isfinite(model.dual_gap_)):
        raise ValueError("ElasticNet produced nonfinite fit evidence")
    entries = [{"column": column, "kind": kind, "coefficient": float(coefficient)}
        for (kind, column), coefficient in zip(
            [(kind, column) for kind in ("value", "missing_indicator") for column in columns],
            coefficients)]
    selected = {entry["column"] for entry in entries if entry["coefficient"] != 0.0}
    return {"coefficients": entries, "coefficient_scale": "standardized_inputs",
        "selected_columns": [column for column in columns if column in selected],
        "intercept": float(model.intercept_), "converged": True,
        "n_iter": int(model.n_iter_), "dual_gap": float(model.dual_gap_)}


def fit_elasticnet_predictions(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """No fold selection here: the caller supplies qualified training and a fixed recipe."""
    forbid_fitting("engine.v2.models.training.magnitude_elasticnet.fit_elasticnet_predictions")
    forbid_v2_fitting("engine.v2.models.training.magnitude_elasticnet.fit_elasticnet_predictions")
    recipe = validate_elasticnet_recipe(recipe)
    return _fit(recipe, train_features, train_labels, prediction_features, columns=columns,
                nonnegative=True, result_schema="magnitude_elasticnet_fit.v1")


def fit_regression_elasticnet_predictions(recipe, train_features, train_labels, prediction_features, *, columns):
    """Fit an explicit domain-aware recipe without inferring source or target admission."""
    forbid_fitting("engine.v2.models.training.magnitude_elasticnet.fit_regression_elasticnet_predictions")
    forbid_v2_fitting("engine.v2.models.training.magnitude_elasticnet.fit_regression_elasticnet_predictions")
    recipe = validate_regression_elasticnet_recipe(recipe)
    return _fit(recipe, train_features, train_labels, prediction_features, columns=columns,
                nonnegative=recipe["target_domain"] == "nonnegative",
                result_schema="regression_elasticnet_fit.v1")


def _fit(recipe, train_features, train_labels, prediction_features, *, columns, nonnegative, result_schema):
    if not isinstance(columns, (list, tuple)) or list(columns) != recipe["features"]:
        raise ValueError("ElasticNet matrix columns must exactly match the ordered recipe subset")
    train, labels, prediction = _matrices(train_features, train_labels, prediction_features, len(columns),
                                        nonnegative=nonnegative)
    model = build_estimator(recipe)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train, labels)
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
            raise ValueError("ElasticNet convergence failed; no retry permitted")
        evidence = _coefficient_evidence(model[-1], columns)
        predictions = np.asarray(model.predict(prediction), dtype=float)
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("ElasticNet predictions require one finite value per prediction row")
    return {"schema_version": result_schema, "recipe_id": content_hash(recipe),
        "predictions": (np.maximum(predictions, 0.0) if nonnegative else predictions).tolist(), **evidence}
