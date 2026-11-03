"""Source-free per-horizon fits with explicit numeric route and target domain."""
import math
import time
from copy import deepcopy

import numpy as np

from engine.v2.models.training.fit_inputs import FitInputs
from engine.v2.models.training.fit_routes import check_fitting_allowed, fit_execution
from engine.v2.research.horizon_fit_inputs import (
    _matrix,
    _require_target,
    fit_evidence,
    prepare_horizon_inputs,
)


class HorizonInputs(dict):
    """Ordered horizon -> FitInputs mapping with evidence built from the same rows."""

    def __init__(self, inputs, evidence, target_column):
        super().__init__(inputs)
        self.evidence = evidence
        self.target_column = target_column


class HorizonFitFailure(RuntimeError):
    """Fit/forecast-validation failure; original exception is retained as __cause__."""


def build_fit_inputs(indexed, training_ids, evaluation_ids, horizons, columns, labels, *, target_column, recipe=None):
    """Build one FitInputs per horizon and its frozen evidence from the same ordered rows.

    Assembles per-horizon matrices for bounded offline research fitting after cohort
    admission. This is intentionally not a production inference entrypoint; production
    prediction uses its own model serving path.
    """
    _require_target(target_column)
    training_ids, evaluation_ids = list(training_ids), list(evaluation_ids)
    horizons, columns = list(horizons), list(columns)
    labels = np.array(labels, dtype=float)
    requires_event_dates = isinstance(recipe, dict) and recipe.get("requires_event_dates") is True
    inputs, evidence = {}, {}
    for horizon in horizons:
        training_keys, training_values, training_matrix = _matrix(indexed, training_ids, horizon, columns)
        prediction_keys, prediction_values, prediction_matrix = _matrix(indexed, evaluation_ids, horizon, columns)
        train_event_dates = ([indexed[event, horizon].get("event_date") for event in training_ids]
                             if requires_event_dates else None)
        inputs[horizon] = FitInputs(training_matrix, labels.copy(), prediction_matrix, columns,
            train_event_dates=train_event_dates)
        evidence[horizon] = fit_evidence(columns, training_values, prediction_values, horizon,
            training_ids, evaluation_ids, training_keys, prediction_keys,
            target_column=target_column, labels=labels.tolist(),
            train_event_dates=train_event_dates)
    return HorizonInputs(inputs, evidence, target_column)


def _prepare_inputs(horizon, horizon_inputs, indexed, training_ids, evaluation_ids, columns, labels, target_column, recipe):
    if horizon_inputs is not None:
        requires_event_dates = isinstance(recipe, dict) and recipe.get("requires_event_dates") is True
        if requires_event_dates and horizon_inputs[horizon].train_event_dates is None:
            raise ValueError("prebuilt horizon inputs require training event dates")
        return deepcopy(horizon_inputs.evidence[horizon]), horizon_inputs[horizon]
    requires_event_dates = isinstance(recipe, dict) and recipe.get("requires_event_dates") is True
    train_event_dates = ([indexed[event, horizon].get("event_date") for event in training_ids]
                         if requires_event_dates else None)
    matrix, prediction_matrix, fit = prepare_horizon_inputs(indexed, training_ids, evaluation_ids,
        horizon, columns, labels, target_column=target_column, train_event_dates=train_event_dates)
    return fit, FitInputs(matrix, labels.copy(), prediction_matrix, columns,
        train_event_dates=train_event_dates)


def _run_fit(route, recipe, inputs, count, target_domain, measure_fit_seconds):
    fit_started = time.monotonic() if measure_fit_seconds else None
    try:
        result = fit_execution(route, recipe, inputs)
        fit_seconds = time.monotonic() - fit_started if measure_fit_seconds else None
        _validate_forecasts(result["predictions"], count, target_domain)
    except Exception as error:
        raise HorizonFitFailure("horizon estimator failed") from error
    return result, fit_seconds


def fitted_forecasts(indexed, training_ids=None, labels=None, evaluation_ids=None, horizons=None, columns=None, *,
                     target_column=None, target_domain, route, recipe, progress=None, diagnostics=None,
                     measure_fit_seconds=False):
    """Fit admitted training-only arrays; caller owns sealed source/target identity.

    Preparation/progress errors propagate directly. Fit failures are wrapped only
    to let compatibility boundaries retain their own error policy. No partial
    predictions/evidence return on failure; callbacks/diagnostics are provisional.
    """
    check_fitting_allowed("fitted_forecasts")
    _validate_domain(route, recipe, target_domain)
    horizon_inputs = indexed if isinstance(indexed, HorizonInputs) else None
    if horizon_inputs is not None:
        evaluation_ids, horizons = training_ids, labels
        target_column = horizon_inputs.target_column
    else:
        training_ids, evaluation_ids = list(training_ids), list(evaluation_ids)
        horizons, columns = list(horizons), list(columns)
        labels = np.asarray(labels, dtype=float)
    evaluation_ids, horizons = list(evaluation_ids), list(horizons)
    if horizon_inputs is not None:
        _validate_prebuilt_identity(horizon_inputs, evaluation_ids, horizons)
    predictions, evidence = {}, []
    for horizon in horizons:
        fit, inputs = _prepare_inputs(horizon, horizon_inputs, indexed, training_ids, evaluation_ids,
            columns, labels, target_column, recipe)
        if progress is not None:
            progress("started", deepcopy(fit))
        result, fit_seconds = _run_fit(route, recipe, inputs, len(evaluation_ids), target_domain, measure_fit_seconds)
        values = result["predictions"]
        if diagnostics is not None and route != "magnitude":
            diagnostics.append({"horizon_sessions": horizon, **{k: v for k, v in result.items() if k != "predictions"}})
        predictions.update({(event, horizon): float(value) for event, value in zip(evaluation_ids, values)})
        evidence.append(fit)
        if progress is not None:
            completed = ({**deepcopy(fit), "fit_seconds": fit_seconds} if measure_fit_seconds
                         else deepcopy(fit))
            progress("completed", completed)
    return predictions, evidence


def _validate_prebuilt_identity(horizon_inputs, evaluation_ids, horizons):
    if horizons != list(horizon_inputs):
        raise ValueError("requested horizons do not match prebuilt fit inputs")
    for horizon in horizons:
        if horizon_inputs.evidence[horizon]["evaluation_event_ids"] != evaluation_ids:
            raise ValueError("requested evaluation IDs do not match prebuilt fit evidence")


def _validate_domain(route, recipe, target_domain):
    if (type(target_domain) is not str or target_domain not in ("signed", "nonnegative")
            or type(route) is not str or route not in ("magnitude", "magnitude_candidate", "regression_candidate")):
        raise ValueError("horizon fits require an explicit closed target domain and numeric route")
    if route == "regression_candidate":
        if type(recipe) is not dict or recipe.get("target_domain") != target_domain:
            raise ValueError("regression recipe and horizon target domain differ")
    elif target_domain != "nonnegative":
        raise ValueError("magnitude routes require the nonnegative target domain")


def _validate_forecasts(values, count, target_domain):
    if (len(values) != count or any(isinstance(value, bool)
            or not isinstance(value, (int, float)) or not math.isfinite(value)
            or (target_domain == "nonnegative" and value < 0) for value in values)):
        raise ValueError
