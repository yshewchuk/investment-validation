"""Source-free per-horizon fits with explicit numeric route and target domain."""
import math
import time
from copy import deepcopy

import numpy as np

from engine.v2.models.training.fit_routes import check_fitting_allowed, fit_execution
from engine.v2.research.horizon_fit_inputs import prepare_horizon_inputs


class HorizonFitFailure(RuntimeError):
    """Fit/forecast-validation failure; original exception is retained as __cause__."""


def fitted_forecasts(indexed, training_ids, labels, evaluation_ids, horizons, columns, *,
                     target_column, target_domain, route, recipe, progress=None, diagnostics=None,
                     measure_fit_seconds=False):
    """Fit admitted training-only arrays; caller owns sealed source/target identity.

    Preparation/progress errors propagate directly. Fit failures are wrapped only
    to let compatibility boundaries retain their own error policy. No partial
    predictions/evidence return on failure; callbacks/diagnostics are provisional.
    """
    check_fitting_allowed("fitted_forecasts")
    _validate_domain(route, recipe, target_domain)
    training_ids, evaluation_ids = list(training_ids), list(evaluation_ids)
    horizons, columns = list(horizons), list(columns)
    labels = np.asarray(labels, dtype=float)
    predictions, evidence = {}, []
    for horizon in horizons:
        matrix, prediction_matrix, fit = prepare_horizon_inputs(indexed, training_ids, evaluation_ids,
            horizon, columns, labels, target_column=target_column)
        if progress is not None:
            progress("started", deepcopy(fit))
        try:
            fit_started = time.monotonic() if measure_fit_seconds else None
            result = fit_execution(route, recipe, matrix, labels.copy(), prediction_matrix, columns=columns)
            fit_seconds = time.monotonic() - fit_started if measure_fit_seconds else None
            values = result["predictions"]
            if diagnostics is not None and route != "magnitude":
                diagnostics.append({"horizon_sessions": horizon, **{k: v for k, v in result.items() if k != "predictions"}})
            _validate_forecasts(values, len(evaluation_ids), target_domain)
        except Exception as error:
            raise HorizonFitFailure("horizon estimator failed") from error
        predictions.update({(event, horizon): float(value) for event, value in zip(evaluation_ids, values)})
        evidence.append(fit)
        if progress is not None:
            completed = ({**deepcopy(fit), "fit_seconds": fit_seconds} if measure_fit_seconds
                         else deepcopy(fit))
            progress("completed", completed)
    return predictions, evidence


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
