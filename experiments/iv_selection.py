"""Pure signed per-event training-mean blended-selection metrics."""
import math

from experiments.magnitude_search_selection import (
    ZeroComparatorError,
    _field,
    _finite_nonnegative,
    _finite_number,
    _identifier,
    _mapping,
    _positive_int,
    _rows,
)

_SIGNED_METRICS = ("rmse", "mae")


def training_mean_rows(inputs, validation, *, target, horizon_sessions=1):
    """Build the signed comparator from an admitted fold cohort and its journaled targets."""
    _positive_int(horizon_sessions, "horizon_sessions")
    inputs = _mapping(inputs, "training_mean inputs")
    validation = _mapping(validation, "training_mean validation")
    column = _field(target, "column", "training_mean target")
    prediction_column = _field(target, "prediction_column", "training_mean target")
    training = _rows(_field(inputs, "training", "training_mean inputs"),
                     "training_mean training")
    if not training:
        raise ValueError("training_mean requires admitted training labels")
    labels = []
    for row in training:
        labels.append(_finite_number(_field(row, column, "training_mean training row"),
                                     "training_mean training " + column))
    mean = math.fsum(labels) / len(labels)
    result = []
    for row in _rows(_field(validation, "targets", "training_mean validation"),
                     "training_mean targets"):
        event_id = _identifier(_field(row, "event_id", "training_mean target row"),
                               "training_mean target event_id")
        label = _finite_number(_field(row, column, "training_mean target row"),
                               "training_mean target " + column)
        result.append({"event_id": event_id, "horizon_sessions": horizon_sessions,
                       column: label, prediction_column: mean})
    return result


def _events(rows, target):
    column = _field(target, "column", "signed target")
    prediction_column = _field(target, "prediction_column", "signed target")
    selected = {}
    for row in _rows(rows, "signed event rows"):
        event_id = _identifier(_field(row, "event_id", "signed event row"),
                               "signed event row event_id")
        horizon = _positive_int(_field(row, "horizon_sessions", "signed event row"),
                                "signed event row horizon_sessions")
        actual = _finite_number(_field(row, column, "signed event row"),
                                "signed event row " + column,
                                "selection metrics require finite numeric rows")
        predicted = _finite_number(_field(row, prediction_column, "signed event row"),
                                   "signed event row " + prediction_column,
                                   "selection metrics require finite numeric rows")
        if horizon != 1:
            continue
        if event_id in selected:
            raise ValueError("lead-1 selection requires one row per event")
        selected[event_id] = (float(actual), float(predicted))
    if not selected:
        raise ValueError("blended selection requires lead-1 event rows")
    return selected


def _metrics(events):
    errors = [actual - predicted for actual, predicted in events.values()]
    return {
        "count": len(events),
        "rmse": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
        "mae": math.fsum(abs(error) for error in errors) / len(errors),
    }


def signed_event_score(candidate_rows, baseline_rows, *, metric_weights, target):
    """Score aligned signed lead-1 forecasts against the training_mean baseline."""
    weights = _mapping(metric_weights, "metric_weights")
    if set(weights) != set(_SIGNED_METRICS):
        raise ValueError("metric_weights must have exactly rmse and mae")
    for name in _SIGNED_METRICS:
        _finite_nonnegative(_field(weights, name, "metric_weights"),
                            f"metric_weights {name}")
    candidate = _events(candidate_rows, target)
    baseline = _events(baseline_rows, target)
    if (candidate.keys() != baseline.keys()
            or any(candidate[key][0] != baseline[key][0] for key in candidate)):
        raise ValueError("candidate and training_mean rows must contain the same events and targets")
    raw = _metrics(candidate)
    reference = _metrics(baseline)
    normalized = {}
    for name in _SIGNED_METRICS:
        denominator = reference[name]
        numerator = raw[name]
        if denominator == 0.0:
            if numerator != 0.0:
                raise ZeroComparatorError(
                    "cannot normalize a nonzero candidate error by zero training_mean error")
            normalized[name] = 1.0
        else:
            normalized[name] = numerator / denominator
    blend = math.fsum(weights[name] * normalized[name] for name in _SIGNED_METRICS)
    return {"metrics": raw, "normalized": normalized, "blend_score": blend}
