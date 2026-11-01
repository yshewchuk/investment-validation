"""Pure per-event SIZE blended-selection metrics and tie-band rule."""
import math
from collections.abc import Mapping
from statistics import stdev

__all__ = ["blended_event_score", "history_mean_rows", "history_mean_prediction", "selection_tie_band"]


class ZeroComparatorError(ValueError):
    """Raised when a nonzero candidate error is normalized by a zero history_mean error."""


_METRIC_NAMES = ("rmse", "mae", "large_move_underprediction")


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _rows(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    return value


def _field(row, key, name):
    row = _mapping(row, name)
    if key not in row:
        raise ValueError(f"{name} missing {key}")
    return row[key]


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _finite_number(value, name, message=None):
    if type(value) not in (int, float):
        raise ValueError(message or f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(message or f"{name} must be a finite number")
    return value


def _finite_nonnegative(value, name):
    value = _finite_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _finite_positive(value, name):
    value = _finite_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _event_rows(rows, threshold):
    _finite_positive(threshold, "large_move_threshold_pct")
    selected = {}
    for row in _rows(rows, "blended event rows"):
        event_id = _identifier(_field(row, "event_id", "blended event row"),
                               "blended event row event_id")
        horizon = _positive_int(_field(row, "horizon_sessions", "blended event row"),
                                "blended event row horizon_sessions")
        actual = _finite_number(_field(row, "absolute_move_pct", "blended event row"),
                                "blended event row absolute_move_pct",
                                "selection metrics require finite numeric rows")
        predicted = _finite_number(
            _field(row, "predicted_absolute_move_pct", "blended event row"),
            "blended event row predicted_absolute_move_pct",
            "selection metrics require finite numeric rows")
        if horizon != 1:
            continue
        if event_id in selected:
            raise ValueError("lead-1 selection requires one row per event")
        selected[event_id] = (float(actual), float(predicted))
    if not selected:
        raise ValueError("blended selection requires lead-1 event rows")
    return selected


def _metrics(events, threshold):
    errors = [abs(actual - predicted) for actual, predicted in events.values()]
    large = [(actual, predicted) for actual, predicted in events.values() if actual >= threshold]
    under = [max(actual - predicted, 0.0) for actual, predicted in large]
    return {
        "count": len(events),
        "large_move_count": len(large),
        "rmse": math.sqrt(math.fsum(error * error for error in errors) / len(errors)),
        "mae": math.fsum(errors) / len(errors),
        "large_move_underprediction": math.fsum(under) / len(under) if under else 0.0,
    }


def history_mean_prediction(feature, training_mean):
    """Causal numeric fallback only: no target lookup or cross-horizon selection."""
    n_prior = _nonnegative_int(_field(feature, "n_prior", "history_mean feature"), "history_mean n_prior")
    prior = _field(feature, "mean_prior_abs_move", "history_mean feature")
    if prior is not None:
        _finite_number(prior, "history_mean feature mean_prior_abs_move")
    elif n_prior:
        raise ValueError("history_mean prior features require a finite mean prior")
    return prior if n_prior else _finite_number(training_mean, "history_mean training mean")


def history_mean_rows(inputs, validation, *, horizon_sessions=1):
    """Build the comparator from an admitted fold cohort and its journaled targets."""
    _positive_int(horizon_sessions, "horizon_sessions")
    inputs = _mapping(inputs, "history_mean inputs")
    validation = _mapping(validation, "history_mean validation")
    training = _rows(_field(inputs, "training", "history_mean inputs"), "history_mean training")
    if not training:
        raise ValueError("history_mean requires admitted training labels")
    for row in training:
        _finite_number(_field(row, "absolute_move_pct", "history_mean training row"),
                       "history_mean training absolute_move_pct")
    fallback = math.fsum(row["absolute_move_pct"] for row in training) / len(training)
    features = {}
    for row in _rows(_field(inputs, "features", "history_mean inputs"), "history_mean features"):
        event_id = _identifier(_field(row, "event_id", "history_mean feature row"),
                               "history_mean feature event_id")
        feature_horizon = _positive_int(_field(row, "horizon_sessions", "history_mean feature row"),
                                        "history_mean feature horizon_sessions")
        n_prior = _nonnegative_int(_field(row, "n_prior", "history_mean feature row"),
                                   "history_mean feature n_prior")
        mean_prior = _field(row, "mean_prior_abs_move", "history_mean feature row")
        if mean_prior is not None:
            _finite_number(mean_prior, "history_mean feature mean_prior_abs_move")
        elif n_prior:
            raise ValueError("history_mean prior features require a finite mean prior")
        key = (event_id, feature_horizon)
        if key in features:
            raise ValueError("history_mean duplicate feature keys")
        features[key] = row
    result = []
    for target in _rows(_field(validation, "targets", "history_mean validation"),
                        "history_mean targets"):
        event_id = _identifier(_field(target, "event_id", "history_mean target row"),
                               "history_mean target event_id")
        label = _finite_number(_field(target, "absolute_move_pct", "history_mean target row"),
                               "history_mean target absolute_move_pct")
        key = (event_id, horizon_sessions)
        if key not in features:
            raise ValueError("history_mean missing requested feature")
        feature = features[key]
        predicted = history_mean_prediction(feature, fallback)
        result.append({"event_id": event_id, "horizon_sessions": horizon_sessions,
            "absolute_move_pct": label, "predicted_absolute_move_pct": predicted})
    return result


def blended_event_score(candidate_rows, history_rows, *, metric_weights, large_move_threshold_pct):
    """Score aligned one-row-per-event lead-1 forecasts against history_mean."""
    _finite_positive(large_move_threshold_pct, "large_move_threshold_pct")
    weights = _mapping(metric_weights, "metric_weights")
    for name in _METRIC_NAMES:
        _finite_nonnegative(_field(weights, name, "metric_weights"), f"metric_weights {name}")
    candidate = _event_rows(candidate_rows, large_move_threshold_pct)
    history = _event_rows(history_rows, large_move_threshold_pct)
    if candidate.keys() != history.keys() or any(candidate[key][0] != history[key][0] for key in candidate):
        raise ValueError("candidate and history_mean rows must contain the same events and targets")
    raw = _metrics(candidate, large_move_threshold_pct)
    baseline = _metrics(history, large_move_threshold_pct)
    normalized = {}
    for name in ("rmse", "mae", "large_move_underprediction"):
        denominator = baseline[name]
        numerator = raw[name]
        if denominator == 0.0:
            if numerator != 0.0:
                raise ZeroComparatorError("cannot normalize a nonzero candidate error by zero history_mean error")
            normalized[name] = 1.0
        else:
            normalized[name] = numerator / denominator
    blend = math.fsum(metric_weights[name] * normalized[name] for name in normalized)
    return {"metrics": raw, "normalized": normalized, "blend_score": blend}


def selection_tie_band(scores, yearly_scores, *, floor):
    """Return candidate-ID winner and max(sample-SE, floor) diagnostics."""
    _finite_nonnegative(floor, "tie_band_floor")
    score_rows = _rows(scores, "selection score rows")
    if not score_rows:
        raise ValueError("selection tie band requires candidate score rows")
    for row in score_rows:
        _identifier(_field(row, "candidate_id", "selection score row"),
                    "selection score row candidate_id")
        _finite_number(_field(row, "blend_score", "selection score row"),
                       "selection score row blend_score")
    by_candidate = {}
    for row in _rows(yearly_scores, "selection yearly score rows"):
        candidate_id = _identifier(_field(row, "candidate_id", "selection yearly row"),
                                   "selection yearly row candidate_id")
        yearly = _mapping(_field(row, "yearly", "selection yearly row"),
                          "selection yearly row yearly")
        for year in yearly:
            _identifier(year, "selection yearly year")
            _finite_number(_field(yearly[year], "blend_score", "selection yearly year row"),
                           "selection yearly year blend_score")
        by_candidate[candidate_id] = yearly
    best = min(score_rows, key=lambda row: (row["blend_score"], row["candidate_id"]))
    if best["candidate_id"] not in by_candidate:
        raise ValueError("selection tie band missing candidate yearly scores")
    yearly = by_candidate[best["candidate_id"]]
    values = [yearly[year]["blend_score"] for year in sorted(yearly)]
    standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    width = max(standard_error, floor)
    tied = [row for row in scores if row["blend_score"] <= best["blend_score"] + width]
    winner = min(tied, key=lambda row: row["candidate_id"])
    return winner["candidate_id"], {
        "best_blend_score": best["blend_score"],
        "standard_error": standard_error,
        "width": width,
        "tied_candidate_ids": sorted(row["candidate_id"] for row in tied),
    }
