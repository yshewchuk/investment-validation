"""Pure per-event SIZE blended-selection metrics and tie-band rule."""
import math
from statistics import stdev

__all__ = ["blended_event_score", "history_mean_rows", "selection_tie_band"]


def _event_rows(rows, threshold):
    selected = {}
    for row in rows:
        if row["horizon_sessions"] != 1:
            continue
        event_id = row["event_id"]
        if event_id in selected:
            raise ValueError("lead-1 selection requires one row per event")
        actual = row["absolute_move_pct"]
        predicted = row["predicted_absolute_move_pct"]
        if (type(actual) not in (int, float) or type(predicted) not in (int, float)
                or not math.isfinite(actual) or not math.isfinite(predicted)):
            raise ValueError("selection metrics require finite numeric rows")
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


def history_mean_rows(inputs, validation, *, horizon_sessions=1):
    """Build the comparator from an admitted fold cohort and its journaled targets."""
    training = inputs["training"]
    if not training:
        raise ValueError("history_mean requires admitted training labels")
    fallback = math.fsum(row["absolute_move_pct"] for row in training) / len(training)
    features = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    result = []
    for target in validation["targets"]:
        feature = features[target["event_id"], horizon_sessions]
        predicted = feature["mean_prior_abs_move"] if feature["n_prior"] else fallback
        result.append({"event_id": target["event_id"], "horizon_sessions": horizon_sessions,
            "absolute_move_pct": target["absolute_move_pct"], "predicted_absolute_move_pct": predicted})
    return result


def blended_event_score(candidate_rows, history_rows, *, metric_weights, large_move_threshold_pct):
    """Score aligned one-row-per-event lead-1 forecasts against history_mean."""
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
                raise ValueError("cannot normalize a nonzero candidate error by zero history_mean error")
            normalized[name] = 1.0
        else:
            normalized[name] = numerator / denominator
    blend = math.fsum(metric_weights[name] * normalized[name] for name in normalized)
    return {"metrics": raw, "normalized": normalized, "blend_score": blend}


def selection_tie_band(scores, yearly_scores, *, floor):
    """Return candidate-ID winner and max(sample-SE, floor) diagnostics."""
    best = min(scores, key=lambda row: (row["blend_score"], row["candidate_id"]))
    by_candidate = {row["candidate_id"]: row for row in yearly_scores}
    yearly = by_candidate[best["candidate_id"]]["yearly"]
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
