"""Source-free regression forecasts and losses; callers own sealed target identity."""
import math
from numbers import Integral, Real
from statistics import mean


def normalize_forecasts(rows, fold, horizons, *, prediction_column, target_domain, strict_json=False):
    """Require exact event/horizon coverage for an explicit signed/nonnegative domain."""
    if (type(prediction_column) is not str or not prediction_column
            or prediction_column in ("event_id", "horizon_sessions")
            or type(target_domain) is not str or target_domain not in ("signed", "nonnegative")):
        raise ValueError("invalid regression forecast policy")
    expected = {(event, horizon) for event in fold["evaluation_event_ids"] for horizon in horizons}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError
    found = {}
    for row in rows:
        pair, value = _forecast(row, prediction_column, target_domain, strict_json)
        if pair not in expected or pair in found:
            raise ValueError
        found[pair] = {"event_id": pair[0], "horizon_sessions": pair[1], prediction_column: value}
    return [found[pair] for pair in sorted(expected)]


def _forecast(row, column, domain, strict_json):
    if not isinstance(row, dict) or set(row) != {"event_id", "horizon_sessions", column}:
        raise ValueError
    horizon, value = row["horizon_sessions"], row[column]
    if (isinstance(horizon, bool) or not isinstance(horizon, Integral)
            or isinstance(value, bool) or not isinstance(value, Real)
            or (domain == "nonnegative" and value < 0)
            or (strict_json and (type(horizon) is not int or type(value) not in (int, float)))):
        raise ValueError
    value = float(value)
    if not math.isfinite(value):
        raise ValueError
    return (row["event_id"], int(horizon)), value


def error_metrics(errors):
    """Pool equal row weights; scale before squaring to avoid finite-error overflow."""
    errors = list(errors)
    if not errors:
        raise ValueError("empty")
    try:
        if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
               for value in errors):
            raise ValueError("nonfinite")
    except OverflowError:
        raise ValueError("nonfinite") from None
    errors = [abs(float(value)) for value in errors]
    scale = max(errors)
    rmse = scale * math.sqrt(mean((error / scale) ** 2 for error in errors)) if scale else 0.0
    mae = mean(errors)
    if not all(math.isfinite(value) for value in (rmse, mae)):
        raise ValueError("nonfinite")
    return {"count": len(errors), "rmse": rmse, "mae": mae}
