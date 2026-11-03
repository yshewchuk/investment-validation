"""Prediction interval coverage and width, with unrandomized PIT evidence."""
from collections import Counter

from experiments.distribution_prediction_report import (
    ARMS,
    HORIZONS,
    _inputs,
    _mean,
    _number,
    _require,
)

LEVELS = (.5, .8, .9, .95)


def _validate(indexed):
    for row in indexed.values():
        available = row["status"] == "available"
        if available or "intervals" in row:
            _require(type(row["intervals"]) is list and len(row["intervals"]) == len(LEVELS))
            previous = None
            for interval, level in zip(row["intervals"], LEVELS):
                _require(type(interval) is dict and interval["level"] == level)
                lower, upper, width = (interval[key] for key in ("lower", "upper", "width"))
                _require(all(_number(value) for value in (lower, upper, width)) and lower <= upper)
                _require(width == upper - lower and type(interval["covered"]) is bool)
                _require(row["observation"] is None or interval["covered"] == (lower <= row["observation"] <= upper))
                _require(previous is None or (lower <= previous[0] and upper >= previous[1]))
                previous = lower, upper
        for field in ("pit_lower", "pit_upper"):
            if available or field in row:
                _require(_number(row[field]) and row[field] <= 1)
        if "pit_lower" in row and "pit_upper" in row:
            _require(row["pit_lower"] <= row["pit_upper"])


def prediction_diagnostic_report(diagnostics, events):
    """Aggregate supplied rows, keeping unavailable events and original brackets.

    No PIT random draw, midpoint, uniformity claim, calibration fit or source
    access. Native callers bind these rows to the exact frozen score evidence.
    O(rows log rows + events log events) time, O(rows + events) storage.
    """
    metadata, indexed = _inputs(diagnostics, events)
    try:
        _validate(indexed)
        horizons = {}
        for horizon in HORIZONS:
            arms = {}
            for arm in ARMS:
                rows = [indexed.get((event, horizon, arm)) for event in metadata]
                counts = Counter("missing_row" if row is None else row["status"] for row in rows)
                available = [row for row in rows if row is not None and row["status"] == "available"]
                intervals = []
                for index, level in enumerate(LEVELS):
                    covered = [int(row["intervals"][index]["covered"]) for row in available]
                    intervals.append({"level": level, "count": len(available), "covered_count": sum(covered),
                        "observed_coverage": _mean(covered),
                        "mean_width": _mean([row["intervals"][index]["width"] for row in available])})
                arms[arm] = {"population": len(metadata), "diagnostic_population": "available_only",
                    "counts": {status: counts[status] for status in ("available", "unavailable", "missing_target", "missing_row")},
                    "intervals": intervals, "pit_count": len(available)}
            horizons[str(horizon)] = {"arms": arms}
        return {"schema_version": "distribution_prediction_diagnostic_report.v1", "population": len(metadata),
            "horizons": horizons, "events": list(metadata.values()), "rows": list(indexed.values()),
            "pit_policy": "brackets_only_unrandomized", "calibration_identity": "uncalibrated",
            "final_read_authority": "none"}
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)
