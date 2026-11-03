"""Source-free, event-weighted forecast summaries; callers own native admission."""
from collections import Counter
from copy import deepcopy
from datetime import date
from math import fsum, isfinite

from engine.v2.ops.errors import fail

ARMS = ("histogram", "same_mean", "matched_point", "training_empirical")
HORIZONS = (1, 3, 5)


def _require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid prediction report inputs or arithmetic")


def _number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _mean(values):
    result = fsum(value / len(values) for value in values) if values else None
    _require(result is None or isfinite(result))
    return result


def _inputs(rows, events):
    """Index one pass of supplied rows, retaining detached, canonical evidence."""
    try:
        _require(type(events) is list)
        metadata, indexed, observations, years = {}, {}, {}, {}
        for event in events:
            _require(type(event) is dict)
            _require(all(type(event[k]) is str and event[k] for k in ("event_id", "ticker", "event_date")))
            _require(date.fromisoformat(event["event_date"]).isoformat() == event["event_date"])
            _require(event["event_id"] not in metadata)
            metadata[event["event_id"]] = deepcopy(event)
        for row in rows:
            _require(type(row) is dict)
            event, horizon, arm = row["event_id"], row["horizon"], row["arm"]
            _require(type(event) is str and event in metadata and type(horizon) is int and horizon in HORIZONS)
            _require(arm in ARMS and (event, horizon, arm) not in indexed)
            _require(type(row["outer_year"]) is str and bool(row["outer_year"]))
            _require(row["status"] in ("available", "unavailable", "missing_target"))
            observation = row["observation"]
            _require(observation is None if row["status"] == "missing_target" else _number(observation))
            _require(years.setdefault(event, row["outer_year"]) == row["outer_year"])
            _require(observations.setdefault((event, horizon), observation) == observation)
            indexed[event, horizon, arm] = deepcopy(row)
        return dict(sorted(metadata.items())), dict(sorted(indexed.items()))
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)


def _validate_scores(indexed):
    for row in indexed.values():
        if row["status"] != "available":
            continue
        _require(row["crps"] is None or _number(row["crps"]))
        _require(_number(row["rps"]) and (row["log_loss"] == "infinite" or _number(row["log_loss"])))
        _require(type(row["tails"]) is list and len(row["tails"]) == 3)
        for tail, threshold in zip(row["tails"], (5, 10, 20)):
            _require(type(tail) is dict and tail["threshold"] == threshold)
            _require(type(tail["observed_ge"]) is bool and tail["observed_ge"] == (row["observation"] >= threshold))
            _require(_number(tail["probability_ge"]) and tail["probability_ge"] <= 1)


def _arm_summary(rows):
    counts = Counter("missing_row" if row is None else row["status"] for row in rows)
    available = [row for row in rows if row is not None and row["status"] == "available"]
    metrics = {}
    for field in ("crps", "rps", "log_loss"):
        values = [row[field] for row in available if type(row[field]) in (int, float)]
        metrics[field] = {"mean": _mean(values), "count": len(values)}
    infinite = sum(row["log_loss"] == "infinite" for row in available)
    metrics["log_loss"].update(finite_mean=metrics["log_loss"]["mean"], finite_count=metrics["log_loss"]["count"],
        infinite_count=infinite, count=len(available), mean="infinite" if infinite else metrics["log_loss"]["mean"])
    tails = []
    for index, threshold in enumerate((5, 10, 20)):
        observed = [int(row["tails"][index]["observed_ge"]) for row in available]
        tails.append({"threshold": threshold, "count": len(available), "observed_count": sum(observed),
            "observed_frequency": _mean(observed),
            "mean_probability_ge": _mean([row["tails"][index]["probability_ge"] for row in available])})
    return {"population": len(rows), "summary_population": "scoreable_only", "scores": metrics, "tails": tails,
        "counts": {status: counts[status] for status in ("available", "unavailable", "missing_target", "missing_row")},
        "unresolved_crps": sum(row["crps"] is None for row in available)}


def prediction_report(scores, events):
    """All four arms and all horizons, even when a whole arm/horizon is missing.

    Paired improvement is control minus histogram CRPS; positive favors the
    histogram. Missing scores never become zeros. O(rows log rows + events log
    events) time and O(rows + events) retained evidence; no I/O, fits or target access.
    """
    metadata, indexed = _inputs(scores, events)
    try:
        _validate_scores(indexed)
        horizons = {}
        for horizon in HORIZONS:
            arms = {arm: _arm_summary([indexed.get((event, horizon, arm)) for event in metadata]) for arm in ARMS}
            contrasts = {}
            for arm in ARMS[1:]:
                pairs, excluded = [], []
                for event in metadata:
                    pair = [indexed.get((event, horizon, name)) for name in ("histogram", arm)]
                    if all(row is not None and row["status"] == "available" and row["crps"] is not None for row in pair):
                        pairs.append({"event_id": event, "delta": pair[1]["crps"] - pair[0]["crps"]})
                    else:
                        excluded.append(event)
                mean = _mean([row["delta"] for row in pairs])
                contrasts[arm] = {"role": "primary" if horizon == 1 and arm == "same_mean" else "descriptive",
                    "status": "complete" if pairs and not excluded else "incomplete", "population": len(metadata),
                    "scoreable_events": len(pairs), "excluded_event_ids": excluded, "paired_deltas": pairs,
                    "full_population_mean_crps_improvement": mean if not excluded else None,
                    "scoreable_mean_crps_improvement": mean}
            horizons[str(horizon)] = {"arms": arms, "contrasts": contrasts}
        return {"schema_version": "distribution_prediction_report.v1", "population": len(metadata),
            "primary": "h1_same_mean_minus_histogram_crps", "horizons": horizons,
            "events": list(metadata.values()), "rows": list(indexed.values()),
            "calibration_identity": "uncalibrated", "final_read_authority": "none"}
    except (KeyError, TypeError, ValueError, OverflowError):
        _require(False)
