"""Conditional cluster uncertainty for frozen forecast scores and tail gaps."""
from datetime import date

import numpy as np

from engine.v2.research.progress import Progress, stage
from experiments.distribution_prediction_report import ARMS, HORIZONS, _require, prediction_report


def _bootstrap(measures, clusters, *, seed, resamples, confidence):
    """One full-roster cluster draw per replicate, shared by every measurement."""
    labels = sorted(set(clusters.values()))
    positions = {label: index for index, label in enumerate(labels)}
    values = np.zeros((len(labels), len(measures), 2))
    for column, rows in enumerate(measures):
        for row in rows:
            values[positions[clusters[row["event_id"]]], column] += (row["delta"], 1)
    samples = np.full((resamples, len(measures)), np.nan)
    rng, progress = np.random.default_rng(seed), Progress("prediction conditional uncertainty", resamples)
    with stage("prediction conditional uncertainty"):
        for iteration in range(resamples):
            if labels:
                weights = np.bincount(rng.integers(0, len(labels), size=len(labels)), minlength=len(labels))
                sums = np.einsum("i,ijk->jk", weights, values)
                _require(np.isfinite(sums).all())
                np.divide(sums[:, 0], sums[:, 1], out=samples[iteration], where=sums[:, 1] > 0)
            progress.tick()
    results = []
    for column in range(len(measures)):
        usable = samples[:, column][~np.isnan(samples[:, column])]
        count = int(np.count_nonzero(values[:, column, 1]))
        result = {"clusters": len(labels), "scoreable_clusters": count, "scoreable_events": len(measures[column]),
            "used_resamples": len(usable), "empty_resamples": resamples - len(usable),
            "status": "unavailable_fewer_than_two_scoreable_clusters"}
        if count >= 2 and len(usable):
            interval = np.quantile(usable, [(1 - confidence) / 2, (1 + confidence) / 2], method="linear")
            _require(np.isfinite(interval).all())
            result.update(status="available", interval=interval.tolist())
        elif count >= 2:
            result["status"] = "unavailable_no_scoreable_resamples"
        results.append(result)
    return results


def prediction_uncertainty(scores, events, *, seed, resamples, block_days, block_origin, confidence):
    """Augment one validated score report using caller-supplied sealed policy.

    CRPS improvements and probability-minus-observed tail gaps share draws
    across arms/horizons. Percentiles condition on nonempty scoreable draws,
    not refitting, independent-event sampling, significance or promotion.
    Fixed 45 measures use O(events + resamples) workspace and
    O(resamples * occupied_clusters) work; native callers bound policy/bytes.
    """
    _require(type(seed) is int and seed >= 0 and type(resamples) is int and resamples > 0)
    _require(type(block_days) is int and block_days > 0)
    _require(type(confidence) in (int, float) and 0 < confidence < 1)
    try:
        _require(type(block_origin) is str and date.fromisoformat(block_origin).isoformat() == block_origin)
        origin = date.fromisoformat(block_origin)
        report = prediction_report(scores, events)
        measures, destinations = [], []
        tail_rows = {(horizon, arm, index): [] for horizon in HORIZONS for arm in ARMS for index in range(3)}
        for row in report["rows"]:
            if row["status"] == "available":
                for index, tail in enumerate(row["tails"]):
                    tail_rows[row["horizon"], row["arm"], index].append({"event_id": row["event_id"],
                        "delta": tail["probability_ge"] - int(tail["observed_ge"])})
        for horizon in HORIZONS:
            summary = report["horizons"][str(horizon)]
            for contrast in summary["contrasts"].values():
                measures.append(contrast["paired_deltas"])
                destinations.append(contrast)
            for arm in ARMS:
                for index, tail in enumerate(summary["arms"][arm]["tails"]):
                    measures.append(tail_rows[horizon, arm, index])
                    destinations.append(tail)
                    tail["gap_direction"] = "probability_minus_observed"
                    tail["mean_calibration_gap"] = (tail["mean_probability_ge"] - tail["observed_frequency"]
                        if tail["count"] else None)
        clusters = {"ticker": {row["event_id"]: row["ticker"] for row in report["events"]},
            "calendar": {row["event_id"]: (date.fromisoformat(row["event_date"]) - origin).days // block_days
                for row in report["events"]}}
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for field, membership in clusters.items():
                intervals = _bootstrap(measures, membership, seed=seed, resamples=resamples, confidence=confidence)
                for destination, interval in zip(destinations, intervals):
                    destination.setdefault("conditional_intervals", {})[field] = interval
        report["uncertainty"] = {"seed": seed, "resamples": resamples, "block_days": block_days,
            "block_origin": block_origin, "confidence": confidence, "calendar_clock": "event_date",
            "calendar_blocks": "occupied_fixed_nonoverlapping", "resampling": "joint_full_population_clusters",
            "quantile_method": "linear", "interpretation": "conditional_on_frozen_fits_and_nonempty_scoreable_resamples"}
        return report
    except (KeyError, TypeError, ValueError, OverflowError, FloatingPointError):
        _require(False)
