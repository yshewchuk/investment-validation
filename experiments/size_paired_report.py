"""Matched development reporting over the selected event/horizon rows (default: all three horizons).

Pairing uses those selected rows only and has no selection or holdout authority.
"""
import math
from collections import Counter, defaultdict

import numpy as np

from engine.v2.foundation import content_hash
from experiments.magnitude_report import magnitude_metrics


def _metrics(rows, field):
    values = [{"absolute_move_pct": r["absolute_move_pct"], "predicted_absolute_move_pct": r[field]} for r in rows]
    errors = [r[field] - r["absolute_move_pct"] for r in rows]
    large = [r["absolute_move_pct"] - r[field] for r in rows if r["absolute_move_pct"] >= 10]
    return {**magnitude_metrics(values), "bias": float(np.mean(errors)),
            "large_move_threshold_pct": 10, "large_move_count": len(large),
            "large_move_underprediction": float(np.mean(large)) if large else None}


def _difference(rows):
    a, b = _metrics(rows, "challenger"), _metrics(rows, "incumbent")
    return {key: a[key] - b[key] for key in ("rmse", "mae")}


def _bootstrap(rows, cluster):
    errors = np.array([[r[arm] - r["absolute_move_pct"] for arm in ("challenger", "incumbent")] for r in rows])
    scale = float(np.max(np.abs(errors))) or 1.0
    errors /= scale
    groups = defaultdict(lambda: np.zeros(5))
    for row, error in zip(rows, errors):
        groups[row[cluster]] += np.r_[1, np.abs(error), error ** 2]
    keys, rng = sorted(groups), np.random.default_rng(0)
    aggregates = np.array([groups[key] for key in keys])
    samples = {metric: [] for metric in ("rmse", "mae")}
    for _ in range(2000):
        total = aggregates[rng.integers(0, len(keys), size=len(keys))].sum(axis=0)
        samples["mae"].append(scale * (total[1] - total[2]) / total[0])
        samples["rmse"].append(scale * (np.sqrt(total[3] / total[0]) - np.sqrt(total[4] / total[0])))
    return {"cluster": cluster, "blocks": len(keys), "events": len({r["event_id"] for r in rows}),
        "resamples": 2000, "seed": 0,
        "percentile_95": {m: np.percentile(v, [2.5, 97.5]).tolist() for m, v in samples.items()}}


_IMPLIED_ARM = "implied_move"
_SCALED_ARM = "implied_move_scaled"
_ARMS = ("challenger", "incumbent", "history_mean")
_REPORT_ARMS = (*_ARMS, _IMPLIED_ARM)
_SCALED_REPORT_ARMS = (*_REPORT_ARMS, _SCALED_ARM)


def _implied_bootstrap(rows, cluster, extra_arm=_IMPLIED_ARM):
    """Additive bootstrap of every report arm and its paired comparisons."""
    focus = (_IMPLIED_ARM,) if extra_arm == _IMPLIED_ARM else (_IMPLIED_ARM, extra_arm)
    arms = (*_REPORT_ARMS, *[arm for arm in focus if arm not in _REPORT_ARMS])
    pairs = (("challenger", "incumbent"),) + tuple(
        (arm, other) for arm in focus for other in ("challenger", "incumbent", "history_mean"))
    if extra_arm != _IMPLIED_ARM:
        pairs += ((extra_arm, _IMPLIED_ARM),)
    errors = np.array([[r[arm] - r["absolute_move_pct"] for arm in arms] for r in rows])
    scale = float(np.max(np.abs(errors))) or 1.0
    errors /= scale
    groups = defaultdict(lambda: np.zeros(1 + 2 * len(arms)))
    for row, error in zip(rows, errors):
        groups[row[cluster]] += np.r_[1, np.abs(error), error ** 2]
    keys, rng = sorted(groups), np.random.default_rng(0)
    aggregates = np.array([groups[key] for key in keys])
    arm_samples = {arm: {"rmse": [], "mae": []} for arm in arms}
    difference_samples = {left + "_minus_" + right: {"rmse": [], "mae": []} for left, right in pairs}
    for _ in range(2000):
        total = aggregates[rng.integers(0, len(keys), size=len(keys))].sum(axis=0)
        mae = {arm: scale * total[1 + index] / total[0]
               for index, arm in enumerate(arms)}
        rmse = {arm: scale * np.sqrt(total[1 + len(arms) + index] / total[0])
                for index, arm in enumerate(arms)}
        for arm in arms:
            arm_samples[arm]["mae"].append(mae[arm])
            arm_samples[arm]["rmse"].append(rmse[arm])
        for left, right in pairs:
            name = left + "_minus_" + right
            difference_samples[name]["mae"].append(mae[left] - mae[right])
            difference_samples[name]["rmse"].append(rmse[left] - rmse[right])
    def interval(samples):
        return {metric: np.percentile(values, [2.5, 97.5]).tolist()
                for metric, values in samples.items()}
    return {"cluster": cluster, "blocks": len(keys), "events": len({r["event_id"] for r in rows}),
        "resamples": 2000, "seed": 0, "arms": list(arms),
        "percentile_95": {arm: interval(samples) for arm, samples in arm_samples.items()},
        "difference_percentile_95": {name: interval(samples)
            for name, samples in difference_samples.items()}}


def _implied_values(implied_moves, expected):
    if implied_moves is None:
        return {}
    if not isinstance(implied_moves, dict) or not set(implied_moves) <= expected:
        raise ValueError("implied move coverage differs from paired population")
    values = {}
    for key, value in implied_moves.items():
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid implied move")
        values[key] = value
    return values


def _implied_scales(scales):
    if scales is None:
        return {}
    if not isinstance(scales, dict):
        raise ValueError("invalid implied scale calibration")
    normalized = {}
    for year, record in scales.items():
        if (not isinstance(year, str) or not isinstance(record, dict)
                or set(record) != {"scale", "fit_n"}):
            raise ValueError("invalid implied scale calibration")
        scale, fit_n = record["scale"], record["fit_n"]
        if type(fit_n) is not int or fit_n < 0:
            raise ValueError("invalid implied scale fit count")
        if scale is not None and (type(scale) not in (int, float)
                                  or not math.isfinite(scale) or scale < 0):
            raise ValueError("invalid implied scale")
        normalized[year] = {"scale": scale, "fit_n": fit_n}
    return normalized


def paired_report(events, challenger, incumbent, targets, *, label_reasons=None,
                  horizons=(1, 3, 5), implied_moves=None, implied_move=None,
                  implied_scales=None):
    """Match by forecast availability alone, then join targets; retain all exclusions.

    ``implied_moves`` optionally maps ``(event_id, horizon_sessions)`` to the qualified
    panel's percent-point ``implied_move`` feature; absent or null values abstain only
    for that report-only arm and never shrink the existing full-pair metrics.
    ``implied_move`` is an equivalent singular alias. ``implied_scales`` optionally
    maps each outer year to its training-only ``{"scale", "fit_n"}`` calibration;
    omitted calibration leaves the scaled arm explicitly empty.
    """
    if implied_moves is not None and implied_move is not None:
        raise ValueError("duplicate implied move input")
    if (not isinstance(horizons, (tuple, list)) or not horizons
            or any(type(h) is not int for h in horizons)
            or len(set(horizons)) != len(horizons)
            or not set(horizons) <= {1, 3, 5}
            or (tuple(horizons) != tuple(sorted(horizons))
                and tuple(horizons) != tuple(h for h in (5, 3, 1) if h in set(horizons)))):
        raise ValueError("horizons must be a nonempty unique subset of integer leads 1, 3, 5 in ascending or legacy (5,3,1) order")
    leads = tuple(sorted(horizons))
    def index(rows):
        result = {(r["event_id"], r["horizon_sessions"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("duplicate forecast keys")
        return result
    a, b = index(challenger), index(incumbent)
    metadata = {e["event_id"]: e for e in events}
    if len(metadata) != len(events):
        raise ValueError("duplicate event metadata")
    label_reasons = label_reasons or {event: "unavailable" for event in set(metadata) - set(targets)}
    if set(label_reasons) != set(metadata) - set(targets):
        raise ValueError("label exclusion reasons differ from coverage")
    expected = {(event, h) for event in metadata for h in leads}
    if set(a) != expected or set(b) != expected or not set(targets) <= set(metadata):
        raise ValueError("paired coverage differs from frozen event population")
    for value in targets.values():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid magnitude target")
    for row in challenger:
        value = row["predicted_absolute_move_pct"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid magnitude challenger")
    for row in incumbent:
        for name in ("history_mean", "predicted_absolute_move_pct"):
            value = row[name]
            if value is None and name == "predicted_absolute_move_pct" and row["abstention"] is not None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid comparison value")
    implied = _implied_values(implied_moves if implied_moves is not None else implied_move, expected)
    scales = _implied_scales(implied_scales)
    matched = {e for e in metadata if all(b[e, h]["predicted_absolute_move_pct"] is not None for h in leads)}
    coverage, rows = Counter(), []
    for event, h in sorted(expected):
        meta, baseline = metadata[event], b[event, h]
        reason = baseline["abstention"] or "predicted"
        for arm, status in (("incumbent", reason), ("challenger", "predicted"), ("history_mean", "predicted")):
            coverage[(arm, meta["event_date"][:4], h, baseline["quote_status"], status,
                      "available" if event in targets else label_reasons[event],
                      "complete" if event in matched else "other_horizon_abstention")] += 1
        if event in matched and event in targets:
            day = meta["event_date"]
            raw = implied.get((event, h))
            scale = scales.get(day[:4], {}).get("scale")
            rows.append({"event_id": event, "horizon_sessions": h, "year": day[:4], "ticker": meta["ticker"],
                "quarter": day[:4] + "Q" + str((int(day[5:7]) - 1) // 3 + 1),
                "absolute_move_pct": targets[event], "challenger": a[event, h]["predicted_absolute_move_pct"],
                "incumbent": baseline["predicted_absolute_move_pct"], "history_mean": baseline["history_mean"],
                _IMPLIED_ARM: raw,
                _SCALED_ARM: raw * scale if raw is not None and scale is not None else None})
    available = [row for row in rows if row[_IMPLIED_ARM] is not None]
    scaled_available = [row for row in rows if row[_SCALED_ARM] is not None]
    implied_coverage = {"available_rows": len(available), "total_paired_rows": len(rows)}
    scaled_coverage = {"available_rows": len(scaled_available), "total_paired_rows": len(rows)}
    result = {"schema_version": "size_paired_report.v2", "eligible_events": len(metadata),
        "forecast_matched_events": len(matched), "label_unavailable_events": sorted(set(metadata) - set(targets)),
        "paired_events": len({r["event_id"] for r in rows}), "event_total_weight": 1,
        "horizon_weight": 1 / len(leads),
        "coverage": [{"arm": arm, "year": y, "horizon_sessions": h, "quote_status": q, "forecast_status": s,
                      "label_status": label, "paired_forecasts": pairing, "count": n}
            for (arm, y, h, q, s, label, pairing), n in sorted(coverage.items())], "paired_rows": rows,
        "implied_move_coverage": implied_coverage,
        "implied_move_scaled_coverage": scaled_coverage,
        "implied_move_scaled_calibration": scales,
        "final_read_authority": "none", "claim": "frozen_dataset_development_only",
        "interval_caveat": "does_not_account_for_all_prior_adaptive_design_choices"}
    broad = [{"absolute_move_pct": targets[e], "challenger": a[e, h]["predicted_absolute_move_pct"],
              "history_mean": b[e, h]["history_mean"]} for e, h in sorted(expected) if e in targets]
    result["broad_coverage_metrics"] = ({arm: _metrics(broad, arm) for arm in ("challenger", "history_mean")}
                                         if broad else {})
    result["challenger_abstentions"] = 0  # A failed selected arm refuses the entire report.
    implied_bootstrap = _implied_bootstrap(available, "quarter") if available else None
    scaled_bootstrap = (_implied_bootstrap(scaled_available, "quarter", _SCALED_ARM)
                        if scaled_available else None)
    if available:
        result["like_for_like"] = {
            **implied_coverage,
            "metrics": {arm: _metrics(available, arm) for arm in _REPORT_ARMS},
            "difference": _difference(available),
            "quarter_bootstrap": implied_bootstrap}
    else:
        result["like_for_like"] = {
            **implied_coverage, "status": "unavailable_no_implied_move_rows",
            "metrics": None, "difference": None, "quarter_bootstrap": None}
    if scaled_available:
        result["like_for_like_implied_move_scaled"] = {
            **scaled_coverage,
            "metrics": {arm: _metrics(scaled_available, arm) for arm in _SCALED_REPORT_ARMS},
            "difference": _difference(scaled_available),
            "quarter_bootstrap": scaled_bootstrap}
    else:
        result["like_for_like_implied_move_scaled"] = {
            **scaled_coverage, "status": "unavailable_no_scaled_implied_move_rows",
            "metrics": None, "difference": None, "quarter_bootstrap": None}
    if rows:
        metrics = {arm: _metrics(rows, arm) for arm in _ARMS}
        metrics[_IMPLIED_ARM] = _metrics(available, _IMPLIED_ARM) if available else None
        metrics[_SCALED_ARM] = _metrics(scaled_available, _SCALED_ARM) if scaled_available else None
        quarter_bootstrap = _bootstrap(rows, "quarter")
        quarter_bootstrap["implied_move"] = implied_bootstrap
        quarter_bootstrap["implied_move_scaled"] = scaled_bootstrap
        result.update(metrics=metrics,
            difference=_difference(rows), quarter_bootstrap=quarter_bootstrap,
            issuer_sensitivity=_bootstrap(rows, "ticker"),
            per_year={y: _breakdown([r for r in rows if r["year"] == y]) for y in sorted({r["year"] for r in rows})},
            per_horizon={str(h): _breakdown([r for r in rows if r["horizon_sessions"] == h]) for h in leads})
    else:
        result["comparison_status"] = "unavailable_no_complete_matched_events"
    return {**result, "result_id": content_hash(result)}


def _breakdown(rows):
    available = [row for row in rows if row[_IMPLIED_ARM] is not None]
    scaled = [row for row in rows if row[_SCALED_ARM] is not None]
    arms = {arm: _metrics(rows, arm) for arm in _ARMS}
    arms[_IMPLIED_ARM] = _metrics(available, _IMPLIED_ARM) if available else None
    arms[_SCALED_ARM] = _metrics(scaled, _SCALED_ARM) if scaled else None
    return {"difference": _difference(rows), "arms": arms,
            "implied_move_coverage": {"available_rows": len(available), "total_paired_rows": len(rows)},
            "implied_move_scaled_coverage": {"available_rows": len(scaled), "total_paired_rows": len(rows)}}
