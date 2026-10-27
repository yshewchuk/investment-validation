"""Matched three-horizon development reporting; no selection or holdout authority."""
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


def paired_report(events, challenger, incumbent, targets, *, label_reasons=None):
    """Match by forecast availability alone, then join targets; retain all exclusions."""
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
    expected = {(event, h) for event in metadata for h in (1, 3, 5)}
    if set(a) != expected or set(b) != expected or not set(targets) <= set(metadata):
        raise ValueError("paired coverage differs from frozen event population")
    for value in list(targets.values()) + [r["predicted_absolute_move_pct"] for r in challenger]:
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid magnitude target/challenger")
    for row in incumbent:
        for name in ("history_mean", "predicted_absolute_move_pct"):
            value = row[name]
            if value is None and name == "predicted_absolute_move_pct" and row["abstention"] is not None:
                continue
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid comparison value")
    matched = {e for e in metadata if all(b[e, h]["predicted_absolute_move_pct"] is not None for h in (1, 3, 5))}
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
            rows.append({"event_id": event, "horizon_sessions": h, "year": day[:4], "ticker": meta["ticker"],
                "quarter": day[:4] + "Q" + str((int(day[5:7]) - 1) // 3 + 1),
                "absolute_move_pct": targets[event], "challenger": a[event, h]["predicted_absolute_move_pct"],
                "incumbent": baseline["predicted_absolute_move_pct"], "history_mean": baseline["history_mean"]})
    result = {"schema_version": "size_paired_report.v1", "eligible_events": len(metadata),
        "forecast_matched_events": len(matched), "label_unavailable_events": sorted(set(metadata) - set(targets)),
        "paired_events": len({r["event_id"] for r in rows}), "event_total_weight": 1, "horizon_weight": 1 / 3,
        "coverage": [{"arm": arm, "year": y, "horizon_sessions": h, "quote_status": q, "forecast_status": s,
                      "label_status": label, "paired_forecasts": pairing, "count": n}
            for (arm, y, h, q, s, label, pairing), n in sorted(coverage.items())], "paired_rows": rows,
        "final_read_authority": "none", "claim": "frozen_dataset_development_only",
        "interval_caveat": "does_not_account_for_all_prior_adaptive_design_choices"}
    broad = [{"absolute_move_pct": targets[e], "challenger": a[e, h]["predicted_absolute_move_pct"],
              "history_mean": b[e, h]["history_mean"]} for e, h in sorted(expected) if e in targets]
    result["broad_coverage_metrics"] = ({arm: _metrics(broad, arm) for arm in ("challenger", "history_mean")}
                                         if broad else {})
    result["challenger_abstentions"] = 0  # A failed selected arm refuses the entire report.
    if rows:
        result.update(metrics={arm: _metrics(rows, arm) for arm in ("challenger", "incumbent", "history_mean")},
            difference=_difference(rows), quarter_bootstrap=_bootstrap(rows, "quarter"),
            issuer_sensitivity=_bootstrap(rows, "ticker"),
            per_year={y: _breakdown([r for r in rows if r["year"] == y]) for y in sorted({r["year"] for r in rows})},
            per_horizon={str(h): _breakdown([r for r in rows if r["horizon_sessions"] == h]) for h in (1, 3, 5)})
    else:
        result["comparison_status"] = "unavailable_no_complete_matched_events"
    return {**result, "result_id": content_hash(result)}


def _breakdown(rows):
    return {"difference": _difference(rows), "arms": {arm: _metrics(rows, arm)
            for arm in ("challenger", "incumbent", "history_mean")}}
