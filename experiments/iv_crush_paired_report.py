"""Descriptive signed-IV comparison; no new selection, threshold or release policy."""
import math
from collections import Counter
from statistics import mean

from engine.v2.foundation import content_hash
from experiments.iv_crush_plan import target
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_evidence import normalize_forecasts
from experiments.magnitude_search_request import validate_horizons


def _metrics(rows, arm):
    values = [{"crush_pct_iv30": row["crush_pct_iv30"], "predicted_crush_pct_iv30": row[arm]} for row in rows]
    return {**magnitude_metrics(values, target=target()),
            "bias": mean(row[arm] - row["crush_pct_iv30"] for row in rows)}


def _comparison(rows):
    metrics = {arm: _metrics(rows, arm) for arm in ("challenger", "incumbent", "training_mean")}
    return {"metrics": metrics, "difference": {metric: metrics["challenger"][metric] - metrics["incumbent"][metric]
                                               for metric in ("rmse", "mae")}}


def paired_report(events, challenger, incumbent, targets, *, horizons=(5, 3, 1)):
    """Pair whole events by forecast availability before joining complete admitted targets."""
    leads = validate_horizons(horizons)
    metadata = {event["event_id"]: event for event in events}
    if len(metadata) != len(events) or not metadata or set(targets) != set(metadata):
        raise ValueError("IV report requires exact complete admitted event/target coverage")
    normalized = normalize_forecasts(challenger, {"evaluation_event_ids": list(metadata)}, leads,
                                    target=target(), strict_json=True)
    a = {(row["event_id"], row["horizon_sessions"]): row for row in normalized}
    b = {(row["event_id"], row["horizon_sessions"]): row for row in incumbent}
    expected = {(event, horizon) for event in metadata for horizon in leads}
    if len(b) != len(incumbent) or set(b) != expected:
        raise ValueError("IV comparison forecast coverage differs")
    for value in targets.values():
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("invalid signed IV target")
    for row in incumbent:
        if (type(row["horizon_sessions"]) is not int
                or row["abstention"] not in (None, "unavailable_input", "insufficient_training")):
            raise ValueError("invalid IV comparison coverage")
        for field in ("predicted_crush_pct_iv30", "training_mean"):
            value = row[field]
            if field == "predicted_crush_pct_iv30" and row["abstention"] is not None:
                if value is not None:
                    raise ValueError("abstaining IV comparison has a forecast")
            elif type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("invalid signed IV comparison value")
    matched = {event for event in metadata if all(b[event, h]["abstention"] is None for h in leads)}
    coverage, rows, broad = Counter(), [], []
    for event, horizon in sorted(expected):
        baseline, meta = b[event, horizon], metadata[event]
        for arm, status in (("challenger", "predicted"), ("training_mean", "predicted"),
                            ("incumbent", baseline["abstention"] or "predicted")):
            coverage[arm, meta["event_date"][:4], horizon, status, event in matched] += 1
        row = {"event_id": event, "horizon_sessions": horizon, "year": meta["event_date"][:4],
               "ticker": meta["ticker"], "crush_pct_iv30": targets[event],
               "challenger": a[event, horizon]["predicted_crush_pct_iv30"],
               "training_mean": baseline["training_mean"]}
        broad.append(row)
        if event in matched:
            rows.append({**row, "incumbent": baseline["predicted_crush_pct_iv30"]})
    result = {"schema_version": "iv_crush_paired_report.v1", "target": target(),
        "eligible_events": len(metadata), "forecast_matched_events": len(matched), "paired_events": len(matched),
        "event_total_weight": 1, "horizon_weight": 1 / len(leads), "paired_rows": rows,
        "coverage": [{"arm": arm, "year": year, "horizon_sessions": horizon,
            "forecast_status": status, "paired_forecasts": paired, "count": count}
            for (arm, year, horizon, status, paired), count in sorted(coverage.items())],
        "broad_coverage_metrics": {arm: _metrics(broad, arm) for arm in ("challenger", "training_mean")},
        "challenger_abstentions": 0, "label_unavailable_events": [],
        "claim": "frozen_dataset_development_only", "final_read_authority": "none",
        "comparison_status": "available" if rows else "unavailable_no_complete_matched_events"}
    if rows:
        result.update(**_comparison(rows),
            per_year={year: _comparison([row for row in rows if row["year"] == year])
                      for year in sorted({row["year"] for row in rows})},
            per_horizon={str(h): _comparison([row for row in rows if row["horizon_sessions"] == h]) for h in leads})
    return {**result, "result_id": content_hash(result)}
