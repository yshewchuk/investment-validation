"""Descriptive signed-IV comparison; no new selection, threshold or release policy."""
import math
from collections import Counter
from statistics import mean

from engine.v2.foundation import content_hash
from experiments.iv_crush_plan import target
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_evidence import normalize_forecasts
from experiments.magnitude_search_request import validate_horizons
from experiments.size_paired_report import _bootstrap


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


_ARMS_V2 = ("challenger", "incumbent", "training_mean", "implied_ex_earnings")
_PAIRS = (("challenger", "incumbent"), ("challenger", "training_mean"),
          ("challenger", "implied_ex_earnings"), ("incumbent", "training_mean"))
_TRUTH = "crush_pct_iv30"


def _implied_arm_values(implied, event_ids):
    if not isinstance(implied, dict) or set(implied) != event_ids:
        raise ValueError("IV implied ex-earnings coverage differs")
    values = {}
    for event, record in implied.items():
        if not isinstance(record, dict) or set(record) != {"iv30", "exern_iv30"}:
            raise ValueError("IV implied ex-earnings coverage differs")
        for value in record.values():
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                raise ValueError("invalid IV implied input")
        iv30, exern = record["iv30"], record["exern_iv30"]
        values[event] = (100 * (exern / iv30 - 1)
                         if iv30 is not None and exern is not None and iv30 > 0 and exern > 0
                         else None)
    return values


def paired_report_v2(events, challenger, incumbent, targets, implied, *,
                     horizons=(5, 3, 1), distribution=None):
    """Descriptive four-arm comparison over whole events; no winner or promotion rule."""
    base = paired_report(events, challenger, incumbent, targets, horizons=horizons)
    leads = validate_horizons(horizons)
    metadata = {event["event_id"]: event for event in events}
    event_ids = set(metadata)
    implied_values = _implied_arm_values(implied, event_ids)
    a = {(row["event_id"], row["horizon_sessions"]): row for row in challenger}
    b = {(row["event_id"], row["horizon_sessions"]): row for row in incumbent}
    matched = {row["event_id"] for row in base["paired_rows"]}
    all_rows = []
    for event, horizon in sorted((event, h) for event in event_ids for h in leads):
        meta = metadata[event]
        all_rows.append({"event_id": event, "horizon_sessions": horizon,
            "year": meta["event_date"][:4], "month": meta["event_date"][:7],
            "ticker": meta["ticker"], "crush_pct_iv30": targets[event],
            "challenger": a[event, horizon]["predicted_crush_pct_iv30"],
            "training_mean": b[event, horizon]["training_mean"],
            "incumbent": (b[event, horizon]["predicted_crush_pct_iv30"]
                          if event in matched else None),
            "implied_ex_earnings": implied_values[event]})
    arm_metrics = {}
    for arm in _ARMS_V2:
        subset = [row for row in all_rows if row[arm] is not None]
        arm_metrics[arm] = _metrics(subset, arm) if subset else None
    common = [row for row in all_rows if all(row[arm] is not None for arm in _ARMS_V2)]
    common_events = len({row["event_id"] for row in common})
    common_matched = None
    if common:
        common_metrics = {arm: _metrics(common, arm) for arm in _ARMS_V2}
        common_matched = {"events": common_events, "metrics": common_metrics,
            "difference": {f"{left}_minus_{right}": {
                "rmse": common_metrics[left]["rmse"] - common_metrics[right]["rmse"],
                "mae": common_metrics[left]["mae"] - common_metrics[right]["mae"]}
                for left, right in _PAIRS}}
    bootstrap = None
    if common:
        bootstrap = {cluster_name: {
            f"{left}_minus_{right}": _bootstrap(common, field, truth=_TRUTH, arms=(left, right))
            for left, right in _PAIRS}
            for cluster_name, field in (("ticker", "ticker"), ("event_month", "month"))}
    common_ids = {row["event_id"] for row in common}
    counts = Counter()
    for event, horizon in sorted((event, h) for event in event_ids for h in leads):
        status = ("predicted" if implied_values[event] is not None
                  else "abstained_missing_or_nonpositive_input")
        counts[metadata[event]["event_date"][:4], horizon, status, event in common_ids] += 1
    coverage = list(base["coverage"]) + [
        {"arm": "implied_ex_earnings", "year": year, "horizon_sessions": horizon,
         "forecast_status": status, "paired_forecasts": paired, "count": count}
        for (year, horizon, status, paired), count in counts.items()]
    coverage.sort(key=lambda row: (row["arm"], row["year"], row["horizon_sessions"],
                                   row["forecast_status"], row["paired_forecasts"]))
    if distribution is None:
        challenger_distribution = {"status": "not_supplied"}
    elif not isinstance(distribution, dict) or distribution.get("schema_version") != "iv_distribution.v1":
        raise ValueError("invalid challenger distribution")
    else:
        challenger_distribution = distribution
    result = {"schema_version": "iv_crush_paired_report.v2", "target": target(),
        "eligible_events": base["eligible_events"],
        "forecast_matched_events": base["forecast_matched_events"],
        "common_matched_events": common_events, "arm_metrics": arm_metrics,
        "common_matched": common_matched, "bootstrap": bootstrap, "coverage": coverage,
        "challenger_distribution": challenger_distribution, "label_unavailable_events": [],
        "claim": "frozen_dataset_development_only", "final_read_authority": "none",
        "comparison_status": ("available" if common
                              else "unavailable_no_common_matched_events")}
    return {**result, "result_id": content_hash(result)}
