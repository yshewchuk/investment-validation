"""Collect verified selected-outer and comparator journals with report-only scalar calibration.

This module fits one report scalar and performs no model fits.
"""
import math
import sys
import time

from engine.v2.foundation import content_hash
from experiments.magnitude_search_outer import collect_outer_results
from experiments.magnitude_search_plan import trial_key
from experiments.magnitude_search_records import read_record
from experiments.size_paired_report import paired_report, validated_horizons
from experiments.v2_magnitude_search_run import _admit_driver


def _valid_scalar(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def fit_implied_scale(training_event_ids, labels, features, horizons):
    """Fit one no-intercept k from admitted training event-horizon feature/label pairs."""
    event_ids, leads, pairs = set(training_event_ids), set(validated_horizons(horizons)), []
    for row in features:
        event_id = row["event_id"]
        if event_id not in event_ids or row["horizon_sessions"] not in leads:
            continue
        x, y = row["implied_move"], labels.get(event_id)
        if x is None:
            continue
        if not _valid_scalar(x):
            raise ValueError("invalid implied move")
        if _valid_scalar(y):
            pairs.append((x, y))
    denominator = sum(x * x for x, _ in pairs)
    if not pairs or denominator == 0:
        return {"scale": None, "fit_n": len(pairs)}
    scale = sum(x * y for x, y in pairs) / denominator
    return {"scale": scale if math.isfinite(scale) and scale >= 0 else None,
            "fit_n": len(pairs)}


def _implied_evidence(fold, inputs, horizons, *, calibrate):
    """Extract report-only calibration and evaluation moves from verified inputs."""
    labels = {row["event_id"]: row["absolute_move_pct"] for row in inputs["training"]}
    scale = (fit_implied_scale(fold["training_candidate_event_ids"], labels, inputs["features"], horizons)
             if calibrate else None)
    keys = {(event_id, horizon) for event_id in fold["evaluation_event_ids"] for horizon in horizons}
    moves = {(row["event_id"], row["horizon_sessions"]): row["implied_move"]
             for row in inputs["features"] if (row["event_id"], row["horizon_sessions"]) in keys}
    return scale, moves


def _candidate_evidence(conn, store, run_id, plan):
    """Read-only per-run candidate listing bound to the sealed plan's exact trial keys."""
    rows, evidence_by_key = [], {}
    for group in plan["folds"]:
        outer_year = group["outer"]["year"]
        for fold in group["inner"]:
            inner_year = fold["year"]
            for candidate in plan["candidates"]:
                candidate_id = candidate["candidate_id"]
                key = trial_key(plan, outer_year, inner_year, candidate_id)
                if key not in evidence_by_key:
                    raw_execution = read_record(conn, store, run_id, "execution:" + key)
                    terminal = read_record(conn, store, run_id, "terminal:" + key)
                    execution = raw_execution if isinstance(raw_execution, dict) else {}
                    terminal = terminal if isinstance(terminal, dict) else {}
                    status = execution.get("status", terminal.get("status"))
                    fit_attempt_count = execution.get("fit_attempt_count")
                    completed_fit_count = execution.get("completed_fit_count")
                    diagnostics = execution.get("diagnostics", []) if status == "succeeded" else []
                    terminal_status = terminal.get("status")
                    score_seconds = terminal.get("score_seconds")
                    rss_delta_kib = terminal.get("rss_delta_kib")
                    inner_metrics = terminal.get("inner_metrics")
                    fit_seconds = []
                    for horizon in plan["horizons"]:
                        slot = "fit-completed:" + key + ":" + str(horizon)
                        completed = read_record(conn, store, run_id, slot)
                        if isinstance(completed, dict) and completed.get("fit_seconds") is not None:
                            fit_seconds.append({
                                "horizon_sessions": horizon,
                                "seconds": completed["fit_seconds"],
                            })
                    evidence_by_key[key] = (
                        terminal_status, score_seconds, rss_delta_kib, inner_metrics,
                        status, fit_attempt_count, completed_fit_count, diagnostics, fit_seconds)
                    del raw_execution, execution, terminal
                (terminal_status, score_seconds, rss_delta_kib, inner_metrics,
                 status, fit_attempt_count, completed_fit_count, diagnostics, fit_seconds) = evidence_by_key[key]
                rows.append({
                    "trial_id": key, "run_id": run_id, "outer_year": outer_year,
                    "inner_year": inner_year, "candidate_id": candidate_id,
                    "execution_slot": "execution:" + key, "status": status,
                    "terminal_status": terminal_status,
                    "fit_attempt_count": fit_attempt_count,
                    "completed_fit_count": completed_fit_count,
                    "fit_seconds": fit_seconds,
                    "score_seconds": score_seconds,
                    "rss_delta_kib": rss_delta_kib,
                    "inner_metrics": inner_metrics,
                    "diagnostics": diagnostics})
    return rows


def collect(conn, store, repository, registration, request, plan_ref):
    print("[stage] collect start", file=sys.stderr)
    started = time.monotonic()
    plan = _admit_driver(conn, store, registration, request, plan_ref)
    horizons = plan["horizons"]
    execution = registration.document.get("magnitude_execution")
    calibrate = (isinstance(execution, dict)
                 and isinstance(execution.get("evaluation_labels_before"), str))
    scales, implied_moves = {}, {}
    def implied_inputs(fold, inputs):
        scale, moves = _implied_evidence(fold, inputs, horizons, calibrate=calibrate)
        if calibrate:
            scales[str(fold["year"])] = scale
        implied_moves.update(moves)
    result = collect_outer_results(conn, store, repository, registration, request, plan_ref,
                                   input_callback=implied_inputs)
    forecasts, comparisons, events, comparison_fits = [], [], [], []
    targets = {r["event_id"]: r["absolute_move_pct"] for r in result["predictions"]}
    ids = {event for group in plan["folds"] for event in group["outer"]["evaluation_event_ids"]}
    events = [e for e in registration.document["size_panel"]["context"]["target_events"] if e["event_id"] in ids]
    inner_fits = _candidate_evidence(conn, store, registration.run_id, plan)
    for group in plan["folds"]:
        year = group["outer"]["year"]
        key = content_hash([plan["plan_id"], year])
        forecasts.extend(read_record(conn, store, registration.run_id, "outer-forecasts:" + key))
        key = content_hash([plan["plan_id"], year, "comparison"])
        comparison = read_record(conn, store, registration.run_id, "comparison-terminal:" + key)["result"]
        comparisons.extend(comparison["forecasts"])
        comparison_fits.extend({"outer_year": year, **fit} for fit in comparison["fit_details"])
    report = {"plan_id": plan["plan_id"], "search": result, "paired": paired_report(events, forecasts, comparisons, targets,
                horizons=horizons,
                implied_moves=implied_moves,
                implied_scales=scales,
                label_reasons={o["event_id"]: o["reason"] for f in result["folds"]
                    for o in f["evaluation_inputs"].get("omitted_labels", [])}),
            "preflight": resource_report(plan), "inner_fits": inner_fits, "comparison_fits": comparison_fits,
            "frozen_eligibility": registration.document["size_panel"]["context"]["eligibility_plan"],
            "label_exclusions": [{"outer_year": f["outer_year"], **omission} for f in result["folds"]
                for omission in f["evaluation_inputs"].get("omitted_labels", [])]}
    fallbacks = [{"outer_year": f["outer_year"],
        "horizon_sessions": d["horizon_sessions"], "fallback_counts": d["fallback_counts"]}
        for f in result["folds"] for d in f.get("fit_diagnostics", []) if "fallback_counts" in d]
    if fallbacks:
        report["paired"]["v1_4_fallbacks"] = fallbacks
        report["paired"]["result_id"] = content_hash({k: v for k, v in report["paired"].items() if k != "result_id"})
    print(f"[stage] collect done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return report


def resource_report(plan):
    counts = plan["counts"]
    jobs = {"inner_candidates": counts["candidates"] * counts["inner_folds"],
            "outer_challengers": counts["outer_folds"], "outer_comparators": counts["outer_folds"]}
    invocations = sum(jobs.values())
    return {"maximum_child_invocations": invocations, "child_jobs": jobs,
        "configured_child_wall_envelope_seconds": invocations * plan["request"]["limits"]["child"]["wall_seconds"],
        "counts": plan["counts"], "limits": plan["request"]["limits"],
        "child": "Linux RSS watchdog/RLIMIT_CPU, one CPU affinity, parent wall timeout",
        "parent_and_job": "size_search CLI RSS watchdog and wall alarm; library caller must contain",
        "build": "not invoked; existing separately bounded stored panel required",
        "coverage_and_cost": "unknown_until_authorized_local_execution; counts_not_feasibility",
        "availability": "reconstructed; vendor_publication_and_event_schedule_knowledge_unverified",
        "final_read_authority": "none"}
