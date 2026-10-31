"""Collect verified selected-outer and comparator journals without new fitting."""
import sys
import time

from engine.v2.foundation import content_hash
from experiments.magnitude_search_outer import collect_outer_results
from experiments.magnitude_search_records import read_record
from experiments.size_paired_report import paired_report
from experiments.v2_magnitude_search_run import _admit_driver


def candidate_evidence(conn, store, run_id, plan):
    """Read-only per-run candidate listing bound to the sealed plan's exact trial keys."""
    rows = []
    for group in plan["folds"]:
        outer_year = group["outer"]["year"]
        for fold in group["inner"]:
            inner_year = fold["year"]
            for candidate in plan["candidates"]:
                candidate_id = candidate["candidate_id"]
                key = content_hash([plan["plan_id"], outer_year, inner_year, candidate_id])
                execution = read_record(conn, store, run_id, "execution:" + key)
                terminal = read_record(conn, store, run_id, "terminal:" + key)
                execution = execution if isinstance(execution, dict) else {}
                terminal = terminal if isinstance(terminal, dict) else {}
                status = execution.get("status", terminal.get("status"))
                fits = execution.get("fits") or []
                rows.append({
                    "trial_id": key, "run_id": run_id, "outer_year": outer_year,
                    "inner_year": inner_year, "candidate_id": candidate_id,
                    "execution_slot": "execution:" + key, "status": status,
                    "terminal_status": terminal.get("status"),
                    "fit_attempt_count": execution.get("fit_attempt_count"),
                    "completed_fit_count": execution.get("completed_fit_count"),
                    "fit_seconds": [{"horizon_sessions": fit.get("horizon_sessions"),
                                     "seconds": fit["fit_seconds"]} for fit in fits
                                    if isinstance(fit, dict) and fit.get("fit_seconds") is not None],
                    "score_seconds": terminal.get("score_seconds"),
                    "rss_delta_kib": terminal.get("rss_delta_kib"),
                    "inner_metrics": terminal.get("inner_metrics"),
                    "diagnostics": execution.get("diagnostics", []) if status == "succeeded" else []})
    return rows


def collect(conn, store, repository, registration, request, plan_ref):
    print("[stage] collect start", file=sys.stderr)
    started = time.monotonic()
    plan = _admit_driver(conn, store, registration, request, plan_ref)
    result = collect_outer_results(conn, store, repository, registration, request, plan_ref)
    forecasts, comparisons, events, comparison_fits = [], [], [], []
    targets = {r["event_id"]: r["absolute_move_pct"] for r in result["predictions"]}
    ids = {event for group in plan["folds"] for event in group["outer"]["evaluation_event_ids"]}
    events = [e for e in registration.document["size_panel"]["context"]["target_events"] if e["event_id"] in ids]
    inner_fits = candidate_evidence(conn, store, registration.run_id, plan)
    for group in plan["folds"]:
        year = group["outer"]["year"]
        key = content_hash([plan["plan_id"], year])
        forecasts.extend(read_record(conn, store, registration.run_id, "outer-forecasts:" + key))
        key = content_hash([plan["plan_id"], year, "comparison"])
        comparison = read_record(conn, store, registration.run_id, "comparison-terminal:" + key)["result"]
        comparisons.extend(comparison["forecasts"])
        comparison_fits.extend({"outer_year": year, **fit} for fit in comparison["fit_details"])
    report = {"plan_id": plan["plan_id"], "search": result, "paired": paired_report(events, forecasts, comparisons, targets,
                label_reasons={o["event_id"]: o["reason"] for f in result["folds"]
                    for o in f["evaluation_inputs"].get("omitted_labels", [])}),
            "preflight": resource_report(plan), "inner_fits": inner_fits, "comparison_fits": comparison_fits,
            "frozen_eligibility": registration.document["size_panel"]["context"]["eligibility_plan"],
            "label_exclusions": [{"outer_year": f["outer_year"], **omission} for f in result["folds"]
                for omission in f["evaluation_inputs"].get("omitted_labels", [])]}
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
