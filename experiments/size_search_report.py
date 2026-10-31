"""Collect verified selected-outer and comparator journals without new fitting."""
import sys
import time

from engine.v2.foundation import content_hash
from experiments.magnitude_search_outer import collect_outer_results
from experiments.magnitude_search_plan import trial_key
from experiments.magnitude_search_records import read_record
from experiments.size_paired_report import paired_report
from experiments.v2_magnitude_search_run import _admit_driver


def collect(conn, store, repository, registration, request, plan_ref):
    print("[stage] collect start", file=sys.stderr)
    started = time.monotonic()
    plan = _admit_driver(conn, store, registration, request, plan_ref)
    result = collect_outer_results(conn, store, repository, registration, request, plan_ref)
    forecasts, comparisons, events, inner_fits, comparison_fits = [], [], [], [], []
    targets = {r["event_id"]: r["absolute_move_pct"] for r in result["predictions"]}
    ids = {event for group in plan["folds"] for event in group["outer"]["evaluation_event_ids"]}
    events = [e for e in registration.document["size_panel"]["context"]["target_events"] if e["event_id"] in ids]
    for group in plan["folds"]:
        year = group["outer"]["year"]
        for fold in group["inner"]:
            for candidate in plan["candidates"]:
                key = trial_key(plan, year, fold["year"], candidate["candidate_id"])
                receipt = read_record(conn, store, registration.run_id, "execution:" + key)
                inner_fits.append({"outer_year": year, "inner_year": fold["year"],
                    "candidate_id": candidate["candidate_id"], "execution_slot": "execution:" + key,
                    **{k: receipt[k] for k in ("status", "fit_attempt_count", "completed_fit_count")},
                    "diagnostics": receipt["diagnostics"] if receipt["status"] == "succeeded" else []})
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
