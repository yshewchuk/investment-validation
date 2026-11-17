"""Verified IV development report over the existing search/comparison journals."""
from engine.v2.foundation import content_hash
from engine.v2.research.progress import decision, stage
from experiments.iv_crush_paired_report import paired_report, paired_report_v2
from experiments.iv_crush_plan import target
from experiments.iv_crush_registration import validate_registration
from experiments.iv_distribution_report import collect_distribution, outer_arrays, read_trial_arrays
from experiments.magnitude_search_outer import collect_outer_results
from experiments.magnitude_search_records import read_record
from experiments.magnitude_search_request_schema import regression_request_version
from experiments.size_search_report import _candidate_evidence
from experiments.size_search_report import resource_report as shared_resource_report
from experiments.v2_magnitude_search_run import _admit_driver


def _implied_inputs(fold, inputs, horizons, sink):
    """Collect report-only iv30/exern_iv30 for the fold's evaluation events from verified outer inputs."""
    keys = {(event, h) for event in fold["evaluation_event_ids"] for h in horizons}
    for row in inputs["features"]:
        key = (row["event_id"], row["horizon_sessions"])
        if key in keys:
            sink[key] = {"iv30": row["iv30"], "exern_iv30": row["exern_iv30"]}


def resource_report(plan):
    """Reuse fixed schedule/resource accounting, without asserting measured feasibility."""
    return {**shared_resource_report(plan), "target": target(),
        "parent_and_job": "IV CLI enforces parent RSS/wall and job deadline; library caller must contain",
        "comparison": "causal signed IV HGB refit and all-admitted-label training mean",
        "timing": "IV fit evidence is untimed; configured ceilings are not observed cost"}


def collect(conn, store, repository, registration, request, plan_ref):
    """Collect once, verifying every comparator before reporting; never execute missing work."""
    with stage("IV collect"):
        validate_registration(registration.document)
        plan = _admit_driver(conn, store, registration, request, plan_ref)
        v2 = regression_request_version(request["schema_version"]) == 2
        implied = {}
        result = collect_outer_results(
            conn, store, repository, registration, request, plan_ref,
            input_callback=(lambda fold, inputs: _implied_inputs(fold, inputs, plan["horizons"], implied))
            if v2 else None)
        trial_arrays = read_trial_arrays(conn, store, registration.run_id, plan)
        outer_rows = outer_arrays(plan, result["predictions"])
        selected_ids = {fold["outer_year"]: fold["candidate"]["candidate_id"]
                        for fold in result["folds"]}
        distribution = collect_distribution(plan, trial_arrays, outer_rows, selected_ids)
        trial_arrays.clear()
        forecasts, comparisons, fits = [], [], []
        for group in plan["folds"]:
            year = group["outer"]["year"]
            key = content_hash([plan["plan_id"], year])
            forecasts.extend(read_record(conn, store, registration.run_id, "outer-forecasts:" + key))
            key = content_hash([plan["plan_id"], year, "comparison"])
            comparison = read_record(conn, store, registration.run_id, "comparison-terminal:" + key)["result"]
            comparisons.extend(comparison["forecasts"])
            fits.extend({"outer_year": year, **fit} for fit in comparison["fit_details"])
        ids = {event for group in plan["folds"] for event in group["outer"]["evaluation_event_ids"]}
        events = [event for event in registration.document["regression_plan"]["events"] if event["event_id"] in ids]
        targets = {row["event_id"]: row["crush_pct_iv30"] for row in result["predictions"]}
        if v2:
            decision("IV report", "paired report v2")
            paired = paired_report_v2(events, forecasts, comparisons, targets, implied,
                                      horizons=plan["horizons"], distribution=distribution)
        else:
            decision("IV report", "paired report v1")
            paired = paired_report(events, forecasts, comparisons, targets, horizons=plan["horizons"])
        report = {"schema_version": "iv_crush_search_report.v1", "plan_id": plan["plan_id"], "search": result,
            "paired": paired,
            "preflight": resource_report(plan), "comparison_fits": fits,
            "distribution": distribution,
            "inner_fits": _candidate_evidence(conn, store, registration.run_id, plan),
            "eligibility_policy_ref": plan["request"]["eligibility_policy_ref"],
            "final_read_authority": "none"}
        return {**report, "result_id": content_hash(report)}
