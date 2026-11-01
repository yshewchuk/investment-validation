"""Frozen decisions before outcomes, then conditional book/forecast reporting."""
import sys

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import canonical_json, from_document
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_inputs import load_magnitude_targets
from engine.v2.research.progress import Progress, stage
from experiments.distribution_book import funded_book
from experiments.distribution_market import (
    decision_quotes,
    outcome_quotes,
    read_benchmark,
    read_bridge,
    side_cost,
)
from experiments.distribution_oof import residual_pool
from experiments.distribution_report import forecast_report, paired_report
from experiments.distribution_run_request import PRIMARY_ARMS, require, validate_request
from experiments.distribution_scores import (
    arm_forecasts,
    economic_forecast,
    profit_scores,
    score_forecasts,
)
from experiments.magnitude_search_shards import iter_section


def _sealed_plan(run):
    request = validate_request(run.request)
    sealed = run.record("seal")
    require(isinstance(sealed, dict) and isinstance(sealed.get("request"), dict)
            and canonical_json(request) == canonical_json(sealed["request"]))
    return sealed


@stage("distribution outer evaluation")
def evaluate_outer(run, group, fits, results, plan, books, quotes, opportunities, sections, calendar):
    sealed = _sealed_plan(run)
    try:
        matches = canonical_json(plan) == canonical_json(sealed)
    except (TypeError, ValueError, RecursionError):
        matches = False
    require(matches)
    plan, request, fold = sealed, sealed["request"], group["outer"]
    year, ids = fold["year"], set(fold["evaluation_event_ids"])
    slots = {f"intents:{year}:{alpha}" for alpha in (*request["alphas"], *books)}
    if any(run.record(slot) is not None for slot in sorted(slots)):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution outer group already has recorded intents")
    events = [event for event in run.trials.source.document["size_panel"]["context"]["target_events"] if event["event_id"] in ids]
    require(len(events) == len(ids))
    pool = {h: residual_pool(fits, "outer", year, h) for h in results}
    benchmarks = read_benchmark(run.trials.store, next((item for item in request["benchmarks"] if item["outer_year"] == year), None), fold)
    snapshot = from_document(SnapshotRef, plan["quote_snapshot"])
    all_decisions = decision_quotes(run.trials.repository, snapshot, events, request["alphas"], request["economics"], calendar=calendar)
    rows = {row["event_id"]: row for row in results[1]["rows"]}
    for alpha in request["alphas"]:
        decisions = all_decisions[alpha]
        bridge = read_bridge(run.trials.store, next(item for item in request["bridges"] if item["outer_year"] == year and item["alpha"] == alpha), fold)
        contracts = request["book"]["contracts_per_trade"]
        local, diagnostics = {arm: [] for arm in books[alpha] if arm in PRIMARY_ARMS or plan["available_benchmarks"][arm]}, []
        progress = Progress("distribution decision integration", len(events))
        for event in events:
            key = event["event_id"]
            row, decision = decisions.get(key, (None, None))
            values = arm_forecasts(rows[key], results[1]["tail"], pool[1], benchmarks.get(key), receipt=results[1]["receipt"])
            forecasts = {arm: economic_forecast(value, bridge, decision, scale=contracts * request["economics"]["multiplier"],
                round_trip_cost=2 * side_cost(request["economics"], contracts)) for arm, value in values.items()}
            common = all(forecasts.get(arm) is not None for arm in PRIMARY_ARMS)
            dates = {name: row[field].date().isoformat() if row else None for name, field in (("decision_at", "decision_date"), ("entry_at", "entry_date"))}
            base = {"event_id": key, "ticker": event["ticker"], "event_date": str(event["event_date"])[:10], **dates}
            for arm in local:
                selected = (benchmarks.get(key, {}).get("gate_selected") is True and decision is not None) if arm == "gate" else (
                    common and forecasts.get(arm) is not None and forecasts[arm]["expected_net_profit_usd"] > request["minimum_net_ev_usd"])
                local[arm].append({**base, "selected": selected})
            diagnostics.append({"event_id": key, "forecasts": forecasts, "primary_common": common})
            if alpha == request["primary_alpha"]:
                opportunities.append({"event_id": key, "ticker": event["ticker"], "decision_date": dates["decision_at"] or base["event_date"]})
            progress.tick()
        print(f"  [distribution decisions] year={year} alpha={alpha} opportunities={len(events)} "
              f"primary_common={sum(row['primary_common'] for row in diagnostics)} "
              f"selected={ {arm: sum(row['selected'] for row in rows) for arm, rows in local.items()} }", file=sys.stderr)
        for arm in local:
            books[alpha][arm].extend(local[arm])
        section = run.publish(f"intents:{year}:{alpha}", [{"event_id": row["event_id"], "arms": {arm: local[arm][i] for arm in local}, **row} for i, row in enumerate(diagnostics)])
        run.record(f"intents:{year}:{alpha}", section, replay=False)
        sections.append(section)
    for alpha in request["alphas"]:
        quotes[alpha].update(outcome_quotes(run.trials.repository, snapshot, all_decisions[alpha], alpha, request["economics"], contracts))
        frozen = run.record(f"intents:{year}:{alpha}")
        intents = iter_section(run.trials.store, frozen["name"], frozen["descriptor"], lambda row: (row["position"],))
        sections.append(run.publish(f"profit-scores:{year}:{alpha}", profit_scores(intents, quotes[alpha])))
    binding = run.trials.source.document
    targets = load_magnitude_targets(run.trials.repository, from_document(SnapshotRef, binding["snapshot"]),
        as_of_month=binding["holdouts"]["holdout_as_of_month"], event_ids=sorted(ids),
        labels_before=binding["magnitude_execution"]["evaluation_labels_before"], omit_not_computed=True)
    labels = dict(zip(targets.event_id, targets.absolute_move_pct))
    omitted = {row["event_id"]: {"omitted": row["reason"]} for row in targets.attrs.get("omitted_labels", [])}
    require(set(labels).isdisjoint(omitted) and set(labels) | set(omitted) == ids)
    for horizon, result in results.items():
        sections.append(run.publish(f"scores:{year}:{horizon}", score_forecasts(result, pool[horizon], labels | omitted, outer_year=year, horizon=horizon)))


@stage("distribution final reports")
def finish_reports(run, books, quotes, opportunities, sections):
    request = _sealed_plan(run)["request"]
    reports = []
    for alpha, intents in books.items():
        funded = {arm: funded_book(rows, quotes[alpha], config=request["book"], report_end=request["report_end"])
            if rows else None for arm, rows in intents.items()}
        report = paired_report(opportunities, funded, **request["uncertainty"])
        risk, primary = request["risk"], report["contrasts"]["same_mean"]
        report["risk_checks"] = {arm: {"loss_limit_met": None if summary["unresolved"] else summary["settled_net_profit"] >= -risk["maximum_loss_usd"],
            "settled_drawdown_met": summary["settlement_drawdown"] <= risk["maximum_settled_drawdown_usd"],
            "completeness_met": (summary["funded"] - summary["unresolved"]) / summary["funded"] >= risk["completeness_floor"] if summary["funded"] else None}
            for arm, summary in report["arms"].items()}
        report["uplift_met"] = primary["net_delta_per_opportunity"] >= risk["minimum_uplift_usd"] if primary.get("net_delta_per_opportunity") is not None else None
        for arm, book in funded.items():
            if book is not None:
                sections.extend([run.publish(f"book:{alpha}:{arm}", book["per_event"]),
                                 run.publish(f"curve:{alpha}:{arm}", book["curve"])])
        reports.append({"alpha": alpha, "report": report})
    return {"schema_version": "distribution_run_report.v1", "reports": reports, "sections": sections,
        "forecast": forecast_report((row for section in sections if section["name"].startswith("scores:")
            for row in iter_section(run.trials.store, section["name"], section["descriptor"], lambda row: (row["position"],))),
            opportunities, **request["uncertainty"]),
        "primary_alpha": request["primary_alpha"], "risk_limits": request["risk"],
        "selector_policy": request["selector_policy"], "calibration_identity": "uncalibrated",
        "current_strategy_superiority_claim": False, "capacity": "not_established_no_depth",
        "benchmark_qualification": "imported_fold_contract_origin_assumed", "promotion_authority": "none"}
