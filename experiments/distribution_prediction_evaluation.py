"""Admitted outer target read and deterministic score/diagnostic evidence replay."""
from itertools import chain

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution_diagnostics import (
    interval_diagnostics,
    pit_bracket,
    prepare_empirical_diagnostics,
)
from engine.v2.research.progress import Progress, stage
from experiments.distribution_prediction_baseline import score_training_baseline
from experiments.distribution_prediction_evidence import read_frozen_outer
from experiments.distribution_prediction_scores import arm_forecasts, score_forecasts
from experiments.magnitude_search_evidence import evaluation_targets, validate_targets
from experiments.magnitude_search_inputs import load_inputs

__all__ = ["evaluate_outer", "read_evaluated_outer"]


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution prediction target or score evidence differs")


def _context(run, fits, year):
    frozen, values = read_frozen_outer(run, fits, year)
    context = fits.trials.read_fit_context("outer", year, None)
    _require(canonical_json(context["fold"]) == canonical_json(frozen["fold"])
             and content_hash(context["admission"]) == frozen["admission_id"])
    admitted = context["admission"]
    inputs = load_inputs(run.trials.store, admitted["manifest"], admitted["inputs_id"])
    return frozen, values, context["fold"], inputs["preflight"]


def _targets(preflight, fold, receipt, rows):
    try:
        validate_targets(preflight, fold, receipt, rows, strict_json=True)
        labels = {row["event_id"]: row["absolute_move_pct"] for row in rows}
        omitted = {row["event_id"]: {"omitted": row["reason"]} for row in receipt.get("omitted_labels", [])}
        _require(set(labels).isdisjoint(omitted) and set(labels) | set(omitted) == set(fold["evaluation_event_ids"]))
        return labels | omitted
    except (KeyError, TypeError, ValueError, OverflowError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution evaluation target receipt differs") from None


def _diagnostics(value, targets, year, horizon):
    result, pool = value["result"], value["pool"]
    baseline = prepare_empirical_diagnostics([row["absolute_move_pct"] for row in value["baseline"]["training"]])
    progress = Progress("prediction event diagnostics", len(result["rows"]))
    for forecast in result["rows"]:
        event = forecast["event_id"]
        observation = targets[event]
        arms = arm_forecasts(forecast, result["tail"], pool, receipt=result["receipt"])
        for arm, law in (*arms.items(), ("training_empirical", baseline)):
            row = {"event_id": event, "outer_year": year, "horizon": horizon, "arm": arm,
                   "observation": observation, "status": "unavailable"}
            if isinstance(observation, dict):
                row.update(status="missing_target", observation=None, reason=observation["omitted"])
            elif law is not None and not (isinstance(law, dict) and law["tail"]["policy"] == "unresolved"):
                if isinstance(law, dict):
                    lower, upper = pit_bracket(law, observation)
                    diagnostic = {"intervals": interval_diagnostics(law, observation), "pit_lower": lower, "pit_upper": upper}
                else:
                    diagnostic = (law if callable(law) else prepare_empirical_diagnostics(law))(observation)
                row.update(status="available", **diagnostic)
            if row["status"] == "unavailable":
                row["reason"] = "unavailable_forecast" if law is None else "unresolved_tail"
            yield row
        progress.tick()


def _rows(value, targets, year, horizon):
    scores = chain(score_forecasts(value["result"], value["pool"], targets, outer_year=year, horizon=horizon),
        score_training_baseline(value["baseline"], targets, receipt=value["result"]["receipt"]))
    return list(scores), list(_diagnostics(value, targets, year, horizon))


@stage("prediction admitted outer evaluation")
def evaluate_outer(run, fits, year):
    plan = run.admit()
    run.record("evaluation-start:" + year, {"run_plan_id": plan["run_plan_id"]})
    frozen, values, fold, preflight = _context(run, fits, year)
    frame = evaluation_targets(run.trials.repository, run.trials.source.document, fold)
    receipt, rows = dict(frame.attrs), frame[["event_id", "absolute_move_pct"]].to_dict("records")
    targets = _targets(preflight, fold, receipt, rows)
    prefix = "distribution_prediction_" + year
    result = {"schema_version": "distribution_prediction_evaluation.v1", "run_plan_id": plan["run_plan_id"],
        "outer_year": year, "frozen_id": content_hash(frozen), "final_read_authority": "none",
        "targets": run.publish(prefix + "_targets", rows, count=len(rows)),
        "receipt": run.publish(prefix + "_target_receipt", [receipt], count=1), "horizons": []}
    for horizon, value in values.items():
        scores, diagnostics = _rows(value, targets, year, horizon)
        count = 4 * len(fold["evaluation_event_ids"])
        result["horizons"].append({"horizon": horizon,
            "scores": run.publish(prefix + f"_{horizon}_scores", scores, count=count),
            "diagnostics": run.publish(prefix + f"_{horizon}_diagnostics", diagnostics, count=count)})
    return run.record("evaluated:" + year, result)


@stage("prediction evaluated evidence replay")
def read_evaluated_outer(run, fits, year):
    frozen, values, fold, preflight = _context(run, fits, year)
    record = run.record("evaluated:" + year)
    header = {"schema_version": "distribution_prediction_evaluation.v1", "run_plan_id": frozen["run_plan_id"],
        "outer_year": year, "frozen_id": content_hash(frozen), "final_read_authority": "none"}
    _require(type(record) is dict and set(record) == set(header) | {"targets", "receipt", "horizons"}
        and canonical_json({key: record[key] for key in header}) == canonical_json(header)
        and canonical_json(run.record("evaluation-start:" + year)) == canonical_json({"run_plan_id": frozen["run_plan_id"]}))
    receipt = list(run.read_section(record["receipt"]))
    _require(len(receipt) == 1 and type(record["horizons"]) is list
             and [entry["horizon"] for entry in record["horizons"]] == [1, 3, 5])
    targets = _targets(preflight, fold, receipt[0], list(run.read_section(record["targets"])))
    scores, diagnostics = [], []
    for entry in record["horizons"]:
        _require(set(entry) == {"horizon", "scores", "diagnostics"})
        actual = _rows(values[entry["horizon"]], targets, year, entry["horizon"])
        for name, expected, output in zip(("scores", "diagnostics"), actual, (scores, diagnostics)):
            _require(canonical_json(list(run.read_section(entry[name]))) == canonical_json(expected))
            output.extend(expected)
    return record, scores, diagnostics
