"""Freeze/reverify all-horizon prediction evidence before any evaluation targets."""
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, stage
from experiments.distribution_oof import residual_pool
from experiments.distribution_prediction_baseline import training_baseline

__all__ = ["freeze_outer", "read_frozen_outer"]


def _require(condition):
    if not condition:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "distribution frozen prediction evidence differs")


def _evidence(run, fits, year):
    plan = run.admit()
    context = fits.trials.read_fit_context("outer", year, None)
    fold = context["fold"]
    _require(context["plan"]["plan_id"] == plan["admission_plan_id"]
             and set(context["plan"]["recipe"]["horizons"]) == {1, 3, 5})
    events = fold["evaluation_event_ids"]
    _require(bool(events) and len(events) == len(set(events)) and set(events) <= set(plan["evaluation_event_ids"]))
    header = {"schema_version": "distribution_prediction_frozen.v1", "run_plan_id": plan["run_plan_id"],
        "outer_year": year, "fold": fold, "admission_id": content_hash(context["admission"]),
        "final_read_authority": "none"}
    values = {}
    for horizon in (1, 3, 5):
        result = fits.read("outer", year, horizon)
        receipt, claim = result["receipt"], result["receipt"]["claim"]
        _require(receipt["schema_version"] == "distribution_fitted.v2" and receipt["status"] == "completed"
            and claim["admission_id"] == header["admission_id"] and claim["plan_id"] == plan["admission_plan_id"]
            and (claim["role"], claim["outer_year"], claim["inner_year"], claim["trial"]["horizon"]) == ("outer", year, None, horizon)
            and [row["event_id"] for row in result["rows"]] == events)
        pool, baseline = residual_pool(fits, "outer", year, horizon), training_baseline(fits, year, horizon)
        _require(baseline["result_id"] == receipt["result_id"])
        values[horizon] = {"result": result, "pool": pool, "baseline": baseline}
    return header, values


def _reference(horizon, value):
    return {"horizon": horizon, "receipt": value["result"]["receipt"], "tail": value["result"]["tail"],
            "pool": value["pool"], "baseline": value["baseline"]}


@stage("prediction all-horizon freeze")
def freeze_outer(run, fits, year):
    """Single-use freeze of raw fitted laws and deterministic shared controls.

    Native bounded fits plus stored residual/baseline evidence fully specify
    every forecast. Shared references are stored once per horizon, not expanded
    into an events-times-residuals matrix. This operation never opens targets.
    """
    plan = run.admit()
    run.record("freeze-start:" + year, {"run_plan_id": plan["run_plan_id"]})
    header, values = _evidence(run, fits, year)
    sections, progress = [], Progress("prediction horizon publication", 3)
    for horizon, value in values.items():
        prefix = "distribution_prediction_" + year + "_" + str(horizon)
        forecasts = run.publish(prefix + "_forecasts", value["result"]["rows"], count=len(header["fold"]["evaluation_event_ids"]))
        reference = run.publish(prefix + "_reference", [_reference(horizon, value)], count=1)
        sections.append({"horizon": horizon, "forecasts": forecasts, "reference": reference})
        progress.tick()
    return run.record("frozen:" + year, {**header, "horizons": sections})


@stage("prediction frozen evidence replay")
def read_frozen_outer(run, fits, year):
    """Fresh native/fit/control verification plus CAS equality; no writes/refits/targets.

    A self-consistent content hash alone cannot authenticate caller-supplied
    controls. Rebuild admitted references and compare the stored bytes each time.
    """
    header, values = _evidence(run, fits, year)
    record = run.record("frozen:" + year)
    _require(type(record) is dict and set(record) == set(header) | {"horizons"}
             and canonical_json({key: record[key] for key in header}) == canonical_json(header)
             and canonical_json(run.record("freeze-start:" + year)) == canonical_json({"run_plan_id": header["run_plan_id"]}))
    sections = record["horizons"]
    _require(type(sections) is list and [entry["horizon"] for entry in sections] == [1, 3, 5])
    progress = Progress("prediction horizon verification", 3)
    for entry in sections:
        _require(set(entry) == {"horizon", "forecasts", "reference"})
        value = values[entry["horizon"]]
        _require(canonical_json(list(run.read_section(entry["forecasts"]))) == canonical_json(value["result"]["rows"])
            and canonical_json(list(run.read_section(entry["reference"]))) == canonical_json([_reference(entry["horizon"], value)]))
        progress.tick()
    return record, values
