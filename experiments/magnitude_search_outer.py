"""Bounded selected-candidate outer fits; immutable evidence, no final authority."""
from pathlib import Path

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import ArtifactError, canonical_json, content_hash, from_document
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import require_current
from engine.v2.research.magnitude_history_store import read_stored
from engine.v2.research.magnitude_inputs import (
    load_magnitude_targets,
    magnitude_recipe,
    preflight_magnitude_targets,
    qualify_magnitude_training_targets,
)
from experiments.magnitude_history_inputs import history_binding
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_execution import (
    execution_limits,
    run_fitted_trial,
    validate_execution_limits,
)
from experiments.magnitude_search_plan import read_search_plan
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.magnitude_search_trials import SearchTrials


def _conflict():
    return fail("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation evidence differs from its sealed inputs")


def _equal(left, right):
    if canonical_json(left) != canonical_json(right):
        raise _conflict()


def _native_selection(trials, plan, read):
    """Presence checks make public execution calls strictly completed replays."""
    selection = read("selection")
    if selection is None:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation requires an existing frozen selection")
    attempts = []
    for outer in plan["folds"]:
        for candidate in plan["candidates"]:
            for inner in outer["inner"]:
                args = (outer["outer"]["year"], inner["year"], candidate["candidate_id"])
                key = content_hash([plan["plan_id"], *args])
                execution, terminal = read("execution:" + key), read("terminal:" + key)
                if not isinstance(execution, dict) or not isinstance(terminal, dict):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation requires completed native inner trials")
                try:
                    limits = execution["attempt"]["limits"]
                    validate_execution_limits(limits)
                except (KeyError, TypeError, OpsError):
                    raise _conflict() from None
                attempts.append((args, limits))
    _equal(trials.freeze(), selection)
    for args, limits in attempts:
        trials.execute(*args, limits=limits)
    return selection


def _inputs(repository, binding, fold):
    snapshot = from_document(SnapshotRef, binding["snapshot"])
    month = binding["holdouts"]["holdout_as_of_month"]
    history = history_binding(repository, snapshot, binding["magnitude_plan"],
        binding["magnitude_history_request"], as_of_month=month, event_ids=binding["event_ids"])
    _equal(history, binding["magnitude_history"])
    participants = set(fold["training_candidate_event_ids"] + fold["evaluation_event_ids"])
    preflight = preflight_magnitude_targets(repository, snapshot, as_of_month=month, event_ids=sorted(participants))
    stored = read_stored(repository, from_document(SnapshotRef, history["feature_snapshot"]), history["context"])
    train = qualify_magnitude_training_targets(repository, snapshot, as_of_month=month,
        labels_before=fold["training_labels_before"], event_ids=fold["training_candidate_event_ids"])
    return {"preflight": preflight, "stored_inputs_id": stored["inputs_id"], "training_inputs": dict(train.attrs),
            "training": train[["event_id", "absolute_move_pct"]].to_dict("records"),
            "features": [row for row in stored["rows"] if row["event_id"] in participants]}


def _fit_evidence(inputs, fold, plan, execution):
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    train = [row["event_id"] for row in inputs["training"]]
    result = []
    for horizon in plan["horizons"]:
        documents = []
        for events in (train, fold["evaluation_event_ids"]):
            rows = [indexed[event, horizon] for event in events]
            documents.append({"columns": execution["features"], "row_keys": [
                [row[name] for name in ("event_id", "context_id", "horizon_sessions")] for row in rows],
                "values": [[row[name] for name in execution["features"]] for row in rows]})
        documents[0]["absolute_move_pct"] = [row["absolute_move_pct"] for row in inputs["training"]]
        result.append({"horizon_sessions": horizon, "training_event_ids": train,
            "evaluation_event_ids": fold["evaluation_event_ids"], "training_matrix_id": content_hash(documents[0]),
            "prediction_matrix_id": content_hash(documents[1])})
    return result


def _forecasts(rows, fold, horizons):
    import math

    try:
        expected = {(event, horizon) for event in fold["evaluation_event_ids"] for horizon in horizons}
        found = {}
        if not isinstance(rows, list) or len(rows) != len(expected):
            raise ValueError
        for row in rows:
            if set(row) != {"event_id", "horizon_sessions", "predicted_absolute_move_pct"}:
                raise ValueError
            value, horizon = row["predicted_absolute_move_pct"], row["horizon_sessions"]
            pair = (row["event_id"], horizon)
            if (type(horizon) is not int or type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or pair not in expected or pair in found):
                raise ValueError
            found[pair] = {**row, "predicted_absolute_move_pct": float(value)}
        return [found[pair] for pair in sorted(expected)]
    except (ValueError, TypeError, KeyError, OverflowError):
        raise _conflict() from None


def _validate_targets(inputs, validation, fold):
    """Reconstruct the exact protected native receipt, without reopening labels."""
    import math

    preflight = inputs["preflight"]
    events = set(fold["evaluation_event_ids"])
    consumed = [row for row in preflight["planned_inputs"] if row["event_id"] in events]
    ordered = [row["event_id"] for row in consumed]
    if len(ordered) != len(events) or set(ordered) != events:
        raise _conflict()
    recipe = magnitude_recipe()
    receipt = {"schema_version": "research_magnitude_targets.v1",
        **{name: preflight[name] for name in ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use", "purpose", "computed_moves_version")},
        "labels_before": fold["evaluation_labels_before"], "target_recipe": recipe,
        "target_recipe_id": content_hash(recipe), "consumed_inputs": consumed, "excluded_future_events": []}
    _equal(validation["evaluation_inputs"], {**receipt, "inputs_id": content_hash(receipt), "holdout_exclusions": []})
    if set(validation) != {"evaluation_inputs", "targets"} or [row["event_id"] for row in validation["targets"]] != ordered:
        raise _conflict()
    for row in validation["targets"]:
        value = row["absolute_move_pct"]
        if (set(row) != {"event_id", "absolute_move_pct"} or type(value) not in (int, float)
                or value < 0 or not math.isfinite(value)):
            raise _conflict()


def _result(started, inputs, execution, forecasts, validation, fold, horizons):
    _validate_targets(inputs, validation, fold)
    targets = {row["event_id"]: row["absolute_move_pct"] for row in validation["targets"]}
    predictions = [{**row, "absolute_move_pct": targets[row["event_id"]]} for row in forecasts]
    result = {"schema_version": "magnitude_search_outer_result.v1", "outer_id": started["outer_id"],
        "plan_id": started["plan_id"], "outer_year": fold["year"], "candidate": started["candidate"],
        "selection_id": started["selection_id"], "inputs_id": content_hash(inputs),
        "execution_id": content_hash(execution), "validation_id": content_hash(validation),
        "training_inputs": inputs["training_inputs"], "evaluation_inputs": validation["evaluation_inputs"],
        "fit_attempt_count": execution["fit_attempt_count"], "event_count": len(targets),
        "forecast_count": len(predictions), "predictions": predictions, "metrics": magnitude_metrics(predictions),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon])} for horizon in horizons],
        "prospective_candidate_id": started["prospective_candidate_id"],
        "evaluation_authority": "selected_outer_fold_only", "final_read_authority": "none",
        "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}}
    return {**result, "result_id": content_hash(result)}


def _replay(read, started, fold, plan, execution):
    try:
        _equal(read("start"), started)
        inputs = read("inputs")
        fits = _fit_evidence(inputs, fold, plan, execution)
        forecasts = _forecasts(read("forecasts"), fold, plan["horizons"])
        receipt = {"status": "succeeded", "fits": fits, "fit_attempt_count": len(fits),
                   "completed_fit_count": len(fits), "forecasts_id": content_hash(forecasts)}
        _equal(read("execution"), receipt)
        for fit in fits:
            for state in ("started", "completed"):
                _equal(read("fit-" + state + ":" + str(fit["horizon_sessions"])), fit)
        result = _result(started, inputs, receipt, forecasts, read("validation"), fold, plan["horizons"])
        _equal(read("terminal"), result)
        return result
    except (KeyError, TypeError, ValueError, OverflowError):
        raise _conflict() from None


def execute_outer_fold(conn, store, repository, registration, request, plan_ref, outer_year, *, limits):
    """Refit one frozen inner winner once; replay complete evidence without fitting."""
    validate_execution_limits(limits)
    require_current(conn, OWNER, MIGRATIONS)
    plan = read_search_plan(conn, store, registration, request, plan_ref)
    try:
        fold = next(item["outer"] for item in plan["folds"] if item["outer"]["year"] == outer_year)
    except StopIteration:
        raise fail("INVALID_EXPERIMENT_SPEC", "outer fold is outside the sealed search plan") from None
    binding = registration.document
    fold = {**fold, "evaluation_labels_before": binding["magnitude_execution"]["evaluation_labels_before"]}
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    selection = _native_selection(trials, plan, read)
    chosen = next(item["selected_candidate_id"] for item in selection["selections"] if item["outer_year"] == outer_year)
    candidate = next(item for item in plan["candidates"] if item["candidate_id"] == chosen)
    try:
        sources = source_closure(Path(__file__).resolve().parents[1], [
            "experiments/magnitude_search_outer.py", "experiments/magnitude_search_worker.py"])
    except (OpsError, ArtifactError, OSError, SyntaxError, ValueError):
        raise _conflict() from None
    key = content_hash([plan["plan_id"], outer_year])
    started = {"outer_id": key, "plan_id": plan["plan_id"], "outer_year": outer_year,
        "selection_id": content_hash(selection), "candidate": candidate, "limits": dict(limits),
        "outer_sources": sources, "prospective_candidate_id": selection["prospective_candidate_id"]}
    def saved(slot):
        return read("outer-" + slot + ":" + key)
    def append(slot, body):
        return append_record(conn, store, registration.run_id, "outer-" + slot + ":" + key, body, replay=False)
    execution = {**binding["magnitude_execution"], "fit_recipe": candidate["recipe"]}
    if saved("terminal") is not None:
        return _replay(saved, started, fold, plan, execution)
    forbid_fitting("execute_outer_fold")
    forbid_legacy_fitting("execute_outer_fold")
    resolved = execution_limits(limits)
    append("start", started)
    inputs = _inputs(repository, binding, fold)
    append("inputs", inputs)
    expected = _fit_evidence(inputs, fold, plan, execution)
    fits, pending = [], []
    def progress(state, fit):
        index = len(fits)
        if (state not in ("started", "completed") or index >= len(expected)
                or (state == "started") == bool(pending)):
            raise _conflict()
        _equal(fit, expected[index])
        append("fit-" + state + ":" + str(fit["horizon_sessions"]), fit)
        if state == "started":
            pending.append(fit)
        else:
            fits.append(pending.pop())
    result = run_fitted_trial(inputs, fold, plan, execution, resolved, progress)
    succeeded = result.get("event") == "succeeded"
    if ((succeeded and (pending or len(fits) != len(expected) or set(result) != {"event", "forecasts"}))
            or (not succeeded and (result != {"event": "failed"} or len(pending) != 1))):
        raise _conflict()
    forecasts = _forecasts(result["forecasts"], fold, plan["horizons"]) if succeeded else None
    receipt = {"status": result["event"], "fits": fits, "fit_attempt_count": len(fits) + len(pending),
               "completed_fit_count": len(fits), "forecasts_id": content_hash(forecasts) if succeeded else None}
    append("execution", receipt)
    if not succeeded:
        raise fail("EXPERIMENT_VARIANT_FAILED", "selected outer candidate failed; replacement and retry are forbidden")
    append("forecasts", forecasts)
    # All predictions and successful fit accounting are durable before any outer target opens.
    labels = load_magnitude_targets(repository, from_document(SnapshotRef, binding["snapshot"]),
        as_of_month=binding["holdouts"]["holdout_as_of_month"], labels_before=fold["evaluation_labels_before"],
        event_ids=fold["evaluation_event_ids"])
    validation = {"evaluation_inputs": dict(labels.attrs),
                  "targets": labels[["event_id", "absolute_move_pct"]].to_dict("records")}
    _validate_targets(inputs, validation, fold)
    append("validation", validation)
    append("terminal", _result(started, inputs, receipt, forecasts, validation, fold, plan["horizons"]))
    return _replay(saved, started, fold, plan, execution)


def collect_outer_results(conn, store, repository, registration, request, plan_ref):
    """Read every completed outer fold; pool event errors, never rank by outer loss."""
    require_current(conn, OWNER, MIGRATIONS)
    plan = read_search_plan(conn, store, registration, request, plan_ref)
    pending = []
    for item in plan["folds"]:
        year = item["outer"]["year"]
        key = content_hash([plan["plan_id"], year])
        started = read_record(conn, store, registration.run_id, "outer-start:" + key)
        terminal = read_record(conn, store, registration.run_id, "outer-terminal:" + key)
        if not isinstance(started, dict) or not isinstance(terminal, dict):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold")
        try:
            validate_execution_limits(started["limits"])
        except (KeyError, OpsError):
            raise _conflict() from None
        pending.append((year, started["limits"]))
    folds = [execute_outer_fold(conn, store, repository, registration, request, plan_ref, year, limits=limits)
             for year, limits in pending]
    predictions = [{**row, "outer_year": fold["outer_year"]} for fold in folds for row in fold["predictions"]]
    result = {"schema_version": "magnitude_search_outer_report.v1", "plan_id": plan["plan_id"],
        "folds": folds, "predictions": predictions, "metrics": magnitude_metrics(predictions),
        "event_count": sum(fold["event_count"] for fold in folds), "forecast_count": len(predictions),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon])} for horizon in plan["horizons"]],
        "prospective_candidate_id": folds[-1]["prospective_candidate_id"],
        "winner_policy": "latest_outer_inner_selection_before_its_evaluation",
        "final_read_authority": "none", "holdout_results": folds[0]["holdout_results"]}
    return {**result, "result_id": content_hash(result)}
