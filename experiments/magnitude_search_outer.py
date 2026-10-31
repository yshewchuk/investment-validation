"""Bounded selected-candidate outer fits; immutable evidence, no final authority."""
import sys
import time
from pathlib import Path

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import ArtifactError, canonical_json, content_hash, from_document
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.migrations import require_current
from engine.v2.research.magnitude_inputs import load_magnitude_targets
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_evidence import (
    admit_fold_inputs,
    fit_evidence,
    normalize_forecasts,
    validate_targets,
)
from experiments.magnitude_search_execution import (
    execution_limits,
    run_fitted_trial,
    validate_execution_limits,
)
from experiments.magnitude_search_inputs import load_inputs, publish_inputs
from experiments.magnitude_search_plan import PLAN_V2, PLAN_V3, read_search_plan, trial_key
from experiments.magnitude_search_records import MIGRATIONS, OWNER, append_record, read_record
from experiments.magnitude_search_trials import (
    SearchTrials,
    _completed_fit as _valid_completed_fit,
    _timing as _valid_timing,
)
from experiments.size_fit_diagnostics import evidence as diagnostic_evidence
from experiments.size_fit_diagnostics import keys as diagnostic_keys
from experiments.size_job_budget import resolved_limits
from experiments.size_search_binding import execution_for_candidate


def _conflict():
    return fail("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation evidence differs from its sealed inputs")


def _equal(left, right):
    if canonical_json(left) != canonical_json(right):
        raise _conflict()


def _timed_shape(result, keys, *, legacy=False):
    if not isinstance(result, dict):
        return False
    observed = set(result)
    if observed == keys:
        return legacy
    return observed == keys | {"score_seconds", "rss_delta_kib"} and _valid_timing(result)


def _equal_completed_fit(fit, expected, *, legacy=False):
    if not isinstance(fit, dict):
        raise _conflict()
    if not legacy or "fit_seconds" in fit:
        if not _valid_completed_fit(fit, expected):
            raise _conflict()
        return
    _equal(fit, expected)


def _native_selection(trials, plan, read):
    """Presence checks plus freeze verification make completed replays strict."""
    print("[stage] native_selection start", file=sys.stderr)
    started = time.monotonic()
    selection = read("selection")
    if selection is None:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation requires an existing frozen selection")
    seen = set()
    for outer in plan["folds"]:
        for candidate in plan["candidates"]:
            for inner in outer["inner"]:
                args = (outer["outer"]["year"], inner["year"], candidate["candidate_id"])
                key = trial_key(plan, *args)
                if key in seen:
                    continue
                seen.add(key)
                execution, terminal = read("execution:" + key), read("terminal:" + key)
                if not isinstance(execution, dict) or not isinstance(terminal, dict):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer evaluation requires completed native inner trials")
                try:
                    attempt = execution["attempt"]
                    limits = attempt["limits"]
                    expected = {"limits": limits, "candidate_id": candidate["candidate_id"], "trial_id": key}
                    execution_start = read("execution-start:" + key)
                    if (not isinstance(attempt, dict) or set(attempt) != {"limits", "candidate_id", "trial_id"}
                            or canonical_json(attempt) != canonical_json(expected)
                            or not isinstance(execution_start, dict)
                            or canonical_json(execution_start) != canonical_json(expected)):
                        raise ValueError
                    validate_execution_limits(limits)
                except (KeyError, TypeError, ValueError, OpsError):
                    raise _conflict() from None
                if plan.get("schema_version") in (PLAN_V2, PLAN_V3) and limits != plan["request"]["limits"]["child"]:
                    raise fail("INVALID_EXPERIMENT_SPEC", "native trial limits differ from sealed request")
    _equal(trials.freeze(), selection)
    print(f"[stage] native_selection done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return selection


def _inputs(repository, binding, fold):
    return admit_fold_inputs(repository, binding, fold, history_failure=_conflict(), strict_json=True)


def _forecasts(rows, fold, horizons):
    try:
        return normalize_forecasts(rows, fold, horizons, strict_json=True)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise _conflict() from None


def _validate_targets(inputs, validation, fold):
    try:
        if set(validation) != {"evaluation_inputs", "targets"}:
            raise ValueError
        validate_targets(inputs["preflight"], fold, validation["evaluation_inputs"], validation["targets"], strict_json=True)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise _conflict() from None


def _result(started, inputs, inputs_id, execution, forecasts, validation, fold, horizons):
    _validate_targets(inputs, validation, fold)
    targets = {row["event_id"]: row["absolute_move_pct"] for row in validation["targets"]}
    predictions = [{**row, "absolute_move_pct": targets[row["event_id"]]}
                   for row in forecasts if row["event_id"] in targets]
    result = {"schema_version": "magnitude_search_outer_result.v1", "outer_id": started["outer_id"],
        "plan_id": started["plan_id"], "outer_year": fold["year"], "candidate": started["candidate"],
        "selection_id": started["selection_id"], "inputs_id": inputs_id,
        "execution_id": content_hash(execution), "validation_id": content_hash(validation),
        "training_inputs": inputs["training_inputs"], "evaluation_inputs": validation["evaluation_inputs"],
        "fit_attempt_count": execution["fit_attempt_count"], "event_count": len(targets),
        "forecast_count": len(predictions), "predictions": predictions, "metrics": magnitude_metrics(predictions),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon])} for horizon in horizons],
        "prospective_candidate_id": started["prospective_candidate_id"],
        "evaluation_authority": "selected_outer_fold_only", "final_read_authority": "none",
        "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"},
        **({"fit_diagnostics": execution["diagnostics"]} if "diagnostics" in execution else {})}
    return {**result, "result_id": content_hash(result)}


def _replay(read, store, started, fold, plan, execution, binding):
    try:
        _equal(read("start"), started)
        record = read("inputs")
        inputs_id = read("terminal")["inputs_id"]
        try:
            inputs = load_inputs(store, record, inputs_id)
        except OpsError as error:
            if error.problem.retryable:
                raise
            raise _conflict() from None
        fits = fit_evidence(inputs, fold, plan, execution)
        forecasts = _forecasts(read("forecasts"), fold, plan["horizons"])
        receipt = {"status": "succeeded", "fits": fits, "fit_attempt_count": len(fits),
                   "completed_fit_count": len(fits), "forecasts_id": content_hash(forecasts)}
        timed_worker = False
        if execution.get("schema_version") == "size_candidate_execution.v1":
            worker = read("worker-result")
            if not _timed_shape(worker, {"event", "forecasts"} | diagnostic_keys(binding), legacy=True):
                raise _conflict()
            timed_worker = "score_seconds" in worker
            _equal(worker["forecasts"], forecasts)
            receipt.update(diagnostic_evidence(worker, execution, plan["horizons"]))
        _equal(read("execution"), receipt)
        for fit in fits:
            _equal(read("fit-started:" + str(fit["horizon_sessions"])), fit)
            _equal_completed_fit(read("fit-completed:" + str(fit["horizon_sessions"])), fit,
                                 legacy=not timed_worker)
        result = _result(started, inputs, inputs_id, receipt, forecasts, read("validation"), fold, plan["horizons"])
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
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    selection = _native_selection(trials, plan, read)
    return _execute_admitted(conn, store, repository, registration, plan, fold, selection, limits)


def _execute_admitted(conn, store, repository, registration, plan, fold, selection, limits, *, replay_only=False):
    """Bind each fold independently after this call's complete native admission."""
    binding, outer_year = registration.document, fold["year"]
    fold = {**fold, "evaluation_labels_before": binding["magnitude_execution"]["evaluation_labels_before"]}
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
        return read_record(conn, store, registration.run_id, "outer-" + slot + ":" + key)
    def append(slot, body):
        return append_record(conn, store, registration.run_id, "outer-" + slot + ":" + key, body, replay=False)
    if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7"):
        from experiments.size_comparison_run import execute as compare
        _equal(limits, plan["request"]["limits"]["child"])
        compare(conn, store, repository, registration, plan, fold, limits=limits, replay_only=replay_only)
    execution = execution_for_candidate(binding, candidate)
    if replay_only or saved("terminal") is not None:
        return _replay(saved, store, started, fold, plan, execution, binding)
    forbid_fitting("execute_outer_fold")
    forbid_legacy_fitting("execute_outer_fold")
    resolved = resolved_limits(conn, store, registration, plan, limits) if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7") else execution_limits(limits)
    append("start", started)
    inputs = _inputs(repository, binding, fold)
    manifest, inputs_id = publish_inputs(store, inputs)
    append("inputs", manifest)
    expected = fit_evidence(inputs, fold, plan, execution)
    fits, pending = [], []
    def progress(state, fit):
        index = len(fits)
        if (state not in ("started", "completed") or index >= len(expected)
                or (state == "started") == bool(pending)):
            raise _conflict()
        if state == "started":
            _equal(fit, expected[index])
        else:
            _equal_completed_fit(fit, expected[index])
        append("fit-" + state + ":" + str(fit["horizon_sessions"]), fit)
        if state == "started":
            pending.append(fit)
        else:
            fits.append(pending.pop())
    source = {"manifest": manifest, "inputs_id": inputs_id, "store_root": str(store.root)}
    if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7"):
        resolved = resolved_limits(conn, store, registration, plan, limits)
    result = run_fitted_trial(source, fold, plan, execution, resolved, progress)
    succeeded = result.get("event") == "succeeded"
    if ((succeeded and (pending or len(fits) != len(expected)
                        or not _timed_shape(result, {"event", "forecasts"} | diagnostic_keys(binding))))
            or (not succeeded and (not _timed_shape(result, {"event"}) or result.get("event") != "failed" or len(pending) != 1))):
        raise _conflict()
    forecasts = _forecasts(result["forecasts"], fold, plan["horizons"]) if succeeded else None
    receipt = {"status": result["event"], "fits": fits, "fit_attempt_count": len(fits) + len(pending),
               "completed_fit_count": len(fits), "forecasts_id": content_hash(forecasts) if succeeded else None,
               **diagnostic_evidence(result, execution, plan["horizons"])}
    if diagnostic_keys(binding):
        append("worker-result", result)
    append("execution", receipt)
    if not succeeded:
        raise fail("EXPERIMENT_VARIANT_FAILED", "selected outer candidate failed; replacement and retry are forbidden")
    append("forecasts", forecasts)
    # All predictions and successful fit accounting are durable before any outer target opens.
    labels = load_magnitude_targets(repository, from_document(SnapshotRef, binding["snapshot"]),
        as_of_month=binding["holdouts"]["holdout_as_of_month"], labels_before=fold["evaluation_labels_before"],
        event_ids=fold["evaluation_event_ids"], omit_not_computed=True)
    validation = {"evaluation_inputs": dict(labels.attrs),
                  "targets": labels[["event_id", "absolute_move_pct"]].to_dict("records")}
    _validate_targets(inputs, validation, fold)
    append("validation", validation)
    append("terminal", _result(started, inputs, inputs_id, receipt, forecasts, validation, fold, plan["horizons"]))
    return _replay(saved, store, started, fold, plan, execution, binding)


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
        pending.append((item["outer"], started["limits"]))
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    selection = _native_selection(trials, plan, read)
    folds = [_execute_admitted(conn, store, repository, registration, plan, fold, selection, limits, replay_only=True)
             for fold, limits in pending]
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
