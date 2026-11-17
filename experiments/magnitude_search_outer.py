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
    _requires_event_dates,
    admit_fold_inputs,
    evaluation_targets,
    fit_evidence,
    forecast_target,
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
from experiments.magnitude_search_blend_outer import blend_outer_result
from experiments.magnitude_search_refits import (
    blend_components,
    failed_terminal,
    per_candidate,
    refit_key,
    refit_units,
    required_units,
)
from experiments.magnitude_search_trials import (
    SearchTrials,
)
from experiments.magnitude_search_trials import (
    _completed_fit as _valid_completed_fit,
)
from experiments.magnitude_search_trials import (
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


def _failed(outcome):
    return isinstance(outcome, dict) and outcome.get("status") == "failed"


def _blend_record(plan, candidate_id):
    return next(blend for blend in plan["blends"] if blend["blend_id"] == candidate_id)


def _timed_shape(result, keys, *, legacy=False, untimed=False):
    if not isinstance(result, dict):
        return False
    observed = set(result)
    if untimed:
        return observed == keys
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
                if plan.get("schema_version") in (PLAN_V2, PLAN_V3, "regression_search_plan.v1") and limits != plan["request"]["limits"]["child"]:
                    raise fail("INVALID_EXPERIMENT_SPEC", "native trial limits differ from sealed request")
    _equal(trials.freeze(), selection)
    print(f"[stage] native_selection done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return selection


def _inputs(repository, binding, fold, *, plan=None):
    return admit_fold_inputs(repository, binding, fold, history_failure=_conflict(), strict_json=True, plan=plan)


def _forecasts(rows, fold, horizons, target=None):
    try:
        return normalize_forecasts(rows, fold, horizons, strict_json=True, target=target)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise _conflict() from None


def _validate_targets(inputs, validation, fold):
    try:
        if set(validation) != {"evaluation_inputs", "targets"}:
            raise ValueError
        validate_targets(inputs["preflight"], fold, validation["evaluation_inputs"], validation["targets"], strict_json=True)
    except (ValueError, TypeError, KeyError, OverflowError):
        raise _conflict() from None


def _result(started, inputs, inputs_id, execution, forecasts, validation, fold, horizons, target=None):
    _validate_targets(inputs, validation, fold)
    column = (target or {}).get("column", "absolute_move_pct")
    targets = {row["event_id"]: row[column] for row in validation["targets"]}
    predictions = [{**row, column: targets[row["event_id"]]}
                   for row in forecasts if row["event_id"] in targets]
    result = {"schema_version": "magnitude_search_outer_result.v1", "outer_id": started["outer_id"],
        "plan_id": started["plan_id"], "outer_year": fold["year"], "candidate": started["candidate"],
        "selection_id": started["selection_id"], "inputs_id": inputs_id,
        "execution_id": content_hash(execution), "validation_id": content_hash(validation),
        "evaluation_inputs": validation["evaluation_inputs"],
        "fit_attempt_count": execution["fit_attempt_count"], "event_count": len(targets),
        "forecast_count": len(predictions), "predictions": predictions, "metrics": magnitude_metrics(predictions, target=target),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon], target=target)} for horizon in horizons],
        "prospective_candidate_id": started["prospective_candidate_id"],
        "evaluation_authority": "selected_outer_fold_only", "final_read_authority": "none",
        "holdout_results": {"random": "excluded; not evaluated", "rolling": "excluded; not evaluated"},
        **({"fit_diagnostics": [{k: v for k, v in item.items() if k != "training_event_ids"}
                                for item in execution["diagnostics"]]} if "diagnostics" in execution else {})}
    if target is not None:
        result.update(schema_version="regression_search_outer_result.v1", target=target)
    return {**result, "result_id": content_hash(result)}


def _replay(read, store, started, fold, plan, execution, binding, *, input_callback=None):
    regression = execution.get("schema_version") == "regression_candidate_execution.v1"
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
        forecasts = _forecasts(read("forecasts"), fold, plan["horizons"], forecast_target(plan, execution.get("fit_recipe", {})))
        receipt = {"status": "succeeded", "fits": fits, "fit_attempt_count": len(fits),
                   "completed_fit_count": len(fits), "forecasts_id": content_hash(forecasts)}
        worker = read("worker-result")
        if worker is None and execution.get("schema_version") in ("size_candidate_execution.v1", "regression_candidate_execution.v1"):
            raise _conflict()
        timed_worker = None
        if worker is not None:
            if not _timed_shape(worker, {"event", "forecasts"} | diagnostic_keys(binding),
                                legacy=True, untimed=regression):
                raise _conflict()
            if worker.get("event") != "succeeded":
                raise _conflict()
            timed_worker = "score_seconds" in worker
            _equal(worker["forecasts"], forecasts)
            if diagnostic_keys(binding):
                receipt.update(diagnostic_evidence(worker, execution, plan["horizons"], inputs, fold))
        _equal(read("execution"), receipt)
        for fit in fits:
            _equal(read("fit-started:" + str(fit["horizon_sessions"])), fit)
            marker = read("fit-completed:" + str(fit["horizon_sessions"]))
            marker_timed = isinstance(marker, dict) and "fit_seconds" in marker
            if timed_worker is None:
                if marker_timed:
                    raise _conflict()
            elif marker_timed != timed_worker:
                raise _conflict()
            _equal_completed_fit(marker, fit, legacy=not bool(timed_worker))
        result = _result(started, inputs, inputs_id, receipt, forecasts, read("validation"), fold, plan["horizons"], plan.get("target"))
        _equal(read("terminal"), result)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise _conflict() from None
    if input_callback is not None:
        input_callback(fold, inputs)
    return result


def _replay_failed(saved, store, started, key, outer_year, chosen, fold, plan, execution):
    try:
        _equal(saved("start"), started)
        terminal = saved("terminal")
        if not isinstance(terminal, dict) or "inputs_id" not in terminal:
            raise _conflict()
        _equal(terminal, failed_terminal(key, outer_year, chosen, terminal["inputs_id"]))
        inputs_id = terminal["inputs_id"]
        try:
            inputs = load_inputs(store, saved("inputs"), inputs_id)
        except OpsError as error:
            if error.problem.retryable:
                raise
            raise _conflict() from None
        expected = fit_evidence(inputs, fold, plan, execution)
        worker = saved("worker-result")
        receipt = saved("execution")
        regression = execution.get("schema_version") == "regression_candidate_execution.v1"
        required = {"status", "fits", "fit_attempt_count", "completed_fit_count", "forecasts_id"}
        if (not isinstance(worker, dict) or worker.get("event") != "failed"
                or not _timed_shape(worker, {"event"}, untimed=regression)
                or not isinstance(receipt, dict) or not set(receipt) >= required):
            raise _conflict()
        fits = receipt["fits"]
        if not isinstance(fits, list):
            raise _conflict()
        k = len(fits)
        if receipt["status"] != "failed" or receipt["forecasts_id"] is not None:
            raise _conflict()
        _equal(fits, expected[:k])
        if receipt["completed_fit_count"] != k:
            raise _conflict()
        if (type(receipt["fit_attempt_count"]) is not int
                or type(receipt["completed_fit_count"]) is not int):
            raise _conflict()
        attempts = receipt["fit_attempt_count"]
        if attempts not in (k, k + 1) or (attempts == k + 1 and k + 1 > len(expected)):
            raise _conflict()
        for fit in fits:
            slot = ":" + str(fit["horizon_sessions"])
            _equal(saved("fit-started" + slot), fit)
            _equal_completed_fit(saved("fit-completed" + slot), fit, legacy=False)
        if attempts == k + 1:
            _equal(saved("fit-started:" + str(expected[k]["horizon_sessions"])), expected[k])
        for index, fit in enumerate(expected):
            slot = ":" + str(fit["horizon_sessions"])
            if index < k:
                continue
            if index == k and attempts == k + 1:
                if saved("fit-completed" + slot) is not None:
                    raise _conflict()
                continue
            if (saved("fit-started" + slot) is not None
                    or saved("fit-completed" + slot) is not None):
                raise _conflict()
        rebuilt = {"status": "failed", "fits": receipt["fits"], "fit_attempt_count": receipt["fit_attempt_count"],
                   "completed_fit_count": len(receipt["fits"]), "forecasts_id": None,
                   **diagnostic_evidence(worker, execution, plan["horizons"], inputs, fold)}
        _equal(receipt, rebuilt)
    except (KeyError, TypeError, ValueError):
        raise _conflict() from None
    return terminal


def execute_outer_fold(conn, store, repository, registration, request, plan_ref, outer_year, *, limits,
                       candidate_id=None):
    """Refit one frozen unit (the overall winner, or a winner_rules refit unit by candidate_id) once; replay complete evidence without fitting."""
    validate_execution_limits(limits)
    require_current(conn, OWNER, MIGRATIONS)
    plan = read_search_plan(conn, store, registration, request, plan_ref)
    try:
        fold = next(item["outer"] for item in plan["folds"] if item["outer"]["year"] == outer_year)
    except StopIteration:
        raise fail("INVALID_EXPERIMENT_SPEC", "outer fold is outside the sealed search plan") from None
    if candidate_id is None:
        if per_candidate(plan):
            raise fail("INVALID_EXPERIMENT_SPEC", "refit unit is outside the sealed selection")
    elif not per_candidate(plan):
        raise fail("INVALID_EXPERIMENT_SPEC", "refit unit is outside the sealed selection")
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    selection = _native_selection(trials, plan, read)
    if candidate_id is not None:
        item = next((entry for entry in selection["selections"]
                     if entry["outer_year"] == outer_year), None)
        if item is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "refit unit is outside the sealed selection")
        try:
            units = refit_units(plan, item)
        except (ValueError, TypeError, KeyError):
            raise _conflict() from None
        if candidate_id not in units:
            raise fail("INVALID_EXPERIMENT_SPEC", "refit unit is outside the sealed selection")
    return _execute_admitted(conn, store, repository, registration, plan, fold, selection, limits,
                             candidate_id=candidate_id)


def _execute_admitted(conn, store, repository, registration, plan, fold, selection, limits, *,
                      replay_only=False, input_callback=None, candidate_id=None):
    """Bind each fold independently after this call's complete native admission."""
    binding, outer_year = registration.document, fold["year"]
    execution_key = "regression_execution" if "target" in plan else "magnitude_execution"
    fold = {**fold, "evaluation_labels_before": binding[execution_key]["evaluation_labels_before"]}
    item = next(entry for entry in selection["selections"] if entry["outer_year"] == outer_year)
    overall = item["selected_candidate_id"]
    required = required_units(plan, item) if per_candidate(plan) else [overall]
    chosen = candidate_id or overall
    candidate = next((item for item in plan["candidates"] if item["candidate_id"] == chosen), None)
    if candidate is None:
        raise fail("EXPERIMENT_VARIANT_FAILED", "selected winner is not a plan candidate; blend winner outer evaluation is not supported yet")
    try:
        sources = source_closure(Path(__file__).resolve().parents[1], [
            "experiments/magnitude_search_outer.py", "experiments/magnitude_search_worker.py"])
    except (OpsError, ArtifactError, OSError, SyntaxError, ValueError):
        raise _conflict() from None
    key = refit_key(plan, outer_year, candidate_id)
    started = {"outer_id": key, "plan_id": plan["plan_id"], "outer_year": outer_year,
        "selection_id": content_hash(selection), "candidate": candidate, "limits": dict(limits),
        "outer_sources": sources, "prospective_candidate_id": selection["prospective_candidate_id"]}
    def saved(slot):
        return read_record(conn, store, registration.run_id, "outer-" + slot + ":" + key)
    def append(slot, body):
        return append_record(conn, store, registration.run_id, "outer-" + slot + ":" + key, body, replay=False)
    if "target" in plan and saved("start") is not None and saved("terminal") is None:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed")
    if binding.get("schema_version") in ("native_experiment_registration.v1.6", "native_experiment_registration.v1.7", "native_experiment_registration.v1.8"):
        from experiments.size_comparison_run import execute as compare
        _equal(limits, plan["request"]["limits"]["child"])
        compare(conn, store, repository, registration, plan, fold, limits=limits, replay_only=replay_only)
    execution = execution_for_candidate(binding, candidate)
    regression = execution.get("schema_version") == "regression_candidate_execution.v1"
    if replay_only or saved("terminal") is not None:
        # A failed non-overall slot replays only its exact durable terminal; every other slot
        # (including a committed overall winner, which never commits a failed terminal) uses
        # the ordinary evidence replay.
        if per_candidate(plan):
            terminal = saved("terminal")
            if isinstance(terminal, dict) and terminal.get("status") == "failed":
                if chosen in required:
                    raise _conflict()
                return _replay_failed(saved, store, started, key, outer_year, chosen, fold, plan, execution)
        return _replay(saved, store, started, fold, plan, execution, binding, input_callback=input_callback)
    forbid_fitting("execute_outer_fold")
    forbid_legacy_fitting("execute_outer_fold")
    resolved = resolved_limits(conn, store, registration, plan, limits) if diagnostic_keys(binding) else execution_limits(limits)
    use_plan = "target" in plan or _requires_event_dates(plan)
    inputs = _inputs(repository, binding, fold, plan=plan) if use_plan and saved("start") is None else None
    append("start", started)
    if inputs is None:
        inputs = _inputs(repository, binding, fold, plan=plan) if use_plan else _inputs(repository, binding, fold)
    manifest, inputs_id = publish_inputs(store, inputs)
    append("inputs", manifest)
    expected = fit_evidence(inputs, fold, plan, execution)
    fits, pending = [], []
    def progress(state, fit):
        index = len(fits)
        if (state not in ("started", "completed") or index >= len(expected)
                or (state == "started") == bool(pending)):
            raise _conflict()
        if state == "started" or regression:
            _equal(fit, expected[index])
        else:
            _equal_completed_fit(fit, expected[index])
        append("fit-" + state + ":" + str(fit["horizon_sessions"]), fit)
        if state == "started":
            pending.append(fit)
        else:
            fits.append(pending.pop())
    source = {"manifest": manifest, "inputs_id": inputs_id, "store_root": str(store.root)}
    if diagnostic_keys(binding):
        resolved = resolved_limits(conn, store, registration, plan, limits)
    result = run_fitted_trial(source, fold, plan, execution, resolved, progress)
    succeeded = result.get("event") == "succeeded"
    # On winner_rules plans the overall winner may have failed before any horizon start, refusing with EXPERIMENT_VARIANT_FAILED; legacy plans are unchanged.
    failed_pending_ok = len(pending) in (0, 1) if per_candidate(plan) else len(pending) == 1
    if ((succeeded and (pending or len(fits) != len(expected)
                        or not _timed_shape(result, {"event", "forecasts"} | diagnostic_keys(binding), untimed=regression)))
            or (not succeeded and (not _timed_shape(result, {"event"}, untimed=regression)
                                   or result.get("event") != "failed" or not failed_pending_ok))):
        raise _conflict()
    forecasts = _forecasts(result["forecasts"], fold, plan["horizons"], forecast_target(plan, execution.get("fit_recipe", {}))) if succeeded else None
    if succeeded:
        _equal(result["forecasts"], forecasts)
    receipt = {"status": result["event"], "fits": fits, "fit_attempt_count": len(fits) + len(pending),
               "completed_fit_count": len(fits), "forecasts_id": content_hash(forecasts) if succeeded else None,
               **diagnostic_evidence(result, execution, plan["horizons"], inputs, fold)}
    append("worker-result", result)
    append("execution", receipt)
    if not succeeded:
        if chosen in required:
            raise fail("EXPERIMENT_VARIANT_FAILED", "selected outer candidate failed; replacement and retry are forbidden")
        # A failed non-overall refit is terminal: targets never open and resume never retries it.
        terminal = failed_terminal(key, outer_year, chosen, inputs_id)
        append("terminal", terminal)
        return terminal
    append("forecasts", forecasts)
    # All predictions and successful fit accounting are durable before any outer target opens.
    labels = evaluation_targets(repository, binding, fold) if "target" in plan else load_magnitude_targets(repository, from_document(SnapshotRef, binding["snapshot"]),
        as_of_month=binding["holdouts"]["holdout_as_of_month"], labels_before=fold["evaluation_labels_before"],
        event_ids=fold["evaluation_event_ids"], omit_not_computed=True)
    validation = {"evaluation_inputs": dict(labels.attrs),
                  "targets": labels[["event_id", plan.get("target", {}).get("column", "absolute_move_pct")]].to_dict("records")}
    _validate_targets(inputs, validation, fold)
    append("validation", validation)
    append("terminal", _result(started, inputs, inputs_id, receipt, forecasts, validation, fold, plan["horizons"], plan.get("target")))
    return _replay(saved, store, started, fold, plan, execution, binding, input_callback=input_callback)


def collect_outer_results(conn, store, repository, registration, request, plan_ref, *, input_callback=None):
    """Read every completed outer fold; pool event errors, never rank by outer loss.

    ``input_callback`` optionally receives each sealed outer fold and its verified
    in-memory inputs; callers must not retain those rows.
    """
    require_current(conn, OWNER, MIGRATIONS)
    plan = read_search_plan(conn, store, registration, request, plan_ref)
    refit_plan = per_candidate(plan)
    overall_by_year, required_by_year = {}, {}
    pending = []
    if refit_plan:
        stored = read_record(conn, store, registration.run_id, "selection")
        if not isinstance(stored, dict):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold")
        try:
            by_year = {item["outer_year"]: item for item in stored["selections"]}
        except (KeyError, TypeError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold") from None
        for item in plan["folds"]:
            year = item["outer"]["year"]
            try:
                selection_item = by_year[year]
                overall = selection_item["selected_candidate_id"]
            except (KeyError, TypeError):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold") from None
            overall_by_year[year] = overall
            try:
                units = refit_units(plan, selection_item)
                required_by_year[year] = required_units(plan, selection_item)
            except (ValueError, TypeError, KeyError):
                raise _conflict() from None
            for unit in units:
                key = refit_key(plan, year, unit)
                started = read_record(conn, store, registration.run_id, "outer-start:" + key)
                terminal = read_record(conn, store, registration.run_id, "outer-terminal:" + key)
                if not isinstance(started, dict) or not isinstance(terminal, dict):
                    raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold")
                try:
                    validate_execution_limits(started["limits"])
                except (KeyError, OpsError):
                    raise _conflict() from None
                pending.append((item["outer"], started["limits"], unit))
    else:
        for item in plan["folds"]:
            year = item["outer"]["year"]
            key = refit_key(plan, year)
            started = read_record(conn, store, registration.run_id, "outer-start:" + key)
            terminal = read_record(conn, store, registration.run_id, "outer-terminal:" + key)
            if not isinstance(started, dict) or not isinstance(terminal, dict):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "outer report requires every completed outer fold")
            try:
                validate_execution_limits(started["limits"])
            except (KeyError, OpsError):
                raise _conflict() from None
            pending.append((item["outer"], started["limits"], None))
    def read(slot):
        return read_record(conn, store, registration.run_id, slot)
    trials = SearchTrials(conn, store, repository, registration, request, plan_ref)
    selection = _native_selection(trials, plan, read)
    refits = []
    if refit_plan:
        outcomes, folds, blend_results = {}, [], {}
        for fold, limits, unit in pending:
            year = fold["year"]
            outcome = _execute_admitted(conn, store, repository, registration, plan, fold, selection, limits,
                                        replay_only=True, candidate_id=unit,
                                        input_callback=input_callback if unit == required_by_year[year][0] else None)
            outcomes[(year, unit)] = outcome
            failed = _failed(outcome)
            refits.append({"outer_year": year, "candidate_id": unit,
                           "status": "failed" if failed else "completed", "result": outcome})
        for item in plan["folds"]:
            year = item["outer"]["year"]
            overall = overall_by_year[year]
            # Required units completed (a required failure already raised), so blend results are complete.
            winners = list(dict.fromkeys([overall, *sorted(by_year[year].get("distinct_winner_ids", []))]))
            for candidate_id in winners:
                components = blend_components(plan, candidate_id)
                if components is None:
                    continue
                results = [outcomes[(year, component)] for component in components]
                failed = any(_failed(result) for result in results)
                blend_results[(year, candidate_id)] = None if failed else blend_outer_result(
                    plan, _blend_record(plan, candidate_id), year, results)
                refits.append({"outer_year": year, "candidate_id": candidate_id, "kind": "blend",
                               "component_candidate_ids": components,
                               "status": "failed" if failed else "completed",
                               "result": blend_results[(year, candidate_id)]})
            folds.append(blend_results[(year, overall)] if blend_components(plan, overall) is not None
                         else outcomes[(year, overall)])
    else:
        folds = [_execute_admitted(conn, store, repository, registration, plan, fold, selection, limits,
                                   replay_only=True, input_callback=input_callback)
                 for fold, limits, _ in pending]
    predictions = [{**row, "outer_year": fold["outer_year"]} for fold in folds for row in fold["predictions"]]
    result = {"schema_version": "magnitude_search_outer_report.v1", "plan_id": plan["plan_id"],
        "folds": folds, "predictions": predictions, "metrics": magnitude_metrics(predictions, target=plan.get("target")),
        "event_count": sum(fold["event_count"] for fold in folds), "forecast_count": len(predictions),
        "per_horizon": [{"horizon_sessions": horizon, **magnitude_metrics([
            row for row in predictions if row["horizon_sessions"] == horizon], target=plan.get("target"))} for horizon in plan["horizons"]],
        "prospective_candidate_id": folds[-1]["prospective_candidate_id"],
        "winner_policy": "latest_outer_inner_selection_before_its_evaluation",
        "final_read_authority": "none", "holdout_results": folds[0]["holdout_results"]}
    if refit_plan:
        result["refits"] = refits
    if "target" in plan:
        result.update(schema_version="regression_search_outer_report.v1", target=plan["target"])
    return {**result, "result_id": content_hash(result)}
