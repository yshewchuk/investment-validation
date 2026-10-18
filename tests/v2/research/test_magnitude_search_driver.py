"""Tiny real journals prove serial orchestration, recovery, and selected outer replay."""
# ruff: noqa: F811
import copy
import sqlite3

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import SystemClock, content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_outer as outer
from experiments import magnitude_search_trials as trials
from experiments import v2_magnitude_search_run as driver
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_execution_integrity import native  # noqa: F401
from tests.v2.research.test_magnitude_search_reconciliation import (
    ARGS,
    KEY,
    failed_child,
    interrupt,
)
from tests.v2.research.test_magnitude_search_trial_integrity import (
    durable_evidence,
    journal,  # noqa: F401
    native_labels,
    rows,
)


@pytest.fixture
def ready(native, monkeypatch, request):
    r = native.recorder
    r.registration.document["magnitude_execution"]["evaluation_labels_before"] = "2024-01-01"
    native.plan["folds"][0]["outer"]["evaluation_event_ids"] = ["event"]
    if getattr(request, "param", 1) == 2:
        second = copy.deepcopy(native.plan["folds"][0])
        second["outer"]["year"] = "2025"
        native.plan["folds"].append(second)
    inputs = r._inputs(None)
    inputs["training_inputs"] = {"synthetic": "prior training only"}
    monkeypatch.setattr(driver, "read_search_plan", lambda *a: native.plan)
    monkeypatch.setattr(driver, "SearchTrials", lambda *a: r)
    monkeypatch.setattr(outer, "SearchTrials", lambda *a: r)
    monkeypatch.setattr(outer, "read_search_plan", lambda *a: native.plan)
    monkeypatch.setattr(outer, "_inputs", lambda *a: inputs)
    monkeypatch.setattr(outer, "load_magnitude_targets", lambda *a, **k: native_labels())
    monkeypatch.setattr(outer, "run_fitted_trial", trials.run_fitted_trial)
    native.context = (r.conn, r.store, r.repository, r.registration, r.request, r.plan_ref)
    driver.seal_search_run(*native.context, clock=SystemClock())
    return native


def run(case):
    return driver.run_search(*case.context, limits=LIMITS)


def forbidden(*args, **kwargs):
    pytest.fail("unexpected new scientific work")


def test_real_inner_outer_report_and_no_fit_replay(ready, monkeypatch):
    calls = []
    original = ready.recorder.execute
    def execute(*args, **kwargs):
        if ready.recorder._read("execution:" + content_hash(["plan", *args])) is None:
            calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(ready.recorder, "execute", execute)
    original_outer = outer.run_fitted_trial
    def fitted(*args):
        assert ready.recorder._read("selection") is not None
        calls.append("outer")
        return original_outer(*args)
    monkeypatch.setattr(outer, "run_fitted_trial", fitted)
    result = run(ready)
    assert calls == [("2024", "2023", "z-last"), ARGS, "outer"]
    assert result["schema_version"] == "magnitude_search_outer_report.v1"
    assert result["prospective_candidate_id"] == "a-first"
    assert result["metrics"] == {"count": 3, "rmse": 2.0, "mae": 2.0}
    assert result["predictions"] == [{**row, "absolute_move_pct": 4.0, "outer_year": "2024"} for row in rows(2.0)]
    assert result["final_read_authority"] == "none"
    assert result["result_id"] == content_hash({key: value for key, value in result.items() if key != "result_id"})
    before = durable_evidence(ready)
    for module in (trials, outer):
        monkeypatch.setattr(module, "run_fitted_trial", forbidden)
        monkeypatch.setattr(module, "load_magnitude_targets", forbidden)
    monkeypatch.setattr(ready.recorder, "_inputs", forbidden)
    monkeypatch.setattr(outer, "_inputs", forbidden)
    with no_fit_guard(), legacy_guard():
        assert run(ready) == result
        assert driver.collect_outer_results(*ready.context) == result
        assert driver.seal_search_run(*ready.context, clock=SystemClock())["sealed"] == "plan"
    assert durable_evidence(ready) == before


@pytest.mark.parametrize("stage", ["execution-result", "execution", "forecasts", "validation", "terminal"])
def test_recovery_precedes_every_fresh_trial(ready, monkeypatch, stage):
    interrupt(monkeypatch, ready.recorder, stage, after=True)
    actions = []
    reconcile, execute = ready.recorder.reconcile, ready.recorder.execute
    def repair(*args, **kwargs):
        actions.append(("reconcile", args))
        return reconcile(*args, **kwargs)
    def fresh(*args, **kwargs):
        if ready.recorder._read("execution:" + content_hash(["plan", *args])) is None:
            assert ready.recorder._read("terminal:" + KEY) is not None
            actions.append(("execute", args))
        return execute(*args, **kwargs)
    monkeypatch.setattr(ready.recorder, "reconcile", repair)
    monkeypatch.setattr(ready.recorder, "execute", fresh)
    result = run(ready)
    assert actions[:2] == [("reconcile", ARGS), ("execute", ("2024", "2023", "z-last"))]
    assert result["prospective_candidate_id"] == "a-first"


@pytest.mark.parametrize("slot", ["start", "execution-result", "execution", "terminal", "outer-start", "outer-terminal"])
def test_full_presence_preflight_prevents_fitting(ready, monkeypatch, slot):
    key = content_hash(["plan", "2024"]) if slot.startswith("outer-") else KEY
    ready.recorder._append(slot + ":" + key, {"synthetic": slot})
    before = durable_evidence(ready)
    monkeypatch.setattr(ready.recorder, "execute", forbidden)
    monkeypatch.setattr(ready.recorder, "reconcile", forbidden)
    with pytest.raises(OpsError) as caught:
        run(ready)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT",
        "search contains an unfinished or orphan " + ("outer" if slot.startswith("outer-") else "inner") + " claim").problem
    assert durable_evidence(ready) == before


@pytest.mark.parametrize("failure", ["corrupt", "resource", "sqlite"])
def test_later_claim_failure_stops_before_earlier_missing_fit(ready, monkeypatch, failure):
    append = ready.recorder._append
    def corrupt(slot, body, **kwargs):
        if slot == "execution-result:" + KEY and failure == "corrupt":
            body = copy.deepcopy(body)
            body["attempt"]["candidate_id"] = "wrong"
        return append(slot, body, **kwargs)
    monkeypatch.setattr(ready.recorder, "_append", corrupt)
    interrupt(monkeypatch, ready.recorder, "execution-result", after=True)
    monkeypatch.setattr(ready.recorder, "_append", append)
    before = durable_evidence(ready)
    monkeypatch.setattr(ready.recorder, "execute", forbidden)
    if failure != "corrupt":
        error = sqlite3.OperationalError("synthetic shared failure") if failure == "sqlite" else fail("RESOURCE_UNAVAILABLE", "synthetic shared failure")
        def refuse(*a, **k):
            raise error
        monkeypatch.setattr(ready.recorder, "reconcile", refuse)
    with pytest.raises(sqlite3.Error if failure == "sqlite" else OpsError) as caught:
        run(ready)
    if failure != "corrupt":
        assert caught.value is error
    else:
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search durable execution result differs").problem
    assert durable_evidence(ready) == before


def test_valid_failed_candidate_is_excluded_without_retry(ready, monkeypatch):
    original = trials.run_fitted_trial
    failed_child(monkeypatch, completed=1)
    ready.recorder.execute(*ARGS, limits=LIMITS)
    monkeypatch.setattr(trials, "run_fitted_trial", original)
    result = run(ready)
    assert result["prospective_candidate_id"] == "z-last"
    assert ready.recorder._read("terminal:" + KEY)["status"] == "failed"
    before = durable_evidence(ready)
    monkeypatch.setattr(trials, "run_fitted_trial", forbidden)
    monkeypatch.setattr(outer, "run_fitted_trial", forbidden)
    assert run(ready) == result
    assert durable_evidence(ready) == before


@pytest.mark.parametrize("recover", [False, True])
def test_expected_scoring_refusal_continues_only_after_verified_terminal(ready, monkeypatch, recover):
    if recover:
        interrupt(monkeypatch, ready.recorder, "execution-result", after=True)
    original = trials.magnitude_metrics
    remaining = 1
    def metrics(predictions):
        nonlocal remaining
        if remaining:
            remaining -= 1
            raise fail("EXPERIMENT_VARIANT_FAILED", "synthetic score refusal")
        return original(predictions)
    monkeypatch.setattr(trials, "magnitude_metrics", metrics)
    result = run(ready)
    refused = ARGS if recover else ("2024", "2023", "z-last")
    key = content_hash(["plan", *refused])
    assert ready.recorder._read("terminal:" + key)["status"] == "refused"
    assert result["prospective_candidate_id"] == ("z-last" if recover else "a-first")


@pytest.mark.parametrize("publication", ["absent", "cause", "wrong_problem"])
def test_candidate_named_error_is_not_enough_to_continue(ready, monkeypatch, publication):
    error = fail("EXPERIMENT_VARIANT_FAILED", "candidate named error")
    cause = sqlite3.OperationalError("publication unavailable")
    def execute(*args, **kwargs):
        key = content_hash(["plan", *args])
        if publication != "absent":
            ready.recorder._append("terminal:" + key, {"status": "refused", "problem": {"code": "wrong"}})
        if publication == "cause":
            raise error from cause
        raise error
    monkeypatch.setattr(ready.recorder, "execute", execute)
    if publication == "wrong_problem":
        def reject(*args, **kwargs):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "invalid durable refusal")
        monkeypatch.setattr(ready.recorder, "reconcile", reject)
    else:
        monkeypatch.setattr(ready.recorder, "reconcile", forbidden)
    monkeypatch.setattr(ready.recorder, "freeze", forbidden)
    with pytest.raises(OpsError) as caught:
        run(ready)
    if publication == "wrong_problem":
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "invalid durable refusal").problem
    else:
        assert caught.value is error


@pytest.mark.parametrize("limits", [None, {}, {**LIMITS, "cpu_seconds": True}, {**LIMITS, "wall_seconds": 0}])
def test_limits_admitted_before_any_journal_work(ready, monkeypatch, limits):
    monkeypatch.setattr(driver, "_admit_driver", forbidden)
    with pytest.raises(OpsError) as caught:
        driver.run_search(*ready.context, limits=limits)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search execution requires explicit positive integer resource limits").problem


def test_completed_and_pending_outer_claims_never_retry(ready, monkeypatch):
    r = ready.recorder
    for candidate in ("z-last", "a-first"):
        r.execute("2024", "2023", candidate, limits=LIMITS)
    r.freeze()
    key = content_hash(["plan", "2024"])
    r._append("outer-start:" + key, {"unfinished": True})
    before = durable_evidence(ready)
    monkeypatch.setattr(driver, "execute_outer_fold", forbidden)
    with pytest.raises(OpsError) as caught:
        run(ready)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT",
        "search contains an unfinished or orphan outer claim").problem
    assert durable_evidence(ready) == before


@pytest.mark.parametrize("ready", [2], indirect=True)
@pytest.mark.parametrize("damage", [None, "limits", "terminal"])
def test_later_completed_outer_is_validated_before_earlier_fresh_fit(ready, monkeypatch, damage):
    r = ready.recorder
    for args in driver._roster(ready.plan):
        r.execute(*args, limits=LIMITS)
    r.freeze()
    completed = content_hash(["plan", "2025"])
    if damage == "terminal":
        append = outer.append_record
        original_terminal = []
        def corrupt(conn, store, run_id, slot, body, **kwargs):
            if slot == "outer-terminal:" + completed:
                original_terminal.append(body)
                body = {**body, "result_id": "wrong"}
            return append(conn, store, run_id, slot, body, **kwargs)
        monkeypatch.setattr(outer, "append_record", corrupt)
        with pytest.raises(OpsError) as caught:
            outer.execute_outer_fold(*ready.context, "2025", limits=LIMITS)
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT",
            "outer evaluation evidence differs from its sealed inputs").problem
        monkeypatch.setattr(outer, "append_record", append)
        assert len(original_terminal) == 1
        original = original_terminal[0]
        assert original["result_id"] == content_hash({key: value for key, value in original.items() if key != "result_id"})
        assert original["outer_year"] == "2025" and original["forecast_count"] == 3
        assert r._read("outer-terminal:" + completed) == {**original, "result_id": "wrong"}
    else:
        limits = {**LIMITS, "cpu_seconds": 31} if damage else LIMITS
        original = outer.execute_outer_fold(*ready.context, "2025", limits=limits)
        assert r._read("outer-terminal:" + completed) == original
    assert r._read("outer-execution:" + completed)["status"] == "succeeded"
    assert r._read("outer-start:" + completed) is not None
    if damage is None:
        fitted, calls = outer.run_fitted_trial, []
        def fresh(*args):
            calls.append(args[1]["year"])
            return fitted(*args)
        monkeypatch.setattr(trials, "run_fitted_trial", forbidden)
        monkeypatch.setattr(outer, "run_fitted_trial", fresh)
        result = run(ready)
        assert calls == ["2024"]
        assert result["folds"][1] == original
        return
    before = durable_evidence(ready)
    monkeypatch.setattr(trials, "run_fitted_trial", forbidden)
    monkeypatch.setattr(outer, "run_fitted_trial", forbidden)
    with pytest.raises(OpsError) as caught:
        run(ready)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT",
        "outer evaluation evidence differs from its sealed inputs").problem
    assert r._read("outer-start:" + content_hash(["plan", "2024"])) is None
    assert durable_evidence(ready) == before


@pytest.mark.parametrize("damage", ["status", "problem"])
def test_valid_hash_bad_refusal_is_rejected_by_public_reconcile(ready, monkeypatch, damage):
    r = ready.recorder
    append = r._append
    def corrupt(slot, body, **kwargs):
        if slot.startswith("terminal:"):
            body = copy.deepcopy(body)
            if damage == "status":
                body["status"] = "unexpected"
            else:
                body["problem"]["code"] = "RESOURCE_UNAVAILABLE"
        return append(slot, body, **kwargs)
    def refuse(*args):
        raise fail("EXPERIMENT_VARIANT_FAILED", "synthetic scoring refusal")
    monkeypatch.setattr(r, "_append", corrupt)
    monkeypatch.setattr(trials, "magnitude_metrics", refuse)
    with pytest.raises(OpsError) as caught:
        run(ready)
    code, message = (("EXPERIMENT_VARIANT_FAILED", "synthetic scoring refusal") if damage == "status"
        else ("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs"))
    assert caught.value.problem == fail(code, message).problem
    assert r._read("start:" + KEY) is None
    assert r._read("selection") is None


def test_all_failed_candidates_refuse_before_outer_work(ready, monkeypatch):
    failed_child(monkeypatch, completed=0)
    monkeypatch.setattr(driver, "execute_outer_fold", forbidden)
    with pytest.raises(OpsError) as caught:
        run(ready)
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "all candidates failed for an outer fold").problem
    assert ready.recorder._read("selection") is None


def test_selection_cannot_precede_missing_inner_work(ready, monkeypatch):
    ready.recorder._append("selection", {"premature": True})
    before = durable_evidence(ready)
    monkeypatch.setattr(ready.recorder, "execute", forbidden)
    monkeypatch.setattr(ready.recorder, "reconcile", forbidden)
    with pytest.raises(OpsError) as caught:
        run(ready)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search selection precedes unfinished inner work").problem
    assert durable_evidence(ready) == before
