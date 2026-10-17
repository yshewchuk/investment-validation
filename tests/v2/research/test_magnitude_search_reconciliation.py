"""Durable post-fit publication repair: real journal, fault cuts, and no-fit oracles."""
# ruff: noqa: F811
import copy

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_execution_integrity import native  # noqa: F401
from tests.v2.research.test_magnitude_search_trial_integrity import (
    durable_evidence,
    journal,  # noqa: F401
    rows,
)

ARGS = ("2024", "2023", "a-first")
KEY = content_hash(["plan", *ARGS])
COHORT = content_hash(["plan", "2024", "2023"])


def interrupt(monkeypatch, recorder, stage, after):
    append = recorder._append
    error = fail("RESOURCE_UNAVAILABLE", "synthetic publication interruption")

    def cut(slot, body, **kwargs):
        if slot.startswith(stage + ":"):
            if after:
                append(slot, body, **kwargs)
            raise error
        return append(slot, body, **kwargs)

    monkeypatch.setattr(recorder, "_append", cut)
    with pytest.raises(OpsError) as caught:
        recorder.execute(*ARGS, limits=LIMITS)
    assert caught.value is error
    monkeypatch.setattr(recorder, "_append", append)


def no_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("reconciliation must never launch or resolve a child")
    monkeypatch.setattr(trials, "run_fitted_trial", forbidden)
    monkeypatch.setattr(trials, "execution_limits", forbidden)


def failed_child(monkeypatch, completed):
    original = trials.run_fitted_trial

    def failed(*args):
        started = 0

        def progress(state, evidence):
            nonlocal started
            args[-1](state, evidence)
            if state == "started":
                started += 1
                if started == completed + 1:
                    raise StopIteration
        try:
            original(*args[:-1], progress)
        except StopIteration:
            return {"event": "failed"}
        raise AssertionError("missing fit admission")
    monkeypatch.setattr(trials, "run_fitted_trial", failed)


@pytest.mark.parametrize("stage", ["execution-result", "execution", "forecasts", "validation", "terminal"])
@pytest.mark.parametrize("after", [False, True])
def test_publication_faults_reconcile_only_durable_results(native, monkeypatch, stage, after):
    recorder = native.recorder
    interrupt(monkeypatch, recorder, stage, after)
    before = durable_evidence(native)
    no_launch(monkeypatch)
    if stage == "execution-result" and not after:
        with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
            recorder.reconcile(*ARGS, limits=LIMITS)
        assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert durable_evidence(native) == before
        assert recorder._read("terminal:" + KEY) is None
        return
    durable = recorder._read("execution-result:" + KEY)
    assert durable == {"attempt": {"trial_id": KEY, "candidate_id": "a-first", "limits": LIMITS},
                       "inputs_id": content_hash(recorder._read("cohort:" + COHORT)),
                       "result": {"event": "succeeded", "forecasts": rows(2.0)}}
    if stage == "terminal" or (stage == "validation" and after):
        monkeypatch.setattr(recorder, "_inputs", lambda *a: pytest.fail("durable validation reopened inputs"))
        monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("durable validation reopened labels"))
    with no_fit_guard(), legacy_guard():
        result = recorder.reconcile(*ARGS, limits=LIMITS)
    assert result["status"] == result["terminal"]["status"] == "succeeded"
    assert result["fit_attempt_count"] == result["completed_fit_count"] == 3
    assert result["terminal"]["metrics"] == {"count": 3, "rmse": 2.0, "mae": 2.0}
    assert result["evaluation_authority"] == "none"
    assert recorder._read("execution-result:" + KEY) == durable
    before = durable_evidence(native)
    monkeypatch.setattr(recorder, "_append", lambda *a, **k: pytest.fail("completed replay wrote evidence"))
    monkeypatch.setattr(recorder, "_inputs", lambda *a: pytest.fail("completed replay opened inputs"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("completed replay opened labels"))
    with no_fit_guard(), legacy_guard():
        assert recorder.reconcile(*ARGS, limits=LIMITS) == result
        assert recorder.execute(*ARGS, limits=LIMITS) == result
    assert durable_evidence(native) == before


@pytest.mark.parametrize("stage", ["execution-result", "execution", "terminal"])
@pytest.mark.parametrize("completed", [0, 1, 2])
def test_candidate_failure_reconciliation_retains_exact_attempt_prefix(native, monkeypatch, stage, completed):
    failed_child(monkeypatch, completed)
    interrupt(monkeypatch, native.recorder, stage, after=True)
    no_launch(monkeypatch)
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: pytest.fail("failed trial opened inputs"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("failed trial opened labels"))
    with no_fit_guard(), legacy_guard():
        result = native.recorder.reconcile(*ARGS, limits=LIMITS)
        assert native.recorder.reconcile(*ARGS, limits=LIMITS) == result
    assert result["status"] == result["terminal"]["status"] == "failed"
    assert result["fit_attempt_count"] == completed + 1
    assert result["completed_fit_count"] == completed
    assert len(result["fits"]) == completed
    assert native.recorder._read("forecasts:" + KEY) is None
    with pytest.raises(OpsError) as caught:
        native.recorder.freeze()
    assert caught.value.problem.code == "EXPERIMENT_VARIANT_FAILED"
    assert native.recorder._read("selection") is None


@pytest.mark.parametrize("stage", ["start", "execution-start", "inputs", "fit-started", "fit-completed"])
def test_interrupted_fit_or_admission_cannot_be_reconciled(native, monkeypatch, stage):
    interrupt(monkeypatch, native.recorder, stage, after=True)
    before = durable_evidence(native)
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert durable_evidence(native) == before
    assert native.recorder._read("execution-result:" + KEY) is None


@pytest.mark.parametrize("slot, mutation", [
    ("execution-result", "missing"), ("execution-result", "extra"), ("execution-result", "trial"),
    ("execution-result", "candidate"), ("execution-result", "limits"), ("execution-result", "inputs"),
    ("execution-result", "result_extra"), ("execution-result", "status"), ("execution-result", "forecasts"),
    ("execution-result", "forecast_value"), ("execution-result", "forecast_order"),
    ("execution", "count"), ("execution", "digest"),
    ("forecasts", "value"), ("validation", "targets"), ("terminal", "metrics"),
    ("fit-started", "missing"), ("fit-completed", "missing"), ("fit-completed", "matrix"),
    ("start", "limits"), ("execution-start", "limits"), ("inputs", "inputs"), ("cohort", "training"),
])
def test_valid_hash_mutations_cannot_authorize_reconciliation(native, monkeypatch, slot, mutation):
    append = native.recorder._append

    def corrupt(name, body, **kwargs):
        original_body = body
        if name.startswith(slot + ":"):
            body = copy.deepcopy(body)
            if mutation == "missing":
                return body
            if slot == "execution-result":
                if mutation == "extra":
                    body["extra"] = True
                elif mutation == "trial":
                    body["attempt"]["trial_id"] = "wrong"
                elif mutation == "candidate":
                    body["attempt"]["candidate_id"] = "z-last"
                elif mutation == "limits":
                    body["attempt"]["limits"]["cpu_seconds"] = True
                elif mutation == "inputs":
                    body["inputs_id"] = "wrong"
                elif mutation == "result_extra":
                    body["result"]["extra"] = True
                elif mutation == "status":
                    body["result"]["event"] = "resource_limit"
                elif mutation == "forecasts":
                    body["result"]["forecasts"].pop()
                elif mutation == "forecast_value":
                    body["result"]["forecasts"][0]["predicted_absolute_move_pct"] = 99.0
                else:
                    body["result"]["forecasts"].reverse()
            elif slot == "execution":
                body["fit_attempt_count" if mutation == "count" else "forecasts_id"] = 1
            elif slot == "forecasts":
                body[0]["predicted_absolute_move_pct"] = 99.0
            elif slot == "validation":
                body["targets"][0]["event_id"] = "wrong"
            elif slot == "terminal":
                body["metrics"]["rmse"] = 99.0
            elif slot == "fit-completed":
                body["prediction_matrix_id"] = "wrong"
            elif slot == "start":
                body["execution"]["limits"]["cpu_seconds"] = True
            elif slot == "execution-start":
                body["limits"]["cpu_seconds"] = True
            elif slot == "inputs":
                body["inputs_id"] = "wrong"
            else:
                body["training"][0]["absolute_move_pct"] = 99.0
        persisted = append(name, body, **kwargs)
        return original_body if slot == "validation" else persisted

    monkeypatch.setattr(native.recorder, "_append", corrupt)
    try:
        native.recorder.execute(*ARGS, limits=LIMITS)
    except OpsError:
        pass
    monkeypatch.setattr(native.recorder, "_append", append)
    before = durable_evidence(native)
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert durable_evidence(native) == before


@pytest.mark.parametrize("change", ["source", "limits", "invalid_limits", "candidate", "fold"])
def test_reconciliation_rechecks_seal_trial_and_limit_identity(native, monkeypatch, change):
    interrupt(monkeypatch, native.recorder, "execution-result", after=True)
    before = durable_evidence(native)
    no_launch(monkeypatch)
    args, limits = ARGS, LIMITS
    if change == "source":
        monkeypatch.setattr(native.recorder, "_binding", lambda: (native.plan, {"plan_id": "changed source"}))
    elif change in ("limits", "invalid_limits"):
        limits = {**LIMITS, "cpu_seconds": 31 if change == "limits" else True}
    elif change == "candidate":
        args = (*ARGS[:2], "other")
    else:
        args = ("2025", *ARGS[1:])
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        native.recorder.reconcile(*args, limits=limits)
    assert caught.value.problem.code == ("INVALID_EXPERIMENT_SPEC" if change in
        ("invalid_limits", "candidate", "fold") else "EXPERIMENT_IDENTITY_CONFLICT")
    assert durable_evidence(native) == before


@pytest.mark.parametrize("stage", ["inputs", "labels", "metrics"])
@pytest.mark.parametrize("code", ["RESOURCE_UNAVAILABLE", "RESOURCE_LIMIT_EXCEEDED", "EXPERIMENT_VARIANT_FAILED"])
def test_reconciliation_never_reclassifies_shared_or_resource_failures(native, monkeypatch, stage, code):
    interrupt(monkeypatch, native.recorder, "execution", after=True)
    no_launch(monkeypatch)
    original = fail(code, "synthetic shared or scoring refusal")

    def refuse(*a, **k):
        raise original
    target, name = (native.recorder, "_inputs") if stage == "inputs" else (
        trials, "load_magnitude_targets" if stage == "labels" else "magnitude_metrics")
    monkeypatch.setattr(target, name, refuse)
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert caught.value is original
    terminal = native.recorder._read("terminal:" + KEY)
    if stage == "metrics" and code == "EXPERIMENT_VARIANT_FAILED":
        assert terminal["status"] == "refused"
        assert native.recorder.reconcile(*ARGS, limits=LIMITS)["terminal"] == terminal
    else:
        assert terminal is None
    assert native.recorder._read("selection") is None


def test_real_fixed_child_is_reaped_before_result_becomes_recoverable(native, monkeypatch):
    from experiments import magnitude_search_execution as execution

    children = []
    original = execution.subprocess.Popen
    append = native.recorder._append

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    def publish(slot, body, **kwargs):
        if slot.startswith("execution-result:"):
            assert len(children) == 1
            assert children[0].returncode == 0
            assert children[0].poll() == 0
        if slot.startswith("execution:"):
            raise fail("RESOURCE_UNAVAILABLE", "cut after reaped child result")
        return append(slot, body, **kwargs)

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    monkeypatch.setattr(trials, "run_fitted_trial", execution.run_fitted_trial)
    monkeypatch.setattr(native.recorder, "_append", publish)
    with pytest.raises(OpsError, match="cut after reaped child result"):
        native.recorder.execute(*ARGS, limits=LIMITS)
    monkeypatch.setattr(native.recorder, "_append", append)
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard():
        result = native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert result["completed_fit_count"] == 3
    assert result["terminal"]["metrics"] == {"count": 3, "rmse": 2.0, "mae": 2.0}
    assert len(children) == 1


def test_reconciliation_checks_current_inputs_before_first_validation_read(native, monkeypatch):
    interrupt(monkeypatch, native.recorder, "execution", after=True)
    no_launch(monkeypatch)
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: {"changed": True})
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("changed inputs opened labels"))
    with pytest.raises(OpsError, match="trial native inputs changed after start"):
        native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert native.recorder._read("validation:" + COHORT) is None
    assert native.recorder._read("terminal:" + KEY) is None


def test_reconciled_failures_still_require_all_candidates_and_match_selection(native, monkeypatch):
    native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    failed_child(monkeypatch, completed=1)
    interrupt(monkeypatch, native.recorder, "execution-result", after=True)
    with pytest.raises(OpsError, match="search selection requires every terminal attempt"):
        native.recorder.freeze()
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard():
        native.recorder.reconcile(*ARGS, limits=LIMITS)
        frozen = native.recorder.freeze()
    assert frozen["prospective_candidate_id"] == "z-last"
    assert frozen["selections"][0]["scores"][1] == {"candidate_id": "a-first", "metrics": None}
    assert len(frozen["attempts"]) == 2


def test_failed_scoring_publication_reconciles_to_original_refusal(native, monkeypatch):
    original = trials.run_fitted_trial

    def huge_forecasts(*args):
        return {**original(*args), "forecasts": rows(1e308)}

    monkeypatch.setattr(trials, "run_fitted_trial", huge_forecasts)

    def scoring_refusal(*args):
        raise fail("EXPERIMENT_VARIANT_FAILED", "synthetic candidate scoring refusal")

    monkeypatch.setattr(trials, "magnitude_metrics", scoring_refusal)
    append = native.recorder._append

    def refuse(slot, body, **kwargs):
        if slot.startswith("terminal:"):
            raise fail("RESOURCE_UNAVAILABLE", "synthetic refusal write failure")
        return append(slot, body, **kwargs)

    monkeypatch.setattr(native.recorder, "_append", refuse)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute(*ARGS, limits=LIMITS)
    assert caught.value.problem.code == "EXPERIMENT_VARIANT_FAILED"
    assert caught.value.__cause__.problem.code == "RESOURCE_UNAVAILABLE"
    monkeypatch.setattr(native.recorder, "_append", append)
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard(), pytest.raises(OpsError) as caught:
        native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert caught.value.problem.code == "EXPERIMENT_VARIANT_FAILED"
    result = native.recorder.reconcile(*ARGS, limits=LIMITS)
    assert result["status"] == "succeeded" and result["terminal"]["status"] == "refused"
    assert native.recorder._read("forecasts:" + KEY) == rows(1e308)


def test_missing_forecast_is_committed_before_reusing_shared_validation(native, monkeypatch):
    native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    interrupt(monkeypatch, native.recorder, "execution-result", after=True)
    read = native.recorder._read

    def ordered(slot):
        if slot.startswith("validation:"):
            assert read("forecasts:" + KEY) == rows(2.0)
        return read(slot)

    monkeypatch.setattr(native.recorder, "_read", ordered)
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: pytest.fail("shared validation reopened inputs"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("shared validation reopened labels"))
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard():
        assert native.recorder.reconcile(*ARGS, limits=LIMITS)["terminal"]["status"] == "succeeded"


def test_failed_reconciliation_does_not_read_shared_validation(native, monkeypatch):
    native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    failed_child(monkeypatch, completed=0)
    interrupt(monkeypatch, native.recorder, "execution-result", after=True)
    read = native.recorder._read

    def no_validation(slot):
        if slot.startswith("validation:"):
            pytest.fail("failed reconciliation read unrelated shared validation")
        return read(slot)

    monkeypatch.setattr(native.recorder, "_read", no_validation)
    no_launch(monkeypatch)
    with no_fit_guard(), legacy_guard():
        assert native.recorder.reconcile(*ARGS, limits=LIMITS)["terminal"]["status"] == "failed"


def test_new_recorder_reconciles_catalog_durable_result_after_reopen(native, monkeypatch):
    from engine.v2.ops.catalog import connect

    interrupt(monkeypatch, native.recorder, "execution-result", after=True)
    path = next(row[2] for row in native.conn.execute("PRAGMA database_list") if row[1] == "main")
    peer = connect(path)
    try:
        recorder = trials.SearchTrials(peer, native.store, native.recorder.repository,
            native.recorder.registration, native.recorder.request, native.recorder.plan_ref)
        monkeypatch.setattr(recorder, "_binding", native.recorder._binding)
        monkeypatch.setattr(recorder, "_inputs", native.recorder._inputs)
        no_launch(monkeypatch)
        with no_fit_guard(), legacy_guard():
            recovered = recorder.reconcile(*ARGS, limits=LIMITS)
            assert native.recorder.reconcile(*ARGS, limits=LIMITS) == recovered
        assert recovered["completed_fit_count"] == 3
        assert recovered["terminal"]["status"] == "succeeded"
    finally:
        peer.close()
