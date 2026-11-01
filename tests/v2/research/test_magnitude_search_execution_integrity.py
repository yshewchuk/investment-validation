"""Small independent valid-hash negatives for the native execution evidence chain."""
# ruff: noqa: F811
import copy
import math

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import canonical_json, content_hash, to_document
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.native_registration import magnitude_fitted_execution_document
from engine.v2.research.magnitude_inputs import magnitude_recipe
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_trials as trials
from tests.v2.research.test_magnitude_search_execution import LIMITS
from tests.v2.research.test_magnitude_search_trial_integrity import (
    journal,  # noqa: F401
    native_inputs,
    prepare_recording,
    rows,
)


@pytest.fixture
def native(journal, monkeypatch):
    execution = magnitude_fitted_execution_document("2025-01-01", "history_ridge")
    journal.recorder.registration.document["magnitude_execution"] = execution
    for candidate in journal.plan["candidates"]:
        candidate["recipe"] = execution["fit_recipe"]
    inputs = {**native_inputs(), "training": [{"event_id": "train", "absolute_move_pct": 2.0}],
              "features": [{"event_id": event, "context_id": "history", "horizon_sessions": horizon,
                  **dict.fromkeys(execution["features"], 0.0)}
                  for event in ("event", "train") for horizon in (1, 3, 5)]}
    monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: inputs)
    prepare_recording(journal, monkeypatch)
    def fit(*args):
        progress = args[-1]
        for horizon in (5, 3, 1):
            training = {"columns": execution["features"], "row_keys": [["train", "history", horizon]],
                        "values": [[0.0] * 11], "absolute_move_pct": [2.0]}
            prediction = {"columns": execution["features"], "row_keys": [["event", "history", horizon]],
                          "values": [[0.0] * 11]}
            evidence = {"horizon_sessions": horizon, "training_event_ids": ["train"],
                "evaluation_event_ids": ["event"], "training_matrix_id": content_hash(training),
                "prediction_matrix_id": content_hash(prediction)}
            progress("started", evidence)
            progress("completed", {**evidence, "fit_seconds": 0.25})
        return {"event": "succeeded", "forecasts": rows(2.0),
                "score_seconds": 1.5, "rss_delta_kib": 12}
    monkeypatch.setattr(trials, "run_fitted_trial", fit)
    return journal


@pytest.mark.parametrize("affinity", ["missing", "empty", "oserror"])
def test_completed_native_replay_does_not_query_host_affinity(native, monkeypatch, affinity):
    # The synthetic fit needs no host capability even while creating its receipt.
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    result = native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    before = native.conn.total_changes
    queries = []

    def unavailable(pid):
        queries.append(pid)
        if affinity == "oserror":
            raise OSError("synthetic affinity failure")
        return set()

    if affinity == "missing":
        monkeypatch.delattr(execution.os, "sched_getaffinity")
    else:
        monkeypatch.setattr(execution.os, "sched_getaffinity", unavailable)
    append = native.recorder._append

    def replay_append(slot, body, **kwargs):
        if slot == "selection":
            return append(slot, body, **kwargs)
        pytest.fail("replay appended evidence")

    monkeypatch.setattr(native.recorder, "_append", replay_append)
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: pytest.fail("replay opened native inputs"))
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("replay launched fit"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("replay opened labels"))
    with no_fit_guard(), legacy_guard():
        assert native.recorder.execute("2024", "2023", "a-first", limits=LIMITS) == result
        with pytest.raises(OpsError) as caught:
            native.recorder.execute("2024", "2023", "a-first", limits={**LIMITS, "cpu_seconds": 31})
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search execution accounting differs").problem
        with pytest.raises(OpsError) as caught:
            native.recorder.execute("2024", "2023", "a-first", limits={**LIMITS, "cpu_seconds": True})
        assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
            "search execution requires explicit positive integer resource limits").problem
    assert queries == []
    assert native.conn.total_changes == before


def test_fresh_native_affinity_failure_precedes_claim_and_inputs(native, monkeypatch):
    before = native.conn.total_changes
    queries = []

    def unavailable(pid):
        queries.append(pid)
        raise OSError("synthetic affinity failure")

    monkeypatch.setattr(execution.os, "sched_getaffinity", unavailable, raising=False)
    monkeypatch.setattr(native.recorder, "_append", lambda *a, **k: pytest.fail("failed affinity appended evidence"))
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: pytest.fail("failed affinity opened native inputs"))
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("failed affinity launched fit"))
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE",
        "search execution requires Linux resource containment").problem
    assert queries == [0]
    assert native.conn.total_changes == before


@pytest.mark.parametrize("mutation", [None, "attempt_count", "completed_count", "boolean_count", "fit_digest",
    "missing_started", "missing_completed", "unexpected_started", "boolean_horizon", "boolean_start_limit", "missing_execution_start", "limits", "extra"])
def test_native_freeze_requires_every_valid_hash_execution_link(native, monkeypatch, mutation):
    limits = {**LIMITS, "cpu_seconds": 1}
    append = native.recorder._append
    key = content_hash(["plan", "2024", "2023", "a-first"])
    def corrupt(slot, body, **kwargs):
        if key in slot and mutation:
            body = copy.deepcopy(body)
            if mutation == "boolean_start_limit" and slot.startswith("start:"):
                body["execution"]["limits"]["cpu_seconds"] = True
            if mutation == "missing_execution_start" and slot.startswith("execution-start:"):
                return body
            if mutation == "boolean_horizon" and slot.startswith("fit-started:") and body["horizon_sessions"] == 1:
                body["horizon_sessions"] = True
            if mutation == "missing_started" and slot.startswith("fit-started:"):
                return body
            if mutation == "missing_completed" and slot.startswith("fit-completed:"):
                return body
            if mutation == "unexpected_started" and slot.startswith("fit-started:"):
                body["training_matrix_id"] = "wrong"
            if slot.startswith("execution:"):
                if mutation == "attempt_count":
                    body["fit_attempt_count"] = 2
                elif mutation == "completed_count":
                    body["completed_fit_count"] = 2
                elif mutation == "boolean_count":
                    body["completed_fit_count"] = True
                elif mutation == "fit_digest":
                    body["fits"][0]["prediction_matrix_id"] = "wrong"
                elif mutation == "limits":
                    body["attempt"]["limits"]["cpu_seconds"] += 1
                elif mutation == "extra":
                    body["unplanned"] = True
        return append(slot, body, **kwargs)
    monkeypatch.setattr(native.recorder, "_append", corrupt)
    native.recorder.execute("2024", "2023", "z-last", limits=limits)
    if mutation:
        with pytest.raises(OpsError) as caught:
            native.recorder.execute("2024", "2023", "a-first", limits=limits)
        message = "search trial evidence chain differs" if mutation in ("missing_execution_start", "boolean_start_limit") else "search execution accounting differs"
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
        for action in (native.recorder.freeze,
                       lambda: native.recorder.execute("2024", "2023", "a-first", limits=limits)):
            with pytest.raises(OpsError) as caught:
                action()
            assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
        assert native.recorder._read("selection") is None
    else:
        result = native.recorder.execute("2024", "2023", "a-first", limits=limits)
        assert result["terminal"]["metrics"] == {"count": 3, "rmse": 2.0, "mae": 2.0}
        score_seconds = result["terminal"]["score_seconds"]
        assert isinstance(score_seconds, float) and not isinstance(score_seconds, bool)
        assert math.isfinite(score_seconds) and score_seconds >= 1.5
        assert result["terminal"]["rss_delta_kib"] == 12
        assert result["terminal"]["inner_metrics"] == [
            {"horizon_sessions": horizon, "count": 1, "mae": 2.0, "rmse": 2.0,
             "large_move_underprediction_rate": None, "large_move_mean_shortfall": None}
            for horizon in (5, 3, 1)]
        assert all("fit_seconds" not in fit for fit in result["fits"])
        for fit in result["fits"]:
            marker = native.recorder._read("fit-completed:" + result["attempt"]["trial_id"] + ":" + str(fit["horizon_sessions"]))
            assert marker["fit_seconds"] == 0.25
        frozen = native.recorder.freeze()
        assert frozen["prospective_candidate_id"] == "a-first"
        assert native.recorder.freeze() == frozen


@pytest.mark.parametrize("stage", ["inputs", "started", "completed", "execution"])
def test_shared_errors_never_create_candidate_terminal(native, monkeypatch, stage):
    error = fail("EXPERIMENT_VARIANT_FAILED", "shared infrastructure refusal")
    append = native.recorder._append
    def refuse(slot, body, **kw):
        if slot.startswith({"started": "fit-started:", "completed": "fit-completed:", "execution": "execution:"}[stage]):
            raise error
        return append(slot, body, **kw)
    if stage == "inputs":
        monkeypatch.setattr(native.recorder, "_inputs", lambda *a: (_ for _ in ()).throw(error))
    else:
        monkeypatch.setattr(native.recorder, "_append", refuse)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value is error
    key = content_hash(["plan", "2024", "2023", "a-first"])
    assert native.recorder._read("terminal:" + key) is None
    with pytest.raises(OpsError) as caught:
        native.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem


def test_pre_evidence_native_journal_replays_without_rewriting(native, monkeypatch):
    append = native.recorder._append

    def old_writer(slot, body, **kwargs):
        body = copy.deepcopy(body)
        if slot.startswith("fit-completed:"):
            body.pop("fit_seconds", None)
        elif slot.startswith("execution-result:"):
            body["result"].pop("score_seconds", None)
            body["result"].pop("rss_delta_kib", None)
        elif slot.startswith("execution:"):
            body.pop("score_seconds", None)
            body.pop("rss_delta_kib", None)
            for fit in body["fits"]:
                fit.pop("fit_seconds", None)
        elif slot.startswith("terminal:"):
            body.pop("score_seconds", None)
            body.pop("rss_delta_kib", None)
            body.pop("inner_metrics", None)
        return append(slot, body, **kwargs)

    monkeypatch.setattr(native.recorder, "_append", old_writer)
    first = native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    second = native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    for candidate_id, result in (("a-first", first), ("z-last", second)):
        key = content_hash(["plan", "2024", "2023", candidate_id])
        assert "fit_seconds" not in native.recorder._read("fit-completed:" + key + ":5")
        assert "score_seconds" not in native.recorder._read("execution-result:" + key)["result"]
        assert "score_seconds" not in native.recorder._read("execution:" + key)
        assert "fit_seconds" not in native.recorder._read("execution:" + key)["fits"][0]
        assert "inner_metrics" not in result["terminal"]

    with no_fit_guard(), legacy_guard():
        selection = native.recorder.freeze()
    changes = native.conn.total_changes
    def replay_append(slot, body, **kwargs):
        if slot == "selection":
            return append(slot, body, **kwargs)
        pytest.fail("legacy replay appended")

    monkeypatch.setattr(native.recorder, "_append", replay_append)
    with no_fit_guard(), legacy_guard():
        assert native.recorder.execute("2024", "2023", "a-first", limits=LIMITS) == first
        assert native.recorder.execute("2024", "2023", "z-last", limits=LIMITS) == second
        assert native.recorder.freeze() == selection
    assert native.conn.total_changes == changes


def test_native_failure_still_requires_all_planned_candidates(native, monkeypatch):
    original = trials.run_fitted_trial
    def failed(*args):
        def progress(state, evidence):
            if state == "started":
                args[-1](state, evidence)
                raise StopIteration
        try:
            original(*args[:-1], progress)
        except StopIteration:
            return {"event": "failed", "score_seconds": 1.5, "rss_delta_kib": 12}
        raise AssertionError("missing fit admission")
    monkeypatch.setattr(trials, "run_fitted_trial", failed)
    result = native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    assert result["fit_attempt_count"] == 1 and result["completed_fit_count"] == 0
    with pytest.raises(OpsError) as caught:
        native.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "native completion differs from recording").problem
    with pytest.raises(OpsError) as caught:
        native.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    monkeypatch.setattr(trials, "run_fitted_trial", original)
    native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    frozen = native.recorder.freeze()
    assert frozen["prospective_candidate_id"] == "a-first"
    assert frozen["selections"][0]["scores"][0] == {"candidate_id": "z-last", "metrics": None}


def test_native_metrics_refusal_retains_successful_execution_for_replay_and_freeze(native, monkeypatch):
    native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    metrics = trials.magnitude_metrics
    error = fail("EXPERIMENT_VARIANT_FAILED", "synthetic candidate scoring refusal")

    def refuse(*args):
        raise error

    monkeypatch.setattr(trials, "magnitude_metrics", refuse)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "z-last", limits=LIMITS)
    assert caught.value is error
    key = content_hash(["plan", "2024", "2023", "z-last"])
    receipt = native.recorder._read("execution:" + key)
    assert receipt["status"] == "succeeded"
    assert receipt["fit_attempt_count"] == receipt["completed_fit_count"] == 3
    assert receipt["forecasts_id"] == content_hash(rows(2.0))
    terminal = {"status": "refused", "problem": to_document(
        fail("EXPERIMENT_VARIANT_FAILED", "search trial did not complete").problem),
        "score_seconds": 1.5, "rss_delta_kib": 12}
    assert native.recorder._read("terminal:" + key) == terminal
    before = native.conn.total_changes
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("refused replay fitted"))
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: pytest.fail("refused replay opened inputs"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("refused replay opened labels"))
    with no_fit_guard(), legacy_guard():
        assert native.recorder.execute("2024", "2023", "z-last", limits=LIMITS) == {
            **receipt, "terminal": terminal, "evaluation_authority": "none"}
    assert native.conn.total_changes == before
    monkeypatch.setattr(trials, "magnitude_metrics", metrics)
    with no_fit_guard(), legacy_guard():
        frozen = native.recorder.freeze()
        assert frozen["prospective_candidate_id"] == "a-first"
        assert frozen["selections"][0]["scores"][0] == {"candidate_id": "z-last", "metrics": None}
        before = native.conn.total_changes
        assert native.recorder.freeze() == frozen
    assert native.conn.total_changes == before
    assert native.recorder._read("execution:" + key) == receipt


def test_native_progress_reuses_precomputed_evidence_but_receipts_rebuild_it(native, monkeypatch):
    evidence = native.recorder._fit_evidence
    calls = []

    def counted(*args):
        calls.append(None)
        return evidence(*args)

    monkeypatch.setattr(native.recorder, "_fit_evidence", counted)
    original = trials.run_fitted_trial

    def fit(*args):
        assert len(calls) == 1
        result = original(*args)
        assert len(calls) == 1
        return result

    monkeypatch.setattr(trials, "run_fitted_trial", fit)
    result = native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert result["completed_fit_count"] == 3
    assert len(calls) > 1


@pytest.mark.parametrize("error", [MemoryError(), OSError(), RuntimeError("thread creation failed"), ArithmeticError()])
def test_strict_fit_failure_policy_is_opt_in(native, monkeypatch, error):
    import pandas as pd

    from engine.v2.models.training import magnitude
    from experiments import magnitude_fitted

    def refuse(*a):
        raise error
    monkeypatch.setattr(magnitude, "fit_magnitude_predictions", refuse)
    inputs = native.recorder._inputs(None)
    indexed = {(row["event_id"], row["horizon_sessions"]): row for row in inputs["features"]}
    arguments = (indexed, pd.DataFrame(inputs["training"]), native.plan["folds"][0]["inner"][0],
                 native.plan, native.recorder.registration.document["magnitude_execution"])
    with pytest.raises(OpsError) as caught:
        magnitude_fitted.fitted_forecasts(*arguments)
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed").problem
    if isinstance(error, ArithmeticError):
        with pytest.raises(OpsError) as caught:
            magnitude_fitted.fitted_forecasts(*arguments, strict_failures=True)
        assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed").problem
    else:
        with pytest.raises(type(error)) as caught:
            magnitude_fitted.fitted_forecasts(*arguments, strict_failures=True)
        assert caught.value is error


def test_public_record_cannot_recover_failed_native_forecast_publication(native, monkeypatch):
    append = native.recorder._append
    def refuse(slot, body, **kw):
        if slot.startswith("forecasts:"):
            raise fail("RESOURCE_UNAVAILABLE", "synthetic publication failure")
        return append(slot, body, **kw)
    monkeypatch.setattr(native.recorder, "_append", refuse)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "synthetic publication failure").problem
    monkeypatch.setattr(native.recorder, "_append", append)
    with pytest.raises(OpsError) as caught:
        native.recorder.record("2024", "2023", "a-first", forecasts=rows(2.0))
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "native execution must finish before recording").problem
    key = content_hash(["plan", "2024", "2023", "a-first"])
    assert native.recorder._read("forecasts:" + key) is None
    assert native.recorder._read("terminal:" + key) is None


@pytest.mark.parametrize("submission", ["forecasts", "failure"])
def test_crash_between_native_start_slots_cannot_publish_public_terminal(native, monkeypatch, submission):
    append = native.recorder._append
    error = fail("RESOURCE_UNAVAILABLE", "synthetic execution claim interruption")

    def refuse(slot, body, **kwargs):
        if slot.startswith("execution-start:"):
            raise error
        return append(slot, body, **kwargs)

    monkeypatch.setattr(native.recorder, "_append", refuse)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value is error
    key = content_hash(["plan", "2024", "2023", "a-first"])
    assert native.recorder._read("start:" + key)["execution"]["limits"] == LIMITS
    assert native.recorder._read("execution-start:" + key) is None
    assert native.recorder._read("inputs:" + key) is None
    before = native.conn.total_changes
    monkeypatch.setattr(native.recorder, "_append", append)
    supplied = {"forecasts": rows(2.0)} if submission == "forecasts" else {
        "failure": fail("EXPERIMENT_VARIANT_FAILED", "candidate failure").problem}
    with pytest.raises(OpsError) as caught:
        native.recorder.record("2024", "2023", "a-first", **supplied)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "trial has no admitted native inputs").problem
    assert native.conn.total_changes == before
    assert native.recorder._read("forecasts:" + key) is None
    assert native.recorder._read("terminal:" + key) is None


@pytest.mark.parametrize("mutation", ["list", "status", "forecasts_id"])
def test_malformed_native_receipt_is_typed_and_cannot_open_labels(native, monkeypatch, mutation):
    append = native.recorder._append
    def corrupt(slot, body, **kw):
        if slot.startswith("execution:"):
            body = copy.deepcopy(body)
            if mutation == "list":
                body = []
            else:
                body.pop(mutation)
        return append(slot, body, **kw)
    monkeypatch.setattr(native.recorder, "_append", corrupt)
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("malformed receipt read labels"))
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "native completion differs from recording").problem


def test_public_record_cannot_recover_failed_native_failure_publication(native, monkeypatch):
    original = trials.run_fitted_trial
    def failed(*args):
        def progress(state, evidence):
            args[-1](state, evidence)
            raise StopIteration
        try:
            original(*args[:-1], progress)
        except StopIteration:
            return {"event": "failed", "score_seconds": 1.5, "rss_delta_kib": 12}
        raise AssertionError("missing fit admission")
    append = native.recorder._append
    def refuse(slot, body, **kw):
        if slot.startswith("terminal:"):
            raise fail("RESOURCE_UNAVAILABLE", "synthetic terminal failure")
        return append(slot, body, **kw)
    monkeypatch.setattr(trials, "run_fitted_trial", failed)
    monkeypatch.setattr(native.recorder, "_append", refuse)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "synthetic terminal failure").problem
    monkeypatch.setattr(native.recorder, "_append", append)
    with pytest.raises(OpsError) as caught:
        native.recorder.record("2024", "2023", "a-first", failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "native execution must finish before recording").problem
    key = content_hash(["plan", "2024", "2023", "a-first"])
    assert native.recorder._read("terminal:" + key) is None


def test_completed_native_public_record_replay_is_read_only(native, monkeypatch):
    result = native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    before = native.conn.total_changes
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("replay opened labels"))
    monkeypatch.setattr(native.recorder, "_inputs", lambda *a: pytest.fail("replay opened inputs"))
    monkeypatch.setattr(trials, "run_fitted_trial", lambda *a: pytest.fail("replay launched fit"))
    assert native.recorder.record("2024", "2023", "a-first", forecasts=rows(2.0)) == result["terminal"]
    assert native.conn.total_changes == before


@pytest.mark.parametrize("mutation", ["matrix", "repeated_start", "completion_first"])
def test_parent_refuses_child_progress_before_committing_bad_marker(native, monkeypatch, mutation):
    original = trials.run_fitted_trial
    def corrupt(*args):
        progress = args[-1]
        def changed(state, evidence):
            if mutation == "matrix":
                progress(state, {**evidence, "training_matrix_id": "wrong"})
            elif mutation == "completion_first":
                progress("completed", {**evidence, "fit_seconds": 0.25})
            else:
                progress(state, evidence)
                progress(state, evidence)
        return original(*args[:-1], changed)
    monkeypatch.setattr(trials, "run_fitted_trial", corrupt)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search fit progress differs from admitted inputs").problem
    slots = [row[0] for row in native.conn.execute("SELECT slot FROM magnitude_search_records")]
    assert sum(slot.startswith("fit-started:") for slot in slots) == (1 if mutation == "repeated_start" else 0)
    assert not any(slot.startswith(("fit-completed:", "execution:", "terminal:", "forecasts:")) for slot in slots)


@pytest.mark.parametrize("mutation", ["early_success", "unstarted_failure", "extra_success", "extra_failure"])
def test_parent_refuses_child_completion_before_receipt_or_labels(native, monkeypatch, mutation):
    original = trials.run_fitted_trial
    def corrupt(*args):
        if mutation == "early_success":
            return {"event": "succeeded", "forecasts": rows(2.0),
                    "score_seconds": 1.5, "rss_delta_kib": 12}
        if mutation == "unstarted_failure":
            return {"event": "failed", "score_seconds": 1.5, "rss_delta_kib": 12}
        if mutation == "extra_success":
            return {**original(*args), "extra": True}
        def started(state, evidence):
            args[-1](state, evidence)
            raise ValueError("stop after admitted fit")
        with pytest.raises(ValueError, match="stop after admitted fit"):
            original(*args[:-1], started)
        return {"event": "failed", "score_seconds": 1.5, "rss_delta_kib": 12, "extra": True}
    monkeypatch.setattr(trials, "run_fitted_trial", corrupt)
    with pytest.raises(OpsError) as caught:
        native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search fit completion differs from admitted attempts").problem
    slots = [row[0] for row in native.conn.execute("SELECT slot FROM magnitude_search_records")]
    assert not any(slot.startswith(("execution:", "terminal:", "forecasts:")) for slot in slots)


def expected_native_manifest_id(inputs):
    """Literal small-fixture wire oracle: no publisher, builder, store or returned ID."""
    sections = {}
    members = {"features": inputs["features"], "training": inputs["training"],
        "preflight.event_ids": inputs["preflight"]["event_ids"],
        "preflight.planned_inputs": inputs["preflight"]["planned_inputs"],
        **{"training_inputs." + name: [] for name in (
            "candidate_event_ids", "qualified_event_ids", "read_inputs", "omitted_labels",
            "target_recipe.secondary_diagnostics", "holdout_exclusions")}}
    for section, values in members.items():
        section_rows = (values if section == "features" else
                        [{"position": index, "row": row} for index, row in enumerate(values)])
        entries = []
        if section_rows:
            document = {"schema_version": "magnitude_search_shard.v1", "section": section, "rows": section_rows}
            digest = content_hash(document)
            hex_digest = digest.removeprefix("sha256:")
            identity = content_hash({"content_hash": digest, "schema_ref": "magnitude_search_shard.v1"})
            keys = ([[row["event_id"], row["horizon_sessions"]] for row in values]
                    if section == "features" else [[index] for index in range(len(values))])
            entries = [{"ref": {"schema_version": "artifact_ref.v1.0", "artifact_id": "art_" + identity[7:39],
                "content_hash": digest, "schema_ref": "magnitude_search_shard.v1",
                "byte_size": len(canonical_json(document).encode()),
                "storage_key": f"objects/{hex_digest[:2]}/{hex_digest}"},
                "rows": len(values), "first_key": keys[0], "last_key": keys[-1]}]
        sections[section] = {"row_count": len(values), "shards": entries}
    inline = {"preflight." + name: value for name, value in inputs["preflight"].items()
              if name not in ("event_ids", "planned_inputs")}
    return content_hash({"schema_version": "magnitude_search_cohort_manifest.v1",
                         "inline": inline, "sections": sections})


def expected_native_receipt(succeeded=True):
    """Derive the oracle from synthetic fixture data, never a returned receipt."""
    columns = magnitude_fitted_execution_document("2025-01-01", "history_ridge")["features"]
    inputs = {**native_inputs(), "training": [{"event_id": "train", "absolute_move_pct": 2.0}],
              "features": [{"event_id": event, "context_id": "history", "horizon_sessions": horizon,
                  **dict.fromkeys(columns, 0.0)}
                  for event in ("event", "train") for horizon in (1, 3, 5)]}
    inputs_id = expected_native_manifest_id(inputs)
    fits = []
    for horizon in (5, 3, 1):
        training = {"columns": columns, "row_keys": [["train", "history", horizon]],
                    "values": [[0.0] * 11], "absolute_move_pct": [2.0]}
        prediction = {"columns": columns, "row_keys": [["event", "history", horizon]], "values": [[0.0] * 11]}
        fits.append({"horizon_sessions": horizon, "training_event_ids": ["train"],
            "evaluation_event_ids": ["event"], "training_matrix_id": content_hash(training),
            "prediction_matrix_id": content_hash(prediction)})
    preflight, recipe = inputs["preflight"], magnitude_recipe()
    evaluation = {"schema_version": "research_magnitude_targets.v1",
        **{name: preflight[name] for name in ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use", "purpose", "computed_moves_version")},
        "labels_before": "2024-01-01", "target_recipe": recipe, "target_recipe_id": content_hash(recipe),
        "consumed_inputs": preflight["planned_inputs"], "excluded_future_events": []}
    validation = {"inputs_id": inputs_id,
        "evaluation_inputs": {**evaluation, "inputs_id": content_hash(evaluation), "holdout_exclusions": []},
        "targets": [{"event_id": "event", "absolute_move_pct": 4.0}]}
    inner_metrics = [{"horizon_sessions": horizon, "count": 1, "mae": 2.0, "rmse": 2.0,
        "large_move_underprediction_rate": None, "large_move_mean_shortfall": None}
        for horizon in (5, 3, 1)]
    terminal = ({"status": "succeeded", "inputs_id": inputs_id,
        "validation_id": content_hash(validation), "evaluation_inputs": validation["evaluation_inputs"],
        "predictions": [{**row, "absolute_move_pct": 4.0} for row in rows(2.0)],
        "metrics": {"count": 3, "rmse": 2.0, "mae": 2.0},
        "score_seconds": 1.5, "rss_delta_kib": 12, "inner_metrics": inner_metrics} if succeeded else
        {"status": "failed", "problem": to_document(fail("EXPERIMENT_VARIANT_FAILED",
            "search trial did not complete").problem),
        "score_seconds": 1.5, "rss_delta_kib": 12})
    return {"attempt": {"limits": LIMITS, "candidate_id": "a-first",
            "trial_id": content_hash(["plan", "2024", "2023", "a-first"])},
        "fit_attempt_count": 3 if succeeded else 1, "completed_fit_count": 3 if succeeded else 0,
        "fits": fits if succeeded else [], "status": "succeeded" if succeeded else "failed",
        "forecasts_id": content_hash(rows(2.0)) if succeeded else None,
        "score_seconds": 1.5, "rss_delta_kib": 12,
        "terminal": terminal, "evaluation_authority": "none"}


@pytest.mark.parametrize("replay_method", ["execute", "reconcile"])
@pytest.mark.parametrize("succeeded", [True, False])
@pytest.mark.parametrize("recovery_stage", [None, "execution", "terminal"])
def test_native_response_verifies_each_fit_marker_once_and_replays_fresh(
        native, monkeypatch, succeeded, replay_method, recovery_stage):
    import json
    from collections import Counter

    from engine.v2.foundation import ArtifactStore
    from experiments.magnitude_search_inputs import load_inputs

    if not succeeded:
        def failed(source, inner, plan, execution, limits, progress):
            inputs = load_inputs(ArtifactStore(source["store_root"]), source["manifest"], source["inputs_id"])
            progress("started", trials.fit_evidence(inputs, inner, plan, execution)[0])
            return {"event": "failed", "score_seconds": 1.5, "rss_delta_kib": 12}
        monkeypatch.setattr(trials, "run_fitted_trial", failed)
    opened = Counter()
    read = ArtifactStore.read_verified

    def verified(store, ref):
        payload = read(store, ref)
        record = json.loads(payload)
        if record.get("schema_version") == "magnitude_search_record.v1":
            opened[record["slot"].split(":", 1)[0]] += 1
        return payload

    monkeypatch.setattr(ArtifactStore, "read_verified", verified)
    if recovery_stage is None:
        result = native.recorder.execute("2024", "2023", "a-first", limits=LIMITS)
    else:
        from tests.v2.research.test_magnitude_search_reconciliation import interrupt, no_launch

        interrupt(monkeypatch, native.recorder, recovery_stage, after=False)
        no_launch(monkeypatch)
        opened.clear()
        with no_fit_guard(), legacy_guard():
            result = native.recorder.reconcile("2024", "2023", "a-first", limits=LIMITS)
    expected = expected_native_receipt(succeeded)
    expected["terminal"]["score_seconds"] = result["terminal"]["score_seconds"]
    assert expected["terminal"]["score_seconds"] >= result["score_seconds"]
    assert result == expected
    assert opened["execution-result"] == 1
    assert opened["fit-started"] == (3 if succeeded else 1)
    assert opened["fit-completed"] == (3 if succeeded else 0)
    replay = getattr(native.recorder, replay_method)
    before = native.conn.total_changes
    for _ in range(2):
        opened.clear()
        with no_fit_guard(), legacy_guard():
            assert replay("2024", "2023", "a-first", limits=LIMITS) == expected
        assert opened["execution-result"] == 1
        assert opened["fit-started"] == (3 if succeeded else 1)
        assert opened["fit-completed"] == (3 if succeeded else 0)
        assert native.conn.total_changes == before
    with pytest.raises(OpsError) as caught:
        replay("2024", "2023", "a-first", limits={**LIMITS, "cpu_seconds": 31})
    message = ("search execution accounting differs" if replay_method == "execute"
               else "search durable execution result differs")
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem


@pytest.mark.parametrize("replay_method", ["execute", "reconcile"])
@pytest.mark.parametrize("cpu_seconds", [30, 37])
def test_completed_replay_rechecks_durable_result_after_prior_success(native, replay_method, cpu_seconds):
    import json

    from engine.v2.contracts import ArtifactRef
    from engine.v2.foundation import from_document

    limits = {**LIMITS, "cpu_seconds": cpu_seconds}
    expected = expected_native_receipt()
    expected["attempt"]["limits"] = limits
    replay = getattr(native.recorder, replay_method)
    result = native.recorder.execute("2024", "2023", "a-first", limits=limits)
    expected["terminal"]["score_seconds"] = result["terminal"]["score_seconds"]
    assert result == expected
    repeated = replay("2024", "2023", "a-first", limits=limits)
    assert repeated == expected
    assert repeated["attempt"]["limits"]["cpu_seconds"] == cpu_seconds
    key = content_hash(["plan", "2024", "2023", "a-first"])
    ref_json = native.conn.execute("SELECT ref_json FROM magnitude_search_records WHERE slot=?",
                                  ("execution-result:" + key,)).fetchone()[0]
    ref = from_document(ArtifactRef, json.loads(ref_json))
    path = native.store.root / ref.storage_key
    path.chmod(0o644)
    path.write_bytes(b"corrupt after successful replay")
    with pytest.raises(OpsError) as caught:
        replay("2024", "2023", "a-first", limits=limits)
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.message == "search journal artifact could not be verified"


def test_inner_metrics_limits_large_move_diagnostics_to_threshold_rows():
    scored = [
        {"horizon_sessions": 5, "absolute_move_pct": 12.0, "predicted_absolute_move_pct": 5.0},
        {"horizon_sessions": 5, "absolute_move_pct": 9.9, "predicted_absolute_move_pct": 20.0},
    ]
    [metrics] = trials.SearchTrials._inner_metrics(scored, [5])
    assert metrics["count"] == 2
    assert metrics["large_move_underprediction_rate"] == 1.0
    assert metrics["large_move_mean_shortfall"] == 7.0


def test_inner_metrics_large_move_diagnostics_are_unavailable_below_threshold():
    scored = [
        {"horizon_sessions": 5, "absolute_move_pct": 9.9, "predicted_absolute_move_pct": 20.0},
        {"horizon_sessions": 5, "absolute_move_pct": 2.0, "predicted_absolute_move_pct": 1.0},
    ]
    [metrics] = trials.SearchTrials._inner_metrics(scored, [5])
    assert metrics["count"] == 2
    assert metrics["large_move_underprediction_rate"] is None
    assert metrics["large_move_mean_shortfall"] is None
