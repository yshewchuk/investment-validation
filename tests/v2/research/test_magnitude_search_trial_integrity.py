"""Small real-journal oracles for exact refusal, phase, and valid-hash chain guards."""
import copy
import json
import sqlite3
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES, Problem, SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    SystemClock,
    content_hash,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials
from experiments.magnitude_report import magnitude_metrics


@pytest.fixture
def journal(tmp_path, monkeypatch):
    conn = connect(tmp_path / "catalog.sqlite")
    conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY) STRICT")
    conn.execute("INSERT INTO experiment_runs VALUES ('run')")
    store = ArtifactStore(tmp_path / "store")
    plan = {"plan_id": "plan", "horizons": [5, 3, 1], "candidates": [
        {"candidate_id": "z-last"}, {"candidate_id": "a-first"}], "folds": [{
        "outer": {"year": "2024"}, "inner": [{"year": "2023", "evaluation_event_ids": ["event"],
        "evaluation_labels_before": "2024-01-01"}]}]}
    recorder = trials.SearchTrials(conn, store, None, SimpleNamespace(run_id="run", document={}), {}, None)
    monkeypatch.setattr(recorder, "_binding", lambda: (plan, {"plan_id": "plan"}))
    monkeypatch.setattr(recorder, "_inputs", lambda inner: {"native_inputs": "same"})
    recorder.seal(clock=SystemClock())
    yield SimpleNamespace(conn=conn, store=store, recorder=recorder, plan=plan)
    conn.close()


def rows(value=7):
    return [{"event_id": "event", "horizon_sessions": horizon, "predicted_absolute_move_pct": value}
            for horizon in [1, 3, 5]]


def expected_problem(code):
    return to_document(Problem(code=code, category="internal" if code == "EXPERIMENT_VARIANT_FAILED"
        else "validation", retryable=False, message="search trial did not complete"))


def forge_complete(journal, mutation=None):
    """Publish internally inconsistent evidence with valid content hashes, never corrupt bytes."""
    cohort = content_hash(["plan", "2024", "2023"])
    inputs = {"native_inputs": "same"}
    if mutation != "missing_cohort":
        journal.recorder._append("cohort:" + cohort, inputs)
    for candidate in journal.plan["candidates"]:
        key = content_hash(["plan", "2024", "2023", candidate["candidate_id"]])
        started = {"trial_id": key, "outer_year": "2024", "inner_year": "2023", "candidate": candidate}
        admitted = {"cohort": cohort, "inputs_id": content_hash(inputs)}
        predictions = [{**row, "absolute_move_pct": 4.0} for row in rows()]
        terminal = {"status": "succeeded", "inputs_id": admitted["inputs_id"],
                    "evaluation_inputs": {}, "predictions": predictions, "metrics": magnitude_metrics(predictions)}
        if mutation == "start":
            started["trial_id"] = "wrong"
        elif mutation == "cohort_link":
            admitted["cohort"] = "missing"
        elif mutation == "cohort_digest":
            admitted["inputs_id"] = terminal["inputs_id"] = "wrong"
        elif mutation == "terminal_inputs":
            terminal["inputs_id"] = "wrong"
        elif mutation == "forecasts":
            predictions[0]["predicted_absolute_move_pct"] = 8
            terminal["metrics"] = magnitude_metrics(predictions)
        elif mutation == "metrics":
            terminal["metrics"]["rmse"] = 42
        elif mutation == "status":
            terminal["status"] = "unknown"
        elif mutation in ("failure_policy", "failure_unknown", "infrastructure_terminal", "refused_no_inputs", "refused_no_forecasts"):
            problem = expected_problem("EXPERIMENT_VARIANT_FAILED")
            if mutation == "failure_policy":
                problem["retryable"] = True
            elif mutation == "failure_unknown":
                problem["code"] = "UNREGISTERED"
            elif mutation == "infrastructure_terminal":
                problem.update(code="RESOURCE_UNAVAILABLE", category="resource", retryable=True)
            terminal = {"status": "refused" if mutation.startswith("refused_") else "failed", "problem": problem}
        for prefix, body in (("start", started), ("inputs", admitted), ("forecasts", rows()), ("terminal", terminal)):
            if (mutation, prefix) in (("refused_no_inputs", "inputs"), ("refused_no_forecasts", "forecasts")):
                continue
            journal.recorder._append(prefix + ":" + key, body)


@pytest.mark.parametrize("mutation", ["start", "missing_cohort", "cohort_link", "cohort_digest", "terminal_inputs",
    "forecasts", "metrics", "status", "failure_policy", "failure_unknown", "infrastructure_terminal",
    "refused_no_inputs", "refused_no_forecasts"])
def test_valid_hash_inconsistent_evidence_chain_refuses(journal, mutation):
    forge_complete(journal, mutation)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
    assert journal.recorder._read("selection") is None


def test_equal_metrics_choose_lowest_id_even_when_candidates_are_reversed(journal):
    forge_complete(journal)
    frozen = journal.recorder.freeze()
    assert frozen["selections"][0]["selected_candidate_id"] == "a-first"
    assert frozen["prospective_candidate_id"] == "a-first"
    assert frozen["selections"][0]["scores"][0]["metrics"] == frozen["selections"][0]["scores"][1]["metrics"]


@pytest.mark.parametrize("problem", ["x", Problem(code="UNREGISTERED", category="internal", retryable=False, message="x")])
def test_invalid_problem_refuses_exactly_without_terminal(journal, problem):
    started = journal.recorder.start("2024", "2023", "z-last")
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", failure=problem)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "trial failure requires a registered Problem").problem
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None


def test_failed_terminal_has_exact_registry_policy_and_no_caller_diagnostics(journal):
    journal.recorder.start("2024", "2023", "z-last")
    result = journal.recorder.record("2024", "2023", "z-last", failure=Problem(
        code="EXPERIMENT_VARIANT_FAILED", category="resource", retryable=True, message="private", details={"secret": 7}))
    assert result == {"status": "failed", "problem": expected_problem("EXPERIMENT_VARIANT_FAILED")}


def test_source_closure_failure_is_exact_identity_refusal_before_sealing(journal, monkeypatch):
    monkeypatch.delattr(journal.recorder, "_binding")
    monkeypatch.setattr(trials, "read_search_plan", lambda *args: journal.plan)
    def broken(*args):
        raise OSError("private path")
    monkeypatch.setattr(trials, "source_closure", broken)
    with pytest.raises(OpsError) as caught:
        journal.recorder.seal(clock=SystemClock())
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search recorder sources cannot be verified").problem


def test_changed_shared_cohort_blocks_selection_instead_of_excluding_candidate(journal, monkeypatch):
    journal.recorder.start("2024", "2023", "z-last")
    monkeypatch.setattr(journal.recorder, "_inputs", lambda inner: {"native_inputs": "changed"})
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "a-first")
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    key = content_hash(["plan", "2024", "2023", "a-first"])
    assert journal.recorder._read("start:" + key) is not None
    assert journal.recorder._read("terminal:" + key) is None
    assert journal.recorder._read("inputs:" + key) is None
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "a-first")
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem


@pytest.mark.parametrize("phase, artifact_code", [
    (phase, code) for phase in ("cohort", "inputs", "forecasts", "terminal")
    for code in ("IO_ERROR", "INTEGRITY_FAILED", "UNSAFE_PATH")
] + [("native_inputs", "IO_ERROR"), ("labels", "IO_ERROR")])
def test_infrastructure_failure_never_excludes_candidate_or_allows_freeze(journal, monkeypatch, phase, artifact_code):
    # A fully successful peer would win if this infrastructure failure became a terminal refusal.
    original_inputs = journal.recorder._inputs
    key = content_hash(["plan", "2024", "2023", "z-last"])
    original_publish = ArtifactStore.publish_bytes
    def publish(store, payload, *, schema_ref):
        if json.loads(payload).get("slot", "").startswith(phase + ":"):
            raise ArtifactError(artifact_code, "private path")
        return original_publish(store, payload, schema_ref=schema_ref)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", publish)
    if phase == "native_inputs":
        def inputs(inner):
            raise fail("RESOURCE_UNAVAILABLE", "temporary native input failure")
        monkeypatch.setattr(journal.recorder, "_inputs", inputs)
    if phase in ("native_inputs", "cohort", "inputs"):
        with pytest.raises(OpsError) as caught:
            journal.recorder.start("2024", "2023", "z-last")
    else:
        journal.recorder.start("2024", "2023", "z-last")
        # Supply only the minimal admitted snapshot document needed by record's decoder.
        from engine.v2.contracts import SnapshotRef
        snapshot = SnapshotRef(snapshot_id="snapshot", manifest_hash="hash", table_versions={},
            calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(),
            knowledge_mode_by_table={})
        journal.recorder.registration.document.update(snapshot=to_document(snapshot), holdouts={"holdout_as_of_month": "2026-01"})
        def labels(*args, **kwargs):
            if phase == "labels":
                raise data_fail("RESOURCE_UNAVAILABLE", "temporary label read failure")
            return pd.DataFrame([{"event_id": "event", "absolute_move_pct": 4.0}])
        monkeypatch.setattr(trials, "load_magnitude_targets", labels)
        from engine.v2.data.errors import DataError
        with pytest.raises((OpsError, DataError)) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    expected_code = "RESOURCE_UNAVAILABLE" if artifact_code == "IO_ERROR" else "EXPERIMENT_IDENTITY_CONFLICT"
    message = ("temporary native input failure" if phase == "native_inputs" else
               "temporary label read failure" if phase == "labels" else
               "search journal artifact could not be verified or published")
    assert caught.value.problem == fail(expected_code, message).problem
    assert journal.recorder._read("terminal:" + key) is None
    monkeypatch.setattr(journal.recorder, "_inputs", original_inputs)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", original_publish)
    forge_complete_peer(journal)
    with pytest.raises(OpsError) as freeze_error:
        journal.recorder.freeze()
    assert freeze_error.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None
    with pytest.raises(OpsError) as retry_error:
        journal.recorder.start("2024", "2023", "z-last")
    assert retry_error.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem


def forge_complete_peer(journal):
    full = copy.deepcopy(journal.plan)
    journal.plan["candidates"] = [full["candidates"][1]]
    forge_complete(journal)
    journal.plan["candidates"] = full["candidates"]


@pytest.mark.parametrize("code", ["RESOURCE_UNAVAILABLE", "RESOURCE_LIMIT_EXCEEDED", "TRANSIENT_SOURCE",
    "INTEGRITY_FAILED", "EXPERIMENT_IDENTITY_CONFLICT", "IDEMPOTENCY_CONFLICT", "IDENTITY_CONFLICT",
    "CONTRACT_MISMATCH", "POPULATION_COLLAPSED", "CALENDAR_UNAVAILABLE", "HOLDOUT_ACCESS_DENIED", "INVALID_EXPERIMENT_SPEC"])
def test_submitted_infrastructure_failure_is_not_a_candidate_terminal(journal, code):
    started = journal.recorder.start("2024", "2023", "z-last")
    category, retryable = {**DATA_FAILURE_CODES, **FAILURE_CODES}[code]
    problem = Problem(code=code, category=category, retryable=retryable, message="private")
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", failure=problem)
    assert caught.value.problem == Problem(code=code, category=problem.category, retryable=problem.retryable,
                                           message="search trial did not complete")
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None


def test_non_json_request_is_typed_before_admission():
    with pytest.raises(OpsError) as caught:
        trials.SearchTrials(None, None, None, None, {"invalid": object()}, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "search request must be canonical JSON").problem


def test_interrupted_start_cannot_be_relabelled_as_candidate_fit_failure(journal, monkeypatch):
    def interrupted(inner):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(journal.recorder, "_inputs", interrupted)
    with pytest.raises(RuntimeError, match="^interrupted$"):
        journal.recorder.start("2024", "2023", "z-last")
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "trial has no admitted native inputs").problem
    key = content_hash(["plan", "2024", "2023", "z-last"])
    assert journal.recorder._read("start:" + key) is not None
    assert journal.recorder._read("terminal:" + key) is None


def test_failed_candidate_still_requires_intact_admitted_cohort(journal):
    import os

    from engine.v2.contracts import ArtifactRef
    from engine.v2.foundation import from_document
    for candidate in ("z-last", "a-first"):
        journal.recorder.start("2024", "2023", candidate)
        journal.recorder.record("2024", "2023", candidate, failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
    ref_json = journal.conn.execute("SELECT ref_json FROM magnitude_search_records WHERE slot LIKE 'cohort:%'").fetchone()[0]
    ref = from_document(ArtifactRef, json.loads(ref_json))
    path = journal.store.root / ref.storage_key
    os.chmod(path, 0o644)
    path.write_bytes(b"corrupt")
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified or published").problem
    assert journal.recorder._read("selection") is None


def prepare_recording(journal, monkeypatch):
    snapshot = SnapshotRef(snapshot_id="snapshot", manifest_hash="hash", table_versions={},
        calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(), knowledge_mode_by_table={})
    journal.recorder.registration.document.update(snapshot=to_document(snapshot), holdouts={"holdout_as_of_month": "2026-01"})
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pd.DataFrame([
        {"event_id": "event", "absolute_move_pct": 4.0}]))


@pytest.mark.parametrize("phase", ["start", "readmit", "labels"])
@pytest.mark.parametrize("code", ["CONTRACT_MISMATCH", "POPULATION_COLLAPSED", "EXPERIMENT_VARIANT_FAILED",
    "CALENDAR_UNAVAILABLE", "HOLDOUT_ACCESS_DENIED", "IDENTITY_CONFLICT", "OBJECT_CORRUPT",
    "RESOURCE_UNAVAILABLE", "EVENT_NOT_FOUND", "TRANSIENT_SOURCE"])
def test_shared_admission_refusal_never_becomes_candidate_terminal(journal, monkeypatch, phase, code):
    prepare_recording(journal, monkeypatch)
    original = data_fail(code, "shared native admission failed")
    def refused(*args, **kwargs):
        raise original
    if phase != "start":
        journal.recorder.start("2024", "2023", "z-last")
    monkeypatch.setattr(trials if phase == "labels" else journal.recorder,
                        "load_magnitude_targets" if phase == "labels" else "_inputs", refused)
    with pytest.raises(DataError) as caught:
        if phase == "start":
            journal.recorder.start("2024", "2023", "z-last")
        else:
            journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert caught.value is original
    key = content_hash(["plan", "2024", "2023", "z-last"])
    assert journal.recorder._read("start:" + key) is not None
    assert journal.recorder._read("terminal:" + key) is None
    assert (journal.recorder._read("forecasts:" + key) is None) == (phase == "start")
    forge_complete_peer(journal)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "z-last")
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem


@pytest.mark.parametrize("number", [np.float32(7.25), np.float64(7.25), np.int64(7), Fraction(29, 4)])
def test_numpy_and_real_forecasts_normalize_before_recording_and_replay(journal, monkeypatch, number):
    prepare_recording(journal, monkeypatch)
    for candidate in ("z-last", "a-first"):
        journal.recorder.start("2024", "2023", candidate)
        forecasts = [{**row, "horizon_sessions": np.int64(row["horizon_sessions"])} for row in rows(number)]
        result = journal.recorder.record("2024", "2023", candidate, forecasts=forecasts)
        assert all(type(row["predicted_absolute_move_pct"]) in (int, float) and type(row["horizon_sessions"]) is int
                   for row in result["predictions"])
        assert journal.recorder.record("2024", "2023", candidate, forecasts=rows(float(number))) == result
    assert journal.recorder.freeze()["prospective_candidate_id"] == "a-first"


@pytest.mark.parametrize("field,value", [
    ("predicted_absolute_move_pct", value) for value in (
        np.bool_(True), np.float64("nan"), np.float32("inf"), np.float64(-1), Fraction(-1, 10 ** 400))
] + [("horizon_sessions", value) for value in (True, np.bool_(True), np.float64(1), np.int64(2))])
def test_numeric_normalization_keeps_refusal_guards(journal, field, value):
    forecasts = rows()
    forecasts[0][field] = value
    inner = journal.plan["folds"][0]["inner"][0]
    with pytest.raises(OpsError) as caught:
        journal.recorder._forecasts(forecasts, inner, journal.plan["horizons"])
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage").problem


@pytest.mark.parametrize("write_failure", [None, "artifact", "catalog"])
def test_candidate_scoring_refusal_preserves_original_if_terminal_write_fails(journal, monkeypatch, write_failure):
    prepare_recording(journal, monkeypatch)
    journal.recorder.start("2024", "2023", "z-last")
    original = fail("EXPERIMENT_VARIANT_FAILED", "candidate metrics must be finite")
    def scoring(*args):
        raise original
    monkeypatch.setattr(trials, "magnitude_metrics", scoring)
    append = journal.recorder._append
    def publish(slot, body, **kwargs):
        if slot.startswith("terminal:"):
            if write_failure == "artifact":
                raise fail("RESOURCE_UNAVAILABLE", "unavailable")
            if write_failure == "catalog":
                raise sqlite3.OperationalError("unavailable")
        return append(slot, body, **kwargs)
    monkeypatch.setattr(journal.recorder, "_append", publish)
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert caught.value is original
    key = content_hash(["plan", "2024", "2023", "z-last"])
    expected = None if write_failure else {"status": "refused", "problem": expected_problem("EXPERIMENT_VARIANT_FAILED")}
    assert journal.recorder._read("terminal:" + key) == expected
    monkeypatch.setattr(journal.recorder, "_append", append)
    monkeypatch.setattr(trials, "magnitude_metrics", magnitude_metrics)
    forge_complete_peer(journal)
    if write_failure:
        with pytest.raises(OpsError) as caught:
            journal.recorder.freeze()
        assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
        assert journal.recorder._read("selection") is None
    else:
        assert journal.recorder.record("2024", "2023", "z-last", forecasts=rows()) == expected
        selection = journal.recorder.freeze()["selections"][0]
        assert selection["selected_candidate_id"] == "a-first"
        assert selection["scores"][0] == {"candidate_id": "z-last", "metrics": None}
