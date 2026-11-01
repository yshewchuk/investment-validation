"""Small real-journal oracles for exact refusal, phase, and valid-hash chain guards."""
import copy
import json
import sqlite3
import tempfile
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES, ArtifactRef, Problem, SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    SystemClock,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.migrations import Migration, migrate
from engine.v2.research.magnitude_inputs import magnitude_recipe
from experiments import magnitude_search_records as records
from experiments import magnitude_search_trials as trials
from experiments.magnitude_report import magnitude_metrics
from experiments.magnitude_search_inputs import load_inputs, publish_inputs


@pytest.fixture
def journal(tmp_path, monkeypatch):
    conn = connect(tmp_path / "catalog.sqlite")
    conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY) STRICT")
    conn.execute("INSERT INTO experiment_runs VALUES ('run')")
    store = ArtifactStore(tmp_path / "store")
    # Journal-only recipe metadata; native execution fixtures install the full recipe.
    plan = {"plan_id": "plan", "horizons": [5, 3, 1], "candidates": [
        {"candidate_id": key, "recipe": {}} for key in ("z-last", "a-first")], "folds": [{
        "outer": {"year": "2024"}, "inner": [{"year": "2023", "evaluation_event_ids": ["event"],
        "evaluation_labels_before": "2024-01-01"}]}]}
    recorder = trials.SearchTrials(conn, store, None, SimpleNamespace(run_id="run", document={}), {}, None)
    monkeypatch.setattr(recorder, "_binding", lambda: (plan, {"plan_id": "plan"}))
    monkeypatch.setattr(recorder, "_inputs", lambda inner: native_inputs())
    recorder.seal(clock=SystemClock())
    yield SimpleNamespace(conn=conn, store=store, recorder=recorder, plan=plan)
    conn.close()


def rows(value=7):
    return [{"event_id": "event", "horizon_sessions": horizon, "predicted_absolute_move_pct": value}
            for horizon in [1, 3, 5]]


def native_inputs():
    preflight = {"schema_version": "research_magnitude_preflight.v1", "snapshot_id": "snapshot",
        "holdout_as_of_month": "2026-01", "random_membership_version": "random", "rolling_membership_version": "rolling",
        "population_use": "research", "purpose": "selection", "computed_moves_version": "moves",
        "event_ids": ["event"], "planned_inputs": [{"event_id": "event", "fragment_id": "fragment",
            "object_id": "object", "content_hash": "hash"}]}
    return {"preflight": {**preflight, "preflight_id": content_hash(preflight)},
            "training_inputs": {"candidate_event_ids": [], "qualified_event_ids": [], "read_inputs": [],
                                "omitted_labels": [], "target_recipe": {"secondary_diagnostics": []},
                                "holdout_exclusions": []},
            "training": [], "features": []}


def native_inputs_id(inputs=None):
    """The manifest hash of a cohort (shards are content-addressed, so any store gives the same id)."""
    with tempfile.TemporaryDirectory() as root:
        return publish_inputs(ArtifactStore(Path(root) / "store"), native_inputs() if inputs is None else inputs)[1]


def validation_evidence(value=4.0):
    preflight = native_inputs()["preflight"]
    recipe = magnitude_recipe()
    receipt = {"schema_version": "research_magnitude_targets.v1",
        **{name: preflight[name] for name in ("snapshot_id", "holdout_as_of_month", "random_membership_version",
            "rolling_membership_version", "population_use", "purpose", "computed_moves_version")},
        "labels_before": "2024-01-01", "target_recipe": recipe, "target_recipe_id": content_hash(recipe),
        "consumed_inputs": preflight["planned_inputs"], "excluded_future_events": []}
    return {"inputs_id": native_inputs_id(),
            "evaluation_inputs": {**receipt, "inputs_id": content_hash(receipt), "holdout_exclusions": []},
            "targets": [{"event_id": "event", "absolute_move_pct": value}]}


def native_labels(value=4.0):
    validation = validation_evidence(value)
    labels = pd.DataFrame(validation["targets"])
    labels.attrs.update(validation["evaluation_inputs"])
    return labels


def durable_evidence(journal):
    return (journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall(),
            {path.relative_to(journal.store.root): path.read_bytes()
             for path in (journal.store.root / "objects").rglob("*") if path.is_file()},
            journal.conn.total_changes)


def expected_problem(code):
    return to_document(Problem(code=code, category="internal" if code == "EXPERIMENT_VARIANT_FAILED"
        else "validation", retryable=False, message="search trial did not complete"))


def forge_complete(journal, mutation=None, *, validation_override=None):
    """Publish internally inconsistent evidence with valid content hashes, never corrupt bytes."""
    cohort = content_hash(["plan", "2024", "2023"])
    inputs = native_inputs()
    validation = validation_evidence() if validation_override is None else validation_override
    manifest, inputs_id = publish_inputs(journal.store, inputs)
    if mutation != "missing_cohort":
        journal.recorder._append("cohort:" + cohort, manifest)
    if mutation not in ("missing_validation", "refused_missing_validation"):
        journal.recorder._append("validation:" + cohort, validation)
    for candidate in journal.plan["candidates"]:
        key = content_hash(["plan", "2024", "2023", candidate["candidate_id"]])
        started = {"trial_id": key, "outer_year": "2024", "inner_year": "2023", "candidate": candidate}
        admitted = {"cohort": cohort, "inputs_id": inputs_id}
        submitted = rows()
        predictions = [{**row, "absolute_move_pct": 4.0} for row in submitted]
        terminal = {"status": "succeeded", "inputs_id": admitted["inputs_id"],
                    "validation_id": content_hash(validation), "evaluation_inputs": validation["evaluation_inputs"],
                    "predictions": predictions, "metrics": magnitude_metrics(predictions)}
        if mutation == "start":
            started["trial_id"] = "wrong"
        elif mutation == "cohort_link":
            admitted["cohort"] = "missing"
        elif mutation == "cohort_digest":
            admitted["inputs_id"] = terminal["inputs_id"] = "wrong"
        elif mutation == "inputs_extra":
            admitted["unexpected"] = "field"
        elif mutation == "terminal_inputs":
            terminal["inputs_id"] = "wrong"
        elif mutation == "validation_link":
            terminal["validation_id"] = "wrong"
        elif mutation == "terminal_receipt":
            terminal["evaluation_inputs"] = {}
        elif mutation == "terminal_target":
            predictions[0]["absolute_move_pct"] = 7.0
            terminal["metrics"] = magnitude_metrics(predictions)
        elif mutation == "forecasts":
            predictions[0]["predicted_absolute_move_pct"] = 8
            terminal["metrics"] = magnitude_metrics(predictions)
        elif mutation == "metrics":
            terminal["metrics"]["rmse"] = 42
        elif mutation == "stored_forecast_shape":
            submitted = rows(-1)
            terminal["predictions"] = [{**row, "absolute_move_pct": 4.0} for row in submitted]
            terminal["metrics"] = magnitude_metrics(terminal["predictions"])
        elif mutation == "stored_target_mismatch":
            submitted = rows(1e308)
            terminal["predictions"] = [{**row, "absolute_move_pct": -1e308} for row in submitted]
        elif mutation == "terminal_forecast_huge_integer":
            predictions[0]["predicted_absolute_move_pct"] = 10 ** 400
        elif mutation == "stored_target_huge_integer":
            predictions[0]["absolute_move_pct"] = 10 ** 400
        elif mutation == "status":
            terminal["status"] = "unknown"
        elif mutation == "succeeded_extra":
            terminal["unexpected"] = "field"
        elif mutation in ("failure_policy", "failure_unknown", "infrastructure_terminal", "refused_no_inputs",
                          "refused_no_forecasts", "failed_extra", "refused_extra", "refused_missing_validation"):
            problem = expected_problem("EXPERIMENT_VARIANT_FAILED")
            if mutation == "failure_policy":
                problem["retryable"] = True
            elif mutation == "failure_unknown":
                problem["code"] = "UNREGISTERED"
            elif mutation == "infrastructure_terminal":
                problem.update(code="RESOURCE_UNAVAILABLE", category="resource", retryable=True)
            terminal = {"status": "refused" if mutation.startswith("refused_") else "failed", "problem": problem}
            if mutation in ("failed_extra", "refused_extra"):
                terminal["unexpected"] = "field"
        for prefix, body in (("start", started), ("inputs", admitted), ("forecasts", submitted), ("terminal", terminal)):
            if (mutation, prefix) in (("refused_no_inputs", "inputs"), ("refused_no_forecasts", "forecasts"),
                                     ("failed_extra", "forecasts")):
                continue
            journal.recorder._append(prefix + ":" + key, body)


@pytest.mark.parametrize("mutation", ["start", "missing_cohort", "cohort_link", "cohort_digest", "inputs_extra", "terminal_inputs",
    "forecasts", "metrics", "status", "failure_policy", "failure_unknown", "infrastructure_terminal",
    "refused_no_inputs", "refused_no_forecasts", "stored_forecast_shape", "stored_target_mismatch", "succeeded_extra",
    "terminal_forecast_huge_integer", "stored_target_huge_integer", "failed_extra", "refused_extra",
    "missing_validation", "validation_link", "terminal_receipt", "terminal_target", "refused_missing_validation"])
def test_valid_hash_inconsistent_evidence_chain_refuses(journal, mutation):
    forge_complete(journal, mutation)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
    assert journal.recorder._read("selection") is None


@pytest.mark.parametrize("mutation", [None, "stored_forecast_shape"], ids=["valid", "self-consistent-negative"])
def test_self_consistent_stored_forecasts_require_valid_values(journal, monkeypatch, mutation):
    forge_complete(journal, mutation)
    key = content_hash(["plan", "2024", "2023", "z-last"])
    cohort = content_hash(["plan", "2024", "2023"])
    submitted = journal.recorder._read("forecasts:" + key)
    terminal = journal.recorder._read("terminal:" + key)
    assert submitted == rows(-1 if mutation else 7)
    assert terminal["predictions"] == [{**row, "absolute_move_pct": 4.0} for row in submitted]
    assert terminal["metrics"] == magnitude_metrics(terminal["predictions"])
    assert load_inputs(journal.store, journal.recorder._read("cohort:" + cohort), native_inputs_id()) == native_inputs()
    assert journal.recorder._read("validation:" + cohort) == validation_evidence()
    monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: pytest.fail("native inputs reopened"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels reopened"))
    before = durable_evidence(journal)
    if mutation:
        # A valid public resubmission reaches the immutable-terminal guard; negative
        # resubmissions are rejected earlier by ordinary forecast admission.
        for operation, message in (
            (journal.recorder.freeze, "search trial evidence chain differs"),
            (lambda: journal.recorder.record("2024", "2023", "z-last", forecasts=rows()),
             "terminal trial cannot accept another submission"),
        ):
            with pytest.raises(OpsError) as caught:
                operation()
            assert type(caught.value) is OpsError
            assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", message).problem
            assert journal.recorder._read("selection") is None
            assert durable_evidence(journal) == before
    else:
        assert journal.recorder.record("2024", "2023", "z-last", forecasts=rows()) == terminal
        assert durable_evidence(journal) == before
        frozen = journal.recorder.freeze()
        assert frozen["prospective_candidate_id"] == "a-first"
        before = durable_evidence(journal)
        assert journal.recorder.freeze() == frozen
        assert durable_evidence(journal) == before


def test_equal_metrics_choose_lowest_id_even_when_candidates_are_reversed(journal):
    forge_complete(journal)
    frozen = journal.recorder.freeze()
    assert frozen["selections"][0]["selected_candidate_id"] == "a-first"
    assert frozen["prospective_candidate_id"] == "a-first"
    assert frozen["selections"][0]["scores"][0]["metrics"] == frozen["selections"][0]["scores"][1]["metrics"]


@pytest.mark.parametrize("cross_fold", [False, True], ids=["matching-fold", "other-valid-fold"])
def test_valid_hash_cohort_link_requires_the_exact_fold(journal, monkeypatch, cross_fold):
    prepare_recording(journal, monkeypatch)
    # Nested searches can reuse the same inner evaluation fold in different outer folds.
    other_fold = copy.deepcopy(journal.plan["folds"][0])
    other_fold["outer"]["year"] = "2025"
    other_fold["inner"][0]["evaluation_labels_before"] = "2025-01-01"
    journal.plan["folds"].append(other_fold)
    key = content_hash(["plan", "2024", "2023", "z-last"])
    own_cohort = content_hash(["plan", "2024", "2023"])
    other_cohort = content_hash(["plan", "2025", "2023"])
    append = journal.recorder._append
    def publish(slot, body, **kwargs):
        if cross_fold and slot == "inputs:" + key:
            body = {**body, "cohort": other_cohort}
        return append(slot, body, **kwargs)
    monkeypatch.setattr(journal.recorder, "_append", publish)
    def labels(*args, labels_before, **kwargs):
        result = native_labels()
        result.attrs["labels_before"] = labels_before
        result.attrs["inputs_id"] = content_hash({name: value for name, value in result.attrs.items()
                                                 if name not in ("inputs_id", "holdout_exclusions")})
        return result
    monkeypatch.setattr(trials, "load_magnitude_targets", labels)
    failure = fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem
    terminal = {"status": "failed", "problem": expected_problem(failure.code)}
    for year in ("2025", "2024"):
        started = journal.recorder.start(year, "2023", "z-last")
        # A failed trial still binds its own cohort, without reading a target receipt.
        journal.recorder._append("terminal:" + started["trial_id"], terminal)
        journal.recorder.start(year, "2023", "a-first")
        journal.recorder.record(year, "2023", "a-first", forecasts=rows())
    assert own_cohort != other_cohort
    for year, cohort in (("2024", own_cohort), ("2025", other_cohort)):
        assert load_inputs(journal.store, journal.recorder._read("cohort:" + cohort), native_inputs_id()) == native_inputs()
        validation = validation_evidence()
        validation["evaluation_inputs"] = dict(labels(labels_before=year + "-01-01").attrs)
        assert journal.recorder._read("validation:" + cohort) == validation
    assert journal.recorder._read("inputs:" + key) == {
        "cohort": other_cohort if cross_fold else own_cohort, "inputs_id": native_inputs_id()}
    before = durable_evidence(journal)
    monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: pytest.fail("native inputs reopened"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels reopened"))
    if cross_fold:
        for operation in (journal.recorder.freeze,
                          lambda: journal.recorder.record("2024", "2023", "z-last", failure=failure)):
            with pytest.raises(OpsError) as caught:
                operation()
            assert type(caught.value) is OpsError
            assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
            assert journal.recorder._read("selection") is None
            assert durable_evidence(journal) == before
    else:
        assert journal.recorder.record("2024", "2023", "z-last", failure=failure) == terminal
        assert durable_evidence(journal) == before
        frozen = journal.recorder.freeze()
        assert [selection["selected_candidate_id"] for selection in frozen["selections"]] == ["a-first", "a-first"]
        assert all(selection["scores"][0] == {"candidate_id": "z-last", "metrics": None}
                   for selection in frozen["selections"])
        assert len(frozen["attempts"]) == 4
        assert frozen["prospective_candidate_id"] == "a-first"
        before = durable_evidence(journal)
        assert journal.recorder.freeze() == frozen
        assert durable_evidence(journal) == before


@pytest.mark.parametrize("phase", ["forecasts", "validation", "cohort"])
def test_stored_evidence_read_failure_retains_resource_policy(journal, monkeypatch, phase):
    forge_complete(journal)
    key = content_hash(["plan", "2024", "2023"] + (["z-last"] if phase == "forecasts" else []))
    encoded = journal.conn.execute("SELECT ref_json FROM magnitude_search_records WHERE slot=?",
                                   (phase + ":" + key,)).fetchone()[0]
    target = from_document(ArtifactRef, json.loads(encoded))
    read = ArtifactStore.read_verified
    def unavailable(store, ref):
        if ref == target:
            raise ArtifactError("IO_ERROR", "private path")
        return read(store, ref)
    monkeypatch.setattr(ArtifactStore, "read_verified", unavailable)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "search journal artifact could not be verified").problem
    assert journal.recorder._read("selection") is None


def test_cohort_byte_limit_preserves_unfinished_claim(journal, monkeypatch):
    original_limit = records.MAX_BYTES
    monkeypatch.setattr(records, "MAX_BYTES", 1024)
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "z-last")
    assert caught.value.problem == fail("RESOURCE_LIMIT_EXCEEDED", "search journal record exceeds its byte limit").problem
    key = content_hash(["plan", "2024", "2023", "z-last"])
    assert journal.recorder._read("start:" + key) is not None
    assert journal.recorder._read("terminal:" + key) is None
    monkeypatch.setattr(records, "MAX_BYTES", original_limit)
    forge_complete_peer(journal)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None


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
    journal.recorder._memo.clear()
    journal.recorder._published.clear()  # a fresh instance has no published cohorts and re-admits
    monkeypatch.setattr(journal.recorder, "_inputs", lambda inner: {**native_inputs(), "native_inputs": "changed"})
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
    (phase, code) for phase in ("cohort", "inputs", "forecasts", "validation", "terminal")
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
        snapshot = SnapshotRef(snapshot_id="snapshot", manifest_hash="hash", table_versions={},
            calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(),
            knowledge_mode_by_table={})
        journal.recorder.registration.document.update(snapshot=to_document(snapshot), holdouts={"holdout_as_of_month": "2026-01"})
        def labels(*args, **kwargs):
            if phase == "labels":
                raise data_fail("RESOURCE_UNAVAILABLE", "temporary label read failure")
            return native_labels()
        monkeypatch.setattr(trials, "load_magnitude_targets", labels)
        with pytest.raises((OpsError, DataError)) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    expected_code = "RESOURCE_UNAVAILABLE" if artifact_code == "IO_ERROR" else "EXPERIMENT_IDENTITY_CONFLICT"
    message = ("temporary native input failure" if phase == "native_inputs" else
               "temporary label read failure" if phase == "labels" else
               "search journal artifact could not be published")
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


@pytest.mark.parametrize("invalid", [object(), float("nan"), float("inf"), -float("inf")],
                         ids=["object", "nan", "infinity", "negative-infinity"])
def test_non_json_request_is_typed_before_admission(invalid):
    with pytest.raises(OpsError) as caught:
        trials.SearchTrials(None, None, None, None, {"invalid": invalid}, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "search request must be canonical JSON").problem


@pytest.mark.parametrize("document", [{1: "value"}, {True: "value"}, {None: "value"},
    {"nested": [{2: "value"}]}, {"nested": (1, 2)}])
def test_json_coercion_is_not_canonical_request_admission(document):
    with pytest.raises(OpsError) as caught:
        trials.SearchTrials(None, None, None, None, document, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "search request must be canonical JSON").problem


def test_canonical_request_is_detached():
    request = {"nested": [{"value": 1.5, "enabled": True, "optional": None}]}
    recorder = trials.SearchTrials(None, None, None, None, request, None)
    assert recorder.request == request
    request["nested"][0]["value"] = 9
    assert recorder.request["nested"][0]["value"] == 1.5


@pytest.mark.parametrize("submission", ["both", "neither"])
@pytest.mark.parametrize("state", ["outside_plan", "missing_start", "started"])
def test_record_argument_refusal_respects_admission_and_start_precedence(journal, monkeypatch, state, submission):
    args = ("2024", "2023", "z-last")
    if state == "started":
        journal.recorder.start(*args)
    elif state == "outside_plan":
        args = ("2024", "2022", "z-last")
    def forbidden(*args, **kwargs):
        pytest.fail("invalid submission reached native reads or publication")
    monkeypatch.setattr(journal.recorder, "_inputs", forbidden)
    monkeypatch.setattr(journal.recorder, "_append", forbidden)
    monkeypatch.setattr(trials, "load_magnitude_targets", forbidden)
    kwargs = {"forecasts": rows(), "failure": fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem}
    before = journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall()
    with pytest.raises(OpsError) as caught:
        journal.recorder.record(*args, **(kwargs if submission == "both" else {}))
    expected = {"outside_plan": ("INVALID_EXPERIMENT_SPEC", "trial is outside the sealed search plan"),
        "missing_start": ("EXPERIMENT_IDENTITY_CONFLICT", "trial must be started before recording"),
        "started": ("INVALID_EXPERIMENT_SPEC", "record exactly one forecast submission or failure")}
    assert caught.value.problem == fail(*expected[state]).problem
    assert journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall() == before


@pytest.mark.parametrize("submission", ["candidate_failure", "forecasts", "malformed_forecasts", "noncandidate_failure"])
def test_interrupted_start_preserves_content_validation_before_input_binding(journal, monkeypatch, submission):
    def interrupted(inner):
        raise RuntimeError("interrupted")
    monkeypatch.setattr(journal.recorder, "_inputs", interrupted)
    with pytest.raises(RuntimeError, match="^interrupted$"):
        journal.recorder.start("2024", "2023", "z-last")
    kwargs = {"candidate_failure": {"failure": fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem},
        "forecasts": {"forecasts": rows()}, "malformed_forecasts": {"forecasts": rows(-1)},
        "noncandidate_failure": {"failure": fail("RESOURCE_UNAVAILABLE", "private failure").problem}}[submission]
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", **kwargs)
    expected = (fail("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage")
        if submission == "malformed_forecasts" else fail("RESOURCE_UNAVAILABLE", "search trial did not complete")
        if submission == "noncandidate_failure" else fail("EXPERIMENT_IDENTITY_CONFLICT", "trial has no admitted native inputs"))
    assert caught.value.problem == expected.problem
    key = content_hash(["plan", "2024", "2023", "z-last"])
    assert journal.recorder._read("start:" + key) is not None
    assert journal.recorder._read("terminal:" + key) is None


def test_failed_candidate_still_requires_intact_admitted_cohort(journal):
    for candidate in ("z-last", "a-first"):
        journal.recorder.start("2024", "2023", candidate)
        journal.recorder.record("2024", "2023", candidate, failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
    ref_json = journal.conn.execute("SELECT ref_json FROM magnitude_search_records WHERE slot LIKE 'cohort:%'").fetchone()[0]
    ref = from_document(ArtifactRef, json.loads(ref_json))
    path = journal.store.root / ref.storage_key
    path.chmod(0o644)
    path.write_bytes(b"corrupt")
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified").problem
    assert journal.recorder._read("selection") is None


def prepare_recording(journal, monkeypatch):
    snapshot = SnapshotRef(snapshot_id="snapshot", manifest_hash="hash", table_versions={},
        calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(), knowledge_mode_by_table={})
    journal.recorder.registration.document.update(snapshot=to_document(snapshot), holdouts={"holdout_as_of_month": "2026-01"})
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: native_labels())


@pytest.mark.parametrize("mutation", [None, "actuals", "receipt", "missing_receipt", "validation_id", "extra_field"])
def test_terminal_only_rewrites_cannot_change_the_true_winner(journal, monkeypatch, mutation):
    prepare_recording(journal, monkeypatch)
    append = journal.recorder._append
    rewritten = content_hash(["plan", "2024", "2023", "a-first"])
    def publish(slot, body, **kwargs):
        if slot == "terminal:" + rewritten and mutation:
            body = copy.deepcopy(body)
            if mutation == "actuals":
                for row in body["predictions"]:
                    row["absolute_move_pct"] = 7.0
                body["metrics"] = magnitude_metrics(body["predictions"])
                assert body["metrics"]["rmse"] == 0  # This used to change the winner by the ID tie-break.
            elif mutation == "receipt":
                body["evaluation_inputs"]["labels_before"] = "2099-01-01"
            elif mutation == "missing_receipt":
                del body["evaluation_inputs"]
            elif mutation == "validation_id":
                body["validation_id"] = "wrong"
            else:
                body["unexpected"] = "field"
        return append(slot, body, **kwargs)
    monkeypatch.setattr(journal.recorder, "_append", publish)
    for candidate, value in (("z-last", 4), ("a-first", 7)):
        journal.recorder.start("2024", "2023", candidate)
        journal.recorder.record("2024", "2023", candidate, forecasts=rows(value))
    validation = journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"]))
    assert validation == validation_evidence()
    monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: pytest.fail("native inputs reopened"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels reopened"))
    if mutation:
        for operation in (lambda: journal.recorder.record("2024", "2023", "a-first", forecasts=rows(7)),
                          journal.recorder.freeze):
            with pytest.raises(OpsError) as caught:
                operation()
            assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
        assert journal.recorder._read("selection") is None
    else:
        frozen = journal.recorder.freeze()
        assert frozen["prospective_candidate_id"] == "z-last"
        assert [score["metrics"]["rmse"] for score in frozen["selections"][0]["scores"]] == [0.0, 3.0]
        assert journal.recorder.record("2024", "2023", "a-first", forecasts=rows(7))["metrics"]["rmse"] == 3.0


@pytest.mark.parametrize("field", ["schema_version", "snapshot_id", "holdout_as_of_month", "random_membership_version",
    "rolling_membership_version", "population_use", "purpose", "computed_moves_version", "labels_before",
    "target_recipe", "target_recipe_id", "consumed_inputs", "excluded_future_events", "holdout_exclusions", "inputs_id"])
def test_valid_hash_receipt_rewrite_is_bound_to_admitted_preflight(journal, field):
    validation = validation_evidence()
    receipt = validation["evaluation_inputs"]
    receipt[field] = ({} if field == "target_recipe" else [] if field == "consumed_inputs" else
                      ["unexpected"] if field in ("excluded_future_events", "holdout_exclusions") else "different")
    if field != "inputs_id":
        receipt["inputs_id"] = content_hash({name: value for name, value in receipt.items()
                                            if name not in ("inputs_id", "holdout_exclusions")})
    forge_complete(journal, validation_override=validation)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
    assert journal.recorder._read("selection") is None


@pytest.mark.parametrize("mutation", ["inputs_id", "extra_field", "missing_targets"])
def test_validation_envelope_is_bound_to_the_admitted_cohort(journal, mutation):
    validation = validation_evidence()
    if mutation == "inputs_id":
        validation["inputs_id"] = "different"
    elif mutation == "extra_field":
        validation["unexpected"] = "field"
    else:
        del validation["targets"]
    forge_complete(journal, validation_override=validation)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
    assert journal.recorder._read("selection") is None


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unplanned", "extra_field", "negative", "boolean", "huge"])
def test_validation_targets_require_exact_typed_event_coverage(journal, mutation):
    validation = validation_evidence()
    targets = validation["targets"]
    if mutation == "missing":
        targets.clear()
    elif mutation == "duplicate":
        targets.append(targets[0].copy())
    elif mutation == "unplanned":
        targets[0]["event_id"] = "unknown"
    elif mutation == "extra_field":
        targets[0]["other"] = 4
    else:
        targets[0]["absolute_move_pct"] = {"negative": -1, "boolean": True, "huge": 10 ** 400}[mutation]
    forge_complete(journal, validation_override=validation)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search trial evidence chain differs").problem
    assert journal.recorder._read("selection") is None


@pytest.mark.parametrize("mutation", [None, "receipt_order", "target_order", "duplicate_membership", "unplanned_membership"])
def test_validation_preserves_admitted_object_and_target_order(journal, monkeypatch, mutation):
    prepare_recording(journal, monkeypatch)
    inner = journal.plan["folds"][0]["inner"][0]
    inner["evaluation_event_ids"] = ["later-event", "event"]  # Plan order need not be native chronological order.
    inputs = native_inputs()
    preflight = inputs["preflight"]
    preflight["event_ids"].append("later-event")
    preflight["planned_inputs"].append({"event_id": "later-event", "fragment_id": "later-fragment",
                                       "object_id": "later-object", "content_hash": "later-hash"})
    preflight["preflight_id"] = content_hash({key: value for key, value in preflight.items() if key != "preflight_id"})
    monkeypatch.setattr(journal.recorder, "_inputs", lambda inner: copy.deepcopy(inputs))
    labels = pd.DataFrame([{"event_id": "event", "absolute_move_pct": 4.0},
                           {"event_id": "later-event", "absolute_move_pct": 5.0}])
    labels.attrs.update(validation_evidence()["evaluation_inputs"])
    labels.attrs["consumed_inputs"] = copy.deepcopy(preflight["planned_inputs"])
    if mutation == "receipt_order":
        labels.attrs["consumed_inputs"].reverse()
    elif mutation == "target_order":
        labels = labels.iloc[::-1].reset_index(drop=True)
    elif mutation == "duplicate_membership":
        labels.attrs["consumed_inputs"][1] = labels.attrs["consumed_inputs"][0].copy()
    elif mutation == "unplanned_membership":
        labels.attrs["consumed_inputs"][1]["object_id"] = "unplanned-object"
    labels.attrs["inputs_id"] = content_hash({key: value for key, value in labels.attrs.items()
                                             if key not in ("inputs_id", "holdout_exclusions")})
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: labels)
    started = journal.recorder.start("2024", "2023", "z-last")
    forecasts = rows() + [{**row, "event_id": "later-event"} for row in rows()]
    if mutation is None:
        result = journal.recorder.record("2024", "2023", "z-last", forecasts=forecasts)
        assert result["evaluation_inputs"]["consumed_inputs"] == preflight["planned_inputs"]
    else:
        with pytest.raises(OpsError) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=forecasts)
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search validation evidence differs").problem
        assert journal.recorder._read("terminal:" + started["trial_id"]) is None
        assert journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"])) is None


@pytest.mark.parametrize("membership", ["complete", "omitted", "duplicated_extra", "duplicated_replacement"])
def test_record_requires_exact_preflight_evaluation_membership(journal, monkeypatch, membership):
    prepare_recording(journal, monkeypatch)
    journal.plan["folds"][0]["inner"][0]["evaluation_event_ids"] = ["event", "later-event"]
    inputs = native_inputs()
    preflight = inputs["preflight"]
    preflight["event_ids"].append("later-event")
    planned = preflight["planned_inputs"]
    planned.append({"event_id": "later-event", "fragment_id": "later-fragment",
                    "object_id": "later-object", "content_hash": "later-hash"})
    if membership == "omitted":
        planned.pop()
    elif membership == "duplicated_extra":
        planned.append(planned[0].copy())
    elif membership == "duplicated_replacement":
        planned[1] = planned[0].copy()
    preflight["preflight_id"] = content_hash({name: value for name, value in preflight.items() if name != "preflight_id"})
    # Receipt and targets agree with the malformed preflight, isolating its membership guard.
    labels = pd.DataFrame([{"event_id": row["event_id"], "absolute_move_pct": 4.0} for row in planned])
    labels.attrs.update(validation_evidence()["evaluation_inputs"])
    labels.attrs["consumed_inputs"] = copy.deepcopy(planned)
    labels.attrs["inputs_id"] = content_hash({name: value for name, value in labels.attrs.items()
                                             if name not in ("inputs_id", "holdout_exclusions")})
    monkeypatch.setattr(journal.recorder, "_inputs", lambda inner: copy.deepcopy(inputs))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: labels.copy(deep=True))
    started = journal.recorder.start("2024", "2023", "z-last")
    forecasts = rows() + [{**row, "event_id": "later-event"} for row in rows()]
    before = durable_evidence(journal)
    if membership == "complete":
        result = journal.recorder.record("2024", "2023", "z-last", forecasts=forecasts)
        assert result["status"] == "succeeded"
        assert result["metrics"] == {"count": 6, "rmse": 3.0, "mae": 3.0}
        assert result["evaluation_inputs"]["consumed_inputs"] == planned
        journal.recorder.start("2024", "2023", "a-first")
        journal.recorder.record("2024", "2023", "a-first", forecasts=forecasts)
    else:
        with pytest.raises(OpsError) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=forecasts)
        assert type(caught.value) is OpsError
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search validation evidence differs").problem
        forecast_slot = "forecasts:" + started["trial_id"]
        assert journal.recorder._read(forecast_slot) == forecasts
        assert journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"])) is None
        assert journal.recorder._read("terminal:" + started["trial_id"]) is None
        assert journal.recorder._read("selection") is None
        after = durable_evidence(journal)
        assert [row for row in after[0] if row[1] != forecast_slot] == before[0]
        assert {path: after[1][path] for path in before[1]} == before[1]
        assert len(after[1]) == len(before[1]) + 1
        assert after[2] == before[2] + 1
    monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: pytest.fail("native inputs reopened"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels reopened"))
    if membership == "complete":
        frozen = journal.recorder.freeze()
        assert frozen["prospective_candidate_id"] == "a-first"
        before = durable_evidence(journal)
        assert journal.recorder.record("2024", "2023", "z-last", forecasts=forecasts) == result
        assert journal.recorder.freeze() == frozen
        assert durable_evidence(journal) == before
    else:
        before = durable_evidence(journal)
        for operation, code, message in (
            (lambda: journal.recorder.start("2024", "2023", "z-last"),
             "EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"),
            (lambda: journal.recorder.record("2024", "2023", "z-last", forecasts=forecasts),
             "EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"),
            (journal.recorder.freeze, "EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"),
        ):
            with pytest.raises(OpsError) as caught:
                operation()
            assert caught.value.problem == fail(code, message).problem
            assert journal.recorder._read("selection") is None
            assert durable_evidence(journal) == before


def test_drifted_labels_cannot_overwrite_a_committed_validation_authority(journal, monkeypatch):
    # Later cohort trials reuse the journaled validation, so label drift after the
    # first trial is never re-read and cannot refuse or rewrite the authority.
    prepare_recording(journal, monkeypatch)
    journal.recorder.start("2024", "2023", "z-last")
    journal.recorder.record("2024", "2023", "z-last", forecasts=rows(4))
    journal.recorder.start("2024", "2023", "a-first")
    changed = native_labels(7)
    reloads = []
    def drifted(*args, **kwargs):
        reloads.append(True)
        return changed
    monkeypatch.setattr(trials, "load_magnitude_targets", drifted)
    result = journal.recorder.record("2024", "2023", "a-first", forecasts=rows(7))
    assert reloads == []
    assert result["status"] == "succeeded"
    assert result["validation_id"] == content_hash(validation_evidence())
    assert result["predictions"] == [{**row, "absolute_move_pct": 4.0} for row in rows(7)]
    assert result["metrics"] == {"count": 3, "rmse": 3.0, "mae": 3.0}
    assert journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"])) == validation_evidence()
    frozen = journal.recorder.freeze()
    assert frozen["prospective_candidate_id"] == "z-last"
    before = durable_evidence(journal)
    monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: pytest.fail("native inputs reopened"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels reopened"))
    assert journal.recorder.record("2024", "2023", "a-first", forecasts=rows(7)) == result
    assert journal.recorder.freeze() == frozen
    assert durable_evidence(journal) == before


@pytest.mark.parametrize("field", ["excluded_future_events", "holdout_exclusions"])
def test_native_loader_exclusions_refuse_before_validation_or_terminal_publication(journal, monkeypatch, field):
    prepare_recording(journal, monkeypatch)
    started = journal.recorder.start("2024", "2023", "z-last")
    labels = native_labels()
    labels.attrs[field] = (["event"] if field == "excluded_future_events" else
                           [{"event_id": "event", "memberships": ["random"]}])
    labels.attrs["inputs_id"] = content_hash({key: value for key, value in labels.attrs.items()
                                             if key not in ("inputs_id", "holdout_exclusions")})
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: labels)
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search validation evidence differs").problem
    assert journal.recorder._read("forecasts:" + started["trial_id"]) == rows()
    assert journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"])) is None
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None
    forge_complete_peer(journal)
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem


def test_crash_after_validation_commit_cannot_complete_or_restart_attempt(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    started = journal.recorder.start("2024", "2023", "z-last")
    def crash(*args):
        raise RuntimeError("interrupted after validation")
    monkeypatch.setattr(trials, "magnitude_metrics", crash)
    with pytest.raises(RuntimeError, match="^interrupted after validation$"):
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"])) == validation_evidence()
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None
    for operation in (lambda: journal.recorder.start("2024", "2023", "z-last"),
                      lambda: journal.recorder.record("2024", "2023", "z-last", forecasts=rows())):
        with pytest.raises(OpsError) as caught:
            operation()
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None


def test_validation_association_failure_rolls_back_and_leaves_forecast_claim(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    started = journal.recorder.start("2024", "2023", "z-last")
    journal.conn.execute("""CREATE TRIGGER reject_validation BEFORE INSERT ON magnitude_search_records
        WHEN NEW.slot LIKE 'validation:%' BEGIN SELECT RAISE(ABORT, 'validation commit failed'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="^validation commit failed$"):
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert not journal.conn.in_transaction
    assert journal.recorder._read("forecasts:" + started["trial_id"]) == rows()
    assert journal.recorder._read("validation:" + content_hash(["plan", "2024", "2023"])) is None
    assert journal.recorder._read("terminal:" + started["trial_id"]) is None
    with pytest.raises(OpsError) as caught:
        journal.recorder.freeze()
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt").problem
    assert journal.recorder._read("selection") is None


@pytest.mark.parametrize("other_value", [4.0, 5.0])
def test_interleaved_writers_cannot_establish_different_validation_authorities(journal, monkeypatch, other_value):
    slot = "validation:" + content_hash(["plan", "2024", "2023"])
    path = journal.conn.execute("PRAGMA database_list").fetchone()[2]
    peer = connect(path)
    publish = ArtifactStore.publish_bytes
    pending = True
    def interleave(store, payload, *, schema_ref):
        nonlocal pending
        ref = publish(store, payload, schema_ref=schema_ref)
        if pending and json.loads(payload).get("slot") == slot:
            pending = False
            assert not journal.conn.in_transaction
            records.append_record(peer, journal.store, "run", slot, validation_evidence(other_value))
        return ref
    monkeypatch.setattr(ArtifactStore, "publish_bytes", interleave)
    try:
        if other_value == 4.0:
            assert journal.recorder._append(slot, validation_evidence()) == validation_evidence()
        else:
            with pytest.raises(OpsError) as caught:
                journal.recorder._append(slot, validation_evidence())
            assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
        assert journal.recorder._read(slot) == validation_evidence(other_value)
        assert journal.conn.execute("SELECT count(*) FROM magnitude_search_records WHERE slot=?", (slot,)).fetchone()[0] == 1
    finally:
        peer.close()


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
        journal.recorder._memo.clear()
        journal.recorder._published.clear()  # a fresh instance has no published cohorts and re-admits
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
    before_retry = journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall()
    def no_read(*args, **kwargs):
        pytest.fail("an unfinished claim must not reopen native inputs or labels")
    monkeypatch.setattr(journal.recorder, "_inputs", no_read)
    monkeypatch.setattr(trials, "load_magnitude_targets", no_read)
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "z-last")
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    if phase != "start":
        with pytest.raises(OpsError) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    assert journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall() == before_retry


def test_preclaim_validation_refusals_do_not_consume_an_attempt_or_submission(journal, monkeypatch):
    prepare_recording(journal, monkeypatch)
    before_start = journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall()
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "unplanned")
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "trial is outside the sealed search plan").problem
    assert journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall() == before_start
    started = journal.recorder.start("2024", "2023", "z-last")
    before_forecasts = journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall()
    with monkeypatch.context() as guard:
        guard.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("invalid forecasts read labels"))
        with pytest.raises(OpsError) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=rows(-1))
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage").problem
    assert journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall() == before_forecasts
    assert journal.recorder._read("forecasts:" + started["trial_id"]) is None
    result = journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert result["status"] == "succeeded"
    assert result["metrics"] == {"count": 3, "rmse": 3.0, "mae": 3.0}


def test_public_metrics_refuse_empty_rows_and_preserve_nonempty_arithmetic():
    with pytest.raises(OpsError) as caught:
        magnitude_metrics([])
    assert type(caught.value) is OpsError
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "magnitude metrics require scored rows").problem
    assert magnitude_metrics([{"predicted_absolute_move_pct": 7.0, "absolute_move_pct": 4.0}]) == {
        "count": 1, "rmse": 3.0, "mae": 3.0}


def test_public_metrics_refuse_finite_operands_whose_error_overflows():
    assert magnitude_metrics([{"predicted_absolute_move_pct": 1e308, "absolute_move_pct": 0.0}]) == {
        "count": 1, "rmse": 1e308, "mae": 1e308}
    with pytest.raises(OpsError) as caught:
        magnitude_metrics([{"predicted_absolute_move_pct": 1e308, "absolute_move_pct": -1e308}])
    assert type(caught.value) is OpsError
    assert caught.value.problem == fail("EXPERIMENT_VARIANT_FAILED", "magnitude metrics must be finite").problem


@pytest.mark.parametrize("number", [np.float32(7.25), np.float64(7.25), np.int64(7), Fraction(29, 4)])
def test_numpy_and_real_forecasts_normalize_before_recording_and_replay(journal, monkeypatch, number):
    prepare_recording(journal, monkeypatch)
    for candidate in ("z-last", "a-first"):
        journal.recorder.start("2024", "2023", candidate)
        forecasts = [{**row, "horizon_sessions": np.int64(row["horizon_sessions"])} for row in rows(number)]
        result = journal.recorder.record("2024", "2023", candidate, forecasts=forecasts)
        assert all(type(row["predicted_absolute_move_pct"]) in (int, float) and type(row["horizon_sessions"]) is int
                   for row in result["predictions"])
        assert all(row["predicted_absolute_move_pct"] == float(number) for row in result["predictions"])
        assert journal.recorder.record("2024", "2023", candidate, forecasts=rows(float(number))) == result
    assert journal.recorder.freeze()["prospective_candidate_id"] == "a-first"


@pytest.mark.parametrize("field,value", [
    ("predicted_absolute_move_pct", value) for value in (
        np.bool_(True), np.float64("nan"), np.float32("inf"), np.float64(-1), Fraction(-1, 10 ** 400))
] + [("horizon_sessions", value) for value in (True, np.bool_(True), np.float64(1))])
def test_numeric_normalization_keeps_refusal_guards(journal, field, value):
    forecasts = rows()
    forecasts[0][field] = value
    inner = journal.plan["folds"][0]["inner"][0]
    with pytest.raises(OpsError) as caught:
        journal.recorder._forecasts(forecasts, inner, journal.plan["horizons"])
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC", "trial forecasts require complete finite event-horizon coverage").problem


@pytest.mark.parametrize("write_failure", [None, "artifact", "catalog", "filesystem", "integrity"])
def test_candidate_scoring_refusal_preserves_original_if_terminal_write_fails(journal, monkeypatch, write_failure):
    prepare_recording(journal, monkeypatch)
    journal.recorder.start("2024", "2023", "z-last")
    original = fail("EXPERIMENT_VARIANT_FAILED", "candidate metrics must be finite")
    def scoring(*args):
        raise original
    monkeypatch.setattr(trials, "magnitude_metrics", scoring)
    append = journal.recorder._append
    publication_error = {None: None, "artifact": fail("RESOURCE_UNAVAILABLE", "unavailable"),
        "catalog": sqlite3.OperationalError("unavailable"), "filesystem": OSError("unavailable"),
        "integrity": fail("EXPERIMENT_IDENTITY_CONFLICT", "journal conflict")}[write_failure]
    def publish(slot, body, **kwargs):
        if slot.startswith("terminal:") and publication_error is not None:
            raise publication_error
        return append(slot, body, **kwargs)
    monkeypatch.setattr(journal.recorder, "_append", publish)
    with pytest.raises(OpsError) as caught:
        journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
    assert caught.value is original
    assert caught.value.__cause__ is publication_error
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
        with pytest.raises(OpsError) as caught:
            journal.recorder.start("2024", "2023", "z-last")
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
        with pytest.raises(OpsError) as caught:
            journal.recorder.record("2024", "2023", "z-last", forecasts=rows())
        assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed").problem
    else:
        before = journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall()
        monkeypatch.setattr(journal.recorder, "_inputs", lambda *a: pytest.fail("native inputs reopened"))
        monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels reopened"))
        monkeypatch.setattr(trials, "magnitude_metrics", lambda *a: pytest.fail("refusal rescored"))
        assert journal.recorder.record("2024", "2023", "z-last", forecasts=rows()) == expected
        assert journal.conn.execute("SELECT * FROM magnitude_search_records ORDER BY slot").fetchall() == before
        monkeypatch.setattr(trials, "magnitude_metrics", magnitude_metrics)
        selection = journal.recorder.freeze()["selections"][0]
        assert selection["selected_candidate_id"] == "a-first"
        assert selection["scores"][0] == {"candidate_id": "z-last", "metrics": None}


@pytest.mark.parametrize("operation", ["start", "record", "freeze"])
@pytest.mark.parametrize("schema", ["unmigrated", "older", "newer", "checksum", "unsealed"])
def test_journal_admission_refuses_before_native_or_evidence_reads(tmp_path, monkeypatch, operation, schema):
    conn = connect(tmp_path / "catalog.sqlite")
    try:
        conn.execute("CREATE TABLE experiment_runs(run_id TEXT PRIMARY KEY) STRICT")
        conn.execute("INSERT INTO experiment_runs VALUES ('run')")
        if schema != "unmigrated":
            migrations = records.MIGRATIONS
            if schema == "newer":
                migrations += (Migration(version=2, name="future", statements=()),)
            migrate(conn, records.OWNER, migrations, clock=SystemClock())
        if schema == "older":
            conn.execute("DELETE FROM schema_versions WHERE owner=?", (records.OWNER,))
        elif schema == "checksum":
            conn.execute("UPDATE schema_versions SET checksum='different' WHERE owner=?", (records.OWNER,))
        recorder = trials.SearchTrials(conn, ArtifactStore(tmp_path / "store"), None,
                                       SimpleNamespace(run_id="run"), {}, None)
        def forbidden(*args, **kwargs):
            pytest.fail("admission opened native or evidence payloads")
        monkeypatch.setattr(recorder, "_binding", (lambda: ({}, {"plan_id": "plan"})) if schema == "unsealed" else forbidden)
        monkeypatch.setattr(recorder, "_inputs", forbidden)
        monkeypatch.setattr(ArtifactStore, "read_verified", forbidden)
        monkeypatch.setattr(trials, "load_magnitude_targets", forbidden)
        before = list(conn.iterdump())
        with pytest.raises(OpsError) as caught:
            if operation == "freeze":
                recorder.freeze()
            elif operation == "start":
                recorder.start("2024", "2023", "candidate")
            else:
                recorder.record("2024", "2023", "candidate", forecasts=rows())
        if schema == "unsealed":
            expected = fail("EXPERIMENT_IDENTITY_CONFLICT", "search recording requires its exact sealed plan")
        elif schema == "newer":
            expected = fail("INTEGRITY_FAILED", "magnitude_search schema is newer than this code supports",
                details={"reason": "schema_newer", "owner": records.OWNER, "applied": 2, "supported": 1})
        elif schema == "checksum":
            expected = fail("INTEGRITY_FAILED", "magnitude_search migration differs from the one applied",
                details={"reason": "checksum_mismatch", "owner": records.OWNER, "version": 1})
        else:
            expected = fail("INTEGRITY_FAILED", "magnitude_search schema is missing migrations",
                details={"reason": "schema_older", "owner": records.OWNER, "missing": [1]})
        assert caught.value.problem == expected.problem
        assert list(conn.iterdump()) == before
    finally:
        conn.close()
