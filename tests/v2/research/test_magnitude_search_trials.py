"""Real native inputs and durable trial transitions, with independent selection oracles."""
import copy
import hashlib
import json
import math
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    SystemClock,
    canonical_json,
    from_document,
)
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.magnitude_history_store import read_stored
from experiments import magnitude_search_evidence as evidence
from experiments import magnitude_search_trials as trials
from experiments import native_registration as native
from experiments.magnitude_search_inputs import load_inputs, publish_inputs
from experiments.magnitude_search_plan import seal_search_plan
from tests.v2.research.magnitude_trial_support import (
    CUTOFF,
    HISTORY,
    MONTH,
    ROOT,
    build_history,
    catalog_rows,
    forbid_history_rebuild,
    magnitude_dataset,
    magnitude_source,
    poison_outcomes,
    trial_cases,
    trial_forecasts,
    trial_slots,
)
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401


def _hash(value):
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


@pytest.fixture
def case(base_source, tmp_path, monkeypatch):  # noqa: F811
    conn = connect(tmp_path / "catalog.sqlite")
    base_source.conn.backup(conn)
    shutil.copytree(base_source.store.root, tmp_path / "store")
    store = ArtifactStore(tmp_path / "store")
    source = SimpleNamespace(**{**vars(base_source), "conn": conn, "store": store,
                                "repository": Repository(conn, store)})
    request = {"schema_version": "magnitude_search_request.v1",
        "inner_years": {"2024": ["2023"], "2025": ["2023", "2024"]},
        "space": {"history_ridge": {"alpha": [0.5, 2]}}, "max_candidates": 2, "max_fits": 24}
    sealed = seal_search_plan(conn, store, source.registration, request)
    recorder = trials.SearchTrials(conn, store, source.repository, source.registration, request,
                                  from_document(ArtifactRef, sealed["plan_ref"]))
    recorder.seal(clock=SystemClock())
    forbid_history_rebuild(monkeypatch)
    yield SimpleNamespace(source=source, recorder=recorder, plan=sealed["plan"], request=request,
                          catalog_before=catalog_rows(conn))
    conn.close()


def _all(case, *, fail_case=None, values=None):
    for args in trial_cases(case):
        case.recorder.start(*args)
        if args == fail_case:
            case.recorder.record(*args, failure=fail("EXPERIMENT_VARIANT_FAILED", "private diagnostic").problem)
        else:
            case.recorder.record(*args, forecasts=trial_forecasts(case, args, (values or {}).get(args, 7)))


@contextmanager
def _assert_error(code, message):
    with pytest.raises(OpsError) as caught:
        yield caught
    assert caught.value.problem == fail(code, message).problem


def test_native_complete_journal_pools_metrics_freezes_latest_and_replays(case, monkeypatch):
    poison_outcomes(monkeypatch, case.source.repository, case.source.snapshot,
            ("2023-04-03", "2025-01-07", "2025-12-03"))
    first, second = [item["candidate_id"] for item in case.plan["candidates"]]
    values = {args: (7 if args[0] == "2024" and args[2] == first else
                     0 if args[0] == "2024" else 10 if args[2] == second else
                     7 if args[1] == "2023" else 0) for args in trial_cases(case)}
    with no_fit_guard():
        _all(case, values=values)
        frozen = case.recorder.freeze()
    assert [item["selected_candidate_id"] for item in frozen["selections"]] == [first, second]
    assert frozen["prospective_candidate_id"] == second
    assert frozen["estimator_execution"] == "unverified_submitted_forecasts"
    assert frozen["evaluation_authority"] == "none"
    expected_attempts = []
    for outer in case.plan["folds"]:
        for candidate in case.plan["candidates"]:
            for inner in outer["inner"]:
                key = _hash([case.plan["plan_id"], outer["outer"]["year"], inner["year"], candidate["candidate_id"]])
                expected_attempts.append({"trial_id": key, "terminal_id": _hash(case.recorder._read("terminal:" + key))})
    assert frozen["attempts"] == expected_attempts
    actuals = {"2023": [4, 6, 8, 12], "2024": [10, 8, 20, 4]}
    for selection in frozen["selections"]:
        for score in selection["scores"]:
            errors = [abs(actual - values[selection["outer_year"], year, score["candidate_id"]])
                for year in (("2023",) if selection["outer_year"] == "2024" else ("2023", "2024"))
                for actual in actuals[year] for _ in range(3)]
            assert score["metrics"] == {"count": len(errors),
                "rmse": pytest.approx(math.sqrt(sum(error ** 2 for error in errors) / len(errors))),
                "mae": pytest.approx(sum(errors) / len(errors))}
    before = trial_slots(case)
    monkeypatch.setattr(case.recorder, "_inputs", lambda *a: pytest.fail("replay reopened native inputs"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("replay reopened labels"))
    assert case.recorder.freeze() == frozen
    assert trial_slots(case) == before
    actual = catalog_rows(case.source.conn)
    assert {key: value for key, value in actual.items() if key != "magnitude_search_records"} == {
        key: value for key, value in case.catalog_before.items() if key != "magnitude_search_records"}
    args = trial_cases(case)[0]
    assert case.recorder.record(*args, forecasts=trial_forecasts(case, args, values[args]))["status"] == "succeeded"
    assert trial_slots(case) == before


def test_shared_training_cohort_and_forecasts_committed_before_validation(case, monkeypatch):
    args, other = trial_cases(case)[:2]
    one, two = case.recorder.start(*args), case.recorder.start(*other)
    assert one["inputs"] == two["inputs"]
    train = one["inputs"]["training_inputs"]
    events = sorted(case.source.events, key=lambda event: (event["event_date"], event["event_id"]))
    training_ids = [event["event_id"] for event in events
                    if event["event_date"].date().isoformat() == "2022-06-02"]
    participant_ids = {event["event_id"] for event in events
                       if event["event_date"].date().isoformat() in
                       ("2022-06-02", "2023-01-03", "2023-06-02")}
    assert train["labels_before"] == "2022-12-22"
    assert train["qualified_event_ids"] == train["candidate_event_ids"] == training_ids
    assert one["inputs"]["training"] == [
        {"event_id": event_id, "absolute_move_pct": 2.0} for event_id in training_ids]
    stored = read_stored(case.source.repository, case.source.feature_snapshot, case.source.build["context"])
    columns = ("event_id", "context_id", "horizon_sessions",
               *case.source.registration.document["magnitude_execution"]["features"])
    expected_features = [{name: row[name] for name in columns}
                         for row in stored["rows"] if row["event_id"] in participant_ids]
    assert len(expected_features) == 18 < len(stored["rows"])
    assert one["inputs"]["features"] == expected_features
    assert one["inputs"]["stored_inputs_id"] == stored["inputs_id"]
    full_cohort = {**one["inputs"], "features": [
        row for row in stored["rows"] if row["event_id"] in participant_ids]}
    assert _hash(one["inputs"]) != _hash(full_cohort), "projected cohorts need their own content identity"
    admitted = case.recorder._read("inputs:" + one["trial_id"])
    assert admitted["inputs_id"] == publish_inputs(case.recorder.store, one["inputs"])[1]
    stored_cohort = case.recorder._read("cohort:" + admitted["cohort"])
    assert load_inputs(case.recorder.store, stored_cohort, admitted["inputs_id"]) == one["inputs"]
    original = trials.load_magnitude_targets
    def labels(*a, **kw):
        committed = case.recorder._read("forecasts:" + one["trial_id"])
        assert committed == sorted(trial_forecasts(case, args), key=lambda row: (row["event_id"], row["horizon_sessions"]))
        assert not case.source.conn.in_transaction
        assert kw["labels_before"] == "2024-01-01"
        return original(*a, **kw)
    monkeypatch.setattr(trials, "load_magnitude_targets", labels)
    result = case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    assert result["evaluation_inputs"]["labels_before"] == "2024-01-01"
    assert result["inputs_id"] == admitted["inputs_id"]


@pytest.mark.parametrize("state", ["missing", "started", "submitted"])
def test_incomplete_attempts_refuse_freeze_without_partial_selection(case, monkeypatch, state):
    args = trial_cases(case)[0]
    if state != "missing":
        case.recorder.start(*args)
    if state == "submitted":
        monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
        with pytest.raises(RuntimeError, match="crash"):
            case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    with _assert_error("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.freeze()
    assert case.recorder._read("selection") is None
    if state != "missing":
        with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
            case.recorder.start(*args)


def test_empty_outer_folds_refuse_freeze_before_native_payloads_or_selection(case, monkeypatch):
    request = copy.deepcopy(case.request)
    request["inner_years"] = {}
    recorder = trials.SearchTrials(case.source.conn, case.source.store, case.source.repository,
        case.source.registration, request, case.recorder.plan_ref)
    before = trial_slots(case)
    monkeypatch.setattr(recorder, "_inputs", lambda *a: pytest.fail("native inputs opened"))
    monkeypatch.setattr(case.source.store, "read_verified", lambda *a, **k: pytest.fail("artifact opened"))
    with _assert_error("INVALID_EXPERIMENT_SPEC",
                       "search requires explicit valid axes, inner years and count caps"):
        recorder.freeze()
    assert trial_slots(case) == before
    assert all(slot != "selection" for slot, _ in before)


@pytest.mark.parametrize("path", ["engine/v2/data/repository.py",
    "engine/v2/research/magnitude_history_store.py", "experiments/magnitude_search_execution.py"])
def test_changed_scaling_source_refuses_existing_seal_before_claim_or_inputs(case, monkeypatch, path):
    original = trials.source_closure

    def changed(*args, **kwargs):
        sources = original(*args, **kwargs)
        assert path in sources
        return {**sources, path: "sha256:" + "0" * 64}

    before = trial_slots(case)
    monkeypatch.setattr(trials, "source_closure", changed)
    monkeypatch.setattr(case.recorder, "_inputs", lambda *args: pytest.fail("changed seal opened inputs"))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search recording requires its exact sealed plan"):
        case.recorder.start(*trial_cases(case)[0])
    assert trial_slots(case) == before


def test_failure_excludes_entire_candidate_and_redacts_diagnostics(case):
    first = case.plan["candidates"][0]["candidate_id"]
    failed = ("2025", "2024", first)
    _all(case, fail_case=failed)
    frozen = case.recorder.freeze()
    assert frozen["selections"][1]["scores"][0] == {"candidate_id": first, "metrics": None}
    assert frozen["selections"][1]["scores"][1]["metrics"]["count"] == 24
    assert frozen["selections"][0]["selected_candidate_id"] == first
    assert "private diagnostic" not in str([case.recorder._read(slot) for slot, _ in trial_slots(case)])
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "terminal trial cannot accept another submission"):
        case.recorder.record(*failed, forecasts=trial_forecasts(case, failed))


def test_all_failed_refuses_and_retains_every_failure(case):
    for args in trial_cases(case):
        case.recorder.start(*args)
        case.recorder.record(*args, failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
    before = trial_slots(case)
    with _assert_error("EXPERIMENT_VARIANT_FAILED", "all candidates failed for an outer fold"):
        case.recorder.freeze()
    assert trial_slots(case) == before
    assert len([slot for slot, _ in before if slot.startswith("terminal:")]) == 6


def test_invalid_submissions_refuse_before_any_validation_labels(case, monkeypatch):
    args = trial_cases(case)[0]
    started = case.recorder.start(*args)
    for mutation in ("missing", "duplicate", "extra", "unknown", "extra_field", "boolean_horizon",
                     "boolean_value", "negative", "nan", "infinity", "overflow", "none", "both"):
        rows = trial_forecasts(case, args)
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows[-1] = rows[0].copy()
        elif mutation == "extra":
            rows.append(rows[0].copy())
        elif mutation == "unknown":
            rows[0]["event_id"] = "unplanned"
        elif mutation == "extra_field":
            rows[0]["metric"] = 0
        elif mutation == "boolean_horizon":
            next(row for row in rows if row["horizon_sessions"] == 1)["horizon_sessions"] = True
        elif mutation == "none":
            rows = None
        elif mutation != "both":
            rows[0]["predicted_absolute_move_pct"] = {"boolean_value": True, "negative": -1,
                "nan": float("nan"), "infinity": float("inf"), "overflow": 10 ** 1000}[mutation]
        monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels opened"))
        message = ("record exactly one forecast submission or failure" if mutation in ("none", "both")
                   else "trial forecasts require complete finite event-horizon coverage")
        failure = fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem if mutation == "both" else None
        with _assert_error("INVALID_EXPERIMENT_SPEC", message):
            case.recorder.record(*args, forecasts=rows, failure=failure)
        assert case.recorder._read("forecasts:" + started["trial_id"]) is None
        assert case.recorder._read("terminal:" + started["trial_id"]) is None


def test_unplanned_trial_or_missing_start_refuses_before_native_reads(case, monkeypatch):
    monkeypatch.setattr(case.recorder, "_inputs", lambda *a: pytest.fail("native inputs opened"))
    args = trial_cases(case)[0]
    with _assert_error("INVALID_EXPERIMENT_SPEC", "trial is outside the sealed search plan"):
        case.recorder.start("2024", "2024", args[2])
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "trial must be started before recording"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))


def test_source_and_plan_drift_refuse_before_native_payloads(case, monkeypatch):
    original = trials.source_closure
    def changed(*args):
        return {**original(*args), "experiments/magnitude_search_trials.py": "sha256:" + "0" * 64}
    monkeypatch.setattr(trials, "source_closure", changed)
    monkeypatch.setattr(case.recorder, "_inputs", lambda *a: pytest.fail("native inputs opened"))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search recording requires its exact sealed plan"):
        case.recorder.start(*trial_cases(case)[0])
    monkeypatch.setattr(trials, "source_closure", original)
    request = copy.deepcopy(case.request)
    request["space"]["history_ridge"]["alpha"] = [1, 3]
    sealed = seal_search_plan(case.source.conn, case.source.store, case.source.registration, request)
    other = trials.SearchTrials(case.source.conn, case.source.store, case.source.repository,
        case.source.registration, request, from_document(ArtifactRef, sealed["plan_ref"]))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        other.seal(clock=SystemClock())


def test_native_input_drift_blocks_selection_without_excluding_candidate(case, monkeypatch):
    args = trial_cases(case)[0]
    started = case.recorder.start(*args)
    case.recorder._memo.clear()
    original = case.recorder._inputs
    monkeypatch.setattr(case.recorder, "_inputs", lambda fold: {**original(fold), "stored_inputs_id": "changed"})
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("validation opened"))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "trial native inputs changed after start"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    assert case.recorder._read("terminal:" + started["trial_id"]) is None
    with _assert_error("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.freeze()


def test_native_history_binding_drift_refuses_exactly_before_readmission(case, monkeypatch):
    args = trial_cases(case)[0]
    started = case.recorder.start(*args)
    case.recorder._memo.clear()
    changed = {**case.source.registration.document["magnitude_history"], "unexpected_identity": "changed"}
    monkeypatch.setattr(evidence, "history_binding", lambda *a, **k: changed)
    monkeypatch.setattr(evidence, "preflight_magnitude_targets", lambda *a, **k: pytest.fail("preflight opened"))
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("validation opened"))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search native history binding changed"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    assert case.recorder._read("forecasts:" + started["trial_id"]) is not None
    assert case.recorder._read("terminal:" + started["trial_id"]) is None
    with _assert_error("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
        case.recorder.freeze()
    assert case.recorder._read("selection") is None
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        case.recorder.start(*args)


def test_corrupt_any_link_blocks_freeze(case):
    _all(case)
    for slot_prefix in ("start", "inputs", "cohort", "forecasts", "validation", "terminal"):
        slot, document = next(item for item in trial_slots(case) if item[0].startswith(slot_prefix + ":"))
        ref = from_document(ArtifactRef, json.loads(document))
        path = case.source.store.root / ref.storage_key
        original = path.read_bytes()
        path.chmod(0o644)
        path.write_bytes(b"corrupt")
        try:
            with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal artifact could not be verified"):
                case.recorder.freeze()
            assert case.recorder._read("selection") is None
        finally:
            path.write_bytes(original)
            path.chmod(0o444)


@pytest.mark.parametrize("statement", ["UPDATE magnitude_search_records SET slot='replacement'",
    "DELETE FROM magnitude_search_records", "INSERT OR REPLACE INTO magnitude_search_records SELECT * FROM magnitude_search_records"])
def test_catalog_rejects_rewritten_deleted_or_replaced_records(case, statement):
    before = trial_slots(case)
    with pytest.raises(sqlite3.IntegrityError, match="magnitude search records are immutable"):
        case.source.conn.execute(statement)
    assert trial_slots(case) == before


def test_failure_winning_concurrent_submission_prevents_label_read(case, monkeypatch):
    args = trial_cases(case)[0]
    started = case.recorder.start(*args)
    original = case.recorder._append
    def interleaved(slot, body, **kwargs):
        if slot.startswith("forecasts:"):
            case.recorder.record(*args, failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
        return original(slot, body, **kwargs)
    monkeypatch.setattr(case.recorder, "_append", interleaved)
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: pytest.fail("labels opened"))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search trial phase is already committed"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    assert case.recorder._read("forecasts:" + started["trial_id"]) is None
    assert case.recorder._read("terminal:" + started["trial_id"])["status"] == "failed"


def test_submitted_forecasts_cannot_be_replaced_after_interruption(case, monkeypatch):
    args = trial_cases(case)[0]
    case.recorder.start(*args)
    original = trials.load_magnitude_targets
    monkeypatch.setattr(trials, "load_magnitude_targets", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args, 0))
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search trial phase is already committed"):
        case.recorder.record(*args, failure=fail("EXPERIMENT_VARIANT_FAILED", "fit failed").problem)
    monkeypatch.setattr(trials, "load_magnitude_targets", original)
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
        case.recorder.record(*args, forecasts=trial_forecasts(case, args))


def test_artifact_publication_failure_leaves_no_admitted_start(case, monkeypatch):
    original = ArtifactStore.publish_bytes
    def unavailable(store, data, *, schema_ref):
        if json.loads(data).get("slot", "").startswith("start:"):
            raise ArtifactError("IO_ERROR", "/private/path")
        return original(store, data, schema_ref=schema_ref)
    monkeypatch.setattr(ArtifactStore, "publish_bytes", unavailable)
    before = trial_slots(case)
    with _assert_error("RESOURCE_UNAVAILABLE", "search journal artifact could not be published"):
        case.recorder.start(*trial_cases(case)[0])
    assert trial_slots(case) == before


def test_plan_reference_metadata_drift_cannot_initialize_records(case):
    case.recorder.plan_ref = replace(case.recorder.plan_ref, byte_size=1)
    with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search plan reference differs from its complete binding"):
        case.recorder.seal(clock=SystemClock())


def test_cutoff_equal_validation_label_leaves_unfinished_claim_and_blocks_freeze(tmp_path, monkeypatch):
    events, moves = magnitude_dataset()
    for row in moves:
        if row["ticker"] == "SYNTH_A" and row["event_date"] == "2023-06-02":
            row["available_as_of_date"] = "2024-01-01"
    source = magnitude_source(tmp_path, register=False, build=False, events=events, moves=moves)
    try:
        source.build, source.feature_snapshot = build_history(source, training_start="2022-01-01",
                                                     evaluation_years=["2024", "2025"])
        source.registration = native.register_native(source.conn, source.store, source.repository, source.spec,
            code_root=ROOT, as_of_month=MONTH, magnitude_training_start="2022-01-01", magnitude_labels_before=CUTOFF,
            magnitude_feature_snapshot_id=source.feature_snapshot.snapshot_id,
            magnitude_history_start=HISTORY, magnitude_fit_model="history_ridge")
        request = {"schema_version": "magnitude_search_request.v1", "inner_years": {"2024": ["2023"], "2025": ["2023"]},
            "space": {"history_ridge": {"alpha": [0.5, 2]}}, "max_candidates": 2, "max_fits": 18}
        sealed = seal_search_plan(source.conn, source.store, source.registration, request)
        recorder = trials.SearchTrials(source.conn, source.store, source.repository, source.registration,
            request, from_document(ArtifactRef, sealed["plan_ref"]))
        recorder.seal(clock=SystemClock())
        forbid_history_rebuild(monkeypatch)
        case = SimpleNamespace(source=source, plan=sealed["plan"], recorder=recorder)
        args = trial_cases(case)[0]
        started = recorder.start(*args)
        with pytest.raises(DataError) as caught:
            recorder.record(*args, forecasts=trial_forecasts(case, args))
        assert caught.value.problem == data_fail(
            "EXPERIMENT_VARIANT_FAILED", "magnitude label is not available before cutoff").problem
        assert recorder._read("terminal:" + started["trial_id"]) is None
        assert recorder._read("forecasts:" + started["trial_id"]) is not None
        with _assert_error("EXPERIMENT_VARIANT_FAILED", "search selection requires every terminal attempt"):
            recorder.freeze()
        assert recorder._read("selection") is None
        with _assert_error("EXPERIMENT_IDENTITY_CONFLICT", "search journal slot is already committed"):
            recorder.start(*args)
    finally:
        source.conn.close()
