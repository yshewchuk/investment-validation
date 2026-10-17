"""Exact-attempt collection, immutable native reconciliation and smoke isolation."""
import csv
import errno
import inspect
import json
import os
import sqlite3
import stat
from dataclasses import asdict, replace
from pathlib import Path
from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.errors import fail as data_fail
from engine.v2.foundation import (
    ArtifactError,
    SystemClock,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops import submission
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import Outcome, commit_attempt, complete_cancel, request_cancel
from engine.v2.ops.migrations import checksum
from engine.v2.ops.recovery import begin_epoch, expire_leases, reconcile_attempt
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.schema import MIGRATIONS, OWNER
from engine.v2.ops.stages import registry
from engine.v2.ops.supervisor import Service
from experiments import native_outcomes, native_registration
from experiments import native_sweep_collection as collection
from experiments import v2_gate_collect as cli
from experiments import v2_gate_sweep as sweep
from tests.ops_support import POLICY, REGISTRY, TEST_POLICY, FakeClock, sample
from tests.v2.ops.gate_sweep_support import (
    call_sweep,
    forbid_effect,
    replace_checkout_source,
    scientific_rows,
    sealed_sweep,
    set_terminal_result,
    submitted_job,
    sweep_checkout,
    sweep_request,
    terminal_case,
)
from tests.v2.ops.gate_sweep_support import (
    fixed_sweep_code as fixed_sweep_code,
)
from tests.v2.ops.gate_sweep_support import (
    fixed_sweep_code_maps as fixed_sweep_code_maps,
)
from tests.v2.ops.native_bounds_support import BoundedSqlRows
from tests.v2.research.test_gate_report import MONTH, REPO, build_gate_spec
from tests.v2.research.test_gate_report import source as source


def _collect(source, *, code_root=REPO, document=None, **kwargs):
    result = call_sweep(source, collection.collect_sweep, document, code_root=code_root, **kwargs)
    assert type(result["index"]["attempted_variants"]) is int
    assert all(type(arm["attempted_variants"]) is int for arm in result["index"]["variants"])
    return result


def _run(source, *, code_root=REPO, document=None, **kwargs):
    service = Service(source.conn, source.store.root, registry(), TEST_POLICY,
                      clock=SystemClock(), code_source=code_root)
    def tick(_delay):
        service.tick()
        sleep(0.05)
    try:
        service.start()
        with patch.object(cli, "sleep", tick):
            # One heavy slot sequences the real children; a parallel CI run
            # can exceed 90s for the whole roster. Keep a finite test-only bound.
            return call_sweep(source, cli.run_sweep, document, code_root=code_root, timeout_seconds=300, **kwargs)
    finally:
        service.close()


def _evidence(source):
    return [json.loads(row[0]) for row in source.conn.execute(
        "SELECT evidence_json FROM experiment_runs ORDER BY run_id")]


def _rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _registrations(source):
    return [native_registration.NativeRegistration(item["run_id"], canonical_json(item["binding"]).encode())
            for item in sealed_sweep(source)["registrations"]]


def _claimer(source):
    clock = FakeClock()
    epoch = begin_epoch(source.conn, clock=clock, boot_id="synthetic", pid=1)
    supervisor = Supervisor(epoch, "synthetic")
    def claim():
        return claim_next(source.conn, policy=TEST_POLICY, sample=sample(clock),
                          supervisor=supervisor, clock=clock, registry=registry())
    return clock, claim


def _finish(source, claim, clock, *, problem=None):
    """Commit a synthetic wire receipt; real fitting has separate child acceptance."""
    if problem is not None:
        return commit_attempt(source.conn, claim.attempt_id, claim.fence,
            Outcome(succeeded=False, process_state="exited", failure=problem), clock=clock)
    registration = next(reg for reg in _registrations(source) if reg.variant_id == claim.spec.parameters["variant_id"])
    result = {"schema_version": "native_gate_result.v1.0", "variant_id": registration.variant_id,
              "run_id": registration.run_id, "attempted_variants": 1,
              "provenance": registration.document, "synthetic_collection_fixture": True}
    ref = source.store.publish_bytes(canonical_json(result).encode(), schema_ref="native_gate_result.v1.0")
    def effects(conn):
        register_artifact(conn, ref, claim.attempt_id, clock)
        conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)", (claim.attempt_id, "native_gate_result", ref.artifact_id))
    return commit_attempt(source.conn, claim.attempt_id, claim.fence,
        Outcome(succeeded=True, process_state="exited"), clock=clock, effects=effects)


def _complete(source, jobs):
    clock, claim = _claimer(source)
    attempts = {}
    for _ in jobs:
        attempt = claim()
        _finish(source, attempt, clock)
        attempts[attempt.job_id] = attempt.attempt_id
    assert set(attempts) == {job.job_id for job in jobs}
    return attempts


@pytest.mark.usefixtures("fixed_sweep_code")
def test_fixed_source_inputs_return_independent_real_maps(fixed_sweep_code_maps):
    closures, manifest = fixed_sweep_code_maps
    for module in (collection, native_registration, sweep):
        for entries, expected in closures.items():
            first = module.source_closure(REPO, entries)
            assert first == expected and first is not expected
            first.clear()
            assert module.source_closure(REPO, entries) == expected
    first = sweep.worker_source_manifest(REPO)
    assert first == manifest and first is not manifest
    first.clear()
    assert sweep.worker_source_manifest(REPO) == manifest


@pytest.mark.usefixtures("fixed_sweep_code")
@pytest.mark.parametrize("target", ["other-root", "unknown-entry", "worker-root"])
def test_fixed_source_inputs_refuse_unexpected_discovery(tmp_path, target):
    with pytest.raises(AssertionError, match="fixed source fixture received"):
        if target == "worker-root":
            sweep.worker_source_manifest(tmp_path)
        else:
            collection.source_closure(tmp_path if target == "other-root" else REPO,
                                      ("experiments/v2_gate.py" if target == "other-root" else "unknown.py",))


def test_real_run_records_independent_arms_and_exact_replay(source, tmp_path, monkeypatch):
    ledger = tmp_path / "LEDGER.csv"
    call_sweep(source)
    result = _run(source, ledger_path=ledger)
    index = result["index"]
    assert index["complete"] is True
    assert index["planned_variants"] == index["attempted_variants"] == 2
    assert index["recording_mode"] == "recorded"
    assert index["roster_id"] == content_hash(sealed_sweep(source))
    ledger_intent = content_hash(str(ledger.resolve()))
    intent = {"schema_version": "native_gate_collection.v1.0", "roster_id": content_hash(sealed_sweep(source)),
              "recording_mode": "recorded", "ledger_destination": ledger_intent,
              "collector_source": collection._implementation(REPO)}
    assert index["run_id"] == content_hash(intent)
    assert index["primary_comparison"] == ["candidate", "control"]
    assert [arm["arm_id"] for arm in index["variants"]] == ["candidate", "control"]
    assert len({arm["attempt_id"] for arm in index["variants"]}) == 2
    assert sorted(row["stage"] for row in _rows(ledger)) == ["ran", "ran"]
    assert json.loads(source.store.read_verified(from_document(ArtifactRef, result["index_ref"]))) == index
    for arm, profit, registration in zip(index["variants"], (1.2, 2.8), _registrations(source), strict=True):
        assert arm["state"] == "succeeded" and arm["problem"] is arm["worker_problem"] is None
        report = source.store.read_verified(from_document(ArtifactRef, arm["report_ref"])).decode()
        assert "Recording mode: recorded evaluation" in report
        metrics = json.loads(report.split("```json\n")[1].split("\n```")[0])
        assert metrics["all_candidates"]["mean_mark_pnl"] == pytest.approx(profit)
        assert metrics["provenance"] == registration.document
        assert metrics["variant_id"] == arm["variant_id"]
        outcome = json.loads(source.store.read_verified(from_document(ArtifactRef, arm["outcome_ref"])))
        receipt = json.loads(source.store.read_verified(from_document(ArtifactRef, arm["completion_ref"])))
        assert outcome["schema_version"] == "native_experiment_outcome.v1.0"
        assert outcome["run_id"] == registration.run_id and outcome["variant_id"] == registration.variant_id
        assert outcome["attempted_variants"] == 1 and outcome["failure_code"] is None
        assert outcome["report_ref"] == arm["report_ref"]
        assert receipt == {"schema_version": "native_experiment_completion.v1.0",
                           "outcome_ref": arm["outcome_ref"], "recording_mode": "recorded",
                           "report_ref": arm["report_ref"], "ledger_destination": content_hash(str(ledger.resolve())),
                           "ledger_row_hash": content_hash(outcome["ledger_row"])}
        assert (source.store.root / "native_reports" / registration.run_id / "recorded" / "REPORT.md").read_text() == report
    before, ledger_bytes = scientific_rows(source), ledger.read_bytes()
    monkeypatch.setattr(cli, "sleep", forbid_effect)
    assert call_sweep(source, cli.run_sweep, ledger_path=ledger) == result
    assert scientific_rows(source) == before and ledger.read_bytes() == ledger_bytes
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_real_smoke_run_never_calls_native_outcomes_or_ledger(source, monkeypatch):
    call_sweep(source)
    for name in ("publish_native_outcome", "replay_native_outcome", "export_native_report", "ledger_append"):
        monkeypatch.setattr(native_outcomes, name, forbid_effect)
    result = _run(source, no_ledger=True)
    assert result["index"]["complete"] is True
    assert result["index"]["recording_mode"] == "smoke"
    assert result["index"]["attempted_variants"] == 2
    for arm in result["index"]["variants"]:
        assert arm["outcome_ref"] is arm["completion_ref"] is None
        report = source.store.read_verified(from_document(ArtifactRef, arm["report_ref"])).decode()
        assert "no-ledger smoke; not recorded" in report
    for evidence in _evidence(source):
        assert not {"native_smoke", "native_smoke_receipt", "native_outcome", "native_outcome_receipt"} & evidence.keys()
    assert not list(source.root.rglob("*.csv")) and not list(source.root.rglob("REPORT.md"))
    assert _collect(source, no_ledger=True) == result
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 2


def test_real_mixed_success_unexpected_failure_and_holdout_reconcile_once(source, tmp_path):
    checkout = sweep_checkout(tmp_path)
    worker = checkout / "experiments/native_gate_evaluation.py"
    text = worker.read_text()
    marker = "            result = gate_result(Repository(conn, store), registration, spec)"
    assert text.count(marker) == 1
    replace_checkout_source(worker, text.replace(marker, marker + "\n            if spec.primary_arm_id == 'candidate':\n"
                                        "                raise RuntimeError('synthetic private programming failure')"))
    gate = checkout / "experiments/gate_report.py"
    text = gate.read_text()
    marker = '    calendar = trading_calendar_from_snapshot(repository, snapshot, extend_days=0)'
    assert text.count(marker) == 1
    replace_checkout_source(gate, text.replace(marker, "    if binding['execution_plan']['arms'] == ['refused']:\n"
                                       "        raise fail('HOLDOUT_ACCESS_DENIED', 'synthetic holdout refusal')\n" + marker))
    document = sweep_request()
    document["variants"].append(json.loads(canonical_json(asdict(build_gate_spec(0.5, "refused")))))
    call_sweep(source, document=document, code_root=checkout)
    ledger = tmp_path / "LEDGER.csv"
    result = _run(source, document=document, code_root=checkout, ledger_path=ledger)
    index = result["index"]
    intent_ref = next(evidence["native_sweep_recorded_intent"] for evidence in _evidence(source)
                      if "native_sweep_recorded_intent" in evidence)
    intent = json.loads(source.store.read_verified(from_document(ArtifactRef, intent_ref)))
    assert intent["collector_source"] == collection._COLLECTOR_SOURCES
    assert intent["collector_source"] != collection.source_closure(checkout, collection._COLLECTOR_ENTRIES)
    assert index["complete"] is False and result["index_ref"] is not None
    assert index["planned_variants"] == index["attempted_variants"] == 3
    failed, success, refused = index["variants"]
    assert [arm["state"] for arm in index["variants"]] == ["failed", "succeeded", "refused"]
    assert failed["problem"]["code"] == "EXPERIMENT_VARIANT_FAILED"
    assert failed["worker_problem"]["code"] == "WORKER_FAILED"
    assert failed["worker_problem"]["retryable"] is True and failed["problem"]["retryable"] is False
    assert refused["problem"]["code"] == "HOLDOUT_ACCESS_DENIED"
    assert refused["problem"] == refused["worker_problem"]
    assert failed["report_ref"] is refused["report_ref"] is None and success["report_ref"] is not None
    for arm in (failed, refused):
        assert source.conn.execute("SELECT COUNT(*) FROM attempt_outputs WHERE attempt_id=?",
                                   (arm["attempt_id"],)).fetchone()[0] == 0
    assert sorted(row["stage"] for row in _rows(ledger)) == ["failed", "ran", "refused"]
    assert "synthetic private programming failure" not in canonical_json(index)
    before, ledger_bytes = scientific_rows(source), ledger.read_bytes()
    assert _collect(source, document=document, code_root=checkout, ledger_path=ledger) == result
    assert scientific_rows(source) == before and ledger.read_bytes() == ledger_bytes
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 3


@pytest.mark.usefixtures("fixed_sweep_code")
def test_pending_collection_binds_mode_and_does_not_freeze_index(source, tmp_path):
    call_sweep(source)
    smoke = _collect(source, no_ledger=True)
    recorded = _collect(source, ledger_path=tmp_path / "LEDGER.csv")
    for result in (smoke, recorded):
        assert result["index_ref"] is None and result["index"]["complete"] is False
        assert result["index"]["attempted_variants"] == 0
        assert [arm["state"] for arm in result["index"]["variants"]] == ["queued", "queued"]
    assert smoke["index"]["run_id"] != recorded["index"]["run_id"]
    for result, mode, ledger in ((smoke, "smoke", None),
                                 (recorded, "recorded", tmp_path / "LEDGER.csv")):
        intent = {"schema_version": "native_gate_collection.v1.0",
                  "roster_id": content_hash(sealed_sweep(source)), "recording_mode": mode,
                  "ledger_destination": None if ledger is None else content_hash(str(ledger.resolve())),
                  "collector_source": collection._implementation(REPO)}
        assert result["index"]["run_id"] == content_hash(intent)
    assert len({arm["job_id"] for result in (smoke, recorded) for arm in result["index"]["variants"]}) == 2
    assert not list(source.root.rglob("*.csv"))
    assert all(not any(key.endswith("_index") for key in evidence) for evidence in _evidence(source))
    before = scientific_rows(source)
    jobs_before = [tuple(row) for row in source.conn.execute("SELECT * FROM jobs ORDER BY job_id")]
    changes = source.conn.total_changes
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, ledger_path=tmp_path / "different.csv")
    assert scientific_rows(source) == before and not (tmp_path / "different.csv").exists()
    assert [tuple(row) for row in source.conn.execute("SELECT * FROM jobs ORDER BY job_id")] == jobs_before
    assert source.conn.total_changes == changes
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


@pytest.mark.usefixtures("fixed_sweep_code")
def test_retry_wait_receipt_stays_pending_without_recording(source, tmp_path, monkeypatch):
    call_sweep(source)
    call_sweep(source, sweep.submit_sweep)
    clock, claim = _claimer(source)
    attempt = claim()
    # Native workers currently have a once-only policy. Model a persisted retry
    # receipt explicitly; collection must not assume every failed attempt ended its job.
    _finish(source, attempt, clock, problem=fail("RESOURCE_UNAVAILABLE", "synthetic retry").problem)
    source.conn.execute("UPDATE jobs SET state='retry_wait' WHERE job_id=?", (attempt.job_id,))
    before = scientific_rows(source)
    for name in ("publish_native_outcome", "replay_native_outcome", "export_native_report", "ledger_append"):
        monkeypatch.setattr(native_outcomes, name, forbid_effect)
    ledger = tmp_path / "LEDGER.csv"
    result = _collect(source, ledger_path=ledger)
    assert result["index_ref"] is None and result["index"]["complete"] is False
    assert result["index"]["attempted_variants"] == 1
    arm = next(arm for arm in result["index"]["variants"] if arm["job_id"] == attempt.job_id)
    assert arm["state"] == "retry_wait" and arm["attempted_variants"] == 1
    assert all(arm[key] is None for key in ("attempt_id", "problem", "worker_problem", "report_ref", "outcome_ref", "completion_ref"))
    assert not ledger.exists()
    assert all(not {"native_outcome", "native_outcome_receipt", "native_sweep_recorded_index"} & item.keys()
               for item in _evidence(source))
    saved = scientific_rows(source)
    assert saved != before  # Only the collection intent is newly bound.
    assert _collect(source, ledger_path=ledger) == result and scientific_rows(source) == saved
    job = submission.get_job(source.conn, attempt.job_id)
    assert job.state == "retry_wait" and job.attempt_count == 1 and job.failure.code == "RESOURCE_UNAVAILABLE"


@pytest.mark.usefixtures("fixed_sweep_code")
def test_recorded_r4_cannot_later_be_replaced_by_success(source, tmp_path):
    call_sweep(source)
    call_sweep(source, sweep.submit_sweep)
    clock, claim = _claimer(source)
    failed = claim()
    diagnostic = source.store.publish_bytes(b"synthetic worker diagnostic", schema_ref="worker_diagnostic.v1.0")
    worker_problem = replace(fail("WORKER_FAILED", "synthetic worker failure").problem,
                             diagnostic_ref=diagnostic.artifact_id)
    _finish(source, failed, clock, problem=worker_problem)
    ledger = tmp_path / "LEDGER.csv"
    result = _collect(source, ledger_path=ledger)
    arm = next(arm for arm in result["index"]["variants"] if arm["job_id"] == failed.job_id)
    assert arm["attempt_id"] == failed.attempt_id
    assert arm["worker_problem"] == to_document(worker_problem)
    assert arm["worker_problem"]["diagnostic_ref"] == diagnostic.artifact_id
    assert source.store.read_verified(diagnostic) == b"synthetic worker diagnostic"
    assert arm["problem"]["code"] == "EXPERIMENT_VARIANT_FAILED"
    assert [row["stage"] for row in _rows(ledger)] == ["failed"]
    registration = next(item for item in _registrations(source) if item.variant_id == arm["variant_id"])
    before, rows = scientific_rows(source), ledger.read_bytes()
    with pytest.raises(OpsError) as error:
        native_outcomes.publish_native_outcome(source.conn, source.store, registration,
            report="A later successful report", attempted=True, ledger_path=ledger,
            max_evidence_bytes=collection.MAX_GATE_RESULT_BYTES)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert scientific_rows(source) == before and ledger.read_bytes() == rows
    assert _collect(source, ledger_path=ledger) == result


def test_changed_collector_source_and_inputs_cannot_rebind_saved_intent(source, tmp_path, monkeypatch):
    checkout = sweep_checkout(tmp_path)
    monkeypatch.setattr(collection, "_COLLECTOR_ROOT", checkout)
    monkeypatch.setattr(collection, "_COLLECTOR_SOURCES", collection.source_closure(checkout, collection._COLLECTOR_ENTRIES))
    call_sweep(source, code_root=checkout)
    _collect(source, code_root=checkout, no_ledger=True)
    before = scientific_rows(source)
    collector = checkout / "experiments/native_sweep_collection.py"
    collector.write_text(collector.read_text() + "\n# synthetic source drift\n")
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, code_root=checkout, no_ledger=True)
    assert scientific_rows(source) == before
    collector.write_bytes(REPO.joinpath("experiments/native_sweep_collection.py").read_bytes())
    document = sweep_request()
    document["variants"][0]["economic_params"]["fill"] = 0.25
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, code_root=checkout, document=document, no_ledger=True)
    assert scientific_rows(source) == before
    assert _collect(source, code_root=checkout, no_ledger=True)["index_ref"] is None


@pytest.mark.parametrize("entry", ["native_sweep_collection.py", "v2_gate_collect.py"])
def test_caller_coordinator_entry_must_match_installed_source(source, tmp_path, entry):
    checkout = sweep_checkout(tmp_path)
    call_sweep(source, code_root=checkout)
    before = scientific_rows(source)
    path = checkout / "experiments" / entry
    path.write_text(path.read_text() + "\n# synthetic caller-only source drift\n")
    assert collection._COLLECTOR_ROOT == REPO
    ledger = tmp_path / "LEDGER.csv"
    with pytest.raises(OpsError) as error:
        _collect(source, code_root=checkout, ledger_path=ledger)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert scientific_rows(source) == before and not ledger.exists()
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.parametrize("failure", ["ledger", "receipt", "export", "index"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_interrupted_publication_reconciles_without_duplicate_rows(source, tmp_path, monkeypatch, failure):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    attempts = _complete(source, jobs)
    ledger = tmp_path / "LEDGER.csv"
    original = (native_outcomes.ledger_append if failure == "ledger" else native_outcomes._publish
                if failure == "receipt" else native_outcomes.export_native_report if failure == "export"
                else collection._bind)
    def interrupted(*args, **kwargs):
        if failure == "receipt" and args[1].get("schema_version") != "native_experiment_completion.v1.0":
            return original(*args, **kwargs)
        if failure == "index" and not args[3].endswith("_index"):
            return original(*args, **kwargs)
        raise OSError("synthetic publication interruption")
    with monkeypatch.context() as patch:
        target, name = ((native_outcomes, "ledger_append") if failure == "ledger" else
                        (native_outcomes, "_publish") if failure == "receipt" else
                        (native_outcomes, "export_native_report") if failure == "export" else
                        (collection, "_bind"))
        patch.setattr(target, name, interrupted)
        with pytest.raises(OpsError) as error:
            _collect(source, ledger_path=ledger)
        assert error.value.code == "RESOURCE_UNAVAILABLE"
        assert error.value.problem.retryable is True
    assert not any(any(key.endswith("_index") for key in evidence) for evidence in _evidence(source))
    saved_evidence = _evidence(source)
    result = _collect(source, ledger_path=ledger)
    for saved, reconciled in zip(saved_evidence, _evidence(source), strict=True):
        assert {key: reconciled[key] for key in saved} == saved
    index = result["index"]
    assert index["complete"] is True and result["index_ref"] is not None
    assert index["attempted_variants"] == 2
    assert {arm["job_id"]: arm["attempt_id"] for arm in index["variants"]} == attempts
    assert json.loads(source.store.read_verified(from_document(ArtifactRef, result["index_ref"]))) == index
    expected_rows = []
    for arm, registration in zip(index["variants"], _registrations(source), strict=True):
        assert arm["state"] == "succeeded" and arm["problem"] is arm["worker_problem"] is None
        assert arm["attempted_variants"] == 1
        outcome = json.loads(source.store.read_verified(from_document(ArtifactRef, arm["outcome_ref"])))
        receipt = json.loads(source.store.read_verified(from_document(ArtifactRef, arm["completion_ref"])))
        assert outcome["run_id"] == registration.run_id and outcome["variant_id"] == registration.variant_id
        assert type(outcome["attempted_variants"]) is int and outcome["attempted_variants"] == 1
        assert outcome["failure_code"] is None and outcome["report_ref"] == arm["report_ref"]
        assert receipt == {"schema_version": "native_experiment_completion.v1.0",
                           "outcome_ref": arm["outcome_ref"], "recording_mode": "recorded",
                           "report_ref": arm["report_ref"], "ledger_destination": content_hash(str(ledger.resolve())),
                           "ledger_row_hash": content_hash(outcome["ledger_row"])}
        report = source.store.read_verified(from_document(ArtifactRef, arm["report_ref"]))
        assert (source.store.root / "native_reports" / registration.run_id / "recorded" / "REPORT.md").read_bytes() == report
        expected_rows.append(outcome["ledger_row"])
    assert _rows(ledger) == expected_rows
    before, ledger_bytes = scientific_rows(source), ledger.read_bytes()
    assert _collect(source, ledger_path=ledger) == result
    assert scientific_rows(source) == before and ledger.read_bytes() == ledger_bytes
    assert dict(source.conn.execute("SELECT job_id, attempt_id FROM attempts")) == attempts


@pytest.mark.usefixtures("fixed_sweep_code")
def test_collection_preserves_native_publication_runtime_error_policy(source, tmp_path, monkeypatch):
    call_sweep(source)
    _complete(source, call_sweep(source, sweep.submit_sweep))
    ledger = tmp_path / "LEDGER.csv"
    def interrupted(*args, **kwargs):
        raise RuntimeError("private native publication interruption")
    with monkeypatch.context() as patch:
        patch.setattr(native_outcomes, "ledger_append", interrupted)
        with pytest.raises(OpsError) as error:
            _collect(source, ledger_path=ledger)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and error.value.problem.retryable is True
    assert "private native" not in str(error.value)
    assert not ledger.exists()
    assert all("native_sweep_recorded_index" not in item for item in _evidence(source))
    result = _collect(source, ledger_path=ledger)
    assert result["index"]["complete"] is True and len(_rows(ledger)) == 2


@pytest.mark.usefixtures("fixed_sweep_code")
def test_preexisting_different_arm_outcome_is_not_blindly_replayed(source, tmp_path):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    _complete(source, jobs)
    ledger = tmp_path / "LEDGER.csv"
    registration = _registrations(source)[0]
    native_outcomes.publish_native_outcome(source.conn, source.store, registration,
        report="A different earlier single-arm report", attempted=True, ledger_path=ledger)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, ledger_path=ledger)
    saved = native_outcomes.replay_native_outcome(source.conn, source.store, registration, ledger_path=ledger)
    assert source.store.read_verified(from_document(ArtifactRef, saved["outcome"]["report_ref"])) == b"A different earlier single-arm report"
    assert len(_rows(ledger)) == 1
    assert not any(any(key.endswith("_index") for key in evidence) for evidence in _evidence(source))


@pytest.mark.parametrize("damage", ["identity", "wrong-schema", "matched-wrong-schema", "wrong-name", "no-output", "terminal-attempt"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_terminal_output_is_exact_and_missing_result_is_r4(source, tmp_path, damage):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    _complete(source, jobs)
    attempt = source.conn.execute("SELECT * FROM attempts WHERE job_id=?", (jobs[0].job_id,)).fetchone()
    output = source.conn.execute("SELECT * FROM attempt_outputs WHERE attempt_id=?", (attempt["attempt_id"],)).fetchone()
    if damage in ("identity", "wrong-schema", "matched-wrong-schema"):
        ref = artifact(source.conn, source.store, output["artifact_id"])
        result = json.loads(source.store.read_verified(ref))
        if damage == "identity":
            result["variant_id"] = "sha256:" + "0" * 64
        if damage == "matched-wrong-schema":
            result["schema_version"] = "incorrect.v1.0"
        replacement = source.store.publish_bytes(canonical_json(result).encode(),
            schema_ref="incorrect.v1.0" if damage in {"wrong-schema", "matched-wrong-schema"} else ref.schema_ref)
        register_artifact(source.conn, replacement, None, SystemClock())
        source.conn.execute("UPDATE attempt_outputs SET artifact_id=? WHERE attempt_id=?",
                            (replacement.artifact_id, attempt["attempt_id"]))
    elif damage == "wrong-name":
        source.conn.execute("UPDATE attempt_outputs SET name='different_output' WHERE attempt_id=?", (attempt["attempt_id"],))
    elif damage == "no-output":
        source.conn.execute("DELETE FROM attempt_outputs WHERE attempt_id=?", (attempt["attempt_id"],))
    else:
        source.conn.execute("UPDATE jobs SET attempt_count=2, fence=2 WHERE job_id=?", (jobs[0].job_id,))
    ledger = tmp_path / "LEDGER.csv"
    if damage == "no-output":
        result = _collect(source, ledger_path=ledger)
        assert result["index"]["complete"] is False
        assert result["index"]["variants"][0]["problem"]["code"] == "EXPERIMENT_VARIANT_FAILED"
        assert sorted(row["stage"] for row in _rows(ledger)) == ["failed", "ran"]
    else:
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            _collect(source, ledger_path=ledger)
        assert not ledger.exists()


@pytest.mark.parametrize("arguments,code", [(["--purpose", "final"], "HOLDOUT_ACCESS_DENIED"),
                                           ([], "INVALID_EXPERIMENT_SPEC"),
                                           (["--no-ledger", "--ledger", "unused.csv"], "INVALID_EXPERIMENT_SPEC")])
def test_cli_refuses_purpose_or_missing_mode_before_catalog(tmp_path, capsys, monkeypatch, arguments, code):
    monkeypatch.setattr(cli, "_catalog", forbid_effect)
    absent = tmp_path / "absent"
    args = ["collect", "--roster", str(absent), "--catalog", str(absent), "--store-root", str(absent),
            "--code-root", str(REPO), "--as-of-month", MONTH] + arguments
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": code}
    assert not absent.exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "30"])
def test_run_rejects_bad_deadline_before_inputs(timeout, monkeypatch):
    monkeypatch.setattr(cli, "collect_sweep", forbid_effect)
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        cli.run_sweep(None, None, None, {}, code_root=REPO, timeout_seconds=timeout)


@pytest.mark.parametrize("state", ["queued-cancel", "live-cancel", "recovered-cancel", "lease-lost"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_cancellation_and_recovery_use_exact_fenced_terminal_attempt(source, tmp_path, state):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    clock, claim = _claimer(source)
    if state == "queued-cancel":
        request_cancel(source.conn, jobs[0].job_id, None, clock=clock)
        failed_job, failed_attempt, attempted = jobs[0].job_id, None, 0
    else:
        attempt = claim()
        failed_job, failed_attempt, attempted = attempt.job_id, attempt.attempt_id, 1
        if state != "lease-lost":
            request_cancel(source.conn, attempt.job_id, attempt.attempt_id, clock=clock)
        if state == "live-cancel":
            complete_cancel(source.conn, attempt.job_id, process_state="verified_dead", clock=clock)
        else:
            clock.advance(121)
            assert expire_leases(source.conn, clock=clock) == [attempt.attempt_id]
            reconcile_attempt(source.conn, attempt.attempt_id, process_state="verified_dead", clock=clock)
        row = source.conn.execute("SELECT fence FROM jobs WHERE job_id=?", (failed_job,)).fetchone()
        assert row[0] > attempt.fence
    successful = claim()
    _finish(source, successful, clock)
    ledger = tmp_path / "LEDGER.csv"
    result = _collect(source, ledger_path=ledger)
    assert result["index_ref"] is not None and result["index"]["complete"] is False
    assert result["index"]["attempted_variants"] == 1 + attempted
    arm = next(item for item in result["index"]["variants"] if item["job_id"] == failed_job)
    assert {item["job_id"]: item["attempt_id"] for item in result["index"]["variants"]} == {
        failed_job: failed_attempt, successful.job_id: successful.attempt_id}
    assert arm["attempted_variants"] == attempted
    assert arm["state"] == "refused"
    assert arm["problem"] == arm["worker_problem"]
    assert arm["problem"]["code"] == ("LEASE_LOST" if state == "lease-lost" else "CANCELLED")
    assert arm["problem"]["retryable"] is (state == "lease-lost")
    assert arm["outcome_ref"] is arm["completion_ref"] is arm["report_ref"] is None
    assert [row["stage"] for row in _rows(ledger)] == ["ran"]
    assert _collect(source, ledger_path=ledger) == result and len(_rows(ledger)) == 1


@pytest.mark.usefixtures("fixed_sweep_code")
def test_partial_collection_keeps_successful_sibling_and_registered_retryable_refusal(source, tmp_path):
    call_sweep(source)
    call_sweep(source, sweep.submit_sweep)
    clock, claim = _claimer(source)
    _finish(source, claim(), clock)
    ledger = tmp_path / "LEDGER.csv"
    partial = _collect(source, ledger_path=ledger)
    assert partial["index_ref"] is None and partial["index"]["complete"] is False
    assert partial["index"]["attempted_variants"] == 1 and len(_rows(ledger)) == 1
    successful = next(arm for arm in partial["index"]["variants"] if arm["state"] == "succeeded")
    _finish(source, claim(), clock, problem=fail("RESOURCE_UNAVAILABLE", "synthetic resource refusal").problem)
    final = _collect(source, ledger_path=ledger)
    assert final["index_ref"] is not None and final["index"]["complete"] is False
    assert final["index"]["attempted_variants"] == 2
    assert next(arm for arm in final["index"]["variants"] if arm["state"] == "succeeded") == successful
    refusal = next(arm for arm in final["index"]["variants"] if arm["state"] == "refused")
    assert refusal["problem"] == refusal["worker_problem"]
    assert refusal["problem"]["code"] == "RESOURCE_UNAVAILABLE" and refusal["problem"]["retryable"] is True
    assert refusal["outcome_ref"] is None and len(_rows(ledger)) == 1
    assert _collect(source, ledger_path=ledger) == final


@pytest.mark.usefixtures("fixed_sweep_code")
def test_blocked_job_is_terminal_and_freezes_an_incomplete_index(source, tmp_path):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    problem = fail("DEPENDENCY_FAILED", "upstream dependency did not succeed").problem
    # Exercise the terminal receipt that ops dependency propagation can persist.
    source.conn.execute("UPDATE jobs SET state='blocked', failure_json=? WHERE job_id=?",
                        (canonical_json(to_document(problem)), jobs[0].job_id))
    clock, claim = _claimer(source)
    _finish(source, claim(), clock)
    ledger = tmp_path / "LEDGER.csv"
    result = _collect(source, ledger_path=ledger)
    assert result["index_ref"] is not None and result["index"]["complete"] is False
    assert result["index"]["planned_variants"] == 2 and result["index"]["attempted_variants"] == 1
    blocked, completed = result["index"]["variants"]
    assert blocked["job_id"] == jobs[0].job_id and blocked["state"] == "refused"
    assert blocked["attempt_id"] is None and blocked["attempted_variants"] == 0
    assert blocked["problem"] == blocked["worker_problem"] == to_document(problem)
    assert blocked["report_ref"] is blocked["outcome_ref"] is blocked["completion_ref"] is None
    assert completed["state"] == "succeeded" and [row["stage"] for row in _rows(ledger)] == ["ran"]
    assert json.loads(source.store.read_verified(from_document(ArtifactRef, result["index_ref"]))) == result["index"]
    assert _collect(source, ledger_path=ledger) == result and len(_rows(ledger)) == 1


@pytest.mark.usefixtures("fixed_sweep_code")
def test_observer_timeout_never_claims_or_cancels_jobs(source, monkeypatch):
    call_sweep(source)
    pending = _collect(source, no_ledger=True)
    ticks = 0
    def clock():
        nonlocal ticks
        ticks += 1
        return min(ticks - 1, 1)
    monkeypatch.setattr(cli, "monotonic", clock)
    monkeypatch.setattr(cli, "sleep", forbid_effect)
    result = call_sweep(source, cli.run_sweep, no_ledger=True, timeout_seconds=0.5)
    assert result == pending
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    assert source.conn.execute("SELECT COUNT(*) FROM supervisor_epochs").fetchone()[0] == 0
    assert not (source.store.root / "supervisor.lock").exists()


def test_invalid_collector_syntax_is_typed_and_preserves_intent(source, tmp_path, monkeypatch):
    checkout = sweep_checkout(tmp_path)
    monkeypatch.setattr(collection, "_COLLECTOR_ROOT", checkout)
    monkeypatch.setattr(collection, "_COLLECTOR_SOURCES", collection.source_closure(checkout, collection._COLLECTOR_ENTRIES))
    call_sweep(source, code_root=checkout)
    _collect(source, code_root=checkout, no_ledger=True)
    before = scientific_rows(source)
    anchor = _registrations(source)[0]
    evidence = json.loads(source.conn.execute(
        "SELECT evidence_json FROM experiment_runs WHERE run_id=?", (anchor.run_id,)).fetchone()[0])
    intent_ref = from_document(ArtifactRef, evidence["native_sweep_smoke_intent"])
    intent_bytes = source.store.read_verified(intent_ref)
    checkout.joinpath("experiments/native_sweep_collection.py").write_text("def invalid(:\n")
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, code_root=checkout, no_ledger=True)
    assert scientific_rows(source) == before
    saved_evidence = json.loads(source.conn.execute(
        "SELECT evidence_json FROM experiment_runs WHERE run_id=?", (anchor.run_id,)).fetchone()[0])
    assert saved_evidence == evidence
    assert source.store.read_verified(from_document(ArtifactRef, saved_evidence["native_sweep_smoke_intent"])) == intent_bytes


@pytest.fixture
def terminal(tmp_path):
    case = terminal_case(tmp_path)
    yield case
    case.conn.close()


def test_terminal_reader_uses_registered_bytes_without_preliminary_file_scan(terminal, monkeypatch):
    monkeypatch.setattr(terminal.store, "verify", forbid_effect)
    attempt, result, problem, worker_problem = collection._terminal(
        terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert attempt["attempt_id"] == "attempt-1" and result == terminal.result
    assert problem is worker_problem is None


def test_terminal_rejects_catalog_reference_rebound_to_different_valid_result(terminal):
    # Both payloads satisfy the result schema and sealed provenance. Only the
    # catalog reference's retained ID is inconsistent with its new bytes.
    payload = canonical_json({**terminal.result, "synthetic_observation": "replacement"}).encode()
    replacement = terminal.store.publish_bytes(payload, schema_ref=terminal.ref.schema_ref)
    assert replacement.artifact_id != terminal.ref.artifact_id
    rebound = replace(replacement, artifact_id=terminal.ref.artifact_id)
    terminal.conn.execute("UPDATE artifacts SET ref_json=? WHERE artifact_id=?",
                          (canonical_json(to_document(rebound)), terminal.ref.artifact_id))
    assert terminal.store.read_verified(rebound) == payload
    assert terminal.conn.execute("SELECT artifact_id FROM attempt_outputs").fetchone()[0] == terminal.ref.artifact_id
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


def test_latest_failed_attempt_never_adopts_the_earlier_successful_output(terminal, monkeypatch):
    worker_problem = fail("WORKER_FAILED", "latest child failed").problem
    terminal.conn.execute("INSERT INTO attempts VALUES ('attempt-2', 'job-1', 2, 3, 'failed', 'ended-later', ?)",
                          (canonical_json(to_document(worker_problem)),))
    terminal.job.state, terminal.job.attempt_count, terminal.job.fence = "failed", 2, 3
    terminal.job.failure = worker_problem
    assert terminal.conn.execute("SELECT COUNT(*) FROM attempts WHERE job_id='job-1'").fetchone()[0] == 2
    assert terminal.conn.execute("SELECT attempt_id FROM attempt_outputs").fetchone()[0] == "attempt-1"
    monkeypatch.setattr(terminal.store, "read_verified", forbid_effect)
    attempt, result, problem, original = collection._terminal(
        terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert attempt["attempt_id"] == "attempt-2" and result is None
    assert problem.code == "EXPERIMENT_VARIANT_FAILED" and problem.retryable is False
    assert original == worker_problem and original.retryable is True


@pytest.mark.parametrize("state,code", [("failed", "WORKER_FAILED"), ("cancelled", "CANCELLED")])
def test_unsuccessful_attempt_fence_cannot_be_ahead_of_job(terminal, state, code):
    terminal.job.state, terminal.job.failure = state, fail(code, "terminal failure").problem
    terminal.conn.execute("UPDATE attempts SET state=?, fence=3", (state,))
    terminal.conn.execute("DELETE FROM attempt_outputs")
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


@pytest.mark.parametrize("damage", [
    "missing-ended-at", "attempt-state", "attempt-fence-ahead", "success-fence-behind", "two-outputs",
    "noncanonical-bytes", "result-schema", "attempted-bool", "attempted-zero", "run-id", "provenance",
    "failed-with-output", "refusal-code", "refusal-category", "refusal-retryable", "success-with-refusal",
])
def test_terminal_contract_rejects_each_independent_damaged_clause(terminal, damage):
    conn, job = terminal.conn, terminal.job
    if damage == "missing-ended-at":
        conn.execute("UPDATE attempts SET ended_at=NULL")
    elif damage == "attempt-state":
        conn.execute("UPDATE attempts SET state='failed'")
    elif damage == "attempt-fence-ahead":
        conn.execute("UPDATE attempts SET fence=3")
    elif damage == "success-fence-behind":
        job.fence = 3
    elif damage == "two-outputs":
        conn.execute("INSERT INTO attempt_outputs VALUES ('attempt-1', 'extra', ?)", (terminal.ref.artifact_id,))
    elif damage == "noncanonical-bytes":
        set_terminal_result(terminal, payload=canonical_json(terminal.result).encode() + b"\n")
    elif damage in {"result-schema", "attempted-bool", "attempted-zero", "run-id", "provenance"}:
        key, value = {"result-schema": ("schema_version", "incorrect.v1.0"),
                      "attempted-bool": ("attempted_variants", True),
                      "attempted-zero": ("attempted_variants", 0),
                      "run-id": ("run_id", "another-run"),
                      "provenance": ("provenance", {"fixture": "another-registration"})}[damage]
        terminal.result[key] = value
        set_terminal_result(terminal)
    else:
        job.failure = fail("RESOURCE_UNAVAILABLE", "synthetic registered refusal").problem
        if damage != "success-with-refusal":
            job.state = "failed"
            conn.execute("UPDATE attempts SET state='failed'")
        if damage.startswith("refusal-"):
            conn.execute("DELETE FROM attempt_outputs")
            key, value = {"refusal-code": ("code", "UNREGISTERED_CODE"),
                          "refusal-category": ("category", "internal"),
                          "refusal-retryable": ("retryable", False)}[damage]
            job.failure = replace(job.failure, **{key: value})
    with pytest.raises(OpsError) as error:
        collection._terminal(conn, terminal.store, job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False


def test_result_size_cap_refuses_metadata_before_opening_object(terminal, monkeypatch):
    ref = replace(terminal.ref, byte_size=collection.MAX_GATE_RESULT_BYTES + 1)
    terminal.conn.execute("UPDATE artifacts SET ref_json=?", (canonical_json(to_document(ref)),))
    monkeypatch.setattr(terminal.store, "verify", forbid_effect)
    monkeypatch.setattr(terminal.store, "read_verified", forbid_effect)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)


def test_documented_collection_byte_caps_are_fixed():
    assert collection.MAX_GATE_RESULT_BYTES == 16 * 1024 * 1024
    assert collection.MAX_FAILURE_BYTES == 1024 * 1024
    assert submission.MAX_JOB_FAILURE_BYTES == 1024 * 1024


def test_result_exactly_at_byte_cap_remains_readable(terminal, monkeypatch):
    monkeypatch.setattr(collection, "MAX_GATE_RESULT_BYTES", terminal.ref.byte_size)
    _, result, problem, _ = collection._terminal(
        terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert result == terminal.result and problem is None


@pytest.mark.parametrize("damage", ["oversized-backing-file", "symlink-backing-file"])
def test_terminal_artifact_read_keeps_registered_size_and_no_follow_guards(terminal, tmp_path, damage):
    path = terminal.store.root / terminal.ref.storage_key
    payload = path.read_bytes()
    path.unlink()
    if damage == "oversized-backing-file":
        path.write_bytes(payload + b"extra bytes beyond registration")
    else:
        target = tmp_path / "untrusted-target"
        target.write_bytes(payload)
        path.symlink_to(target)
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("encoding,wide_text", [("UTF-8", False), ("UTF-16le", False), ("UTF-16le", True)])
def test_recovered_failure_is_bounded_before_json_decode(tmp_path, monkeypatch, encoding, wide_text):
    case = terminal_case(tmp_path, encoding=encoding)
    try:
        case.job.state, case.job.fence = "cancelled", 3
        # Three-byte UTF-8 characters can fit the UTF-16 SQL cap but exceed the decoded UTF-8 cap.
        text = ("漢" * (collection.MAX_FAILURE_BYTES // 3 + 1) if wide_text
                else "x" * (collection.MAX_FAILURE_BYTES + 1))
        case.conn.execute("UPDATE attempts SET state='cancelled', failure_json=?", (text,))
        case.conn.execute("DELETE FROM attempt_outputs")
        bounded = BoundedSqlRows(case.conn, max_bytes=collection.MAX_FAILURE_BYTES *
                                 (2 if encoding.startswith("UTF-16") else 1), column="failure_json",
                                 sql_fragment="end as failure_json from attempts")
        monkeypatch.setattr(collection.json, "loads", forbid_effect)
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            collection._terminal(bounded, case.store, case.job, case.registration)
        sql_size = len(text.encode(encoding))
        assert bounded.sizes == [sql_size if wide_text else None]
    finally:
        case.conn.close()


def test_recovered_failure_exactly_at_byte_cap_retains_its_problem(terminal, monkeypatch):
    problem = fail("CANCELLED", "synthetic cancellation").problem
    payload = canonical_json(to_document(problem))
    monkeypatch.setattr(collection, "MAX_FAILURE_BYTES", len(payload.encode()))
    terminal.job.state, terminal.job.fence = "cancelled", 3
    terminal.conn.execute("UPDATE attempts SET state='cancelled', failure_json=?", (payload,))
    terminal.conn.execute("DELETE FROM attempt_outputs")
    _, result, collected, worker_problem = collection._terminal(
        terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert result is None and collected == worker_problem == problem


def test_recovered_failure_requires_text_storage_even_for_valid_problem_json(terminal):
    payload = canonical_json(to_document(fail("CANCELLED", "synthetic cancellation").problem))
    terminal.job.state, terminal.job.fence = "cancelled", 3
    terminal.conn.execute("UPDATE attempts SET state='cancelled', failure_json=CAST(? AS BLOB)", (payload,))
    terminal.conn.execute("DELETE FROM attempt_outputs")
    assert terminal.conn.execute("SELECT typeof(failure_json) FROM attempts").fetchone()[0] == "blob"
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


@pytest.mark.parametrize("characters,accepted", [(600 * 1024, True), (1024 * 1024 + 1, False)])
def test_utf16_failure_uses_encoded_sql_allowance_and_utf8_contract(tmp_path, monkeypatch, characters, accepted):
    case = terminal_case(tmp_path, encoding="UTF-16le")
    try:
        problem = fail("CANCELLED", "synthetic cancellation", details={"padding": "x" * characters}).problem
        payload = canonical_json(to_document(problem))
        case.job.state, case.job.fence = "cancelled", 3
        case.conn.execute("UPDATE attempts SET state='cancelled', failure_json=?", (payload,))
        case.conn.execute("DELETE FROM attempt_outputs")
        stored_bytes = case.conn.execute("SELECT length(CAST(failure_json AS BLOB)) FROM attempts").fetchone()[0]
        bounded = BoundedSqlRows(case.conn, max_bytes=2 * collection.MAX_FAILURE_BYTES, column="failure_json",
                                 sql_fragment="end as failure_json from attempts")
        if accepted:
            assert len(payload.encode("utf-8")) < 1024 * 1024 < stored_bytes < 2 * 1024 * 1024
            _, result, collected, original = collection._terminal(bounded, case.store, case.job, case.registration)
            assert result is None and collected == original == problem
        else:
            assert stored_bytes > 2 * 1024 * 1024
            monkeypatch.setattr(collection.json, "loads", forbid_effect)
            with pytest.raises(OpsError) as error:
                collection._terminal(bounded, case.store, case.job, case.registration)
            assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
        assert bounded.sizes == [stored_bytes if accepted else None]
    finally:
        case.conn.close()


@pytest.mark.parametrize("case", ["none-mode", "integer-mode", "string-mode", "missing-ledger", "contradictory",
                                  "empty", "directory", "symlink", "integer-path", "bytes-path", "nul-path"])
def test_invalid_collection_mode_or_destination_refuses_before_submission(tmp_path, monkeypatch, case):
    target = tmp_path / "ledger.csv"
    link = tmp_path / "ledger-link.csv"
    link.symlink_to(target)
    arguments = {"none-mode": {"no_ledger": None}, "integer-mode": {"no_ledger": 1},
        "string-mode": {"no_ledger": "yes"}, "missing-ledger": {},
        "contradictory": {"no_ledger": True, "ledger_path": target},
        "empty": {"ledger_path": ""}, "directory": {"ledger_path": tmp_path},
        "symlink": {"ledger_path": link}, "integer-path": {"ledger_path": 1},
        "bytes-path": {"ledger_path": b"ledger.csv"}, "nul-path": {"ledger_path": "ledger\x00.csv"}}[case]
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(None, None, None, {}, code_root=REPO, as_of_month=MONTH, **arguments)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC" and error.value.problem.retryable is False
    assert not target.exists()


@pytest.mark.parametrize("kind", ["fifo", "device", "socket-mode"])
def test_existing_nonregular_ledger_refuses_without_opening_or_submitting(tmp_path, monkeypatch, kind):
    ledger = tmp_path / "ledger.csv"
    if kind == "fifo":
        os.mkfifo(ledger)
    elif kind == "device":
        ledger = Path(os.devnull)
        assert stat.S_ISCHR(ledger.lstat().st_mode)
    else:
        # Socket creation is unavailable in some test sandboxes; exercise its
        # lstat result without requiring permission to create a socket.
        original = Path.lstat
        def metadata(path, *args, **kwargs):
            return SimpleNamespace(st_mode=stat.S_IFSOCK) if path == ledger else original(path, *args, **kwargs)
        monkeypatch.setattr(Path, "lstat", metadata)
    monkeypatch.setattr(collection, "_implementation", forbid_effect)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "open", forbid_effect)
        with pytest.raises(OpsError) as error:
            collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                     code_root=REPO, as_of_month=MONTH, ledger_path=ledger)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC" and error.value.problem.retryable is False


@pytest.mark.parametrize("existing", [True, False])
def test_ledger_preflight_accepts_regular_or_absent_nested_destination(tmp_path, monkeypatch, existing):
    ledger = tmp_path / "ledger.csv" if existing else tmp_path / "missing-parent" / "ledger.csv"
    if existing:
        ledger.write_bytes(b"existing ledger bytes")
    reached = RuntimeError("admitted ledger reached source verification")
    def admitted(*args):
        raise reached
    monkeypatch.setattr(collection, "_implementation", admitted)
    with pytest.raises(RuntimeError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, ledger_path=ledger)
    assert error.value is reached
    assert ledger.read_bytes() == b"existing ledger bytes" if existing else not ledger.parent.exists()


@pytest.mark.parametrize("kind", ["file-parent", "loop-parent", "long-name"])
def test_invalid_ledger_parent_refuses_before_source_or_submission(tmp_path, monkeypatch, kind):
    parent = tmp_path / "parent"
    if kind == "file-parent":
        parent.write_text("not a directory")
    elif kind == "loop-parent":
        parent.symlink_to(parent)
    else:
        parent = tmp_path / ("x" * 256)
    ledger = parent / "ledger.csv"
    monkeypatch.setattr(collection, "_implementation", forbid_effect)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, ledger_path=ledger)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC" and error.value.problem.retryable is False
    assert str(ledger) not in str(error.value)
    assert sorted(path.name for path in tmp_path.iterdir()) == ([] if kind == "long-name" else ["parent"])


@pytest.mark.parametrize("number", [errno.EACCES, errno.EPERM, errno.EIO, errno.EMFILE])
def test_ledger_preflight_storage_error_remains_retryable(tmp_path, monkeypatch, number):
    ledger = tmp_path / "ledger.csv"
    def interrupted(*args, **kwargs):
        raise OSError(number, "private ledger metadata failure")
    monkeypatch.setattr(Path, "lstat", interrupted)
    monkeypatch.setattr(collection, "_implementation", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, ledger_path=ledger)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and error.value.problem.retryable is True
    assert "private ledger" not in str(error.value)


def test_empty_ledger_path_refuses_as_invalid_text_before_filesystem_inspection(monkeypatch):
    with monkeypatch.context() as guard:
        for method in ("is_symlink", "is_dir", "resolve", "stat", "lstat"):
            guard.setattr(Path, method, forbid_effect)
        guard.setattr(collection, "submit_sweep", forbid_effect)
        with pytest.raises(OpsError, match="sweep ledger destination must be a valid path") as error:
            collection.collect_sweep(None, None, None, {}, code_root=REPO, as_of_month=MONTH, ledger_path="")
    assert error.value.code == "INVALID_EXPERIMENT_SPEC" and error.value.problem.retryable is False


def test_active_transaction_refuses_before_submission_without_ending_caller_transaction(monkeypatch):
    conn = sqlite3.connect(":memory:", isolation_level=None)
    try:
        conn.execute("BEGIN")
        monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
        with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
            collection.collect_sweep(conn, None, None, {}, code_root=REPO, as_of_month=MONTH, no_ledger=True)
        assert conn.in_transaction
    finally:
        conn.close()


@pytest.mark.parametrize("exception", [OSError, sqlite3.OperationalError])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_collection_io_interruptions_are_exactly_retryable_resource_failures(monkeypatch, exception):
    def interrupted(*args, **kwargs):
        raise exception("synthetic temporary interruption")
    monkeypatch.setattr(collection, "submit_sweep", interrupted)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and error.value.problem.retryable is True


@pytest.mark.parametrize("code", ["INPUT_CHANGED", "RESOURCE_UNAVAILABLE"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_collection_preserves_delegated_typed_submission_refusal(monkeypatch, code):
    original = fail(code, "delegated submission refusal", details={"data_failure_code": "OperationalError"})
    def refused(*args, **kwargs):
        raise original
    monkeypatch.setattr(collection, "submit_sweep", refused)
    monkeypatch.setattr(collection, "_bind", forbid_effect)
    store = SimpleNamespace(publish_bytes=forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), store, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value is original and error.value.problem is original.problem


@pytest.mark.parametrize("exception", [ValueError, TypeError, KeyError, RecursionError, SyntaxError, RuntimeError])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_collect_preserves_unexpected_delegate_programming_errors(monkeypatch, exception):
    original = exception("unexpected implementation defect")
    def damaged(*args, **kwargs):
        raise original
    monkeypatch.setattr(collection, "submit_sweep", damaged)
    with pytest.raises(exception) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value is original


@pytest.mark.parametrize("target,exception", [
    ("registered_artifact", KeyError), ("registered_artifact", TypeError),
    ("registered_artifact", RuntimeError), ("read_verified", ValueError),
    ("read_verified", TypeError), ("read_verified", KeyError),
    ("read_verified", RecursionError), ("read_verified", RuntimeError),
])
def test_terminal_preserves_unexpected_reader_programming_errors(terminal, monkeypatch, target, exception):
    original = exception("unexpected terminal implementation defect")
    def damaged(*args, **kwargs):
        raise original
    monkeypatch.setattr(collection if target == "registered_artifact" else terminal.store, target, damaged)
    with pytest.raises(exception) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value is original


@pytest.mark.usefixtures("fixed_sweep_code")
def test_collect_preserves_key_error_from_inside_terminal_reader(source, monkeypatch):
    call_sweep(source)
    _complete(source, call_sweep(source, sweep.submit_sweep))
    original = KeyError("unexpected terminal implementation defect")
    def damaged(*args, **kwargs):
        raise original
    monkeypatch.setattr(collection, "registered_artifact", damaged)
    with pytest.raises(KeyError) as error:
        _collect(source, no_ledger=True)
    assert error.value is original


@pytest.mark.parametrize("reader", ["collector", "observer"])
@pytest.mark.parametrize("exception", [ValueError, RecursionError, TypeError, KeyError])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_collection_job_read_preserves_programming_errors(source, monkeypatch, reader, exception):
    original = exception("unexpected receipt reader implementation defect")
    def damaged(*args, **kwargs):
        raise original
    if reader == "collector":
        call_sweep(source)
        monkeypatch.setattr(collection, "get_job", damaged)
    else:
        pending = {"index": {"complete": False, "variants": [{"job_id": "job-1"}]}, "index_ref": None}
        monkeypatch.setattr(cli, "collect_sweep", lambda *args, **kwargs: pending)
        monkeypatch.setattr(cli, "get_job", damaged)
        monkeypatch.setattr(cli, "sleep", forbid_effect)
    with pytest.raises(exception) as error:
        if reader == "collector":
            _collect(source, no_ledger=True)
        else:
            cli.run_sweep(source.conn, source.store, None, {}, code_root=REPO, no_ledger=True)
    assert error.value is original


@pytest.mark.parametrize("target", ["publish_bytes", "verify_native_registration"])
@pytest.mark.parametrize("exception", [ValueError, TypeError, KeyError, RecursionError, RuntimeError])
def test_bind_preserves_unexpected_publication_programming_errors(terminal, monkeypatch, target, exception):
    original = exception("unexpected bind implementation defect")
    def damaged(*args, **kwargs):
        raise original
    monkeypatch.setattr(collection, "_stored_evidence", lambda *_: {})
    monkeypatch.setattr(terminal.store if target == "publish_bytes" else native_registration, target, damaged)
    with pytest.raises(exception) as error:
        collection._bind(terminal.conn, terminal.store, terminal.registration, "intent",
                         {"schema_version": "native_gate_collection.v1.0"})
    assert error.value is original and not terminal.conn.in_transaction


@pytest.mark.parametrize("reference", [[], {}, "not-an-object",
                                      {"schema_version": "artifact_ref.v" + "1" * 5000 + ".0"}])
def test_bind_types_only_malformed_stored_reference(terminal, monkeypatch, reference):
    monkeypatch.setattr(collection, "_stored_evidence", lambda *_: {"intent": reference})
    monkeypatch.setattr(terminal.store, "publish_bytes", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection._bind(terminal.conn, terminal.store, terminal.registration, "intent",
                         {"schema_version": "native_gate_collection.v1.0"})
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert not terminal.conn.in_transaction


@pytest.mark.parametrize("kind", ["intent", "index"])
def test_bind_rejects_oversized_reference_before_any_object_read(terminal, monkeypatch, kind):
    schema = "native_gate_collection.v1.0" if kind == "intent" else "native_gate_sweep_index.v1.0"
    document = {"schema_version": schema}
    expected = artifact_reference(canonical_json(document).encode(), schema)
    saved = to_document(replace(expected, byte_size=10 ** 12))
    monkeypatch.setattr(collection, "_stored_evidence", lambda *_: {kind: saved})
    monkeypatch.setattr(terminal.store, "read_verified", forbid_effect)
    monkeypatch.setattr(terminal.store, "publish_bytes", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection._bind(terminal.conn, terminal.store, terminal.registration, kind, document)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert not terminal.conn.in_transaction


@pytest.mark.parametrize("schema", ["native_gate_collection.v1.0", "native_experiment_report.v1.0",
                                    "native_gate_sweep_index.v1.0"])
@pytest.mark.parametrize("code", ["INTEGRITY_FAILED", "UNSAFE_PATH"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_collect_preserves_delegated_artifact_publication_error_identity(source, monkeypatch, schema, code):
    call_sweep(source)
    _complete(source, call_sweep(source, sweep.submit_sweep))
    original = ArtifactError(code, "private artifact publication detail")
    publish, calls = source.store.publish_bytes, []
    def refused(data, *, schema_ref):
        if schema_ref == schema:
            calls.append(schema_ref)
            raise original
        return publish(data, schema_ref=schema_ref)
    monkeypatch.setattr(source.store, "publish_bytes", refused)
    with pytest.raises(ArtifactError) as error:
        _collect(source, no_ledger=True)
    assert error.value is original and error.value.code == code
    assert calls == [schema]
    assert not any("native_sweep_smoke_index" in evidence for evidence in _evidence(source))


@pytest.mark.usefixtures("fixed_sweep_code")
def test_collect_types_raw_catalog_integrity_constraints(monkeypatch):
    def damaged(*args, **kwargs):
        raise sqlite3.IntegrityError("private catalog detail")
    monkeypatch.setattr(collection, "submit_sweep", damaged)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert "private" not in canonical_json(to_document(error.value.problem))


@pytest.mark.parametrize("spelling", ["relative", "symlinked-parent"])
def test_collection_normalizes_ledger_destination_before_binding(tmp_path, monkeypatch, spelling):
    target = tmp_path / "durable"
    target.mkdir()
    parent = tmp_path / "alias"
    parent.symlink_to(target, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    requested = "durable/LEDGER.csv" if spelling == "relative" else parent / "LEDGER.csv"
    assert collection._mode(False, requested) == ("recorded", target / "LEDGER.csv")
    assert not (target / "LEDGER.csv").exists()


@pytest.mark.parametrize("exception,code,retryable", [(OSError, "RESOURCE_UNAVAILABLE", True),
    (sqlite3.OperationalError, "RESOURCE_UNAVAILABLE", True),
    (sqlite3.IntegrityError, "EXPERIMENT_IDENTITY_CONFLICT", False)])
def test_observer_types_polling_interruptions_without_retrying(monkeypatch, exception, code, retryable):
    pending = {"index": {"complete": False, "variants": [{"job_id": "job-1"}]}, "index_ref": None}
    calls = []
    def collect(*args, **kwargs):
        calls.append(kwargs)
        return pending
    def interrupted(*args, receipt_decode_failure_code, max_failure_bytes):
        assert receipt_decode_failure_code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert max_failure_bytes == submission.MAX_JOB_FAILURE_BYTES
        raise exception("private observer failure detail")
    monkeypatch.setattr(cli, "collect_sweep", collect)
    monkeypatch.setattr(cli, "get_job", interrupted)
    monkeypatch.setattr(cli, "sleep", forbid_effect)
    with pytest.raises(OpsError) as error:
        cli.run_sweep(None, None, None, {}, code_root=REPO, no_ledger=True, timeout_seconds=1)
    assert error.value.code == code and error.value.problem.retryable is retryable
    assert len(calls) == 1 and "private observer" not in canonical_json(to_document(error.value.problem))


@pytest.mark.parametrize("race", ["changed-variant", "different-reference", "same-reference"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_bind_reservation_race_preserves_the_committed_winner(source, monkeypatch, race):
    call_sweep(source)
    anchor = _registrations(source)[0]
    key = "native_sweep_smoke_intent"
    document = {"schema_version": "native_gate_collection.v1.0", "test_intent": "expected"}
    other = document if race == "same-reference" else {**document, "test_intent": "competing"}
    winning_ref = source.store.publish_bytes(canonical_json(other).encode(), schema_ref=other["schema_version"])
    verify = native_registration.verify_native_registration
    committed = None
    def race_after_verification(conn, store, registration, **kwargs):
        nonlocal committed
        verify(conn, store, registration, **kwargs)
        committed = json.loads(conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                                           (anchor.run_id,)).fetchone()[0])
        committed["concurrent_evidence"] = {"preserved": True}
        if race == "changed-variant":
            committed["variant_id"] = "another-variant"
        else:
            committed[key] = to_document(winning_ref)
        conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                     (canonical_json(committed), anchor.run_id))
    monkeypatch.setattr(native_registration, "verify_native_registration", race_after_verification)
    if race == "same-reference":
        assert collection._bind(source.conn, source.store, anchor, key, document) == to_document(winning_ref)
    else:
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            collection._bind(source.conn, source.store, anchor, key, document)
    saved = json.loads(source.conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                                          (anchor.run_id,)).fetchone()[0])
    assert saved == committed
    assert not source.conn.in_transaction


def test_observer_uses_remaining_deadline_and_stops_without_zero_sleep(monkeypatch):
    pending = {"index": {"complete": False, "variants": [{"job_id": "job-1"}]}, "index_ref": None}
    now, delays, collections, reads = [10.0], [], [], []
    def collect(*args, **kwargs):
        collections.append(kwargs)
        return pending
    def advance(delay):
        assert delay > 0, "a deadline reached exactly must not sleep again"
        assert len(delays) < 3, "observer exceeded its fixed deadline"
        delays.append(delay)
        now[0] += delay
    def read_job(conn, job_id, **kwargs):
        assert kwargs == {"max_failure_bytes": 1024 * 1024,
                          "receipt_decode_failure_code": "EXPERIMENT_IDENTITY_CONFLICT"}
        reads.append(job_id)
        return SimpleNamespace(state="queued")
    monkeypatch.setattr(cli, "collect_sweep", collect)
    monkeypatch.setattr(cli, "get_job", read_job)
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli, "sleep", advance)
    assert cli.run_sweep(None, None, None, {}, code_root=REPO, timeout_seconds=1.25, no_ledger=True) == pending
    assert delays == [1, 0.25] and now[0] == 11.25
    assert reads == ["job-1"] * 3
    assert len(collections) == 2 and collections[0] == collections[1]


@pytest.mark.parametrize("action", ["collect", "run"])
@pytest.mark.parametrize("complete", [True, False])
def test_cli_dispatches_requested_action_and_prints_result_and_exit(tmp_path, monkeypatch, capsys, action, complete):
    roster = tmp_path / "roster.json"
    document = {"test_roster": "passed unchanged"}
    roster.write_text(json.dumps(document))
    conn, repository = SimpleNamespace(closed=False), object()
    conn.close = lambda: setattr(conn, "closed", True)
    result = {"index_ref": {"test_ref": "immutable"} if complete else None,
              "index": {"variants": [], "complete": complete}}
    calls = []
    def selected(got_conn, store, got_repository, got_document, **kwargs):
        calls.append(kwargs)
        assert got_conn is conn and got_repository is repository and got_document == document
        assert store.root == tmp_path / "objects"
        return result
    def connect(path, *, must_exist):
        assert path == str(tmp_path / "catalog.sqlite") and must_exist is True
        return conn
    monkeypatch.setattr(cli, "connect", connect)
    monkeypatch.setattr(cli, "Repository", lambda got_conn, store: repository)
    monkeypatch.setattr(cli, "run_sweep" if action == "run" else "collect_sweep", selected)
    monkeypatch.setattr(cli, "collect_sweep" if action == "run" else "run_sweep", forbid_effect)
    (tmp_path / "catalog.sqlite").touch()
    argv = [action, "--roster", str(roster), "--catalog", str(tmp_path / "catalog.sqlite"),
            "--store-root", str(tmp_path / "objects"), "--code-root", str(REPO),
            "--as-of-month", MONTH, "--scope", "shadow", "--event-id", "event-1",
            "--no-ledger", "--timeout-seconds", "2.5"]
    assert cli.main(argv) == (0 if complete else 2)
    output = capsys.readouterr()
    assert output.out == json.dumps(result, sort_keys=True) + "\n" and output.err == ""
    expected = {"code_root": REPO, "as_of_month": MONTH, "scope": "shadow", "event_ids": ["event-1"],
                "no_ledger": True, "ledger_path": None}
    assert calls == [{**expected, **({"timeout_seconds": 2.5} if action == "run" else {})}]
    assert conn.closed is True


@pytest.mark.parametrize("catalog_state", ["unmigrated", "old"])
def test_collect_requires_current_catalog_before_metadata_or_ledger_writes(tmp_path, catalog_state):
    conn = sqlite3.connect(tmp_path / "catalog.sqlite", isolation_level=None)
    conn.row_factory = sqlite3.Row
    ledger = tmp_path / "LEDGER.csv"
    try:
        if catalog_state == "old":
            conn.execute("CREATE TABLE schema_versions (owner TEXT, version INTEGER, checksum TEXT)")
            conn.execute("INSERT INTO schema_versions VALUES (?,?,?)", (OWNER, 1, checksum(MIGRATIONS[0])))
        before = list(conn.iterdump())
        store = SimpleNamespace(publish_bytes=forbid_effect)
        repository = SimpleNamespace(resolve_pinned=forbid_effect)
        with pytest.raises(OpsError) as error:
            collection.collect_sweep(conn, store, repository, sweep_request(), code_root=REPO,
                                     as_of_month=MONTH, ledger_path=ledger)
        assert error.value.code == "INTEGRITY_FAILED" and error.value.problem.retryable is False
        assert error.value.problem.details["reason"] == "schema_older"
        assert list(conn.iterdump()) == before and not conn.in_transaction and not ledger.exists()
    finally:
        conn.close()


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("case", ["null", "normal", "unicode", "malformed", "wrong-type", "oversized",
                                  "unicode-oversized", "exact-cap", "utf16-valid", "utf16-exact-cap",
                                  "utf16-oversized", "utf16-unicode-oversized"])
def test_job_failure_is_bounded_before_decode_on_all_receipt_paths(tmp_path, monkeypatch, reader, case):
    job = submitted_job(tmp_path, encoding="UTF-16le" if case.startswith("utf16-") else "UTF-8",
                        strict=case != "wrong-type")
    try:
        padding = ("x" * (600 * 1024) if case == "utf16-valid" else
                   "漢" * (400 * 1024) if case in {"unicode-oversized", "utf16-unicode-oversized"} else
                   "x" * (1024 * 1024 + 1) if case in {"oversized", "utf16-oversized"} else
                   "漢字🙂" if case == "unicode" else "small")
        problem = fail("RESOURCE_UNAVAILABLE", "receipt failure", details={"padding": padding}).problem
        payload = canonical_json(to_document(problem))
        if case in {"exact-cap", "utf16-exact-cap"}:
            problem = replace(problem, details={"padding": "x" * (1024 * 1024 - len(payload.encode()) + len(padding))})
            payload = canonical_json(to_document(problem))
            assert len(payload.encode()) == 1024 * 1024
        stored = (None if case == "null" else "{invalid JSON" if case == "malformed" else
                  payload.encode() if case == "wrong-type" else payload)
        job.conn.execute("UPDATE jobs SET state='failed', failure_json=? WHERE job_id=?", (stored, job.job_id))
        bounded = BoundedSqlRows(job.conn, max_bytes=submission.MAX_JOB_FAILURE_BYTES *
                                 (2 if case.startswith("utf16-") else 1), column="failure_json",
                                 sql_fragment="end as failure_json from jobs")
        def read():
            if reader == "get-job":
                return submission.get_job(bounded, job.job_id, max_failure_bytes=submission.MAX_JOB_FAILURE_BYTES)
            return submission.submit_graph(bounded, REGISTRY, POLICY, [job.request], clock=job.clock,
                                           max_failure_bytes=submission.MAX_JOB_FAILURE_BYTES)[0]
        if case in {"wrong-type", "oversized", "unicode-oversized", "utf16-oversized", "utf16-unicode-oversized"}:
            original_load = submission.load_json
            def guarded_load(cls, value):
                if cls.__name__ == "Problem":
                    forbid_effect()
                return original_load(cls, value)
            monkeypatch.setattr(submission, "load_json", guarded_load)
            with pytest.raises(OpsError) as error:
                read()
            assert error.value.code == "INTEGRITY_FAILED" and error.value.problem.retryable is False
        elif case == "malformed":
            with pytest.raises(OpsError) as error:
                read()
            assert error.value.code == "INTEGRITY_FAILED"
        else:
            if case == "utf16-valid":
                sql_size = job.conn.execute("SELECT length(CAST(failure_json AS BLOB)) FROM jobs").fetchone()[0]
                assert len(payload.encode()) < 1024 * 1024 < sql_size < 2 * 1024 * 1024
            elif case == "utf16-exact-cap":
                assert job.conn.execute("SELECT length(CAST(failure_json AS BLOB)) FROM jobs").fetchone()[0] == 2 * 1024 * 1024
            receipt = read()
            assert receipt.job_id == job.job_id and receipt.failure == (None if case == "null" else problem)
        sql_size = (None if stored is None or not isinstance(stored, str)
                    else len(stored.encode("UTF-16le" if case.startswith("utf16-") else "UTF-8")))
        expected = None if sql_size is None or sql_size > bounded.max_bytes else sql_size
        assert bounded.sizes == [expected]
        assert not job.conn.in_transaction
        assert job.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        job.conn.close()


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("options", [{}, {"max_failure_bytes": None}], ids=["omitted", "explicit-none"])
def test_default_job_failure_reads_preserve_oversized_receipt(tmp_path, reader, options):
    job = submitted_job(tmp_path)
    try:
        previous = submission.get_job(job.conn, job.job_id)
        problem = fail("RESOURCE_UNAVAILABLE", "uncapped receipt", details={"padding": "x" * (1024 * 1024)}).problem
        payload = canonical_json(to_document(problem))
        assert len(payload.encode()) > submission.MAX_JOB_FAILURE_BYTES
        job.conn.execute("UPDATE jobs SET failure_json=?", (payload,))
        if reader == "get-job":
            receipt = submission.get_job(job.conn, job.job_id, **options)
        else:
            receipt = submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock, **options)[0]
        assert receipt == replace(previous, failure=problem)
        assert job.conn.execute("SELECT failure_json FROM jobs").fetchone()[0] == payload
        assert not job.conn.in_transaction
    finally:
        job.conn.close()


@pytest.mark.parametrize("options", [{}, {"max_failure_bytes": None}], ids=["omitted", "explicit-none"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_default_sweep_submission_preserves_oversized_receipt(source, options):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    problem = fail("RESOURCE_UNAVAILABLE", "uncapped sweep receipt",
                   details={"padding": "x" * (1024 * 1024)}).problem
    payload = canonical_json(to_document(problem))
    assert len(payload.encode()) > submission.MAX_JOB_FAILURE_BYTES
    source.conn.execute("UPDATE jobs SET failure_json=? WHERE job_id=?", (payload, jobs[0].job_id))
    assert call_sweep(source, sweep.submit_sweep, **options) == [replace(jobs[0], failure=problem), *jobs[1:]]
    assert source.conn.execute("SELECT failure_json FROM jobs WHERE job_id=?", (jobs[0].job_id,)).fetchone()[0] == payload


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("case", ["oversized", "malformed", "wrong-type", "invalid-encoding"])
def test_failure_budget_policy_only_reclassifies_size_admission(tmp_path, reader, case):
    job = submitted_job(tmp_path, strict=case != "wrong-type")
    try:
        problem = fail("RESOURCE_UNAVAILABLE", "synthetic stored failure").problem
        payload = canonical_json(to_document(problem))
        if case == "invalid-encoding":
            job.conn.execute("UPDATE jobs SET failure_json=CAST(x'ff' AS TEXT)")
        else:
            value = {"oversized": payload, "malformed": "{broken", "wrong-type": b"{}"}[case]
            job.conn.execute("UPDATE jobs SET failure_json=?", (value,))
        options = {"max_failure_bytes": 64, "receipt_decode_failure_code": "EXPERIMENT_IDENTITY_CONFLICT"}
        with pytest.raises(OpsError) as error:
            if reader == "get-job":
                submission.get_job(job.conn, job.job_id, **options)
            else:
                submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock, **options)
        assert error.value.code == ("EXPERIMENT_IDENTITY_CONFLICT" if case == "oversized" else "INTEGRITY_FAILED")
        assert error.value.problem.retryable is False
        assert not job.conn.in_transaction
    finally:
        job.conn.close()


@pytest.mark.usefixtures("fixed_sweep_code")
def test_collection_forwards_fixed_failure_budget_on_initial_replay_and_later_reads(source, monkeypatch):
    call_sweep(source)
    call_sweep(source, sweep.submit_sweep)
    submit, read = collection.submit_sweep, submission.get_job
    submissions, reads = [], []
    expected = {"max_failure_bytes": 1024 * 1024, "receipt_decode_failure_code": "EXPERIMENT_IDENTITY_CONFLICT"}
    def observed_submit(*args, **kwargs):
        assert {key: kwargs[key] for key in expected} == expected
        submissions.append(True)
        return submit(*args, **kwargs)
    def observed_read(stage):
        def get_job(conn, job_id, **kwargs):
            assert kwargs == expected
            reads.append((stage, job_id))
            return read(conn, job_id, **kwargs)
        return get_job
    monkeypatch.setattr(collection, "submit_sweep", observed_submit)
    monkeypatch.setattr(submission, "get_job", observed_read("initial-replay"))
    monkeypatch.setattr(collection, "get_job", observed_read("collector"))
    result = _collect(source, no_ledger=True)
    job_ids = [arm["job_id"] for arm in result["index"]["variants"]]
    assert submissions == [True]
    assert reads == [(stage, job_id) for stage in ("initial-replay", "collector") for job_id in job_ids]


@pytest.mark.usefixtures("fixed_sweep_code")
def test_oversized_job_failure_refuses_public_collect_before_intent_or_ledger(source, tmp_path, monkeypatch):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    problem = fail("RESOURCE_UNAVAILABLE", "large receipt", details={"padding": "x" * (1024 * 1024 + 1)}).problem
    source.conn.execute("UPDATE jobs SET state='failed', failure_json=? WHERE job_id=?",
                        (canonical_json(to_document(problem)), jobs[0].job_id))
    before = scientific_rows(source)
    original_load = submission.load_json
    def guarded_load(cls, value):
        if cls.__name__ == "Problem":
            forbid_effect()
        return original_load(cls, value)
    monkeypatch.setattr(submission, "load_json", guarded_load)
    ledger = tmp_path / "LEDGER.csv"
    with pytest.raises(OpsError) as error:
        _collect(source, ledger_path=ledger)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert scientific_rows(source) == before and not ledger.exists()
    assert all(not any(key.startswith("native_sweep_") for key in evidence) for evidence in _evidence(source))


@pytest.mark.parametrize("entry", ["native_sweep_collection.py", "v2_gate_collect.py"])
def test_other_code_root_coordinator_bytes_cannot_describe_installed_collector(tmp_path, monkeypatch, entry):
    checkout = sweep_checkout(tmp_path)
    path = checkout / "experiments" / entry
    path.write_text(path.read_text() + "\n# different unexecuted coordinator source\n")
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=checkout, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


def test_unavailable_import_time_collector_snapshot_refuses_before_submission(monkeypatch):
    monkeypatch.setattr(collection, "_COLLECTOR_SOURCE_ERROR", "synthetic snapshot capture failure")
    monkeypatch.setattr(collection, "source_closure", forbid_effect)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


@pytest.mark.parametrize("number,retryable", [
    (errno.EIO, True), (errno.EMFILE, True), (errno.ENFILE, True),
    (errno.EACCES, False), (errno.EPERM, False), (errno.ENOENT, False),
    (errno.ENOTDIR, False), (errno.EISDIR, False), (errno.EINVAL, False), (None, False),
])
def test_collector_source_reads_distinguish_resources_from_unverifiable_code(monkeypatch, number, retryable):
    original, reads = Path.open, []
    target = collection._COLLECTOR_ROOT / "experiments/v2_gate_collect.py"
    def interrupted(path, *args, **kwargs):
        opened_mode = args[0] if args else kwargs.get("mode", "r")
        if path == target:
            reads.append((path, opened_mode))
            assert opened_mode == "rb"
            raise OSError(number, "private source read detail")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", interrupted)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert reads == [(target, "rb")]
    assert error.value.code == ("RESOURCE_UNAVAILABLE" if retryable else "EXPERIMENT_IDENTITY_CONFLICT")
    assert error.value.problem.retryable is retryable
    assert "private" not in canonical_json(to_document(error.value.problem))


def test_source_read_interruption_can_retry_without_replacing_import_snapshot(monkeypatch):
    original, snapshot = Path.open, collection._COLLECTOR_SOURCES
    interruptions = []
    def interrupted(path, *args, **kwargs):
        if path == collection._COLLECTOR_ROOT / "experiments/v2_gate_collect.py" and not interruptions:
            interruptions.append(path)
            raise OSError(errno.EIO, "private temporary source read detail")
        return original(path, *args, **kwargs)
    admitted = RuntimeError("verified sources reached submission")
    def submitted(*args, **kwargs):
        raise admitted
    monkeypatch.setattr(Path, "open", interrupted)
    monkeypatch.setattr(collection, "submit_sweep", submitted)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and error.value.problem.retryable is True
    with pytest.raises(RuntimeError) as retried:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert retried.value is admitted
    assert collection._COLLECTOR_SOURCES is snapshot and collection._COLLECTOR_SOURCE_ERROR is None


@pytest.mark.parametrize("number,code", [(errno.EIO, "RESOURCE_UNAVAILABLE"),
                                       (errno.EACCES, "EXPERIMENT_IDENTITY_CONFLICT")])
def test_caller_source_read_errors_keep_the_same_policy_after_installed_verification(tmp_path, monkeypatch, number, code):
    target = tmp_path / "experiments/v2_gate_collect.py"
    target.parent.mkdir()
    target.write_bytes((REPO / "experiments/v2_gate_collect.py").read_bytes())
    original, reads = Path.open, []
    def interrupted(path, *args, **kwargs):
        if path == target:
            reads.append(path)
            raise OSError(number, "private caller source read detail")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", interrupted)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=tmp_path, as_of_month=MONTH, no_ledger=True)
    assert reads == [target] and error.value.code == code
    assert error.value.problem.retryable is (number == errno.EIO)
    assert "private" not in canonical_json(to_document(error.value.problem))


def test_changed_installed_source_refuses_before_caller_closure(monkeypatch):
    changed = {**collection._COLLECTOR_SOURCES, collection._COLLECTOR_ENTRIES[0]: "sha256:" + "0" * 64}
    calls = []
    def closure(*args):
        calls.append(args)
        assert len(calls) == 1, "known installed-source drift must refuse before caller reads"
        return changed
    monkeypatch.setattr(collection, "source_closure", closure)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False
    assert calls == [(collection._COLLECTOR_ROOT, collection._COLLECTOR_ENTRIES)]


@pytest.mark.parametrize("kind", ["missing", "file", "file-parent", "invalid-type", "nul"])
def test_invalid_code_root_refuses_before_source_reads_or_submission(tmp_path, monkeypatch, kind):
    file = tmp_path / "file"
    file.write_bytes(b"not a directory")
    root = {"missing": tmp_path / "missing", "file": file, "file-parent": file / "child",
            "invalid-type": None, "nul": "\x00"}[kind]
    monkeypatch.setattr(collection, "source_closure", forbid_effect)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=root, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC" and error.value.problem.retryable is False


def test_code_root_stat_io_is_retryable_before_source_reads_or_submission(monkeypatch):
    original = Path.stat
    def interrupted(path, *args, **kwargs):
        if path == REPO:
            raise OSError(errno.EIO, "private stat failure")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", interrupted)
    monkeypatch.setattr(collection, "source_closure", forbid_effect)
    monkeypatch.setattr(collection, "submit_sweep", forbid_effect)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and error.value.problem.retryable is True


@pytest.mark.parametrize("fault,code", [("io", "RESOURCE_UNAVAILABLE"), ("sqlite", "RESOURCE_UNAVAILABLE"),
                                       ("invalid-json", "INVALID_EXPERIMENT_SPEC")])
def test_cli_serializes_exact_raw_failure_code_without_private_details(tmp_path, monkeypatch, capsys, fault, code):
    def read(_path):
        if fault == "invalid-json":
            return b"{invalid private input"
        raise (OSError if fault == "io" else sqlite3.OperationalError)("private failure detail")
    monkeypatch.setattr(cli, "read_native_staged_input", read)
    monkeypatch.setattr(cli, "connect", forbid_effect)
    argv = ["collect", "--roster", str(tmp_path / "roster.json"), "--catalog", str(tmp_path / "catalog.sqlite"),
            "--store-root", str(tmp_path / "objects"), "--code-root", str(REPO),
            "--as-of-month", MONTH, "--no-ledger"]
    assert cli.main(argv) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"refused": code} and output.err == ""
    assert "private" not in output.out and not (tmp_path / "catalog.sqlite").exists()


@pytest.mark.parametrize("payload", [b"{broken", b"[]", b"{}", b'{"schema_version":null}', b"[" * 2000 + b"]" * 2000])
def test_terminal_decoding_translates_actual_malformed_evidence(terminal, payload):
    set_terminal_result(terminal, payload=payload)
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


@pytest.mark.parametrize("field", ["reference", "failure"])
@pytest.mark.parametrize("payload", ["{broken", "[]", "null", "{}", "1" * 5000,
                                    '{"nested":' + "[" * 2000 + "]" * 2000 + "}"])
def test_terminal_types_malformed_metadata_and_recovered_failures(terminal, field, payload):
    if field == "reference":
        terminal.conn.execute("UPDATE artifacts SET ref_json=?", (payload,))
    else:
        terminal.job.state = "failed"
        terminal.conn.execute("UPDATE attempts SET state='failed', failure_json=?", (payload,))
        terminal.conn.execute("DELETE FROM attempt_outputs")
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


@pytest.mark.parametrize("damage", ["malformed", "code", "category", "retryable"])
def test_terminal_decodes_attempt_failure_even_when_job_refusal_is_authoritative(terminal, damage):
    terminal.job.state, terminal.job.attempt_count, terminal.job.fence = "failed", 2, 3
    terminal.job.failure = fail("RESOURCE_UNAVAILABLE", "authoritative refusal").problem
    if damage == "malformed":
        payload = "{broken"
    else:
        changes = {"code": {"code": "UNREGISTERED_CODE"}, "category": {"category": "internal"},
                   "retryable": {"retryable": False}}[damage]
        stored = replace(fail("RESOURCE_UNAVAILABLE", "stored attempt refusal").problem, **changes)
        payload = canonical_json(to_document(stored))
        assert json.loads(payload) == to_document(stored)
    terminal.conn.execute(
        "INSERT INTO attempts VALUES ('attempt-2', 'job-1', 2, 3, 'failed', 'ended-later', ?)", (payload,))
    terminal.conn.execute("DELETE FROM attempt_outputs")
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


def test_terminal_refuses_unencodable_stored_artifact_path(terminal):
    damaged = replace(terminal.ref, content_hash="sha256:\ud800", storage_key="objects/\ud800/\ud800")
    # Escaped JSON is valid UTF-8 text, but the decoded path cannot be encoded by os.open.
    terminal.conn.execute("UPDATE artifacts SET ref_json=?", (json.dumps(to_document(damaged)),))
    with pytest.raises(OpsError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and error.value.problem.retryable is False


@pytest.mark.parametrize("payload", ["{broken", "[]", "null", '{"nested":' + "[" * 2000 + "]" * 2000 + "}"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_malformed_native_evidence_refuses_before_collection_effects(source, tmp_path, payload):
    call_sweep(source)
    source.conn.execute("UPDATE experiment_runs SET evidence_json=?", (payload,))
    before = scientific_rows(source)
    ledger = tmp_path / "LEDGER.csv"
    with pytest.raises(OpsError) as error:
        _collect(source, ledger_path=ledger)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and not ledger.exists()
    assert scientific_rows(source) == before
    assert source.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


@pytest.mark.usefixtures("fixed_sweep_code")
def test_submit_evidence_cap_is_opt_in_and_forwarded_before_staging(source, monkeypatch):
    call_sweep(source)
    anchor = _registrations(source)[0]
    evidence = json.loads(source.conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                                            (anchor.run_id,)).fetchone()[0])
    evidence["unrelated_padding"] = "漢" * 5000
    payload = canonical_json(evidence)
    source.conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?", (payload, anchor.run_id))
    # Existing callers do not acquire the collector's new size policy by default.
    jobs = call_sweep(source, sweep.submit_sweep)
    assert len(jobs) == 2
    before = scientific_rows(source)
    monkeypatch.setattr(source.store, "publish_bytes", forbid_effect)
    with pytest.raises(OpsError) as error:
        call_sweep(source, sweep.submit_sweep, max_evidence_bytes=len(payload.encode()) - 1)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert scientific_rows(source) == before


@pytest.mark.parametrize("field,payload,code", [
    ("failure_json", "{broken", "INTEGRITY_FAILED"),
    ("failure_json", "[" * 2000 + "]" * 2000, "INTEGRITY_FAILED"),
    ("queue_reason_json", "{broken", "EXPERIMENT_IDENTITY_CONFLICT"),
])
def test_observer_types_persisted_receipt_corruption_at_job_read(tmp_path, monkeypatch, field, payload, code):
    job = submitted_job(tmp_path)
    try:
        job.conn.execute(f"UPDATE jobs SET {field}=?", (payload,))
        pending = {"index": {"complete": False, "variants": [{"job_id": job.job_id}]}, "index_ref": None}
        monkeypatch.setattr(cli, "collect_sweep", lambda *args, **kwargs: pending)
        monkeypatch.setattr(cli, "sleep", forbid_effect)
        with pytest.raises(OpsError) as error:
            cli.run_sweep(job.conn, None, None, {}, code_root=REPO, no_ledger=True)
        assert error.value.code == code and error.value.problem.retryable is False
    finally:
        job.conn.close()


@pytest.mark.parametrize("action", ["collect", "run"])
@pytest.mark.parametrize("field", ["queue_reason_json", "progress_json", "resources_json", "failure_json"])
@pytest.mark.parametrize("payload", ["{broken", "[" * 2000 + "]" * 2000], ids=["malformed", "recursive"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_initial_collection_replay_types_corrupt_receipt_before_effects(
        source, tmp_path, monkeypatch, action, field, payload):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    if field in {"progress_json", "resources_json"}:
        _, claim = _claimer(source)
        attempt = claim()
        if field == "resources_json":
            source.conn.execute("UPDATE attempts SET resources_json=? WHERE attempt_id=?",
                                (payload, attempt.attempt_id))
        else:
            source.conn.execute("INSERT INTO progress_events VALUES (?,0,?,'progress','synthetic-time',?)",
                                (attempt.attempt_id, attempt.job_id, payload))
    else:
        source.conn.execute(f"UPDATE jobs SET {field}=? WHERE job_id=?", (payload, jobs[0].job_id))
    before = list(source.conn.iterdump())
    monkeypatch.setattr(collection, "_bind", forbid_effect)
    monkeypatch.setattr(native_outcomes, "ledger_append", forbid_effect)
    monkeypatch.setattr(cli, "sleep", forbid_effect)
    ledger = tmp_path / "LEDGER.csv"
    with pytest.raises(OpsError) as error:
        call_sweep(source, collection.collect_sweep if action == "collect" else cli.run_sweep, ledger_path=ledger)
    assert error.value.code == ("INTEGRITY_FAILED" if field == "failure_json" else "EXPERIMENT_IDENTITY_CONFLICT")
    assert error.value.problem.retryable is False
    assert list(source.conn.iterdump()) == before and not source.conn.in_transaction and not ledger.exists()


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("options", [{}, {"receipt_decode_failure_code": None},
                                    {"receipt_decode_failure_code": "EXPERIMENT_IDENTITY_CONFLICT"},
                                    {"receipt_decode_failure_code": "RESOURCE_UNAVAILABLE"}],
                         ids=["omitted", "explicit-none", "sweep-r6", "other-registered-code"])
def test_shared_job_receipt_corruption_policy_is_opt_in(tmp_path, reader, options):
    job = submitted_job(tmp_path)
    try:
        job.conn.execute("UPDATE jobs SET queue_reason_json='{broken'")
        before = list(job.conn.iterdump())
        expected = OpsError if options.get("receipt_decode_failure_code") else json.JSONDecodeError
        with pytest.raises(expected) as error:
            if reader == "get-job":
                submission.get_job(job.conn, job.job_id, **options)
            else:
                submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock, **options)
        if expected is OpsError:
            assert error.value.code == options["receipt_decode_failure_code"]
            assert error.value.problem.retryable is (error.value.code == "RESOURCE_UNAVAILABLE")
        assert list(job.conn.iterdump()) == before and not job.conn.in_transaction
    finally:
        job.conn.close()


@pytest.mark.parametrize("options", [{}, {"receipt_decode_failure_code": None},
                                    {"receipt_decode_failure_code": "EXPERIMENT_IDENTITY_CONFLICT"},
                                    {"receipt_decode_failure_code": "RESOURCE_UNAVAILABLE"}],
                         ids=["omitted", "explicit-none", "sweep-r6", "other-registered-code"])
@pytest.mark.usefixtures("fixed_sweep_code")
def test_shared_sweep_submission_receipt_corruption_policy_is_opt_in(source, options):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    source.conn.execute("UPDATE jobs SET queue_reason_json='{broken' WHERE job_id=?", (jobs[0].job_id,))
    before = list(source.conn.iterdump())
    expected = OpsError if options.get("receipt_decode_failure_code") else json.JSONDecodeError
    with pytest.raises(expected) as error:
        call_sweep(source, sweep.submit_sweep, **options)
    if expected is OpsError:
        assert error.value.code == options["receipt_decode_failure_code"]
        assert error.value.problem.retryable is (error.value.code == "RESOURCE_UNAVAILABLE")
    assert list(source.conn.iterdump()) == before and not source.conn.in_transaction


@pytest.mark.parametrize("reader", ["get-job", "submit-graph", "submit-sweep"])
@pytest.mark.parametrize("code", ["UNREGISTERED_CODE", True, [], {}])
def test_invalid_receipt_decode_policy_refuses_before_catalog_access(reader, code):
    conn = SimpleNamespace(execute=forbid_effect)
    with pytest.raises(OpsError) as error:
        if reader == "get-job":
            submission.get_job(conn, "synthetic-job", receipt_decode_failure_code=code)
        elif reader == "submit-graph":
            submission.submit_graph(conn, REGISTRY, POLICY, [], clock=SystemClock(),
                                    receipt_decode_failure_code=code)
        else:
            sweep.submit_sweep(conn, None, None, {}, code_root=REPO, as_of_month=MONTH,
                               receipt_decode_failure_code=code)
    assert error.value.code == "INVALID_REQUEST" and error.value.problem.retryable is False


@pytest.mark.parametrize("reader", ["get-job", "submit-graph", "submit-sweep"])
@pytest.mark.parametrize("limit", [0, -1, True, 1.0, "1", [], {}, (2**63 - 1) // 2 + 1])
def test_invalid_failure_budget_refuses_before_catalog_or_submission(reader, limit):
    conn = SimpleNamespace(execute=forbid_effect)
    with pytest.raises(OpsError) as error:
        if reader == "get-job":
            submission.get_job(conn, "synthetic-job", max_failure_bytes=limit)
        elif reader == "submit-graph":
            submission.submit_graph(conn, REGISTRY, POLICY, [], clock=SystemClock(), max_failure_bytes=limit)
        else:
            sweep.submit_sweep(conn, None, None, {}, code_root=REPO, as_of_month=MONTH, max_failure_bytes=limit)
    assert error.value.code == "INVALID_REQUEST" and error.value.problem.retryable is False


@pytest.mark.parametrize("limit", [1, (2**63 - 1) // 2])
def test_failure_budget_accepts_inclusive_positive_integer_endpoints(tmp_path, limit):
    job = submitted_job(tmp_path, encoding="UTF-16le")
    try:
        assert submission.get_job(job.conn, job.job_id, max_failure_bytes=limit).failure is None
        assert submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock,
                                       max_failure_bytes=limit)[0].job_id == job.job_id
    finally:
        job.conn.close()


def _cli_args(tmp_path, *, action="collect", roster=None):
    return [action, "--roster", str(roster or tmp_path / "roster.json"),
            "--catalog", str(tmp_path / "catalog.sqlite"), "--store-root", str(tmp_path / "objects"),
            "--code-root", str(REPO), "--as-of-month", MONTH, "--no-ledger"]


@pytest.mark.parametrize("kind", ["missing", "directory", "fifo", "file-parent"])
def test_cli_catalog_requires_existing_regular_file_before_connect(tmp_path, monkeypatch, capsys, kind):
    (tmp_path / "roster.json").write_text("{}")
    path = tmp_path / "catalog.sqlite"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "file-parent":
        path.write_text("not a directory")
        path = path / "child"
    argv = _cli_args(tmp_path)
    argv[argv.index("--catalog") + 1] = str(path)
    monkeypatch.setattr(cli, "connect", forbid_effect)
    assert cli.main(argv) == 2
    output = capsys.readouterr()
    assert output.out == '{"refused": "INVALID_EXPERIMENT_SPEC"}\n' and output.err == ""
    assert not (tmp_path / "objects").exists()
    if kind == "missing":
        assert not path.exists()


@pytest.mark.parametrize("number", [errno.EACCES, errno.EPERM, errno.EIO, errno.EMFILE, errno.ENOMEM])
def test_cli_catalog_stat_resource_errors_remain_retryable(tmp_path, monkeypatch, capsys, number):
    (tmp_path / "roster.json").write_text("{}")
    catalog = tmp_path / "catalog.sqlite"
    original = Path.stat
    def unavailable(path, *args, **kwargs):
        if path == catalog:
            raise OSError(number, "private catalog stat error")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", unavailable)
    monkeypatch.setattr(cli, "connect", forbid_effect)
    assert cli.main(_cli_args(tmp_path)) == 2
    output = capsys.readouterr()
    assert output.out == '{"refused": "RESOURCE_UNAVAILABLE"}\n' and output.err == ""
    assert not (tmp_path / "objects").exists()


def test_cli_catalog_stat_does_not_reclassify_later_open_failure(tmp_path, monkeypatch, capsys):
    (tmp_path / "roster.json").write_text("{}")
    catalog = tmp_path / "catalog.sqlite"
    catalog.touch()
    def disappear(path, *, must_exist):
        assert Path(path) == catalog and must_exist
        catalog.unlink()
        raise sqlite3.OperationalError("private catalog open failure after stat")
    monkeypatch.setattr(cli, "connect", disappear)
    assert cli.main(_cli_args(tmp_path)) == 2
    output = capsys.readouterr()
    assert output.out == '{"refused": "RESOURCE_UNAVAILABLE"}\n' and output.err == ""
    assert not catalog.exists() and not (tmp_path / "objects").exists()


@pytest.mark.parametrize("path_case", ["missing", "directory", "symlink"])
def test_cli_invalid_roster_paths_refuse_before_catalog(tmp_path, monkeypatch, capsys, path_case):
    path = tmp_path / "input"
    if path_case == "directory":
        path.mkdir()
    elif path_case == "symlink":
        target = tmp_path / "target.json"
        target.write_text("{}")
        path.symlink_to(target)
    monkeypatch.setattr(cli, "connect", forbid_effect)
    assert cli.main(_cli_args(tmp_path, roster=path)) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"refused": "INVALID_EXPERIMENT_SPEC"} and output.err == ""


@pytest.mark.parametrize("number,code", [(errno.EACCES, "INVALID_EXPERIMENT_SPEC"),
    (errno.EPERM, "INVALID_EXPERIMENT_SPEC"), (errno.ENOTDIR, "INVALID_EXPERIMENT_SPEC"),
    (errno.EINVAL, "INVALID_EXPERIMENT_SPEC"), (errno.ENAMETOOLONG, "INVALID_EXPERIMENT_SPEC"),
    (errno.ENOENT, "INVALID_EXPERIMENT_SPEC"), (errno.ELOOP, "INVALID_EXPERIMENT_SPEC"),
    (errno.EIO, "RESOURCE_UNAVAILABLE"), (errno.EMFILE, "RESOURCE_UNAVAILABLE"),
    (errno.ENOMEM, "RESOURCE_UNAVAILABLE")])
def test_cli_roster_io_distinguishes_bad_paths_from_resource_failures(tmp_path, monkeypatch, capsys, number, code):
    def read(path):
        raise OSError(number, "private path detail")
    monkeypatch.setattr(cli, "read_native_staged_input", read)
    monkeypatch.setattr(cli, "connect", forbid_effect)
    assert cli.main(_cli_args(tmp_path)) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"refused": code} and output.err == ""


def test_cli_recursive_roster_is_invalid_input_before_catalog(tmp_path, monkeypatch, capsys):
    def recurse(payload):
        raise RecursionError("private decoder detail")
    monkeypatch.setattr(cli, "read_native_staged_input", lambda path: b"{}")
    monkeypatch.setattr(cli, "json", SimpleNamespace(loads=recurse, dumps=json.dumps))
    monkeypatch.setattr(cli, "connect", forbid_effect)
    assert cli.main(_cli_args(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC"}


@pytest.mark.parametrize("error", [ArtifactError("UNSAFE_PATH", "private artifact path"),
                                    data_fail("INPUT_CHANGED", "private input detail")])
def test_cli_preserves_exact_typed_artifact_and_data_codes(tmp_path, monkeypatch, capsys, error):
    (tmp_path / "catalog.sqlite").touch()
    conn = SimpleNamespace(closed=False)
    conn.close = lambda: setattr(conn, "closed", True)
    monkeypatch.setattr(cli, "read_native_staged_input", lambda path: b"{}")
    monkeypatch.setattr(cli, "connect", lambda *args, **kwargs: conn)
    monkeypatch.setattr(cli, "Repository", lambda *args: object())
    def refused(*args, **kwargs):
        raise error
    monkeypatch.setattr(cli, "collect_sweep", refused)
    assert cli.main(_cli_args(tmp_path)) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {"refused": error.code} and output.err == "" and conn.closed


@pytest.mark.parametrize("error_type", [ValueError, KeyError, TypeError])
def test_cli_does_not_rebrand_an_unexpected_action_bug_as_bad_input(tmp_path, monkeypatch, error_type):
    (tmp_path / "catalog.sqlite").touch()
    conn = SimpleNamespace(closed=False)
    conn.close = lambda: setattr(conn, "closed", True)
    monkeypatch.setattr(cli, "read_native_staged_input", lambda path: b"{}")
    monkeypatch.setattr(cli, "connect", lambda *args, **kwargs: conn)
    monkeypatch.setattr(cli, "Repository", lambda *args: object())
    original = error_type("implementation defect")
    def broken(*args, **kwargs):
        raise original
    monkeypatch.setattr(cli, "collect_sweep", broken)
    with pytest.raises(error_type) as caught:
        cli.main(_cli_args(tmp_path))
    assert caught.value is original and conn.closed


def test_library_and_cli_observer_defaults_are_one_hour(tmp_path, monkeypatch, capsys):
    (tmp_path / "catalog.sqlite").touch()
    assert inspect.signature(cli.run_sweep).parameters["timeout_seconds"].default == 3600
    conn = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli, "read_native_staged_input", lambda path: b"{}")
    monkeypatch.setattr(cli, "connect", lambda *args, **kwargs: conn)
    monkeypatch.setattr(cli, "Repository", lambda *args: object())
    def run(*args, **kwargs):
        assert kwargs["timeout_seconds"] == 3600
        return {"index": {"complete": False}, "index_ref": None}
    monkeypatch.setattr(cli, "run_sweep", run)
    assert cli.main(_cli_args(tmp_path, action="run")) == 2
    assert json.loads(capsys.readouterr().out)["index"]["complete"] is False


@pytest.mark.parametrize("payload", [b"[]", b"null", b"[" * 2000 + b"]" * 2000])
def test_cli_nonobject_roster_is_invalid_before_catalog(tmp_path, monkeypatch, capsys, payload):
    monkeypatch.setattr(cli, "read_native_staged_input", lambda path: payload)
    monkeypatch.setattr(cli, "connect", forbid_effect)
    assert cli.main(_cli_args(tmp_path)) == 2
    assert json.loads(capsys.readouterr().out) == {"refused": "INVALID_EXPERIMENT_SPEC"}


@pytest.mark.parametrize("reader", ["get-job", "submit-replay"])
def test_job_failure_invalid_encoding_is_typed_before_receipt_decoding(tmp_path, reader):
    job = submitted_job(tmp_path)
    try:
        job.conn.execute("UPDATE jobs SET failure_json=CAST(x'ff' AS TEXT)")
        with pytest.raises(sqlite3.OperationalError, match="Could not decode to UTF-8"):
            job.conn.execute("SELECT failure_json FROM jobs").fetchone()
        with pytest.raises(OpsError) as error:
            if reader == "get-job":
                submission.get_job(job.conn, job.job_id)
            else:
                submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock)
        assert error.value.code == "INTEGRITY_FAILED" and error.value.problem.retryable is False
    finally:
        job.conn.close()


@pytest.mark.parametrize("reader", ["get-job", "submit-replay"])
@pytest.mark.parametrize("exception", [TypeError, KeyError])
@pytest.mark.parametrize("stored_failure", [False, True])
def test_job_failure_decoder_programming_errors_propagate(tmp_path, monkeypatch, reader, exception, stored_failure):
    job = submitted_job(tmp_path)
    if stored_failure:
        payload = canonical_json(to_document(fail("WORKER_FAILED", "stored worker failure").problem))
        job.conn.execute("UPDATE jobs SET failure_json=?", (payload,))
    original = exception("unexpected failure decoder implementation defect")
    decode = submission.load_json
    def damaged(cls, value):
        if cls.__name__ == "Problem":
            raise original
        return decode(cls, value)
    monkeypatch.setattr(submission, "load_json", damaged)
    try:
        with pytest.raises(exception) as error:
            if reader == "get-job":
                submission.get_job(job.conn, job.job_id)
            else:
                submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock)
        assert error.value is original
        assert job.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    finally:
        job.conn.close()


@pytest.mark.parametrize("reader", ["get-job", "submit-replay"])
@pytest.mark.parametrize("payload", ["[]", "{}", "1" * 5000,
    '{"schema_version":"problem.v' + "1" * 5000 + '.0"}'])
def test_malformed_job_failure_shapes_remain_typed(tmp_path, reader, payload):
    job = submitted_job(tmp_path)
    try:
        job.conn.execute("UPDATE jobs SET failure_json=?", (payload,))
        with pytest.raises(OpsError) as error:
            if reader == "get-job":
                submission.get_job(job.conn, job.job_id)
            else:
                submission.submit_graph(job.conn, REGISTRY, POLICY, [job.request], clock=job.clock)
        assert error.value.code == "INTEGRITY_FAILED" and error.value.problem.retryable is False
    finally:
        job.conn.close()
