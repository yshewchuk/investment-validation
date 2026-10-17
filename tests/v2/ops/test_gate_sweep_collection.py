"""Exact-attempt collection, immutable native reconciliation and smoke isolation."""
import csv
import json
import sqlite3
from dataclasses import asdict, replace
from time import sleep
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    SystemClock,
    canonical_json,
    from_document,
    to_document,
)
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
from tests.ops_support import TEST_POLICY, FakeClock, sample
from tests.v2.ops.gate_sweep_support import (
    call_sweep,
    forbid_effect,
    replace_checkout_source,
    scientific_rows,
    sealed_sweep,
    set_terminal_result,
    sweep_checkout,
    sweep_request,
    terminal_case,
)
from tests.v2.research.test_gate_report import MONTH, REPO, build_gate_spec
from tests.v2.research.test_gate_report import source as source


def _collect(source, *, code_root=REPO, document=None, **kwargs):
    return call_sweep(source, collection.collect_sweep, document, code_root=code_root, **kwargs)


def _run(source, *, code_root=REPO, document=None, **kwargs):
    service = Service(source.conn, source.store.root, registry(), TEST_POLICY,
                      clock=SystemClock(), code_source=code_root)
    def tick(_delay):
        service.tick()
        sleep(0.05)
    try:
        service.start()
        with patch.object(cli, "sleep", tick):
            return call_sweep(source, cli.run_sweep, document, code_root=code_root, timeout_seconds=90, **kwargs)
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
    for _ in jobs:
        _finish(source, claim(), clock)


def test_real_run_records_independent_arms_and_exact_replay(source, tmp_path, monkeypatch):
    ledger = tmp_path / "LEDGER.csv"
    call_sweep(source)
    result = _run(source, ledger_path=ledger)
    index = result["index"]
    assert index["complete"] is True
    assert index["planned_variants"] == index["attempted_variants"] == 2
    assert index["recording_mode"] == "recorded"
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


def test_pending_collection_binds_mode_and_does_not_freeze_index(source, tmp_path):
    call_sweep(source)
    smoke = _collect(source, no_ledger=True)
    recorded = _collect(source, ledger_path=tmp_path / "LEDGER.csv")
    for result in (smoke, recorded):
        assert result["index_ref"] is None and result["index"]["complete"] is False
        assert result["index"]["attempted_variants"] == 0
        assert [arm["state"] for arm in result["index"]["variants"]] == ["queued", "queued"]
    assert smoke["index"]["run_id"] != recorded["index"]["run_id"]
    assert len({arm["job_id"] for result in (smoke, recorded) for arm in result["index"]["variants"]}) == 2
    assert not list(source.root.rglob("*.csv"))
    assert all(not any(key.endswith("_index") for key in evidence) for evidence in _evidence(source))
    before = scientific_rows(source)
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, ledger_path=tmp_path / "different.csv")
    assert scientific_rows(source) == before and not (tmp_path / "different.csv").exists()


def test_changed_collector_source_and_inputs_cannot_rebind_saved_intent(source, tmp_path):
    checkout = sweep_checkout(tmp_path)
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


@pytest.mark.parametrize("failure", ["ledger", "receipt", "export", "index"])
def test_interrupted_publication_reconciles_without_duplicate_rows(source, tmp_path, monkeypatch, failure):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    _complete(source, jobs)
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
    result = _collect(source, ledger_path=ledger)
    assert result["index"]["complete"] and result["index_ref"] is not None
    assert len(_rows(ledger)) == 2
    assert _collect(source, ledger_path=ledger) == result and len(_rows(ledger)) == 2


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


@pytest.mark.parametrize("damage", ["identity", "wrong-schema", "wrong-name", "no-output", "terminal-attempt"])
def test_terminal_output_is_exact_and_missing_result_is_r4(source, tmp_path, damage):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    _complete(source, jobs)
    attempt = source.conn.execute("SELECT * FROM attempts WHERE job_id=?", (jobs[0].job_id,)).fetchone()
    output = source.conn.execute("SELECT * FROM attempt_outputs WHERE attempt_id=?", (attempt["attempt_id"],)).fetchone()
    if damage in ("identity", "wrong-schema"):
        ref = artifact(source.conn, source.store, output["artifact_id"])
        result = json.loads(source.store.read_verified(ref))
        if damage == "identity":
            result["variant_id"] = "sha256:" + "0" * 64
        replacement = source.store.publish_bytes(canonical_json(result).encode(),
            schema_ref="incorrect.v1.0" if damage == "wrong-schema" else ref.schema_ref)
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
def test_cli_refuses_purpose_or_missing_mode_before_catalog(tmp_path, capsys, arguments, code):
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
def test_cancellation_and_recovery_use_exact_fenced_terminal_attempt(source, tmp_path, state):
    call_sweep(source)
    jobs = call_sweep(source, sweep.submit_sweep)
    clock, claim = _claimer(source)
    if state == "queued-cancel":
        request_cancel(source.conn, jobs[0].job_id, None, clock=clock)
        failed_job, attempted = jobs[0].job_id, 0
    else:
        attempt = claim()
        failed_job, attempted = attempt.job_id, 1
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
    _finish(source, claim(), clock)
    ledger = tmp_path / "LEDGER.csv"
    result = _collect(source, ledger_path=ledger)
    assert result["index_ref"] is not None and result["index"]["complete"] is False
    assert result["index"]["attempted_variants"] == 1 + attempted
    arm = next(item for item in result["index"]["variants"] if item["job_id"] == failed_job)
    assert arm["state"] == "refused"
    assert arm["problem"] == arm["worker_problem"]
    assert arm["problem"]["code"] == ("LEASE_LOST" if state == "lease-lost" else "CANCELLED")
    assert arm["problem"]["retryable"] is (state == "lease-lost")
    assert arm["outcome_ref"] is arm["completion_ref"] is arm["report_ref"] is None
    assert [row["stage"] for row in _rows(ledger)] == ["ran"]
    assert _collect(source, ledger_path=ledger) == result and len(_rows(ledger)) == 1


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


def test_observer_timeout_never_claims_or_cancels_jobs(source, monkeypatch):
    call_sweep(source)
    pending = _collect(source, no_ledger=True)
    clock = iter((0, 1))
    monkeypatch.setattr(cli, "monotonic", lambda: next(clock))
    monkeypatch.setattr(cli, "sleep", forbid_effect)
    result = call_sweep(source, cli.run_sweep, no_ledger=True, timeout_seconds=0.5)
    assert result == pending
    assert source.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    assert source.conn.execute("SELECT COUNT(*) FROM supervisor_epochs").fetchone()[0] == 0
    assert not (source.store.root / "supervisor.lock").exists()


def test_invalid_collector_syntax_is_typed_and_preserves_intent(source, tmp_path):
    checkout = sweep_checkout(tmp_path)
    call_sweep(source, code_root=checkout)
    _collect(source, code_root=checkout, no_ledger=True)
    before = scientific_rows(source)
    checkout.joinpath("experiments/native_sweep_collection.py").write_text("def invalid(:\n")
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        _collect(source, code_root=checkout, no_ledger=True)
    assert scientific_rows(source) == before


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
    with pytest.raises(ArtifactError) as error:
        collection._terminal(terminal.conn, terminal.store, terminal.job, terminal.registration)
    assert error.value.code == ("INTEGRITY_FAILED" if damage == "oversized-backing-file" else "UNSAFE_PATH")


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
        monkeypatch.setattr(collection.json, "loads", forbid_effect)
        with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
            collection._terminal(case.conn, case.store, case.job, case.registration)
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
def test_collection_io_interruptions_are_exactly_retryable_resource_failures(monkeypatch, exception):
    def interrupted(*args, **kwargs):
        raise exception("synthetic temporary interruption")
    monkeypatch.setattr(collection, "submit_sweep", interrupted)
    with pytest.raises(OpsError) as error:
        collection.collect_sweep(SimpleNamespace(in_transaction=False), None, None, {},
                                 code_root=REPO, as_of_month=MONTH, no_ledger=True)
    assert error.value.code == "RESOURCE_UNAVAILABLE" and error.value.problem.retryable is True


@pytest.mark.parametrize("race", ["changed-variant", "different-reference", "same-reference"])
def test_bind_reservation_race_preserves_the_committed_winner(source, monkeypatch, race):
    call_sweep(source)
    anchor = _registrations(source)[0]
    key = "native_sweep_smoke_intent"
    document = {"schema_version": "native_gate_collection.v1.0", "test_intent": "expected"}
    other = document if race == "same-reference" else {**document, "test_intent": "competing"}
    winning_ref = source.store.publish_bytes(canonical_json(other).encode(), schema_ref=other["schema_version"])
    verify = native_registration.verify_native_registration
    committed = None
    def race_after_verification(conn, store, registration):
        nonlocal committed
        verify(conn, store, registration)
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
    now, delays, collections = [10.0], [], []
    def collect(*args, **kwargs):
        collections.append(kwargs)
        return pending
    def advance(delay):
        assert delay > 0, "a deadline reached exactly must not sleep again"
        assert len(delays) < 3, "observer exceeded its fixed deadline"
        delays.append(delay)
        now[0] += delay
    monkeypatch.setattr(cli, "collect_sweep", collect)
    monkeypatch.setattr(cli, "get_job", lambda *args: SimpleNamespace(state="queued"))
    monkeypatch.setattr(cli, "monotonic", lambda: now[0])
    monkeypatch.setattr(cli, "sleep", advance)
    assert cli.run_sweep(None, None, None, {}, code_root=REPO, timeout_seconds=1.25, no_ledger=True) == pending
    assert delays == [1, 0.25] and now[0] == 11.25
    assert len(collections) == 2 and collections[0] == collections[1]


@pytest.mark.parametrize("action", ["collect", "run"])
@pytest.mark.parametrize("complete", [True, False])
def test_cli_dispatches_requested_action_and_prints_result_and_exit(tmp_path, monkeypatch, capsys, action, complete):
    roster = tmp_path / "roster.json"
    document = {"test_roster": "passed unchanged"}
    roster.write_text(json.dumps(document))
    conn, repository = SimpleNamespace(closed=False), object()
    conn.close = lambda: setattr(conn, "closed", True)
    result = {"index": {"complete": complete, "variants": []},
              "index_ref": {"test_ref": "immutable"} if complete else None}
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
    argv = [action, "--roster", str(roster), "--catalog", str(tmp_path / "catalog.sqlite"),
            "--store-root", str(tmp_path / "objects"), "--code-root", str(REPO),
            "--as-of-month", MONTH, "--scope", "shadow", "--event-id", "event-1",
            "--no-ledger", "--timeout-seconds", "2.5"]
    assert cli.main(argv) == (0 if complete else 2)
    output = capsys.readouterr()
    assert json.loads(output.out) == result and output.err == ""
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
