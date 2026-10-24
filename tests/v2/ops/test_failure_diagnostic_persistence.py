"""Oversized coordinator failures publish complete diagnostics before fenced completion.

No subprocesses, market-data fixture or automatic recovery protocol is involved.
An interrupted publication may leave a complete content-addressed orphan; the
catalog must never claim completion without its registered full diagnostic.
"""
import json
import os
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from engine.v2.contracts import Problem
from engine.v2.foundation import ArtifactError, ArtifactStore, artifact_reference, format_timestamp
from engine.v2.ops import catalog, failure_diagnostics, lifecycle, recovery, scheduler, submission
from engine.v2.ops import supervisor as supervisor_module
from engine.v2.ops.checkpoints import register_artifact, registered_artifact
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.migrations import migrate
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor
from engine.v2.ops.schema import MIGRATIONS, OWNER
from engine.v2.ops.submission import get_job
from engine.v2.ops.supervisor import Service
from tests.ops_support import FakeClock
from tests.v2.ops.native_bounds_support import catalog_rows, object_bytes
from tests.v2.ops.test_receipt_failure_bounds import coordinator_failure, coordinator_inputs, exact_problem_bytes
from tools.v2_ops_fixtures import POLICY, REGISTRY, enqueue_claim, request, sample

CAP = 1024 * 1024


def service_for(conn, clock, store):
    service = Service.__new__(Service)
    service.conn, service.clock, service.store = conn, clock, store
    service.launches, service._progress_state = {}, {}
    service._claim_keepalive_cache = None
    service._renewer = None
    return service


@pytest.fixture
def case(tmp_path):
    clock = FakeClock()
    path = tmp_path / "ops.sqlite"
    conn = catalog.connect(path)
    migrate(conn, OWNER, MIGRATIONS, clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    claim = enqueue_claim(conn, clock, Supervisor(epoch, "boot"))
    store = ArtifactStore(tmp_path / "store")
    value = SimpleNamespace(conn=conn, path=path, clock=clock, claim=claim, store=store,
                            service=service_for(conn, clock, store))
    try:
        yield value
    finally:
        value.conn.close()


@pytest.fixture(scope="module")
def full_problem():
    return replace(coordinator_failure(), stage="decision-validation", trace_id="trace-original",
                   diagnostic_ref="art_previous_diagnostic", dependency_refs=("job_dependency",),
                   retry_after_seconds=37)


def assert_summary(original, summary, ref):
    assert (summary.code, summary.category, summary.retryable) == (
        original.code, original.category, original.retryable)
    assert summary.schema_version == "problem.v1.0"
    assert summary.diagnostic_ref == ref.artifact_id
    assert summary.stage is None and summary.trace_id is None
    assert summary.dependency_refs == () and summary.retry_after_seconds is None
    assert summary.details == {}
    assert summary.message == "failure details preserved in diagnostic artifact"
    assert len(catalog.dumps_failure(summary).encode()) <= CAP


def assert_original(store, original, ref):
    payload = catalog.dumps(original).encode()
    assert ref == artifact_reference(payload, "problem.v1.0")
    assert store.read_verified(ref) == payload
    assert catalog.load_json(Problem, payload.decode()) == original


@pytest.mark.parametrize("size", [512, CAP])
def test_small_and_exact_bound_problems_keep_identical_canonical_bytes(size):
    original = make_problem("RESOURCE_UNAVAILABLE", "small failure", stage="stage",
                            details={"padding": ""}, dependency_refs=("job_a",),
                            retry_after_seconds=19)
    original = replace(original, trace_id="trace", diagnostic_ref="art_existing")
    overhead = len(catalog.dumps(original).encode())
    original = replace(original, details={"padding": "x" * (size - overhead)})
    payload = catalog.dumps(original)
    assert len(payload.encode()) == size

    def forbidden(*args, **kwargs):
        pytest.fail("a bounded Problem must not publish or alter its existing diagnostic")

    summary, artifact = failure_diagnostics.prepare_failure(
        original, SimpleNamespace(publish_bytes=forbidden))
    assert summary is original and artifact is None
    assert catalog.dumps_failure(summary) == payload
    assert catalog.MAX_JOB_FAILURE_BYTES == CAP


@pytest.mark.parametrize("entrypoint", ["prepare", "storeless"])
@pytest.mark.parametrize("fill", ["x", "漢"], ids=["ascii", "multibyte"])
@pytest.mark.parametrize("excess", [0, 1], ids=["last-admitted-byte", "first-refused-byte"])
def test_failure_writer_exact_byte_boundaries(tmp_path, entrypoint, fill, excess):
    original, payload = exact_problem_bytes(CAP + excess, fill=fill)
    assert catalog.dumps(original) == payload
    store = ArtifactStore(tmp_path)
    if entrypoint == "prepare":
        summary, ref = failure_diagnostics.prepare_failure(original, store)
        if excess:
            assert_summary(original, summary, ref)
            assert ref == artifact_reference(payload.encode("utf-8"), "problem.v1.0")
            assert store.read_verified(ref) == payload.encode("utf-8")
            assert len(object_bytes(store)) == 1
        else:
            assert summary is original and ref is None
            assert catalog.dumps_failure(summary) == payload
            assert object_bytes(store) == {}
    else:
        if excess:
            with pytest.raises(OpsError) as captured:
                catalog.dumps_failure(original)
            assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
            assert captured.value.problem.retryable is False
            assert captured.value.problem.message == "failure requires a diagnostic artifact before persistence"
            assert isinstance(captured.value.__cause__, OpsError)
            assert captured.value.__cause__.problem is original
        else:
            assert catalog.dumps_failure(original) == payload
        assert object_bytes(store) == {}
    assert catalog.dumps(original) == payload


@pytest.mark.parametrize("field", [
    "details", "message", "unicode-message", "stage", "trace_id", "dependency_refs", "diagnostic_ref",
])
def test_every_unbounded_metadata_field_moves_intact_to_the_diagnostic(tmp_path, field):
    original = make_problem("RESOURCE_UNAVAILABLE", "original failure", stage="stage",
                            details={"ordinary": [1, "unchanged"]}, dependency_refs=("job_a",),
                            retry_after_seconds=37)
    original = replace(original, trace_id="trace", diagnostic_ref="art_previous")
    padding = "x" * CAP
    value = ({"details": {"large": padding}, "dependency_refs": (padding,),
              "unicode-message": "漢" * (CAP // 3 + 1)}.get(field, padding))
    original = replace(original, **{("message" if field == "unicode-message" else field): value})
    before = catalog.dumps(original)
    assert len(before.encode()) > CAP
    store = ArtifactStore(tmp_path)
    summary, ref = failure_diagnostics.prepare_failure(original, store)
    assert_summary(original, summary, ref)
    assert_original(store, original, ref)
    assert catalog.dumps(original) == before


def test_summary_message_is_fixed_and_contains_no_original_metadata(tmp_path):
    store = ArtifactStore(tmp_path)
    first = make_problem("VALIDATION_FAILED", "private-original-message",
                         details={"padding": "x" * CAP})
    second = replace(first, message="a-different-private-message", stage="private-stage")
    summary_a, _ = failure_diagnostics.prepare_failure(first, store)
    summary_b, _ = failure_diagnostics.prepare_failure(second, store)
    assert summary_a.message == summary_b.message
    assert "private" not in summary_a.message


@pytest.mark.parametrize("oversized", [False, True])
def test_explicit_retryability_override_is_preserved(tmp_path, oversized):
    original = replace(make_problem("VALIDATION_FAILED", "explicit retry decision"), retryable=True)
    if oversized:
        original = replace(original, details={"padding": "x" * CAP})
    store = ArtifactStore(tmp_path)
    summary, ref = failure_diagnostics.prepare_failure(original, store)
    assert summary.retryable is True
    assert (summary.code, summary.category) == (original.code, original.category)
    if oversized:
        assert_summary(original, summary, ref)
        assert_original(store, original, ref)
    else:
        assert summary is original and ref is None
        assert catalog.dumps_failure(original) == catalog.dumps(original)


@pytest.mark.parametrize("changes", [
    {"code": "NOT_REGISTERED"}, {"code": None}, {"code": []}, {"category": "internal"},
    {"schema_version": "problem.v99.0"}, {"retryable": 0}, {"retryable": 1},
])
@pytest.mark.parametrize("oversized", [False, True], ids=["small", "large"])
@pytest.mark.parametrize("entrypoint", ["prepare", "service"])
def test_invalid_problem_classification_refuses_before_publication(case, monkeypatch, changes, oversized, entrypoint):
    original = replace(make_problem("VALIDATION_FAILED", "invalid envelope",
                                    details={"padding": "x" * CAP} if oversized else {}), **changes)

    def forbidden(*args, **kwargs):
        pytest.fail("an invalid failure classification must not be serialized or published")

    monkeypatch.setattr(case.store, "publish_bytes", forbidden)
    monkeypatch.setattr(failure_diagnostics, "dumps", forbidden)
    before = catalog_rows(case.conn)
    with pytest.raises(OpsError) as captured:
        if entrypoint == "prepare":
            failure_diagnostics.prepare_failure(original, case.store)
        else:
            case.service._commit_failure(case.claim, {"exit_code": 4}, original)
    assert captured.value.code == "INTEGRITY_FAILED"
    assert captured.value.problem.retryable is False
    assert captured.value.problem.message == "failure diagnostic requires a valid Problem classification"
    assert isinstance(captured.value.__cause__, OpsError)
    assert captured.value.__cause__.problem is original
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction
    assert object_bytes(case.store) == {}


def test_service_keeps_real_coordinator_findings_and_previous_reference(case):
    inputs, problems = coordinator_inputs(), []

    def coordinator_validation(*unused):
        try:
            validate([], **inputs)
        except OpsError as exc:
            original = replace(exc.problem, stage="decision-validation", trace_id="trace-original",
                               diagnostic_ref="art_previous_diagnostic",
                               dependency_refs=("job_dependency",), retry_after_seconds=37)
            problems.append(original)
            raise OpsError(original) from exc

    case.service._commit_success = coordinator_validation
    case.service._finish(SimpleNamespace(claim=case.claim), {"exit_code": 0})
    assert len(problems) == 1
    original = problems[0]
    assert len(original.details["findings"]) == 12000
    receipt = get_job(case.conn, case.claim.job_id)
    summary = receipt.failure
    ref = registered_artifact(case.conn, summary.diagnostic_ref)
    assert_summary(original, summary, ref)
    assert_original(case.store, original, ref)
    assert receipt.state == "failed" and not receipt.failure.retryable
    for table in ("jobs", "attempts"):
        row = case.conn.execute(f"SELECT state, failure_json FROM {table}").fetchone()
        assert tuple(row) == ("failed", catalog.dumps(summary))
    registration = case.conn.execute("SELECT producer_attempt_id FROM artifacts").fetchone()
    assert registration[0] == case.claim.attempt_id
    assert len(object_bytes(case.store)) == 1
    assert not case.conn.in_transaction


@pytest.mark.parametrize("retryable", [False, True])
def test_service_preserves_small_failure_bytes_and_retry_semantics(case, retryable):
    original = make_problem("RESOURCE_UNAVAILABLE" if retryable else "VALIDATION_FAILED",
                            "small original failure", details={"reason": "unchanged"},
                            stage="coordinator", dependency_refs=("job_a",))
    original = replace(original, diagnostic_ref="art_preexisting", trace_id="trace")
    case.service._commit_failure(case.claim, {"exit_code": 4}, original)
    assert get_job(case.conn, case.claim.job_id).state == ("retry_wait" if retryable else "failed")
    for table in ("jobs", "attempts"):
        assert case.conn.execute(f"SELECT failure_json FROM {table}").fetchone()[0] == catalog.dumps(original)
    assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert object_bytes(case.store) == {}


def test_large_retryable_failure_keeps_retry_policy_and_full_problem(case):
    original = make_problem("RESOURCE_UNAVAILABLE", "retryable coordinator failure",
                            details={"padding": "x" * CAP}, retry_after_seconds=19)
    case.service._commit_failure(case.claim, {"exit_code": 4}, original)
    receipt = get_job(case.conn, case.claim.job_id)
    assert receipt.state == "retry_wait" and receipt.failure.retryable
    # The fixture's first-attempt retry delay is one second, not retry_after_seconds=19.
    assert receipt.next_eligible_at == format_timestamp(case.clock.now() + timedelta(seconds=1))
    ref = registered_artifact(case.conn, receipt.failure.diagnostic_ref)
    assert_summary(original, receipt.failure, ref)
    assert_original(case.store, original, ref)


@pytest.mark.parametrize("writer", ["serialize", "commit", "end-attempt", "advance-job"])
def test_storeless_writers_refuse_oversized_problem_without_catalog_mutation(case, full_problem, writer):
    before = catalog_rows(case.conn)
    outcome = lifecycle.Outcome(False, "verified_dead", 4, full_problem)
    with pytest.raises(OpsError) as captured:
        if writer == "serialize":
            catalog.dumps_failure(full_problem)
        elif writer == "commit":
            lifecycle.commit_attempt(case.conn, case.claim.attempt_id, case.claim.fence, outcome,
                                     clock=case.clock)
        else:
            with catalog.transaction(case.conn):
                if writer == "end-attempt":
                    lifecycle.end_attempt(case.conn, case.claim.attempt_id, "failed", outcome,
                                          case.clock.now())
                else:
                    job = case.conn.execute("SELECT * FROM jobs").fetchone()
                    lifecycle.advance_job(case.conn, job, outcome, case.clock.now())
    assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert captured.value.problem.retryable is False
    assert captured.value.problem.message == "failure requires a diagnostic artifact before persistence"
    assert isinstance(captured.value.__cause__, OpsError)
    assert captured.value.__cause__.problem is full_problem
    assert catalog_rows(case.conn) == before
    assert not case.conn.in_transaction
    assert object_bytes(case.store) == {}


@pytest.mark.parametrize("error,code", [
    (OSError("private storage detail"), "RESOURCE_UNAVAILABLE"),
    (ArtifactError("HASH_MISMATCH", "private artifact detail"), "INTEGRITY_FAILED"),
])
def test_failed_initial_publication_never_commits_or_loses_original_error(
        case, full_problem, monkeypatch, error, code):
    before = catalog_rows(case.conn)

    def unavailable(*args, **kwargs):
        assert not case.conn.in_transaction
        raise error

    monkeypatch.setattr(case.store, "publish_bytes", unavailable)
    with pytest.raises(OpsError) as captured:
        case.service._commit_failure(case.claim, {"exit_code": 4}, full_problem)
    assert isinstance(captured.value, failure_diagnostics.DiagnosticPublicationError)
    assert captured.value.code == code
    assert captured.value.problem.message == "failure diagnostic could not be published"
    assert "private" not in captured.value.problem.message
    assert isinstance(captured.value.__cause__, OpsError)
    assert captured.value.__cause__.problem is full_problem
    assert catalog_rows(case.conn) == before
    assert object_bytes(case.store) == {}
    assert not case.conn.in_transaction


def fail_at_supervisor_boundary(case, monkeypatch, problem, entrypoint):
    def failure(*args, **kwargs):
        raise OpsError(problem)

    if entrypoint == "finish":
        monkeypatch.setattr(case.service, "_commit_success", failure)
        return lambda: case.service._finish(SimpleNamespace(claim=case.claim), {"exit_code": 0})
    case.service.code_source = case.path.parent
    monkeypatch.setattr(supervisor_module, "worker_source_manifest", failure)
    return lambda: case.service._launch(case.claim)


@pytest.mark.parametrize("entrypoint", ["finish", "launch"])
@pytest.mark.parametrize("error,code", [
    (OSError("private storage detail"), "RESOURCE_UNAVAILABLE"),
    (ArtifactError("HASH_MISMATCH", "private artifact detail"), "INTEGRITY_FAILED"),
])
def test_tick_contains_publication_failure_and_preserves_other_workers(
        case, full_problem, monkeypatch, capsys, entrypoint, error, code):
    service, conn, clock = case.service, case.conn, case.clock
    identity = Supervisor(conn.execute("SELECT supervisor_epoch FROM attempts").fetchone()[0], "boot")
    other = enqueue_claim(conn, clock, identity, key="other")
    assert other is not None
    service.boot, service.root = "boot", case.path.parent
    service.policy, service.registry, service.identity = DEFAULT_POLICY, REGISTRY, identity
    service.observed, service.running = {}, {}
    for name in ("_clock_check", "_reclaim_terminal_staging", "_reconcile_publication_status",
                 "_reconcile_computed_moves_refresh", "_reconcile_native_score_batch_shadow",
                 "_reconcile_native_parity", "_reconcile_pool_nightly_refresh"):
        monkeypatch.setattr(service, name, lambda: None)
    monkeypatch.setattr(service, "_track_steps", lambda *args: None)
    monkeypatch.setattr(service, "_observation_due", lambda *args: False)
    monkeypatch.setattr(supervisor_module, "sample_capacity", lambda *args, **kwargs: sample(clock))
    monkeypatch.setattr(supervisor_module, "prove_ownership_gone", lambda *args, **kwargs:
                        recovery.OwnershipProof(True, (), (), ()))
    fail_at_supervisor_boundary(case, monkeypatch, full_problem, entrypoint)
    claims = iter([case.claim, None, None] if entrypoint == "launch" else [None, None, None])
    monkeypatch.setattr(supervisor_module, "claim_next", lambda *args, **kwargs: next(claims))
    stopped, polled, stranded = [], [], []
    monkeypatch.setattr(supervisor_module.executor, "stop", lambda *args, **kwargs: stopped.append(args))
    real_strand = service._strand

    def strand(claim, problem):
        stranded.append((claim, problem))
        real_strand(claim, problem)

    monkeypatch.setattr(service, "_strand", strand)

    def unavailable(*args, **kwargs):
        assert not conn.in_transaction
        raise error

    monkeypatch.setattr(case.store, "publish_bytes", unavailable)

    def poll(conn, running, **kwargs):
        polled.append(running.claim.attempt_id)
        return {"done": running.claim is case.claim, "exit_code": 0, "memory": 0}

    monkeypatch.setattr(supervisor_module.executor, "poll", poll)
    read_fd, write_fd = os.pipe()
    try:
        if entrypoint == "finish":
            service.running[case.claim.attempt_id] = SimpleNamespace(
                claim=case.claim, failure=None, result_fd=read_fd)
            service.launches[case.claim.attempt_id] = object()
            service._progress_state[case.claim.attempt_id] = object()
        survivor = SimpleNamespace(claim=other, failure=None)
        service.running[other.attempt_id] = survivor
        assert service.tick() is True
        assert service.running == {other.attempt_id: survivor} and stopped == []
        assert case.claim.attempt_id not in service.launches
        assert case.claim.attempt_id not in service._progress_state
        assert polled.count(other.attempt_id) == 1
        assert len(stranded) == 1 and stranded[0][0] is case.claim
        assert stranded[0][1].code == code
        report = json.loads(capsys.readouterr().err)
        assert report == {"event": "attempt_left_for_recovery", "job_id": case.claim.job_id,
                          "attempt_id": case.claim.attempt_id,
                          "problem": {"code": code, "category": stranded[0][1].category,
                                      "retryable": stranded[0][1].retryable,
                                      "message": "failure diagnostic could not be published"}}
        attempt = conn.execute("SELECT state, ended_at, failure_json FROM attempts WHERE attempt_id=?",
                               (case.claim.attempt_id,)).fetchone()
        assert tuple(attempt) == ("starting", None, None)
        assert get_job(conn, case.claim.job_id).state == "running"
        assert conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert object_bytes(case.store) == {} and not conn.in_transaction
        assert conn.execute("SELECT COUNT(*) FROM resource_reservations WHERE released_at IS NULL").fetchone()[0] == 2

        # The failed attempt receives no more heartbeats. Its peer continues,
        # and ordinary lease expiry/reconciliation settles only the abandoned one.
        clock.advance(supervisor_module.LEASE_SECONDS - 1)
        assert service.tick() is True
        clock.advance(2)
        assert service.tick() is True
        assert service.running == {other.attempt_id: survivor} and stopped == []
        assert polled.count(other.attempt_id) == 3
        assert get_job(conn, other.job_id).state == "running"
        receipt = get_job(conn, case.claim.job_id)
        assert receipt.state == "retry_wait" and receipt.failure.code == "LEASE_LOST"
        assert conn.execute("SELECT COUNT(*) FROM resource_reservations WHERE released_at IS NULL").fetchone()[0] == 1
    finally:
        if entrypoint == "launch" or case.claim.attempt_id in service.running:
            os.close(read_fd)
        os.close(write_fd)


@pytest.mark.parametrize("entrypoint", ["finish", "launch"])
@pytest.mark.parametrize("error", [OSError("storage unavailable"), ArtifactError("HASH_MISMATCH", "mismatch")])
def test_failed_publication_after_lease_expiry_strands_without_completion(
        case, full_problem, monkeypatch, entrypoint, error):
    boundary = fail_at_supervisor_boundary(case, monkeypatch, full_problem, entrypoint)

    def slow_failure(*args, **kwargs):
        case.clock.advance(supervisor_module.LEASE_SECONDS + 1)
        raise error

    monkeypatch.setattr(case.store, "publish_bytes", slow_failure)
    boundary()
    job = case.conn.execute("SELECT state, fence, failure_json FROM jobs").fetchone()
    attempt = case.conn.execute("SELECT state, ended_at, failure_json FROM attempts").fetchone()
    assert tuple(job) == ("running", case.claim.fence + 1, None)
    assert tuple(attempt) == ("recovery_pending", None, None)
    assert case.conn.execute("SELECT released_at FROM resource_reservations").fetchone()[0] is None
    assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert object_bytes(case.store) == {} and not case.conn.in_transaction


@pytest.mark.parametrize("entrypoint", ["finish", "launch"])
@pytest.mark.parametrize("phase", ["classification", "publication", "commit-storage", "commit-integrity"])
def test_supervisor_boundary_does_not_swallow_unrelated_errors(case, full_problem, monkeypatch, entrypoint, phase):
    original = (RuntimeError("unexpected publication error") if phase == "publication" else
                OpsError(make_problem("RESOURCE_UNAVAILABLE" if phase == "commit-storage" else
                                      "INTEGRITY_FAILED", "unrelated commit error")))
    problem = replace(full_problem, category="invalid") if phase == "classification" else full_problem
    boundary = fail_at_supervisor_boundary(case, monkeypatch, problem, entrypoint)

    def broken(*args, **kwargs):
        raise original

    if phase == "publication":
        monkeypatch.setattr(case.store, "publish_bytes", broken)
    elif phase.startswith("commit"):
        monkeypatch.setattr(supervisor_module, "commit_attempt", broken)
    monkeypatch.setattr(case.service, "_strand", lambda *args: pytest.fail("not a publication refusal"))
    before = catalog_rows(case.conn)
    with pytest.raises((OpsError, RuntimeError)) as captured:
        boundary()
    if phase == "classification":
        assert captured.value.code == "INTEGRITY_FAILED"
        assert captured.value.problem.message == "failure diagnostic requires a valid Problem classification"
        assert captured.value.__cause__.problem is problem
    else:
        assert captured.value is original
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction


@pytest.mark.parametrize("mismatch", ["reference", "schema", "success"])
def test_failure_artifact_must_match_the_failed_outcome(case, full_problem, mismatch):
    summary, ref = failure_diagnostics.prepare_failure(full_problem, case.store)
    before = catalog_rows(case.conn)
    if mismatch == "reference":
        summary = replace(summary, diagnostic_ref="art_other")
    elif mismatch == "schema":
        ref = replace(ref, schema_ref="other.v1.0")
    outcome = (lifecycle.Outcome(True, "verified_dead", 0) if mismatch == "success" else
               lifecycle.Outcome(False, "verified_dead", 4, summary))
    with pytest.raises(OpsError) as captured:
        lifecycle.commit_attempt(case.conn, case.claim.attempt_id, case.claim.fence, outcome,
                                 clock=case.clock, failure_artifact=ref)
    assert captured.value.code == "INTEGRITY_FAILED"
    assert captured.value.problem.message == "failure diagnostic does not match its receipt"
    assert catalog_rows(case.conn) == before
    assert not case.conn.in_transaction


def test_fence_lost_after_publication_leaves_only_an_unregistered_complete_object(case, full_problem):
    summary, ref = failure_diagnostics.prepare_failure(full_problem, case.store)
    case.conn.execute("UPDATE jobs SET fence=fence+1")
    before = catalog_rows(case.conn)
    with pytest.raises(OpsError) as captured:
        lifecycle.commit_attempt(case.conn, case.claim.attempt_id, case.claim.fence,
                                 lifecycle.Outcome(False, "verified_dead", 4, summary),
                                 clock=case.clock, failure_artifact=ref)
    assert captured.value.code == "LEASE_LOST"
    assert captured.value.problem.message == "the attempt no longer holds the job's fence"
    assert catalog_rows(case.conn) == before
    assert_original(case.store, full_problem, ref)
    assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert not case.conn.in_transaction


def test_service_lease_expiry_during_publication_uses_existing_recovery(case, full_problem, monkeypatch):
    publish = case.store.publish_bytes
    published = []

    def slow_publication(*args, **kwargs):
        assert not case.conn.in_transaction
        ref = publish(*args, **kwargs)
        published.append(ref)
        case.clock.advance(3600)
        return ref

    monkeypatch.setattr(case.store, "publish_bytes", slow_publication)
    case.service._commit_failure(case.claim, {"exit_code": 4}, full_problem)
    job = case.conn.execute("SELECT state, fence, failure_json FROM jobs").fetchone()
    attempt = case.conn.execute("SELECT state, ended_at, failure_json FROM attempts").fetchone()
    assert tuple(job) == ("running", case.claim.fence + 1, None)
    assert tuple(attempt) == ("recovery_pending", None, None)
    assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
    assert len(published) == 1
    assert_original(case.store, full_problem, published[0])
    assert not case.conn.in_transaction


def test_final_transaction_failure_rolls_back_registration_and_completion(case, full_problem):
    summary, ref = failure_diagnostics.prepare_failure(full_problem, case.store)
    before = catalog_rows(case.conn)

    class RefuseCommit:
        def __getattr__(self, name):
            return getattr(case.conn, name)

        def execute(self, sql, *args, **kwargs):
            if sql == "COMMIT":
                assert case.conn.in_transaction
                assert registered_artifact(case.conn, ref.artifact_id) == ref
                assert get_job(case.conn, case.claim.job_id).state == "failed"
                raise RuntimeError("injected final transaction failure")
            return case.conn.execute(sql, *args, **kwargs)

    with pytest.raises(RuntimeError, match="injected final transaction failure"):
        lifecycle.commit_attempt(RefuseCommit(), case.claim.attempt_id, case.claim.fence,
                                 lifecycle.Outcome(False, "verified_dead", 4, summary),
                                 clock=case.clock, failure_artifact=ref)
    assert catalog_rows(case.conn) == before
    assert not case.conn.in_transaction
    assert_original(case.store, full_problem, ref)
    case.service._commit_failure(case.claim, {"exit_code": 4}, full_problem)
    assert registered_artifact(case.conn, ref.artifact_id) == ref
    assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 1
    assert len(object_bytes(case.store)) == 1


def test_complete_publication_orphan_survives_reopen_and_idempotent_republication(case, full_problem):
    class SimulatedCrash(BaseException):
        pass

    def crash(point):
        assert not case.conn.in_transaction
        if point == "linked":
            raise SimulatedCrash()

    before = catalog_rows(case.conn)
    case.service.store = ArtifactStore(case.store.root, fault=crash)
    with pytest.raises(SimulatedCrash):
        case.service._commit_failure(case.claim, {"exit_code": 4}, full_problem)
    assert catalog_rows(case.conn) == before
    ref = artifact_reference(catalog.dumps(full_problem).encode(), "problem.v1.0")
    assert_original(case.store, full_problem, ref)
    objects = object_bytes(case.store)
    assert len(objects) == 1
    case.conn.close()
    case.conn = catalog.connect(case.path, must_exist=True)
    store = ArtifactStore(case.store.root)
    assert catalog_rows(case.conn) == before
    assert_original(store, full_problem, ref)
    # Re-publishing known original bytes is idempotent. This does not claim
    # a supervisor restart automatically discovers or retries orphaned errors.
    summary, repeated = failure_diagnostics.prepare_failure(full_problem, store)
    assert repeated == ref and object_bytes(store) == objects
    assert_summary(full_problem, summary, repeated)
    assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0


@pytest.mark.parametrize("writer", ["blocked-insert", "descendants", "cancel", "reconcile", "reconcile-cancel", "scheduler"])
def test_generated_failure_writers_share_the_strict_ceiling_and_atomic_refusal(case, monkeypatch, writer):
    """Inject a large generated envelope at each writer boundary, without huge fixture catalogs."""
    module = {"blocked-insert": submission, "descendants": lifecycle, "cancel": lifecycle,
              "reconcile": recovery, "reconcile-cancel": recovery, "scheduler": scheduler}[writer]
    if writer in {"cancel", "reconcile-cancel"}:
        lifecycle.request_cancel(case.conn, case.claim.job_id, case.claim.attempt_id, clock=case.clock)
    if writer.startswith("reconcile"):
        case.clock.advance(121)
        recovery.expire_leases(case.conn, clock=case.clock)
    if writer == "blocked-insert":
        case.conn.execute("UPDATE jobs SET state='failed', active_attempt_id=NULL")
    if writer == "descendants":
        submission.submit(case.conn, REGISTRY, POLICY,
                          request("child", dependency_job_ids=(case.claim.job_id,)), clock=case.clock)
    if writer == "scheduler":
        lifecycle.commit_attempt(case.conn, case.claim.attempt_id, case.claim.fence,
                                 lifecycle.Outcome(True, "verified_dead", 0), clock=case.clock)
        submission.submit(case.conn, REGISTRY, POLICY, request("queued", resource_class="legacy_score"),
                          clock=case.clock)
        def low():
            return replace(sample(case.clock), host_available_bytes=5 << 30)
        supervisor = Supervisor(case.conn.execute("SELECT supervisor_epoch FROM attempts").fetchone()[0], "boot")
        assert scheduler.claim_next(case.conn, policy=DEFAULT_POLICY, sample=low(),
                                    supervisor=supervisor, clock=case.clock) is None
        case.clock.advance(scheduler.HEADROOM_CEILING_WINDOW_SECONDS + 1)
    generated = []

    def oversized(*args, **kwargs):
        problem = replace(make_problem(*args, **kwargs), details={"padding": "x" * CAP})
        generated.append(problem)
        return problem

    monkeypatch.setattr(module, "make_problem", oversized)
    before = catalog_rows(case.conn)
    with pytest.raises(OpsError) as captured:
        if writer == "blocked-insert":
            submission.submit_graph(case.conn, REGISTRY, POLICY, [request("earlier"),
                request("blocked", dependency_job_ids=(case.claim.job_id,))], clock=case.clock)
        elif writer == "descendants":
            with catalog.transaction(case.conn):
                lifecycle.block_descendants(case.conn, case.claim.job_id, case.clock.now())
        elif writer == "cancel":
            lifecycle.complete_cancel(case.conn, case.claim.job_id, process_state="verified_dead", clock=case.clock)
        elif writer.startswith("reconcile"):
            recovery.reconcile_attempt(case.conn, case.claim.attempt_id, process_state="verified_dead", clock=case.clock)
        else:
            scheduler.claim_next(case.conn, policy=DEFAULT_POLICY, sample=low(),
                                 supervisor=supervisor, clock=case.clock)
    assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert captured.value.problem.message == "failure requires a diagnostic artifact before persistence"
    assert isinstance(captured.value.__cause__, OpsError)
    assert captured.value.__cause__.problem is generated[0]
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction
    assert object_bytes(case.store) == {}


def test_registration_conflict_rolls_back_failure_completion(case, full_problem):
    summary, ref = failure_diagnostics.prepare_failure(full_problem, case.store)
    with catalog.transaction(case.conn):
        register_artifact(case.conn, replace(ref, schema_ref="conflicting.v1.0"), case.claim.attempt_id, case.clock)
    before = catalog_rows(case.conn)
    with pytest.raises(OpsError) as captured:
        lifecycle.commit_attempt(case.conn, case.claim.attempt_id, case.claim.fence,
                                 lifecycle.Outcome(False, "verified_dead", 4, summary),
                                 clock=case.clock, failure_artifact=ref)
    assert captured.value.code == "INTEGRITY_FAILED"
    assert captured.value.problem.message == "artifact identity conflict"
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction
    assert_original(case.store, full_problem, ref)


@pytest.mark.parametrize("registrar", [catalog.register_artifact, register_artifact], ids=["catalog", "checkpoints"])
def test_shared_registration_preserves_bytes_provenance_and_noop_clock(case, registrar):
    ref = case.store.publish_bytes(b"diagnostic", schema_ref="problem.v1.0")
    expected = (ref.artifact_id, catalog.dumps(ref), case.claim.attempt_id, format_timestamp(case.clock.now()))
    with catalog.transaction(case.conn):
        assert registrar(case.conn, ref, case.claim.attempt_id, case.clock) is None
    assert tuple(case.conn.execute("SELECT * FROM artifacts").fetchone()) == expected
    before = catalog_rows(case.conn)

    def forbidden_clock():
        pytest.fail("existing registration must not read the clock")

    clock = SimpleNamespace(now=forbidden_clock)
    with catalog.transaction(case.conn):
        assert registrar(case.conn, ref, None, clock) is None
    assert catalog_rows(case.conn) == before
    with pytest.raises(OpsError) as captured, catalog.transaction(case.conn):
        registrar(case.conn, replace(ref, schema_ref="conflicting.v1.0"), None, clock)
    assert captured.value.code == "INTEGRITY_FAILED"
    assert captured.value.problem.message == "artifact identity conflict"
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction


@pytest.mark.parametrize("registrar", [catalog.register_artifact, register_artifact], ids=["catalog", "checkpoints"])
def test_shared_registration_rolls_back_with_its_callers_transaction(case, registrar):
    ref = case.store.publish_bytes(b"diagnostic", schema_ref="problem.v1.0")
    before = catalog_rows(case.conn)
    original = RuntimeError("caller commit failed")
    with pytest.raises(RuntimeError) as captured, catalog.transaction(case.conn):
        registrar(case.conn, ref, case.claim.attempt_id, case.clock)
        assert case.conn.in_transaction
        assert tuple(case.conn.execute("SELECT * FROM artifacts").fetchone()) == (
            ref.artifact_id, catalog.dumps(ref), case.claim.attempt_id, format_timestamp(case.clock.now()))
        raise original
    assert captured.value is original
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction


def test_checkpoint_registration_wrapper_preserves_arguments_and_exception_identity(case, monkeypatch):
    ref = case.store.publish_bytes(b"diagnostic", schema_ref="problem.v1.0")
    original = RuntimeError("unexpected catalog error")

    def broken(conn, supplied_ref, attempt_id, clock):
        assert conn is case.conn and supplied_ref is ref and clock is case.clock
        assert attempt_id == case.claim.attempt_id
        raise original

    monkeypatch.setattr(catalog, "register_artifact", broken)
    before = catalog_rows(case.conn)
    with pytest.raises(RuntimeError) as captured:
        register_artifact(case.conn, ref, case.claim.attempt_id, case.clock)
    assert captured.value is original
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction


def test_unexpected_publication_error_propagates_unchanged(case, full_problem, monkeypatch):
    original = RuntimeError("unexpected delegate error")

    def broken(*args, **kwargs):
        raise original

    monkeypatch.setattr(case.store, "publish_bytes", broken)
    before = catalog_rows(case.conn)
    with pytest.raises(RuntimeError) as captured:
        case.service._commit_failure(case.claim, {"exit_code": 4}, full_problem)
    assert captured.value is original
    assert catalog_rows(case.conn) == before and not case.conn.in_transaction
