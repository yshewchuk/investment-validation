"""Issue #125: ``models_rollback`` -- a bounded production deployment rollback.

These tests drive the real operator plan/submit/supervisor/worker path against
a synthetic release store staged and promoted with the real ``deployment``
APIs; nothing stubs ``deployment.rollback``, dispatch, submission or pointer
storage. They mirror the promote e2e in ``tests/test_v2_ops_training_job.py``
and its ``_release_fixture``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.v2.foundation import SystemClock, content_hash
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
    deployment,
)
from engine.v2.ops import cli, stages, training
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.plans import request_from_plan
from engine.v2.ops.submission import NamespacePolicy, get_job, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, run_until

REPO = Path(__file__).resolve().parents[3]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _release_fixture(release_id, *, intercept, coefficient):
    """A one-binding synthetic ``ModelRelease`` plus its matching inventory,
    the same shape ``tests/test_v2_ops_training_job.py`` and
    ``tests/v2/ops/test_training_promote_guard.py`` stage."""
    payload = json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": intercept,
                     "coefficients": [coefficient]}],
    }, sort_keys=True).encode()
    member_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))

    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    inv_artifact = ModelArtifactInventory(
        artifact_id="m1", role="size", strategy_ids=("*",), compatible_clock_ids=("entry-close",),
        target_contract_ref="return.v1", ordered_features=("x",), members=(inv_member,))
    inv_binding = ReleaseBinding(
        role="size", strategy_id="*", clock_id="entry-close", artifact_id="m1",
        ordered_features=("x",), required_member_kinds=("estimator",))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(inv_artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def _stage_one(root, release_id, *, intercept, coefficient):
    release, inventory, payloads = _release_fixture(
        release_id, intercept=intercept, coefficient=coefficient)
    deployment.stage_release(root, release, inventory, payloads)
    deployment.mark_staging_succeeded(root, release_id)


def _stage_pair(root):
    """Stage and mark two distinct synthetic releases ``r1`` and ``r2``."""
    _stage_one(root, "r1", intercept=1.0, coefficient=2.0)
    _stage_one(root, "r2", intercept=10.0, coefficient=20.0)


def _service(conn, root, clock):
    service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=root)
    service.start()
    return service


def _failure_report(service, conn, job_id):
    failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                           (job_id,)).fetchone()[0]
    attempt = conn.execute("SELECT attempt_id FROM attempts WHERE job_id = ? "
                           "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    diagnostic = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics"
    stderr = diagnostic / "worker.stderr"
    return f"{failure}\n{stderr.read_text() if stderr.is_file() else '(no stderr)'}"


def _output_refs(conn, job_id):
    return {row["name"]: row["artifact_id"] for row in conn.execute(
        "SELECT ao.name, ao.artifact_id FROM attempt_outputs ao JOIN attempts a "
        "ON a.attempt_id = ao.attempt_id WHERE a.job_id = ?", (job_id,)).fetchall()}


def _history_bytes(root):
    return {path.name: path.read_bytes() for path in sorted((root / "history").glob("*.json"))}


def test_rollback_through_supervisor_restores_prior_release(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    try:
        _stage_pair(tmp_path)
        deployment.promote(tmp_path, "r1")
        deployment.promote(tmp_path, "r2")
        assert deployment.current_pointer(tmp_path).release_id == "r2"
        assert deployment.rollback_target(tmp_path) == "r1"

        plan = training.rollback_plan(release_root=str(tmp_path))
        assert plan["parameters"]["incumbent_release_id"] == "r2"
        assert plan["parameters"]["incumbent_sequence"] == 1
        assert plan["parameters"]["target_release_id"] == "r1"
        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "rollback-e2e"), clock=clock)
        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, receipt.job_id, timeout=90)
            assert state == "succeeded", _failure_report(service, conn, receipt.job_id)
            ref_id = _output_refs(conn, receipt.job_id)["pointer_state"]
            document = json.loads(service.store.read_verified(
                artifact(conn, service.store, ref_id)))
        finally:
            service.close()

        assert document["action"] == "rollback"
        assert document["release_id"] == "r1"
        assert document["previous_release_id"] == "r2"
        assert deployment.current_pointer(tmp_path).release_id == "r1"
        history = list(deployment.pointer_history(tmp_path))
        assert [entry.action for entry in history] == ["promote", "promote", "rollback"]
        assert history[-1].release_id == "r1"
        assert history[-1].previous_release_id == "r2"
    finally:
        conn.close()


def test_rollback_with_no_prior_release_is_typed_validation_failed(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    try:
        _stage_pair(tmp_path)
        deployment.promote(tmp_path, "r1")  # exactly one history entry: nothing to undo
        before_pointer = (tmp_path / "DEPLOYED").read_bytes()
        before_history = _history_bytes(tmp_path)
        with pytest.raises(deployment.NoPriorRelease):
            deployment.rollback_target(tmp_path)

        plan = training.rollback_plan(release_root=str(tmp_path))
        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "rollback-none"), clock=clock)
        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, receipt.job_id, timeout=90)
            assert state == "failed", _failure_report(service, conn, receipt.job_id)
            job = get_job(conn, receipt.job_id)
            assert job.failure.code == "VALIDATION_FAILED"
            assert job.failure.retryable is False
            assert job.failure.details == {}
            assert job.failure.diagnostic_ref is not None
            ref = artifact(conn, service.store, job.failure.diagnostic_ref)
            details = json.loads(service.store.read_verified(ref))
            assert details == {"exception_class": "NoPriorRelease"}
            assert _output_refs(conn, receipt.job_id) == {}
        finally:
            service.close()

        assert (tmp_path / "DEPLOYED").read_bytes() == before_pointer
        assert _history_bytes(tmp_path) == before_history
        assert deployment.current_pointer(tmp_path).release_id == "r1"
        assert len(deployment.pointer_history(tmp_path)) == 1
    finally:
        conn.close()


def test_rollback_through_supervisor_on_empty_store_is_typed_no_prior_release(tmp_path):
    """A store with no pointer at all still reaches ``deployment.rollback``.

    Nothing was ever promoted, so the plan pins ``(None, None, None)``. The
    live pointer is still absent, so the worker must let the real
    ``deployment.rollback`` run and surface its typed ``NoPriorRelease`` --
    never a bare worker failure -- and leave no pointer or history behind.
    """
    conn, clock, _ = catalog(tmp_path)
    try:
        with pytest.raises(deployment.NoPriorRelease):
            deployment.rollback_target(tmp_path)
        plan = training.rollback_plan(release_root=str(tmp_path))
        assert plan["parameters"]["incumbent_release_id"] is None
        assert plan["parameters"]["incumbent_sequence"] is None
        assert plan["parameters"]["target_release_id"] is None

        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "rollback-empty"), clock=clock)
        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, receipt.job_id, timeout=90)
            assert state == "failed", _failure_report(service, conn, receipt.job_id)
            job = get_job(conn, receipt.job_id)
            assert job.failure.code == "VALIDATION_FAILED"
            assert job.failure.retryable is False
            assert job.failure.diagnostic_ref is not None
            ref = artifact(conn, service.store, job.failure.diagnostic_ref)
            details = json.loads(service.store.read_verified(ref))
            assert details == {"exception_class": "NoPriorRelease"}
            assert _output_refs(conn, receipt.job_id) == {}
        finally:
            service.close()

        assert deployment.current_pointer(tmp_path) is None
        assert not (tmp_path / "history").exists()
    finally:
        conn.close()


def test_unreadable_rollback_history_at_plan_time_is_typed_validation_failed(tmp_path):
    """A corrupt history file refuses the plan, typed, without mutating the store.

    Real stage/promote APIs build an incumbent plus a prior release, then the
    store's actual history JSON is corrupted. The real ``rollback_plan`` must
    surface the resolver's own ``StagingRefused`` (``HISTORY_UNREADABLE``) as
    typed ``VALIDATION_FAILED`` -- never pinning a bogus target -- and leave
    both the live pointer and the corrupt history bytes exactly as they were.
    """
    _stage_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    deployment.promote(tmp_path, "r2")
    history_path = sorted((tmp_path / "history").glob("*.json"))[0]
    history_path.write_bytes(b"{ not json")
    pointer_before = (tmp_path / "DEPLOYED").read_bytes()
    history_before = _history_bytes(tmp_path)

    with pytest.raises(OpsError) as excinfo:
        training.rollback_plan(release_root=str(tmp_path))
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.details == {"exception_class": "StagingRefused"}

    assert (tmp_path / "DEPLOYED").read_bytes() == pointer_before
    assert _history_bytes(tmp_path) == history_before
    assert deployment.current_pointer(tmp_path).release_id == "r2"


def test_rollback_resubmission_same_key_returns_same_job(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    try:
        _stage_pair(tmp_path)
        deployment.promote(tmp_path, "r1")
        deployment.promote(tmp_path, "r2")

        plan = training.rollback_plan(release_root=str(tmp_path))
        first = submit(conn, stages.registry(), POLICY,
                       request_from_plan(plan, "rollback-idem"), clock=clock)
        before = submit(conn, stages.registry(), POLICY,
                        request_from_plan(plan, "rollback-idem"), clock=clock)
        assert before.job_id == first.job_id

        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, first.job_id, timeout=90)
            assert state == "succeeded", _failure_report(service, conn, first.job_id)
            history_after = list(deployment.pointer_history(tmp_path))
            assert [entry.action for entry in history_after] == [
                "promote", "promote", "rollback"]

            after = submit(conn, stages.registry(), POLICY,
                           request_from_plan(plan, "rollback-idem"), clock=clock)
            assert after.job_id == first.job_id
            assert after.state == "succeeded"
            for _ in range(5):
                service.tick()
            assert list(deployment.pointer_history(tmp_path)) == history_after
            assert deployment.current_pointer(tmp_path).release_id == "r1"
        finally:
            service.close()
    finally:
        conn.close()


def test_stale_rollback_plan_after_unrelated_promotion_is_refused_without_mutation(tmp_path):
    """A plan made for r2->r1 must not undo a later promotion.

    The store advances by an ordinary promote to ``r3`` instead of the plan's
    rollback, so the live pointer no longer names the pinned incumbent and the
    history holds a ``promote`` (not a ``rollback``) at the pinned sequence plus
    one.  The job fails typed and neither pointer nor history moves.
    """
    conn, clock, _ = catalog(tmp_path)
    try:
        _stage_pair(tmp_path)
        _stage_one(tmp_path, "r3", intercept=30.0, coefficient=40.0)
        deployment.promote(tmp_path, "r1")
        deployment.promote(tmp_path, "r2")

        plan = training.rollback_plan(release_root=str(tmp_path))
        deployment.promote(tmp_path, "r3")
        before_pointer = (tmp_path / "DEPLOYED").read_bytes()
        before_history = _history_bytes(tmp_path)

        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "rollback-stale"), clock=clock)
        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, receipt.job_id, timeout=90)
            assert state == "failed", _failure_report(service, conn, receipt.job_id)
            job = get_job(conn, receipt.job_id)
            assert job.failure.code == "VALIDATION_FAILED"
            assert job.failure.retryable is False
            assert _output_refs(conn, receipt.job_id) == {}
        finally:
            service.close()

        assert (tmp_path / "DEPLOYED").read_bytes() == before_pointer
        assert _history_bytes(tmp_path) == before_history
        assert deployment.current_pointer(tmp_path).release_id == "r3"
        assert [entry.action for entry in deployment.pointer_history(tmp_path)] == [
            "promote", "promote", "promote"]
    finally:
        conn.close()


def test_lost_receipt_recovery_reports_original_rollback_and_keeps_later_promotion(tmp_path):
    """A resubmitted plan recognizes its already-recorded swap.

    The original attempt ran the real ``deployment.rollback`` (its worker
    receipt was lost), then a later ``promote`` moved the store on.  Submitting
    the saved plan under a NEW idempotency key must report that exact recorded
    rollback state and must not swap the pointer again -- leaving the later
    promotion and the whole history untouched.
    """
    conn, clock, _ = catalog(tmp_path)
    try:
        _stage_pair(tmp_path)
        _stage_one(tmp_path, "r3", intercept=30.0, coefficient=40.0)
        deployment.promote(tmp_path, "r1")
        deployment.promote(tmp_path, "r2")

        plan = training.rollback_plan(release_root=str(tmp_path))
        lost = deployment.rollback(tmp_path)
        assert (lost.action, lost.release_id, lost.previous_release_id) == ("rollback", "r1", "r2")
        deployment.promote(tmp_path, "r3")
        live_before = (tmp_path / "DEPLOYED").read_bytes()
        history_before = _history_bytes(tmp_path)
        assert deployment.current_pointer(tmp_path).release_id == "r3"

        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "rollback-recovered"), clock=clock)
        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, receipt.job_id, timeout=90)
            assert state == "succeeded", _failure_report(service, conn, receipt.job_id)
            ref_id = _output_refs(conn, receipt.job_id)["pointer_state"]
            document = json.loads(service.store.read_verified(
                artifact(conn, service.store, ref_id)))
        finally:
            service.close()

        assert document["action"] == "rollback"
        assert document["release_id"] == "r1"
        assert document["previous_release_id"] == "r2"
        assert document["sequence"] == 2
        assert deployment.current_pointer(tmp_path).release_id == "r3"
        assert (tmp_path / "DEPLOYED").read_bytes() == live_before
        assert _history_bytes(tmp_path) == history_before
    finally:
        conn.close()


def test_crash_window_pointer_written_history_not_appended_recovers_recorded_rollback(
        tmp_path, monkeypatch):
    """A crash between the real pointer write and the history append.

    The original attempt runs the real ``deployment.rollback`` with a narrowly
    patched ``_append_history`` that raises after ``DEPLOYED`` has moved, so the
    live pointer names the rollback while ``history/`` stops one entry short.
    Restoring ``_append_history`` and resubmitting the saved plan under a NEW
    key must report that exact pointer state -- the pointer itself is the
    recovery receipt, so ``_recorded_rollback``'s pointer fallback is what
    runs -- without a second pointer swap or history repair.

    The store then advances for real: a synthetic ``r3`` is staged and promoted
    through the real ``deployment`` API, whose own crash repair appends the
    missing rollback history entry before it promotes. The same pinned plan,
    submitted under another NEW key, must again report its original rollback
    state -- now found in the repaired history -- and leave the later ``r3``
    pointer and every history byte unchanged. Nothing seeds output receipts or
    stubs the worker/deployment producer.
    """
    conn, clock, _ = catalog(tmp_path)
    try:
        _stage_pair(tmp_path)
        deployment.promote(tmp_path, "r1")
        deployment.promote(tmp_path, "r2")

        plan = training.rollback_plan(release_root=str(tmp_path))
        assert plan["parameters"]["incumbent_release_id"] == "r2"
        assert plan["parameters"]["incumbent_sequence"] == 1
        assert plan["parameters"]["target_release_id"] == "r1"

        original_append = deployment._append_history

        def crash_after_pointer(_root, _state):
            raise RuntimeError("simulated crash between pointer write and history append")

        monkeypatch.setattr(deployment, "_append_history", crash_after_pointer)
        try:
            with pytest.raises(RuntimeError):
                deployment.rollback(tmp_path)
        finally:
            monkeypatch.setattr(deployment, "_append_history", original_append)

        crashed = deployment.current_pointer(tmp_path)
        assert (crashed.sequence, crashed.action, crashed.release_id,
                crashed.previous_release_id) == (2, "rollback", "r1", "r2")
        assert [entry.sequence for entry in deployment.pointer_history(tmp_path)] == [0, 1]
        live_before = (tmp_path / "DEPLOYED").read_bytes()
        history_before = _history_bytes(tmp_path)

        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "rollback-crash-window"), clock=clock)
        service = _service(conn, tmp_path, clock)
        try:
            state = run_until(service, conn, receipt.job_id, timeout=90)
            assert state == "succeeded", _failure_report(service, conn, receipt.job_id)
            ref_id = _output_refs(conn, receipt.job_id)["pointer_state"]
            document = json.loads(service.store.read_verified(
                artifact(conn, service.store, ref_id)))

            assert document["action"] == "rollback"
            assert document["release_id"] == "r1"
            assert document["previous_release_id"] == "r2"
            assert document["sequence"] == 2
            assert (tmp_path / "DEPLOYED").read_bytes() == live_before
            assert _history_bytes(tmp_path) == history_before
            assert deployment.current_pointer(tmp_path).sequence == 2

            # Stage a real r3 and promote it through the real deployment API.
            # The promote's own _repair_history appends the crash-windowed
            # rollback at sequence 2 before advancing, so history becomes
            # [0, 1, 2] and r3 lands at sequence 3 with the pointer naming it.
            _stage_one(tmp_path, "r3", intercept=30.0, coefficient=40.0)
            deployment.promote(tmp_path, "r3")
            assert deployment.current_pointer(tmp_path).release_id == "r3"
            assert [entry.sequence for entry in deployment.pointer_history(tmp_path)] == [
                0, 1, 2, 3]
            live_r3 = (tmp_path / "DEPLOYED").read_bytes()
            history_r3 = _history_bytes(tmp_path)

            second = submit(conn, stages.registry(), POLICY,
                            request_from_plan(plan, "rollback-crash-window-r3"), clock=clock)
            state = run_until(service, conn, second.job_id, timeout=90)
            assert state == "succeeded", _failure_report(service, conn, second.job_id)
            ref_id = _output_refs(conn, second.job_id)["pointer_state"]
            recovered = json.loads(service.store.read_verified(
                artifact(conn, service.store, ref_id)))
        finally:
            service.close()

        assert recovered["action"] == "rollback"
        assert recovered["release_id"] == "r1"
        assert recovered["previous_release_id"] == "r2"
        assert recovered["sequence"] == 2
        assert deployment.current_pointer(tmp_path).release_id == "r3"
        assert (tmp_path / "DEPLOYED").read_bytes() == live_r3
        assert _history_bytes(tmp_path) == history_r3
    finally:
        conn.close()


def test_cli_plan_and_submit_rollback_never_calls_deployment_inline(tmp_path, capsys):
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "rollback",
                     "--release-root", str(tmp_path)]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["plan"]["kind"] == "rollback"
    assert document["plan"]["parameters"]["release_root"] == str(tmp_path.resolve())
    plan_ref = document["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "cli-rollback-1"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["kind"] == "models_rollback"
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        row = conn.execute("SELECT state FROM jobs WHERE job_id=?",
                           (receipt["job_id"],)).fetchone()
        assert row is not None
        assert row["state"] == "queued"
    finally:
        conn.close()


def test_cli_plan_rollback_rejects_release_id_without_saving_a_plan(tmp_path, capsys):
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        before = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        conn.close()

    assert cli.main(["--root", str(root), "plan", "rollback",
                     "--release-root", str(tmp_path),
                     "--release-id", "r1"]) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["code"] == "INVALID_REQUEST"

    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        after = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        assert after == before
    finally:
        conn.close()


def test_cli_plan_rollback_rejects_explicit_empty_release_id_without_saving_a_plan(
        tmp_path, capsys):
    """An explicitly supplied empty ``--release-id`` is still a supplied value.

    The shared argument's default is ``None`` so an omitted id is
    distinguishable from ``--release-id ""``; rollback refuses the explicit
    empty value as ``INVALID_REQUEST`` before saving any plan artifact.
    """
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        before = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
    finally:
        conn.close()

    assert cli.main(["--root", str(root), "plan", "rollback",
                     "--release-root", str(tmp_path),
                     "--release-id", ""]) == 2
    body = json.loads(capsys.readouterr().out)
    assert body["code"] == "INVALID_REQUEST"

    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        after = conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0]
        assert after == before
    finally:
        conn.close()


def test_rollback_plan_implementation_ref_is_built_from_the_shared_root_helper(tmp_path):
    """``rollback_plan`` pins its worker source manifest from the ops package's
    own ``default_checkout_root`` helper (``engine.v2.ops.experiments``), the
    same checkout root ``rollback_plan`` itself uses -- never a module-local
    ``Path(__file__)`` derivation or a direct ``engine.paths`` import.
    Recomputing the expected ref from that helper alone proves the plan builds
    from it."""
    from engine.v2.ops.experiments import default_checkout_root
    from engine.v2.ops.fingerprints import worker_source_manifest

    plan = training.rollback_plan(release_root=str(tmp_path))
    assert plan["implementation_ref"] == content_hash(
        worker_source_manifest(default_checkout_root()))
