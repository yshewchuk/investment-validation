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

from engine.v2.foundation import SystemClock
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


def _stage_pair(root):
    """Stage and mark two distinct synthetic releases ``r1`` and ``r2``."""
    for release_id, intercept, coefficient in (("r1", 1.0, 2.0), ("r2", 10.0, 20.0)):
        release, inventory, payloads = _release_fixture(
            release_id, intercept=intercept, coefficient=coefficient)
        deployment.stage_release(root, release, inventory, payloads)
        deployment.mark_staging_succeeded(root, release_id)


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

        plan = training.rollback_plan(release_root=str(tmp_path))
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
