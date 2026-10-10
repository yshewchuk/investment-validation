"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.effects_graph import backup_effect
from engine.v2.ops.errors import fail
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_effects_graph import (
    SESSION, _commit, _open, _params, _row, _seed_decisions, _submit_and_claim,
)


def test_backup_copies_artifacts_and_restores_new_root(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b1", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="b1", owner="worker", target=backup, clock=clock, store=store)
    assert manifest["artifacts"]["evidence.bin"]["content_hash"] == ref.content_hash
    restored = restore_backup(backup, tmp_path / "restored")
    assert (restored / "artifacts" / "evidence.bin").read_bytes() == b"evidence"
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"
    with pytest.raises(Exception):
        restore_backup(backup, tmp_path / "restored")


def test_backup_copy_failure_remains_retryable(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b2", {"evidence.bin": ref}, clock=clock)
    def crash(_name):
        raise RuntimeError("copy interrupted")
    with pytest.raises(RuntimeError):
        run_backup(conn, key="b2", owner="worker", target=tmp_path / "backup", clock=clock,
                   store=store, fault=crash)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "running"
    clock.advance(301)
    run_backup(conn, key="b2", owner="retry", target=tmp_path / "backup", clock=clock, store=store)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"


def test_manifest_success_before_ack_is_idempotent_and_keys_are_safe(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    with pytest.raises(Exception):
        prepare_backup(conn, "../escape", {"evidence.bin": ref}, clock=clock)
    prepare_backup(conn, "b3", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    def crash(point):
        if point == "after_manifest_before_ack":
            raise RuntimeError("ack lost")
    with pytest.raises(RuntimeError):
        run_backup(conn, key="b3", owner="worker", target=backup, clock=clock, store=store,
                   fault=crash)
    first = (backup / "b3.manifest.json").read_bytes()
    clock.advance(301)
    second = run_backup(conn, key="b3", owner="retry", target=backup, clock=clock, store=store)
    assert (backup / "b3.manifest.json").read_bytes() == first
    assert second["cutoff"] == "2026-09-12T00:00:00.000000Z"
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b3'").fetchone()[0] == "delivered"


def test_prepare_backup_retry_returns_same_effect_without_rebuilding_payload(tmp_path):
    """P2-C07: ``prepare_backup`` called again for a key that already has a
    pending/running ``backup`` effect must return that SAME effect, not
    build a new payload -- a new payload (fresh ``cutoff``) for the same
    logical key would make ``outbox.enqueue`` raise ``IDEMPOTENCY_CONFLICT``."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    first_id = prepare_backup(conn, "b6", {"evidence.bin": ref}, clock=clock)
    original = conn.execute("SELECT payload_json FROM outbox WHERE effect_id=?", (first_id,)).fetchone()[0]

    clock.advance(3600)
    second_id = prepare_backup(conn, "b6", {"evidence.bin": ref}, clock=clock)
    assert second_id == first_id
    assert conn.execute("SELECT payload_json FROM outbox WHERE effect_id=?",
                        (first_id,)).fetchone()[0] == original
    assert conn.execute("SELECT COUNT(*) FROM outbox WHERE kind='backup' AND logical_key='b6'"
                        ).fetchone()[0] == 1


def test_backup_rejects_symlink_target_before_writing(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b4", {"evidence.bin": ref}, clock=clock)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "backup"
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(Exception):
        run_backup(conn, key="b4", owner="worker", target=target, clock=clock, store=store)
    assert not (outside / "b4.sqlite").exists()


def test_run_backup_retry_after_the_effect_already_delivered_succeeds(tmp_path):
    """#98: a retry after a crash between run_backup's own complete() (state=delivered)
    and the caller's later watermark/attempt-commit step must succeed as an idempotent
    replay, not fail permanently with STALE_EXPECTATION."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b4", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    first = run_backup(conn, key="b4", owner="worker", target=backup, clock=clock, store=store)
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b4'").fetchone()[0] == "delivered"
    manifest_before_retry = (backup / "b4.manifest.json").read_bytes()

    # Simulate the retry that happens when the caller crashed after run_backup's own
    # complete() but before it wrote the watermark/committed the attempt: run_backup is
    # invoked again with the SAME key, while the outbox row is already delivered.
    second = run_backup(conn, key="b4", owner="worker", target=backup, clock=clock, store=store)
    assert second == first
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b4'").fetchone()[0] == "delivered"
    # No second backup was performed: the manifest file on disk is unchanged.
    assert (backup / "b4.manifest.json").read_bytes() == manifest_before_retry


def test_backup_coordinator_retry_after_delivery_before_commit_attempt_succeeds(tmp_path,
                                                                                 monkeypatch):
    """#98 at the COORDINATOR level: the fault here is injected after
    ``backup_effect``'s own ``run_backup`` delivered the effect (outbox row
    ``delivered``, manifest durable) but before the attempt's own
    ``commit_attempt`` runs — the exact boundary ``backup.py``'s
    ``_delivered_or_stale`` replay branch targets. The call-level test above
    only simulates the crash at the ``run_backup`` call site, and the
    existing coordinator tests only inject faults INSIDE ``backup_effect``
    (before it returns); neither exercises this window, which the nightly
    job-retry path traverses for real. The retry re-attempts the SAME job
    through the scheduler's own retry machinery (a retryable ``commit_attempt``
    failure settles the crashed attempt and moves the job to ``retry_wait``;
    a fresh claim after the backoff delay starts attempt 2), and must
    complete with exactly ONE delivered backup effect, the manifest
    unchanged, the backup watermark written, the second attempt committed
    and the first attempt settled as failed -- nothing left dangling."""
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])

        first = _submit_and_claim(conn, clock, supervisor, kind="backup", key="backup-1",
                                  parameters=_params("backup", SESSION, scope))
        result = backup_effect(conn, store, first, root, clock=clock)
        assert result == (None, ())
        key = conn.execute("SELECT logical_key FROM outbox WHERE kind='backup'").fetchone()[0]
        assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == \
            "delivered"
        manifest_path = root / "backups" / scope / (key + ".manifest.json")
        manifest_before = manifest_path.read_bytes()

        # The fault: this attempt's commit_attempt never completes — the
        # coordinator dies right after backup_effect returned, leaving the
        # delivered effect durably DONE but the attempt uncommitted.
        def crash(*args, **kwargs):
            raise RuntimeError("crash before commit_attempt")

        monkeypatch.setattr("tests.v2.ops.test_v2_ops_effects_graph.commit_attempt", crash)
        with pytest.raises(RuntimeError, match="crash before commit_attempt"):
            _commit(conn, clock, first, result)
        monkeypatch.undo()
        assert conn.execute("SELECT state FROM attempts WHERE attempt_id=?",
                            (first.attempt_id,)).fetchone()[0] not in ("succeeded", "failed")

        # Retry the SAME job, through the mechanism this suite's own
        # retry-after-failure tests use (test_v2_ops_stages_core_kinds.py's
        # ``_fail_once``): ``commit_attempt`` with a retryable failure settles
        # the crashed attempt as failed and moves the job to ``retry_wait``;
        # once the backup kind's first backoff step (bounded, (5, 30)) has
        # elapsed, a fresh claim of the same idempotency key starts attempt 2
        # on the same job_id. Before #98 that second attempt's run_backup
        # found only the delivered row and failed permanently with
        # STALE_EXPECTATION.
        state = commit_attempt(
            conn, first.attempt_id, first.fence,
            Outcome(succeeded=False, process_state="verified_dead", exit_code=1,
                    failure=fail("WORKER_FAILED", "coordinator died before commit").problem),
            clock=clock)
        assert state == "retry_wait"
        clock.advance(5)

        second = _submit_and_claim(conn, clock, supervisor, kind="backup", key="backup-1",
                                   parameters=_params("backup", SESSION, scope))
        assert second.job_id == first.job_id
        assert second.attempt_number == 2
        retried = backup_effect(conn, store, second, root, clock=clock)
        assert retried == (None, ())
        _commit(conn, clock, second, retried)

        # Exactly one delivered effect — no duplicate backup ran.
        rows = conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchall()
        assert [r["state"] for r in rows] == ["delivered"]
        # The manifest on disk is the FIRST attempt's, byte for byte.
        assert manifest_path.read_bytes() == manifest_before
        backup_wm = conn.execute("SELECT occurrence FROM watermarks WHERE pipeline='nightly' "
                                 "AND scope=? AND stage='backup'", (scope,)).fetchone()
        assert backup_wm["occurrence"] == SESSION
        # The retried attempt is ultimately committed successfully...
        assert conn.execute("SELECT state FROM attempts WHERE attempt_id=?",
                            (second.attempt_id,)).fetchone()[0] == "succeeded"
        assert conn.execute("SELECT state FROM jobs WHERE job_id=?",
                            (second.job_id,)).fetchone()[0] == "succeeded"
        # ...and the superseded first attempt is settled, not left dangling
        # (same assertion the lease-recovery retry test makes for the
        # attempt it supersedes, test_v2_ops_coordinator_lease.py).
        crashed = conn.execute("SELECT state, process_state FROM attempts WHERE attempt_id=?",
                               (first.attempt_id,)).fetchone()
        assert (crashed["state"], crashed["process_state"]) == ("failed", "verified_dead")
    finally:
        conn.close()
