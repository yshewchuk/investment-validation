"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from tests.ops_support import catalog


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
