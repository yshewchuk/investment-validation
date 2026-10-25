"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import errno
from dataclasses import replace
import hashlib
import json
import os
import sqlite3
import threading

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops import backup as backup_module
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.catalog import integrity_errors
from engine.v2.ops.effects_graph import backup_effect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_effects_graph import (
    SESSION,
    _commit,
    _open,
    _params,
    _row,
    _seed_decisions,
    _submit_and_claim,
)


def _row_counts(conn):
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    return {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name in tables}


def test_verified_artifact_copy_rejects_hash_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"evidence")
    with pytest.raises(OpsError) as refused:
        backup_module._copy_verified_file(
            source, destination, expected_size=len(b"evidence"),
            expected_hash="sha256:" + "0" * 64, stage="synthetic artifact")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert destination.read_bytes() == b"evidence"


def test_transient_backup_database_open_error_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "transient-read", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="transient-read", owner="worker", target=backup,
               clock=clock, store=store)
    source = backup / "transient-read.sqlite"
    original_open = backup_module.os.open

    def fail_source_open(path, *args, **kwargs):
        if os.fspath(path) == os.fspath(source):
            raise OSError(errno.EIO, "synthetic transient read error")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(backup_module.os, "open", fail_source_open)
    with pytest.raises(OpsError) as refused:
        backup_module._checked_file_hash(source, stage="synthetic database")

    assert refused.value.code == "WORKER_FAILED"
    assert refused.value.problem.retryable is True


def test_backup_verifies_database_and_restore_never_overwrites(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    conn.execute("CREATE TABLE snapshot_probe (value TEXT)")
    conn.execute("INSERT INTO snapshot_probe VALUES ('captured')")
    conn.commit()
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b1", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="b1", owner="worker", target=backup,
                          clock=clock, store=store)
    assert manifest["database_rows"] == _row_counts(conn)
    database_path = backup / "b1.sqlite"
    assert manifest["database"] == "sha256:" + hashlib.sha256(database_path.read_bytes()).hexdigest()
    snapshot = sqlite3.connect(database_path)
    try:
        assert snapshot.execute("SELECT value FROM snapshot_probe").fetchone()[0] == "captured"
    finally:
        snapshot.close()
    restored = restore_backup(backup, tmp_path / "restored")
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INVALID_REQUEST"
    assert (restored / "ops.sqlite").is_file()


def test_restore_rejects_unsafe_manifest_key_before_writing(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "safe-key", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="safe-key", owner="worker", target=backup,
               clock=clock, store=store)
    manifest_path = backup / "safe-key.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["backup_key"] = "../../escape"
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_delivered_retry_returns_stored_receipt_without_rereading_files(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "delivered-receipt"
    prepare_backup(conn, key, {}, clock=clock)
    backup = tmp_path / "backup"
    original = run_backup(conn, key=key, owner="worker", target=backup,
                          clock=clock, store=store)
    database = backup / f"{key}.sqlite"
    database.write_bytes(b"damaged after delivery")

    assert run_backup(conn, key=key, owner="retry", target=backup,
                      clock=clock, store=store) == original


@pytest.mark.parametrize("component", ["manifest", "database"])
def test_restore_rejects_oversized_manifest_and_symlink_inputs(tmp_path, component):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "checked-input", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="checked-input", owner="worker", target=backup,
               clock=clock, store=store)
    if component == "manifest":
        manifest_path = backup / "checked-input.manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest_path.write_bytes(
            manifest_bytes + b" " * (backup_module.MAX_MANIFEST_BYTES + 1 - len(manifest_bytes)))
    else:
        path = backup / "checked-input.sqlite"
        external = tmp_path / "database-outside"
        external.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(external)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"


def test_restore_rejects_tampered_backup_without_blocking_retry(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "tampered-restore", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="tampered-restore", owner="worker", target=backup,
               clock=clock, store=store)
    path = backup / "tampered-restore.sqlite"
    original = path.read_bytes()
    path.write_bytes(original + b"tampered")
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert not destination.exists()
    path.write_bytes(original)
    assert restore_backup(backup, destination) == destination


def test_restore_checks_current_manifest_row_counts(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "restore-counts", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="restore-counts", owner="worker", target=backup,
               clock=clock, store=store)
    manifest_path = backup / "restore-counts.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    table = next(iter(manifest["database_rows"]))
    manifest["database_rows"][table] += 1
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert not destination.exists()


@pytest.mark.parametrize("field", ["database", "schema_version", "cutoff"])
def test_restore_rejects_missing_required_manifest_fields_before_writing(tmp_path, field):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "missing-field", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="missing-field", owner="worker", target=backup,
               clock=clock, store=store)
    manifest_path = backup / "missing-field.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop(field)
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert not destination.exists()


def test_backup_publication_lock_serializes_same_key(tmp_path):
    target = tmp_path / "backup"
    target.mkdir()
    first_entered = threading.Event()
    second_attempted = threading.Event()
    second_entered = threading.Event()
    release_first = threading.Event()

    def first_operation():
        first_entered.set()
        assert release_first.wait(timeout=2)

    def first_attempt():
        backup_module._with_backup_lock(target, "same-key", first_operation)

    def second_attempt():
        second_attempted.set()
        backup_module._with_backup_lock(
            target, "same-key", second_entered.set)

    first = threading.Thread(target=first_attempt)
    second = threading.Thread(target=second_attempt)
    first.start()
    assert first_entered.wait(timeout=2)
    second.start()
    try:
        assert second_attempted.wait(timeout=2)
        assert not second_entered.wait(timeout=0.05)
    finally:
        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

    assert second_entered.is_set()


def test_backup_counts_and_copies_one_read_snapshot(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    conn.execute("CREATE TABLE snapshot_probe (value TEXT)")
    conn.execute("INSERT INTO snapshot_probe VALUES ('before')")
    conn.commit()
    database_path = conn.execute("PRAGMA database_list").fetchone()[2]
    sibling = sqlite3.connect(database_path, isolation_level=None)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "snapshot-race", {}, clock=clock)
    original_backup_to = backup_module.backup_to

    def commit_between_count_and_copy(source, destination):
        sibling.execute("INSERT INTO snapshot_probe VALUES ('after')")
        original_backup_to(source, destination)

    monkeypatch.setattr(backup_module, "backup_to", commit_between_count_and_copy)
    backup = tmp_path / "backup"
    try:
        manifest = run_backup(conn, key="snapshot-race", owner="worker", target=backup,
                              clock=clock, store=store)
    finally:
        sibling.close()

    snapshot = sqlite3.connect(backup / "snapshot-race.sqlite")
    try:
        assert snapshot.execute("SELECT COUNT(*) FROM snapshot_probe").fetchone()[0] == 1
        assert manifest["database_rows"]["snapshot_probe"] == 1
    finally:
        snapshot.close()
    assert conn.execute("SELECT COUNT(*) FROM snapshot_probe").fetchone()[0] == 2


def test_backup_source_snapshot_read_error_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "snapshot-read-error", {}, clock=clock)

    def fail_count(_conn):
        raise sqlite3.OperationalError("synthetic source read error")

    monkeypatch.setattr(backup_module, "_table_row_counts", fail_count)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="snapshot-read-error", owner="worker",
                   target=tmp_path / "backup", clock=clock, store=store)

    assert refused.value.code == "WORKER_FAILED"
    assert refused.value.problem.retryable is True
    assert not conn.in_transaction


@pytest.mark.parametrize("artifact_name", [
    "evidence/bin",
    "..",
    "evidence" + chr(92) + "bin",
    "evidence" + chr(0) + "bin",
])
def test_backup_rejects_unsafe_artifact_name_before_claim(tmp_path, artifact_name):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    backup = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        prepare_backup(conn, "unsafe-name", {artifact_name: ref}, clock=clock)

    assert refused.value.code == "INVALID_REQUEST"
    assert conn.execute("SELECT effect_id FROM outbox WHERE kind='backup'").fetchone() is None
    assert not backup.exists()


def test_prepare_backup_refuses_artifact_maps_before_enqueue(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    reference = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    malformed = replace(reference, byte_size=True)
    requests = [[], {"evidence.bin": {}}, {"evidence.bin": malformed},
                {"same-name.bin": reference}, {"same-name.bin": reference}]

    for index, artifacts in enumerate(requests):
        with pytest.raises(OpsError) as refused:
            prepare_backup(conn, f"artifact-request-{index}", artifacts, clock=clock)
        assert refused.value.code == "INVALID_REQUEST"

    assert conn.execute("SELECT effect_id FROM outbox WHERE kind='backup'").fetchone() is None


def test_controlled_drill_verifier_rejects_file_over_configured_limit(tmp_path, monkeypatch):
    from tools.v2_controlled_failure_drill import _manifest_is_consistent

    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "drill-bounded", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="drill-bounded", owner="worker", target=backup,
                          clock=clock, store=store)
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 4)

    assert not _manifest_is_consistent(manifest, backup, "drill-bounded")


def test_backup_published_hash_rejects_symlink_replacement(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "hash-race", {}, clock=clock)
    target = tmp_path / "backup"
    external = tmp_path / "external.sqlite"
    external.write_bytes(b"outside")
    original_hash = backup_module._checked_file_hash

    def replace_database(path, *, stage, expected_size=None, keepalive=None):
        if stage == "backup database":
            path.unlink()
            path.symlink_to(external)
        return original_hash(path, stage=stage, expected_size=expected_size,
                             keepalive=keepalive)

    monkeypatch.setattr(backup_module, "_checked_file_hash", replace_database)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="hash-race", owner="worker", target=target,
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert external.read_bytes() == b"outside"
    assert not (target / "hash-race.manifest.json").exists()


def test_sqlite_backup_copy_error_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "copy-error", {}, clock=clock)

    def fail_copy(_conn, staged):
        staged.write_bytes(b"partial staging bytes")
        raise sqlite3.OperationalError("synthetic copy failure")

    monkeypatch.setattr("engine.v2.ops.backup.backup_to", fail_copy)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="copy-error", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "WORKER_FAILED"
    assert refused.value.problem.retryable is True
    assert not list((tmp_path / "backup").glob(".*.staging.sqlite"))


def test_backup_staging_io_error_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "staging-error", {}, clock=clock)

    def fail_staging(**_kwargs):
        raise OSError("synthetic staging failure")

    monkeypatch.setattr(backup_module.tempfile, "mkstemp", fail_staging)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="staging-error", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "WORKER_FAILED"
    assert refused.value.problem.retryable is True


def test_backup_refuses_failed_sqlite_integrity_check(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "bad-integrity", {}, clock=clock)
    monkeypatch.setattr("engine.v2.ops.catalog.integrity_errors",
                        lambda _conn: ["synthetic integrity failure"])

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="bad-integrity", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["first_problem"] == "synthetic integrity failure"
    assert not (tmp_path / "backup" / "bad-integrity.sqlite").exists()


def test_backup_refuses_row_count_mismatch(tmp_path):
    staged = tmp_path / "staged.sqlite"
    probe = sqlite3.connect(staged)
    try:
        probe.execute("CREATE TABLE evidence (value TEXT)")
        probe.execute("INSERT INTO evidence VALUES ('one')")
        probe.commit()
    finally:
        probe.close()

    with pytest.raises(OpsError) as refused:
        backup_module._verify_staged(staged, {"evidence": 0})

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.details["tables"] == ["evidence"]


@pytest.mark.parametrize("stale_bytes", [b"", b"SQLite format 3\x00"],
                         ids=["zero-byte", "truncated-header"])
def test_backup_replaces_stale_final_database_only_with_verified_copy(tmp_path, stale_bytes):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b7", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / "b7.sqlite"
    stale.write_bytes(stale_bytes)

    manifest = run_backup(
        conn, key="b7", owner="worker", target=backup, clock=clock, store=store)

    assert stale.stat().st_size > len(stale_bytes)
    rebuilt = sqlite3.connect(stale)
    try:
        assert integrity_errors(rebuilt) == []
        assert _row_counts(rebuilt) == _row_counts(conn)
    finally:
        rebuilt.close()
    assert manifest["database_rows"] == _row_counts(conn)
    assert conn.execute(
        "SELECT state FROM outbox WHERE logical_key='b7'").fetchone()[0] == "delivered"


def test_backup_failure_before_ack_remains_retryable(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b2", {}, clock=clock)
    def crash(_name):
        raise RuntimeError("publication interrupted")
    with pytest.raises(RuntimeError):
        run_backup(conn, key="b2", owner="worker", target=tmp_path / "backup", clock=clock,
                   store=store, fault=crash)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "running"
    clock.advance(301)
    run_backup(conn, key="b2", owner="retry", target=tmp_path / "backup", clock=clock, store=store)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"


@pytest.mark.parametrize("unsafe_prior", ["symlink", "oversized"])
def test_retry_rebuilds_unsafe_prior_database(tmp_path, monkeypatch, unsafe_prior):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "unsafe-prior", {}, clock=clock)
    backup = tmp_path / "backup"

    def crash_after_publish(point):
        if point == "after_manifest_before_ack":
            raise RuntimeError("ack lost")

    with pytest.raises(RuntimeError, match="ack lost"):
        run_backup(conn, key="unsafe-prior", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash_after_publish)
    database = backup / "unsafe-prior.sqlite"
    prior_inode = database.lstat().st_ino
    external = tmp_path / "external.sqlite"
    external_bytes = None
    if unsafe_prior == "symlink":
        external.write_bytes(database.read_bytes())
        external_bytes = external.read_bytes()
        database.unlink()
        database.symlink_to(external)
    else:
        cap = database.stat().st_size
        with database.open("ab") as stream:
            stream.write(b"oversized prior")
        monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", cap)
        assert database.stat().st_size > backup_module.MAX_BACKUP_FILE_BYTES
    clock.advance(301)

    result = run_backup(conn, key="unsafe-prior", owner="retry", target=backup,
                        clock=clock, store=store)

    assert result["database_rows"] == _row_counts(conn)
    assert not database.is_symlink()
    assert database.lstat().st_ino != prior_inode
    if external_bytes is not None:
        assert external.read_bytes() == external_bytes


def test_retry_rebuilds_when_same_count_source_contents_change(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    conn.execute("CREATE TABLE synthetic_change (value TEXT)")
    conn.execute("INSERT INTO synthetic_change VALUES ('before')")
    conn.commit()
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "same-count-change", {}, clock=clock)
    backup = tmp_path / "backup"

    def crash_after_publish(point):
        if point == "after_manifest_before_ack":
            raise RuntimeError("ack lost")

    with pytest.raises(RuntimeError, match="ack lost"):
        run_backup(conn, key="same-count-change", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash_after_publish)
    manifest_path = backup / "same-count-change.manifest.json"
    prior = json.loads(manifest_path.read_text())
    conn.execute("UPDATE synthetic_change SET value='after'")
    conn.commit()
    clock.advance(301)

    result = run_backup(conn, key="same-count-change", owner="retry", target=backup,
                        clock=clock, store=store)
    updated = json.loads(manifest_path.read_text())
    snapshot = sqlite3.connect(backup / "same-count-change.sqlite")
    try:
        assert snapshot.execute("SELECT value FROM synthetic_change").fetchone()[0] == "after"
    finally:
        snapshot.close()

    assert result["database_rows"] == prior["database_rows"]
    assert updated["database"] != prior["database"]
    assert updated["database_rows"] == prior["database_rows"]





def test_manifest_success_before_ack_is_idempotent_and_keys_are_safe(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    with pytest.raises(ArtifactError, match="UNSAFE_PATH"):
        prepare_backup(conn, "../escape", {}, clock=clock)
    prepare_backup(conn, "b3", {}, clock=clock)
    backup = tmp_path / "backup"
    def crash(point):
        if point == "after_manifest_before_ack":
            raise RuntimeError("ack lost")
    with pytest.raises(RuntimeError):
        run_backup(conn, key="b3", owner="worker", target=backup, clock=clock, store=store,
                   fault=crash)
    first = json.loads((backup / "b3.manifest.json").read_text())
    clock.advance(301)
    second = run_backup(conn, key="b3", owner="retry", target=backup, clock=clock, store=store)
    assert second["database_rows"] == _row_counts(conn)
    independent_hash = hashlib.sha256((backup / "b3.sqlite").read_bytes()).hexdigest()
    assert second["database"] == "sha256:" + independent_hash
    assert second["artifacts"] == first["artifacts"]
    assert second["cutoff"] == "2026-09-12T00:00:00.000000Z"
    assert conn.execute("SELECT state FROM outbox WHERE logical_key='b3'").fetchone()[0] == "delivered"


def test_prepare_backup_retry_returns_same_effect_without_rebuilding_payload(tmp_path):
    """P2-C07: ``prepare_backup`` called again for a key that already has a
    pending/running ``backup`` effect must return that SAME effect, not
    build a new payload -- a new payload (fresh ``cutoff``) for the same
    logical key would make ``outbox.enqueue`` raise ``IDEMPOTENCY_CONFLICT``."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    first_id = prepare_backup(conn, "b6", {}, clock=clock)
    original = conn.execute("SELECT payload_json FROM outbox WHERE effect_id=?", (first_id,)).fetchone()[0]

    clock.advance(3600)
    second_id = prepare_backup(conn, "b6", {}, clock=clock)
    assert second_id == first_id
    assert conn.execute("SELECT payload_json FROM outbox WHERE effect_id=?",
                        (first_id,)).fetchone()[0] == original
    assert conn.execute("SELECT COUNT(*) FROM outbox WHERE kind='backup' AND logical_key='b6'"
                        ).fetchone()[0] == 1


def test_backup_rejects_symlink_target_before_writing(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b4", {}, clock=clock)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "backup"
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b4", owner="worker", target=target, clock=clock, store=store)
    assert refused.value.code == "INVALID_REQUEST"
    assert not (outside / "b4.sqlite").exists()


def test_run_backup_retry_after_the_effect_already_delivered_succeeds(tmp_path):
    """#98: a retry after a crash between run_backup's own complete() (state=delivered)
    and the caller's later watermark/attempt-commit step must succeed as an idempotent
    replay, not fail permanently with STALE_EXPECTATION."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b4", {}, clock=clock)
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
