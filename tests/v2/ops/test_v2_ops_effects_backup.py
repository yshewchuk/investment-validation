"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from engine.v2.foundation import ArtifactError, ArtifactStore, to_document
from engine.v2.ops import backup as backup_module
from engine.v2.ops import catalog as catalog_module
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.catalog import integrity_errors
from engine.v2.ops.effects_graph import backup_effect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import file_hash
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


class _InjectedCrash(Exception):
    pass


def _row_counts(conn):
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    return {name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
            for name in tables}


def test_artifact_store_verify_respects_explicit_byte_cap(tmp_path):
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"larger than cap", schema_ref="evidence.v1.0")
    with pytest.raises(ArtifactError):
        store.verify(ref, max_bytes=4)

    source = store.root / ref.storage_key
    source.chmod(0o644)
    source.write_bytes(b"grown beyond the registered size")
    with pytest.raises(ArtifactError):
        store.verify(ref, max_bytes=ref.byte_size)


def test_restore_requires_artifacts_mapping(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"must not be silently omitted", schema_ref="evidence.v1.0")
    prepare_backup(conn, "missing-artifacts", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="missing-artifacts", owner="worker", target=backup,
               clock=clock, store=store)
    manifest_path = backup / "missing-artifacts.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["artifacts"]
    manifest_path.write_text(json.dumps(manifest))

    destination = tmp_path / "restored"
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)
    assert refused.value.code == "INTEGRITY_FAILED"
    assert not destination.exists()


@pytest.mark.parametrize(("field", "value", "remove"), [
    ("schema_version", None, True),
    ("schema_version", "catalog_backup.v9.0", False),
    ("backup_key", "different-key", False),
    ("database_rows", {}, False),
])
def test_restore_rejects_manifest_identity_or_row_count_mismatch(
        tmp_path, field, value, remove):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "restore-manifest-check", {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="restore-manifest-check", owner="worker", target=backup,
               clock=clock, store=store)
    manifest_path = backup / "restore-manifest-check.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if remove:
        manifest.pop(field)
    else:
        manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))

    destination = tmp_path / "restored"
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)
    assert refused.value.code == "INTEGRITY_FAILED"
    assert not destination.exists()


def test_backup_manifest_publication_oserror_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "publish-error", {}, clock=clock)

    def fail_publish(*_args, **_kwargs):
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(backup_module, "_publish_manifest", fail_publish)
    with pytest.raises(OpsError) as failed:
        run_backup(conn, key="publish-error", owner="worker",
                   target=tmp_path / "backup", clock=clock, store=store)
    assert failed.value.code == "WORKER_FAILED"
    assert failed.value.problem.retryable is True


def test_backup_integrity_scan_renews_keepalive(tmp_path, monkeypatch):
    source = sqlite3.connect(tmp_path / "source.sqlite")
    source.execute("CREATE TABLE sample (value INTEGER)")
    source.commit()
    renewed = []

    def long_integrity_scan(conn):
        conn.execute(
            "WITH RECURSIVE n(x) AS (VALUES(0) UNION ALL "
            "SELECT x + 1 FROM n WHERE x < 20000) SELECT MAX(x) FROM n").fetchone()
        return []

    monkeypatch.setattr(catalog_module, "integrity_errors", long_integrity_scan)
    backup_module.backup_to(
        source, tmp_path / "copy.sqlite", keepalive=lambda: renewed.append(True))
    source.close()

    assert renewed


def test_sqlite_progress_callback_preserves_lease_loss():
    progress = backup_module._Progress(
        "synthetic progress", keepalive=lambda: (_ for _ in ()).throw(
            fail("LEASE_LOST", "synthetic lease loss")))
    progress.last_keepalive = 0

    assert progress.pulse() == 1
    with pytest.raises(OpsError) as lost:
        progress.finish()
    assert lost.value.code == "LEASE_LOST"


def test_backup_copies_artifacts_and_restores_new_root(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b1", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="b1", owner="worker", target=backup, clock=clock, store=store)
    assert manifest["artifacts"]["evidence.bin"]["content_hash"] == ref.content_hash
    restored = restore_backup(backup, tmp_path / "restored")
    artifact = restored / "artifacts" / "evidence.bin"
    assert artifact.read_bytes() == b"evidence"
    original = artifact.read_bytes()
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert artifact.read_bytes() == original


def test_delivered_backup_rejects_symlinked_artifacts_directory(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"delivered artifact", schema_ref="evidence.v1.0")
    prepare_backup(conn, "delivered-link", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    receipt = run_backup(conn, key="delivered-link", owner="worker", target=backup,
                         clock=clock, store=store)

    moved = tmp_path / "moved-artifacts"
    (backup / "artifacts").rename(moved)
    (backup / "artifacts").symlink_to(moved, target_is_directory=True)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="delivered-link", owner="worker", target=backup,
                   clock=clock, store=store)
    assert refused.value.code == "INTEGRITY_FAILED"
    assert receipt["artifacts"]


def test_restore_rejects_unsafe_manifest_artifact_before_creating_destination(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"safe", schema_ref="evidence.v1.0")
    prepare_backup(conn, "unsafe-restore", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="unsafe-restore", owner="worker", target=backup,
               clock=clock, store=store)
    manifest_path = backup / "unsafe-restore.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["../escape"] = manifest["artifacts"].pop("evidence.bin")
    manifest_path.write_text(json.dumps(manifest))
    destination = tmp_path / "unsafe-restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()


def test_restore_rejects_symlinked_artifact_and_removes_partial_destination(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"safe", schema_ref="evidence.v1.0")
    prepare_backup(conn, "symlink-restore", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="symlink-restore", owner="worker", target=backup,
               clock=clock, store=store)
    artifact = backup / "artifacts" / "evidence.bin"
    artifact.unlink()
    artifact.symlink_to(store.verify(ref))
    destination = tmp_path / "symlink-restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()


@pytest.mark.parametrize("stale_bytes", [b"", b"SQLite format 3\x00"],
                         ids=["zero-byte", "truncated-header"])
def test_backup_atomically_replaces_unverified_key_database(tmp_path, stale_bytes):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b7", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / "b7.sqlite"
    stale.write_bytes(stale_bytes)

    manifest = run_backup(
        conn, key="b7", owner="worker", target=backup, clock=clock, store=store)

    assert manifest["database_file"] == "b7.sqlite"
    assert file_hash(stale) == manifest["database"]
    assert manifest["database_rows"] == _row_counts(conn)
    rebuilt = sqlite3.connect(stale)
    try:
        assert integrity_errors(rebuilt) == []
        assert _row_counts(rebuilt) == _row_counts(conn)
    finally:
        rebuilt.close()
    assert conn.execute(
        "SELECT state FROM outbox WHERE logical_key='b7'").fetchone()[0] == "delivered"


def test_staged_row_count_mismatch_is_refused_with_table_details(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b8", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def corrupt(conn, destination, *, progress=None, keepalive=None):
        real_backup_to(conn, destination, progress=progress, keepalive=keepalive)
        damaged = sqlite3.connect(destination)
        try:
            damaged.execute("DELETE FROM outbox")
            damaged.commit()
        finally:
            damaged.close()

    monkeypatch.setattr(backup_module, "backup_to", corrupt)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b8", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["tables"] == ["outbox"]


def test_backup_counts_and_copy_share_one_snapshot(tmp_path, monkeypatch):
    """A writer that commits after the source counts are read must not leak
    its row into the backup: counts and ``backup_to`` use one read snapshot."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    conn.execute("CREATE TABLE synthetic_change (value INTEGER)")
    conn.commit()
    prepare_backup(conn, "b13", {}, clock=clock)
    before = _row_counts(conn)
    database = tmp_path / "ops.sqlite"
    real_backup_to = backup_module.backup_to
    observed = []

    def racing_backup_to(source, destination, *, progress=None, keepalive=None):
        writer = sqlite3.connect(database)
        try:
            writer.execute("INSERT INTO synthetic_change VALUES (1)")
            writer.commit()
        finally:
            writer.close()
        observed.append(_row_counts(source))
        real_backup_to(source, destination, progress=progress, keepalive=keepalive)

    monkeypatch.setattr(backup_module, "backup_to", racing_backup_to)
    manifest = run_backup(conn, key="b13", owner="worker", target=tmp_path / "backup",
                          clock=clock, store=store)

    assert observed == [before]
    assert _row_counts(conn)["synthetic_change"] == 1
    backup = sqlite3.connect(tmp_path / "backup" / manifest["database_file"])
    try:
        assert _row_counts(backup) == before
        assert backup.execute("SELECT COUNT(*) FROM synthetic_change").fetchone()[0] == 0
    finally:
        backup.close()
    assert manifest["database_rows"] == before


def test_retry_ignores_crash_staging_files_as_results(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b12", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / f"{backup_module._stage_prefix('b12')}crash.staging.sqlite"
    other = backup / f"{backup_module._stage_prefix('b13')}crash.staging.sqlite"
    stale.write_bytes(b"junk")
    other.write_bytes(b"other")

    manifest = run_backup(
        conn, key="b12", owner="worker", target=backup, clock=clock, store=store)

    assert stale.exists()
    assert other.exists()
    assert manifest["database_file"] == "b12.sqlite"
    assert manifest["database_file"] not in {stale.name, other.name}
    assert file_hash(backup / manifest["database_file"]) == manifest["database"]


def test_backup_copy_failure_remains_retryable(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b2", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    old_destination = backup / "artifacts" / "evidence.bin"
    old_destination.parent.mkdir(parents=True)
    old_destination.write_bytes(b"old evidence")
    def crash(_name):
        raise _InjectedCrash("copy interrupted")
    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="b2", owner="worker", target=backup, clock=clock,
                   store=store, fault=crash)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "running"
    assert old_destination.read_bytes() == b"old evidence"
    assert list(old_destination.parent.glob(".evidence.bin.*.pending")) == []
    clock.advance(301)
    run_backup(conn, key="b2", owner="retry", target=backup, clock=clock, store=store)
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"


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


def test_unsafe_artifact_name_is_refused_before_database_install(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b21", {"../escape": ref}, clock=clock)
    backup = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b21", owner="worker", target=backup, clock=clock, store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not list(backup.glob("*.sqlite"))


def test_backup_rejects_symlinked_artifact_source(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "symlink-artifact", {"evidence.bin": ref}, clock=clock)
    source = tmp_path / "changed-source"
    source.symlink_to(store.verify(ref))
    monkeypatch.setattr(store, "verify", lambda _ref, **_kwargs: source)
    backup = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="symlink-artifact", owner="worker", target=backup,
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not list(backup.glob("*.manifest.json"))


@pytest.mark.parametrize("mutation", [
    {"artifact_id": 123},
    {"storage_key": None},
    {"unexpected": "field"},
], ids=["wrong-type", "null-storage-key", "unknown-field"])
def test_malformed_payload_artifact_metadata_is_refused_before_backup(
        tmp_path, mutation):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    malformed = {**to_document(ref), **mutation}
    prepare_backup(conn, "bad-ref", {"evidence.bin": malformed}, clock=clock)
    backup = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="bad-ref", owner="worker", target=backup,
                   clock=clock, store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not list(backup.glob("*.sqlite"))
    assert not list(backup.glob("*.manifest.json"))


def test_per_key_lock_serializes_two_threads(tmp_path):
    target = tmp_path / "backup"
    target.mkdir()
    first_entered = threading.Event()
    second_attempting = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()
    order = []

    def first():
        with backup_module._backup_lock(target, "same-key"):
            order.append("first-enter")
            first_entered.set()
            release.wait(5)
            order.append("first-exit")

    def second():
        first_entered.wait(5)
        second_attempting.set()
        with backup_module._backup_lock(target, "same-key"):
            order.append("second-enter")
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(5)
    second_thread.start()
    assert second_attempting.wait(5)
    assert not second_entered.wait(0.1)
    release.set()
    first_thread.join(5)
    second_thread.join(5)

    assert second_entered.is_set()
    assert order == ["first-enter", "first-exit", "second-enter"]


def test_different_key_locks_are_independent(tmp_path):
    target = tmp_path / "backup"
    target.mkdir()
    entered = threading.Event()

    def other():
        with backup_module._backup_lock(target, "other-key"):
            entered.set()

    with backup_module._backup_lock(target, "one-key"):
        thread = threading.Thread(target=other)
        thread.start()
        assert entered.wait(5)
        thread.join(5)


def test_catalog_backup_to_reports_page_progress(tmp_path):
    conn, _, _ = catalog(tmp_path)
    destination = tmp_path / "copy.sqlite"
    seen = []

    backup_module.backup_to(conn, destination, progress=lambda *args: seen.append(args))

    assert destination.is_file()
    assert seen
    _status, remaining, total = seen[-1]
    assert remaining == 0
    assert total > 0
    assert all(0 <= pending <= count for _, pending, count in seen)
    assert all(count == total for _, _, count in seen)
    assert all(left[1] >= right[1] for left, right in zip(seen, seen[1:]))


def test_backup_rejects_symlink_target_before_writing(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b4", {"evidence.bin": ref}, clock=clock)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = tmp_path / "backup"
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b4", owner="worker", target=target, clock=clock, store=store)
    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not list(outside.glob("b4.*.sqlite"))
    assert not (outside / "b4.sqlite").exists()


def test_backup_rejects_symlinked_artifact_directory_before_staging(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "symlink-artifacts", {"evidence.bin": ref}, clock=clock)
    target = tmp_path / "backup"
    target.mkdir()
    outside = tmp_path / "outside-artifacts"
    outside.mkdir()
    (target / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="symlink-artifacts", owner="worker", target=target,
                   clock=clock, store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not (target / "symlink-artifacts.sqlite").exists()
    assert not (outside / "evidence.bin").exists()


@pytest.mark.parametrize("key", ["../escape", "/absolute", "a/b", "", ".", "a\\b"])
def test_run_backup_rejects_unsafe_key_before_creating_target(tmp_path, key):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    target = tmp_path / "backup"

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key=key, owner="worker", target=target, clock=clock,
                   store=store)

    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False
    assert not target.exists()


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


def test_delivered_retry_refuses_corrupted_artifact(tmp_path):
    """A delivered receipt is not success when its durable files later change."""
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b4", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key="b4", owner="worker", target=backup, clock=clock, store=store)

    (backup / "artifacts" / "evidence.bin").write_bytes(b"corrupted")
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b4", owner="worker", target=backup, clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_backup_coordinator_retry_after_delivery_before_commit_attempt_succeeds(tmp_path,
                                                                                 monkeypatch):
    """#98 at the COORDINATOR level: the fault here is injected after
    ``backup_effect``'s own ``run_backup`` delivered the effect (outbox row
    ``delivered``, manifest durable) but before the attempt's own
    ``commit_attempt`` runs — the exact boundary ``backup.py``'s delivered
    replay branch targets. The retry re-attempts the SAME job through the
    scheduler's own retry machinery (a retryable ``commit_attempt`` failure
    settles the crashed attempt and moves the job to ``retry_wait``; a fresh
    claim after the backoff delay starts attempt 2), and must complete with
    exactly ONE delivered backup effect, the manifest unchanged, the backup
    watermark written, the second attempt committed and the first attempt
    settled as failed -- nothing left dangling."""
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
            raise _InjectedCrash("crash before commit_attempt")

        monkeypatch.setattr("tests.v2.ops.test_v2_ops_effects_graph.commit_attempt", crash)
        with pytest.raises(_InjectedCrash, match="crash before commit_attempt"):
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


def test_manifest_database_path_uses_current_name_or_legacy_fallback(tmp_path):
    assert backup_module.manifest_database_path(
        tmp_path, {"backup_key": "b1", "database_file": "b1.abc.sqlite"}) == \
        tmp_path / "b1.abc.sqlite"
    assert backup_module.manifest_database_path(
        tmp_path, {"backup_key": "b1"}) == tmp_path / "b1.sqlite"


def test_restore_accepts_legacy_manifest_without_size_fields(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "legacy", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(
        conn, key="legacy", owner="worker", target=backup, clock=clock, store=store)
    (backup / manifest["database_file"]).rename(backup / "legacy.sqlite")
    manifest.pop("database_file")
    manifest.pop("database_size")
    manifest.pop("database_rows")
    manifest["artifacts"]["evidence.bin"].pop("byte_size")
    (backup / "legacy.manifest.json").write_text(json.dumps(manifest))

    restored = restore_backup(backup, tmp_path / "legacy-restored")

    assert (restored / "ops.sqlite").is_file()
    assert (restored / "artifacts" / "evidence.bin").read_bytes() == b"evidence"


def test_ack_loss_reuses_verified_backup_without_recopy(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"ack-loss evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "ack-loss", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"

    def crash_after_manifest(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash()

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="ack-loss", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash_after_manifest)
    manifest_path = backup / "ack-loss.manifest.json"
    manifest_before = manifest_path.read_bytes()
    clock.advance(3600)

    def unexpected_rebuild(*_args, **_kwargs):
        pytest.fail("verified backup was rebuilt")

    monkeypatch.setattr(backup_module, "_stage_copy", unexpected_rebuild)
    receipt = run_backup(conn, key="ack-loss", owner="worker", target=backup,
                         clock=clock, store=store)

    assert receipt == json.loads(manifest_before)
    assert manifest_path.read_bytes() == manifest_before
    assert conn.execute(
        "SELECT state FROM outbox WHERE logical_key='ack-loss'").fetchone()[0] == "delivered"


def test_relative_backup_root_retry_after_manifest_crash_succeeds(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"relative retry", schema_ref="evidence.v1.0")
    prepare_backup(conn, "relative", {"evidence.bin": ref}, clock=clock)
    monkeypatch.chdir(tmp_path)

    def crash_after_manifest(point):
        if point == "after_manifest_before_ack":
            raise _InjectedCrash()

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="relative", owner="worker", target="backup",
                   clock=clock, store=store, fault=crash_after_manifest)
    clock.advance(3600)

    receipt = run_backup(conn, key="relative", owner="worker", target="backup",
                         clock=clock, store=store)

    assert receipt["database_rows"]
    assert (tmp_path / "backup" / "relative.manifest.json").is_file()
    assert conn.execute(
        "SELECT state FROM outbox WHERE logical_key='relative'").fetchone()[0] == "delivered"
    delivered_receipt = run_backup(
        conn, key="relative", owner="worker", target="backup", clock=clock, store=store)
    assert delivered_receipt == receipt


def test_durable_manifest_propagates_lease_loss(tmp_path, monkeypatch):
    manifest = {
        "backup_key": "lease",
        "database": "sha256:" + "0" * 64,
        "database_size": 0,
        "database_rows": {},
    }

    def lose_lease(*_args, **_kwargs):
        raise fail("LEASE_LOST", "synthetic lease loss")

    monkeypatch.setattr(backup_module, "verified_file_hash", lose_lease)
    with pytest.raises(OpsError) as lost:
        backup_module._verify_durable_manifest(tmp_path, "lease", manifest)
    assert lost.value.code == "LEASE_LOST"


def test_durable_manifest_with_corrupt_database_is_not_reused(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"retry evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "resume", {"evidence.bin": ref}, clock=clock)
    payload = json.loads(conn.execute(
        "SELECT payload_json FROM outbox WHERE kind='backup' AND logical_key='resume'").fetchone()[0])
    refs = backup_module._validated_artifact_names(payload)
    backup = tmp_path / "backup"

    def crash_after_manifest(name):
        if name == "after_manifest_before_ack":
            raise _InjectedCrash()

    with pytest.raises(_InjectedCrash):
        run_backup(conn, key="resume", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash_after_manifest)
    manifest_path = backup / "resume.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert backup_module._durable_manifest(manifest_path, "resume", payload, refs) == manifest

    manifest["artifacts"] = {}
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(OpsError) as mismatch:
        backup_module._durable_manifest(manifest_path, "resume", payload, refs)
    assert mismatch.value.code == "INTEGRITY_FAILED"

    manifest["artifacts"] = {"evidence.bin": {
        "content_hash": ref.content_hash, "byte_size": ref.byte_size}}
    manifest_path.write_text(json.dumps(manifest))
    artifact = backup / "artifacts" / "evidence.bin"
    artifact.write_bytes(b"corrupted retry artifact")
    assert backup_module._durable_manifest(manifest_path, "resume", payload, refs) is None
    clock.advance(3600)
    repaired = run_backup(conn, key="resume", owner="worker", target=backup,
                          clock=clock, store=store)
    assert repaired["artifacts"] == manifest["artifacts"]
    assert backup_module._durable_manifest(manifest_path, "resume", payload, refs) == repaired
    assert artifact.read_bytes() == b"retry evidence"
    database = backup / manifest["database_file"]
    database.write_bytes(database.read_bytes() + b"corruption")
    assert backup_module._durable_manifest(manifest_path, "resume", payload, refs) is None


def test_backup_destination_fsync_failure_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "disk-error", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()

    def fail_fsync(_fd):
        raise OSError("synthetic destination failure")

    monkeypatch.setattr(backup_module.os, "fsync", fail_fsync)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="disk-error", owner="worker", target=backup,
                   clock=clock, store=store)

    assert refused.value.code == "WORKER_FAILED"
    assert refused.value.problem.retryable is True


def test_single_filename_rejects_unsafe_names(tmp_path):
    assert backup_module._single_filename("plain.sqlite") == "plain.sqlite"
    for name in ("../escape", "/absolute", "a/b", "", ".", "a\\b"):
        with pytest.raises(ValueError):
            backup_module._single_filename(name)
        with pytest.raises(ValueError):
            backup_module.manifest_database_path(
                tmp_path, {"backup_key": "safe", "database_file": name})
        with pytest.raises(ValueError):
            backup_module.manifest_database_path(tmp_path, {"backup_key": name})


def test_prepare_backup_rejects_unsafe_key(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    with pytest.raises(OpsError) as refused:
        prepare_backup(conn, "../escape", {}, clock=clock)
    assert refused.value.code == "INVALID_REQUEST"


def test_backup_manifest_records_database_hash_and_source_rows(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "rows", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(
        conn, key="rows", owner="worker", target=backup, clock=clock, store=store)
    database = backup / manifest["database_file"]
    assert manifest["database"] == file_hash(database)
    assert manifest["database_rows"] == _row_counts(conn)
    copied = sqlite3.connect(database)
    try:
        assert integrity_errors(copied) == []
        assert _row_counts(copied) == _row_counts(conn)
    finally:
        copied.close()


def test_backup_keepalive_runs_outside_caller_transaction(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "heartbeat", {}, clock=clock)
    observed_transactions = []

    def keepalive():
        observed_transactions.append(conn.in_transaction)
        assert not conn.in_transaction

    run_backup(conn, key="heartbeat", owner="worker", target=tmp_path / "backup",
               clock=clock, store=store, keepalive=keepalive)
    assert observed_transactions
    assert not any(observed_transactions)


def test_sparse_file_over_the_byte_cap_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 1024)
    sparse = tmp_path / "sparse.bin"
    with sparse.open("wb") as stream:
        stream.truncate(1025)

    with pytest.raises(OpsError) as refused:
        backup_module.verified_file_hash(
            sparse, expected_size=1025, stage="sparse database")
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_verified_file_hash_refuses_growth_after_open(tmp_path, monkeypatch):
    class GrowingStream:
        def __init__(self):
            self.read_once = False

        def read(self, _size):
            if self.read_once:
                return b""
            self.read_once = True
            return b"xx"

        def close(self):
            pass

    monkeypatch.setattr(backup_module, "_open_checked",
                        lambda _path, **_kwargs: (GrowingStream(), 1))

    with pytest.raises(OpsError, match="grew while reading") as refused:
        backup_module.verified_file_hash(tmp_path / "growing", expected_size=None,
                                         stage="growing file")

    assert refused.value.code == "INTEGRITY_FAILED"


@pytest.mark.parametrize("tamper", ["database", "artifact"], ids=["database-hash", "artifact-hash"])
def test_restore_refuses_corrupt_database_or_artifact(tmp_path, tamper):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "restore-corrupt", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(
        conn, key="restore-corrupt", owner="worker", target=backup, clock=clock, store=store)

    if tamper == "database":
        path = backup / manifest["database_file"]
        content = bytearray(path.read_bytes())
        content[0] ^= 1
        path.write_bytes(content)
    else:
        (backup / "artifacts" / "evidence.bin").write_bytes(b"tamper!!")

    destination = tmp_path / "restore-corrupt-output"
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()


def test_source_catalog_over_byte_cap_is_refused_before_table_scan(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "oversized", {}, clock=clock)
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 1)
    monkeypatch.setattr(backup_module, "_table_row_counts",
                        lambda _conn: pytest.fail("oversized source was scanned"))

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="oversized", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert not list((tmp_path / "backup").glob("*.sqlite"))