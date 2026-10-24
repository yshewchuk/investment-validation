"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from engine.v2.foundation import ArtifactStore, to_document
from engine.v2.ops import backup as backup_module
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
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INVALID_REQUEST"
    assert refused.value.problem.retryable is False


@pytest.mark.parametrize("stale_bytes", [b"", b"SQLite format 3\x00"],
                         ids=["zero-byte", "truncated-header"])
def test_backup_publishes_verified_database_beside_unreferenced_stale_path(
        tmp_path, stale_bytes):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b7", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / "b7.sqlite"
    stale.write_bytes(stale_bytes)

    manifest = run_backup(
        conn, key="b7", owner="worker", target=backup, clock=clock, store=store)

    # The stale key-named path is never the published database: the manifest
    # selects a fresh immutable file, whatever remains at <key>.sqlite.
    assert manifest["database_file"] != "b7.sqlite"
    assert stale.read_bytes() == stale_bytes
    published = backup / manifest["database_file"]
    assert file_hash(published) == manifest["database"]
    assert manifest["database_content_hash"].startswith("sha256:")
    rebuilt = sqlite3.connect(published)
    try:
        assert integrity_errors(rebuilt) == []
        assert _row_counts(rebuilt) == _row_counts(conn)
        assert manifest["database_content_hash"] == backup_module._full_logical_digest(rebuilt)
    finally:
        rebuilt.close()
    assert manifest["database_rows"] == _row_counts(conn)
    assert conn.execute(
        "SELECT state FROM outbox WHERE logical_key='b7'").fetchone()[0] == "delivered"


def test_staged_row_count_mismatch_is_refused_with_table_details(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b8", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def corrupt(conn, destination, *, progress=None):
        real_backup_to(conn, destination, progress=progress)
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

    def racing_backup_to(source, destination, *, progress=None):
        writer = sqlite3.connect(database)
        try:
            writer.execute("INSERT INTO synthetic_change VALUES (1)")
            writer.commit()
        finally:
            writer.close()
        observed.append(_row_counts(source))
        real_backup_to(source, destination, progress=progress)

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


def test_staged_content_mismatch_is_refused_without_row_count_change(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    conn.execute("CREATE TABLE synthetic_content (value TEXT)")
    conn.execute("INSERT INTO synthetic_content VALUES ('original')")
    conn.commit()
    prepare_backup(conn, "b15", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def corrupt_content(source, destination, *, progress=None):
        real_backup_to(source, destination, progress=progress)
        damaged = sqlite3.connect(destination)
        try:
            damaged.execute("UPDATE synthetic_content SET value='tampered'")
            damaged.commit()
        finally:
            damaged.close()

    monkeypatch.setattr(backup_module, "backup_to", corrupt_content)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b15", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["reason"] == "content_digest"


def test_fresh_copy_outbox_state_tamper_is_refused_as_content_digest(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b22", {}, clock=clock)
    real_backup_to = backup_module.backup_to

    def tamper(source, destination, *, progress=None):
        real_backup_to(source, destination, progress=progress)
        damaged = sqlite3.connect(destination)
        try:
            damaged.execute("UPDATE outbox SET attempts=attempts+1")
            damaged.commit()
        finally:
            damaged.close()

    monkeypatch.setattr(backup_module, "backup_to", tamper)
    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b22", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert refused.value.problem.details["reason"] == "content_digest"


def test_retry_removes_only_matching_crash_staging_files(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "b12", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    stale = backup / f"{backup_module._stage_prefix('b12')}crash.staging.sqlite"
    other = backup / f"{backup_module._stage_prefix('b13')}crash.staging.sqlite"
    stale.write_bytes(b"junk")
    other.write_bytes(b"other")

    run_backup(conn, key="b12", owner="worker", target=backup, clock=clock, store=store)

    assert not stale.exists()
    assert other.exists()


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


def test_artifact_hash_mismatch_refuses_non_retryable_integrity_failure(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"good bytes", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b19", {"evidence.bin": ref}, clock=clock)
    object_path = store.root.joinpath(*ref.storage_key.split("/"))
    data = bytearray(object_path.read_bytes())
    data[0] ^= 0xFF
    object_path.unlink()
    object_path.write_bytes(bytes(data))

    def forbid(_ref):
        raise AssertionError("store.verify must not run")

    monkeypatch.setattr(store, "verify", forbid)

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b19", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_oversized_backing_object_is_refused_before_verify(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"x" * (4 << 20), schema_ref="evidence.v1.0")
    prepare_backup(conn, "b27", {"evidence.bin": ref}, clock=clock)
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 2 << 20)
    calls = []

    def spy(_ref):
        calls.append(_ref)
        raise AssertionError("verify must not run before the size preflight")

    monkeypatch.setattr(store, "verify", spy)

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b27", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert calls == []
    assert not list((tmp_path / "backup").rglob("evidence.bin"))


def test_size_mismatched_backing_object_is_refused_before_verify(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    mismatched = {**to_document(ref), "byte_size": ref.byte_size - 1}
    prepare_backup(conn, "b28", {"evidence.bin": mismatched}, clock=clock)
    calls = []

    def spy(_ref):
        calls.append(_ref)
        raise AssertionError("verify must not run before the size preflight")

    monkeypatch.setattr(store, "verify", spy)

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="b28", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert calls == []


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
        with backup_module._backup_lock(target, "same-key"):
            order.append("second-enter")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(5)
    second_thread.start()
    assert second_thread.is_alive()
    assert "second-enter" not in order
    release.set()
    first_thread.join(5)
    second_thread.join(5)

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


def test_restore_reads_manifest_within_the_existing_bound(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "cap", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="cap", owner="worker", target=backup,
                          clock=clock, store=store)
    (backup / manifest["database_file"]).rename(backup / "cap.sqlite")
    manifest["artifacts"]["evidence.bin"]["padding"] = (
        "x" * (backup_module.MAX_MANIFEST_BYTES + 1))
    (backup / "cap.manifest.json").write_text(json.dumps(manifest))
    assert (backup / "cap.manifest.json").stat().st_size > \
        backup_module.MAX_MANIFEST_BYTES
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()


def test_restore_catches_deeply_nested_manifest_json(tmp_path):
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "nested.manifest.json").write_text("[" * 100_000)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_restore_rejects_unsafe_database_filename(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "unsafe", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="unsafe", owner="worker", target=backup,
                          clock=clock, store=store)
    manifest["database_file"] = "../escape.sqlite"
    (backup / "unsafe.manifest.json").write_text(json.dumps(manifest))
    destination = tmp_path / "restored"

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, destination)

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert not destination.exists()


@pytest.mark.parametrize("referenced", ["database", "artifact"])
@pytest.mark.parametrize("damage", ["missing", "symlink"])
def test_restore_missing_or_symlinked_source_is_an_integrity_failure(
        tmp_path, referenced, damage):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "restore", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="restore", owner="worker", target=backup,
                          clock=clock, store=store)
    source = (backup / manifest["database_file"] if referenced == "database"
              else backup / "artifacts" / "evidence.bin")
    if damage == "missing":
        source.unlink()
    else:
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(source.read_bytes())
        source.unlink()
        source.symlink_to(replacement)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


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


def test_delivered_backup_retry_refuses_tampered_artifact(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "tampered-delivery", {"evidence.bin": ref}, clock=clock)
    backup_root = tmp_path / "backup"
    run_backup(conn, key="tampered-delivery", owner="worker", target=backup_root,
               clock=clock, store=store)
    (backup_root / "artifacts" / "evidence.bin").write_bytes(b"partial")

    with pytest.raises(OpsError) as refused:
        run_backup(conn, key="tampered-delivery", owner="retry", target=backup_root,
                   clock=clock, store=store)

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
    manifest.pop("database_content_hash")
    manifest["artifacts"]["evidence.bin"].pop("byte_size")
    (backup / "legacy.manifest.json").write_text(json.dumps(manifest))

    restored = restore_backup(backup, tmp_path / "legacy-restored")

    assert (restored / "ops.sqlite").is_file()
    assert (restored / "artifacts" / "evidence.bin").read_bytes() == b"evidence"


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


def test_full_logical_digest_changes_on_schema_change():
    conn = sqlite3.connect(":memory:")
    try:
        before = backup_module._full_logical_digest(conn, stage="schema-test")
        conn.execute("CREATE TABLE schema_probe (value TEXT)")
        after = backup_module._full_logical_digest(conn, stage="schema-test")
    finally:
        conn.close()
    assert after != before


def test_restore_refuses_database_over_the_byte_cap(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "cap", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="cap", owner="worker", target=backup,
                          clock=clock, store=store)
    size = (backup / manifest["database_file"]).stat().st_size
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", size - 1)
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_restore_refuses_a_published_database_size_mismatch(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "size", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="size", owner="worker", target=backup,
                          clock=clock, store=store)
    manifest["database_size"] += 1
    (backup / "size.manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False


def test_restore_refuses_a_published_database_hash_mismatch(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "hash", {}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="hash", owner="worker", target=backup,
                          clock=clock, store=store)
    published = backup / manifest["database_file"]
    data = bytearray(published.read_bytes())
    data[-1] ^= 0xFF
    published.write_bytes(data)

    with pytest.raises(OpsError) as refused:
        restore_backup(backup, tmp_path / "restored")

    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False
    assert "mismatch" in str(refused.value)
    assert not (tmp_path / "restored" / "ops.sqlite").exists()


def test_backup_keepalive_runs_outside_snapshot_transaction(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "heartbeat", {}, clock=clock)
    observed_transactions = []
    renewals = []
    original_digest = backup_module._full_logical_digest

    def digest_with_heartbeat(snapshot, *, stage="database-digest", keepalive=None):
        observed_transactions.append(conn.in_transaction)
        if keepalive is not None:
            keepalive()
        return original_digest(snapshot, stage=stage)

    def keepalive():
        assert not conn.in_transaction
        renewals.append(True)

    monkeypatch.setattr(backup_module, "_full_logical_digest", digest_with_heartbeat)
    run_backup(conn, key="heartbeat", owner="worker", target=tmp_path / "backup",
               clock=clock, store=store, keepalive=keepalive)

    assert observed_transactions and not any(observed_transactions)
    assert renewals


def test_sparse_file_over_the_byte_cap_is_refused_without_a_destination(
        tmp_path, monkeypatch):
    monkeypatch.setattr(backup_module, "MAX_BACKUP_FILE_BYTES", 1024)
    sparse = tmp_path / "sparse.bin"
    with sparse.open("wb") as stream:
        stream.truncate(1025)
    destination = tmp_path / "copy.bin"

    with pytest.raises(OpsError) as refused:
        backup_module._stream_copy(sparse, destination, stage="sparse copy")
    assert refused.value.code == "INTEGRITY_FAILED"
    assert refused.value.problem.retryable is False

    assert not destination.exists()


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

    with pytest.raises(OpsError) as refused:
        backup_module.verified_file_hash(tmp_path / "growing", expected_size=None,
                                         stage="growing file")

    assert refused.value.code == "INTEGRITY_FAILED"


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