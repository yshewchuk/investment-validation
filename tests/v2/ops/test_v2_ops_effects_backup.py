"""Local immutable backup and isolated restore controls."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

import engine.v2.ops.backup as backup_module
from engine.v2.foundation import ArtifactError, ArtifactStore
from engine.v2.ops.backup import prepare_backup, restore_backup, run_backup
from engine.v2.ops.effects_graph import backup_effect
from engine.v2.ops.errors import fail, OpsError
from engine.v2.ops.lifecycle import Outcome, commit_attempt
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_effects_graph import (
    SESSION, _commit, _open, _params, _row, _seed_decisions, _submit_and_claim,
)


def test_backup_copies_artifacts_and_restores_new_root(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "b1", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    recorded_uris = []
    recorded_sql = []
    real_connect = backup_module.sqlite3.connect

    def recording_connect(database, *args, **kwargs):
        connection = real_connect(database, *args, **kwargs)
        if kwargs.get("uri"):
            recorded_uris.append(database)
            connection.set_trace_callback(recorded_sql.append)
        return connection

    monkeypatch.setattr(backup_module.sqlite3, "connect", recording_connect)
    manifest = run_backup(conn, key="b1", owner="worker", target=backup, clock=clock, store=store)
    assert manifest["database_byte_size"] == (backup / "b1.sqlite").stat().st_size
    assert recorded_uris == [(backup / "b1.sqlite").resolve().as_uri() + "?mode=ro"]
    assert any(statement.upper() == "PRAGMA QUICK_CHECK" for statement in recorded_sql)
    assert manifest["artifacts"]["evidence.bin"]["content_hash"] == ref.content_hash
    assert manifest["database"] == "sha256:" + hashlib.sha256(
        (backup / "b1.sqlite").read_bytes()).hexdigest()
    restored = restore_backup(backup, tmp_path / "restored")
    assert (restored / "artifacts" / "evidence.bin").read_bytes() == b"evidence"
    assert conn.execute("SELECT state FROM outbox WHERE kind='backup'").fetchone()[0] == "delivered"
    legacy_manifest = {name: value for name, value in manifest.items() if name != "database_byte_size"}
    (backup / "b1.manifest.json").write_bytes(
        json.dumps(legacy_manifest, sort_keys=True, separators=(",", ":")).encode())
    restored_legacy = restore_backup(backup, tmp_path / "restored-legacy-manifest")
    assert "sha256:" + hashlib.sha256(
        (restored_legacy / "ops.sqlite").read_bytes()).hexdigest() == manifest["database"]
    with pytest.raises(OpsError) as error:
        restore_backup(backup, tmp_path / "restored")
    assert error.value.code == "INVALID_REQUEST"
    real_read_bounded_json = backup_module.read_bounded_json
    recorded_max_bytes = []

    def recording_read_bounded_json(path, max_bytes):
        recorded_max_bytes.append(max_bytes)
        return real_read_bounded_json(path, max_bytes)

    monkeypatch.setattr(backup_module, "read_bounded_json", recording_read_bounded_json)
    manifest_path = next(backup.glob("*.manifest.json"))
    manifest_path.write_bytes(b"x" * (1024 * 1024 + 1))
    with pytest.raises(OpsError) as error:
        restore_backup(backup, tmp_path / "restored-oversized-manifest")
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == "backup manifest is invalid"
    assert recorded_max_bytes == [backup_module._MAX_MANIFEST_BYTES]


def test_restore_rejects_manifest_size_mismatches(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "size-mismatch", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key="size-mismatch", owner="worker", target=backup,
                          clock=clock, store=store)
    manifest_path = next(backup.glob("*.manifest.json"))
    destination = tmp_path / "restored"

    modified_manifest = {
        **manifest, "database_byte_size": manifest["database_byte_size"] + 1,
    }
    manifest_path.write_bytes(
        json.dumps(modified_manifest, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(OpsError) as error:
        restore_backup(backup, destination)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == "restored catalog size mismatch"
    assert not destination.exists()

    modified_manifest = {
        **manifest,
        "artifacts": {
            **manifest["artifacts"],
            "evidence.bin": {
                **manifest["artifacts"]["evidence.bin"],
                "byte_size": manifest["artifacts"]["evidence.bin"]["byte_size"] + 1,
            },
        },
    }
    manifest_path.write_bytes(
        json.dumps(modified_manifest, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(OpsError) as error:
        restore_backup(backup, destination)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == "restored artifact size mismatch"


def test_backup_rejects_artifact_size_mismatch(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    prepare_backup(conn, "bad-artifact-size", {"evidence.bin": ref}, clock=clock)
    backup = tmp_path / "backup"

    def short_copy(source, destination):
        destination.write_bytes(b"x")

    monkeypatch.setattr(backup_module.shutil, "copyfile", short_copy)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="bad-artifact-size", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == "backup artifact size mismatch"
    state = conn.execute(
        "SELECT state FROM outbox WHERE kind='backup' AND logical_key='bad-artifact-size'"
    ).fetchone()[0]
    assert state == "running"


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


def test_backup_reject_fresh_catalog_integrity_violation(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript(
        "CREATE TABLE backup_parent(id INTEGER PRIMARY KEY);"
        "CREATE TABLE backup_child(parent_id INTEGER REFERENCES backup_parent(id));"
        "INSERT INTO backup_child(parent_id) VALUES (42);"
    )
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "bad-source-integrity", {}, clock=clock)
    backup = tmp_path / "backup"
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="bad-source-integrity", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "catalog backup failed its integrity check"
    assert error.value.problem.retryable is True
    assert not (backup / "bad-source-integrity.sqlite").exists()


def test_backup_rejects_non_ok_quick_check(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "bad-quick-check", {}, clock=clock)
    real_connect = backup_module.sqlite3.connect

    class _BadResult:
        def fetchall(self):
            return [("not ok",)]

    class _ReadOnlyProbe:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, statement):
            if statement.upper() == "PRAGMA QUICK_CHECK":
                return _BadResult()
            return self.connection.execute(statement)

        def close(self):
            self.connection.close()

    def bad_quick_check(database, *args, **kwargs):
        connection = real_connect(database, *args, **kwargs)
        if kwargs.get("uri"):
            return _ReadOnlyProbe(connection)
        return connection

    monkeypatch.setattr(backup_module.sqlite3, "connect", bad_quick_check)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="bad-quick-check", owner="worker", target=tmp_path / "backup",
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup database verification failed"
    assert error.value.problem.retryable is True
    assert not (tmp_path / "backup" / "bad-quick-check.sqlite").exists()


@pytest.mark.parametrize(("contents", "message"), [
    (b"", "backup database is empty"),
    (b"not a sqlite database", "backup database verification failed"),
    ("oversized", "backup database size differs from its page count"),
])
def test_backup_bad_copy_remains_retryable(tmp_path, monkeypatch, contents, message):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "bad-copy", {}, clock=clock)
    backup = tmp_path / "backup"
    real_backup_to = backup_module.backup_to

    def bad_copy(_conn, destination):
        if contents == "oversized":
            real_backup_to(_conn, destination)
            with destination.open("ab") as stream:
                stream.write(b"x")
        else:
            destination.write_bytes(contents)

    monkeypatch.setattr(backup_module, "backup_to", bad_copy)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="bad-copy", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == message
    assert error.value.problem.retryable is True
    assert not (backup / "bad-copy.sqlite").exists()
    state = conn.execute(
        "SELECT state FROM outbox WHERE kind='backup' AND logical_key='bad-copy'"
    ).fetchone()[0]
    assert state == "running"

    monkeypatch.setattr(backup_module, "backup_to", real_backup_to)
    clock.advance(301)
    manifest = run_backup(conn, key="bad-copy", owner="retry", target=backup,
                          clock=clock, store=store)
    assert manifest["database"] == "sha256:" + hashlib.sha256(
        (backup / "bad-copy.sqlite").read_bytes()).hexdigest()
    assert conn.execute(
        "SELECT state FROM outbox WHERE kind='backup' AND logical_key='bad-copy'"
    ).fetchone()[0] == "delivered"


def test_backup_effect_preserves_safe_refusal_message(tmp_path, monkeypatch):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])
        claim = _submit_and_claim(
            conn, clock, supervisor, kind="backup", key="backup-cause",
            parameters=_params("backup", SESSION, scope))

        def reject_copy(_database, **_kwargs):
            raise fail("BACKUP_FAILED", "backup database is empty")

        monkeypatch.setattr(backup_module, "_verified_database_hash", reject_copy)
        with pytest.raises(OpsError) as error:
            backup_effect(conn, store, claim, root, clock=clock)
        assert error.value.code == "BACKUP_FAILED"
        assert error.value.problem.message == "backup database is empty"
        assert error.value.problem.retryable is True
        row = conn.execute(
            "SELECT state, receipt_json FROM outbox WHERE kind='backup'"
        ).fetchone()
        assert row["state"] == "pending"
        assert json.loads(row["receipt_json"]) == {"error": "backup database is empty"}
    finally:
        conn.close()


def test_backup_effect_verifier_crash_records_generic_error(tmp_path, monkeypatch):
    conn, clock, supervisor, store, root = _open(tmp_path)
    try:
        scope = "shadow"
        _seed_decisions(conn, clock, scope, SESSION, predictions=[_row("evt-1", "pred")])
        claim = _submit_and_claim(
            conn, clock, supervisor, kind="backup", key="backup-cause",
            parameters=_params("backup", SESSION, scope))

        def crash(_database, **_kwargs):
            raise RuntimeError("simulated verifier crash")

        monkeypatch.setattr(backup_module, "_verified_database_hash", crash)
        with pytest.raises(RuntimeError, match="simulated verifier crash"):
            backup_effect(conn, store, claim, root, clock=clock)
        row = conn.execute(
            "SELECT state, receipt_json FROM outbox WHERE kind='backup'"
        ).fetchone()
        assert row["state"] == "pending"
        assert json.loads(row["receipt_json"]) == {"error": "backup_failed"}
    finally:
        conn.close()


def test_backup_to_oserror_is_retryable_and_cleans_copy(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "copy-oserror", {}, clock=clock)
    backup = tmp_path / "backup"

    def interrupted_copy(_conn, destination):
        destination.write_bytes(b"partial")
        raise OSError("private detail")

    monkeypatch.setattr(backup_module, "backup_to", interrupted_copy)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="copy-oserror", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup database copy or verification failed"
    assert error.value.problem.retryable is True
    assert not (backup / "copy-oserror.sqlite").exists()


def test_backup_cleanup_failure_is_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "cleanup-failure", {}, clock=clock)
    backup = tmp_path / "backup"

    real_backup_to = backup_module.backup_to

    def invalid_backup_to(_conn, destination):
        real_backup_to(_conn, destination)
        with destination.open("r+b") as stream:
            stream.write(b"x")

    real_unlink = Path.unlink
    cleanup_attempts = []

    def selective_unlink(self, *args, **kwargs):
        if self.name == "cleanup-failure.sqlite":
            cleanup_attempts.append(self)
            raise OSError("simulated cleanup failure")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(backup_module, "backup_to", invalid_backup_to)
    monkeypatch.setattr(Path, "unlink", selective_unlink)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="cleanup-failure", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == (
        "backup database verification failed; backup database cleanup failed")
    assert error.value.problem.retryable is True
    assert (backup / "cleanup-failure.sqlite").exists()
    assert cleanup_attempts == [backup / "cleanup-failure.sqlite"]


def test_backup_reject_existing_copy_with_invalid_sqlite(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "fk-copy", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    destination = backup / "fk-copy.sqlite"
    backup_module.backup_to(conn, destination)
    with destination.open("r+b") as stream:
        stream.write(b"x")
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="fk-copy", owner="worker", target=backup, clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup database verification failed"
    assert error.value.problem.retryable is True
    assert not destination.exists()
    state = conn.execute(
        "SELECT state FROM outbox WHERE kind='backup' AND logical_key='fk-copy'"
    ).fetchone()[0]
    assert state == "running"


def test_backup_rejects_existing_copy_with_foreign_key_violation(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "fk-existing", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    destination = backup / "fk-existing.sqlite"
    existing = backup_module.sqlite3.connect(destination)
    existing.execute("PRAGMA foreign_keys=OFF")
    existing.executescript(
        "CREATE TABLE existing_parent(id INTEGER PRIMARY KEY);"
        "CREATE TABLE existing_child(parent_id INTEGER REFERENCES existing_parent(id));"
        "INSERT INTO existing_child(parent_id) VALUES (42);"
    )
    existing.commit()
    existing.close()
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="fk-existing", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup database verification failed"
    assert error.value.problem.retryable is True
    assert not destination.exists()


def test_backup_rejects_oversized_manifest(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "oversized-manifest", {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    (backup / "oversized-manifest.manifest.json").write_bytes(b"x" * (1024 * 1024 + 1))

    real_open = Path.open
    read_sizes = []

    class _RecordingStream:
        def __init__(self, stream):
            self._stream = stream

        def __enter__(self):
            self._stream.__enter__()
            return self

        def __exit__(self, *args):
            return self._stream.__exit__(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            return self._stream.read(size)

    def recording_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else "r")
        if self.name == "oversized-manifest.manifest.json" and mode == "rb":
            return _RecordingStream(stream)
        return stream

    monkeypatch.setattr(Path, "open", recording_open)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="oversized-manifest", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup manifest exceeds 1 MiB"
    assert error.value.problem.retryable is True
    assert read_sizes == [1024 * 1024 + 1]


@pytest.mark.parametrize(("contents", "message"), [
    (b"{", "backup manifest is invalid"),
    (b"\xff", "backup manifest is invalid"),
    (b"{}", "backup manifest differs"),
])
def test_backup_rejects_invalid_existing_manifest(tmp_path, contents, message):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "bad-manifest", {}, clock=clock)
    backup = tmp_path / "backup"

    def crash(point):
        if point == "after_manifest_before_ack":
            raise RuntimeError("ack lost")

    with pytest.raises(RuntimeError, match="ack lost"):
        run_backup(conn, key="bad-manifest", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash)
    (backup / "bad-manifest.manifest.json").write_bytes(contents)
    clock.advance(301)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="bad-manifest", owner="retry", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == message


def test_backup_manifest_read_error_is_typed_retryable(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "unreadable-manifest"
    prepare_backup(conn, key, {}, clock=clock)
    backup = tmp_path / "backup"
    backup.mkdir()
    manifest_path = backup / (key + ".manifest.json")
    manifest_path.write_bytes(b"{}")
    real_open = Path.open

    def unreadable_manifest(self, *args, **kwargs):
        if self == manifest_path:
            raise OSError("simulated manifest read failure")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unreadable_manifest)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup manifest could not be read"
    assert error.value.problem.retryable is True
    assert conn.execute(
        "SELECT state FROM outbox WHERE kind='backup' AND logical_key=?", (key,)
    ).fetchone()[0] == "running"


def test_delivered_backup_replay_rejects_same_size_database_hash_mismatch(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "replay-hash-mismatch"
    prepare_backup(conn, key, {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    database = backup / (key + ".sqlite")
    original_size = database.stat().st_size

    copy = sqlite3.connect(database)
    version = copy.execute("PRAGMA user_version").fetchone()[0]
    copy.execute(f"PRAGMA user_version = {version + 1}")
    copy.close()
    assert database.stat().st_size == original_size

    with pytest.raises(OpsError) as error:
        run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup database copy hash mismatch"
    assert error.value.problem.retryable is True


def test_restore_reject_malformed_manifest_before_creating_destination(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "b1.manifest.json").write_bytes(b"{}")
    destination = tmp_path / "destination"
    with pytest.raises(OpsError) as error:
        restore_backup(source, destination)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == "backup manifest is invalid"
    assert not destination.exists()


def test_backup_fsyncs_database_copy(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "fsync-copy", {}, clock=clock)
    backup = tmp_path / "backup"
    publication_events = []
    real_fsync = backup_module.os.fsync
    real_replace = backup_module.os.replace

    def recording_fsync(fd):
        if Path(f"/proc/self/fd/{fd}").resolve().name == "fsync-copy.sqlite":
            publication_events.append("database_fsync")
        return real_fsync(fd)

    def recording_replace(source, destination):
        publication_events.append("manifest_replace")
        return real_replace(source, destination)

    real_open = Path.open
    open_modes = []

    def recording_open(self, *args, **kwargs):
        stream = real_open(self, *args, **kwargs)
        mode = kwargs.get("mode", args[0] if args else "r")
        if self.name == "fsync-copy.sqlite":
            open_modes.append(mode)
        return stream

    monkeypatch.setattr(backup_module.os, "fsync", recording_fsync)
    monkeypatch.setattr(backup_module.os, "replace", recording_replace)
    monkeypatch.setattr(Path, "open", recording_open)
    run_backup(conn, key="fsync-copy", owner="worker", target=backup,
               clock=clock, store=store)
    assert publication_events.count("database_fsync") == 1
    assert publication_events.index("database_fsync") < publication_events.index("manifest_replace")
    assert "rb+" in open_modes


@pytest.mark.parametrize("failed_stage", ["fsync", "hash"])
def test_backup_fsync_or_hash_failure_is_retryable(tmp_path, monkeypatch, failed_stage):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "copy-" + failed_stage + "-failure"
    prepare_backup(conn, key, {}, clock=clock)
    backup = tmp_path / "backup"
    real_fsync = backup_module.os.fsync
    real_hash = backup_module.file_hash

    def fail_fsync(_fd):
        raise OSError("simulated database fsync failure")

    def fail_hash(path, **kwargs):
        if Path(path).name == key + ".sqlite":
            raise OSError("simulated database hash failure")
        return real_hash(path, **kwargs)

    if failed_stage == "fsync":
        monkeypatch.setattr(backup_module.os, "fsync", fail_fsync)
    else:
        monkeypatch.setattr(backup_module, "file_hash", fail_hash)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup database verification failed"
    assert error.value.problem.retryable is True
    assert not (backup / (key + ".sqlite")).exists()
    assert not (backup / (key + ".manifest.json")).exists()


def test_backup_accept_manifest_at_size_cap(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "cap-manifest", {}, clock=clock)
    backup = tmp_path / "backup"

    def crash(point):
        if point == "after_manifest_before_ack":
            raise RuntimeError("ack lost")

    with pytest.raises(RuntimeError, match="ack lost"):
        run_backup(conn, key="cap-manifest", owner="worker", target=backup,
                   clock=clock, store=store, fault=crash)
    manifest_path = backup / "cap-manifest.manifest.json"
    manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(
        manifest + b" " * (1024 * 1024 - len(manifest)))
    clock.advance(301)
    run_backup(conn, key="cap-manifest", owner="retry", target=backup,
               clock=clock, store=store)
    assert conn.execute(
        "SELECT state FROM outbox WHERE kind='backup' AND logical_key='cap-manifest'"
    ).fetchone()[0] == "delivered"


def test_backup_rejects_manifest_exceeding_size_limit(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    prepare_backup(conn, "small-limit", {}, clock=clock)
    backup = tmp_path / "backup"
    monkeypatch.setattr(backup_module, "_MAX_MANIFEST_BYTES", 1)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key="small-limit", owner="worker", target=backup,
                   clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.message == "backup manifest exceeds 1 MiB"
    assert error.value.problem.retryable is True
    assert not (backup / "small-limit.manifest.json").exists()


def test_manifest_success_before_ack_is_idempotent_and_keys_are_safe(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    ref = store.publish_bytes(b"evidence", schema_ref="evidence.v1.0")
    with pytest.raises(ArtifactError, match="UNSAFE_PATH"):
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
    conn.execute("CREATE TABLE backup_growth(value TEXT)")
    conn.executemany("INSERT INTO backup_growth(value) VALUES (?)", [("x" * 1000,)] * 1000)
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


@pytest.mark.parametrize("damage", [b"", b"not sqlite"])
def test_delivered_backup_replay_reject_damaged_database(tmp_path, damage):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "replay-damaged"
    prepare_backup(conn, key, {}, clock=clock)
    backup = tmp_path / "backup"
    run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    (backup / (key + ".sqlite")).write_bytes(damage)
    with pytest.raises(OpsError) as error:
        run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.retryable is True


@pytest.mark.parametrize("kind", ["database", "artifact"])
def test_backup_refuses_files_over_max_bytes(tmp_path, kind, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "max-bytes-" + kind

    if kind == "database":
        monkeypatch.setattr(backup_module, "MAX_BACKUP_BYTES", 1)
        prepare_backup(conn, key, {}, clock=clock)
    else:
        catalog_bytes = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute(
            "PRAGMA page_size").fetchone()[0]
        monkeypatch.setattr(backup_module, "MAX_BACKUP_BYTES", catalog_bytes)
        artifact_ref = store.publish_bytes(
            b"x" * (catalog_bytes + 1), schema_ref="evidence.v1.0")
        prepare_backup(conn, key, {"artifact.bin": artifact_ref}, clock=clock)

        def fail_verify(ref):
            raise AssertionError("verify should not be called")

        monkeypatch.setattr(store, "verify", fail_verify)

    backup = tmp_path / "backup"
    with pytest.raises(OpsError) as error:
        run_backup(conn, key=key, owner="worker", target=backup, clock=clock, store=store)
    assert error.value.code == "BACKUP_FAILED"
    assert error.value.problem.retryable is True


@pytest.mark.parametrize(
    ("kind", "expected_message"),
    [("catalog", "restored catalog exceeds size limit"),
     ("artifact", "restored artifact exceeds size limit")],
)
def test_restore_refuses_files_over_max_backup_bytes(tmp_path, kind, expected_message,
                                                      monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path / "objects")
    key = "restore-cap-" + kind
    catalog_bytes = conn.execute("PRAGMA page_count").fetchone()[0] * conn.execute(
        "PRAGMA page_size").fetchone()[0]
    artifacts = {}
    if kind == "artifact":
        artifacts["evidence.bin"] = store.publish_bytes(
            b"x" * (catalog_bytes + 1), schema_ref="evidence.v1.0")
    prepare_backup(conn, key, artifacts, clock=clock)
    backup = tmp_path / "backup"
    manifest = run_backup(conn, key=key, owner="worker", target=backup,
                          clock=clock, store=store)
    limit = 1 if kind == "catalog" else manifest["database_byte_size"]
    monkeypatch.setattr(backup_module, "MAX_BACKUP_BYTES", limit)

    destination = tmp_path / "restore"
    with pytest.raises(OpsError) as error:
        restore_backup(backup, destination)
    assert error.value.code == "INTEGRITY_FAILED"
    assert error.value.problem.message == expected_message
    if kind == "catalog":
        assert not destination.exists()
    else:
        assert not (destination / "artifacts" / "evidence.bin").exists()
