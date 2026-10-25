"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    format_timestamp,
    from_document,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.outbox import claim, complete, enqueue

MAX_BACKUP_FILE_BYTES = 1 << 40
_COPY_CHUNK = 1024 * 1024


def _open_checked_file(path, *, expected_size, stage):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ELOOP, errno.ENOTDIR):
            raise fail("INTEGRITY_FAILED", f"{stage} is missing or unsafe") from exc
        raise fail("WORKER_FAILED", f"{stage} could not be opened") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BACKUP_FILE_BYTES
                or (expected_size is not None
                    and (type(expected_size) is not int or expected_size != info.st_size))):
            raise fail("INTEGRITY_FAILED", f"{stage} is not a bounded regular file")
        stream = os.fdopen(fd, "rb")
        fd = -1
        return stream, info.st_size
    finally:
        if fd >= 0:
            os.close(fd)


def _checked_file_hash(path, *, stage, expected_size=None):
    stream, size = _open_checked_file(path, expected_size=expected_size, stage=stage)
    digest = hashlib.sha256()
    copied = 0
    with stream:
        while copied <= size:
            try:
                chunk = stream.read(min(_COPY_CHUNK, size + 1 - copied))
            except OSError as exc:
                raise fail("WORKER_FAILED", f"{stage} could not be read") from exc
            if not chunk:
                break
            copied += len(chunk)
            digest.update(chunk)
    if copied != size:
        raise fail("INTEGRITY_FAILED", f"{stage} size changed while reading")
    return "sha256:" + digest.hexdigest()


def _copy_verified_file(source, destination, *, expected_size, expected_hash, stage):
    stream, size = _open_checked_file(source, expected_size=expected_size, stage=stage)
    digest = hashlib.sha256()
    copied = 0
    with stream:
        try:
            output = Path(destination).open("wb")
        except OSError as exc:
            raise fail("WORKER_FAILED", f"{stage} output could not be opened") from exc
        with output:
            while True:
                try:
                    chunk = stream.read(_COPY_CHUNK)
                except OSError as exc:
                    raise fail("WORKER_FAILED", f"{stage} could not be read") from exc
                if not chunk:
                    break
                copied += len(chunk)
                if copied > size:
                    raise fail("INTEGRITY_FAILED", f"{stage} grew while reading")
                digest.update(chunk)
                try:
                    output.write(chunk)
                except OSError as exc:
                    raise fail("WORKER_FAILED", f"{stage} output could not be written") from exc
            try:
                output.flush()
                os.fsync(output.fileno())
            except OSError as exc:
                raise fail("WORKER_FAILED", f"{stage} output could not be synced") from exc
    if copied != size or "sha256:" + digest.hexdigest() != expected_hash:
        raise fail("INTEGRITY_FAILED", f"{stage} size or hash mismatch")


def prepare_backup(conn, key, artifacts, *, clock):
    """Enqueue one ``backup`` effect for ``key``, idempotent across attempts.

    A retry (after ``fail_effect`` returns the effect to pending) must call
    this again before re-running ``run_backup``, but the effect already
    fixed its own ``cutoff`` and artifact list on the FIRST attempt — a
    retry that rebuilt the payload with the clock's current time would hand
    ``outbox.enqueue`` a different payload hash for the same logical key and
    fail with ``IDEMPOTENCY_CONFLICT`` (P2-C07). So an existing ``backup``
    effect for this key is returned as-is, with no new payload built at all.
    """
    if len(safe_relative_path(key)) != 1:
        raise fail("INVALID_REQUEST", "unsafe backup key")
    with transaction(conn):
        existing = conn.execute("SELECT effect_id FROM outbox WHERE kind='backup' AND logical_key=?",
                                (key,)).fetchone()
        if existing is not None:
            return existing["effect_id"]
        payload = {"backup_key": key, "artifacts": artifacts,
                   "cutoff": format_timestamp(clock.now()), "schema_version": "catalog_backup.v1.0"}
        return enqueue(conn, "backup", key, payload)


def _no_keepalive():
    return None


def _delivered_receipt(conn, kind, logical_key):
    """The stored receipt for an already-``delivered`` outbox row under
    ``(kind, logical_key)``, or ``None`` if no such row is delivered yet.

    #98: lets a retry after a crash between the effect's own commit and the
    caller's attempt commit treat an already-completed effect as success,
    instead of failing permanently with STALE_EXPECTATION because
    ``claim()`` no longer matches a delivered row.
    """
    row = conn.execute(
        "SELECT receipt_json FROM outbox WHERE kind=? AND logical_key=? AND state='delivered'",
        (kind, logical_key)).fetchone()
    return None if row is None else json.loads(row["receipt_json"])


def _delivered_or_stale(conn, kind, logical_key):
    """``_delivered_receipt`` or STALE_EXPECTATION: the claim-failure branch."""
    delivered = _delivered_receipt(conn, kind, logical_key)
    if delivered is None:
        raise fail("STALE_EXPECTATION", "backup effect is not pending")
    return delivered


def _table_row_counts(conn):
    """Deterministic row count per non-internal table, keyed and ordered by name."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    counts = {}
    for name in tables:
        quoted = '"' + name.replace('"', '""') + '"'
        counts[name] = conn.execute(f"SELECT COUNT(*) FROM {quoted}").fetchone()[0]
    return counts


def _verify_staged(staged, source_rows):
    """Verify a staged copy's row counts against the source snapshot."""
    probe = sqlite3.connect(staged)
    try:
        staged_rows = _table_row_counts(probe)
    finally:
        probe.close()
    if staged_rows != source_rows:
        differing = [name for name in sorted(set(source_rows) | set(staged_rows))
                     if source_rows.get(name) != staged_rows.get(name)]
        raise fail("INTEGRITY_FAILED", "backup row counts differ from source",
                   details={"tables": differing})


def _stage_database(conn, target, key, database, source_rows):
    try:
        stage_fd, stage_name = tempfile.mkstemp(
            dir=str(target), prefix=f".{key}.", suffix=".staging.sqlite")
        os.close(stage_fd)
        os.unlink(stage_name)
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup database staging failed") from exc
    staged = Path(stage_name)
    try:
        try:
            backup_to(conn, staged)
        except sqlite3.Error as exc:
            raise fail("WORKER_FAILED", "SQLite backup copy failed") from exc
        _verify_staged(staged, source_rows)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, database)
        staged = None
        fsync_directory(target)
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup database publication failed") from exc
    finally:
        if staged is not None:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass


def _copy_artifact(store, target, name, ref, fault):
    """Atomically publish one verified artifact and run its fault hook."""
    try:
        source = store.verify(ref)
        destination = target / "artifacts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage_fd, stage_name = tempfile.mkstemp(
            dir=str(destination.parent), prefix=f".{name}.", suffix=".staging")
        os.close(stage_fd)
        os.unlink(stage_name)
    except ArtifactError as exc:
        raise fail("INTEGRITY_FAILED", "backup artifact is invalid or oversized") from exc
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup artifact staging failed") from exc
    staged = Path(stage_name)
    try:
        _copy_verified_file(source, staged, expected_size=ref.byte_size,
                            expected_hash=ref.content_hash, stage="backup artifact")
        os.replace(staged, destination)
        staged = None
        fsync_directory(destination.parent)
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup artifact publication failed") from exc
    finally:
        if staged is not None:
            try:
                staged.unlink()
            except FileNotFoundError:
                pass
    if fault:
        fault(name)
    return {"content_hash": ref.content_hash, "byte_size": ref.byte_size}


def _validate_backup_artifact_names(artifacts):
    for name in artifacts:
        try:
            if len(safe_relative_path(name)) != 1:
                raise ValueError("unsafe backup artifact name")
        except (ArtifactError, ValueError) as exc:
            raise fail("INVALID_REQUEST", "unsafe backup artifact name") from exc


def _copy_artifacts(payload, store, target, fault, keepalive):
    copied = {}
    for name, document in payload.get("artifacts", {}).items():
        keepalive()
        ref = from_document(ArtifactRef, document)
        copied[name] = _copy_artifact(store, target, name, ref, fault)
    return copied


def _publish_manifest(manifest_path, manifest):
    """Atomically write the verified manifest with a directory fsync."""
    temporary = manifest_path.with_suffix(".pending")
    try:
        temporary.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        fsync_directory(manifest_path.parent)
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup manifest publication failed") from exc


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=_no_keepalive):
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _delivered_or_stale(conn, "backup", key)
    payload = json.loads(effect["payload_json"])
    _validate_backup_artifact_names(payload.get("artifacts", {}))
    target = Path(target)
    if os.path.lexists(target) and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    target.mkdir(parents=True, exist_ok=True)
    database = target / (key + ".sqlite")
    manifest_path = target / (key + ".manifest.json")
    try:
        conn.execute("BEGIN")
        try:
            source_rows = _table_row_counts(conn)
            _stage_database(conn, target, key, database, source_rows)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    except sqlite3.Error as exc:
        raise fail("WORKER_FAILED", "backup source snapshot failed") from exc
    copied = _copy_artifacts(payload, store, target, fault, keepalive)
    manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                "cutoff": payload["cutoff"],
                "database": _checked_file_hash(database, stage="backup database"),
                "database_rows": source_rows, "artifacts": copied}
    _publish_manifest(manifest_path, manifest)
    if fault:
        fault("after_manifest_before_ack")
    complete(conn, effect["effect_id"], manifest, owner=owner,
             claim_token=effect["claim_token"], clock=clock)
    return manifest


def restore_backup(source, destination):
    """Restore into a new isolated root and verify every database/artifact byte."""
    source, destination = Path(source), Path(destination)
    if os.path.lexists(destination):
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    manifest = json.loads(manifests[0].read_text())
    destination.mkdir(parents=True)
    database = destination / "ops.sqlite"
    database.write_bytes((source / (manifest["backup_key"] + ".sqlite")).read_bytes())
    probe = sqlite3.connect(database)
    try:
        if integrity_errors(probe) or file_hash(database) != manifest["database"]:
            raise fail("INTEGRITY_FAILED", "restored catalog failed verification")
    finally:
        probe.close()
    for name, info in manifest["artifacts"].items():
        path = destination / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / "artifacts" / name).read_bytes())
        if file_hash(path) != info["content_hash"]:
            raise fail("INTEGRITY_FAILED", "restored artifact failed verification")
    return destination
