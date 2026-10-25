"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import ctypes
import errno
import fcntl
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
from engine.v2.ops.outbox import claim, complete, enqueue

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BACKUP_FILE_BYTES = 1 << 40
_COPY_CHUNK = 1024 * 1024


def _with_backup_lock(target, key, operation):
    lock_path = target / f".{key}.backup.lock"
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup publication lock could not be opened") from exc
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return operation()
    except OSError as exc:
        raise fail("WORKER_FAILED", "backup publication lock failed") from exc
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _open_checked_file(path, *, expected_size, stage, max_bytes=MAX_BACKUP_FILE_BYTES):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.ELOOP, errno.ENOTDIR):
            raise fail("INTEGRITY_FAILED", f"{stage} is missing or unsafe") from exc
        raise fail("WORKER_FAILED", f"{stage} could not be opened") from exc
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes
                or (expected_size is not None
                    and (type(expected_size) is not int or expected_size != info.st_size))):
            raise fail("INTEGRITY_FAILED", f"{stage} is not a bounded regular file")
        stream = os.fdopen(fd, "rb")
        fd = -1
        return stream, info.st_size
    finally:
        if fd >= 0:
            os.close(fd)


def _read_bounded_bytes(path, max_bytes, *, stage):
    stream, size = _open_checked_file(
        path, expected_size=None, stage=stage, max_bytes=max_bytes)
    try:
        data = stream.read(max_bytes + 1)
    except OSError as exc:
        raise fail("WORKER_FAILED", f"{stage} could not be read") from exc
    finally:
        stream.close()
    if len(data) != size or len(data) > max_bytes:
        raise fail("INTEGRITY_FAILED", f"{stage} size changed while reading")
    return data


def _checked_file_hash(path, *, stage, expected_size=None, keepalive=None):
    stream, size = _open_checked_file(
        path, expected_size=expected_size, stage=stage,
        max_bytes=MAX_BACKUP_FILE_BYTES)
    digest = hashlib.sha256()
    copied = 0
    with stream:
        while copied <= size:
            if keepalive is not None:
                keepalive()
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


def _copy_verified_file(source, destination, *, expected_size, expected_hash, stage,
                        keepalive=None):
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
                if keepalive is not None:
                    keepalive()
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
    _validate_backup_artifact_names(artifacts)
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


def _table_row_counts(conn, keepalive=None):
    """Deterministic row count per non-internal table, keyed and ordered by name."""
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    counts = {}
    for name in tables:
        if keepalive is not None:
            keepalive()
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
    if not isinstance(artifacts, dict):
        raise fail("INVALID_REQUEST", "backup artifacts must be a mapping")
    for name, reference in artifacts.items():
        try:
            if len(safe_relative_path(name)) != 1:
                raise ValueError("unsafe backup artifact name")
        except (ArtifactError, ValueError) as exc:
            raise fail("INVALID_REQUEST", "unsafe backup artifact name") from exc
        if not isinstance(reference, ArtifactRef):
            try:
                from_document(ArtifactRef, reference)
            except (ArtifactError, KeyError, TypeError, ValueError) as exc:
                raise fail("INVALID_REQUEST", "backup artifact reference is invalid") from exc


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


def _valid_sha256(value):
    return (isinstance(value, str) and len(value) == 71
            and value.startswith("sha256:")
            and all(character in "0123456789abcdef" for character in value[7:]))


def _validate_manifest_artifact(name, info, *, require_size):
    try:
        safe_name = isinstance(name, str) and len(safe_relative_path(name)) == 1
    except (ArtifactError, ValueError):
        safe_name = False
    if not safe_name or not isinstance(info, dict):
        raise fail("INTEGRITY_FAILED", "backup manifest artifact entry is invalid")
    size = info.get("byte_size")
    if (not _valid_sha256(info.get("content_hash"))
            or (size is None and require_size)
            or (size is not None and (type(size) is not int or size < 0))):
        raise fail("INTEGRITY_FAILED", "backup manifest artifact entry is invalid")


def _validate_manifest_artifacts(artifacts, *, require_sizes=False):
    if not isinstance(artifacts, dict):
        raise fail("INTEGRITY_FAILED", "backup manifest artifacts are invalid")
    for name, info in artifacts.items():
        _validate_manifest_artifact(name, info, require_size=require_sizes)


def _validate_manifest_identity(path, manifest, expected_key, receipt):
    key = manifest.get("backup_key")
    try:
        safe_key = isinstance(key, str) and len(safe_relative_path(key)) == 1
    except (ArtifactError, ValueError):
        safe_key = False
    if (not safe_key or path.name != f"{key}.manifest.json"
            or (expected_key is not None and key != expected_key)
            or (receipt is not None and manifest != receipt)):
        raise fail("INTEGRITY_FAILED", "backup manifest identity is invalid")


def _validate_manifest_rows(rows):
    if rows is None:
        return
    if (not isinstance(rows, dict)
            or any(not isinstance(name, str) or type(count) is not int or count < 0
                   for name, count in rows.items())):
        raise fail("INTEGRITY_FAILED", "backup manifest row counts are invalid")


def _read_backup_manifest(path, *, stage, expected_key=None, receipt=None):
    try:
        manifest = json.loads(_read_bounded_bytes(
            path, MAX_MANIFEST_BYTES, stage=stage))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from exc
    if not isinstance(manifest, dict):
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid")
    _validate_manifest_identity(path, manifest, expected_key, receipt)
    rows = manifest.get("database_rows")
    _validate_manifest_rows(rows)
    _validate_manifest_artifacts(manifest.get("artifacts"), require_sizes=rows is not None)
    if not _valid_sha256(manifest.get("database")):
        raise fail("INTEGRITY_FAILED", "backup manifest database hash is invalid")
    return manifest


def _verify_delivered_backup(conn, key, target, keepalive=None):
    receipt = _delivered_or_stale(conn, "backup", key)
    target = Path(target)
    manifest_path = target / f"{key}.manifest.json"
    manifest = _read_backup_manifest(
        manifest_path, stage="delivered backup manifest", expected_key=key,
        receipt=receipt)
    database_hash = manifest["database"]
    artifacts = manifest["artifacts"]
    database_rows = manifest.get("database_rows")

    database = target / f"{key}.sqlite"
    with tempfile.TemporaryDirectory(dir=target) as verify_dir:
        checked_database = Path(verify_dir) / "database.sqlite"
        _copy_verified_file(
            database, checked_database, expected_size=None,
            expected_hash=database_hash, stage="delivered backup database",
            keepalive=keepalive)
        try:
            probe = sqlite3.connect(checked_database)
            try:
                callback_errors = []

                def renew_lease():
                    if keepalive is None:
                        return 0
                    try:
                        keepalive()
                    except Exception as exc:
                        callback_errors.append(exc)
                        return 1
                    return 0

                probe.set_progress_handler(renew_lease, 10000)
                try:
                    problems = integrity_errors(probe)
                    actual_rows = (_table_row_counts(probe, keepalive=keepalive)
                                   if database_rows is not None else database_rows)
                except sqlite3.Error:
                    if callback_errors:
                        raise callback_errors[0]
                    raise
                finally:
                    probe.set_progress_handler(None, 0)
                if problems or (database_rows is not None and actual_rows != database_rows):
                    raise fail("INTEGRITY_FAILED", "delivered backup database is invalid")
            finally:
                probe.close()
        except sqlite3.Error as exc:
            raise fail("INTEGRITY_FAILED", "delivered backup database is invalid") from exc

    for name, info in artifacts.items():
        actual_hash = _checked_file_hash(
            target / "artifacts" / name, stage="delivered backup artifact",
            expected_size=info.get("byte_size"), keepalive=keepalive)
        if actual_hash != info.get("content_hash"):
            raise fail("INTEGRITY_FAILED", "delivered backup artifact hash differs")
    return receipt


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=_no_keepalive):
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _verify_delivered_backup(conn, key, target, keepalive=keepalive)
    payload = json.loads(effect["payload_json"])
    _validate_backup_artifact_names(payload.get("artifacts", {}))
    target = Path(target)
    if os.path.lexists(target) and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    target.mkdir(parents=True, exist_ok=True)

    def publish_attempt():
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

    return _with_backup_lock(target, key, publish_attempt)


def _rename_directory_noreplace(source, destination):
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except (AttributeError, OSError) as exc:
        raise fail("WORKER_FAILED", "atomic no-overwrite restore is unavailable") from exc
    renameat2.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                          ctypes.c_char_p, ctypes.c_uint)
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise fail("INVALID_REQUEST", "restore destination already exists")
    raise fail("WORKER_FAILED", "restore destination could not be published")


def restore_backup(source, destination):
    """Restore verified backup inputs into a new root without overwriting."""
    source, destination = Path(source), Path(destination)
    if os.path.lexists(destination):
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    manifest = _read_backup_manifest(manifests[0], stage="backup manifest")
    key = manifest["backup_key"]
    artifacts = manifest["artifacts"]
    database_hash = manifest["database"]
    database_rows = manifest.get("database_rows")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise fail("WORKER_FAILED", "restore destination parent could not be created") from exc

    with tempfile.TemporaryDirectory(
            prefix=f".{destination.name}.restore-", dir=destination.parent) as staging:
        staged_root = Path(staging)
        database = staged_root / "ops.sqlite"
        _copy_verified_file(
            source / f"{key}.sqlite", database, expected_size=None,
            expected_hash=database_hash, stage="restore database")
        try:
            probe = sqlite3.connect(database)
            try:
                if (integrity_errors(probe)
                        or (database_rows is not None
                            and _table_row_counts(probe) != database_rows)):
                    raise fail("INTEGRITY_FAILED", "restored catalog failed verification")
            finally:
                probe.close()
        except sqlite3.Error as exc:
            raise fail("INTEGRITY_FAILED", "restored catalog is invalid") from exc

        for name, info in artifacts.items():
            path = staged_root / "artifacts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _copy_verified_file(
                source / "artifacts" / name, path, expected_size=info.get("byte_size"),
                expected_hash=info["content_hash"], stage="restore artifact")
        _rename_directory_noreplace(staged_root, destination)
    return destination
