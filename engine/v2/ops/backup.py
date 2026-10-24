"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    ArtifactError,
    ensure_directory,
    format_timestamp,
    from_document,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.outbox import claim, complete, enqueue

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BACKUP_FILE_BYTES = 1 << 40
_CHUNK = 1 << 20


def _log(message):
    print(f"[backup] {message}", file=sys.stderr, flush=True)


def _single_filename(value):
    if not isinstance(value, str):
        raise ValueError("not a safe single filename")
    try:
        parts = safe_relative_path(value)
    except ArtifactError:
        raise ValueError("not a safe single filename") from None
    if len(parts) != 1:
        raise ValueError("not a safe single filename")
    return parts[0]


def _validated_key(key):
    try:
        parts = safe_relative_path(key)
    except ArtifactError:
        raise fail("INVALID_REQUEST", "unsafe backup key") from None
    if len(parts) != 1:
        raise fail("INVALID_REQUEST", "unsafe backup key")
    return key


def manifest_database_path(root, manifest):
    name = manifest.get("database_file")
    if name is None:
        name = _single_filename(manifest.get("backup_key")) + ".sqlite"
    return Path(root) / _single_filename(name)


def _stage_prefix(key):
    return f".bkp-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}."


def _full_logical_digest(conn, *, stage=None):
    digest = hashlib.sha256()
    for line in conn.iterdump():
        digest.update(line.encode("utf-8"))
    return "sha256:" + digest.hexdigest()


def _open_checked(path, *, stage, expected_size=None, oserror_code=None):
    """Open a regular non-symlink file within the byte cap; refuse otherwise."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        if oserror_code is not None:
            raise fail(oserror_code, f"{stage} is missing or unreadable") from None
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise fail("INTEGRITY_FAILED", f"{stage} is not a regular file")
        if info.st_size > MAX_BACKUP_FILE_BYTES:
            raise fail("INTEGRITY_FAILED", f"{stage} exceeds the file size limit")
        if expected_size is not None and info.st_size != expected_size:
            raise fail("INTEGRITY_FAILED", f"{stage} size mismatch")
        return os.fdopen(fd, "rb"), info.st_size
    except BaseException:
        os.close(fd)
        raise


def _stream_copy(source, destination, *, expected_hash=None, expected_size=None,
                 stage, fault=None, oserror_code=None):
    stream, size = _open_checked(source, stage=stage, expected_size=expected_size,
                                 oserror_code=oserror_code)
    try:
        try:
            ensure_directory(destination.parent)
            fd, temporary_name = tempfile.mkstemp(
                dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".pending")
        except OSError:
            raise fail("WORKER_FAILED", f"{stage} destination is unavailable") from None
        temporary = Path(temporary_name)
        try:
            digest = hashlib.sha256()
            copied = 0
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = stream.read(_CHUNK)
                    if not chunk:
                        break
                    if copied + len(chunk) > size:
                        raise fail("INTEGRITY_FAILED", f"{stage} grew while copying")
                    try:
                        out.write(chunk)
                    except OSError:
                        raise fail("WORKER_FAILED", f"{stage} destination write failed") from None
                    digest.update(chunk)
                    copied += len(chunk)
                try:
                    out.flush()
                    os.fsync(out.fileno())
                except OSError:
                    raise fail("WORKER_FAILED", f"{stage} destination flush failed") from None
            if copied != size:
                raise fail("INTEGRITY_FAILED", f"{stage} shrank while copying")
            actual = "sha256:" + digest.hexdigest()
            if expected_hash is not None and actual != expected_hash:
                raise fail("INTEGRITY_FAILED", f"{stage} hash mismatch")
            if fault is not None:
                fault(destination.name)
            try:
                os.replace(temporary, destination)
                temporary = None
                fsync_directory(destination.parent)
            except OSError:
                raise fail("WORKER_FAILED", f"{stage} destination publish failed") from None
            return copied, actual
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
    finally:
        stream.close()


def prepare_backup(conn, key, artifacts, *, clock):
    _validated_key(key)
    with transaction(conn):
        existing = conn.execute(
            "SELECT effect_id FROM outbox WHERE kind='backup' AND logical_key=?",
            (key,)).fetchone()
        if existing is not None:
            return existing["effect_id"]
        payload = {"backup_key": key, "artifacts": artifacts,
                   "cutoff": format_timestamp(clock.now()),
                   "schema_version": "catalog_backup.v1.0"}
        return enqueue(conn, "backup", key, payload)


@contextmanager
def _read_snapshot(conn):
    """One deferred read transaction, so the source counts, digest and page
    copy all describe the same snapshot."""
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _table_row_counts(conn):
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    return {name: conn.execute(
        'SELECT COUNT(1) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
        for name in tables}


def _verify_staged(staged, source_rows, source_digest):
    probe = sqlite3.connect(staged)
    try:
        staged_rows = _table_row_counts(probe)
        if staged_rows != source_rows:
            differing = [name for name in sorted(set(source_rows) | set(staged_rows))
                         if source_rows.get(name) != staged_rows.get(name)]
            raise fail("INTEGRITY_FAILED", "backup row counts differ from source",
                       details={"tables": differing})
        staged_digest = _full_logical_digest(probe)
    finally:
        probe.close()
    if staged_digest != source_digest:
        raise fail("INTEGRITY_FAILED", "backup content digest differs from source",
                   details={"reason": "content_digest"})


@contextmanager
def _backup_lock(target, key):
    fd = os.open(target / f".{key}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _remove_stale_stages(target, key):
    """Remove crash-leftover staging databases for this key only."""
    prefix = _stage_prefix(key)
    for entry in target.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(".staging.sqlite"):
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


def _stage_copy(conn, target, key):
    """Write the source's current snapshot into a fresh staging database."""
    stage_fd, stage_name = tempfile.mkstemp(
        dir=str(target), prefix=_stage_prefix(key), suffix=".staging.sqlite")
    os.close(stage_fd)
    os.unlink(stage_name)
    staged = Path(stage_name)
    try:
        backup_to(conn, staged)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _install_stage(staged, target, key, source_rows, source_digest):
    try:
        _verify_staged(staged, source_rows, source_digest)
        for _ in range(10):
            name = f"{key}.{uuid.uuid4().hex}.sqlite"
            if not (target / name).exists():
                break
        else:
            raise fail("WORKER_FAILED", "could not allocate a unique backup name")
        _copied, digest = _stream_copy(staged, target / name, stage="database-install")
        return name, digest
    finally:
        staged.unlink(missing_ok=True)


def _copy_artifacts(refs, store, target, fault):
    copied = {}
    for name, ref in refs.items():
        try:
            source = store.root.joinpath(*safe_relative_path(ref.storage_key))
            _stream_copy(source, target / "artifacts" / name,
                         expected_hash=ref.content_hash, expected_size=ref.byte_size,
                         stage=f"artifact {name}", fault=fault)
        except ArtifactError:
            raise fail("INTEGRITY_FAILED", "backup artifact storage key is unsafe") from None
        except OSError:
            raise fail("INTEGRITY_FAILED",
                       "backup artifact is missing or unreadable") from None
        copied[name] = {"content_hash": ref.content_hash, "byte_size": ref.byte_size}
    return copied


def _validated_artifact_names(payload):
    artifacts = payload.get("artifacts", {})
    try:
        refs = {}
        for name, document in artifacts.items():
            _single_filename(name)
            refs[name] = from_document(ArtifactRef, document)
        return refs
    except (AttributeError, ArtifactError, KeyError, TypeError, ValueError):
        raise fail("INVALID_REQUEST", "invalid backup artifact metadata") from None


def _publish_manifest(manifest_path, manifest):
    """Atomically write a manifest with a directory fsync."""
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ensure_directory(manifest_path.parent)
    fd, name = tempfile.mkstemp(
        dir=str(manifest_path.parent), prefix=f".{manifest_path.name}.", suffix=".pending")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
        temporary = None
        fsync_directory(manifest_path.parent)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _verified_file_hash(path, *, expected_size, stage):
    stream, _ = _open_checked(path, stage=stage, expected_size=expected_size,
                              oserror_code="INTEGRITY_FAILED")
    try:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
        return "sha256:" + digest.hexdigest()
    finally:
        stream.close()


def _verified_database(target, manifest):
    if (type(manifest.get("database_size")) is not int
            or manifest["database_size"] < 0
            or not isinstance(manifest.get("database_rows"), dict)
            or not isinstance(manifest.get("database_content_hash"), str)):
        return False
    database = Path(target) / _single_filename(manifest["database_file"])
    if _verified_file_hash(database, expected_size=manifest["database_size"],
                           stage="durable backup database") != manifest.get("database"):
        return False
    probe = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        return (not integrity_errors(probe)
                and _table_row_counts(probe) == manifest["database_rows"]
                and _full_logical_digest(probe) == manifest["database_content_hash"])
    finally:
        probe.close()


def _verified_artifacts(target, artifacts):
    if not isinstance(artifacts, dict):
        return False
    for name, info in artifacts.items():
        _single_filename(name)
        if (not isinstance(info, dict) or not isinstance(info.get("content_hash"), str)
                or type(info.get("byte_size")) is not int or info["byte_size"] < 0):
            return False
        path = Path(target) / "artifacts" / name
        if _verified_file_hash(path, expected_size=info["byte_size"],
                               stage=f"durable backup artifact {name}") != info["content_hash"]:
            return False
    return True


def _verify_durable_manifest(target, key, manifest):
    try:
        return (isinstance(manifest, dict) and manifest.get("backup_key") == key
                and isinstance(manifest.get("database"), str)
                and _verified_database(target, manifest)
                and _verified_artifacts(target, manifest.get("artifacts")))
    except Exception:
        return False


def _durable_manifest(manifest_path, key):
    """The manifest a crashed attempt already published for ``key``, or
    ``None``. It is the backup's own durable evidence -- trusted at exactly the
    same level as the delivered ``receipt_json`` replay -- so a retry
    acknowledges it unchanged instead of writing a second backup whose fresh
    bytes would orphan the evidence a crash left behind."""
    if not manifest_path.is_file():
        return None
    from engine.v2.ops.workflows.commands import read_bounded_bytes

    try:
        manifest = json.loads(read_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES))
    except (OSError, ValueError, RecursionError):
        return None
    if (isinstance(manifest, dict) and manifest.get("backup_key") == key
            and _verify_durable_manifest(manifest_path.parent, key, manifest)):
        return manifest
    return None


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=None):
    _validated_key(key)
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        row = conn.execute(
            "SELECT receipt_json FROM outbox WHERE kind='backup' AND logical_key=? "
            "AND state='delivered'", (key,)).fetchone()
        if row is None:
            raise fail("STALE_EXPECTATION", "backup effect is not pending")
        return json.loads(row["receipt_json"])
    payload = json.loads(effect["payload_json"])
    refs = _validated_artifact_names(payload)
    target = Path(target)
    if target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    ensure_directory(target)
    manifest_path = target / (key + ".manifest.json")
    started = time.monotonic()
    _log(f"run_backup {key}: start")
    keepalive and keepalive()
    with _backup_lock(target, key):
        _remove_stale_stages(target, key)
        durable = _durable_manifest(manifest_path, key)
        if durable is not None:
            _log(f"run_backup {key}: complete from durable manifest")
            complete(conn, effect["effect_id"], durable, owner=owner,
                     claim_token=effect["claim_token"], clock=clock)
            return durable
        staged = None
        try:
            with _read_snapshot(conn) as snapshot:
                source_rows = _table_row_counts(snapshot)
                source_digest = _full_logical_digest(snapshot)
                staged = _stage_copy(snapshot, target, key)
        except BaseException:
            if staged is not None:
                staged.unlink(missing_ok=True)
            raise
        database_file, database_hash = _install_stage(
            staged, target, key, source_rows, source_digest)
        copied = _copy_artifacts(refs, store, target, fault)
        manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                    "cutoff": payload["cutoff"], "database": database_hash,
                    "database_file": database_file,
                    "database_content_hash": source_digest,
                    "database_rows": source_rows,
                    "database_size": (target / database_file).stat().st_size,
                    "artifacts": copied}
        _publish_manifest(manifest_path, manifest)
    keepalive and keepalive()
    if fault:
        fault("after_manifest_before_ack")
    _log(f"run_backup {key}: complete elapsed={time.monotonic() - started:.1f}s")
    complete(conn, effect["effect_id"], manifest, owner=owner,
             claim_token=effect["claim_token"], clock=clock)
    return manifest


def _restore_metadata(source, manifest):
    database_path = manifest_database_path(source, manifest)
    artifacts = _restore_artifacts(manifest.get("artifacts", {}))
    database_size = manifest.get("database_size")
    if database_size is not None and (type(database_size) is not int or database_size < 0):
        raise ValueError("invalid database size")
    database_hash = manifest["database"]
    if not isinstance(database_hash, str):
        raise ValueError("invalid database hash")
    return database_path, database_hash, database_size, artifacts


def _restore_artifacts(artifacts):
    if not isinstance(artifacts, dict):
        raise ValueError("invalid artifact metadata")
    for name, info in artifacts.items():
        _single_filename(name)
        if not isinstance(info, dict) or not isinstance(info.get("content_hash"), str):
            raise ValueError("invalid artifact metadata")
        size = info.get("byte_size")
        if size is not None and (type(size) is not int or size < 0):
            raise ValueError("invalid artifact size")
    return artifacts


def restore_backup(source, destination):
    """Restore into a new isolated root and verify every database/artifact byte."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    from engine.v2.ops.workflows.commands import read_bounded_bytes

    try:
        manifest = json.loads(read_bounded_bytes(manifests[0], MAX_MANIFEST_BYTES))
    except (OSError, ValueError, RecursionError) as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is unreadable") from exc
    try:
        database_path, database_hash, database_size, artifacts = _restore_metadata(
            source, manifest)
        database_name = database_path.name
    except (ArtifactError, AttributeError, KeyError, TypeError, ValueError) as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from exc
    started = time.monotonic()
    _log(f"restore_backup: start database={database_name}")
    ensure_directory(destination)
    database = destination / "ops.sqlite"
    _stream_copy(database_path, database, expected_hash=database_hash,
                 expected_size=database_size, stage="restore database",
                 oserror_code="INTEGRITY_FAILED")
    probe = sqlite3.connect(database)
    try:
        if integrity_errors(probe):
            raise fail("INTEGRITY_FAILED", "restored catalog failed verification")
    finally:
        probe.close()
    for name, info in artifacts.items():
        expected = info.get("byte_size")
        _stream_copy(source / "artifacts" / name, destination / "artifacts" / name,
                     expected_hash=info["content_hash"],
                     expected_size=expected if isinstance(expected, int) else None,
                     stage=f"restore artifact {name}",
                     oserror_code="INTEGRITY_FAILED")
    _log(f"restore_backup: complete elapsed={time.monotonic() - started:.1f}s")
    return destination