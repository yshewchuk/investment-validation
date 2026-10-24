"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import resource
import sqlite3
import stat
import sys
import tempfile
import time
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
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.outbox import claim, complete, enqueue

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BACKUP_FILE_BYTES = 1 << 40
_CHUNK = 1 << 20


def _log(message):
    print(f"[backup] {message}", file=sys.stderr, flush=True)


def _rss_max_kib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


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


def _open_checked(path, *, stage, expected_size=None, oserror_code=None):
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


def _verify_staged(staged, source_rows):
    if staged.stat().st_size > MAX_BACKUP_FILE_BYTES:
        raise fail("INTEGRITY_FAILED", "staged catalog exceeds the backup size limit")
    probe = sqlite3.connect(staged)
    try:
        if integrity_errors(probe):
            raise fail("INTEGRITY_FAILED", "staged catalog integrity check failed")
        staged_rows = _table_row_counts(probe)
        if staged_rows != source_rows:
            differing = [name for name in sorted(set(source_rows) | set(staged_rows))
                         if source_rows.get(name) != staged_rows.get(name)]
            raise fail("INTEGRITY_FAILED", "backup row counts differ from source",
                       details={"tables": differing})
    finally:
        probe.close()


@contextmanager
def _backup_lock(target, key):
    fd = os.open(target / f".{key}.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _stage_copy(conn, target, key, *, keepalive=None):
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    total_bytes = page_size * conn.execute("PRAGMA page_count").fetchone()[0]
    if total_bytes > MAX_BACKUP_FILE_BYTES:
        raise fail("INTEGRITY_FAILED", "source catalog exceeds the backup size limit")
    fd, name = tempfile.mkstemp(dir=str(target), prefix=_stage_prefix(key),
                                suffix=".staging.sqlite")
    os.close(fd)
    os.unlink(name)
    staged = Path(name)
    started = last_log = last_keepalive = time.monotonic()
    _log(f"database-copy: start bytes={total_bytes} rss_max_kib={_rss_max_kib()}")

    def progress(_status, remaining, total):
        nonlocal last_log, last_keepalive
        now = time.monotonic()
        if keepalive and now - last_keepalive >= 30:
            keepalive()
            last_keepalive = now
        if now - last_log >= 60:
            total = total * page_size
            done = total - remaining * page_size
            elapsed = now - started
            rate = done / elapsed
            eta = (total - done) / rate if rate else 0
            _log(f"database-copy: {done}/{total} bytes rate={rate:.0f}B/s "
                 f"eta={eta:.0f}s rss_max_kib={_rss_max_kib()}")
            last_log = now

    try:
        backup_to(conn, staged, progress=progress)
        elapsed = time.monotonic() - started
        _log(f"database-copy: complete bytes={total_bytes} elapsed={elapsed:.1f}s "
             f"rss_max_kib={_rss_max_kib()}")
        keepalive and keepalive()
        return staged
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _install_stage(staged, target, key, source_rows, *, keepalive=None):
    try:
        _verify_staged(staged, source_rows)
        name = key + ".sqlite"
        size = staged.stat().st_size
        os.replace(staged, target / name)
        try:
            fsync_directory(target)
        except OSError:
            raise fail("WORKER_FAILED", "backup database publish failed") from None
        digest = verified_file_hash(target / name, expected_size=size,
                                    stage="installed backup database")
        return name, digest
    finally:
        staged.unlink(missing_ok=True)


def _copy_artifacts(refs, store, target, fault, *, keepalive=None):
    copied = {}
    for name, ref in refs.items():
        source = store.verify(ref)
        destination = target / "artifacts" / name
        ensure_directory(destination.parent)
        fd, pending = tempfile.mkstemp(dir=str(destination.parent),
                                      prefix=f".{name}.", suffix=".pending")
        temporary = Path(pending)
        started = last_log = last_keepalive = time.monotonic()
        digest = hashlib.sha256()
        total = 0
        _log(f"artifact {name}: start bytes={ref.byte_size} rss_max_kib={_rss_max_kib()}")
        try:
            with os.fdopen(fd, "wb") as out, source.open("rb") as src:
                while chunk := src.read(_CHUNK):
                    out.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
                    now = time.monotonic()
                    if keepalive and now - last_keepalive >= 30:
                        keepalive()
                        last_keepalive = now
                    if now - last_log >= 60:
                        elapsed = now - started
                        rate = total / elapsed
                        eta = (ref.byte_size - total) / rate if rate else 0
                        _log(f"artifact {name}: {total}/{ref.byte_size} bytes "
                             f"rate={rate:.0f}B/s eta={eta:.0f}s "
                             f"rss_max_kib={_rss_max_kib()}")
                        last_log = now
                out.flush()
                os.fsync(out.fileno())
            actual = "sha256:" + digest.hexdigest()
            if total != ref.byte_size or actual != ref.content_hash:
                raise fail("INTEGRITY_FAILED", f"artifact {name} verification failed")
            if fault:
                fault(name)
            os.replace(temporary, destination)
            fsync_directory(destination.parent)
            elapsed = time.monotonic() - started
            _log(f"artifact {name}: complete bytes={total} elapsed={elapsed:.1f}s "
                 f"rss_max_kib={_rss_max_kib()}")
            copied[name] = {"content_hash": actual, "byte_size": total}
        finally:
            temporary.unlink(missing_ok=True)
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
            temporary.unlink(missing_ok=True)


def verified_file_hash(path, *, expected_size, stage):
    """Hash a regular file without following links or reading beyond its cap."""
    stream, size = _open_checked(path, stage=stage, expected_size=expected_size,
                                 oserror_code="INTEGRITY_FAILED")
    try:
        digest = hashlib.sha256()
        copied = 0
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            if copied + len(chunk) > size:
                raise fail("INTEGRITY_FAILED", f"{stage} grew while reading")
            digest.update(chunk)
            copied += len(chunk)
        if copied != size:
            raise fail("INTEGRITY_FAILED", f"{stage} shrank while reading")
        return "sha256:" + digest.hexdigest()
    finally:
        stream.close()


def _verified_database(target, manifest, *, keepalive=None):
    if (type(manifest.get("database_size")) is not int
            or manifest["database_size"] < 0
            or not isinstance(manifest.get("database_rows"), dict)):
        return False
    database = Path(target) / (_single_filename(manifest["backup_key"]) + ".sqlite")
    if verified_file_hash(database, expected_size=manifest["database_size"],
                           stage="durable backup database") != manifest.get("database"):
        return False
    probe = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        return not integrity_errors(probe) and _table_row_counts(probe) == manifest["database_rows"]
    finally:
        probe.close()


def _verify_durable_manifest(target, key, manifest, *, keepalive=None):
    try:
        return (isinstance(manifest, dict) and manifest.get("backup_key") == key
                and isinstance(manifest.get("database"), str)
                and _verified_database(target, manifest, keepalive=keepalive))
    except Exception:
        return False


def _durable_manifest(manifest_path, key, payload, refs, *, keepalive=None):
    """Return a verified prior only when it also matches this effect payload."""
    if not manifest_path.is_file():
        return None
    from engine.v2.ops.workflows.commands import read_bounded_bytes

    try:
        manifest = json.loads(read_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES))
    except (OSError, ValueError, RecursionError):
        return None
    if (not isinstance(manifest, dict) or manifest.get("backup_key") != key
            or not _verify_durable_manifest(
                manifest_path.parent, key, manifest, keepalive=keepalive)):
        return None
    expected_artifacts = {
        name: {"content_hash": ref.content_hash, "byte_size": ref.byte_size}
        for name, ref in refs.items()
    }
    if (manifest.get("schema_version") != payload.get("schema_version")
            or manifest.get("cutoff") != payload.get("cutoff")
            or manifest.get("artifacts") != expected_artifacts):
        raise fail("INTEGRITY_FAILED", "backup manifest differs from requested effect")
    return manifest


def _delivered_or_stale(conn, key):
    row = conn.execute(
        "SELECT receipt_json FROM outbox WHERE kind='backup' AND logical_key=? "
        "AND state='delivered'", (key,)).fetchone()
    if row is None:
        raise fail("STALE_EXPECTATION", "backup effect is not pending")
    return json.loads(row["receipt_json"])


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=None):
    _validated_key(key)
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _delivered_or_stale(conn, key)
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
        durable = _durable_manifest(manifest_path, key, payload, refs,
                                     keepalive=keepalive)
        if durable is not None:
            _log(f"run_backup {key}: complete from durable manifest")
            complete(conn, effect["effect_id"], durable, owner=owner,
                     claim_token=effect["claim_token"], clock=clock)
            return durable
        _log(f"run_backup {key}: no verified reusable result; rebuilding")
        database_path = conn.execute("PRAGMA database_list").fetchone()[2]
        if not database_path:
            raise fail("WORKER_FAILED", "backup snapshot requires a file-backed catalog")
        snapshot_conn = sqlite3.connect(Path(database_path).resolve().as_uri() + "?mode=ro",
                                         uri=True)
        staged = None
        try:
            with _read_snapshot(snapshot_conn) as snapshot:
                page_size = snapshot.execute("PRAGMA page_size").fetchone()[0]
                page_count = snapshot.execute("PRAGMA page_count").fetchone()[0]
                if page_size * page_count > MAX_BACKUP_FILE_BYTES:
                    raise fail("INTEGRITY_FAILED", "source catalog exceeds the backup size limit")
                source_rows = _table_row_counts(snapshot)
                staged = _stage_copy(snapshot, target, key, keepalive=keepalive)
        except BaseException:
            if staged is not None:
                staged.unlink(missing_ok=True)
            raise
        finally:
            snapshot_conn.close()
        database_file, database_hash = _install_stage(
            staged, target, key, source_rows, keepalive=keepalive)
        copied = _copy_artifacts(refs, store, target, fault, keepalive=keepalive)
        manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                    "cutoff": payload["cutoff"], "database": database_hash,
                    "database_file": database_file,
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


def restore_backup(source, destination):
    """Restore into a new isolated root; refuse an existing destination."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    manifest = json.loads(manifests[0].read_text())
    destination.mkdir(parents=True)
    database = destination / "ops.sqlite"
    database_source = manifest_database_path(source, manifest)
    database.write_bytes(database_source.read_bytes())
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