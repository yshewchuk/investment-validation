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
from engine.v2.foundation.artifacts import read_bounded_bytes
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.outbox import claim, complete, enqueue

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BACKUP_FILE_BYTES = 1 << 40
_CHUNK = 1 << 20


def _log(message):
    print(f"[backup] {message}", file=sys.stderr, flush=True)


class _Progress:
    def __init__(self, stage, total=None, *, keepalive=None):
        self.stage, self.total, self.keepalive = stage, total, keepalive
        self.started = self.last_log = self.last_keepalive = time.monotonic()
        self.done = 0
        _log(f"{stage}: start total={total if total is not None else 'unknown'} "
             f"rss_max_kib={_rss_max_kib()}")

    def update(self, done):
        self.done = done
        now = time.monotonic()
        if self.keepalive and now - self.last_keepalive >= 30:
            self.keepalive()
            self.last_keepalive = now
        if now - self.last_log >= 60:
            elapsed = now - self.started
            rate = done / elapsed
            eta = ((self.total - done) / rate if rate and self.total is not None
                   else "unknown")
            _log(f"{self.stage}: {done}/{self.total if self.total is not None else 'unknown'} "
                 f"units rate={rate:.0f}/s eta={eta if eta == 'unknown' else f'{eta:.0f}s'} "
                 f"rss_max_kib={_rss_max_kib()}")
            self.last_log = now

    def pulse(self):
        """Count one SQLite callback; SQLite calls every 10,000 VM instructions."""
        self.update(self.done + 1)
        return 0

    def finish(self, outcome="complete"):
        elapsed = time.monotonic() - self.started
        _log(f"{self.stage}: {outcome} done={self.done} elapsed={elapsed:.1f}s "
             f"rss_max_kib={_rss_max_kib()}")


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
        _single_filename(key)
    except ValueError:
        raise fail("INVALID_REQUEST", "unsafe backup key") from None
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


def _table_row_counts(conn, *, keepalive=None):
    tables = [row[0] for row in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]
    progress = _Progress("database row counts", keepalive=keepalive)
    conn.set_progress_handler(progress.pulse, 10000)
    try:
        counts = {name: conn.execute(
            'SELECT COUNT(1) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
            for name in tables}
    except BaseException:
        progress.finish("failed")
        raise
    else:
        progress.finish()
        return counts
    finally:
        conn.set_progress_handler(None, 0)


def _verify_staged(staged, source_rows, *, keepalive=None):
    if staged.stat().st_size > MAX_BACKUP_FILE_BYTES:
        raise fail("INTEGRITY_FAILED", "staged catalog exceeds the backup size limit")
    probe = sqlite3.connect(staged)
    try:
        progress = _Progress("staged database integrity", keepalive=keepalive)
        probe.set_progress_handler(progress.pulse, 10000)
        try:
            if integrity_errors(probe):
                raise fail("INTEGRITY_FAILED", "staged catalog integrity check failed")
        except BaseException:
            progress.finish("failed")
            raise
        else:
            progress.finish()
        finally:
            probe.set_progress_handler(None, 0)
        staged_rows = _table_row_counts(probe, keepalive=keepalive)
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
        _verify_staged(staged, source_rows, keepalive=keepalive)
        try:
            fd = os.open(staged, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            raise fail("WORKER_FAILED", "backup database stage sync failed") from None
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
        source_stream, _ = _open_checked(
            source, stage=f"artifact {name}", expected_size=ref.byte_size,
            oserror_code="INTEGRITY_FAILED")
        try:
            fd, pending = tempfile.mkstemp(dir=str(destination.parent),
                                          prefix=f".{name}.", suffix=".pending")
        except BaseException:
            source_stream.close()
            raise
        temporary = Path(pending)
        started = last_log = last_keepalive = time.monotonic()
        digest = hashlib.sha256()
        total = 0
        _log(f"artifact {name}: start bytes={ref.byte_size} rss_max_kib={_rss_max_kib()}")
        try:
            with os.fdopen(fd, "wb") as out, source_stream as src:
                while chunk := src.read(_CHUNK):
                    if total + len(chunk) > ref.byte_size:
                        raise fail("INTEGRITY_FAILED", f"artifact {name} grew while reading")
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
            source_stream.close()
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


def verified_file_hash(path, *, expected_size, stage, keepalive=None):
    """Hash a regular file without following links or reading beyond its cap."""
    stream, size = _open_checked(path, stage=stage, expected_size=expected_size,
                                 oserror_code="INTEGRITY_FAILED")
    progress = _Progress(stage, size, keepalive=keepalive)
    try:
        digest = hashlib.sha256()
        copied = 0
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            if copied + len(chunk) > size:
                raise fail("INTEGRITY_FAILED", f"{stage} grew while reading")
            digest.update(chunk)
            copied += len(chunk)
            progress.update(copied)
        if copied != size:
            raise fail("INTEGRITY_FAILED", f"{stage} shrank while reading")
        result = "sha256:" + digest.hexdigest()
    except BaseException:
        progress.finish("failed")
        raise
    else:
        progress.finish()
        return result
    finally:
        stream.close()


def _copy_verified_file(source, destination, *, expected_size, expected_hash, stage):
    stream, size = _open_checked(source, stage=stage, expected_size=expected_size,
                                 oserror_code="INTEGRITY_FAILED")
    progress = _Progress(stage, size)
    digest = hashlib.sha256()
    copied = 0
    try:
        with stream, Path(destination).open("wb") as out:
            for chunk in iter(lambda: stream.read(_CHUNK), b""):
                copied += len(chunk)
                if copied > size:
                    raise fail("INTEGRITY_FAILED", f"{stage} grew while reading")
                digest.update(chunk)
                out.write(chunk)
                progress.update(copied)
            out.flush()
            os.fsync(out.fileno())
        actual = "sha256:" + digest.hexdigest()
        if copied != size or actual != expected_hash:
            raise fail("INTEGRITY_FAILED", f"{stage} size or hash mismatch")
    except BaseException:
        progress.finish("failed")
        raise
    else:
        progress.finish()
    finally:
        stream.close()


def _verified_database(target, manifest, *, keepalive=None):
    if (type(manifest.get("database_size")) is not int
            or manifest["database_size"] < 0
            or not isinstance(manifest.get("database_rows"), dict)):
        return False
    database = manifest_database_path(target, manifest)
    if verified_file_hash(database, expected_size=manifest["database_size"],
                           stage="durable backup database", keepalive=keepalive) != manifest.get("database"):
        return False
    probe = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
    try:
        progress = _Progress("durable database integrity", keepalive=keepalive)
        probe.set_progress_handler(progress.pulse, 10000)
        try:
            errors = integrity_errors(probe)
        except BaseException:
            progress.finish("failed")
            raise
        else:
            progress.finish()
        finally:
            probe.set_progress_handler(None, 0)
        return not errors and _table_row_counts(
            probe, keepalive=keepalive) == manifest["database_rows"]
    finally:
        probe.close()


def _verify_durable_manifest(target, key, manifest, *, keepalive=None):
    try:
        if (not isinstance(manifest, dict) or manifest.get("backup_key") != key
                or not isinstance(manifest.get("database"), str)
                or not _verified_database(target, manifest, keepalive=keepalive)):
            return False
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            return False
        for name, info in artifacts.items():
            _single_filename(name)
            if (not isinstance(info, dict) or type(info.get("byte_size")) is not int
                    or info["byte_size"] < 0):
                return False
            path = Path(target) / "artifacts" / name
            if verified_file_hash(path, expected_size=info["byte_size"],
                                  stage=f"durable backup artifact {name}",
                                  keepalive=keepalive) != info.get("content_hash"):
                return False
        return True
    except Exception:
        return False


def _durable_manifest(manifest_path, key, payload, refs, *, keepalive=None):
    """Return a verified prior only when it also matches this effect payload."""
    if not manifest_path.is_file():
        return None

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
        "SELECT receipt_json, payload_json FROM outbox WHERE kind='backup' AND logical_key=? "
        "AND state='delivered'", (key,)).fetchone()
    if row is None:
        raise fail("STALE_EXPECTATION", "backup effect is not pending")
    return json.loads(row["receipt_json"]), json.loads(row["payload_json"])


def _verified_delivered_backup_receipt(conn, key, target, *, keepalive=None):
    receipt, payload = _delivered_or_stale(conn, key)
    refs = _validated_artifact_names(payload)
    target = Path(target)
    if target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    if refs and (target / "artifacts").is_symlink():
        raise fail("INTEGRITY_FAILED", "delivered backup artifacts directory is a symlink")
    durable = _durable_manifest(target / (key + ".manifest.json"), key,
                                payload, refs, keepalive=keepalive)
    if durable is None or durable != receipt:
        raise fail("INTEGRITY_FAILED", "delivered backup files failed verification")
    return receipt


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=None):
    _validated_key(key)
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        return _verified_delivered_backup_receipt(
            conn, key, target, keepalive=keepalive)
    payload = json.loads(effect["payload_json"])
    refs = _validated_artifact_names(payload)
    target = Path(target)
    if target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    ensure_directory(target)
    artifact_root = target / "artifacts"
    if refs and artifact_root.is_symlink():
        raise fail("INVALID_REQUEST", "backup artifacts directory is a symlink")
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
                source_rows = _table_row_counts(snapshot, keepalive=keepalive)
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


def _validate_restore_artifacts(source, artifacts):
    for name, info in artifacts.items():
        _single_filename(name)
        if (not isinstance(info, dict) or not isinstance(info.get("content_hash"), str)
                or ("byte_size" in info
                    and (type(info["byte_size"]) is not int or info["byte_size"] < 0))):
            raise ValueError("invalid artifact metadata")
    artifact_root = source / "artifacts"
    if artifacts and (artifact_root.is_symlink() or not artifact_root.is_dir()):
        raise ValueError("invalid artifact directory")


def _read_restore_manifest(source, manifest_path):
    try:
        manifest = json.loads(read_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES))
        if not isinstance(manifest, dict):
            raise ValueError("manifest must be an object")
        database_source = manifest_database_path(source, manifest)
        database_hash = manifest.get("database")
        database_size = manifest.get("database_size")
        artifacts = manifest.get("artifacts", {})
        if (not isinstance(database_hash, str)
                or (database_size is not None
                    and (type(database_size) is not int or database_size < 0))
                or not isinstance(artifacts, dict)):
            raise ValueError("invalid manifest fields")
        _validate_restore_artifacts(source, artifacts)
        return database_source, database_hash, database_size, artifacts
    except (OSError, ValueError, TypeError, AttributeError, KeyError, RecursionError):
        raise fail("INTEGRITY_FAILED", "backup manifest is malformed or unsafe") from None


def _restore_contents(source, destination, database_source, database_hash,
                      database_size, artifacts):
    try:
        database = destination / "ops.sqlite"
        _copy_verified_file(database_source, database, expected_size=database_size,
                            expected_hash=database_hash, stage="backup database")
        try:
            probe = sqlite3.connect(database)
            progress = _Progress("restored database integrity")
            probe.set_progress_handler(progress.pulse, 10000)
            try:
                try:
                    errors = integrity_errors(probe)
                except BaseException:
                    progress.finish("failed")
                    raise
                else:
                    progress.finish()
                if errors:
                    raise fail("INTEGRITY_FAILED", "restored catalog failed integrity check")
            finally:
                probe.set_progress_handler(None, 0)
                probe.close()
        except sqlite3.DatabaseError:
            raise fail("INTEGRITY_FAILED", "restored catalog failed integrity check") from None
        for name, info in artifacts.items():
            path = destination / "artifacts" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            _copy_verified_file(source / "artifacts" / name, path,
                                expected_size=info.get("byte_size"),
                                expected_hash=info["content_hash"],
                                stage=f"backup artifact {name}")
    except BaseException:
        import shutil
        shutil.rmtree(destination, ignore_errors=True)
        raise


def restore_backup(source, destination):
    """Restore into a new isolated root; refuse an existing destination."""
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_symlink():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    database_source, database_hash, database_size, artifacts = _read_restore_manifest(
        source, manifests[0])
    try:
        destination.mkdir(parents=True)
    except FileExistsError:
        raise fail("INVALID_REQUEST", "restore destination already exists") from None
    _restore_contents(source, destination, database_source, database_hash,
                      database_size, artifacts)
    return destination