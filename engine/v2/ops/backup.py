"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

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
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.outbox import claim, complete, enqueue

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BACKUP_FILE_BYTES = 1 << 40
_COPY_CHUNK = 1024 * 1024


def _read_bounded_bytes(path, max_bytes):
    """Read a capped regular file without following links or blocking on FIFOs."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise ValueError("file cannot be read safely") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("file is not a bounded regular file")
        data = bytearray()
        while len(data) <= max_bytes:
            chunk = os.read(fd, min(_COPY_CHUNK, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > max_bytes:
            raise ValueError("file grew beyond its byte limit")
        return bytes(data)
    finally:
        os.close(fd)


def _open_checked_file(path, *, expected_size, stage):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise fail("INTEGRITY_FAILED", f"{stage} is missing or unsafe") from exc
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
                    raise fail("INTEGRITY_FAILED", f"{stage} could not be read") from exc
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


def _verified_prior_backup(manifest_path, database, source_rows, key, payload):
    """The existing manifest when it and ``database`` are a verified prior
    result for this payload, else ``None``.

    A retry after the manifest was published but before the acknowledgement
    rebuilds a byte-different SQLite copy from the same source. Reusing this
    verified pair keeps such a retry byte-idempotent instead of refusing the
    rebuilt database's different hash.
    """
    try:
        manifest = json.loads(_read_bounded_bytes(manifest_path, MAX_MANIFEST_BYTES))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if (manifest.get("backup_key") != key
            or manifest.get("schema_version") != payload["schema_version"]
            or manifest.get("cutoff") != payload["cutoff"]):
        return None
    prior_rows = manifest.get("database_rows")
    if prior_rows is not None and prior_rows != source_rows:
        raise fail("INTEGRITY_FAILED", "backup source rows changed after manifest publication")
    if prior_rows is None:
        return None
    expected_hash = manifest.get("database")
    if not isinstance(expected_hash, str) or not database.exists():
        return None
    try:
        probe = sqlite3.connect(database)
        try:
            if integrity_errors(probe) or _table_row_counts(probe) != source_rows:
                return None
        finally:
            probe.close()
        if file_hash(database) != expected_hash:
            return None
    except (OSError, sqlite3.Error):
        return None
    return manifest


def _verify_staged(staged, source_rows):
    """Verify a staged copy against the source row counts and integrity."""
    probe = sqlite3.connect(staged)
    try:
        staged_rows = _table_row_counts(probe)
        problems = integrity_errors(probe)
    finally:
        probe.close()
    if staged_rows != source_rows:
        differing = [name for name in sorted(set(source_rows) | set(staged_rows))
                     if source_rows.get(name) != staged_rows.get(name)]
        raise fail("INTEGRITY_FAILED", "backup row counts differ from source",
                   details={"tables": differing})
    if problems:
        raise fail("INTEGRITY_FAILED", "backup integrity check failed",
                   details={"first_problem": problems[0]})


def _stage_database(conn, target, key, database, source_rows):
    """Build, verify and atomically install a fresh staged database copy."""
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


def _copy_artifacts(payload, store, target, fault, keepalive):
    """Copy every payload artifact into the target and return their manifest entries."""
    copied = {}
    for name, document in payload.get("artifacts", {}).items():
        keepalive()
        ref = from_document(ArtifactRef, document)
        if "/" in name or name in ("", ".", ".."):
            raise fail("INVALID_REQUEST", "unsafe backup artifact name")
        copied[name] = _copy_artifact(store, target, name, ref, fault)
    return copied


def _publish_manifest(manifest_path, manifest, prior):
    """Match a prior manifest, or atomically write a new one with a directory fsync."""
    if prior is not None:
        if prior != manifest:
            raise fail("INTEGRITY_FAILED", "backup manifest differs")
        return
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
    target = Path(target)
    if target.exists() and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    target.mkdir(parents=True, exist_ok=True)
    database = target / (key + ".sqlite")
    source_rows = _table_row_counts(conn)
    manifest_path = target / (key + ".manifest.json")
    prior = _verified_prior_backup(manifest_path, database, source_rows, key, payload)
    if prior is None:
        _stage_database(conn, target, key, database, source_rows)
    copied = _copy_artifacts(payload, store, target, fault, keepalive)
    manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                "cutoff": payload["cutoff"], "database": file_hash(database),
                "database_rows": source_rows, "artifacts": copied}
    _publish_manifest(manifest_path, manifest, prior)
    if fault:
        fault("after_manifest_before_ack")
    complete(conn, effect["effect_id"], manifest, owner=owner,
             claim_token=effect["claim_token"], clock=clock)
    return manifest


def restore_backup(source, destination):
    """Restore into a new isolated root and verify every database/artifact byte."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    try:
        manifest = json.loads(_read_bounded_bytes(manifests[0], MAX_MANIFEST_BYTES))
        backup_key = manifest["backup_key"]
        if not isinstance(backup_key, str) or len(safe_relative_path(backup_key)) != 1:
            raise ValueError("unsafe backup key")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from exc
    destination.mkdir(parents=True)
    database = destination / "ops.sqlite"
    _copy_verified_file(source / (backup_key + ".sqlite"), database,
                        expected_size=None, expected_hash=manifest["database"],
                        stage="restore database")
    probe = sqlite3.connect(database)
    try:
        if integrity_errors(probe):
            raise fail("INTEGRITY_FAILED", "restored catalog failed integrity check")
    finally:
        probe.close()
    for name, info in manifest["artifacts"].items():
        if not isinstance(name, str) or len(safe_relative_path(name)) != 1:
            raise fail("INTEGRITY_FAILED", "backup artifact name is unsafe")
        path = destination / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _copy_verified_file(source / "artifacts" / name, path,
                            expected_size=info.get("byte_size"),
                            expected_hash=info["content_hash"], stage="restore artifact")
    return destination
