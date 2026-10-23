"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
import tempfile
from contextlib import contextmanager
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

#: Prior manifests are advisory metadata; reads beyond this bound are invalid.
MAX_MANIFEST_BYTES = 1024 * 1024


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


@contextmanager
def _read_snapshot(conn):
    """One explicit deferred read transaction, so counts and a backup agree.

    ``BEGIN`` (not ``BEGIN IMMEDIATE``) only takes a read lock on first access,
    leaving writers free in WAL mode. Commit on success, roll back on anything
    else. Callers keep this open only around counting and ``backup_to``: never
    around artifact copying or manifest publication.
    """
    if conn.in_transaction:
        raise RuntimeError("nested catalog transaction: backup needs a clean connection")
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


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


def _read_manifest(manifest_path):
    """Parse a prior manifest without following non-regular files or reading
    past ``MAX_MANIFEST_BYTES``; any invalid prior is reported as ``None``."""
    try:
        fd = os.open(manifest_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        chunks = []
        remaining = MAX_MANIFEST_BYTES + 1
        while remaining > 0:
            chunk = os.read(fd, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) > MAX_MANIFEST_BYTES:
        return None
    try:
        return json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _verified_prior_backup(manifest_path, database, source_rows, key, payload):
    """The existing manifest when it and ``database`` are a verified prior
    result for this payload, else ``None``."""
    manifest = _read_manifest(manifest_path)
    if not isinstance(manifest, dict):
        return None
    if (manifest.get("backup_key") != key
            or manifest.get("schema_version") != payload["schema_version"]
            or manifest.get("cutoff") != payload["cutoff"]
            or manifest.get("database_rows") != source_rows):
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
    """Verify a staged copy's per-table row counts against the source.

    ``backup_to`` already owns the integrity/foreign-key check.
    """
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


@contextmanager
def _backup_lock(target, key):
    """Serialize overlapping attempts for one target/key while they inspect,
    replace and publish that key's database and manifest."""
    path = target / f".{key}.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _remove_stale_stages(target, key):
    """Remove crash-leftover staging databases for this key only."""
    prefix = f".{key}."
    suffix = ".staging.sqlite"
    for entry in target.iterdir():
        if entry.name.startswith(prefix) and entry.name.endswith(suffix):
            try:
                entry.unlink()
            except FileNotFoundError:
                pass


def _stage_copy(conn, target, key):
    """Write the source's current snapshot into a fresh staging database."""
    stage_fd, stage_name = tempfile.mkstemp(
        dir=str(target), prefix=f".{key}.", suffix=".staging.sqlite")
    os.close(stage_fd)
    os.unlink(stage_name)
    staged = Path(stage_name)
    try:
        backup_to(conn, staged)
    except BaseException:
        _discard_staged(staged)
        raise
    return staged


def _discard_staged(staged):
    try:
        staged.unlink()
    except FileNotFoundError:
        pass


def _install_stage(staged, target, database, source_rows):
    """Verify and atomically install a staged copy after the read snapshot ends."""
    try:
        _verify_staged(staged, source_rows)
        with staged.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(staged, database)
        staged = None
        fsync_directory(target)
    finally:
        if staged is not None:
            _discard_staged(staged)


def _copy_artifact(store, target, name, ref, fault):
    """Copy one artifact, verify its hash and run the per-artifact fault hook."""
    source = store.verify(ref)
    destination = target / "artifacts" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    if fault:
        fault(name)
    if file_hash(destination) != ref.content_hash:
        raise fail("INTEGRITY_FAILED", "backup artifact hash mismatch")
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
    fd, name = tempfile.mkstemp(
        dir=str(manifest_path.parent), prefix=f".{manifest_path.name}.",
        suffix=".pending")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
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
    manifest_path = target / (key + ".manifest.json")
    with _backup_lock(target, key):
        _remove_stale_stages(target, key)
        staged = None
        try:
            with _read_snapshot(conn):
                source_rows = _table_row_counts(conn)
                prior = _verified_prior_backup(manifest_path, database, source_rows, key, payload)
                if prior is None:
                    staged = _stage_copy(conn, target, key)
        except BaseException:
            if staged is not None:
                _discard_staged(staged)
            raise
        if staged is not None:
            _install_stage(staged, target, database, source_rows)
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
