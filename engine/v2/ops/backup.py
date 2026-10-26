"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import (
    format_timestamp,
    from_document,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.ops.catalog import backup_to, integrity_errors, transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.outbox import claim, complete, enqueue
from engine.v2.ops.workflows.commands import read_bounded_json

_MAX_MANIFEST_BYTES = 1024 * 1024
MAX_BACKUP_BYTES = 64 * 1024**3
BACKUP_RETENTION = 3


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


def _verified_database_hash(database, *, check_integrity=True):
    """Return the hash after read-only quick_check and optional integrity checks."""
    try:
        size = database.stat().st_size
        if size > MAX_BACKUP_BYTES:
            raise fail("BACKUP_FAILED", "backup database exceeds size limit")
        if size == 0:
            raise fail("BACKUP_FAILED", "backup database is empty")
        probe = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            page_count = probe.execute("PRAGMA page_count").fetchone()[0]
            page_size = probe.execute("PRAGMA page_size").fetchone()[0]
            if size != page_count * page_size:
                raise fail("BACKUP_FAILED", "backup database size differs from its page count")
            if probe.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
                raise fail("BACKUP_FAILED", "backup database verification failed")
            if check_integrity and integrity_errors(probe):
                raise fail("BACKUP_FAILED", "backup database verification failed")
        finally:
            probe.close()
        with database.open("rb+") as stream:
            os.fsync(stream.fileno())
        return file_hash(database)
    except (OSError, sqlite3.Error) as error:
        raise fail("BACKUP_FAILED", "backup database verification failed") from error


def _verify_delivered_copy(database, receipt):
    """Verify a delivered backup's database copy against its stored receipt."""
    try:
        size = receipt["database_byte_size"]
        expected_hash = receipt["database"]
    except (KeyError, TypeError) as error:
        raise fail("BACKUP_FAILED", "backup receipt is invalid") from error
    if (not isinstance(size, int) or isinstance(size, bool) or size <= 0
            or not isinstance(expected_hash, str) or not expected_hash):
        raise fail("BACKUP_FAILED", "backup receipt is invalid")
    try:
        actual_size = database.stat().st_size
    except OSError as error:
        raise fail("BACKUP_FAILED", "backup database copy is missing or inaccessible") from error
    if actual_size != size:
        raise fail("BACKUP_FAILED", "backup database copy size mismatch")
    if _verified_database_hash(database) != expected_hash:
        raise fail("BACKUP_FAILED", "backup database copy hash mismatch")


def _copy_verified_database(conn, database):
    try:
        is_new = not database.exists()
        if is_new:
            backup_to(conn, database)
        return _verified_database_hash(database, check_integrity=not is_new)
    except (OpsError, OSError, sqlite3.Error) as error:
        message = (error.problem.message if isinstance(error, OpsError)
                   else "backup database copy or verification failed")
        try:
            database.unlink(missing_ok=True)
        except OSError as unlink_error:
            raise fail("BACKUP_FAILED", f"{message}; backup database cleanup failed") from unlink_error
        raise fail("BACKUP_FAILED", message) from error


def _verify_existing_manifest(manifest_path, manifest):
    if not manifest_path.exists():
        return
    try:
        with manifest_path.open("rb") as stream:
            captured = stream.read(_MAX_MANIFEST_BYTES + 1)
    except OSError as error:
        raise fail("BACKUP_FAILED", "backup manifest could not be read") from error
    if len(captured) > _MAX_MANIFEST_BYTES:
        raise fail("BACKUP_FAILED", "backup manifest exceeds 1 MiB")
    try:
        existing = json.loads(captured)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from error
    if existing != manifest:
        raise fail("INTEGRITY_FAILED", "backup manifest differs")


def _prune_backups(target, current_key):
    """Keep only the newest ``BACKUP_RETENTION`` validated backup pairs."""
    key_pattern = re.compile(r"bkp[0-9a-f]{24}")
    candidates = []
    for database in target.glob("bkp*.sqlite"):
        key = database.name[: -len(".sqlite")]
        manifest_path = target / (key + ".manifest.json")
        if key_pattern.fullmatch(key) is None or not manifest_path.exists():
            continue
        try:
            manifest = read_bounded_json(manifest_path, _MAX_MANIFEST_BYTES)
            if (not isinstance(manifest, dict) or manifest.get("backup_key") != key
                    or not isinstance(manifest.get("cutoff"), str)):
                continue
            _verify_delivered_copy(database, manifest)
            candidates.append((manifest["cutoff"], key, database, manifest_path))
        except (OSError, ValueError, TypeError, RecursionError, OpsError):
            continue
    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    survivors = 0
    for _cutoff, key, database, manifest_path in candidates:
        if key == current_key:
            continue
        if survivors < BACKUP_RETENTION - 1:
            survivors += 1
            continue
        for path in (database, manifest_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                print("[backup-retention] BACKUP_FAILED: backup pair deletion failed",
                      file=sys.stderr)


def _copy_backup_artifact(store, target, name, document, fault):
    ref = from_document(ArtifactRef, document)
    if "/" in name or name in ("", ".", ".."):
        raise fail("INVALID_REQUEST", "unsafe backup artifact name")
    artifact_path = store.root.joinpath(*store._object_parts(ref))
    try:
        artifact_size = artifact_path.stat().st_size
    except OSError as error:
        raise fail("BACKUP_FAILED", "backup artifact is missing or inaccessible") from error
    if artifact_size > MAX_BACKUP_BYTES:
        raise fail("BACKUP_FAILED", "backup artifact exceeds size limit")
    source = store.verify(ref)
    if source.stat().st_size != ref.byte_size:
        raise fail("INTEGRITY_FAILED", "backup artifact size mismatch")
    destination = target / "artifacts" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if destination.stat().st_size != ref.byte_size:
        raise fail("INTEGRITY_FAILED", "backup artifact size mismatch")
    if fault:
        fault(name)
    if file_hash(destination) != ref.content_hash:
        raise fail("INTEGRITY_FAILED", "backup artifact hash mismatch")
    return {"content_hash": ref.content_hash, "byte_size": ref.byte_size}


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=_no_keepalive):
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        receipt = _delivered_or_stale(conn, "backup", key)
        _verify_delivered_copy(Path(target) / (key + ".sqlite"), receipt)
        return receipt
    payload = json.loads(effect["payload_json"])
    target = Path(target)
    if target.exists() and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    target.mkdir(parents=True, exist_ok=True)
    database = target / (key + ".sqlite")
    database_hash = _copy_verified_database(conn, database)
    copied = {}
    for name, document in payload.get("artifacts", {}).items():
        keepalive()
        copied[name] = _copy_backup_artifact(store, target, name, document, fault)
    manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                "cutoff": payload["cutoff"], "database": database_hash,
                "database_byte_size": database.stat().st_size,
                "artifacts": copied}
    manifest_path = target / (key + ".manifest.json")
    _verify_existing_manifest(manifest_path, manifest)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise fail("BACKUP_FAILED", "backup manifest exceeds 1 MiB")
    temporary = manifest_path.with_suffix(".pending")
    temporary.write_bytes(encoded)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    fsync_directory(manifest_path.parent)
    if fault:
        fault("after_manifest_before_ack")
    complete(conn, effect["effect_id"], manifest, owner=owner,
             claim_token=effect["claim_token"], clock=clock)
    _prune_backups(target, key)
    return manifest


def _read_restore_manifest(source):
    manifests = list(source.glob("*.manifest.json"))
    if len(manifests) != 1:
        raise fail("INTEGRITY_FAILED", "backup manifest is missing or ambiguous")
    try:
        manifest = read_bounded_json(manifests[0], _MAX_MANIFEST_BYTES)
    except (OSError, ValueError, RecursionError) as error:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from error
    if (not isinstance(manifest, dict)
            or not isinstance(manifest.get("backup_key"), str)
            or not isinstance(manifest.get("database"), str)
            or not isinstance(manifest.get("artifacts"), dict)):
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid")
    return manifest


def _restore_catalog(source, destination, manifest):
    database_source = source / (manifest["backup_key"] + ".sqlite")
    try:
        actual_size = database_source.stat().st_size
    except OSError as error:
        raise fail("INTEGRITY_FAILED", "backup manifest is invalid") from error
    if actual_size > MAX_BACKUP_BYTES:
        raise fail("INTEGRITY_FAILED", "restored catalog exceeds size limit")
    expected_size = manifest.get("database_byte_size", actual_size)
    if (not isinstance(expected_size, int) or isinstance(expected_size, bool)
            or expected_size <= 0 or expected_size != actual_size):
        raise fail("INTEGRITY_FAILED", "restored catalog size mismatch")
    destination.mkdir(parents=True)
    database = destination / "ops.sqlite"
    shutil.copyfile(database_source, database)
    if database.stat().st_size != expected_size:
        raise fail("INTEGRITY_FAILED", "restored catalog size mismatch")
    probe = sqlite3.connect(database)
    try:
        if integrity_errors(probe) or file_hash(database) != manifest["database"]:
            raise fail("INTEGRITY_FAILED", "restored catalog failed verification")
    finally:
        probe.close()


def restore_backup(source, destination):
    """Restore into a new isolated root and verify every database/artifact byte."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise fail("INVALID_REQUEST", "restore destination already exists")
    manifest = _read_restore_manifest(source)
    _restore_catalog(source, destination, manifest)
    for name, info in manifest["artifacts"].items():
        path = destination / "artifacts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact_source = source / "artifacts" / name
        artifact_size = artifact_source.stat().st_size
        if artifact_size > MAX_BACKUP_BYTES:
            raise fail("INTEGRITY_FAILED", "restored artifact exceeds size limit")
        if artifact_size != info["byte_size"]:
            raise fail("INTEGRITY_FAILED", "restored artifact size mismatch")
        shutil.copyfile(artifact_source, path)
        if path.stat().st_size != info["byte_size"]:
            raise fail("INTEGRITY_FAILED", "restored artifact size mismatch")
        if file_hash(path) != info["content_hash"]:
            raise fail("INTEGRITY_FAILED", "restored artifact failed verification")
    return destination
