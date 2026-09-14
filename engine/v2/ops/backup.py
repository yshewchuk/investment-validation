"""Durable local backup effect; remote mirroring is deliberately out of scope."""
from __future__ import annotations

import json
import os
import sqlite3
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


def run_backup(conn, *, key, owner, target, clock, store, fault=None, keepalive=_no_keepalive):
    effect = claim(conn, "backup", owner=owner, clock=clock, logical_key=key)
    if effect is None:
        raise fail("STALE_EXPECTATION", "backup effect is not pending")
    payload = json.loads(effect["payload_json"])
    target = Path(target)
    if target.exists() and target.is_symlink():
        raise fail("INVALID_REQUEST", "backup target is a symlink")
    target.mkdir(parents=True, exist_ok=True)
    database = target / (key + ".sqlite")
    if not database.exists():
        backup_to(conn, database)
    probe = sqlite3.connect(database)
    try:
        if integrity_errors(probe):
            raise fail("INTEGRITY_FAILED", "backup integrity check failed")
    finally:
        probe.close()
    copied = {}
    for name, document in payload.get("artifacts", {}).items():
        keepalive()
        ref = from_document(ArtifactRef, document)
        if "/" in name or name in ("", ".", ".."):
            raise fail("INVALID_REQUEST", "unsafe backup artifact name")
        source = store.verify(ref)
        destination = target / "artifacts" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        if fault:
            fault(name)
        if file_hash(destination) != ref.content_hash:
            raise fail("INTEGRITY_FAILED", "backup artifact hash mismatch")
        copied[name] = {"content_hash": ref.content_hash, "byte_size": ref.byte_size}
    manifest = {"schema_version": payload["schema_version"], "backup_key": key,
                "cutoff": payload["cutoff"], "database": file_hash(database),
                "artifacts": copied}
    manifest_path = target / (key + ".manifest.json")
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise fail("INTEGRITY_FAILED", "backup manifest differs")
    temporary = manifest_path.with_suffix(".pending")
    temporary.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, manifest_path)
    fsync_directory(manifest_path.parent)
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
