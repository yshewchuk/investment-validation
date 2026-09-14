"""Immutable local releases with a fenced, monotone current pointer.

Remote targets require their own conditional-write adapter and are not enabled.
The local pointer can be reconciled after filesystem success but catalog failure.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path

from engine.v2.foundation import (
    content_hash,
    ensure_directory,
    fsync_directory,
    safe_relative_path,
)
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import file_hash
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.outbox import enqueue, watermark, watermark_would_conflict


def stage_release(conn, store, release_id, occurrence, files, *, expected_current, gates, clock,
                  claim=None):
    safe_relative_path(release_id)
    if "/" in release_id:
        raise fail("INVALID_REQUEST", "release identity must be one path segment")
    required = {"decision", "projection", "security", "engineering"}
    binding = content_hash({"release_id": release_id, "occurrence": occurrence, "files": files})
    eligible = all(_gate_is_bound(kind, gates.get(kind), binding, store) for kind in required)
    if eligible and claim is None:
        raise fail("LEASE_LOST", "eligible release mutation requires a fenced claim")
    manifest = {"schema_version": "release_manifest.v1.0", "release_id": release_id,
                "occurrence": occurrence, "files": files, "gates": gates}
    digest = content_hash(manifest)
    for name, ref in files.items():
        safe_relative_path(name)
        store.verify(ref)
    with transaction(conn):
        if eligible:
            verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        old = conn.execute("SELECT manifest_hash FROM releases WHERE release_id=?", (release_id,)).fetchone()
        if old and old[0] != digest:
            raise fail("IDEMPOTENCY_CONFLICT", "release manifest changed")
        if not old:
            conn.execute("INSERT INTO releases(release_id,occurrence,manifest_json,manifest_hash,"
                         "expected_current,eligible) VALUES (?,?,?,?,?,?)",
                         (release_id, occurrence, dumps(manifest), digest, expected_current, int(eligible)))
        if eligible:
            enqueue(conn, "publication", release_id, {"release_id": release_id, "manifest_hash": digest})
    return {"release_id": release_id, "eligible": eligible, "manifest_hash": digest}


def _gate_is_bound(kind, gate, binding, store):
    return (isinstance(gate, dict) and gate.get("ok") is True
            and isinstance(gate.get("receipt_ref"), str)
            and gate["receipt_ref"].startswith("sha256:")
            and isinstance(gate.get("input_hash"), str)
            and gate["input_hash"] == binding
            and _verify_gate_artifact(kind, gate, binding, store))


def _verify_gate_artifact(kind, gate, binding, store):
    try:
        from engine.v2.contracts import ArtifactRef
        from engine.v2.foundation import ArtifactError, from_document
        ref = from_document(ArtifactRef, gate["receipt_artifact"])
        if ref.content_hash != gate["receipt_ref"]:
            return False
        document = json.loads(store.read_verified(ref))
        return (document.get("kind") == kind and document.get("status") == "passed"
                and document.get("input_hash") == binding)
    except (ArtifactError, KeyError, TypeError, ValueError):
        return False


def materialize(store, target, release_id, files):
    target = Path(target)
    _safe_target(target)
    ensure_directory(target / "releases")
    final = target / "releases" / release_id
    if final.is_symlink():
        raise fail("INTEGRITY_FAILED", "release directory is a symlink")
    staging = target / "releases" / ("pending-" + uuid.uuid4().hex)
    ensure_directory(staging)
    for name, ref in files.items():
        safe_relative_path(name)
        dest = staging / name
        ensure_directory(dest.parent)
        shutil.copyfile(store.verify(ref), dest)
        with dest.open("rb") as stream:
            os.fsync(stream.fileno())
        dest.chmod(0o444)
    _verify_files(staging, files)
    for directory in sorted((p for p in staging.rglob("*") if p.is_dir()), reverse=True):
        fsync_directory(directory)
    fsync_directory(staging)
    if final.exists():
        _verify_files(final, files)
        shutil.rmtree(staging)
    else:
        staging.rename(final)
        fsync_directory(final.parent)
    return final


def _verify_files(directory, files):
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(files):
        raise fail("INTEGRITY_FAILED", "release file population differs")
    for name, ref in files.items():
        path = directory / name
        if path.is_symlink() or file_hash(path) != ref.content_hash:
            raise fail("INTEGRITY_FAILED", "release bytes differ")


def current(target):
    path = Path(target) / "CURRENT"
    if not path.exists():
        return None
    value = path.read_text().strip()
    if len(safe_relative_path(value)) != 1:
        raise fail("INTEGRITY_FAILED", "current pointer is unsafe")
    return value


def _safe_target(target):
    if target.exists() and target.is_symlink():
        raise fail("INTEGRITY_FAILED", "publication target is a symlink")
    if (target / "releases").exists() and (target / "releases").is_symlink():
        raise fail("INTEGRITY_FAILED", "release root is a symlink")


def publish_local(conn, claim, store, target, release_id, *, scope, clock, generation="",
                  fault=None):
    row = conn.execute("SELECT * FROM releases WHERE release_id=?", (release_id,)).fetchone()
    if row is None or not row["eligible"]:
        raise fail("PUBLICATION_REFUSED", "release gates are not all valid")
    from engine.v2.contracts import ArtifactRef
    from engine.v2.foundation import from_document
    manifest = json.loads(row["manifest_json"])
    files = {name: from_document(ArtifactRef, ref) for name, ref in manifest["files"].items()}
    materialize(store, target, release_id, files)
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        observed = current(target)
        if observed not in (row["expected_current"], release_id):
            raise fail("STALE_EXPECTATION", "current release changed")
        newest = conn.execute("SELECT MAX(occurrence) FROM releases WHERE published_at IS NOT NULL").fetchone()[0]
        if newest and newest > row["occurrence"]:
            raise fail("STALE_EXPECTATION", "an older release cannot replace a newer release")
        # guide §5.4/§5.5 item 1: every FORESEEABLE refusal -- watermark
        # conflict included -- must be known before CURRENT ever moves. This
        # is a pure precheck of the exact writes ``_acknowledge`` below will
        # make; it never itself writes. Without it, the old bug was that the
        # pointer swap ran first and ``_acknowledge``'s own watermark() call
        # could still raise afterwards, leaving CURRENT naming a release the
        # catalog never acknowledged. An actual crash between the swap and
        # the acknowledgement (the ``fault`` hook below) is a separate,
        # unavoidable window this precheck cannot close -- that is the
        # existing O24 recoverable-on-retry design, unchanged.
        for stage in ("publication", "delivery"):
            if watermark_would_conflict(conn, "nightly", scope, stage, row["occurrence"],
                                        release_id, generation=generation):
                raise fail("IDEMPOTENCY_CONFLICT",
                          "a different release already completed this occurrence")
        if observed != release_id:
            pointer = Path(target) / ("CURRENT." + uuid.uuid4().hex)
            pointer.write_text(release_id + "\n")
            with pointer.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(pointer, Path(target) / "CURRENT")
            fsync_directory(Path(target))
        if fault:
            fault("pointer_before_ack")
        _acknowledge(conn, row, scope, clock, generation=generation)
    return {"release_id": release_id, "delivered": current(target) == release_id}


def _acknowledge(conn, row, scope, clock, *, generation=""):
    from engine.v2.foundation import format_timestamp
    stamp = format_timestamp(clock.now())
    conn.execute("UPDATE releases SET published_at=?,delivered_at=? WHERE release_id=?",
                 (stamp, stamp, row["release_id"]))
    conn.execute("UPDATE outbox SET state='delivered',attempts=attempts+1,receipt_json=? "
                 "WHERE kind='publication' AND logical_key=?",
                 (dumps({"release_id": row["release_id"], "verified_at": stamp}), row["release_id"]))
    for stage in ("publication", "delivery"):
        watermark(conn, "nightly", scope, stage, row["occurrence"], row["release_id"], clock=clock,
                 generation=generation)
