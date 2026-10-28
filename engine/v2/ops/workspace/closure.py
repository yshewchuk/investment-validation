"""Read-only verification of one snapshot's complete required closure.

An experiment workspace reads a pinned snapshot straight from the production catalog and store
(an overlay, never a copy), so everything that snapshot needs must still be there, unchanged,
when the workspace opens and again before it seals. ``verify_snapshot_closure`` re-derives every
identity from catalog rows (``Repository.resolve_full``) and then checks what that resolution
does not prove: dataset-version parents, the parent chain, the committed import receipt and its
lineage, coverage and capture rows, and every object file. It only reads: catalog queries run on
the caller's connection (a read-only one for the workspace) and the store is only ``lstat``-ed
(``deep=True`` also re-hashes each object through ``objects.verify_object_path``).

Failure semantics: every defect raises ``SNAPSHOT_CLOSURE_BROKEN`` whose ``details["reason"]``
names it; nothing is repaired, skipped or retried. Raw receipts and normalizations behind
``daily_market`` revisions are outside this closure: a snapshot reader scans fragments only.
Progress goes to stderr at a bounded cadence; the result is a value, never stdout.
"""
from __future__ import annotations

import os
import stat
import sys
import time
from dataclasses import dataclass

from engine.v2.data import errors, objects
from engine.v2.data.repository import Repository
from engine.v2.foundation import CONTENT_HASH_PREFIX, ArtifactStore, content_hash

__all__ = ["SNAPSHOT_CLOSURE_SCHEMA", "SnapshotClosure", "verify_snapshot_closure"]

SNAPSHOT_CLOSURE_SCHEMA = "snapshot_closure.v1.0"
_MAX_CHAIN = 100_000
_PROGRESS_SECONDS = 30.0


@dataclass(frozen=True)
class SnapshotClosure:
    """``identity_digest`` covers catalog facts only (stable across process restarts);
    ``stat_digest`` also covers each object file's inode and mtime (same-process seal check)."""

    snapshot_id: str
    manifest_hash: str
    parent_chain: tuple[str, ...]
    commit_receipt_ids: tuple[str, ...]
    lineage_receipt_ids: tuple[str, ...]
    object_count: int
    object_bytes: int
    identity_digest: str
    stat_digest: str


def _broken(reason: str, **details):
    return errors.fail("SNAPSHOT_CLOSURE_BROKEN", "snapshot closure is incomplete or changed",
                       details={"reason": reason, **details})


def _digest(value) -> str:
    return content_hash(value)


def _log(message: str) -> None:
    print("[closure] " + message, file=sys.stderr, flush=True)


def _dataset_parents(conn, resolved) -> None:
    for name, manifest in sorted(resolved.table_manifests.items()):
        parent = manifest.parent_dataset_version_id
        if parent is not None and conn.execute(
                "SELECT 1 FROM data_dataset_versions WHERE dataset_version_id = ?",
                (parent,)).fetchone() is None:
            raise _broken("dataset_parent_missing", table=name, dataset_version_id=parent)


def _parent_chain(conn, snapshot) -> tuple[str, ...]:
    chain, seen, current = [], {snapshot.snapshot_id}, snapshot.parent_snapshot_id
    while current is not None:
        if current in seen or len(chain) >= _MAX_CHAIN:
            raise _broken("parent_chain_invalid", snapshot_id=current)
        row = conn.execute("SELECT parent_snapshot_id FROM data_snapshots WHERE snapshot_id = ?",
                           (current,)).fetchone()
        if row is None:
            raise _broken("parent_missing", snapshot_id=current)
        seen.add(current)
        chain.append(current)
        current = row["parent_snapshot_id"]
    return tuple(chain)


def _committed(conn, receipt_id: str) -> bool:
    row = conn.execute("SELECT status FROM data_import_receipts WHERE receipt_id = ?",
                       (receipt_id,)).fetchone()
    return row is not None and row["status"] == "committed"


def _commit_receipts(conn, snapshot_id: str) -> tuple[str, ...]:
    rows = conn.execute(
        "SELECT receipt_id FROM data_import_receipts WHERE result_snapshot_id = ? "
        "AND status = 'committed' ORDER BY receipt_id", (snapshot_id,)).fetchall()
    if not rows:
        raise _broken("commit_receipt_missing", snapshot_id=snapshot_id)
    return tuple(row["receipt_id"] for row in rows)


def _lineage(conn, receipt_ids) -> tuple[str, ...]:
    found: set[str] = set()
    for start in receipt_ids:
        hops, current = {start}, start
        while True:
            row = conn.execute("SELECT base_receipt_id FROM data_receipt_lineage "
                               "WHERE receipt_id = ?", (current,)).fetchone()
            if row is None:
                break
            current = row["base_receipt_id"]
            if current in hops or len(hops) > _MAX_CHAIN:
                raise _broken("lineage_cycle", receipt_id=current)
            if not _committed(conn, current):
                raise _broken("lineage_receipt_missing", receipt_id=current)
            hops.add(current)
            found.add(current)
    return tuple(sorted(found))


def _coverage_rows(conn, snapshot_id: str) -> list:
    rows = conn.execute(
        "SELECT * FROM data_snapshot_coverage WHERE snapshot_id = ? "
        "ORDER BY table_name, coverage_id",
        (snapshot_id,)).fetchall()
    for receipt_id in sorted({row["import_receipt_id"] for row in rows}):
        if not _committed(conn, receipt_id):
            raise _broken("coverage_receipt_missing", receipt_id=receipt_id)
    return [list(row) for row in rows]


def _capture_rows(conn, receipt_ids) -> list:
    rows: list = []
    for receipt_id in receipt_ids:
        rows += [list(row) for row in conn.execute(
            "SELECT * FROM data_price_captures "
            "WHERE receipt_id = ? ORDER BY capture_id", (receipt_id,))]
    return rows


def _receipt_facts(conn, receipt_ids) -> list:
    return [list(conn.execute("SELECT * FROM data_import_receipts WHERE receipt_id = ?",
                              (receipt_id,)).fetchone()) for receipt_id in sorted(receipt_ids)]


def _object_stats(store: ArtifactStore, refs, deep: bool) -> tuple[list, int]:
    ordered = sorted(refs, key=lambda ref: ref.object_id)
    items, total = [], 0
    started = last = time.monotonic()
    for index, ref in enumerate(ordered, 1):
        digest = ref.content_hash.removeprefix(CONTENT_HASH_PREFIX)
        try:
            info = os.lstat(store.root / "objects" / digest[:2] / digest)
        except FileNotFoundError:
            raise _broken("object_missing", object_id=ref.object_id) from None
        except OSError:
            raise _broken("object_unreadable", object_id=ref.object_id) from None
        if not stat.S_ISREG(info.st_mode) or info.st_size != ref.byte_size:
            raise _broken("object_changed", object_id=ref.object_id)
        if deep:
            try:
                objects.verify_object_path(store, ref)
            except errors.DataError as exc:
                raise _broken("object_changed", object_id=ref.object_id, cause=exc.code) from exc
        items.append([ref.object_id, info.st_size, info.st_ino, info.st_mtime_ns])
        total += info.st_size
        if time.monotonic() - last >= _PROGRESS_SECONDS:
            last = time.monotonic()
            _log(f"{index}/{len(ordered)} objects, {total} bytes, {last - started:.0f}s elapsed")
    return items, total


def verify_snapshot_closure(conn, store: ArtifactStore, snapshot_id: str, manifest_hash: str,
                            *, deep: bool = False) -> SnapshotClosure:
    """Verify ``snapshot_id``'s whole closure against ``manifest_hash``; read-only."""
    started = time.monotonic()
    _log(f"stage=verify start snapshot={snapshot_id} deep={deep}")
    try:
        resolved = Repository(conn).resolve_full(snapshot_id)
    except errors.DataError as exc:
        raise _broken("snapshot_unresolved", snapshot_id=snapshot_id, cause=exc.code) from exc
    snapshot = resolved.snapshot
    if snapshot.manifest_hash != manifest_hash:
        raise _broken("manifest_mismatch", snapshot_id=snapshot_id)
    _dataset_parents(conn, resolved)
    chain = _parent_chain(conn, snapshot)
    commits = _commit_receipts(conn, snapshot_id)
    lineage = _lineage(conn, commits)
    coverage = _coverage_rows(conn, snapshot_id)
    captures = _capture_rows(conn, commits)
    items, total = _object_stats(store, resolved.objects, deep)
    identity = _digest({
        "schema": SNAPSHOT_CLOSURE_SCHEMA, "snapshot_id": snapshot_id,
        "manifest_hash": manifest_hash, "parent_chain": list(chain),
        "commit_receipts": list(commits), "lineage": list(lineage), "coverage": coverage,
        "receipts": _receipt_facts(conn, (*commits, *lineage)),
        "captures": captures, "records": len(resolved.records),
        "objects": [[ref.object_id, ref.content_hash, ref.byte_size]
                    for ref in sorted(resolved.objects, key=lambda r: r.object_id)]})
    _log(f"stage=verify done objects={len(items)} bytes={total} "
         f"elapsed={time.monotonic() - started:.1f}s")
    return SnapshotClosure(
        snapshot_id=snapshot_id, manifest_hash=manifest_hash, parent_chain=chain,
        commit_receipt_ids=commits, lineage_receipt_ids=lineage, object_count=len(items),
        object_bytes=total, identity_digest=identity, stat_digest=_digest(items))
