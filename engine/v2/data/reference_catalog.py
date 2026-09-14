"""Per-import-receipt reference inputs in the data catalog (schema v5).

Guide §14: legacy snapshot metadata and model/registry artifacts are
separately pinned compatibility inputs, not part of snapshot identity. An
import that commits (or reuses) a snapshot records the exact reference files
it pinned in ``data_import_reference_inputs``, keyed by its own receipt. The
same snapshot can therefore carry different reference refs on different
receipts: identical data with a retrained model reuses the snapshot and still
records the new artifact.

This module is legacy-free and never imports ``reference_inputs`` (which
reaches legacy code through ``legacy_adapter``), so ``catalog.commit_snapshot``
can insert rows inside its own transaction without loading a legacy module.

* :func:`insert_reference_inputs` runs inside ``commit_snapshot``'s
  transaction, right after the receipt row.
* :func:`reference_inputs_for_snapshot` returns the rows of the most recent
  committed receipt in ``scope`` whose resulting snapshot is ``snapshot_id``.
  It refuses with ``SNAPSHOT_NOT_READY`` when that receipt is missing or
  recorded no reference inputs.
* :func:`pinned_materialization_refs` turns those rows into the three pinned
  inputs a ``LegacyMaterializationRequest`` needs.
"""
from __future__ import annotations

import dataclasses
import sqlite3
from collections.abc import Sequence

from engine.v2.contracts import ObjectRef

from . import errors
from .legacy_materialization import format_pinned_ref

__all__ = [
    "CALENDAR_KIND",
    "LEGACY_SNAPSHOT_KIND",
    "REFERENCE_KINDS",
    "REFERENCE_OBJECT_KIND",
    "ReferenceInput",
    "insert_reference_inputs",
    "pinned_materialization_refs",
    "reference_inputs_for_snapshot",
]

CALENDAR_KIND = "calendar"
LEGACY_SNAPSHOT_KIND = "legacy_snapshot"
#: Every kind ``reference_inputs.LEGACY_REFERENCE_INPUTS_V1`` declares.
REFERENCE_KINDS: tuple[str, ...] = (
    CALENDAR_KIND, "model_registry", "structure_champions", "chooser_analog_pool",
    LEGACY_SNAPSHOT_KIND, "champion_artifact", "tier4_serving_cache",
)
#: ``objects.publish_legacy_file`` labels every object it publishes this way,
#: whatever the file format; a ref rebuilt from a row must carry the same kind.
REFERENCE_OBJECT_KIND = "parquet_fragment"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReferenceInput:
    """One pinned reference file: its legacy path relative to ``engine.paths.ROOT``."""

    kind: str
    legacy_path: str
    object_id: str
    content_hash: str
    byte_size: int

    def object_ref(self) -> ObjectRef:
        return ObjectRef(kind=REFERENCE_OBJECT_KIND, object_id=self.object_id,
                         content_hash=self.content_hash, byte_size=self.byte_size)


def insert_reference_inputs(conn: sqlite3.Connection, receipt_id: str,
                            inputs: Sequence[ReferenceInput]) -> None:
    """Insert one row per input. The caller holds the receipt's own transaction."""
    for item in inputs:
        if item.kind not in REFERENCE_KINDS:
            raise errors.fail("CONTRACT_MISMATCH", "unknown reference input kind",
                              details={"kind": item.kind, "path": item.legacy_path})
        conn.execute(
            "INSERT INTO data_import_reference_inputs (receipt_id, legacy_path, kind, object_id, "
            "content_hash, byte_size) VALUES (?, ?, ?, ?, ?, ?)",
            (receipt_id, item.legacy_path, item.kind, item.object_id, item.content_hash,
             item.byte_size))


def reference_inputs_for_snapshot(conn: sqlite3.Connection, *, scope: str,
                                  snapshot_id: str) -> tuple[ReferenceInput, ...]:
    """The reference inputs of the newest committed receipt for ``snapshot_id`` in ``scope``."""
    receipt = conn.execute(
        "SELECT receipt_id FROM data_import_receipts WHERE scope = ? AND status = 'committed' "
        "AND result_snapshot_id = ? ORDER BY registered_at DESC, rowid DESC LIMIT 1",
        (scope, snapshot_id)).fetchone()
    if receipt is None:
        raise errors.fail("SNAPSHOT_NOT_READY", "no committed import receipt for this snapshot",
                          details={"scope": scope, "snapshot_id": snapshot_id})
    rows = conn.execute(
        "SELECT kind, legacy_path, object_id, content_hash, byte_size FROM "
        "data_import_reference_inputs WHERE receipt_id = ? ORDER BY legacy_path",
        (receipt[0],)).fetchall()
    if not rows:
        raise errors.fail("SNAPSHOT_NOT_READY", "the snapshot's newest import receipt pinned no "
                          "reference inputs", details={"receipt_id": receipt[0]})
    return tuple(ReferenceInput(kind=r[0], legacy_path=r[1], object_id=r[2], content_hash=r[3],
                                byte_size=r[4]) for r in rows)


def pinned_materialization_refs(inputs: Sequence[ReferenceInput]) -> dict:
    """``{"legacy_snapshot_object_ref", "registry_and_model_refs", "calendar_refs"}``."""
    snapshots = [item for item in inputs if item.kind == LEGACY_SNAPSHOT_KIND]
    calendar = tuple(format_pinned_ref(item.legacy_path, item.content_hash)
                     for item in inputs if item.kind == CALENDAR_KIND)
    if len(snapshots) != 1 or not calendar:
        raise errors.fail("SNAPSHOT_NOT_READY", "reference inputs lack the legacy SNAPSHOT or "
                          "the calendar", details={"kinds": sorted({i.kind for i in inputs})})
    registry = tuple(format_pinned_ref(item.legacy_path, item.content_hash)
                     for item in sorted(inputs, key=lambda i: i.legacy_path)
                     if item.kind not in (CALENDAR_KIND, LEGACY_SNAPSHOT_KIND))
    return {"legacy_snapshot_object_ref": snapshots[0].object_ref(),
            "registry_and_model_refs": registry, "calendar_refs": calendar}
