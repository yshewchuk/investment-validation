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
* :func:`committed_receipt_for_snapshot` returns the receipt_id of the most
  recent committed receipt in ``scope`` whose resulting snapshot is
  ``snapshot_id`` — ``None`` if none exists. This "latest" resolution is only
  ever correct at PLAN time (a fresh plan should pick up the newest
  generation); external review finding #5 (2026-09-14) is a launch-time
  caller re-running this same "latest" query and picking up a reference-only
  reimport committed after the plan was built. Callers after plan time must
  read a specific receipt_id a plan already pinned, via
  :func:`reference_inputs_for_receipt`, never call this again.
* :func:`reference_inputs_for_snapshot` is the plan-time convenience that
  chains the two: the rows of the most recent committed receipt in ``scope``
  whose resulting snapshot is ``snapshot_id``. It refuses with
  ``SNAPSHOT_NOT_READY`` when that receipt is missing or recorded no
  reference inputs.
* :func:`reference_inputs_for_receipt` returns the rows of ONE SPECIFIC
  receipt, by id — the launch-time-safe read.
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
    "committed_receipt_for_snapshot",
    "insert_reference_inputs",
    "pinned_materialization_refs",
    "reference_inputs_for_receipt",
    "reference_inputs_for_snapshot",
]

CALENDAR_KIND = "calendar"
LEGACY_SNAPSHOT_KIND = "legacy_snapshot"
#: Every kind ``reference_inputs.LEGACY_REFERENCE_INPUTS_V1`` declares.
REFERENCE_KINDS: tuple[str, ...] = (
    CALENDAR_KIND, "model_registry", "structure_champions", "chooser_analog_pool",
    LEGACY_SNAPSHOT_KIND, "champion_artifact", "tier4_serving_cache",
    "pnl_sim_history", "recalibration_pairs",
)
#: Task brief 2026-09-14: model-output kinds a snapshot-mode launch must have
#: pinned before ``legacy_score`` can run against it — see
#: :func:`pinned_materialization_refs`.
_REQUIRED_MODEL_OUTPUT_KINDS = frozenset({"pnl_sim_history", "recalibration_pairs"})
#: ``objects.publish_legacy_file`` labels every object it publishes this way,
#: whatever the file format; a ref rebuilt from a row must carry the same kind.
REFERENCE_OBJECT_KIND = "parquet_fragment"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ReferenceInput:
    """One pinned reference file: its legacy path relative to ``engine.paths.ROOT``.

    ``fold``: the Tier-4 monthly fold (``YYYYMM``) current at import time,
    recorded only for :data:`_REQUIRED_MODEL_OUTPUT_KINDS` (task brief
    2026-09-14) — ``""`` for every other kind, which is not tied to a fold.
    """

    kind: str
    legacy_path: str
    object_id: str
    content_hash: str
    byte_size: int
    fold: str = ""

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
            "content_hash, byte_size, fold) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (receipt_id, item.legacy_path, item.kind, item.object_id, item.content_hash,
             item.byte_size, item.fold))


_COMMITTED_RECEIPT_FOR_SNAPSHOT = (
    "SELECT receipt_id FROM data_import_receipts WHERE scope = ? AND status = 'committed' "
    "AND result_snapshot_id = ? ORDER BY registered_at DESC, rowid DESC LIMIT 1"
)


def committed_receipt_for_snapshot(conn: sqlite3.Connection, *, scope: str,
                                   snapshot_id: str) -> str | None:
    """The receipt_id of the newest committed receipt for ``snapshot_id`` in
    ``scope`` — ``None`` if none exists. PLAN-TIME ONLY (external review #5):
    a launch-time caller must instead read a specific receipt_id a plan
    already pinned (:func:`reference_inputs_for_receipt`), never call this."""
    row = conn.execute(_COMMITTED_RECEIPT_FOR_SNAPSHOT, (scope, snapshot_id)).fetchone()
    return row[0] if row else None


def reference_inputs_for_receipt(conn: sqlite3.Connection, *,
                                 receipt_id: str) -> tuple[ReferenceInput, ...]:
    """The reference inputs of ONE SPECIFIC committed receipt, by id — never
    re-resolved. Refuses ``SNAPSHOT_NOT_READY`` if it pinned no reference
    inputs (or does not exist)."""
    rows = conn.execute(
        "SELECT kind, legacy_path, object_id, content_hash, byte_size, fold FROM "
        "data_import_reference_inputs WHERE receipt_id = ? ORDER BY legacy_path",
        (receipt_id,)).fetchall()
    if not rows:
        raise errors.fail("SNAPSHOT_NOT_READY", "the import receipt pinned no reference inputs",
                          details={"receipt_id": receipt_id})
    return tuple(ReferenceInput(kind=r[0], legacy_path=r[1], object_id=r[2], content_hash=r[3],
                                byte_size=r[4], fold=r[5]) for r in rows)


def reference_inputs_for_snapshot(conn: sqlite3.Connection, *, scope: str,
                                  snapshot_id: str) -> tuple[ReferenceInput, ...]:
    """The reference inputs of the newest committed receipt for ``snapshot_id``
    in ``scope`` — plan-time convenience; see :func:`committed_receipt_for_snapshot`."""
    receipt_id = committed_receipt_for_snapshot(conn, scope=scope, snapshot_id=snapshot_id)
    if receipt_id is None:
        raise errors.fail("SNAPSHOT_NOT_READY", "no committed import receipt for this snapshot",
                          details={"scope": scope, "snapshot_id": snapshot_id})
    return reference_inputs_for_receipt(conn, receipt_id=receipt_id)


def pinned_materialization_refs(inputs: Sequence[ReferenceInput]) -> dict:
    """``{"legacy_snapshot_object_ref", "registry_and_model_refs", "calendar_refs"}``.

    Refuses ``SNAPSHOT_NOT_READY`` (task brief 2026-09-14) when the pinned set
    lacks the legacy SNAPSHOT, the calendar, or either
    :data:`_REQUIRED_MODEL_OUTPUT_KINDS` file — the plan-time guard for
    ``legacy_score`` in snapshot mode: an older snapshot imported before these
    two kinds existed has no such rows, and must refuse rather than launch
    silently without them.
    """
    snapshots = [item for item in inputs if item.kind == LEGACY_SNAPSHOT_KIND]
    calendar = tuple(format_pinned_ref(item.legacy_path, item.content_hash)
                     for item in inputs if item.kind == CALENDAR_KIND)
    kinds = {item.kind for item in inputs}
    if len(snapshots) != 1 or not calendar or not _REQUIRED_MODEL_OUTPUT_KINDS <= kinds:
        raise errors.fail("SNAPSHOT_NOT_READY", "reference inputs lack the legacy SNAPSHOT, the "
                          "calendar, or a required Tier-4-derived model output",
                          details={"kinds": sorted(kinds)})
    registry = tuple(format_pinned_ref(item.legacy_path, item.content_hash)
                     for item in sorted(inputs, key=lambda i: i.legacy_path)
                     if item.kind not in (CALENDAR_KIND, LEGACY_SNAPSHOT_KIND))
    return {"legacy_snapshot_object_ref": snapshots[0].object_ref(),
            "registry_and_model_refs": registry, "calendar_refs": calendar}
