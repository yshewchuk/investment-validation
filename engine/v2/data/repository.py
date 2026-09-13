"""Exact snapshot resolution — phase-2 guide §8.1.

``Repository.resolve(snapshot_id)`` rebuilds a ``SnapshotRef`` entirely from
catalog rows, never from ``data_snapshot_heads``: it loads the snapshot row,
every table's dataset version, and every fragment's full ordered membership,
then feeds them back through the *same public builders* that mint identity at
commit time (``manifests.fragment_record``, ``.dataset_manifest``,
``.snapshot_ref`` — ``engine/v2/data/catalog.py`` uses the same three), and
compares the freshly recomputed id/manifest_hash against the catalog's own
primary key at each level. A mismatch — corrupt or missing membership, an
edited row, a dropped join partner — surfaces as ``MANIFEST_CORRUPT``, this
package's judgement call (task brief decision 1: §11 has no
``INTEGRITY_FAILED``, the guide prose's name for the same case). An unknown
``snapshot_id`` is ``SNAPSHOT_NOT_FOUND``.

Every ``FragmentRecord`` field the v1 catalog schema could not store on its
own — only ``input_receipt_refs`` — was added as v2 (task brief decision 3,
``engine/v2/data/schema.py``); every other field (object/contract refs via a
join, partition_key, row_count, byte/logical hashes, key/time bounds,
import_request_hash) was already a v1 column, so no ``manifest_hash`` column
is needed on ``data_fragments`` either: rebuilding through
``manifests.fragment_record`` recomputes it deterministically from the other
stored fields, the same way the original commit did.

One read-only SQLite transaction (:func:`_read_only`) covers the whole walk,
so a concurrent commit elsewhere can never hand back a torn snapshot, and a
resolved ``SnapshotRef`` never changes underneath a caller holding it — every
row this reads is append-only.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and this package's own ``manifests``/``objects``/
``errors`` — never ``engine.v2.ops``.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager

from engine.v2.contracts import ObjectRef, SnapshotRef, TableContractRef
from engine.v2.data import manifests
from engine.v2.data.errors import fail
from engine.v2.data.objects import FragmentInspection

__all__ = ["Repository"]


@contextmanager
def _read_only(conn: sqlite3.Connection):
    if conn.in_transaction:
        raise RuntimeError("nested data catalog transaction")
    conn.execute("BEGIN")
    try:
        yield conn
    finally:
        conn.execute("ROLLBACK")


class Repository:
    """Read-only access to one already-migrated data catalog connection."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def resolve(self, snapshot_id: str) -> SnapshotRef:
        with _read_only(self._conn) as conn:
            snap_row = conn.execute(
                "SELECT * FROM data_snapshots WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
            if snap_row is None:
                raise fail("SNAPSHOT_NOT_FOUND", "unknown snapshot id",
                          details={"snapshot_id": snapshot_id})
            table_rows = conn.execute(
                "SELECT table_name, dataset_version_id FROM data_snapshot_tables "
                "WHERE snapshot_id = ?", (snapshot_id,)).fetchall()
            table_versions = {row["table_name"]: self._manifest(conn, row["dataset_version_id"])
                              for row in table_rows}
            snap = manifests.snapshot_ref(
                table_versions, calendar_version=snap_row["calendar_version"],
                source_priority_version=snap_row["source_priority_version"],
                finality_receipt_refs=tuple(json.loads(snap_row["finality_receipt_refs_json"])),
                parent_snapshot_id=snap_row["parent_snapshot_id"])
            if (snap.snapshot_id != snap_row["snapshot_id"]
                    or snap.manifest_hash != snap_row["manifest_hash"]):
                raise fail("MANIFEST_CORRUPT", "snapshot does not match its catalog identity",
                          details={"snapshot_id": snapshot_id})
            return snap

    def _manifest(self, conn: sqlite3.Connection, dataset_version_id: str):
        dsv_row = conn.execute(
            "SELECT * FROM data_dataset_versions WHERE dataset_version_id = ?",
            (dataset_version_id,)).fetchone()
        if dsv_row is None:
            raise fail("MANIFEST_CORRUPT", "snapshot references an unknown dataset version",
                      details={"dataset_version_id": dataset_version_id})
        contract_ref = self._contract_ref(conn, dsv_row["contract_id"])
        records = self._records(conn, dataset_version_id, contract_ref)
        evidence = json.loads(dsv_row["evidence_json"])
        manifest = manifests.dataset_manifest(
            contract_ref, records, knowledge_mode=dsv_row["knowledge_mode"],
            coverage_receipt_refs=tuple(evidence["coverage_receipt_refs"]),
            availability_evidence_refs=tuple(evidence["availability_evidence_refs"]),
            parent_dataset_version_id=dsv_row["parent_dataset_version_id"])
        ref = manifest.dataset_version_ref
        if (ref.dataset_version_id != dsv_row["dataset_version_id"]
                or ref.manifest_hash != dsv_row["manifest_hash"]):
            raise fail("MANIFEST_CORRUPT", "dataset version does not match its catalog identity",
                      details={"dataset_version_id": dataset_version_id})
        return manifest

    def _contract_ref(self, conn: sqlite3.Connection, contract_id: str) -> TableContractRef:
        row = conn.execute("SELECT definition_hash FROM data_contracts WHERE contract_id = ?",
                           (contract_id,)).fetchone()
        if row is None:
            raise fail("MANIFEST_CORRUPT", "dataset version references an unknown contract",
                      details={"contract_id": contract_id})
        return TableContractRef(contract_id=contract_id, definition_hash=row["definition_hash"])

    def _records(self, conn: sqlite3.Connection, dataset_version_id: str,
                contract_ref: TableContractRef) -> list:
        rows = conn.execute(
            "SELECT f.* FROM data_version_fragments vf JOIN data_fragments f "
            "ON f.fragment_id = vf.fragment_id WHERE vf.dataset_version_id = ? "
            "ORDER BY vf.ordinal", (dataset_version_id,)).fetchall()
        return [self._record(conn, row, contract_ref) for row in rows]

    def _record(self, conn: sqlite3.Connection, row: sqlite3.Row, contract_ref: TableContractRef):
        object_row = conn.execute("SELECT * FROM data_objects WHERE object_id = ?",
                                  (row["object_id"],)).fetchone()
        if object_row is None:
            raise fail("MANIFEST_CORRUPT", "fragment references an unknown object",
                      details={"object_id": row["object_id"]})
        object_ref = ObjectRef(kind=object_row["kind"], object_id=object_row["object_id"],
                               content_hash=object_row["content_hash"],
                               byte_size=object_row["byte_size"])
        bounds = json.loads(row["key_bounds_json"])
        time_bounds = (json.loads(row["time_bounds_json"]) if row["time_bounds_json"] is not None
                      else {"time_min": None, "time_max": None})
        inspection = FragmentInspection(
            object_ref=object_ref, partition_key=row["partition_key"], row_count=row["row_count"],
            byte_hash=row["byte_hash"], logical_content_hash=row["logical_content_hash"],
            primary_key_min=tuple(bounds["primary_key_min"]),
            primary_key_max=tuple(bounds["primary_key_max"]),
            time_min=time_bounds["time_min"], time_max=time_bounds["time_max"])
        record = manifests.fragment_record(
            inspection, contract_ref,
            input_receipt_refs=tuple(json.loads(row["input_receipt_refs_json"])),
            import_request_hash=row["import_request_hash"])
        if record.fragment_id != row["fragment_id"]:
            raise fail("MANIFEST_CORRUPT", "fragment does not match its catalog identity",
                      details={"fragment_id": row["fragment_id"]})
        return record
