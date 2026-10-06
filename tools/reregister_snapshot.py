"""Rebuild one pinned snapshot's neutral re-registration inventory from catalog SQL.

``neutral_inventory`` reads catalog metadata only: no ``data_contracts``, object
bytes or mutable head; the caller attests ``content_hash(payload)``.

``export`` (the CLI) additionally checks the explicit pins against the mutable
head, verifies every object descriptor against the artifact store and publishes
the inventory as one atomically written, canonical-JSON export file. The
catalog is only ever opened read-only; no register path lives here.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sqlite3
import sys
import uuid
from pathlib import Path

from engine.v2.contracts import ArtifactRef
from engine.v2.data import errors
from engine.v2.foundation import (
    CONTENT_HASH_PREFIX,
    ArtifactError,
    ArtifactStore,
    canonical_json,
    content_hash,
)

SCHEMA_VERSION = "reregister_snapshot.v1"
EXPORT_SCHEMA_VERSION = "reregister_snapshot_export.v1"
PRICE_HISTORY = "price_history"
COMPUTED_MOVES = "computed_moves"
_MAX_LINEAGE_DEPTH = 64


def _execute(conn, statement, parameters=()):
    try:
        return conn.execute(statement, parameters)
    except sqlite3.Error as exc:
        raise errors.fail("INPUT_CHANGED", "catalog metadata is not readable") from exc


@contextlib.contextmanager
def _read_transaction(conn):
    if conn.in_transaction:
        raise errors.fail("INPUT_CHANGED", "a catalog transaction is already active")
    _execute(conn, "BEGIN")
    try:
        yield conn
    finally:
        _execute(conn, "ROLLBACK")


def _stored_json(raw, kind):
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise errors.fail("MANIFEST_CORRUPT", "stored metadata is not valid JSON") from exc
    if not isinstance(value, kind):
        raise errors.fail("MANIFEST_CORRUPT", "stored metadata has an unexpected shape")
    return value


def _stored_mapping(raw, fields):
    value = _stored_json(raw, dict)
    if any(not isinstance(value.get(name), kinds) for name, kinds in fields):
        raise errors.fail("MANIFEST_CORRUPT", "stored metadata is missing a required field")
    return value


def _shape(value, kinds):
    if not isinstance(value, kinds):
        raise errors.fail("MANIFEST_CORRUPT", "stored metadata has an unexpected shape")
    return value


def _items(value, kinds):
    return [_shape(item, kinds) for item in value]


_TIME_BOUNDS_FIELDS = (("time_min", (str, type(None))), ("time_max", (str, type(None))))


def _time_bounds(raw):
    if raw is None:
        return {"time_min": None, "time_max": None}
    value = _stored_json(raw, dict)
    if any(name not in value for name, _ in _TIME_BOUNDS_FIELDS):
        raise errors.fail("MANIFEST_CORRUPT", "stored metadata is missing a required field")
    return {name: _shape(value[name], kinds) for name, kinds in _TIME_BOUNDS_FIELDS}


def _fragment_document(row):
    bounds = _stored_mapping(row["key_bounds_json"], (("primary_key_min", list), ("primary_key_max", list)))
    times = _time_bounds(row["time_bounds_json"])
    return {"fragment_id": row["fragment_id"],
            "object": {"kind": row["kind"], "object_id": row["object_id"],
                       "content_hash": row["content_hash"], "byte_size": row["byte_size"]},
            "partition_key": row["partition_key"], "row_count": row["row_count"],
            "primary_key_min": _items(bounds["primary_key_min"], (str, int, float, bool)),
            "primary_key_max": _items(bounds["primary_key_max"], (str, int, float, bool)),
            "time_min": times["time_min"], "time_max": times["time_max"]}


def _fragments(conn, dataset_version_id, contract_id):
    rows = _execute(conn,
        "SELECT vf.ordinal, f.fragment_id, f.contract_id AS fragment_contract_id, f.partition_key,"
        " f.row_count, f.byte_hash, f.key_bounds_json, f.time_bounds_json, o.object_id, o.kind,"
        " o.content_hash, o.byte_size FROM data_version_fragments vf"
        " LEFT JOIN data_fragments f ON f.fragment_id = vf.fragment_id"
        " LEFT JOIN data_objects o ON o.object_id = f.object_id"
        " WHERE vf.dataset_version_id = ? ORDER BY vf.ordinal", (dataset_version_id,)).fetchall()
    if [row["ordinal"] for row in rows] != list(range(len(rows))):
        raise errors.fail("INPUT_CHANGED", "dataset version fragment membership is incomplete")
    for row in rows:
        if row["fragment_id"] is None or row["object_id"] is None:
            raise errors.fail("INPUT_CHANGED", "dataset version fragment membership is incomplete")
        if (row["fragment_contract_id"] != contract_id
                or row["byte_hash"] != row["content_hash"]):
            raise errors.fail("MANIFEST_CORRUPT", "stored fragment identity is inconsistent")
    fragments = [_fragment_document(row) for row in rows]
    return fragments


def _tables(conn, snapshot_id):
    rows = _execute(conn,
        "SELECT st.table_name, st.dataset_version_id, v.contract_id, v.knowledge_mode,"
        " v.row_count, v.evidence_json FROM data_snapshot_tables st"
        " LEFT JOIN data_dataset_versions v ON v.dataset_version_id = st.dataset_version_id"
        " WHERE st.snapshot_id = ? ORDER BY st.table_name", (snapshot_id,)).fetchall()
    tables = []
    for row in rows:
        if row["contract_id"] is None:
            raise errors.fail("INPUT_CHANGED", "snapshot table membership is incomplete")
        fragments = _fragments(conn, row["dataset_version_id"], row["contract_id"])
        if sum(item["row_count"] for item in fragments) != row["row_count"]:
            raise errors.fail("MANIFEST_CORRUPT", "stored dataset row count is inconsistent")
        evidence = _stored_mapping(row["evidence_json"],
                                   (("coverage_receipt_refs", list),
                                    ("availability_evidence_refs", list)))
        tables.append({"table_name": row["table_name"],
                       "dataset_version_id": row["dataset_version_id"],
                       "contract_id": row["contract_id"],
                       "knowledge_mode": row["knowledge_mode"],
                       "coverage_receipt_refs": _items(evidence["coverage_receipt_refs"], str),
                       "availability_evidence_refs": _items(evidence["availability_evidence_refs"], str),
                       "fragments": fragments})
    return tables


def _lineage(conn, receipt_id):
    visited, edges, current = {receipt_id}, [], receipt_id
    for _ in range(_MAX_LINEAGE_DEPTH):
        row = _execute(conn, "SELECT base_receipt_id FROM data_receipt_lineage"
                             " WHERE receipt_id = ?", (current,)).fetchone()
        if row is None:
            return {"receipt_ids": sorted(visited), "edges": sorted(edges)}, visited
        base = row["base_receipt_id"]
        if base in visited:
            raise errors.fail("INPUT_CHANGED", "receipt lineage is cyclic")
        parent = _execute(conn, "SELECT status FROM data_import_receipts WHERE receipt_id = ?",
                          (base,)).fetchone()
        if parent is None or parent["status"] != "committed":
            raise errors.fail("INPUT_CHANGED", "receipt lineage ancestor is not committed")
        visited.add(base)
        edges.append([current, base])
        current = base
    raise errors.fail("INPUT_CHANGED", "receipt lineage exceeds the supported depth")


def _captures(conn, tables, lineage_ids):
    by_name = {table["table_name"]: table for table in tables}
    captures = {}
    if PRICE_HISTORY in by_name:
        marks = ", ".join("?" for _ in lineage_ids)
        rows = _execute(conn,
            "SELECT capture_id, receipt_id, ticker, source_kind, source_hash, retrieved_at,"
            " outcome, rows_added, rows_tombstoned, created_at, contract_id"
            " FROM data_price_captures WHERE contract_id = ? AND receipt_id IN"
            f" ({marks}) ORDER BY capture_id",
            (by_name[PRICE_HISTORY]["contract_id"], *sorted(lineage_ids))).fetchall()
        captures[PRICE_HISTORY] = [dict(row) for row in rows]
    if COMPUTED_MOVES in by_name:
        rows = _execute(conn,
            "SELECT capture_id, ticker, created_at, contract_id, outcome"
            " FROM data_computed_moves_captures WHERE contract_id = ? ORDER BY capture_id",
            (by_name[COMPUTED_MOVES]["contract_id"],)).fetchall()
        captures[COMPUTED_MOVES] = [dict(row) for row in rows]
    return captures


def _validate_pin(scope, snapshot_id, receipt_id, generation):
    for name, value in (("scope", scope), ("snapshot_id", snapshot_id), ("receipt_id", receipt_id)):
        if not isinstance(value, str) or not value:
            raise errors.fail("INPUT_CHANGED", f"pinned {name} must be a non-empty string")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise errors.fail("INPUT_CHANGED", "pinned generation must be a positive integer")


def neutral_inventory(conn, *, scope, snapshot_id, receipt_id, generation):
    """Rebuild the pinned snapshot's neutral inventory from catalog SQL."""
    _validate_pin(scope, snapshot_id, receipt_id, generation)
    with _read_transaction(conn):
        receipt = _execute(conn, "SELECT status, scope, result_snapshot_id FROM data_import_receipts"
                           " WHERE receipt_id = ?", (receipt_id,)).fetchone()
        if (receipt is None or receipt["status"] != "committed"
                or receipt["scope"] != scope or receipt["result_snapshot_id"] != snapshot_id):
            raise errors.fail("INPUT_CHANGED", "pinned receipt does not match scope and snapshot")
        snapshot = _execute(conn, "SELECT calendar_version, source_priority_version,"
                            " finality_receipt_refs_json FROM data_snapshots"
                            " WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if snapshot is None:
            raise errors.fail("INPUT_CHANGED", "unknown snapshot")
        tables = _tables(conn, snapshot_id)
        references = [dict(row) for row in _execute(conn,
            "SELECT kind, legacy_path, object_id, content_hash, byte_size, fold"
            " FROM data_import_reference_inputs WHERE receipt_id = ? ORDER BY legacy_path",
            (receipt_id,)).fetchall()]
        lineage, lineage_ids = _lineage(conn, receipt_id)
        finality = _items(_stored_json(snapshot["finality_receipt_refs_json"], list), str)
        return {"schema_version": SCHEMA_VERSION, "scope": scope, "snapshot_id": snapshot_id,
                "generation": generation, "receipt_id": receipt_id,
                "calendar_version": snapshot["calendar_version"],
                "source_priority_version": snapshot["source_priority_version"],
                "finality_receipt_refs": finality,
                "tables": tables, "references": references, "lineage": lineage,
                "captures": _captures(conn, tables, lineage_ids)}


def _open_read_only(catalog_path):
    """The one catalog open: a ``mode=ro`` URI, so no write or DDL is possible."""
    try:
        conn = sqlite3.connect(Path(catalog_path).resolve().as_uri() + "?mode=ro",
                               uri=True, isolation_level=None, timeout=5.0)
    except sqlite3.Error as exc:
        raise errors.fail("INPUT_CHANGED", "the catalog could not be opened read-only") from exc
    conn.row_factory = sqlite3.Row
    return conn


def _verify_head(conn, *, scope, snapshot_id, receipt_id, generation):
    row = _execute(conn, "SELECT snapshot_id, generation, update_receipt_ref"
                         " FROM data_snapshot_heads WHERE scope = ?", (scope,)).fetchone()
    if (row is None or row["snapshot_id"] != snapshot_id or row["generation"] != generation
            or row["update_receipt_ref"] != receipt_id):
        raise errors.fail("INPUT_CHANGED", "the mutable head does not match the explicit pins")


def _artifact_ref(descriptor):
    content_hash_value = descriptor["content_hash"]
    if (not isinstance(content_hash_value, str)
            or not content_hash_value.startswith(CONTENT_HASH_PREFIX)
            or isinstance(descriptor["byte_size"], bool)
            or not isinstance(descriptor["byte_size"], int)):
        raise errors.fail("OBJECT_CORRUPT", "a pinned object descriptor cannot be verified")
    digest = content_hash_value.removeprefix(CONTENT_HASH_PREFIX)
    return ArtifactRef(artifact_id=descriptor["object_id"], content_hash=content_hash_value,
                       schema_ref=descriptor["kind"], byte_size=descriptor["byte_size"],
                       storage_key=f"objects/{digest[:2]}/{digest}")


def _object_descriptors(inventory):
    for table in inventory["tables"]:
        for fragment in table["fragments"]:
            yield fragment["object"]
    yield from inventory["references"]


def _verify_objects(inventory, objects_root):
    """Re-hash every recorded object; return its verified path."""
    store = ArtifactStore(objects_root)
    paths = []
    for descriptor in _object_descriptors(inventory):
        try:
            paths.append(store.verify(_artifact_ref(descriptor)))
        except ArtifactError as exc:
            raise errors.fail("OBJECT_CORRUPT",
                              "a pinned object does not match the object store") from exc
    return paths


def _refuse_destination_collisions(out_path, source_paths):
    out_real = os.path.realpath(out_path)
    for source_path in source_paths:
        source_real = os.path.realpath(source_path)
        same_file = (out_real == source_real or
                     (os.path.exists(out_path) and os.path.exists(source_path)
                      and os.path.samefile(out_path, source_path)))
        if same_file:
            raise errors.fail("INPUT_CHANGED", "the export destination resolves to an input",
                              details={"path": source_real})


def _export_bytes(inventory):
    wrapper = {"schema_version": EXPORT_SCHEMA_VERSION, "inventory": inventory,
               "content_hash": content_hash(inventory)}
    return canonical_json(wrapper).encode("utf-8") + b"\n"


def _publish(out_path, data, recheck):
    """Temp file in the existing parent, fsync, head recheck, then one replace.

    The destination is never opened or truncated before the rename; any failure
    before it removes the temp and leaves the prior destination byte-identical.
    """
    out_path = Path(out_path)
    tmp = out_path.parent / f".{out_path.name}.{uuid.uuid4().hex}.part"
    replaced = False
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            handle = os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        recheck()
        os.replace(tmp, out_path)
        replaced = True
    except OSError as exc:
        raise errors.fail("INPUT_CHANGED", "the export file could not be written") from exc
    finally:
        if not replaced:
            with contextlib.suppress(OSError):
                os.unlink(tmp)


def export_inventory(conn, *, scope, snapshot_id, receipt_id, generation,
                     catalog_path, objects_root, out):
    """Export the pinned snapshot's verified neutral inventory, or refuse typed."""
    _validate_pin(scope, snapshot_id, receipt_id, generation)
    _verify_head(conn, scope=scope, snapshot_id=snapshot_id, receipt_id=receipt_id,
                 generation=generation)
    inventory = neutral_inventory(conn, scope=scope, snapshot_id=snapshot_id,
                                  receipt_id=receipt_id, generation=generation)
    object_paths = _verify_objects(inventory, objects_root)
    _refuse_destination_collisions(out, [catalog_path, *object_paths])
    data = _export_bytes(inventory)
    _publish(out, data, lambda: _verify_head(conn, scope=scope, snapshot_id=snapshot_id,
                                             receipt_id=receipt_id, generation=generation))


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write one pinned snapshot's verified export")
    export.add_argument("--snapshot-id", required=True)
    export.add_argument("--receipt-id", required=True)
    export.add_argument("--generation", required=True, type=int)
    export.add_argument("--scope", required=True)
    export.add_argument("--catalog", required=True, type=Path)
    export.add_argument("--objects", required=True, type=Path, help="ArtifactStore root")
    export.add_argument("--out", required=True, type=Path, help="export file destination")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    conn = None
    try:
        conn = _open_read_only(args.catalog)
        export_inventory(conn, scope=args.scope, snapshot_id=args.snapshot_id,
                         receipt_id=args.receipt_id, generation=args.generation,
                         catalog_path=args.catalog, objects_root=args.objects, out=args.out)
    except errors.DataError as exc:
        print(json.dumps({"refused": exc.code, "message": exc.problem.message},
                         sort_keys=True), file=sys.stderr)
        return 2
    finally:
        if conn is not None:
            conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
