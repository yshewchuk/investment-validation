"""Rebuild one pinned snapshot's neutral re-registration inventory from catalog SQL.

``neutral_inventory`` reads catalog metadata only: no ``data_contracts``, object
bytes or mutable head; the caller attests ``content_hash(payload)``.

``export`` (the CLI) additionally checks the explicit pins against the mutable
head, verifies every object descriptor against the artifact store and publishes
the inventory as one atomically written, canonical-JSON export file. ``register``
re-verifies one such export against the live catalog and object store and returns
a :class:`VerifiedInventory`; the catalog is only ever opened read-only.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import math
import os
import sqlite3
import sys
import uuid
from pathlib import Path, PurePath

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
                            " finality_receipt_refs_json, knowledge_mode_by_table_json"
                            " FROM data_snapshots"
                            " WHERE snapshot_id = ?", (snapshot_id,)).fetchone()
        if snapshot is None:
            raise errors.fail("INPUT_CHANGED", "unknown snapshot")
        tables = _tables(conn, snapshot_id)
        if {table["table_name"] for table in tables} != set(
                _stored_json(snapshot["knowledge_mode_by_table_json"], dict)):
            raise errors.fail("INPUT_CHANGED", "snapshot table membership is incomplete")
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


_SQLITE_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")


def _refuse_destination_collisions(out_path, catalog_path, object_paths, objects_root):
    out_real = os.path.realpath(out_path)
    catalog_real = os.path.realpath(catalog_path)
    for refused in (catalog_real,
                    *(catalog_real + suffix for suffix in _SQLITE_SIDECAR_SUFFIXES)):
        if out_real == refused:
            raise errors.fail("INPUT_CHANGED", "the export destination resolves to an input",
                              details={"path": refused})
    namespace_real = os.path.realpath(os.path.join(objects_root, "objects"))
    if PurePath(out_real).is_relative_to(PurePath(namespace_real)):
        raise errors.fail("INPUT_CHANGED", "the export destination resolves to an input",
                          details={"path": out_real})
    for source_path in (catalog_path, *object_paths):
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
    _refuse_destination_collisions(out, catalog_path, object_paths, objects_root)
    data = _export_bytes(inventory)
    _publish(out, data, lambda: _verify_head(conn, scope=scope, snapshot_id=snapshot_id,
                                             receipt_id=receipt_id, generation=generation))


# Register verification: validate a neutral export against the live catalog and
# object store with typed refusals only; no catalog write happens here.

VERIFIED_SCHEMA_VERSION = "reregister_snapshot_verified.v1"
_MAX_JSON_DEPTH = 32
_KNOWLEDGE_MODES = {"observed", "attested_stable", "reconstructed"}
_PRICE_CAPTURE_KEYS = {"capture_id", "receipt_id", "ticker", "source_kind", "source_hash", "retrieved_at", "outcome", "rows_added", "rows_tombstoned", "created_at", "contract_id"}
_PRICE_CAPTURE_TEXT = ("capture_id", "receipt_id", "ticker", "source_kind", "source_hash", "retrieved_at", "outcome", "created_at", "contract_id")
_MOVES_CAPTURE_KEYS = {"capture_id", "ticker", "created_at", "contract_id", "outcome"}
_POS, _NAT, _MODE = object(), object(), object()
_SPECIAL = {_POS: lambda v: type(v) is int and v > 0, _NAT: lambda v: type(v) is int and v >= 0,
            _MODE: lambda v: type(v) is str and v in _KNOWLEDGE_MODES}
_EXPORT_SCHEMA = {
    "schema_version": str, "content_hash": str, "inventory": {
        "schema_version": str, "scope": str, "snapshot_id": str, "receipt_id": str,
        "calendar_version": str, "source_priority_version": str, "generation": _POS,
        "finality_receipt_refs": [str], "tables": [{
            "table_name": str, "dataset_version_id": str, "contract_id": str,
            "knowledge_mode": _MODE, "coverage_receipt_refs": [str],
            "availability_evidence_refs": [str], "fragments": [{
                "fragment_id": str, "partition_key": str, "row_count": _NAT,
                "object": {"kind": str, "object_id": str, "content_hash": str, "byte_size": int},
                "primary_key_min": [(str, int, bool, float)],
                "primary_key_max": [(str, int, bool, float)],
                "time_min": (str, type(None)), "time_max": (str, type(None))}]}],
        "references": [{
            "kind": str, "legacy_path": str, "object_id": str, "content_hash": str,
            "byte_size": int, "fold": (str, int, type(None))}],
        "lineage": {"receipt_ids": [str], "edges": [[str]]}, "captures": dict}}


def _corrupt(name):
    return errors.fail("MANIFEST_CORRUPT", f"the exported {name} has an unexpected shape")


def _utf8(text):
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _corrupt("string") from exc


def _walk_json(root):
    stack = [(root, 1)]
    while stack:
        value, depth = stack.pop()
        kind = type(value)
        if kind is dict:
            if depth > _MAX_JSON_DEPTH:
                raise _corrupt("nesting depth")
            if (type(value.get("schema_version")) is str and value["schema_version"].startswith(("table_contract.", "column_contract."))) or {"contract_id", "table_name", "columns"} <= set(value):
                raise errors.fail("UNSUPPORTED_CONTRACT", "the export embeds a contract document; only contract_id refs are allowed")
            for key, item in value.items():
                _utf8(key)
                stack.append((item, depth + 1))
        elif kind is list:
            if depth > _MAX_JSON_DEPTH:
                raise _corrupt("nesting depth")
            stack.extend((item, depth + 1) for item in value)
        elif kind is str:
            _utf8(value)
        elif kind is float:
            if not math.isfinite(value):
                raise _corrupt("number")
        elif kind is not int and kind is not bool and value is not None:
            raise _corrupt("value")


def _match(value, spec, name):
    if type(spec) is dict:
        if type(value) is not dict or set(value) != set(spec):
            raise _corrupt(name)
        for key, sub in spec.items():
            _match(value[key], sub, name)
    elif type(spec) is list:
        if type(value) is not list:
            raise _corrupt(name)
        for item in value:
            _match(item, spec[0], name)
    else:
        check = _SPECIAL.get(spec)
        ok = check(value) if check is not None else type(value) in (spec if type(spec) is tuple else (spec,))
        if not ok:
            raise _corrupt(name)


def _capture_rows(captures, tables):
    if type(captures) is not dict or not set(captures) <= set(tables):
        raise _corrupt("captures")
    for table, rows in captures.items():
        if table == PRICE_HISTORY:
            keys, text, counts = _PRICE_CAPTURE_KEYS, _PRICE_CAPTURE_TEXT, ("rows_added", "rows_tombstoned")
        elif table == COMPUTED_MOVES:
            keys, text, counts = _MOVES_CAPTURE_KEYS, _MOVES_CAPTURE_KEYS, ()
        else:
            raise _corrupt("captures")
        if type(rows) is not list:
            raise _corrupt("captures")
        for row in rows:
            if type(row) is not dict or set(row) != keys:
                raise _corrupt("captures")
            if (any(type(row[name]) is not str for name in text)
                    or any(type(row[name]) is not int or row[name] < 0 for name in counts)):
                raise _corrupt("captures")


def _validate_export(wrapper):
    _walk_json(wrapper)
    _match(wrapper, _EXPORT_SCHEMA, "export wrapper")
    inventory = wrapper["inventory"]
    if wrapper["schema_version"] != EXPORT_SCHEMA_VERSION or inventory["schema_version"] != SCHEMA_VERSION:
        raise _corrupt("export wrapper")
    if any(len(edge) != 2 for edge in inventory["lineage"]["edges"]):
        raise _corrupt("lineage edges")
    names = [table["table_name"] for table in inventory["tables"]]
    if len(set(names)) != len(names):
        raise _corrupt("tables")
    _capture_rows(inventory["captures"], names)
    return inventory


def _load_export(inventory_path):
    try:
        raw = Path(inventory_path).read_bytes()
    except OSError as exc:
        raise errors.fail("INPUT_CHANGED", "the inventory file could not be read") from exc
    try:
        wrapper = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise errors.fail("MANIFEST_CORRUPT", "the inventory file is not valid JSON") from exc
    inventory = _validate_export(wrapper)
    if wrapper["content_hash"] != content_hash(inventory):
        raise _corrupt("content hash")
    return inventory


def _membership(tables):
    return {table["table_name"]: (len(table["fragments"]), frozenset(
        (fragment["fragment_id"], fragment["object"]["object_id"], fragment["partition_key"])
        for fragment in table["fragments"])) for table in tables}


def _verify_live_inventory(inventory, live):
    if _membership(inventory["tables"]) != _membership(live["tables"]):
        raise errors.fail("CONTRACT_MISMATCH", "the exported table or fragment membership does not match the current catalog")
    encoding = {"sort_keys": True, "ensure_ascii": True, "separators": (",", ":")}
    if json.dumps(live, **encoding) != json.dumps(inventory, **encoding):
        raise errors.fail("INPUT_CHANGED", "the exported inventory does not match the current catalog")


@dataclasses.dataclass(frozen=True, kw_only=True)
class VerifiedInventory:
    payload_bytes: bytes

    def to_bytes(self) -> bytes:
        return self.payload_bytes


def register(conn, *, inventory_path, objects_root):
    inventory = _load_export(inventory_path)
    live = neutral_inventory(conn, scope=inventory["scope"], snapshot_id=inventory["snapshot_id"],
                              receipt_id=inventory["receipt_id"], generation=inventory["generation"])
    _verify_live_inventory(inventory, live)
    _verify_objects(inventory, objects_root)
    payload = {"schema_version": VERIFIED_SCHEMA_VERSION, "scope": inventory["scope"],
               "snapshot_id": inventory["snapshot_id"], "generation": inventory["generation"],
               "receipt_id": inventory["receipt_id"], "content_hash": content_hash(inventory)}
    return VerifiedInventory(payload_bytes=(canonical_json(payload) + "\n").encode("utf-8"))


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
    register_cmd = commands.add_parser(
        "register", help="verify one export and write its verified inventory to stdout")
    register_cmd.add_argument("--inventory", required=True, type=Path, help="export file to verify")
    register_cmd.add_argument("--catalog", required=True, type=Path)
    register_cmd.add_argument("--objects", required=True, type=Path, help="ArtifactStore root")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    conn = None
    try:
        conn = _open_read_only(args.catalog)
        if args.command == "register":
            verified = register(conn, inventory_path=args.inventory, objects_root=args.objects)
            sys.stdout.buffer.write(verified.to_bytes())
        else:
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
