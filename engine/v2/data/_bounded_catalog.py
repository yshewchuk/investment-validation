"""Scalar-only catalog admission prerequisite; not a resolver or public factory.

The future bounded-session caller owns the already-open read transaction and
must still transfer/decode, enforce the UTF-8 counter, and verify native identity.
"""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass

from engine.v2.data import errors

# Fixed future transfer projection. Tokens ending in '?' permit SQL NULL;
# '#' marks INTEGER, all other fields require TEXT. Never include storage_key,
# registered_at or commit_receipt_ref. Parents/evidence are values, not walks.
_SNAPSHOT = """s.snapshot_id s.parent_snapshot_id? s.manifest_hash s.calendar_version
s.source_priority_version s.finality_receipt_refs_json s.knowledge_mode_by_table_json"""
_TABLE = """t.snapshot_id t.table_name t.dataset_version_id d.dataset_version_id
 d.contract_id d.parent_dataset_version_id? d.manifest_hash d.logical_content_hash
 d.row_count# d.knowledge_mode d.evidence_json d.partition_logical_hashes_json
 c.contract_id c.schema_version c.table_name c.definition_hash c.definition_json"""
_FRAGMENT = """m.dataset_version_id m.ordinal# m.fragment_id f.fragment_id f.object_id
 f.contract_id f.partition_key f.row_count# f.byte_hash f.logical_content_hash
 f.key_bounds_json f.time_bounds_json? f.import_request_hash f.input_receipt_refs_json
 o.object_id o.kind o.content_hash o.byte_size#"""
_TABLES = "data_snapshot_tables t"
_MEMBERS = (_TABLES + " JOIN data_version_fragments m "
            "ON m.dataset_version_id=t.dataset_version_id")
_TABLE_JOINS = (_TABLES + " LEFT JOIN data_dataset_versions d "
                "ON d.dataset_version_id=t.dataset_version_id "
                "LEFT JOIN data_contracts c ON c.contract_id=d.contract_id")
_FRAGMENT_JOINS = (_MEMBERS + " LEFT JOIN data_fragments f ON f.fragment_id=m.fragment_id "
                   "LEFT JOIN data_objects o ON o.object_id=f.object_id "
                   "LEFT JOIN data_contracts c ON c.contract_id=f.contract_id")
_LIMITS = {"max_catalog_tables", "max_catalog_fragments", "max_catalog_bytes",
           "max_fragments", "max_partitions"}
_RESOURCE_CODES = {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR,
                   sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_FULL, sqlite3.SQLITE_NOMEM}


@dataclass(frozen=True)
class CatalogProbe:
    """Counts and stored bytes only; never an identity or admission token."""

    tables: int
    catalog_fragments: int
    stored_bytes: int
    fragments: int
    partitions: int


def _check_cap(value: int, cap: int) -> None:
    if value > cap:
        raise errors.fail("RESULT_LIMIT_EXCEEDED", "catalog probe exceeds limit")


def _count(cursor, source, where, params, cap, *, projection="1"):
    value = cursor.execute(
        f"SELECT COUNT(*) FROM (SELECT {projection} FROM {source} "
        f"WHERE {where} LIMIT ?)", (*params, cap + 1)).fetchone()[0]
    _check_cap(value, cap)
    return value


def _measure(cursor, fields, source, where, params, count, total, cap, *, order=""):
    probes = []
    for token in fields.split():
        field = token.rstrip("?#")
        allowed = "'integer'" if token.endswith("#") else "'text'"
        if token.endswith("?"):
            allowed += ",'null'"
        probes.extend((f"typeof({field}) IN ({allowed})", f"length(CAST({field} AS BLOB))"))
    rows = cursor.execute(
        f"SELECT {', '.join(probes)} FROM {source} WHERE {where} {order} LIMIT ?",
        (*params, count + 1))
    seen = 0
    for row in rows:
        seen += 1
        if seen > count:
            raise errors.fail("MANIFEST_CORRUPT", "catalog probe membership changed")
        for valid, size in zip(row[::2], row[1::2], strict=True):
            if not valid:
                raise errors.fail("MANIFEST_CORRUPT", "catalog probe has invalid field or join")
            total += 0 if size is None else size
            _check_cap(total, cap)
    if seen != count:
        raise errors.fail("MANIFEST_CORRUPT", "catalog probe has missing membership")
    return total


def _validate_inputs(conn, snapshot_id, table_name, limits):
    if (not isinstance(conn, sqlite3.Connection)
            or type(snapshot_id) is not str or not snapshot_id
            or type(table_name) is not str or not table_name
            or type(limits) is not dict or set(limits) != _LIMITS
            or any(type(v) is not int or not 0 < v < sys.maxsize for v in limits.values())):
        raise errors.fail("CONTRACT_MISMATCH", "invalid catalog probe configuration")
    try:
        snapshot_id.encode("utf-8")
        table_name.encode("utf-8")
    except UnicodeEncodeError:
        raise errors.fail("CONTRACT_MISMATCH", "invalid catalog probe configuration") from None


def probe_catalog(conn: sqlite3.Connection, snapshot_id: str, table_name: str,
                  limits: dict[str, int]) -> CatalogProbe:
    """Enforce five caps; only fixed-width integers/NULL leave SQLite.

    Caller keeps its read transaction open through transfer and owns cleanup.
    These payload limits do not bound SQLite internal work.
    """
    _validate_inputs(conn, snapshot_id, table_name, limits)
    cursor = None
    try:
        if not conn.in_transaction:
            raise errors.fail("CONTRACT_MISMATCH", "catalog probe requires a read transaction")
        cursor = conn.cursor()
        cursor.row_factory = None
        if not _count(cursor, "data_snapshots", "snapshot_id=?", (snapshot_id,), 1):
            raise errors.fail("SNAPSHOT_UNRESOLVED", "catalog probe snapshot unavailable")
        where, params = "t.snapshot_id=?", (snapshot_id,)
        tables = _count(cursor, _TABLES, where, params, limits["max_catalog_tables"])
        selected_where, selected_params = where + " AND t.table_name=?", (*params, table_name)
        if not _count(cursor, _TABLES, selected_where, selected_params, 1):
            raise errors.fail("CONTRACT_MISMATCH", "catalog probe table unavailable")
        catalog_fragments = _count(cursor, _MEMBERS, where, params,
                                   limits["max_catalog_fragments"])
        fragments = _count(cursor, _MEMBERS, selected_where, selected_params,
                           limits["max_fragments"])
        partitions = _count(cursor, _FRAGMENT_JOINS, selected_where, selected_params,
                            limits["max_partitions"], projection="DISTINCT f.partition_key")
        cap = limits["max_catalog_bytes"]
        total = _measure(cursor, _SNAPSHOT, "data_snapshots s", "s.snapshot_id=?",
                         params, 1, 0, cap)
        total = _measure(cursor, _TABLE, _TABLE_JOINS, where, params, tables, total, cap,
                         order="ORDER BY t.table_name")
        # Validate the fragment's contract join without charging its payload again:
        # the fixed schedule charges contract fields once per snapshot table.
        missing = _count(cursor, _FRAGMENT_JOINS, where + " AND c.contract_id IS NULL",
                         params, limits["max_catalog_fragments"])
        if missing:
            raise errors.fail("MANIFEST_CORRUPT", "catalog probe has missing contract")
        total = _measure(cursor, _FRAGMENT, _FRAGMENT_JOINS, where, params,
                         catalog_fragments, total, cap, order="ORDER BY t.table_name, m.ordinal")
        return CatalogProbe(tables, catalog_fragments, total, fragments, partitions)
    except sqlite3.Error as exc:
        code = getattr(exc, "sqlite_errorcode", None)
        resource = isinstance(code, int) and code & 255 in _RESOURCE_CODES
        raise errors.fail("RESOURCE_UNAVAILABLE" if resource else "MANIFEST_CORRUPT",
                          "catalog probe unavailable" if resource else "catalog probe malformed") from None
    finally:
        if cursor is not None:
            cursor.close()
