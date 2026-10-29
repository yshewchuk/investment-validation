"""Copy the pinned snapshot's catalog closure rows into the workspace catalog."""
from __future__ import annotations

import hashlib
import sqlite3
import sys
import time

from engine.v2.data import errors as data_errors
from engine.v2.ops.errors import OpsError, fail

_SNAPSHOTS = (
    "(WITH RECURSIVE chain(id) AS ("
    "SELECT :s UNION "
    "SELECT s.parent_snapshot_id FROM data_snapshots s JOIN chain "
    "ON s.snapshot_id = chain.id WHERE s.parent_snapshot_id IS NOT NULL) "
    "SELECT id FROM chain)"
)
_PINNED_VERSIONS = (
    "(SELECT dataset_version_id FROM data_snapshot_tables WHERE snapshot_id = :s)"
)
_ALL_VERSIONS = (
    "(WITH RECURSIVE versions(id) AS ("
    "SELECT dataset_version_id FROM data_snapshot_tables WHERE snapshot_id = :s UNION "
    "SELECT v.parent_dataset_version_id FROM data_dataset_versions v JOIN versions "
    "ON v.dataset_version_id = versions.id "
    "WHERE v.parent_dataset_version_id IS NOT NULL) SELECT id FROM versions)"
)
_FRAGMENTS = (
    "(SELECT fragment_id FROM data_version_fragments WHERE dataset_version_id IN "
    + _PINNED_VERSIONS + ")"
)

_TABLES = (
    ("data_contracts",
     "contract_id IN (SELECT contract_id FROM data_dataset_versions "
     "WHERE dataset_version_id IN " + _ALL_VERSIONS + ") OR "
     "contract_id IN (SELECT contract_id FROM data_fragments "
     "WHERE fragment_id IN " + _FRAGMENTS + ")"),
    ("data_objects",
     "object_id IN (SELECT object_id FROM data_fragments WHERE fragment_id IN "
     + _FRAGMENTS + ")"),
    ("data_fragments", "fragment_id IN " + _FRAGMENTS),
    ("data_dataset_versions", "dataset_version_id IN " + _ALL_VERSIONS),
    ("data_version_fragments", "dataset_version_id IN " + _PINNED_VERSIONS),
    ("data_snapshots", "snapshot_id IN " + _SNAPSHOTS),
    ("data_snapshot_tables", "snapshot_id = :s"),
)

_CHUNK = 5000
_BATCH_BYTES = 16 << 20
_MAX_ROWS = 8_000_000
_MAX_BYTES = 3 << 30
_REQUIRED = ("data_snapshots", "data_snapshot_tables",
             "data_dataset_versions", "data_contracts")


class _Budget:
    def __init__(self):
        self.rows = 0
        self.bytes = 0

    def add(self, table: str, row) -> None:
        self.rows += 1
        self.bytes += len(repr(tuple(row)).encode("utf-8"))
        if self.rows > _MAX_ROWS:
            raise _refuse("copy_rows_over_cap", table=table)
        if self.bytes > _MAX_BYTES:
            raise _refuse("copy_bytes_over_cap", table=table)


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace closure copy refused",
                details={"reason": reason, **details})


def _log(message: str) -> None:
    print("[workspace] " + message, file=sys.stderr, flush=True)


def _digest(rows, budget: _Budget, table: str):
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        budget.add(table, row)
        digest.update(repr(tuple(row)).encode("utf-8"))
        count += 1
    return digest.hexdigest(), count


def _read(conn: sqlite3.Connection, table: str, where: str, sid: str):
    return conn.execute("SELECT * FROM %s WHERE %s ORDER BY 1, 2" % (table, where),
                        {"s": sid})


def _insert_rows(src, dst, sid, budget: _Budget) -> dict:
    counts = {}
    for table, where in _TABLES:
        cursor = _read(src, table, where, sid)
        source_columns = tuple(column[0] for column in cursor.description)
        destination_columns = tuple(
            column[0] for column in
            dst.execute("SELECT * FROM %s LIMIT 0" % table).description)
        if source_columns != destination_columns:
            raise _refuse("schema_mismatch", table=table)
        placeholders = ",".join("?" * len(source_columns))
        statement = "INSERT OR IGNORE INTO %s VALUES (%s)" % (table, placeholders)
        count = 0
        batch = []
        batch_bytes = 0
        for row in cursor:
            budget.add(table, row)
            batch.append(row)
            batch_bytes += len(repr(tuple(row)).encode("utf-8"))
            if len(batch) >= _CHUNK or batch_bytes > _BATCH_BYTES:
                dst.executemany(statement, batch)
                count += len(batch)
                batch = []
                batch_bytes = 0
        if batch:
            dst.executemany(statement, batch)
            count += len(batch)
        counts[table] = count
    return counts


def _verify_rows(src, dst, sid, started, budget: _Budget) -> None:
    for table, where in _TABLES:
        source_digest, source_count = _digest(
            _read(src, table, where, sid), budget, table)
        if source_count <= 0 and table in _REQUIRED:
            raise _refuse("source_row_missing", table=table)
        destination_count = dst.execute(
            "SELECT COUNT(*) FROM %s" % table).fetchone()[0]
        if destination_count != source_count:
            raise _refuse("unexpected_rows", table=table)
        destination_digest, _ = _digest(
            _read(dst, table, where, sid), budget, table)
        if source_digest != destination_digest:
            raise _refuse("row_mismatch", table=table)
        _log(f"stage=copy table={table} rows={source_count} "
             f"{time.monotonic() - started:.2f}s elapsed")


def _verify_catalog(dst) -> None:
    if dst.execute("PRAGMA foreign_key_check").fetchone():
        raise _refuse("foreign_key_violation")


def _chain(dst, sid) -> list[str]:
    chain = []
    current = sid
    while True:
        row = dst.execute(
            "SELECT parent_snapshot_id FROM data_snapshots WHERE snapshot_id = ?",
            (current,)).fetchone()
        parent = row[0] if row else None
        if parent is None:
            break
        chain.append(parent)
        current = parent
    return chain


def copy_closure_rows(workspace) -> dict:
    started = time.monotonic()
    try:
        pinned = workspace.source_snapshot()
        sid = pinned.snapshot_id
        _log(f"stage=copy start snapshot={sid}")
        src, dst = workspace._source_conn, workspace.conn
        if dst.in_transaction:
            raise _refuse("workspace_invalid")
        src.execute("BEGIN")
        try:
            dst.execute("BEGIN IMMEDIATE")
            try:
                dst.execute("PRAGMA defer_foreign_keys = ON")
                budget = _Budget()
                counts = _insert_rows(src, dst, sid, budget)
                _verify_rows(src, dst, sid, started, budget)
                # Identical digests above prove the rows equal the source; resolve_full cannot run inside this transaction.
                _verify_catalog(dst)
                workspace.source_snapshot()
                chain = _chain(dst, sid)
                _log(f"stage=copy done snapshot={sid} tables={len(counts)} "
                     f"{time.monotonic() - started:.2f}s elapsed")
                dst.commit()
            except BaseException:
                dst.rollback()
                raise
        finally:
            src.rollback()
        return {"snapshot_id": sid, "manifest_hash": pinned.manifest_hash,
                "chain": chain, "rows": counts}
    except (OpsError, data_errors.DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc
