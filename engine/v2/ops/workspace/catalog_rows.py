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


def _refuse(reason: str, **details):
    return fail("WORKSPACE_REFUSED", "research workspace closure copy refused",
                details={"reason": reason, **details})


def _log(message: str) -> None:
    print("[workspace] " + message, file=sys.stderr, flush=True)


def _digest(rows):
    digest = hashlib.sha256()
    count = 0
    for row in rows:
        digest.update(repr(tuple(row)).encode("utf-8"))
        count += 1
    return digest.hexdigest(), count


def _read(conn: sqlite3.Connection, table: str, where: str, sid: str):
    return conn.execute("SELECT * FROM %s WHERE %s ORDER BY 1, 2" % (table, where),
                        {"s": sid})


def _insert_rows(src, dst, sid) -> dict:
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
        batch = cursor.fetchmany(_CHUNK)
        while batch:
            dst.executemany(statement, batch)
            count += len(batch)
            batch = cursor.fetchmany(_CHUNK)
        counts[table] = count
    return counts


def _verify_rows(src, dst, sid, started) -> None:
    for table, where in _TABLES:
        source_digest, source_count = _digest(_read(src, table, where, sid))
        destination_digest, _ = _digest(_read(dst, table, where, sid))
        if source_count <= 0:
            raise _refuse("source_row_missing", table=table)
        if source_digest != destination_digest:
            raise _refuse("row_mismatch", table=table)
        destination_count = dst.execute(
            "SELECT COUNT(*) FROM %s" % table).fetchone()[0]
        if destination_count != source_count:
            raise _refuse("unexpected_rows", table=table)
        _log(f"stage=copy table={table} rows={source_count} "
             f"{time.monotonic() - started:.2f}s elapsed")


def _verify_catalog(dst) -> None:
    if dst.execute("PRAGMA foreign_key_check").fetchall():
        raise _refuse("foreign_key_violation")


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
                counts = _insert_rows(src, dst, sid)
                _verify_rows(src, dst, sid, started)
                # Identical digests above prove the rows equal the source; resolve_full cannot run inside this transaction.
                _verify_catalog(dst)
                workspace.source_snapshot()
                dst.commit()
            except BaseException:
                dst.rollback()
                raise
        finally:
            src.rollback()
        chain = [row[0] for row in dst.execute(
            "SELECT snapshot_id FROM data_snapshots WHERE snapshot_id <> ? ORDER BY 1",
            (sid,)).fetchall()]
        _log(f"stage=copy done snapshot={sid} tables={len(counts)} "
             f"{time.monotonic() - started:.2f}s elapsed")
        return {"snapshot_id": sid, "manifest_hash": pinned.manifest_hash,
                "chain": chain, "rows": counts}
    except (OpsError, data_errors.DataError):
        raise
    except Exception as exc:
        raise _refuse("storage_failure", cause=type(exc).__name__) from exc
