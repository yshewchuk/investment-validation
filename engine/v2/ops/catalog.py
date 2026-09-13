"""The operations catalog connection and its one transaction shape — phase-1 guide §6.1.

Stdlib SQLite on a local filesystem. Every connection enables foreign keys,
WAL, a bounded busy timeout and FULL synchronization, and a connection that
cannot is refused rather than used half-configured.

Transactions are ``BEGIN IMMEDIATE``: the write lock is taken at the start, so
two claimers serialize at BEGIN instead of both reading a stale reservation
table. They are short by contract — no network call, fit or long file write
happens inside one (contracts §11.2). Nesting is refused, because a nested
block that silently committed the outer work is exactly how an effect escapes
its fence.

Backups use the SQLite online backup API. Copying only the main database file
while WAL holds committed pages is not a consistent backup (§6.1), and
``tests/test_v2_ops_catalog.py`` demonstrates the rows it loses.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from engine.v2.foundation import canonical_json, from_document, to_document
from engine.v2.ops.errors import fail

__all__ = [
    "BUSY_TIMEOUT_MS",
    "backup_to",
    "connect",
    "dumps",
    "integrity_errors",
    "load_json",
    "transaction",
]

BUSY_TIMEOUT_MS = 5_000
_SYNC_FULL = 2


def connect(path: Path | str, *, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> sqlite3.Connection:
    """Open the catalog with its required pragmas, verified rather than assumed."""
    conn = sqlite3.connect(str(path), isolation_level=None, timeout=busy_timeout_ms / 1000.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
        mode = conn.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        conn.execute("PRAGMA synchronous = FULL")
        foreign = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous").fetchone()[0]
    except BaseException:
        conn.close()
        raise
    if str(mode).lower() != "wal" or foreign != 1 or sync != _SYNC_FULL:
        conn.close()
        raise fail("INTEGRITY_FAILED",
                   "catalog connection could not enable WAL, foreign keys and FULL sync")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """One short immediate transaction; commit on success, roll back on anything else."""
    if conn.in_transaction:
        raise RuntimeError("nested catalog transaction: effects commit in the caller's transaction")
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def backup_to(conn: sqlite3.Connection, dest: Path | str) -> None:
    """A consistent copy through the online backup API, integrity-checked."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(dest)
    target = sqlite3.connect(str(dest))
    try:
        conn.backup(target)
        problems = integrity_errors(target)
    finally:
        target.close()
    if problems:
        raise fail("INTEGRITY_FAILED", "catalog backup failed its integrity check",
                   details={"first_problem": problems[0]})


def integrity_errors(conn: sqlite3.Connection) -> list[str]:
    """``PRAGMA integrity_check`` and ``foreign_key_check`` findings; empty when sound."""
    rows = [row[0] for row in conn.execute("PRAGMA integrity_check")]
    keys = conn.execute("PRAGMA foreign_key_check").fetchall()
    return [r for r in rows if r != "ok"] + [f"foreign key violation in {k[0]}" for k in keys]


def dumps(value: Any) -> str:
    """Canonical JSON of a contract value, for a JSON column."""
    return canonical_json(to_document(value))


def load_json(cls: type, text: str | None) -> Any:
    """Strictly decode a JSON column back into its contract type."""
    return None if text is None else from_document(cls, json.loads(text))
