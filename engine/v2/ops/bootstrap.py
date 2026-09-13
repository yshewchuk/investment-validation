"""Open a catalog: connect with verified pragmas, then migrate every owner.

A catalog is never used at a schema version this code did not create: newer
schemas and edited migrations are refused by :mod:`engine.v2.ops.migrations`
before any job table is read.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from engine.v2.foundation import Clock
from engine.v2.ledger.decisions import SCHEMA
from engine.v2.ops import schema
from engine.v2.ops.catalog import connect
from engine.v2.ops.migrations import Migration, migrate

__all__ = ["open_catalog"]


def open_catalog(path: Path | str, *, clock: Clock) -> sqlite3.Connection:
    """A migrated, ready catalog connection."""
    conn = connect(path)
    try:
        migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock)
        migrate(conn, "ledger", (Migration(1, "decision_authority", SCHEMA),), clock=clock)
    except BaseException:
        conn.close()
        raise
    return conn
