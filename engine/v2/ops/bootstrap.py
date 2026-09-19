"""Open a catalog: connect with verified pragmas, then migrate every owner.

A catalog is never used at a schema version this code did not create: newer
schemas and edited migrations are refused by :mod:`engine.v2.ops.migrations`
before any job table is read.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from engine.v2.data import schema as data_schema
from engine.v2.foundation import Clock
from engine.v2.ledger.calibration import SCHEMA_V3
from engine.v2.ledger.decisions import SCHEMA, SCHEMA_V2
from engine.v2.ops import schema
from engine.v2.ops.catalog import connect
from engine.v2.ops.migrations import Migration, migrate

__all__ = ["open_catalog"]

#: engine.v2.data.schema declares its migrations as plain tuples, never
#: engine.v2.ops.migrations.Migration objects (phase-2 guide §3.3: the data
#: package never imports ops). This is the one place that wraps them, on
#: ops's higher layer, before handing them to the existing migration
#: machinery — the same shape submit()/etc. already expect.
_DATA_MIGRATIONS = tuple(
    Migration(version, name, statements) for version, name, statements in data_schema.MIGRATIONS
)


def open_catalog(path: Path | str, *, clock: Clock) -> sqlite3.Connection:
    """A migrated, ready catalog connection."""
    conn = connect(path)
    try:
        migrate(conn, schema.OWNER, schema.MIGRATIONS, clock=clock)
        migrate(conn, "ledger", (Migration(1, "decision_authority", SCHEMA),
                                 Migration(2, "decision_generation_divergence", SCHEMA_V2),
                                 Migration(3, "calibration_state", SCHEMA_V3)),
               clock=clock)
        migrate(conn, data_schema.OWNER, _DATA_MIGRATIONS, clock=clock)
    except BaseException:
        conn.close()
        raise
    return conn
