"""Explicit, checksummed schema migrations, one sequence per owner — §6.1, O32.

Each migration is **one transaction**. SQLite DDL is transactional, so a
migration interrupted at any statement — by an exception or by the process
being killed — leaves the previous version intact rather than half a schema.

Two refusals keep an upgrade honest:

* a schema **newer** than this code is refused, because older code writing to a
  table it does not understand is how a column's meaning gets lost;
* a recorded migration whose **checksum** differs from the code's copy is
  refused. An edited migration is a new migration, never a rewrite of an
  applied one.

Owners share one ``schema_versions`` table and keep separate sequences. The
ledger's decision tables live in the same SQLite file as the job tables and
keep their own migrations, so ownership survives colocation (§4).
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from engine.v2.foundation import Clock, content_hash, format_timestamp
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import fail

__all__ = ["Migration", "applied_versions", "checksum", "migrate", "require_current"]

_VERSIONS_DDL = """CREATE TABLE IF NOT EXISTS schema_versions (
    owner TEXT NOT NULL,
    version INTEGER NOT NULL CHECK (version > 0),
    name TEXT NOT NULL,
    checksum TEXT NOT NULL,
    applied_at TEXT NOT NULL,
    PRIMARY KEY (owner, version)
) STRICT"""


@dataclass(frozen=True)
class Migration:
    """One numbered schema step: its statements run in one transaction."""

    version: int
    name: str
    statements: tuple[str, ...]


def checksum(migration: Migration) -> str:
    return content_hash({"version": migration.version, "name": migration.name,
                         "statements": list(migration.statements)})


def applied_versions(conn: sqlite3.Connection, owner: str) -> dict[int, str]:
    """``{version: checksum}`` already recorded for ``owner``."""
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' "
                          "AND name = 'schema_versions'").fetchone()
    if exists is None:
        return {}
    rows = conn.execute("SELECT version, checksum FROM schema_versions WHERE owner = ?",
                        (owner,))
    return {int(row[0]): str(row[1]) for row in rows}


def migrate(conn: sqlite3.Connection, owner: str, migrations: Sequence[Migration], *,
            clock: Clock, fault: Callable[[str], None] | None = None) -> list[int]:
    """Apply every pending migration for ``owner``; return the versions applied now."""
    _validate_sequence(owner, migrations)
    with transaction(conn):
        conn.execute(_VERSIONS_DDL)
        _check_applied(owner, applied_versions(conn, owner), migrations)
    done: list[int] = []
    for migration in migrations:
        with transaction(conn):
            applied = applied_versions(conn, owner)
            _check_applied(owner, applied, migrations)
            if migration.version not in applied:
                _apply(conn, owner, migration, clock, fault)
                done.append(migration.version)
    return done


def require_current(conn: sqlite3.Connection, owner: str,
                    migrations: Sequence[Migration]) -> None:
    """Refuse to operate on a schema that is older or newer than this code."""
    applied = applied_versions(conn, owner)
    _check_applied(owner, applied, migrations)
    missing = [m.version for m in migrations if m.version not in applied]
    if missing:
        raise fail("INTEGRITY_FAILED", f"{owner} schema is missing migrations",
                   details={"reason": "schema_older", "owner": owner, "missing": missing})


def _validate_sequence(owner: str, migrations: Sequence[Migration]) -> None:
    versions = [m.version for m in migrations]
    if versions != list(range(1, len(versions) + 1)):
        raise ValueError(f"{owner} migrations must be numbered 1..n in order, got {versions}")


def _check_applied(owner: str, applied: dict[int, str],
                   migrations: Sequence[Migration]) -> None:
    known = {m.version: checksum(m) for m in migrations}
    newer = sorted(v for v in applied if v not in known)
    if newer:
        raise fail("INTEGRITY_FAILED", f"{owner} schema is newer than this code supports",
                   details={"reason": "schema_newer", "owner": owner,
                            "applied": newer[-1], "supported": len(known)})
    for version, found in sorted(applied.items()):
        if known[version] != found:
            raise fail("INTEGRITY_FAILED", f"{owner} migration differs from the one applied",
                       details={"reason": "checksum_mismatch", "owner": owner,
                                "version": version})


def _apply(conn: sqlite3.Connection, owner: str, migration: Migration, clock: Clock,
           fault: Callable[[str], None] | None) -> None:
    for index, statement in enumerate(migration.statements):
        conn.execute(statement)
        if fault is not None:
            fault(f"{owner}:{migration.version}:{index}")
    conn.execute(
        "INSERT INTO schema_versions (owner, version, name, checksum, applied_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (owner, migration.version, migration.name, checksum(migration),
         format_timestamp(clock.now())))
