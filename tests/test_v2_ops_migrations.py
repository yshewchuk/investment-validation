"""Focused tests for the migration framework's table-recreate slice (issue #133).

Covers ``engine/v2/ops/ARCHITECTURE.md``'s R1-R6 contract for the opt-in
``Migration.recreate_tables`` flag: enforcement off before ``BEGIN
IMMEDIATE``, ``PRAGMA foreign_key_check`` inside the transaction before
commit, typed non-retryable ``INTEGRITY_FAILED`` with a full rollback on any
violation, enforcement restored on every path, and the flag being
checksum-protected while historical unflagged checksums stay unchanged.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import replace

import pytest

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ops.bootstrap import _DATA_MIGRATIONS, _data_migration
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import Migration, applied_versions, checksum, migrate
from tests.ops_support import FakeClock, catalog


def _H(label: str) -> str:
    """A distinct, valid ``sha256:`` hash derived from ``label``."""
    return content_hash({"probe": label})


def _catalog_snapshot(conn) -> bytes:
    """Byte snapshot of the connection's committed catalog state, WAL
    included: SQLite's backup API copies the live committed pages (including
    any committed WAL frames) into a temporary in-memory database, whose
    ``serialize()`` bytes are returned; the destination is always closed."""
    dest = sqlite3.connect(":memory:")
    try:
        conn.backup(dest)
        return dest.serialize()
    finally:
        dest.close()


#: An orphaned ``job_dependencies`` row: both sides dangle, so the insert is
#: only accepted with enforcement off (and only caught by the fk check then).
_ORPHAN_INSERT = ("INSERT INTO job_dependencies (child_job_id, parent_job_id,"
                 " required_output_contract) VALUES ('orphan-child',"
                 " 'ghost-parent', 'rows.v1.0')")

#: The table-recreate dance for ``data_normalizations`` (parent) while the
#: committed ``data_daily_market_revisions`` child must not be touched.
_RECREATE_NORMALIZATIONS = (
    """CREATE TABLE data_normalizations_rebuild (
        normalization_id TEXT PRIMARY KEY,
        raw_hash TEXT NOT NULL,
        normalizer_id TEXT NOT NULL,
        contract_id TEXT NOT NULL REFERENCES data_contracts(contract_id),
        normalized_hash TEXT NOT NULL,
        artifact_ref_json TEXT NOT NULL,
        row_count INTEGER NOT NULL CHECK (row_count >= 0),
        created_at TEXT NOT NULL,
        UNIQUE (raw_hash, normalizer_id, contract_id)
    ) STRICT""",
    "INSERT INTO data_normalizations_rebuild SELECT normalization_id, raw_hash,"
    " normalizer_id, contract_id, normalized_hash, artifact_ref_json, row_count,"
    " created_at FROM data_normalizations",
    "DROP TABLE data_normalizations",
    "ALTER TABLE data_normalizations_rebuild RENAME TO data_normalizations",
)


def _seed_parent_and_child(conn, clock):
    """One committed ``data_normalizations`` parent and one committed
    ``data_daily_market_revisions`` child referencing it; returns the child
    row as committed."""
    ts = format_timestamp(clock.now())
    conn.execute("INSERT INTO data_contracts (contract_id, schema_version, table_name,"
                 " definition_hash, definition_json, registered_at) VALUES"
                 " ('c-1', 'table_contract.v1.0', 'daily_market', ?, '{}', ?)",
                 (_H("contract-1"), ts))
    conn.execute("INSERT INTO data_import_receipts (receipt_id, attempt_id, fence,"
                 " source_manifest_hash, result_snapshot_id, status, problem_json,"
                 " registered_at) VALUES ('r-1', 'a-1', 1, ?, NULL, 'failed', NULL, ?)",
                 (_H("receipt-1"), ts))
    conn.execute("INSERT INTO data_raw_receipts (raw_receipt_id, source, endpoint,"
                 " request_hash, raw_hash, artifact_ref_json, response_kind, request_json,"
                 " response_meta_json, received_at) VALUES"
                 " ('rr-1', 'src', '/e', ?, ?, '{}', 'complete', '{}', '{}', ?)",
                 (_H("request-1"), _H("raw-1"), ts))
    conn.execute("INSERT INTO data_normalizations (normalization_id, raw_hash,"
                 " normalizer_id, contract_id, normalized_hash, artifact_ref_json,"
                 " row_count, created_at) VALUES ('n-1', ?, 'norm-1', 'c-1', ?, '{}',"
                 " 1, ?)", (_H("norm-raw-1"), _H("normalized-1"), ts))
    conn.execute("INSERT INTO data_daily_market_revisions (revision_id,"
                 " import_receipt_id, raw_receipt_id, normalization_id, ticker,"
                 " session_date, source, source_priority, finality_rank, revision_number,"
                 " deleted, row_hash, created_at) VALUES"
                 " ('rev-1', 'r-1', 'rr-1', 'n-1', 'AAA', '2026-01-05', 'src', 0, 1,"
                 " 0, 0, ?, ?)", (_H("row-1"), ts))
    return conn.execute("SELECT * FROM data_daily_market_revisions").fetchone()


def test_recreate_migration_preserves_child_rows_and_fk(tmp_path):
    """(a) A real catalog with a committed parent and referencing child
    survives an opted-in recreate of the parent: the child row is untouched,
    ``PRAGMA foreign_key_check`` is empty afterwards, and enforcement is
    restored to ON on the success path (R5) — a dangling child insert still
    fails the ordinary way."""
    conn, clock, _ = catalog(tmp_path)
    child_before = tuple(_seed_parent_and_child(conn, clock))
    migration = Migration(1, "recreate_normalizations", _RECREATE_NORMALIZATIONS,
                          recreate_tables=True)
    assert migrate(conn, "test", [migration], clock=clock) == [1]
    assert tuple(conn.execute(
        "SELECT * FROM data_daily_market_revisions").fetchone()) == child_before
    assert conn.execute("SELECT row_count FROM data_normalizations WHERE"
                        " normalization_id = 'n-1'").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert applied_versions(conn, "test") == {1: checksum(migration)}
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO data_daily_market_revisions (revision_id,"
                     " import_receipt_id, raw_receipt_id, normalization_id, ticker,"
                     " session_date, source, source_priority, finality_rank,"
                     " revision_number, deleted, row_hash, created_at) VALUES"
                     " ('rev-2', 'r-1', 'rr-1', 'missing', 'BBB', '2026-01-06', 'src',"
                     " 0, 1, 0, 0, ?, ?)", (_H("row-2"), format_timestamp(clock.now())))


def test_fk_violation_is_typed_and_rolls_back_everything(tmp_path):
    """(b) A recreate-flagged migration that leaves a dangling FK is refused
    with a non-retryable ``INTEGRITY_FAILED``, no row values in public
    details, and the serialized catalog snapshot byte-identical afterwards:
    each snapshot is a ``backup()`` copy of the connection's committed state
    into an in-memory database, so committed WAL frames are part of the
    compared bytes: no DDL/data and no ``schema_versions`` row survived
    (R3/R4), and enforcement is back ON (R5)."""
    conn, clock, _ = catalog(tmp_path)
    baseline = _catalog_snapshot(conn)
    migration = Migration(1, "orphan_dependency", (_ORPHAN_INSERT,),
                          recreate_tables=True)
    with pytest.raises(OpsError) as err:
        migrate(conn, "test", [migration], clock=clock)
    problem = err.value.problem
    assert err.value.code == "INTEGRITY_FAILED"
    assert problem.retryable is False
    assert problem.details == {"reason": "foreign_key_check_failed", "owner": "test",
                               "version": 1, "tables": ["job_dependencies"],
                               "violation_count": 1}
    public = problem.message + json.dumps(problem.details) + str(err.value)
    assert "ghost-parent" not in public and "orphan-child" not in public
    assert _catalog_snapshot(conn) == baseline
    assert conn.execute("SELECT COUNT(*) FROM schema_versions"
                        " WHERE owner = 'test'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM job_dependencies").fetchone()[0] == 0
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_failing_first_migration_leaves_fresh_catalog_byte_identical(tmp_path):
    """First-time creation of ``schema_versions`` is rolled back with the
    first failed migration: a failing initial migration on a fresh file
    leaves the serialized catalog snapshot byte-identical, each snapshot a
    ``backup()`` copy of the connection's committed state into an in-memory
    database so committed WAL frames are part of the compared bytes, table
    not even created."""
    clock = FakeClock()
    path = tmp_path / "first.sqlite"
    conn = connect(path)
    baseline = _catalog_snapshot(conn)
    migration = Migration(1, "dangling_child", (
        "CREATE TABLE probe_parent (pid TEXT PRIMARY KEY) STRICT",
        "CREATE TABLE probe_child (cid TEXT PRIMARY KEY, pid TEXT NOT NULL"
        " REFERENCES probe_parent(pid)) STRICT",
        "INSERT INTO probe_parent VALUES ('p1')",
        "INSERT INTO probe_child VALUES ('c1', 'gone')",
        "DROP TABLE probe_parent",
    ), recreate_tables=True)
    with pytest.raises(OpsError) as err:
        migrate(conn, "test", [migration], clock=clock)
    assert err.value.code == "INTEGRITY_FAILED"
    assert _catalog_snapshot(conn) == baseline
    assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'schema_versions'"
                        " OR name LIKE 'probe_%'").fetchone() is None
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_ordinary_migration_runs_enforced_and_skips_the_fk_check(tmp_path):
    """R1 + requirement 4: an unflagged pending migration runs with
    enforcement ON — a dangling insert is refused by SQLite itself
    (``IntegrityError``, not the framework's ``OpsError``, so the
    ``foreign_key_check`` step never ran) — and still rolls back."""
    conn, clock, _ = catalog(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        migrate(conn, "test", [Migration(1, "plain", (_ORPHAN_INSERT,))], clock=clock)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM job_dependencies").fetchone()[0] == 0


def test_historical_checksums_stay_identical_and_the_flag_is_protected():
    """Unflagged migrations hash over exactly the old payload, so every
    historical checksum is unchanged; a true flag joins the payload, so
    toggling it changes the checksum."""
    plain = Migration(3, "three", ("A", "B"))
    assert checksum(plain) == content_hash({"version": 3, "name": "three",
                                            "statements": ["A", "B"]})
    flagged = replace(plain, recreate_tables=True)
    assert checksum(flagged) != checksum(plain)
    assert checksum(flagged) == content_hash({"version": 3, "name": "three",
                                              "statements": ["A", "B"],
                                              "recreate_tables": True})


def test_toggling_the_flag_on_an_applied_migration_is_refused(tmp_path):
    """Both directions of the toggle on an APPLIED migration are refused as
    checksum mismatches, never silently accepted or re-run."""
    clock = FakeClock()
    plain = Migration(1, "probe", ("CREATE TABLE probe(v INTEGER) STRICT",))
    conn = connect(tmp_path / "toggle.sqlite")
    migrate(conn, "test", [plain], clock=clock)
    with pytest.raises(OpsError) as flagged_after_plain:
        migrate(conn, "test", [replace(plain, recreate_tables=True)], clock=clock)
    assert flagged_after_plain.value.problem.details["reason"] == "checksum_mismatch"
    conn.close()

    other = connect(tmp_path / "toggle2.sqlite")
    migrate(other, "test", [replace(plain, recreate_tables=True)], clock=clock)
    with pytest.raises(OpsError) as plain_after_flagged:
        migrate(other, "test", [plain], clock=clock)
    assert plain_after_flagged.value.problem.details["reason"] == "checksum_mismatch"
    other.close()


def test_each_migration_stays_its_own_transaction_and_a_fix_retries(tmp_path):
    """R4/R6: a failing flagged migration 2 does not undo applied migration
    1; correcting the still-pending migration and re-running applies it."""
    conn, clock, _ = catalog(tmp_path)
    first = Migration(1, "first", ("CREATE TABLE step_one(v INTEGER) STRICT",))
    broken = Migration(2, "broken", (_ORPHAN_INSERT,), recreate_tables=True)
    with pytest.raises(OpsError):
        migrate(conn, "test", [first, broken], clock=clock)
    assert applied_versions(conn, "test") == {1: checksum(first)}
    assert conn.execute("SELECT COUNT(*) FROM job_dependencies").fetchone()[0] == 0
    fixed = Migration(2, "broken", ("CREATE TABLE step_two(v INTEGER) STRICT",),
                      recreate_tables=True)
    assert migrate(conn, "test", [first, fixed], clock=clock) == [2]
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_data_migration_tuples_pass_the_flag_through_bootstrap():
    """Bootstrap keeps wrapping plain 3-tuples (historical checksums intact,
    versions before 13 never flagged) and passes an optional 4th element
    through for the v13 recreate opt-in."""
    plain = _data_migration((1, "n", ("A",)))
    assert plain.recreate_tables is False
    assert checksum(plain) == checksum(Migration(1, "n", ("A",)))
    assert _data_migration((1, "n", ("A",), True)).recreate_tables is True
    assert _data_migration((1, "n", ("A",), False)).recreate_tables is False
    assert all(not m.recreate_tables for m in _DATA_MIGRATIONS if m.version < 13)
    assert [m.version for m in _DATA_MIGRATIONS if m.recreate_tables] == [13]
