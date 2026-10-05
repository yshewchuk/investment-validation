"""Regression for the data-owner v13 ``data_normalizations`` recreate.

Issue #133's contract (``engine/v2/data/ARCHITECTURE.md``): v13 rebuilds the
table through the migration framework's opt-in enforcement-off recreate and
removes exactly v10's ``UNIQUE (raw_hash, normalizer_id, contract_id)``, so
normalization values, the sole ``normalization_id`` primary key, the
``contract_id`` FK and committed ``data_daily_market_revisions`` child
references all survive. The catalog is a real disk-backed file built through
v12, seeded with a committed parent/child pair, then reopened so the real
``migrate`` applies the pending production step; this test fails if v13 is
absent or its schema behaviour is wrong.
"""
from __future__ import annotations

import sqlite3

import pytest

from engine.v2.data import schema as data_schema
from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ops.bootstrap import _DATA_MIGRATIONS
from engine.v2.ops.catalog import connect, transaction
from engine.v2.ops.migrations import applied_versions, checksum, migrate
from tests.ops_support import FakeClock
from tests.test_v2_ops_migrations import _seed_parent_and_child

_UNIQUE_TRIPLE = frozenset({"raw_hash", "normalizer_id", "contract_id"})


def _unique_index_column_sets(conn, table: str) -> set[frozenset]:
    """The column set of every UNIQUE index/constraint on ``table``."""
    return {frozenset(row[2] for row in conn.execute(f"PRAGMA index_info({name})"))
            for _seq, name, unique, _origin, _partial
            in conn.execute(f"PRAGMA index_list({table})") if unique}


def test_v13_recreate_drops_only_the_unique_triple(tmp_path):
    v13 = next((m for m in _DATA_MIGRATIONS if m.version == 13), None)
    assert v13 is not None, ("the data owner has no version-13 migration: the "
                             "planned data_normalizations recreate (#133) is "
                             "missing")
    assert v13.recreate_tables, "data v13 must opt into the recreate procedure"
    clock = FakeClock()
    path = tmp_path / "catalog.sqlite"

    setup = connect(path)
    assert migrate(setup, data_schema.OWNER,
                   tuple(m for m in _DATA_MIGRATIONS if m.version < 13),
                   clock=clock) == list(range(1, 13))
    with transaction(setup):
        child_before = tuple(_seed_parent_and_child(setup, clock))
        norm_before = tuple(setup.execute("SELECT * FROM data_normalizations"
                                          " WHERE normalization_id = 'n-1'").fetchone())
    setup.close()

    conn = connect(path)  # reopened: production v13 is the only pending step
    assert migrate(conn, data_schema.OWNER, _DATA_MIGRATIONS, clock=clock) == [13]
    assert applied_versions(conn, data_schema.OWNER)[13] == checksum(v13)
    assert tuple(conn.execute("SELECT * FROM data_normalizations"
                              " WHERE normalization_id = 'n-1'").fetchone()) == norm_before
    assert tuple(conn.execute("SELECT * FROM data_daily_market_revisions"
                              " WHERE revision_id = 'rev-1'").fetchone()) == child_before
    assert [row["name"] for row in conn.execute("PRAGMA table_info(data_normalizations)")
            if row["pk"]] == ["normalization_id"]  # sole primary key
    assert _UNIQUE_TRIPLE not in _unique_index_column_sets(conn, "data_normalizations")
    ts = format_timestamp(clock.now())
    with pytest.raises(sqlite3.IntegrityError):  # contract_id FK intact and enforced
        conn.execute("INSERT INTO data_normalizations (normalization_id, raw_hash,"
                     " normalizer_id, contract_id, normalized_hash,"
                     " artifact_ref_json, row_count, created_at) VALUES"
                     " ('n-2', ?, 'norm-2', 'missing-contract', ?, '{}', 1, ?)",
                     (content_hash({"v13": "raw-2"}), content_hash({"v13": "hash-2"}), ts))
    with pytest.raises(sqlite3.IntegrityError):  # child FK to normalization_id intact
        conn.execute("INSERT INTO data_daily_market_revisions (revision_id,"
                     " import_receipt_id, raw_receipt_id, normalization_id, ticker,"
                     " session_date, source, source_priority, finality_rank,"
                     " revision_number, deleted, row_hash, created_at) VALUES"
                     " ('rev-2', 'r-1', 'rr-1', 'missing', 'BBB', '2026-01-06',"
                     " 'src', 0, 1, 0, 0, ?, ?)",
                     (content_hash({"v13": "row-2"}), ts))
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.close()
