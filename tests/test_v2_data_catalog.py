"""D04 (schema half): the data-owner catalog is checksummed, idempotent,
owner-scoped, and its append-only/CAS invariants hold in real SQLite.

Phase-2 guide §6 (tables and invariants 1-8), §12 D04. This file proves the
SQL half only — object hashing, manifest construction, and idempotent-payload
comparison on a duplicate ID are later tasks (P2-2/P2-3); the duplicate-PK
test here only proves SQLite refuses a naive duplicate primary key, not the
full "same content is accepted" comparison.

Every row is built through small composable helpers in this file rather than
production insert code: ``engine/v2/data/schema.py`` deliberately contains no
Python insert/commit logic (phase-2 guide §4), so a real minimal
contract -> object -> fragment -> dataset version -> membership -> snapshot ->
snapshot-table -> head -> receipt chain is assembled by hand here, over a real
``open_catalog`` connection (never mocked).
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from engine.v2.data import schema as data_schema
from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import Migration, migrate
from tests.ops_support import FakeClock


def H(label: str) -> str:
    """A distinct, valid ``sha256:`` hash derived from ``label``."""
    return content_hash({"label": label})


def ts(clock: FakeClock) -> str:
    return format_timestamp(clock.now())


# --------------------------------------------------------------------------
# minimal valid chain: contract -> object -> fragment -> dataset version ->
# membership -> snapshot -> snapshot table -> head -> receipt
# --------------------------------------------------------------------------


def insert_contract(conn, clock, *, contract_id="legacy.daily_market.v1",
                    table_name="daily_market"):
    conn.execute(
        "INSERT INTO data_contracts (contract_id, schema_version, table_name, "
        "definition_hash, definition_json, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
        (contract_id, "table_contract.v1.0", table_name, H(contract_id),
         json.dumps({"table_name": table_name}), ts(clock)),
    )
    return contract_id


def insert_object(conn, clock, *, object_id="obj-1", byte_size=100):
    conn.execute(
        "INSERT INTO data_objects (object_id, kind, content_hash, byte_size, "
        "storage_key, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
        (object_id, "parquet", H(object_id), byte_size, f"objects/{object_id}", ts(clock)),
    )
    return object_id


def insert_fragment(conn, clock, *, fragment_id="frag-1", object_id, contract_id,
                    partition_key="2026", row_count=10, time_bounds=None):
    time_json = json.dumps(time_bounds) if time_bounds is not None else None
    conn.execute(
        "INSERT INTO data_fragments (fragment_id, object_id, contract_id, partition_key, "
        "row_count, byte_hash, logical_content_hash, key_bounds_json, time_bounds_json, "
        "import_request_hash, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (fragment_id, object_id, contract_id, partition_key, row_count, H(fragment_id + "-byte"),
         H(fragment_id + "-logical"),
         json.dumps({"primary_key_min": ["a"], "primary_key_max": ["z"]}), time_json,
         H(fragment_id + "-request"), ts(clock)),
    )
    return fragment_id


def insert_dataset_version(conn, clock, *, dataset_version_id="dsv-1", contract_id,
                           parent=None, knowledge_mode="reconstructed",
                           evidence=None, row_count=10):
    evidence = evidence if evidence is not None else {
        "coverage_receipt_refs": [], "availability_evidence_refs": [],
    }
    conn.execute(
        "INSERT INTO data_dataset_versions (dataset_version_id, contract_id, "
        "parent_dataset_version_id, manifest_hash, logical_content_hash, row_count, "
        "knowledge_mode, evidence_json, registered_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (dataset_version_id, contract_id, parent, H(dataset_version_id + "-manifest"),
         H(dataset_version_id + "-logical"), row_count, knowledge_mode, json.dumps(evidence),
         ts(clock)),
    )
    return dataset_version_id


def insert_membership(conn, *, dataset_version_id, fragment_id, ordinal=0):
    conn.execute(
        "INSERT INTO data_version_fragments (dataset_version_id, ordinal, fragment_id) "
        "VALUES (?, ?, ?)",
        (dataset_version_id, ordinal, fragment_id),
    )


def insert_snapshot(conn, clock, *, snapshot_id="snap-1", parent=None,
                    knowledge_mode_by_table=None, receipt_ref="receipt-pending"):
    modes = knowledge_mode_by_table or {"daily_market": "reconstructed"}
    conn.execute(
        "INSERT INTO data_snapshots (snapshot_id, parent_snapshot_id, manifest_hash, "
        "calendar_version, source_priority_version, finality_receipt_refs_json, "
        "knowledge_mode_by_table_json, commit_receipt_ref, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (snapshot_id, parent, H(snapshot_id + "-manifest"), "calendar.v1", "priority.v1",
         json.dumps([]), json.dumps(modes), receipt_ref, ts(clock)),
    )
    return snapshot_id


def insert_snapshot_table(conn, *, snapshot_id, table_name, dataset_version_id):
    conn.execute(
        "INSERT INTO data_snapshot_tables (snapshot_id, table_name, dataset_version_id) "
        "VALUES (?, ?, ?)",
        (snapshot_id, table_name, dataset_version_id),
    )


def insert_head(conn, clock, *, scope="shadow", snapshot_id, generation=1,
                receipt_ref="receipt-head"):
    conn.execute(
        "INSERT INTO data_snapshot_heads (scope, snapshot_id, generation, updated_at, "
        "update_receipt_ref) VALUES (?, ?, ?, ?, ?)",
        (scope, snapshot_id, generation, ts(clock), receipt_ref),
    )


def insert_receipt(conn, clock, *, receipt_id="receipt-1", attempt_id="attempt-1", fence=1,
                   status="committed", result_snapshot_id="snap-1", problem=None):
    conn.execute(
        "INSERT INTO data_import_receipts (receipt_id, attempt_id, fence, "
        "source_manifest_hash, result_snapshot_id, status, problem_json, registered_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (receipt_id, attempt_id, fence, H(receipt_id + "-source"), result_snapshot_id, status,
         json.dumps(problem) if problem is not None else None, ts(clock)),
    )
    return receipt_id


def build_chain(conn, clock, *, table_name="daily_market", knowledge_mode="reconstructed",
                evidence=None, scope="shadow"):
    """One complete, valid chain from a contract through a committed head."""
    contract_id = insert_contract(conn, clock, table_name=table_name)
    object_id = insert_object(conn, clock)
    fragment_id = insert_fragment(conn, clock, object_id=object_id, contract_id=contract_id)
    dsv_id = insert_dataset_version(conn, clock, contract_id=contract_id,
                                    knowledge_mode=knowledge_mode, evidence=evidence)
    insert_membership(conn, dataset_version_id=dsv_id, fragment_id=fragment_id)
    snapshot_id = insert_snapshot(conn, clock, knowledge_mode_by_table={table_name: knowledge_mode})
    insert_snapshot_table(conn, snapshot_id=snapshot_id, table_name=table_name,
                          dataset_version_id=dsv_id)
    receipt_id = insert_receipt(conn, clock, result_snapshot_id=snapshot_id)
    insert_head(conn, clock, scope=scope, snapshot_id=snapshot_id, receipt_ref=receipt_id)
    return {
        "contract_id": contract_id, "object_id": object_id, "fragment_id": fragment_id,
        "dataset_version_id": dsv_id, "snapshot_id": snapshot_id, "receipt_id": receipt_id,
    }


def catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    return conn, clock


# --------------------------------------------------------------------------
# bootstrap: applied once, idempotent, owner-scoped
# --------------------------------------------------------------------------


def test_bootstrap_applies_once_and_is_separate_from_ops_and_ledger(tmp_path):
    conn, clock = catalog(tmp_path)
    rows = conn.execute(
        "SELECT owner, version, name FROM schema_versions WHERE owner = 'data'"
    ).fetchall()
    assert [tuple(r) for r in rows] == [
        ("data", 1, "snapshot_catalog"), ("data", 2, "fragment_input_receipt_refs"),
        ("data", 3, "import_receipt_scope"), ("data", 4, "dataset_version_partition_hashes"),
        ("data", 5, "import_reference_inputs"), ("data", 6, "import_reference_input_fold"),
        ("data", 7, "price_captures"), ("data", 8, "receipt_lineage"),
        ("data", 9, "price_captures_contract_scope"),
        ("data", 10, "incremental_eod_controls"),
        ("data", 11, "generic_incremental_revisions"),
        ("data", 12, "computed_moves_captures"),
    ]
    ops_versions = {r[0] for r in conn.execute(
        "SELECT version FROM schema_versions WHERE owner = 'ops'")}
    ledger_versions = {r[0] for r in conn.execute(
        "SELECT version FROM schema_versions WHERE owner = 'ledger'")}
    assert ops_versions and ledger_versions  # separate sequences, both non-empty
    conn.close()

    # A second open applies nothing new: same data rows, no error.
    conn2 = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    rows2 = conn2.execute(
        "SELECT owner, version, name FROM schema_versions WHERE owner = 'data'"
    ).fetchall()
    assert [tuple(r) for r in rows2] == [tuple(r) for r in rows]
    conn2.close()


def test_edited_migration_is_a_checksum_mismatch(tmp_path):
    conn, clock = catalog(tmp_path)
    conn.close()
    # A fresh connection, re-migrated with the same version/name but edited
    # statement text: migrations.py must refuse it rather than silently
    # accept a rewrite of an applied migration. The module's own MIGRATIONS
    # tuple is never mutated — only a local copy is built here. Every other
    # applied version is passed through unedited, or `migrate` would report
    # the later real version as "newer than this code supports" before ever
    # reaching the version-1 checksum comparison this test is about.
    version, name, statements = data_schema.MIGRATIONS[0]
    edited = (Migration(version, name, statements + ("SELECT 1",)),
             *(Migration(v, n, s) for v, n, s in data_schema.MIGRATIONS[1:]))
    conn2 = sqlite3.connect(str(tmp_path / "catalog.sqlite"), isolation_level=None)
    conn2.execute("PRAGMA foreign_keys = ON")
    try:
        with pytest.raises(OpsError, match="differs from the one applied") as excinfo:
            migrate(conn2, data_schema.OWNER, edited, clock=clock)
        assert excinfo.value.problem.details["reason"] == "checksum_mismatch"
    finally:
        conn2.close()


def test_recorded_data_version_newer_than_code_is_refused(tmp_path):
    conn, clock = catalog(tmp_path)
    with pytest.raises(OpsError, match="newer than this code supports") as excinfo:
        migrate(conn, data_schema.OWNER, (), clock=clock)
    assert excinfo.value.problem.details["reason"] == "schema_newer"
    conn.close()


# --------------------------------------------------------------------------
# append-only tables: insert once, then UPDATE and DELETE both raise
# --------------------------------------------------------------------------


def test_append_only_contract(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    _assert_immutable(conn, "data_contracts", f"contract_id = '{contract_id}'",
                      "table_name = 'other'")


def test_append_only_object(tmp_path):
    conn, clock = catalog(tmp_path)
    object_id = insert_object(conn, clock)
    _assert_immutable(conn, "data_objects", f"object_id = '{object_id}'", "byte_size = 999")


def test_append_only_fragment(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    object_id = insert_object(conn, clock)
    fragment_id = insert_fragment(conn, clock, object_id=object_id, contract_id=contract_id)
    _assert_immutable(conn, "data_fragments", f"fragment_id = '{fragment_id}'", "row_count = 999")


def test_append_only_dataset_version(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    dsv_id = insert_dataset_version(conn, clock, contract_id=contract_id)
    _assert_immutable(conn, "data_dataset_versions", f"dataset_version_id = '{dsv_id}'",
                      "row_count = 999")


def test_append_only_version_fragment(tmp_path):
    conn, clock = catalog(tmp_path)
    ids = build_chain(conn, clock)
    _assert_immutable(conn, "data_version_fragments",
                      f"dataset_version_id = '{ids['dataset_version_id']}'", "ordinal = 5")


def test_append_only_snapshot(tmp_path):
    conn, clock = catalog(tmp_path)
    snapshot_id = insert_snapshot(conn, clock)
    _assert_immutable(conn, "data_snapshots", f"snapshot_id = '{snapshot_id}'",
                      "calendar_version = 'other'")


def test_append_only_snapshot_table(tmp_path):
    conn, clock = catalog(tmp_path)
    ids = build_chain(conn, clock)
    where = f"snapshot_id = '{ids['snapshot_id']}' AND table_name = 'daily_market'"
    _assert_immutable(conn, "data_snapshot_tables", where,
                      f"dataset_version_id = '{ids['dataset_version_id']}'")


def test_append_only_import_receipt(tmp_path):
    conn, clock = catalog(tmp_path)
    ids = build_chain(conn, clock)
    _assert_immutable(conn, "data_import_receipts", f"receipt_id = '{ids['receipt_id']}'",
                      "status = 'failed'")


def _assert_immutable(conn, table, where, set_clause):
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(f"UPDATE {table} SET {set_clause} WHERE {where}")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(f"DELETE FROM {table} WHERE {where}")


# --------------------------------------------------------------------------
# duplicate primary key
# --------------------------------------------------------------------------


def test_duplicate_primary_key_different_content_raises(tmp_path):
    conn, clock = catalog(tmp_path)
    insert_contract(conn, clock, contract_id="dupe", table_name="daily_market")
    with pytest.raises(sqlite3.IntegrityError):
        insert_contract(conn, clock, contract_id="dupe", table_name="trades")


# --------------------------------------------------------------------------
# negative CHECKs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bad_hash", [
    "sha256:" + "A" * 64,       # uppercase
    "sha256:" + "0" * 63,       # short
    "0" * 64,                   # no prefix
    "sha256:" + "g" * 64,       # non-hex
])
def test_malformed_hash_rejected(tmp_path, bad_hash):
    conn, clock = catalog(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO data_contracts (contract_id, schema_version, table_name, "
            "definition_hash, definition_json, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("bad", "table_contract.v1.0", "daily_market", bad_hash, "{}", ts(clock)),
        )


def test_invalid_json_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO data_contracts (contract_id, schema_version, table_name, "
            "definition_hash, definition_json, registered_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("bad", "table_contract.v1.0", "daily_market", H("bad"), "not json", ts(clock)),
        )


def test_unknown_knowledge_mode_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    with pytest.raises(sqlite3.IntegrityError):
        insert_dataset_version(conn, clock, contract_id=contract_id, knowledge_mode="guessed")


def test_observed_with_empty_evidence_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    with pytest.raises(sqlite3.IntegrityError):
        insert_dataset_version(conn, clock, contract_id=contract_id, knowledge_mode="observed",
                               evidence={"coverage_receipt_refs": [], "availability_evidence_refs": []})


def test_observed_with_evidence_accepted(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    insert_dataset_version(conn, clock, contract_id=contract_id, knowledge_mode="observed",
                           evidence={"coverage_receipt_refs": [],
                                     "availability_evidence_refs": ["receipt-a"]})


def test_committed_receipt_with_null_snapshot_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    with pytest.raises(sqlite3.IntegrityError):
        insert_receipt(conn, clock, status="committed", result_snapshot_id=None)


def test_failed_receipt_with_snapshot_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    ids = build_chain(conn, clock)
    with pytest.raises(sqlite3.IntegrityError):
        insert_receipt(conn, clock, receipt_id="receipt-2", status="failed",
                       result_snapshot_id=ids["snapshot_id"])


def test_failed_receipt_with_null_snapshot_accepted(tmp_path):
    conn, clock = catalog(tmp_path)
    insert_receipt(conn, clock, status="failed", result_snapshot_id=None)


# --------------------------------------------------------------------------
# snapshot-table trigger: bound dataset version must match its own table_name
# --------------------------------------------------------------------------


def test_snapshot_table_contract_mismatch_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    trades_contract = insert_contract(conn, clock, contract_id="legacy.trades.v1", table_name="trades")
    object_id = insert_object(conn, clock)
    fragment_id = insert_fragment(conn, clock, fragment_id="frag-trades", object_id=object_id,
                                  contract_id=trades_contract)
    dsv_id = insert_dataset_version(conn, clock, dataset_version_id="dsv-trades",
                                    contract_id=trades_contract)
    insert_membership(conn, dataset_version_id=dsv_id, fragment_id=fragment_id)
    snapshot_id = insert_snapshot(conn, clock, knowledge_mode_by_table={"daily_market": "reconstructed"})
    with pytest.raises(sqlite3.IntegrityError, match="does not match"):
        insert_snapshot_table(conn, snapshot_id=snapshot_id, table_name="daily_market",
                              dataset_version_id=dsv_id)


# --------------------------------------------------------------------------
# head: insert-at-1, CAS-shaped update, delete refused
# --------------------------------------------------------------------------


def test_head_insert_at_generation_two_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    ids = build_chain(conn, clock, scope="unused")
    with pytest.raises(sqlite3.IntegrityError):
        insert_head(conn, clock, scope="other", snapshot_id=ids["snapshot_id"], generation=2)


def test_head_update_to_next_generation_succeeds(tmp_path):
    conn, clock = catalog(tmp_path)
    first = build_chain(conn, clock, scope="shadow")
    second_snapshot = insert_snapshot(conn, clock, snapshot_id="snap-2",
                                      parent=first["snapshot_id"])
    conn.execute(
        "UPDATE data_snapshot_heads SET snapshot_id = ?, generation = 2, updated_at = ?, "
        "update_receipt_ref = ? WHERE scope = 'shadow' AND snapshot_id = ? AND generation = 1",
        (second_snapshot, ts(clock), "receipt-2", first["snapshot_id"]),
    )
    row = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'"
    ).fetchone()
    assert tuple(row) == (second_snapshot, 2)


def test_head_update_skipping_generation_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    build_chain(conn, clock, scope="shadow")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE data_snapshot_heads SET generation = 3, updated_at = ? WHERE scope = 'shadow'",
            (ts(clock),))


def test_head_update_reusing_generation_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    build_chain(conn, clock, scope="shadow")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE data_snapshot_heads SET generation = 1, updated_at = ? WHERE scope = 'shadow'",
            (ts(clock),))


def test_head_update_changing_scope_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    build_chain(conn, clock, scope="shadow")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE data_snapshot_heads SET scope = 'other', generation = 2, updated_at = ? "
            "WHERE scope = 'shadow'", (ts(clock),))


def test_head_update_missing_snapshot_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    build_chain(conn, clock, scope="shadow")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "UPDATE data_snapshot_heads SET snapshot_id = 'no-such-snapshot', generation = 2, "
            "updated_at = ? WHERE scope = 'shadow'", (ts(clock),))


def test_head_delete_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    build_chain(conn, clock, scope="shadow")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("DELETE FROM data_snapshot_heads WHERE scope = 'shadow'")


# --------------------------------------------------------------------------
# foreign keys
# --------------------------------------------------------------------------


def test_fragment_unknown_object_fk_rejected(tmp_path):
    conn, clock = catalog(tmp_path)
    contract_id = insert_contract(conn, clock)
    with pytest.raises(sqlite3.IntegrityError):
        insert_fragment(conn, clock, object_id="no-such-object", contract_id=contract_id)
