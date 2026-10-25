"""Synthetic SQL only: the probe deliberately does not authenticate identities."""
import sqlite3
import sys

import pytest

from engine.v2.data import _bounded_catalog as probes
from engine.v2.data.errors import DataError

LIMITS = dict(max_catalog_tables=2, max_catalog_fragments=4, max_catalog_bytes=10_000,
              max_fragments=2, max_partitions=2)
# Independent schema/rows, intentionally without constraints so corruption can be
# exercised. Two bindings reuse the same dataset; two fragments reuse one object.
ROWS = {
    "snapshots": dict(snapshot_id="s", parent_snapshot_id=None, manifest_hash="h",
                      calendar_version="cal", source_priority_version="priority",
                      finality_receipt_refs_json="[]", knowledge_mode_by_table_json="{}"),
    "snapshot_tables": dict(snapshot_id="s", table_name="a", dataset_version_id="d"),
    "dataset_versions": dict(dataset_version_id="d", contract_id="c", parent_dataset_version_id=None,
                             manifest_hash="h", logical_content_hash="l", row_count=0,
                             knowledge_mode="reconstructed", evidence_json="{}",
                             partition_logical_hashes_json="{}"),
    "contracts": dict(contract_id="c", schema_version="v", table_name="a", definition_hash="h",
                      definition_json="{}"),
    "version_fragments": dict(dataset_version_id="d", ordinal=0, fragment_id="f1"),
    "fragments": dict(fragment_id="f1", object_id="o", contract_id="c", partition_key="p1",
                      row_count=0, byte_hash="b", logical_content_hash="l", key_bounds_json="{}",
                      time_bounds_json=None, import_request_hash="i", input_receipt_refs_json="[]"),
    "objects": dict(object_id="o", kind="k", content_hash="h", byte_size=0),
}

class ScalarCursor(sqlite3.Cursor):
    def _scalar(self, row):
        assert row is None or all(type(v) is int or v is None for v in row)
        return row

    def fetchone(self):
        return self._scalar(super().fetchone())

    def __next__(self):
        return self._scalar(super().__next__())

    def fetchall(self):
        pytest.fail("probe attempted bulk transfer")


class ScalarConnection(sqlite3.Connection):
    def cursor(self):
        return super().cursor(factory=ScalarCursor)


def insert(conn, table, row):
    conn.execute(f"INSERT INTO data_{table} ({','.join(row)}) VALUES "
                 f"({','.join('?' for _ in row)})", tuple(row.values()))


def populate(conn):
    for table, row in ROWS.items():
        conn.execute(f"CREATE TABLE data_{table} ({','.join(row)})")
        insert(conn, table, row)
    insert(conn, "snapshot_tables", {**ROWS["snapshot_tables"], "table_name": "z"})
    insert(conn, "version_fragments", {**ROWS["version_fragments"], "ordinal": 1, "fragment_id": "f2"})
    insert(conn, "fragments", {**ROWS["fragments"], "fragment_id": "f2", "partition_key": "p2"})
    conn.commit()
    conn.execute("BEGIN")


@pytest.fixture
def catalog():
    conn = sqlite3.connect(":memory:", factory=ScalarConnection)
    populate(conn)
    yield conn
    conn.close()


def probe(conn, **changes):
    return probes.probe_catalog(conn, "s", "a", {**LIMITS, **changes})


def refused(conn, code, **changes):
    with pytest.raises(DataError) as caught:
        probe(conn, **changes)
    assert caught.value.code == code
    assert conn.in_transaction
    return caught.value


def test_exact_counts_and_byte_schedule(catalog):
    result = probe(catalog)
    # 17 snapshot + 31/table + 22/membership; zero-row fragments still count.
    assert result == probes.CatalogProbe(2, 4, 167, 2, 2)
    assert probe(catalog, max_catalog_bytes=167) == result
    refused(catalog, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=166)


@pytest.mark.parametrize("cap", ["max_catalog_tables", "max_catalog_fragments",
                                "max_fragments", "max_partitions"])
def test_each_count_one_over(catalog, cap):
    refused(catalog, "RESULT_LIMIT_EXCEEDED", **{cap: LIMITS[cap] - 1})


@pytest.mark.parametrize("field", ["evidence_json", "dataset_version_id"])
def test_unrequested_huge_metadata_and_ids_never_transfer(catalog, field):
    unrelated = {**ROWS["dataset_versions"], "dataset_version_id": "other", "contract_id": "c"}
    unrelated[field] = "雪\0" * 100_000
    insert(catalog, "dataset_versions", unrelated)
    catalog.execute("UPDATE data_snapshot_tables SET dataset_version_id=? WHERE table_name='z'",
                    (unrelated["dataset_version_id"],))
    catalog.text_factory = lambda _: pytest.fail("TEXT crossed SQL boundary")
    refused(catalog, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=167)


@pytest.mark.parametrize("table", ["dataset_versions", "contracts", "fragments", "objects"])
def test_missing_join_refuses(catalog, table):
    catalog.execute(f"DELETE FROM data_{table}")
    refused(catalog, "MANIFEST_CORRUPT")


@pytest.mark.parametrize("table,field,value", [
    ("snapshots", "calendar_version", b"not text"),
    ("dataset_versions", "row_count", "0"),
    ("contracts", "definition_json", None),
    ("version_fragments", "ordinal", 1.5),
    ("fragments", "row_count", "0"),
    ("objects", "byte_size", None),
])
def test_types_refuse_without_decode(catalog, table, field, value):
    catalog.execute(f"UPDATE data_{table} SET {field}=?", (value,))
    refused(catalog, "MANIFEST_CORRUPT")


def test_empty_membership_counts_zero(catalog):
    catalog.execute("DELETE FROM data_version_fragments")
    assert probe(catalog) == probes.CatalogProbe(2, 0, 79, 0, 0)


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_blob_lengths_include_nul_and_database_encoding(catalog, encoding):
    copy = sqlite3.connect(":memory:", factory=ScalarConnection)
    copy.execute(f"PRAGMA encoding='{encoding}'")
    populate(copy)
    copy.execute("UPDATE data_snapshots SET calendar_version=?", ("雪\0é",))
    copy.commit()
    copy.execute("BEGIN")
    copy.text_factory = lambda _: pytest.fail("TEXT crossed SQL boundary")
    expected = 170 if encoding == "UTF-8" else 334
    assert probe(copy, max_catalog_bytes=expected).stored_bytes == expected
    refused(copy, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=expected - 1)
    copy.close()


def test_only_scalar_reads_preserve_connection_and_transaction(catalog):
    def deny_writes(action, *_):
        return sqlite3.SQLITE_OK if action in {
            sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
        } else sqlite3.SQLITE_DENY

    catalog.set_authorizer(deny_writes)
    def original_factory(*_):
        pytest.fail("connection row factory used")
    catalog.row_factory = original_factory
    catalog.text_factory = lambda _: pytest.fail("TEXT crossed SQL boundary")
    assert probe(catalog).stored_bytes == 167
    refused(catalog, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=1)
    assert catalog.row_factory is original_factory
    catalog.set_authorizer(None)
    catalog.rollback()  # The caller still owns cleanup, on success and refusal.
    assert not catalog.in_transaction


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, sys.maxsize])
def test_invalid_limits(catalog, value):
    refused(catalog, "CONTRACT_MISMATCH", max_catalog_bytes=value)


def test_configuration_snapshot_and_schema_refusals(catalog):
    class CustomLimits(dict):
        pass

    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        probes.probe_catalog(catalog, "s", "a", CustomLimits(LIMITS))
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        probes.probe_catalog(catalog, "s", "a", {**LIMITS, "extra": 1})
    with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
        probes.probe_catalog(catalog, "missing", "a", LIMITS)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        probes.probe_catalog(catalog, "s", "missing", LIMITS)
    catalog.execute("DROP TABLE data_objects")
    refused(catalog, "MANIFEST_CORRUPT")
    catalog.rollback()
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        probe(catalog)


@pytest.mark.parametrize("code,expected", [
    ("native", "CONTRACT_MISMATCH"), (None, "MANIFEST_CORRUPT"),
    (sqlite3.SQLITE_CORRUPT, "MANIFEST_CORRUPT"),
    *[(code | (2 << 8), "RESOURCE_UNAVAILABLE") for code in
      (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR,
       sqlite3.SQLITE_CANTOPEN, sqlite3.SQLITE_FULL, sqlite3.SQLITE_NOMEM)],
])
def test_sqlite_failure_mapping_is_numeric_and_redacted(catalog, monkeypatch, code, expected):
    exc = sqlite3.OperationalError("private payload; database is locked")
    if code == "native":
        exc = probes.errors.fail("CONTRACT_MISMATCH", "native")
    elif code is not None:
        exc.sqlite_errorcode = code

    def fail(*_, **__):
        raise exc

    monkeypatch.setattr(probes, "_count", fail)
    caught = refused(catalog, expected)
    if code == "native":
        assert caught is exc
    assert "private" not in str(caught)
    assert caught.problem.retryable == (expected == "RESOURCE_UNAVAILABLE")


def test_unused_fields_are_not_charged_or_transferred(catalog):
    for table, column in [("snapshots", "commit_receipt_ref"),
                          ("snapshots", "registered_at"), ("objects", "storage_key")]:
        catalog.execute(f"ALTER TABLE data_{table} ADD COLUMN {column}")
        catalog.execute(f"UPDATE data_{table} SET {column}=?", (b"x" * 100_000,))
    assert probe(catalog, max_catalog_bytes=167).stored_bytes == 167


def test_closed_connection_maps_catalog_error():
    conn = sqlite3.connect(":memory:")
    conn.close()
    with pytest.raises(DataError, match="MANIFEST_CORRUPT"):
        probe(conn)
