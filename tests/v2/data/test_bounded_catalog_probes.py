"""Synthetic SQL only: the probe deliberately does not authenticate identities."""
import sqlite3
import sys

import pytest

from engine.v2.data import _bounded_catalog as probes
from engine.v2.data import schema
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


def refused(conn, code, *, message=None, **changes):
    with pytest.raises(DataError) as caught:
        probe(conn, **changes)
    assert caught.value.code == code
    assert caught.value.problem.message == (message or {
        "RESULT_LIMIT_EXCEEDED": "catalog probe exceeds limit",
        "CONTRACT_MISMATCH": "invalid catalog probe configuration",
        "MANIFEST_CORRUPT": "catalog probe has invalid field or join",
    }[code])
    assert caught.value.problem.details == {}
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
    refused(catalog, "MANIFEST_CORRUPT", message=("catalog probe has missing contract"
            if table == "fragments" else "catalog probe has invalid field or join"))


@pytest.mark.parametrize("table,field", [
    (table, field) for table, row in ROWS.items() for field in row
    if field not in {"snapshot_id", "table_name", "dataset_version_id", "fragment_id",
                     "contract_id", "object_id"}
])
def test_payload_types_refuse_without_decode(catalog, table, field):
    catalog.execute(f"UPDATE data_{table} SET {field}=?", (b"wrong SQL type",))
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


@pytest.mark.parametrize("cap", sorted(LIMITS))
@pytest.mark.parametrize("value", [None, True, 0, -1, 1.5, "1", sys.maxsize])
def test_invalid_limits(catalog, cap, value):
    refused(catalog, "CONTRACT_MISMATCH", **{cap: value})


def test_configuration_snapshot_and_schema_refusals(catalog):
    class CustomLimits(dict):
        pass

    with pytest.raises(DataError, match="^CONTRACT_MISMATCH: invalid catalog probe configuration$"):
        probes.probe_catalog(catalog, "s", "a", CustomLimits(LIMITS))
    with pytest.raises(DataError, match="^CONTRACT_MISMATCH: invalid catalog probe configuration$"):
        probes.probe_catalog(catalog, "s", "a", {**LIMITS, "extra": 1})
    with pytest.raises(DataError, match="^SNAPSHOT_UNRESOLVED: catalog probe snapshot unavailable$"):
        probes.probe_catalog(catalog, "missing", "a", LIMITS)
    with pytest.raises(DataError, match="^CONTRACT_MISMATCH: catalog probe table unavailable$"):
        probes.probe_catalog(catalog, "s", "missing", LIMITS)
    catalog.execute("DROP TABLE data_objects")
    refused(catalog, "MANIFEST_CORRUPT", message="catalog probe malformed")
    catalog.rollback()
    with pytest.raises(DataError, match="^CONTRACT_MISMATCH: catalog probe requires a read transaction$"):
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
    message = ("native" if code == "native" else "catalog probe unavailable"
               if expected == "RESOURCE_UNAVAILABLE" else "catalog probe malformed")
    caught = refused(catalog, expected, message=message)
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
    with pytest.raises(DataError, match="^MANIFEST_CORRUPT: catalog probe malformed$"):
        probe(conn)


@pytest.mark.parametrize("key", sorted(LIMITS))
def test_missing_limit_refuses(catalog, key):
    with pytest.raises(DataError, match="^CONTRACT_MISMATCH: invalid catalog probe configuration$"):
        probes.probe_catalog(catalog, "s", "a", {k: v for k, v in LIMITS.items() if k != key})


@pytest.mark.parametrize("position,value", [(0, None), (1, None), (1, ""), (1, 1),
                                          (2, None), (2, ""), (2, 1), (3, None), (3, [])])
def test_invalid_holder_and_arguments(catalog, position, value):
    args = [catalog, "s", "a", LIMITS]
    args[position] = value
    with pytest.raises(DataError, match="^CONTRACT_MISMATCH: invalid catalog probe configuration$"):
        probes.probe_catalog(*args)


def test_isolated_fragment_contract_join_refuses(catalog):
    catalog.execute("UPDATE data_fragments SET contract_id='absent'")
    refused(catalog, "MANIFEST_CORRUPT", message="catalog probe has missing contract")


def test_distinct_selected_partitions_exclude_unrelated_table(catalog):
    catalog.execute("UPDATE data_fragments SET partition_key='p1'")
    insert(catalog, "dataset_versions", {**ROWS["dataset_versions"], "dataset_version_id": "other"})
    insert(catalog, "version_fragments", {**ROWS["version_fragments"],
                                        "dataset_version_id": "other", "fragment_id": "f3"})
    insert(catalog, "fragments", {**ROWS["fragments"], "fragment_id": "f3", "partition_key": "p3"})
    catalog.execute("UPDATE data_snapshot_tables SET dataset_version_id='other' WHERE table_name='z'")
    result = probe(catalog, max_fragments=2, max_partitions=1)
    assert (result.tables, result.catalog_fragments, result.fragments, result.partitions) == (2, 3, 2, 1)


@pytest.mark.parametrize("table,field,charge", [("snapshots", "parent_snapshot_id", 1),
    ("dataset_versions", "parent_dataset_version_id", 2), ("fragments", "time_bounds_json", 4)])
def test_nullable_payload_nonnull_bytes_are_charged(catalog, table, field, charge):
    catalog.execute(f"UPDATE data_{table} SET {field}=?", ("雪\0",))
    expected = 167 + 4 * charge
    assert probe(catalog, max_catalog_bytes=expected).stored_bytes == expected
    refused(catalog, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=expected - 1)


@pytest.mark.parametrize("expected,message", [(0, "changed"), (3, "has missing membership")])
def test_membership_mismatch_messages(catalog, expected, message):
    cursor = catalog.cursor()
    full = "membership changed" if message == "changed" else message
    with pytest.raises(DataError, match=f"^MANIFEST_CORRUPT: catalog probe {full}$"):
        probes._measure(cursor, "s.calendar_version", "data_snapshots s", "1", (), expected, 0, 100)
    cursor.close()


def test_production_schema_ddl_with_synthetic_rows():
    conn = sqlite3.connect(":memory:", factory=ScalarConnection)
    conn.execute("PRAGMA foreign_keys=ON")
    for migration in schema.MIGRATIONS:
        for sql in migration[2]:
            conn.execute(sql)
    for table in ("contracts", "objects", "fragments", "dataset_versions", "version_fragments",
                  "snapshots", "snapshot_tables"):
        row = {k: "sha256:" + "0" * 64 if k.endswith("hash") else v for k, v in ROWS[table].items()}
        if table not in {"version_fragments", "snapshot_tables"}:
            row["registered_at"] = "2024-01-01T00:00:00Z"
        if table == "objects":
            row["storage_key"] = "not-opened"
        if table == "snapshots":
            row["commit_receipt_ref"] = "not-followed"
        insert(conn, table, row)
    conn.commit()
    conn.execute("BEGIN")
    conn.text_factory = lambda _: pytest.fail("TEXT crossed SQL boundary")
    result = probe(conn)
    assert (result.tables, result.catalog_fragments, result.fragments, result.partitions) == (1, 1, 1, 1)
    assert result.stored_bytes == 630  # 70 original bytes + eight hash expansions of 70.
    conn.close()


@pytest.mark.parametrize("field,tables", [
    ("dataset_version_id", ("dataset_versions", "snapshot_tables", "version_fragments")),
    ("contract_id", ("contracts", "dataset_versions", "fragments")),
    ("fragment_id", ("fragments", "version_fragments")),
    ("object_id", ("objects", "fragments")),
])
def test_linked_identifier_types_refuse_without_losing_joins(catalog, field, tables):
    for table in tables:
        catalog.execute(f"UPDATE data_{table} SET {field}=?", (b"bad identifier",))
    refused(catalog, "MANIFEST_CORRUPT")


@pytest.mark.parametrize("table,where", [("snapshot_tables", "table_name='z'"), ("contracts", "1")])
def test_names_refuse_without_losing_selected_table(catalog, table, where):
    catalog.execute(f"UPDATE data_{table} SET table_name=? WHERE {where}", (b"bad name",))
    refused(catalog, "MANIFEST_CORRUPT")


@pytest.mark.parametrize("position", [1, 2])
@pytest.mark.parametrize("value", ["\ud800", "\udfff", "private\ud800suffix", "\ud800\udfff"])
def test_unencodable_request_refuses_before_sql(catalog, monkeypatch, position, value):
    def forbidden(*_):
        pytest.fail("invalid request reached catalog cursor")

    monkeypatch.setattr(ScalarConnection, "cursor", forbidden)
    args = [catalog, "s", "a", LIMITS]
    args[position] = value
    settings = catalog.row_factory, catalog.text_factory
    with pytest.raises(DataError) as caught:
        probes.probe_catalog(*args)
    assert caught.value.code == "CONTRACT_MISMATCH"
    assert caught.value.problem.message == "invalid catalog probe configuration"
    assert not caught.value.problem.retryable and caught.value.problem.details == {}
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert catalog.in_transaction and settings == (catalog.row_factory, catalog.text_factory)


@pytest.mark.parametrize("position", [1, 2])
@pytest.mark.parametrize("value", ["雪", "x\0𝄞"])
def test_utf8_request_text_remains_accepted(catalog, position, value):
    args = [catalog, "s", "a", LIMITS]
    args[position] = value
    if position == 1:
        catalog.execute("UPDATE data_snapshots SET snapshot_id=?", (value,))
        catalog.execute("UPDATE data_snapshot_tables SET snapshot_id=?", (value,))
    else:
        catalog.execute("UPDATE data_snapshot_tables SET table_name=? WHERE table_name='a'", (value,))
    result = probes.probe_catalog(*args)
    assert (result.tables, result.catalog_fragments, result.fragments, result.partitions) == (2, 4, 2, 2)
