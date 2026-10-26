"""Synthetic payload transfer only; arbitrary IDs deliberately prove no identity."""
import json
import sqlite3

import pytest

from engine.v2.data import _bounded_catalog_transfer as transfer
from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.foundation import to_document
from tests.v2.data.test_bounded_catalog_probes import LIMITS, populate


def build(path=":memory:", encoding="UTF-8"):
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA encoding='{encoding}'")
    conn.execute("PRAGMA automatic_index=OFF")
    conn.row_factory = sqlite3.Row
    populate(conn)
    conn.execute("UPDATE data_snapshot_tables SET table_name=CASE table_name WHEN 'a' THEN 'z' ELSE 'a' END")
    conn.execute("UPDATE data_version_fragments SET ordinal=1-ordinal")
    conn.execute("UPDATE data_snapshots SET finality_receipt_refs_json=?,knowledge_mode_by_table_json=?",
                 ('["synthetic-finality"]', '{"a":"reconstructed","z":"reconstructed"}'))
    conn.execute("UPDATE data_contracts SET definition_json=?",
                 (json.dumps(to_document(IV_CRUSH_CONTRACT)),))
    conn.execute('UPDATE data_dataset_versions SET evidence_json=?',
                 ('{"coverage_receipt_refs":["coverage"],"availability_evidence_refs":["availability"]}',))
    conn.execute('UPDATE data_dataset_versions SET partition_logical_hashes_json=?', ('{"p":"hash"}',))
    conn.execute('UPDATE data_fragments SET key_bounds_json=?',
                 ('{"primary_key_min":["first",1,false],"primary_key_max":["last",9,true]}',))
    conn.execute('UPDATE data_fragments SET input_receipt_refs_json=?', ('["input"]',))
    conn.commit()
    conn.execute("BEGIN")
    return conn


@pytest.fixture
def catalog():
    conn = build()
    yield conn
    conn.close()


def read(conn, **changes):
    return transfer.transfer_catalog(conn, "s", "a", {**LIMITS, "max_catalog_bytes": 100_000,
                                                      **changes})


def refuse(conn, code="MANIFEST_CORRUPT", **changes):
    with pytest.raises(DataError) as caught:
        read(conn, **changes)
    message = {"MANIFEST_CORRUPT": "catalog transfer malformed",
               "CONTRACT_MISMATCH": "invalid catalog transfer configuration",
               "RESULT_LIMIT_EXCEEDED": "catalog probe exceeds limit",
               "RESOURCE_UNAVAILABLE": "catalog transfer unavailable",
               "IDENTITY_CONFLICT": "native"}[code]
    assert caught.value.problem == transfer.errors.make_problem(code, message)
    assert "private" not in str(caught.value) and caught.value.__cause__ is None
    if code in {"MANIFEST_CORRUPT", "RESOURCE_UNAVAILABLE"}:
        assert caught.value.__suppress_context__
    assert caught.value.problem.retryable == (code == "RESOURCE_UNAVAILABLE")
    assert conn.in_transaction
    return caught.value


def test_order_repeated_charges_detachment_no_resolver_or_objects(catalog, monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("resolver/object access")
    for name in ("resolve", "resolve_full", "scan"):
        monkeypatch.setattr(Repository, name, forbidden)
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    queries = []
    catalog.set_trace_callback(queries.append)
    result = read(catalog)
    assert all(q.startswith("SELECT ") and "SELECT *" not in q for q in queries)
    assert [r["t.table_name"] for r in result.tables] == ["a", "z"]
    assert [r["m.ordinal"] for r in result.fragments] == [0, 1, 0, 1]
    assert result.tables[0]["c.definition_json"] == IV_CRUSH_CONTRACT
    assert result.snapshot["s.finality_receipt_refs_json"] == ("synthetic-finality",)
    assert result.snapshot["s.knowledge_mode_by_table_json"] == {"a": "reconstructed", "z": "reconstructed"}
    assert result.snapshot["s.parent_snapshot_id"] is None
    assert all(row["d.parent_dataset_version_id"] is None for row in result.tables)
    assert all(row["f.time_bounds_json"] is None for row in result.fragments)
    assert to_document(result.snapshot) == {
        "s.snapshot_id": "s", "s.parent_snapshot_id": None, "s.manifest_hash": "h",
        "s.calendar_version": "cal", "s.source_priority_version": "priority",
        "s.finality_receipt_refs_json": ["synthetic-finality"],
        "s.knowledge_mode_by_table_json": {"a": "reconstructed", "z": "reconstructed"}}
    assert to_document(result.tables) == [{
        "t.snapshot_id": "s", "t.table_name": name, "t.dataset_version_id": "d",
        "d.dataset_version_id": "d", "d.contract_id": "c", "d.parent_dataset_version_id": None,
        "d.manifest_hash": "h", "d.logical_content_hash": "l", "d.row_count": 0,
        "d.knowledge_mode": "reconstructed", "d.evidence_json": {
            "coverage_receipt_refs": ["coverage"], "availability_evidence_refs": ["availability"]},
        "d.partition_logical_hashes_json": {"p": "hash"}, "c.contract_id": "c",
        "c.schema_version": "v", "c.table_name": "a", "c.definition_hash": "h",
        "c.definition_json": to_document(IV_CRUSH_CONTRACT)} for name in ["a", "z"]]
    assert to_document(result.fragments) == [{
        "m.dataset_version_id": "d", "m.ordinal": ordinal, "m.fragment_id": fid,
        "f.fragment_id": fid, "f.object_id": "o", "f.contract_id": "c", "f.partition_key": part,
        "f.row_count": 0, "f.byte_hash": "b", "f.logical_content_hash": "l",
        "f.key_bounds_json": {"primary_key_min": ["first", 1, False], "primary_key_max": ["last", 9, True]},
        "f.time_bounds_json": None, "f.import_request_hash": "i", "f.input_receipt_refs_json": ["input"],
        "o.object_id": "o", "o.kind": "k", "o.content_hash": "h", "o.byte_size": 0}
        for ordinal, fid, part in [(0, "f2", "p2"), (1, "f1", "p1")] * 2]
    assert type(result.fragments[0]["f.key_bounds_json"].primary_key_min) is tuple
    assert [type(x) for x in result.fragments[0]["f.key_bounds_json"].primary_key_min] == [str, int, bool]
    assert type(result.tables[0]["d.evidence_json"].coverage_receipt_refs) is tuple
    assert type(result.fragments[0]["f.input_receipt_refs_json"]) is tuple
    assert result.utf8_bytes == result.probe.stored_bytes - 14  # 2 table + 4*3 integer bytes.
    assert read(catalog, max_catalog_bytes=result.probe.stored_bytes) == result
    refuse(catalog, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=result.probe.stored_bytes - 1)
    result.tables[0]["d.partition_logical_hashes_json"]["x"] = "mutated"
    assert result.tables[1]["d.partition_logical_hashes_json"] == {"p": "hash"}
    assert read(catalog).tables[0]["d.partition_logical_hashes_json"] == {"p": "hash"}
    assert catalog.row_factory is sqlite3.Row and catalog.text_factory is str


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
@pytest.mark.parametrize("late_group", [False, True])
def test_utf8_counter_includes_multibyte_nul_and_is_independent(encoding, late_group, monkeypatch):
    conn = build(encoding=encoding)
    try:
        original = read(conn)
        text = "雪" * 20_000 + "\0é"
        field = "fragments SET partition_key" if late_group else "snapshots SET calendar_version"
        conn.execute(f"UPDATE data_{field}=?", (text,))
        result = read(conn, max_catalog_bytes=400_000)
        expected = original.utf8_bytes + (4 if late_group else 1) * (len(text.encode()) - (2 if late_group else 3))
        assert result.utf8_bytes == expected
        if encoding != "UTF-8":
            assert expected > result.probe.stored_bytes
        cap = max(result.probe.stored_bytes, expected)
        assert read(conn, max_catalog_bytes=cap).utf8_bytes == expected
        monkeypatch.setattr(transfer, "_decode", lambda *_: pytest.fail("JSON parsed before cap"))
        refuse(conn, "RESULT_LIMIT_EXCEEDED", max_catalog_bytes=cap - 1)
    finally:
        conn.close()


def test_all_probes_pass_before_first_payload(catalog, monkeypatch):
    catalog.execute("UPDATE data_contracts SET definition_json=?", ("private" * 100_000,))
    monkeypatch.setattr(transfer, "_transfer", lambda *_: pytest.fail("payload transferred"))
    refuse(catalog, "RESULT_LIMIT_EXCEEDED")


@pytest.mark.parametrize("table,field,bad", [
    ("snapshots", "finality_receipt_refs_json", '["private",'),
    ("snapshots", "finality_receipt_refs_json", '[1]'),
    ("snapshots", "knowledge_mode_by_table_json", '{"x":"private","x":"observed"}'),
    ("dataset_versions", "evidence_json", '{}'),
    ("dataset_versions", "partition_logical_hashes_json", '{"private":true}'),
    ("contracts", "definition_json", '{"schema_version":"table_contract.v999.0"}'),
    ("fragments", "key_bounds_json", '{"primary_key_min":[NaN],"primary_key_max":[]}'),
    ("fragments", "time_bounds_json", '{"time_min":1,"time_max":null}'),
    ("fragments", "time_bounds_json", 'null'),
    ("fragments", "input_receipt_refs_json", 'null'),
])
def test_json_syntax_duplicates_and_native_shapes_refuse(catalog, table, field, bad):
    catalog.execute(f"UPDATE data_{table} SET {field}=?", (bad,))
    assert refuse(catalog).problem.message == "catalog transfer malformed"


def test_recursion_error_during_json_decode_is_redacted(catalog, monkeypatch):
    def deep(*_, **__):
        raise RecursionError("private")
    monkeypatch.setattr(transfer.json, "loads", deep)
    refuse(catalog)


@pytest.mark.parametrize("field,value", [("definition_hash", "not-a-hash"), ("primary_key", ["missing"])])
def test_full_native_contract_checks_through_transfer(catalog, field, value):
    contract = {**to_document(IV_CRUSH_CONTRACT), field: value}
    catalog.execute("UPDATE data_contracts SET definition_json=?", (json.dumps(contract),))
    refuse(catalog)


def test_transfer_rechecks_types_after_scalar_admission(catalog, monkeypatch):
    probe = transfer.probes.probe_catalog
    def corrupt(*args):
        result = probe(*args)
        catalog.execute("UPDATE data_snapshots SET calendar_version=x'80'")
        return result
    monkeypatch.setattr(transfer.probes, "probe_catalog", corrupt)
    with pytest.raises(DataError) as caught:
        read(catalog)
    assert caught.value.problem == transfer.errors.make_problem(
        "MANIFEST_CORRUPT", "catalog transfer has invalid field")


def test_invalid_stored_unicode_is_redacted(catalog):
    catalog.execute("UPDATE data_snapshots SET calendar_version=CAST(x'80' AS TEXT)")
    assert refuse(catalog).problem.message == "catalog transfer malformed"


def test_same_wal_read_view_across_concurrent_commit(tmp_path, monkeypatch):
    path = tmp_path / "catalog.sqlite"
    conn = build(path)
    conn.rollback()
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("BEGIN")
    writer = sqlite3.connect(path)
    probe = transfer.probes.probe_catalog
    def interleave(*args):
        result = probe(*args)
        writer.execute("UPDATE data_snapshots SET calendar_version='new'")
        writer.commit()
        return result
    monkeypatch.setattr(transfer.probes, "probe_catalog", interleave)
    try:
        assert read(conn).snapshot["s.calendar_version"] == "cal"
        conn.rollback()
        conn.execute("BEGIN")
        assert read(conn).snapshot["s.calendar_version"] == "new"
    finally:
        writer.close()
        conn.close()


@pytest.mark.parametrize("exc,code", [
    (sqlite3.OperationalError("private database is locked"), "MANIFEST_CORRUPT"),
    (UnicodeEncodeError("utf-8", "\ud800", 0, 1, "private"), "MANIFEST_CORRUPT"),
    (transfer.errors.fail("IDENTITY_CONFLICT", "native"), "IDENTITY_CONFLICT"),
    *[(sqlite3.OperationalError("private"), code) for code in (sqlite3.SQLITE_BUSY,
        sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_IOERR, sqlite3.SQLITE_CANTOPEN,
        sqlite3.SQLITE_FULL, sqlite3.SQLITE_NOMEM)],
])
def test_transfer_failures_preserve_native_or_map_numeric_sql(catalog, monkeypatch, exc, code):
    if type(code) is int:
        exc.sqlite_errorcode, code = code | (2 << 8), "RESOURCE_UNAVAILABLE"
    def fail(*_):
        raise exc
    monkeypatch.setattr(transfer, "_transfer", fail)
    caught = refuse(catalog, code)
    if code == "IDENTITY_CONFLICT":
        assert caught is exc


@pytest.mark.parametrize("count", [0, 2])
def test_transfer_membership_guard(catalog, count):
    message = "membership changed" if count == 0 else "has missing membership"
    with pytest.raises(DataError) as caught:
        transfer._transfer(catalog.cursor(), "s.calendar_version", "data_snapshots s", "1", (),
                           count, 0, 100)
    assert caught.value.problem == transfer.errors.make_problem(
        "MANIFEST_CORRUPT", "catalog transfer " + message)


def test_transfer_rechecks_types_and_preserves_nullable_text(catalog):
    catalog.execute("UPDATE data_snapshots SET parent_snapshot_id='parent'")
    catalog.execute('UPDATE data_fragments SET time_bounds_json=?',
                   ('{"time_min":null,"time_max":null}',))
    result = read(catalog)
    assert result.snapshot["s.parent_snapshot_id"] == "parent"
    assert result.fragments[0]["f.time_bounds_json"].time_min is None
    catalog.execute('UPDATE data_fragments SET time_bounds_json=?',
                   ('{"time_min":"2001-01-01","time_max":"2001-01-02"}',))
    assert to_document(read(catalog).fragments[0]["f.time_bounds_json"]) == {
        "time_min": "2001-01-01", "time_max": "2001-01-02"}
    catalog.execute("UPDATE data_snapshots SET calendar_version=x'80'")
    with pytest.raises(DataError) as caught:
        transfer._transfer(catalog.cursor(), "s.calendar_version", "data_snapshots s", "1", (),
                           1, 0, 100)
    assert caught.value.problem == transfer.errors.make_problem(
        "MANIFEST_CORRUPT", "catalog transfer has invalid field")


def test_empty_membership_nullable_values_and_unsupported_configuration(catalog):
    catalog.execute("DELETE FROM data_version_fragments")
    assert read(catalog).fragments == ()
    catalog.text_factory = bytes
    refuse(catalog, "CONTRACT_MISMATCH")
    catalog.text_factory = lambda _: pytest.fail("custom text factory called")
    refuse(catalog, "CONTRACT_MISMATCH")
    catalog.text_factory, catalog.row_factory = str, None
    refuse(catalog, "CONTRACT_MISMATCH")
    catalog.row_factory = sqlite3.Row
    catalog.rollback()
    with pytest.raises(DataError) as caught:
        read(catalog)
    assert caught.value.problem == transfer.errors.make_problem(
        "CONTRACT_MISMATCH", "catalog probe requires a read transaction")
