"""Synthetic native scanner parity and request-local iterator lifetime."""
import sqlite3
from datetime import datetime

import pyarrow as pa
import pytest

from engine.v2.contracts.data import DataQuery, KeyPredicate
from engine.v2.data import bounded_snapshot, errors, objects
from engine.v2.data.repository import Repository
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

LIMITS = dict(max_catalog_tables=3, max_catalog_fragments=6, max_catalog_bytes=100_000,
              max_fragments=3, max_partitions=3)
TABLE = "earnings_events"
COLUMNS = ("event_id", "ticker", "event_date", "session", "date_conflict",
           "event_cluster_id", "src_orats")


@pytest.fixture
def fixture(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    contract = contract_for(TABLE)
    ref = contract_ref_for(contract)
    records = []
    for year in (2020, 2021):
        rows = [dict(event_id=f"{year}-{ticker}", ticker=ticker,
            event_date=datetime(year, 1, 2), year=year, session="BMO", session_src="orats",
            annc_tod=None, src_orats=True, src_oquants=False, src_nasdaq=False,
            src_yfinance=False, date_agree=True, date_conflict=False, updated_at=None,
            event_cluster_id=None, claim_count=None, reconciliation=None)
            for ticker in ("AAA", "BBB")]
        records.append(publish_and_inspect(store, contract, ref, rows, str(year)))
    other = contract_for("securities")
    other_record = publish_and_inspect(store, other, contract_ref_for(other),
        [dict(ticker="UNSELECTED", year=2020, first_date=None, last_date=None, mcap_usd=1.5e9,
            mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
            n_obs=250, src="orats")], "2020")
    snapshot = commit_tables(conn, clock, {TABLE: records, "securities": [other_record]},
                             {TABLE: contract, "securities": other})
    yield conn, store, snapshot, ref, records, other_record
    conn.set_authorizer(None)
    conn.close()


def opened(fixture, *, store=True):
    conn, native_store, snapshot, *_ = fixture
    return bounded_snapshot.open_bounded_snapshot(
        Repository(conn, native_store if store else None, pin_snapshots=True), snapshot,
        table_name=TABLE, limits=LIMITS)


def query(fixture, **changes):
    return DataQuery(**{ "snapshot_id": fixture[2].snapshot_id, "table_contract_ref": fixture[3],
        "columns": COLUMNS, "key_filter": (KeyPredicate(column="ticker", operator="in",
            values=("AAA", "BBB")),), "order_by": ("event_id",),
        "max_batch_rows": 1, "max_result_rows": 4, **changes})


def scan(session, request, **changes):
    return session.scan(request, table_name=TABLE, **{"max_object_bytes": 1_000_000, **changes})


def rows(batches):
    return [row for batch in batches for row in batch.to_pylist()]


def error(code, call):
    with pytest.raises(errors.DataError) as caught:
        call()
    assert caught.value.code == code
    return caught.value


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_real_arrow_native_parity_no_sql_and_no_nonselected_object(fixture, monkeypatch, batch_size):
    conn, store, snapshot, ref, _, other = fixture
    key_filter = (KeyPredicate(column="year", operator="eq", values=(2020,)),
                  KeyPredicate(column="ticker", operator="in", values=("AAA",)))
    request = query(fixture, key_filter=key_filter, max_batch_rows=min(batch_size, 2),
                    max_result_rows=2)
    oracle = Repository(conn, store)
    expected_bound = oracle.scan_population_bound(snapshot.snapshot_id, table_name=TABLE,
        table_contract_ref=ref, key_filter=key_filter)
    expected = list(oracle.scan(request, table_name=TABLE, max_object_bytes=1_000_000))
    reads = []
    native_read = objects.read_verified_bytes
    def guarded(store, obj, **kwargs):
        assert obj.object_id != other.object_ref.object_id
        reads.append(obj.object_id)
        return native_read(store, obj, **kwargs)
    monkeypatch.setattr(objects, "read_verified_bytes", guarded)
    with opened(fixture) as session:
        conn.set_authorizer(lambda *_: sqlite3.SQLITE_DENY)
        assert session.scan_population_bound(snapshot.snapshot_id, table_name=TABLE,
            table_contract_ref=ref, key_filter=key_filter) == expected_bound == 2
        actual = list(scan(session, request))
        assert rows(actual) == rows(expected)
        assert all(batch.schema.names == list(COLUMNS) for batch in actual)
        assert all(batch.num_rows <= batch_size for batch in actual)
    assert len(reads) == 1


@pytest.mark.parametrize("cap", [None, True, False, 0, -1, 1.5, "1000000"])
def test_object_cap_is_mandatory_and_native(fixture, cap):
    with opened(fixture) as session:
        actual = error("QUERY_NOT_BOUNDED", lambda: list(scan(session, query(fixture),
                                                              max_object_bytes=cap)))
        expected = error("QUERY_NOT_BOUNDED", lambda: objects.validate_max_object_bytes(cap))
        assert actual.problem == expected.problem
        error("QUERY_NOT_BOUNDED", lambda: list(session.scan(query(fixture), table_name=TABLE)))


def test_missing_store_and_nonselected_table_refuse_before_objects(fixture, monkeypatch):
    def forbidden(*_, **__):
        pytest.fail("object opened on refused scan")
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    with opened(fixture, store=False) as session:
        error("CONTRACT_MISMATCH", lambda: list(scan(session, query(fixture))))
    with opened(fixture) as session:
        for table in ("securities", "unknown", None, []):
            error("CONTRACT_MISMATCH", lambda: list(session.scan(query(fixture), table_name=table,
                                                                 max_object_bytes=1_000_000)))
            error("CONTRACT_MISMATCH", lambda: session.scan_population_bound(fixture[2].snapshot_id,
                table_name=table, table_contract_ref=fixture[3]))


@pytest.mark.parametrize("changes", [{"columns": ("missing",)}, {"max_result_rows": 1},
    {"max_batch_rows": 0},
    {"table_contract_ref": contract_ref_for(contract_for("securities"))}])
def test_native_query_failures_have_identical_envelopes(fixture, changes):
    request = query(fixture, **changes)
    conn, store, *_ = fixture
    with pytest.raises(errors.DataError) as native:
        list(Repository(conn, store).scan(request, table_name=TABLE, max_object_bytes=1_000_000))
    with opened(fixture) as session:
        actual = error(native.value.code, lambda: list(scan(session, request)))
        assert actual.problem == native.value.problem


def test_recorded_byte_cap_has_native_refusal_and_no_fallback(fixture, monkeypatch):
    conn, store, *_ = fixture
    with pytest.raises(errors.DataError) as native:
        list(Repository(conn, store).scan(query(fixture), table_name=TABLE, max_object_bytes=1))
    def forbidden(*_, **__):
        pytest.fail("uncapped object path used")
    monkeypatch.setattr(objects, "verify_object_path", forbidden)
    with opened(fixture) as session:
        actual = error(native.value.code, lambda: list(scan(session, query(fixture), max_object_bytes=1)))
        assert actual.problem == native.value.problem


def test_single_started_scan_release_on_close_and_exhaustion(fixture):
    with opened(fixture) as session:
        first = scan(session, query(fixture))
        second = scan(session, query(fixture))  # Construction alone does not acquire the slot.
        assert next(first).num_rows == 1
        error("CONTRACT_MISMATCH", lambda: next(second))
        first.close()
        assert len(rows(scan(session, query(fixture)))) == 4
        assert len(rows(scan(session, query(fixture)))) == 4


def test_started_and_never_started_iterator_lifetime_guard(fixture):
    with opened(fixture) as session:
        never_started = scan(session, query(fixture))
        started = scan(session, query(fixture))
        assert next(started).num_rows == 1
    for iterator in (never_started, started):
        error("CONTRACT_MISMATCH", lambda: next(iterator))
    error("CONTRACT_MISMATCH", lambda: session.scan_population_bound(None,
        table_name=None, table_contract_ref=None))
    error("CONTRACT_MISMATCH", lambda: list(session.scan(None, table_name=None)))


def test_late_native_failure_preserved_and_slot_released(fixture, monkeypatch):
    primary = errors.fail("MANIFEST_CORRUPT", "synthetic late stream failure", stage="scan",
                          details={"synthetic": [1, "native diagnostic"]})
    closed = []
    def failing(*_, **__):
        try:
            yield pa.record_batch({"event_id": ["provisional"]})
            raise primary
        finally:
            closed.append(True)
    monkeypatch.setattr(Repository, "_execute_scan", failing)
    with opened(fixture) as session:
        stream = scan(session, query(fixture))
        assert next(stream).to_pylist() == [{"event_id": "provisional"}]
        assert error("MANIFEST_CORRUPT", lambda: next(stream)) is primary
        assert closed == [True]
        replacement = scan(session, query(fixture))
        next(replacement)
        replacement.close()
        assert closed == [True, True]


def test_context_exit_closes_started_native_stream(fixture, monkeypatch):
    closed = []
    def streaming(*_, **__):
        try:
            yield pa.record_batch({"event_id": ["provisional"]})
            yield pa.record_batch({"event_id": ["later"]})
        finally:
            closed.append(True)
    monkeypatch.setattr(Repository, "_execute_scan", streaming)
    with opened(fixture) as session:
        stream = scan(session, query(fixture))
        next(stream)
        assert not closed
    assert closed == [True]


def test_zero_candidate_bound_scan_uses_no_objects(fixture, monkeypatch):
    key_filter = (KeyPredicate(column="year", operator="eq", values=(1900,)),)
    def forbidden(*_, **__):
        pytest.fail("zero candidate scan opened object")
    monkeypatch.setattr(objects, "read_verified_bytes", forbidden)
    with opened(fixture) as session:
        assert session.scan_population_bound(fixture[2].snapshot_id, table_name=TABLE,
            table_contract_ref=fixture[3], key_filter=key_filter) == 0
        assert rows(scan(session, query(fixture, key_filter=key_filter,
                                       max_result_rows=0, max_batch_rows=1))) == []


def test_unknown_snapshot_and_wrong_population_ref_native_codes(fixture):
    with opened(fixture) as session:
        error("SNAPSHOT_NOT_FOUND", lambda: list(scan(session, query(fixture, snapshot_id="missing"))))
        error("SNAPSHOT_NOT_FOUND", lambda: session.scan_population_bound("missing",
            table_name=TABLE, table_contract_ref=fixture[3]))
        error("CONTRACT_MISMATCH", lambda: session.scan_population_bound(fixture[2].snapshot_id,
            table_name=TABLE, table_contract_ref=contract_ref_for(contract_for("securities"))))


def test_same_iterator_reentrant_resume_is_refused(fixture, monkeypatch):
    handles = {}
    def reentrant(*_, **__):
        error("CONTRACT_MISMATCH", lambda: next(handles["stream"]))
        error("CONTRACT_MISMATCH", lambda: handles["stream"].close())
        yield pa.record_batch({"event_id": ["outer"]})
    monkeypatch.setattr(Repository, "_execute_scan", reentrant)
    with opened(fixture) as session:
        handles["stream"] = scan(session, query(fixture))
        assert next(handles["stream"]).to_pylist() == [{"event_id": "outer"}]
        handles["stream"].close()


@pytest.mark.parametrize("consumer_failure", [False, True])
def test_exit_cleanup_error_preserves_primary_and_always_closes_session(
        fixture, monkeypatch, consumer_failure):
    primary = RuntimeError("synthetic consumer failure")
    cleanup = RuntimeError("synthetic native close failure")
    def streaming(*_, **__):
        try:
            yield pa.record_batch({"event_id": ["provisional"]})
        finally:
            raise cleanup
    monkeypatch.setattr(Repository, "_execute_scan", streaming)
    with pytest.raises(RuntimeError) as caught:
        with opened(fixture) as session:
            stream = scan(session, query(fixture))
            next(stream)
            if consumer_failure:
                raise primary
    assert caught.value is (primary if consumer_failure else cleanup)
    error("CONTRACT_MISMATCH", lambda: session.resolve(fixture[2].snapshot_id))
    error("CONTRACT_MISMATCH", lambda: next(stream))


def test_closed_lifetime_wins_over_cap_and_iterator_protocol(fixture):
    with opened(fixture) as session:
        stream = scan(session, query(fixture))
    for cap in (None, False, 0, "bad"):
        error("CONTRACT_MISMATCH", lambda: session.scan(None, table_name=None, max_object_bytes=cap))
    error("CONTRACT_MISMATCH", lambda: session.scan(None, table_name=None))
    error("CONTRACT_MISMATCH", lambda: iter(stream))
    error("CONTRACT_MISMATCH", stream.close)


def test_single_object_byte_cap_equality_and_one_under_native_parity(fixture):
    import pyarrow.parquet as pq
    conn, store, _, _, records, _ = fixture
    obj = records[0].object_ref
    payload = objects.read_verified_bytes(store, obj, max_bytes=obj.byte_size)
    with pa.BufferReader(payload) as source, pq.ParquetFile(source) as parquet:
        expanded = sum(parquet.metadata.row_group(i).total_byte_size
                       for i in range(parquet.metadata.num_row_groups))
    cap = max(obj.byte_size, expanded)
    request = query(fixture, key_filter=(KeyPredicate(column="year", operator="eq", values=(2020,)),),
                    max_result_rows=2)
    oracle = Repository(conn, store)
    expected = rows(oracle.scan(request, table_name=TABLE, max_object_bytes=cap))
    assert len(expected) == 2
    native = error("RESULT_LIMIT_EXCEEDED", lambda: list(
        oracle.scan(request, table_name=TABLE, max_object_bytes=cap - 1)))
    with opened(fixture) as session:
        assert rows(scan(session, request, max_object_bytes=cap)) == expected
        actual = error("RESULT_LIMIT_EXCEEDED", lambda: list(
            scan(session, request, max_object_bytes=cap - 1)))
        assert actual.problem == native.problem
