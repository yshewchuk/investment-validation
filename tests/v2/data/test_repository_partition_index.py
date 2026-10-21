"""A pinned ``Repository`` plans a partition-filtered scan over only its partition.

The pinned entry keeps a ``partition_key -> ascending record indices`` index
(built once, with every fragment's primary-key bounds validated), so a pinned
scan whose ``key_filter`` names a partition column hands the planner just that
partition's fragments. Output, bounds and read counters stay identical to the
unpinned repository, which plans every fragment and prunes inside the planner.
"""
import dataclasses

import pytest

from engine.v2.contracts.data import DataQuery, KeyPredicate, TimeInterval
from engine.v2.data import query as query_mod
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT as CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    publish_and_inspect,
)

TICKERS = [("A", "2024-05-01"), ("B", "2024-05-02"),
           ("C", "2024-05-03"), ("D", "2024-05-04")]
MULTI_FRAGMENT = [
    ("A", ("2024-05-01", "2024-05-02")),
    ("A", ("2024-05-03", "2024-05-04", "2024-05-05")),
    ("B", ("2024-05-02",)),
    ("C", ("2024-05-02", "2024-05-03", "2024-05-04", "2024-05-05")),
    ("D", ("2024-05-04", "2024-05-05")),
]


def _move(ticker, day):
    return dict(ticker=ticker, event_date=day, realized_move_pct=1.0,
                available_as_of_date="2024-05-05", implied_move_pct=2.0, event_ordinal=0,
                admitted=False, mean_prior_or_implied=None, n_prior_implied=0,
                skipped=False, computed_at="2024-05-05T00:00:00Z", source_hash="synthetic",
                capture_id="synthetic")


def _build(tmp_path, groups=None):
    conn, clock, store = catalog_and_store(tmp_path)
    ref = contract_ref_for(CONTRACT)
    groups = [(t, (d,)) for t, d in TICKERS] if groups is None else groups
    records = [publish_and_inspect(store, CONTRACT, ref, [_move(t, d) for d in days], t)
               for t, days in groups]
    hashes = {}
    for ticker in sorted({t for t, _ in groups}):
        parts = [days for t, days in groups if t == ticker]
        if len(parts) > 1:
            rows = [_move(ticker, day) for days in parts for day in days]
            hashes[ticker] = publish_and_inspect(
                store, CONTRACT, ref, rows, ticker).logical_content_hash
    snapshot = commit_tables(conn, clock, {"computed_moves": records},
                             {"computed_moves": CONTRACT}, store=store,
                             partition_logical_hashes={"computed_moves": hashes})
    return conn, store, ref, snapshot


def _request(snapshot, ref, *, key_filter=(), time_interval=None, max_result_rows):
    return DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
                     columns=("ticker", "event_date"), key_filter=key_filter,
                     time_interval=time_interval, order_by=CONTRACT.primary_key,
                     max_batch_rows=1, max_result_rows=max_result_rows)


def _rows(repository, request):
    return [row for batch in repository.scan(request, table_name="computed_moves")
            for row in batch.to_pylist()]


def _counter_counts(repository):
    return {table: {field: total for field, total in row.items() if field != "seconds"}
            for table, row in repository.read_counters.snapshot().items()}


def _spy_plans(monkeypatch):
    calls = []
    original = query_mod.plan_scan_population

    def spy(contract, records, **kwargs):
        calls.append(tuple(records))
        return original(contract, records, **kwargs)

    monkeypatch.setattr(query_mod, "plan_scan_population", spy)
    return calls


def test_pinned_and_unpinned_scans_return_identical_rows_and_counters(tmp_path):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        pinned = Repository(conn, store, pin_snapshots=True)
        unpinned = Repository(conn, store)
        interval = TimeInterval(column="event_date", start_inclusive="2024-05-01",
                                end_exclusive="2024-05-03")
        cases = [
            ((KeyPredicate(column="ticker", operator="eq", values=("A",)),), None,
             [{"ticker": "A", "event_date": "2024-05-01"}]),
            ((KeyPredicate(column="ticker", operator="in", values=("A", "C")),), None,
             [{"ticker": "A", "event_date": "2024-05-01"},
              {"ticker": "C", "event_date": "2024-05-03"}]),
            ((KeyPredicate(column="ticker", operator="eq", values=("ZZZ",)),), None, []),
            ((KeyPredicate(column="ticker", operator="eq", values=("B",)),), interval,
             [{"ticker": "B", "event_date": "2024-05-02"}]),
        ]
        for key_filter, time_interval, expected in cases:
            bound = unpinned.scan_population_bound(
                snapshot.snapshot_id, table_name="computed_moves", table_contract_ref=ref,
                key_filter=key_filter, time_interval=time_interval)
            request = _request(snapshot, ref, key_filter=key_filter,
                               time_interval=time_interval, max_result_rows=bound)
            assert _rows(pinned, request) == expected
            assert _rows(unpinned, request) == expected
            assert _counter_counts(pinned) == _counter_counts(unpinned)
    finally:
        conn.close()


def test_partition_predicate_plans_only_that_partition_pinned_vs_every_fragment_unpinned(
        tmp_path, monkeypatch):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        key_filter = (KeyPredicate(column="ticker", operator="eq", values=("A",)),)
        request = _request(snapshot, ref, key_filter=key_filter, max_result_rows=1)
        expected = [{"ticker": "A", "event_date": "2024-05-01"}]
        calls = _spy_plans(monkeypatch)
        assert _rows(Repository(conn, store, pin_snapshots=True), request) == expected
        assert [len(records) for records in calls] == [1]
        calls.clear()
        assert _rows(Repository(conn, store), request) == expected
        assert [len(records) for records in calls] == [len(TICKERS)]
    finally:
        conn.close()


@pytest.mark.parametrize("bound_only", [False, True])
def test_corrupt_bound_on_an_unfiltered_fragment_refuses_and_is_never_cached(
        tmp_path, monkeypatch, bound_only):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        original = Repository._records

        def corrupting(self, catalog_conn, dataset_version_id, contract_ref):
            records = list(original(self, catalog_conn, dataset_version_id, contract_ref))
            records[-1] = dataclasses.replace(
                records[-1], primary_key_min=("ZZZ", "2099-01-01"),
                primary_key_max=("AAA", "2000-01-01"))
            return records

        monkeypatch.setattr(Repository, "_records", corrupting)
        repository = Repository(conn, store, pin_snapshots=True)
        key_filter = (KeyPredicate(column="ticker", operator="eq", values=("A",)),)
        request = _request(snapshot, ref, key_filter=key_filter, max_result_rows=1)
        for _ in range(2):
            with pytest.raises(DataError) as failure:
                if bound_only:
                    repository.scan_population_bound(
                        snapshot.snapshot_id, table_name="computed_moves",
                        table_contract_ref=ref, key_filter=key_filter)
                else:
                    _rows(repository, request)
            assert failure.value.code == "MANIFEST_CORRUPT"
            assert failure.value.problem.message == "fragment primary-key bounds are invalid"
            assert failure.value.problem.retryable is False
            assert repository._pinned_tables == {}
    finally:
        conn.close()


@pytest.mark.parametrize("as_document", [False, True])
@pytest.mark.parametrize("values,expected,partitions", [
    (("A",), 5, ["A", "A"]), (("A", "C"), 9, ["A", "A", "C"]), (("ZZZ",), 0, []),
])
def test_pinned_bound_and_scan_plan_only_selected_partitions_on_cold_and_warm_cache(
        tmp_path, monkeypatch, values, expected, partitions, as_document):
    conn, store, ref, snapshot = _build(tmp_path, MULTI_FRAGMENT)
    try:
        key_filter = (KeyPredicate(column="ticker", operator="in", values=values),)
        bound_filter = ([dataclasses.asdict(key_filter[0])] if as_document else key_filter)
        repository = Repository(conn, store, pin_snapshots=True)
        calls = _spy_plans(monkeypatch)
        for _ in range(2):
            bound = repository.scan_population_bound(
                snapshot.snapshot_id, table_name="computed_moves",
                table_contract_ref=ref, key_filter=bound_filter)
            assert bound == expected
            rows = _rows(repository, _request(
                snapshot, ref, key_filter=key_filter, max_result_rows=bound))
            assert rows == [{"ticker": ticker, "event_date": day}
                            for ticker, days in MULTI_FRAGMENT if ticker in values for day in days]
        assert [[record.partition_key for record in records] for records in calls] == [
            partitions] * 4
        calls.clear()
        assert Repository(conn, store).scan_population_bound(
            snapshot.snapshot_id, table_name="computed_moves",
            table_contract_ref=ref, key_filter=bound_filter) == expected
        assert [len(records) for records in calls] == [len(MULTI_FRAGMENT)]
    finally:
        conn.close()


@pytest.mark.parametrize("key_filter,code,message", [
    (None, "QUERY_NOT_BOUNDED", "key_filter must be a list or tuple of predicates"),
    ({}, "QUERY_NOT_BOUNDED", "key_filter must be a list or tuple of predicates"),
    ([None], "QUERY_NOT_BOUNDED", "a key_filter predicate is refused: BAD_TYPE"),
    ((KeyPredicate(column="ticker", operator="in", values=None),), "QUERY_NOT_BOUNDED",
     "a key_filter predicate is refused: BAD_TYPE"),
    ((KeyPredicate(column="ticker", operator="in", values=("A", "B", "D")),
      KeyPredicate(column="ticker", operator="in", values=("B", "C"))), "QUERY_NOT_BOUNDED",
     "the same column is filtered twice"),
    ((KeyPredicate(column="event_date", operator="eq", values=("2024-05-02",)),),
     "CONTRACT_MISMATCH", "'event_date' is not a filterable column"),
])
def test_pinned_bounds_preserve_invalid_filter_refusals(tmp_path, key_filter, code, message):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        for pinned in (False, True):
            repository = Repository(conn, store, pin_snapshots=pinned)
            with pytest.raises(DataError) as failure:
                repository.scan_population_bound(
                    snapshot.snapshot_id, table_name="computed_moves",
                    table_contract_ref=ref, key_filter=key_filter)
            assert failure.value.code == code
            assert failure.value.problem.message == message
            assert failure.value.problem.retryable is False
    finally:
        conn.close()


@pytest.mark.parametrize("values,start,end,expected", [
    ((), "2024-05-01", "2024-05-04", 10), (("A", "D"), "2024-05-01", "2024-05-04", 5),
    (("A", "D"), "2024-05-01", "2024-05-06", 7),
    (("A", "D"), "2024-05-03", "2024-05-06", 5), (("A", "D"), "2024-05-06", None, 0),
])
def test_pinned_bounds_preserve_time_filters(tmp_path, monkeypatch, values, start, end, expected):
    conn, store, ref, snapshot = _build(tmp_path, MULTI_FRAGMENT)
    try:
        key_filter = (KeyPredicate(column="ticker", operator="in", values=values),) if values else ()
        interval = TimeInterval(column="event_date", start_inclusive=start,
                                end_exclusive=end)
        calls = _spy_plans(monkeypatch)
        for pinned in (False, True):
            repository = Repository(conn, store, pin_snapshots=pinned)
            assert repository.scan_population_bound(
                snapshot.snapshot_id, table_name="computed_moves", table_contract_ref=ref,
                key_filter=key_filter, time_interval=interval) == expected
            assert [record.partition_key for record in calls[-1]] == [
                ticker for ticker, _ in MULTI_FRAGMENT
                if not pinned or not values or ticker in values]
    finally:
        conn.close()


def test_time_interval_without_a_partition_predicate_plans_every_fragment(tmp_path, monkeypatch):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        interval = TimeInterval(column="event_date", start_inclusive="2024-05-01",
                                end_exclusive="2024-05-04")
        request = _request(snapshot, ref, time_interval=interval, max_result_rows=3)
        calls = _spy_plans(monkeypatch)
        repository = Repository(conn, store, pin_snapshots=True)
        assert _rows(repository, request) == [
            {"ticker": "A", "event_date": "2024-05-01"},
            {"ticker": "B", "event_date": "2024-05-02"},
            {"ticker": "C", "event_date": "2024-05-03"},
        ]
        assert [len(records) for records in calls] == [len(TICKERS)]
    finally:
        conn.close()
