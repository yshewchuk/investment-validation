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


def _move(ticker, day):
    return dict(ticker=ticker, event_date=day, realized_move_pct=1.0,
                available_as_of_date="2024-05-05", implied_move_pct=2.0, event_ordinal=0,
                admitted=False, mean_prior_or_implied=None, n_prior_implied=0,
                skipped=False, computed_at="2024-05-05T00:00:00Z", source_hash="synthetic",
                capture_id="synthetic")


def _build(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    ref = contract_ref_for(CONTRACT)
    records = [publish_and_inspect(store, CONTRACT, ref, [_move(t, d)], t)
               for t, d in TICKERS]
    snapshot = commit_tables(conn, clock, {"computed_moves": records},
                             {"computed_moves": CONTRACT}, store=store)
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


def test_corrupt_bound_on_an_unfiltered_fragment_refuses_every_pinned_scan_and_is_never_cached(
        tmp_path, monkeypatch):
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
                _rows(repository, request)
            assert failure.value.code == "MANIFEST_CORRUPT"
            assert repository._pinned_tables == {}
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
