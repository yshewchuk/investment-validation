"""``Repository(pin_snapshots=True)`` reads a pinned snapshot's catalog membership once.

Pins the read count (the native score batch made thousands of scan calls on one
snapshot and re-read every table's fragment membership on each of them).
"""
import pytest

from engine.v2.contracts.data import DataQuery, KeyPredicate
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT as CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_ref_for,
    publish_and_inspect,
)


def _move(ticker, day):
    return dict(ticker=ticker, event_date=day, realized_move_pct=1.0,
                available_as_of_date="2024-05-05", implied_move_pct=2.0, quarter_ordinal=1,
                skipped=False, computed_at="2024-05-05T00:00:00Z", source_hash="synthetic",
                capture_id="synthetic")


def _build(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    ref = contract_ref_for(CONTRACT)
    records = [publish_and_inspect(store, CONTRACT, ref, [_move(t, d)], t)
               for t, d in [("A", "2024-05-01"), ("B", "2024-05-02"), ("C", "2024-05-03")]]
    snapshot = commit_tables(conn, clock, {"computed_moves": records},
                             {"computed_moves": CONTRACT}, store=store)
    return conn, store, ref, snapshot


def _count_membership_reads(monkeypatch):
    calls = []
    original = Repository._records

    def counting(self, conn, dataset_version_id, contract_ref):
        calls.append(dataset_version_id)
        return original(self, conn, dataset_version_id, contract_ref)

    monkeypatch.setattr(Repository, "_records", counting)
    return calls


def _scan_and_bound(repository, snapshot, ref):
    key_filter = (KeyPredicate(column="ticker", operator="eq", values=("B",)),)
    bound = repository.scan_population_bound(
        snapshot.snapshot_id, table_name="computed_moves", table_contract_ref=ref,
        key_filter=key_filter)
    request = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=ref,
                        columns=("ticker",), key_filter=key_filter, order_by=CONTRACT.primary_key,
                        max_batch_rows=1, max_result_rows=bound)
    rows = [row["ticker"] for batch in repository.scan(request, table_name="computed_moves")
            for row in batch.to_pylist()]
    return bound, rows


def test_pinned_repository_reads_membership_once_across_many_calls(tmp_path, monkeypatch):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        calls = _count_membership_reads(monkeypatch)
        repository = Repository(conn, store, pin_snapshots=True)
        first = _scan_and_bound(repository, snapshot, ref)
        after_first = len(calls)  # one resolve walk + one table-membership read
        assert after_first <= 2
        results = [_scan_and_bound(repository, snapshot, ref) for _ in range(4)]
        assert results == [first] * 4 and first == (1, ["B"])
        assert len(calls) == after_first
    finally:
        conn.close()


def test_default_repository_still_reads_membership_on_every_call(tmp_path, monkeypatch):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        calls = _count_membership_reads(monkeypatch)
        repository = Repository(conn, store)
        for _ in range(5):
            assert _scan_and_bound(repository, snapshot, ref) == (1, ["B"])
        assert len(calls) >= 10
    finally:
        conn.close()


def test_pinned_repository_never_caches_an_unknown_snapshot(tmp_path):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        repository = Repository(conn, store, pin_snapshots=True)
        for _ in range(2):
            with pytest.raises(DataError, match="SNAPSHOT_NOT_FOUND"):
                repository.resolve("snap_does_not_exist")
        assert repository.resolve(snapshot.snapshot_id) == snapshot
    finally:
        conn.close()


def test_mutating_a_returned_ref_cannot_change_later_scans(tmp_path):
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        repository = Repository(conn, store, pin_snapshots=True)
        returned = repository.resolve(snapshot.snapshot_id)
        try:
            returned.table_versions.clear()
        except (TypeError, AttributeError):
            pass  # an immutable mapping is equally safe
        again = repository.resolve(snapshot.snapshot_id)
        assert again == snapshot
        assert again is not returned
        assert _scan_and_bound(repository, snapshot, ref) == (1, ["B"])
    finally:
        conn.close()
