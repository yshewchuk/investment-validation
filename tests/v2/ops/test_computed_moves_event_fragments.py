"""Event-isolated publication through the real synthetic capture/commit path."""
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.contracts import DataQuery, KeyPredicate, TimeInterval
from engine.v2.data import manifests, objects
from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT
from engine.v2.data.errors import DataError, fail
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import computed_moves_store as producer
from tests.data_scan_support import contract_ref_for, publish_and_inspect
from tests.ops_support import catalog
from tests.v2.ops.test_v2_ops_computed_moves_store import (
    _AS_OF,
    _EVENT_DAYS,
    _build_parent,
    _closes_csv,
    _CountingFetcher,
    _event_row,
    _head_row,
    _parameters,
    _series_csv,
    _write_input,
)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events = [_event_row(ticker, day) for ticker in ("AAAA", "BBBB") for day in _EVENT_DAYS]
    _build_parent(conn, clock, store, events_rows=events)
    fetcher = _CountingFetcher(_closes_csv())
    attempt = 0

    def run(tickers=("AAAA", "BBBB"), *, source=fetcher, cached=True):
        nonlocal attempt
        attempt += 1
        monkeypatch.setattr(producer, "target_tickers_from_snapshot", lambda *a, **k: (list(tickers), {}))
        head = _head_row(conn)
        root = tmp_path / f"attempt-{attempt}"
        _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
        parameters = _parameters(head, expected_ids=tickers,
            catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
        return producer.run_computed_moves_refresh(parameters, root, as_of=_AS_OF,
                                                   fetcher=source, use_cached_receipts=cached)

    yield conn, store, fetcher, run
    conn.close()


def _resolved(conn, store):
    repository = Repository(conn, store)
    resolved = repository.resolve_full(_head_row(conn)["snapshot_id"])
    records = repository.fragment_records(resolved.snapshot, "computed_moves")
    return repository, resolved, records


def _scan(repository, snapshot, ticker, day=None):
    contract = COMPUTED_MOVES_CONTRACT
    keys = (KeyPredicate(column="ticker", operator="eq", values=(ticker,)),)
    interval = None if day is None else TimeInterval(
        column="event_date", start_inclusive=day.isoformat(),
        end_exclusive=(day + timedelta(days=1)).isoformat())
    bound = repository.scan_population_bound(snapshot.snapshot_id, table_name="computed_moves",
        table_contract_ref=contract_ref_for(contract), key_filter=keys, time_interval=interval)
    query = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=contract_ref_for(contract),
        columns=tuple(column.name for column in contract.columns), key_filter=keys,
        time_interval=interval, order_by=contract.primary_key, max_batch_rows=max(1, bound),
        max_result_rows=bound)
    return [row for batch in repository.scan(query, table_name="computed_moves") for row in batch.to_pylist()]


def test_capture_pins_singleton_bounds_and_exact_scan_never_opens_siblings(capture, monkeypatch):
    conn, store, _, run = capture
    assert run().status == "complete"
    repository, resolved, records = _resolved(conn, store)
    assert len(records) == 2 * len(_EVENT_DAYS)
    for record in records:
        assert record.row_count == 1
        assert record.primary_key_min == record.primary_key_max
        assert record.partition_key == record.primary_key_min[0]
        assert pd.Timestamp(record.time_min).date().isoformat() == record.primary_key_min[1]
        assert record.time_min == record.time_max
    selected = records[1]
    protected = {record.object_ref.object_id for record in records if record != selected}
    original = objects.verify_object_path
    opened = []

    def verify(store_, ref):
        assert ref.object_id not in protected, "a non-requested synthetic outcome object was touched"
        opened.append(ref.object_id)
        return original(store_, ref)

    monkeypatch.setattr(objects, "verify_object_path", verify)
    rows = _scan(repository, resolved.snapshot, "AAAA", _EVENT_DAYS[1].date())
    assert len(rows) == 1
    assert rows[0]["event_date"] == _EVENT_DAYS[1].date().isoformat()
    assert rows[0]["realized_move_pct"] == pytest.approx(100 / 106)
    assert opened == [selected.object_ref.object_id]


def test_documented_rewrite_invariant_distinguishes_event_and_ticker_fragments():
    document = Path(__file__).parents[3] / "engine/v2/data/ARCHITECTURE.md"
    invariant = document.read_text().split("**Whole-partition rewrite, no legacy append order**", 1)[1]
    invariant = invariant.split("\n- **", 1)[0]
    assert "`price_history` uses one fragment" in invariant
    assert "`computed_moves` captures use ordered, non-overlapping singleton event fragments" in invariant
    assert "Older snapshots may retain one whole-ticker fragment" in invariant


def test_event_layout_preserves_whole_partition_logical_hash(capture):
    conn, store, _, run = capture
    run(("AAAA",))
    repository, resolved, records = _resolved(conn, store)
    rows = _scan(repository, resolved.snapshot, "AAAA")
    whole = publish_and_inspect(store, COMPUTED_MOVES_CONTRACT,
        contract_ref_for(COMPUTED_MOVES_CONTRACT), rows, "AAAA")
    manifest = resolved.table_manifests["computed_moves"]
    assert manifest.partition_logical_hashes == {"AAAA": whole.logical_content_hash}
    old_layout = manifests.dataset_manifest(contract_ref_for(COMPUTED_MOVES_CONTRACT), [whole],
        knowledge_mode="reconstructed", coverage_receipt_refs=(), availability_evidence_refs=())
    assert manifest.logical_content_hash == old_layout.logical_content_hash
    assert manifest.row_count == old_layout.row_count == len(_EVENT_DAYS)
    assert manifest.dataset_version_ref.dataset_version_id != old_layout.dataset_version_ref.dataset_version_id
    manifests.verify_partition_hashes(store, manifest, records, COMPUTED_MOVES_CONTRACT)


def test_identical_cached_retry_preserves_every_fragment_and_head(capture):
    conn, store, fetcher, run = capture
    run()
    before = _head_row(conn)
    _, first, records = _resolved(conn, store)
    result = run()
    assert result.status == "noop"
    assert not result.coverage_advanced
    assert _head_row(conn)["snapshot_id"] == before["snapshot_id"]
    assert _head_row(conn)["generation"] == before["generation"]
    _, second, retry_records = _resolved(conn, store)
    assert second.snapshot == first.snapshot
    assert retry_records == records
    assert fetcher.calls == ["AAAA", "BBBB"]
    assert conn.execute("SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_rewrite_keeps_unselected_fragments_and_hash_without_opening_them(capture, monkeypatch, legacy):
    conn, store, _, run = capture
    write = producer._write_ticker_fragments
    if legacy:
        def old_writer(store_, ticker, rows, *, request_hash):
            if ticker != "BBBB":
                return write(store_, ticker, rows, request_hash=request_hash)
            return (publish_and_inspect(store_, COMPUTED_MOVES_CONTRACT,
                contract_ref_for(COMPUTED_MOVES_CONTRACT), rows, ticker,
                input_receipt_refs=(), import_request_hash=request_hash),)
        monkeypatch.setattr(producer, "_write_ticker_fragments", old_writer)
    run()
    monkeypatch.setattr(producer, "_write_ticker_fragments", write)
    _, first, records = _resolved(conn, store)
    kept = tuple(record for record in records if record.partition_key == "BBBB")
    protected = {record.object_ref.object_id for record in kept}
    original = objects.verify_object_path

    def verify(store_, ref):
        assert ref.object_id not in protected, "an inherited outcome object was reopened"
        return original(store_, ref)

    monkeypatch.setattr(objects, "verify_object_path", verify)
    assert run(("AAAA",), source=_CountingFetcher(_series_csv(200)), cached=False).status == "complete"
    _, second, updated = _resolved(conn, store)
    assert tuple(record for record in updated if record.partition_key == "BBBB") == kept
    assert second.table_manifests["computed_moves"].partition_logical_hashes.get("BBBB") == (
        first.table_manifests["computed_moves"].partition_logical_hashes.get("BBBB"))
    assert len(kept) == (1 if legacy else len(_EVENT_DAYS))
    assert second.snapshot.table_versions["earnings_events"] == first.snapshot.table_versions["earnings_events"]
    assert Repository(conn, store).resolve(first.snapshot.snapshot_id) == first.snapshot


def test_refresh_changes_old_physical_layout_without_changing_logical_content(capture, monkeypatch):
    conn, store, fetcher, run = capture
    write = producer._write_ticker_fragments

    def old_writer(store_, ticker, rows, *, request_hash):
        return (publish_and_inspect(store_, COMPUTED_MOVES_CONTRACT,
            contract_ref_for(COMPUTED_MOVES_CONTRACT), rows, ticker,
            input_receipt_refs=(), import_request_hash=request_hash),)

    monkeypatch.setattr(producer, "_write_ticker_fragments", old_writer)
    run(("AAAA",))
    repository, old, old_records = _resolved(conn, store)
    assert len(old_records) == 1
    monkeypatch.setattr(producer, "_write_ticker_fragments", write)
    assert run(("AAAA",)).status == "complete"
    _, new, new_records = _resolved(conn, store)
    assert len(new_records) == len(_EVENT_DAYS)
    assert old.table_manifests["computed_moves"].logical_content_hash == (
        new.table_manifests["computed_moves"].logical_content_hash)
    assert old.snapshot.snapshot_id != new.snapshot.snapshot_id
    assert repository.fragment_records(old.snapshot, "computed_moves") == old_records
    assert fetcher.calls == ["AAAA"]


@pytest.mark.parametrize("point", ["object_preparation", "catalog_transaction"])
def test_partial_failure_publishes_no_generation_and_cached_retry_recovers(capture, monkeypatch, point):
    conn, store, fetcher, run = capture
    before = _head_row(conn)
    if point == "object_preparation":
        original = objects.inspect_fragment
        calls = 0

        def failure(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise fail("OBJECT_CORRUPT", "synthetic event preparation failure")
            return original(*args, **kwargs)

        target, name = objects, "inspect_fragment"
    else:
        original = producer._insert_captures

        def failure(*args, **kwargs):
            original(*args, **kwargs)
            raise fail("OBJECT_CORRUPT", "synthetic failure after capture-log insert")

        target, name = producer, "_insert_captures"
    monkeypatch.setattr(target, name, failure)
    with pytest.raises(DataError, match="OBJECT_CORRUPT"):
        run(("AAAA",))
    assert _head_row(conn) == before
    assert conn.execute("SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM data_fragments WHERE contract_id=?",
                        (COMPUTED_MOVES_CONTRACT.contract_id,)).fetchone()[0] == 0
    monkeypatch.setattr(target, name, original)
    assert run(("AAAA",)).status == "complete"
    assert fetcher.calls == ["AAAA"]
    _, _, records = _resolved(conn, store)
    assert len(records) == len(_EVENT_DAYS)
    assert conn.execute("SELECT COUNT(*) FROM data_computed_moves_captures").fetchone()[0] == 1
