"""Synthetic native opt-in metadata, immutable defaults and partial refreshes."""
import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import DataQuery, KeyPredicate
from engine.v2.data.computed_move_bounds import COLUMNS, CONTRACT, TABLE, source_bounds
from engine.v2.data.computed_moves import build_event_rows, session_move
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.ops import computed_move_bounds_store as bounds
from engine.v2.ops import computed_moves_store as producer
from engine.v2.ops.errors import OpsError
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
    _write_input,
)


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(producer, "MIN_SCOREABLE", 5)
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    events = [_event_row(ticker, day) for ticker in ("AAAA", "BBBB") for day in _EVENT_DAYS]
    _build_parent(conn, clock, store, events_rows=events)
    fetcher = _CountingFetcher(_closes_csv())
    attempt = 0

    def run(tickers=("AAAA", "BBBB"), *, enabled=False):
        nonlocal attempt
        attempt += 1
        monkeypatch.setattr(producer, "target_tickers_from_snapshot", lambda *a, **k: (list(tickers), {}))
        head = _head_row(conn)
        root = tmp_path / f"attempt-{attempt}"
        _write_input(root, catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path, head=head)
        parameters = _parameters(head, expected_ids=tickers,
            catalog_path=tmp_path / "ops.sqlite", objects_root=tmp_path)
        result = producer.run_computed_moves_refresh(parameters, root, as_of=_AS_OF,
            fetcher=fetcher, include_source_bounds=enabled)
        repo = Repository(conn, store)
        return result, repo, repo.resolve_full(_head_row(conn)["snapshot_id"])

    yield conn, store, fetcher, run
    conn.close()


def read_bounds(repository, snapshot):
    records = repository.fragment_records(snapshot, TABLE)
    count = sum(r.row_count for r in records)
    request = DataQuery(snapshot_id=snapshot.snapshot_id, table_contract_ref=bounds.REF,
        columns=COLUMNS, key_filter=(KeyPredicate(column="ticker", operator="in",
            values=tuple(sorted({r.partition_key for r in records}))),), order_by=CONTRACT.primary_key,
        max_batch_rows=max(1, count), max_result_rows=count)
    return [row for batch in repository.scan(request, table_name=TABLE) for row in batch.to_pylist()]


def test_opt_in_adds_exact_bound_metadata_without_changing_live_objects(capture):
    _, _, fetcher, run = capture
    _, _, live = run()
    assert TABLE not in live.table_manifests
    live_objects = [r.object_ref for r in live.records if r.table_contract_ref.contract_id == "computed_moves.v5"]
    _, repository, first = run(enabled=True)
    assert first.snapshot != live.snapshot
    assert [r.object_ref for r in first.records if r.table_contract_ref.contract_id == "computed_moves.v5"] == live_objects
    rows = read_bounds(repository, first.snapshot)
    outcomes = {r.primary_key_min: r for r in first.records if r.table_contract_ref.contract_id == "computed_moves.v5"}
    assert len(rows) == len(outcomes)
    for row in rows:
        record = outcomes[row["ticker"], row["event_date"]]
        assert row["outcome_object_id"] == record.object_ref.object_id
        assert row["outcome_content_hash"] == record.object_ref.content_hash
        assert row["pre_close_date"] < row["post_close_date"]
        assert row["session"] == "AMC"
    _, repository, repeated = run(enabled=True)
    assert repeated.snapshot == first.snapshot
    assert read_bounds(repository, repeated.snapshot) == rows
    assert fetcher.calls == ["AAAA", "BBBB"]


def test_default_off_partial_refresh_drops_only_rewritten_sidecar(capture):
    _, _, _, run = capture
    _, repository, original = run(enabled=True)
    old = read_bounds(repository, original.snapshot)
    _, repository, partial = run(("AAAA",), enabled=False)
    remaining = read_bounds(repository, partial.snapshot)
    assert remaining == [r for r in old if r["ticker"] == "BBBB"]
    assert read_bounds(repository, original.snapshot) == old
    _, repository, complete = run(("AAAA",), enabled=True)
    assert read_bounds(repository, complete.snapshot) == old


@pytest.mark.parametrize("session,expected", [
    ("BMO", ("2024-01-05", "2024-01-08")),
    ("AMC", ("2024-01-08", "2024-01-10")),
])
def test_source_dates_follow_actual_missing_bar_sequence(session, expected):
    sd = np.array(["2024-01-05", "2024-01-08", "2024-01-10"], dtype="datetime64[ns]")
    assert source_bounds(sd, "2024-01-08", session) == expected
    assert session_move(sd, np.array([100., 110., 121.]), np.datetime64("2024-01-08"), session) == pytest.approx(10.)


@pytest.mark.parametrize("event,session", [
    ("2024-01-08", None), ("2024-01-08", "TYPO"), ("2024-01-08", []),
    ("2024-01-08", pd.NA), ("bad", "AMC"), (None, "BMO"), ([], "AMC"),
    ("2024-02-30", "BMO"), ("20240108", "AMC"), ("NaT", "AMC"),
    ("2024-01-08T00:00:00", "BMO"), ("0001-01-01", "AMC"),
])
def test_source_bounds_refuses_invalid_scalars_before_bracket_lookup(event, session, monkeypatch):
    from engine.v2.data import computed_move_bounds
    from engine.v2.data.errors import DataError

    def forbidden(*args):
        pytest.fail("invalid scalar reached native calendar lookup")

    monkeypatch.setattr(computed_move_bounds, "source_session_indices", forbidden)
    with pytest.raises(DataError) as error:
        source_bounds(object(), event, session)
    assert error.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("days,event,session", [
    (["2024-01-01", "2024-01-10"], "2024-01-10", "BMO"),
    (["2024-01-01", "2024-01-02"], "2024-01-01", "BMO"),
    (["2024-01-01", "2024-01-02"], "2024-01-02", "AMC"),
])
def test_missing_or_overwide_window_has_no_bounds(days, event, session):
    assert source_bounds(np.array(days, dtype="datetime64[ns]"), event, session) == (None, None)


def test_final_skipped_classification_never_creates_dates_from_a_bracket(tmp_path):
    sd = pd.bdate_range("2024-01-01", periods=30).to_numpy()
    events = pd.DataFrame({"event_date": pd.to_datetime(sd[5:10]), "session": ["BMO"] * 5,
                           "src_orats": [False] * 5})
    daily = pd.DataFrame({"date": pd.to_datetime(sd), "implied_move": [2.] * len(sd), "src_iv": ["x"] * len(sd)})
    rows = build_event_rows("AAAA", events, sd, np.arange(len(sd)) + 100., daily,
        computed_at="2024-04-01", source_hash="a" * 64, capture_id="capture_test", min_scoreable=5)
    assert all(r["skipped"] for r in rows)
    store = ArtifactStore(tmp_path)
    outcomes = producer._write_ticker_fragments(store, "AAAA", rows, request_hash="sha256:" + "b" * 64)
    records = bounds.write_bounds(store, rows, events, sd, outcomes)
    import pyarrow.parquet as pq

    from engine.v2.data.objects import verify_object_path
    output = pq.read_table(verify_object_path(store, records[0].object_ref)).to_pylist()
    assert all(r["pre_close_date"] is None and r["post_close_date"] is None for r in output)
    assert len(output) == len(rows)


@pytest.mark.parametrize("invalid", [1, "yes", None])
def test_invalid_opt_in_refuses_before_io(invalid):
    with pytest.raises(OpsError) as error:
        producer.run_computed_moves_refresh(None, "/nonexistent", as_of=_AS_OF, include_source_bounds=invalid)
    assert error.value.code == "INVALID_REQUEST"


def test_opt_in_below_bar_metadata_is_null_without_fetching(capture, monkeypatch):
    _, _, fetcher, run = capture
    monkeypatch.setattr(producer, "MIN_SCOREABLE", 100)
    _, repository, result = run((), enabled=True)
    rows = read_bounds(repository, result.snapshot)
    assert len(rows) == 2 * len(_EVENT_DAYS)
    assert all(row["pre_close_date"] is None and row["post_close_date"] is None for row in rows)
    assert fetcher.calls == []


def test_producer_refuses_availability_that_disagrees_with_actual_source(tmp_path):
    from engine.v2.data.errors import DataError
    sd = pd.bdate_range("2024-01-01", periods=30).to_numpy()
    events = pd.DataFrame({"event_date": pd.to_datetime(sd[5:10]), "session": ["BMO"] * 5,
                           "src_orats": [True] * 5})
    daily = pd.DataFrame({"date": pd.to_datetime(sd), "implied_move": [2.] * len(sd), "src_iv": ["x"] * len(sd)})
    rows = build_event_rows("AAAA", events, sd, np.arange(len(sd)) + 100., daily,
        computed_at="2024-04-01", source_hash="a" * 64, capture_id="capture_test", min_scoreable=5)
    store = ArtifactStore(tmp_path)
    outcomes = producer._write_ticker_fragments(store, "AAAA", rows, request_hash="sha256:" + "b" * 64)
    rows[0]["available_as_of_date"] = "2024-02-01"
    with pytest.raises(DataError, match="native outcome and source bounds differ"):
        bounds.write_bounds(store, rows, events, sd, outcomes)
