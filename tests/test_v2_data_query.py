"""D05, D06: bounded Arrow scans — phase-2 guide §5.3, §8.2, §12.

D05 drives the query validator's refusals; D06 drives real synthetic scans
(real Parquet, real catalog, real ``ArtifactStore``) for exact-match rows in
deterministic key order, typed-null synthesis for a declared-missing nullable
column, batch/result-row bounding, and fragment pruning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import (  # noqa: E402
    ChainQuery,
    ColumnContract,
    DataQuery,
    DependencyPlan,
    KeyPredicate,
    TableContract,
    TableContractRef,
    TimeInterval,
)
from engine.v2.data import objects, query as query_mod  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.objects import partition_logical_hash  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from tests.data_scan_support import (  # noqa: E402
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    hand_built_record,
    publish_and_inspect,
    table_from_rows,
)

_SEC = contract_for("securities")
_SEC_REF = contract_ref_for(_SEC)
_DM = contract_for("daily_market")
_DM_REF = contract_ref_for(_DM)
_OC = contract_for("option_chains")
_OC_REF = contract_ref_for(_OC)
_EE = contract_for("earnings_events")
_EE_REF = contract_ref_for(_EE)


def _option_chains_row(ticker: str, year: int) -> dict:
    from datetime import datetime
    return dict(ticker=ticker, obs_date=datetime(year, 1, 2), year=year, expiry=datetime(year, 2, 16),
               dte=45, strike=100.0, right="C", bid=1.0, ask=1.2, mid=1.1, iv=30.0, delta=0.5,
               spot=100.0, src="orats", src_file="f.parquet", chain_kind="entry", volume=None,
               open_interest=None, bid_size=None, ask_size=None, quote_repaired=False)


def _securities_row(ticker: str, year: int) -> dict:
    return dict(ticker=ticker, year=year, first_date=None, last_date=None, mcap_usd=1.5e9,
               mcap_log=21.1, mcap_raw=1.5, mcap_unit_era="billions", mcap_quantized=False,
               n_obs=250, src="orats")


def _securities_snapshot(tmp_path, tickers=("AAA", "BBB"), years=(2024, 2025)):
    conn, clock, store = catalog_and_store(tmp_path)
    records = []
    for year in years:
        rows = [_securities_row(t, year) for t in tickers]
        records.append(publish_and_inspect(store, _SEC, _SEC_REF, rows, str(year)))
    snap = commit_tables(conn, clock, {"securities": records}, {"securities": _SEC})
    return conn, store, snap


def _daily_market_row(ticker: str, year: int, day: int, *, omit_iv30: bool = False) -> dict:
    from datetime import datetime
    row = dict(ticker=ticker, date=datetime(year, 1, day), year=year, spot=100.0, iv10=30.0,
              iv30=32.0, exern_iv10=29.0, exern_iv30=31.0, implied_move=5.0,
              implied_reconstructed=False, rvol30=28.0, skew=1.1, contango=0.5, fwd90_30=33.0,
              fexern90_30=34.0, iee=0.2, mcap_usd=1e9, mcap_log=20.7,
              mcap_asof=datetime(year, 1, day), mcap_age_days=0.0, src_spot="orats",
              src_iv="orats", src_mcap="orats")
    if omit_iv30:
        row.pop("iv30")
    return row


def _basic_query(**overrides) -> dict:
    base = dict(table_contract_ref=_SEC_REF, columns=("ticker", "year"),
               key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
               order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# D05: refusals
# --------------------------------------------------------------------------


def test_implicit_latest_snapshot_id_is_not_found(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id="latest", **_basic_query())
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_unknown_snapshot_id_is_not_found(tmp_path):
    _conn, store, _snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id="snap_" + "0" * 32, **_basic_query())
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "SNAPSHOT_NOT_FOUND"


def test_mismatched_table_contract_ref_is_contract_mismatch(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(table_contract_ref=_DM_REF))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_unknown_table_name_is_contract_mismatch(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query())
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="daily_market"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_empty_projection_is_query_not_bounded(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(columns=()))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"


def test_predicate_on_non_filterable_column_is_contract_mismatch(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(
        key_filter=(KeyPredicate(column="mcap_usd", operator="eq", values=("1.5e9",)),)))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_order_by_not_the_full_primary_key_is_contract_mismatch(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(order_by=("year", "ticker")))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_missing_bounds_is_query_not_bounded(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(key_filter=(), time_interval=None))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"


def test_limits_above_contract_cap_is_query_not_bounded(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id,
                      **_basic_query(max_batch_rows=60000, max_result_rows=60000))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"


def test_time_interval_column_must_be_observation_time_column(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    records = [publish_and_inspect(store, _DM, _DM_REF, [_daily_market_row("AAA", 2024, 2)], "2024")]
    snap = commit_tables(conn, clock, {"daily_market": records}, {"daily_market": _DM})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(), time_interval=TimeInterval(column="ticker", start_inclusive="2024-01-01"),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="daily_market"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_scan_without_a_store_refuses(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query())
    with pytest.raises(RuntimeError):
        list(repo.scan(query, table_name="securities"))


# --------------------------------------------------------------------------
# D06: real synthetic scans
# --------------------------------------------------------------------------


def test_projection_filter_and_order_across_multiple_fragments(tmp_path):
    """Two ``securities`` fragments (2024, 2025), each with two tickers;
    filtering to one ticker returns exactly its two rows, ticker-then-year."""
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),)))
    rows = [row for batch in repo.scan(query, table_name="securities") for row in batch.to_pylist()]
    assert rows == [{"ticker": "AAA", "year": 2024}, {"ticker": "AAA", "year": 2025}]


def test_in_predicate_returns_matching_rows_in_key_order(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAA", "BBB")),)))
    rows = [row for batch in repo.scan(query, table_name="securities") for row in batch.to_pylist()]
    assert rows == [{"ticker": "AAA", "year": 2024}, {"ticker": "AAA", "year": 2025},
                    {"ticker": "BBB", "year": 2024}, {"ticker": "BBB", "year": 2025}]


def test_time_interval_across_fragments_with_ticker_pinned(tmp_path):
    """order_by leads with ``ticker``; pinning it via ``eq`` makes fragment-
    sequential (year-ascending) reading globally correct for ``date`` too."""
    conn, clock, store = catalog_and_store(tmp_path)
    records = [
        publish_and_inspect(store, _DM, _DM_REF,
                            [_daily_market_row("AAA", 2024, 2), _daily_market_row("AAA", 2024, 3)], "2024"),
        publish_and_inspect(store, _DM, _DM_REF,
                            [_daily_market_row("AAA", 2025, 2), _daily_market_row("AAA", 2025, 3)], "2025"),
    ]
    snap = commit_tables(conn, clock, {"daily_market": records}, {"daily_market": _DM})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        time_interval=TimeInterval(column="date", start_inclusive="2024-01-01",
                                   end_exclusive="2026-01-01"),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    rows = [row for batch in repo.scan(query, table_name="daily_market") for row in batch.to_pylist()]
    dates = [r["date"] for r in rows]
    assert dates == sorted(dates)
    assert len(rows) == 4


def test_hidden_primary_key_columns_dropped_when_not_projected(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(columns=("mcap_usd",)))
    for batch in repo.scan(query, table_name="securities"):
        assert batch.schema.names == ["mcap_usd"]


def _event_row(event_id: str, ticker: str, year: int, day: int) -> dict:
    from datetime import datetime
    return dict(event_id=event_id, ticker=ticker, event_date=datetime(year, 1, day), year=year,
               session="BMO", session_src="orats", annc_tod=None, src_orats=True, src_oquants=True,
               src_nasdaq=False, src_yfinance=False, date_agree=True, date_conflict=False,
               updated_at=None, event_cluster_id=None, claim_count=None, reconciliation=None)


def test_narrow_projection_with_hidden_filter_columns_matches_full_projection(tmp_path):
    """Review P2-C05, decision 1's own proof: projecting only ``event_id``
    with a ticker filter plus a date interval must return exactly the same
    ``event_id``s, in the same order, as the full projection with the same
    filters — even though neither ``ticker`` (the predicate column) nor
    ``event_date`` (the time_interval column) is in the narrow projection.
    Before the fix, ``_execute_scan``'s ``needed`` tuple omitted both, so
    ``query_mod.row_matches`` read them as absent (not merely ``None``) and
    every row was silently excluded. A multi-fragment partition (2024 split
    into two objects) exercises the same bug across a fragment boundary."""
    conn, clock, store = catalog_and_store(tmp_path)
    frag_a = publish_and_inspect(store, _EE, _EE_REF,
                                 [_event_row("AAA_2024-01-02", "AAA", 2024, 2)], "2024")
    frag_b = publish_and_inspect(store, _EE, _EE_REF,
                                 [_event_row("AAA_2024-01-09", "AAA", 2024, 9),
                                  _event_row("BBB_2024-01-03", "BBB", 2024, 3)], "2024")
    part_hash = partition_logical_hash(store, [frag_a.object_ref, frag_b.object_ref], _EE, _EE_REF,
                                       "2024")
    other_year = publish_and_inspect(store, _EE, _EE_REF, [_event_row("AAA_2025-01-02", "AAA", 2025, 2)],
                                     "2025")
    snap = commit_tables(conn, clock, {"earnings_events": [frag_a, frag_b, other_year]},
                         {"earnings_events": _EE}, store=store,
                         partition_logical_hashes={"earnings_events": {"2024": part_hash}})
    repo = Repository(conn, store)
    full_columns = tuple(c.name for c in _EE.columns)
    filters = dict(
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        time_interval=TimeInterval(column="event_date", start_inclusive="2024-01-01",
                                   end_exclusive="2025-01-01"))
    narrow = DataQuery(snapshot_id=snap.snapshot_id, table_contract_ref=_EE_REF, columns=("event_id",),
                       order_by=("event_id",), max_batch_rows=10, max_result_rows=10, **filters)
    full = DataQuery(snapshot_id=snap.snapshot_id, table_contract_ref=_EE_REF, columns=full_columns,
                     order_by=("event_id",), max_batch_rows=10, max_result_rows=10, **filters)
    narrow_ids = [r["event_id"] for b in repo.scan(narrow, table_name="earnings_events")
                 for r in b.to_pylist()]
    full_ids = [r["event_id"] for b in repo.scan(full, table_name="earnings_events")
               for r in b.to_pylist()]
    assert narrow_ids == full_ids == ["AAA_2024-01-02", "AAA_2024-01-09"]


def test_nullable_column_absent_from_an_old_fragment_is_typed_null(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    old = publish_and_inspect(store, _DM, _DM_REF, [_daily_market_row("AAA", 2024, 2, omit_iv30=True)],
                              "2024", omit=frozenset({"iv30"}))
    new = publish_and_inspect(store, _DM, _DM_REF, [_daily_market_row("AAA", 2025, 2)], "2025")
    snap = commit_tables(conn, clock, {"daily_market": [old, new]}, {"daily_market": _DM})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date", "iv30"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    rows = [row for batch in repo.scan(query, table_name="daily_market") for row in batch.to_pylist()]
    assert rows[0]["iv30"] is None
    assert rows[1]["iv30"] == 32.0


def test_required_column_missing_from_a_fragment_is_contract_mismatch(tmp_path):
    """A hand-built record whose real physical file drops a *required*
    column — unreachable through the normal ``inspect_fragment`` publish
    path, which already refuses this at ingest time, so this fixture
    bypasses it directly (see ``data_scan_support.hand_built_record``)."""
    conn, clock, store = catalog_and_store(tmp_path)
    table = table_from_rows(_OC, [_option_chains_row("AAA", 2024)], omit=frozenset({"dte"}))
    record = hand_built_record(
        store, _OC, _OC_REF, table, partition_key="2024", row_count=1,
        primary_key_min=("AAA", "2024-01-02T00:00:00.000000", "2024-02-16T00:00:00.000000", 100.0, "C"),
        primary_key_max=("AAA", "2024-01-02T00:00:00.000000", "2024-02-16T00:00:00.000000", 100.0, "C"))
    snap = commit_tables(conn, clock, {"option_chains": [record]}, {"option_chains": _OC})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_OC_REF,
        columns=("ticker", "obs_date", "expiry", "strike", "right", "dte"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        order_by=("ticker", "obs_date", "expiry", "strike", "right"),
        max_batch_rows=10, max_result_rows=10)
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="option_chains"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_batches_never_exceed_max_batch_rows(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    rows = [_securities_row(f"T{i:02d}", 2024) for i in range(5)]
    record = publish_and_inspect(store, _SEC, _SEC_REF, rows, "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF, columns=("ticker", "year"),
        key_filter=(KeyPredicate(column="year", operator="eq", values=(2024,)),),
        order_by=("ticker", "year"), max_batch_rows=2, max_result_rows=100)
    sizes = [batch.num_rows for batch in repo.scan(query, table_name="securities")]
    assert sizes == [2, 2, 1]


def test_result_limit_exceeded_even_after_some_batches_were_yielded(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    rows = [_securities_row(f"T{i:02d}", 2024) for i in range(5)]
    record = publish_and_inspect(store, _SEC, _SEC_REF, rows, "2024")
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF, columns=("ticker", "year"),
        key_filter=(KeyPredicate(column="year", operator="eq", values=(2024,)),),
        order_by=("ticker", "year"), max_batch_rows=2, max_result_rows=3)
    it = repo.scan(query, table_name="securities")
    first = next(it)
    assert first.num_rows == 2
    with pytest.raises(DataError) as err:
        list(it)
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"


def test_fragment_pruning_shown_by_object_open_counter(tmp_path, monkeypatch):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    calls = []
    original = objects.verify_object_path

    def counting(store_, object_ref):
        calls.append(object_ref.object_id)
        return original(store_, object_ref)

    monkeypatch.setattr(objects, "verify_object_path", counting)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF, columns=("ticker", "year"),
        key_filter=(KeyPredicate(column="year", operator="eq", values=(2024,)),),
        order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
    rows = [row for batch in repo.scan(query, table_name="securities") for row in batch.to_pylist()]
    assert {r["year"] for r in rows} == {2024}
    assert len(calls) == 1  # only the 2024 fragment's object was opened


def test_unknown_projected_column_is_contract_mismatch(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(columns=("ticker", "bogus")))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "CONTRACT_MISMATCH"


def test_max_result_rows_above_contract_cap_is_query_not_bounded(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id,
                      **_basic_query(max_batch_rows=10, max_result_rows=3_000_000))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"


def test_deadline_already_passed_is_deadline_exceeded(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id,
                      **_basic_query(deadline="2000-01-01T00:00:00.000000Z"))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "DEADLINE_EXCEEDED"


def test_boundary_duplicate_key_across_fragments_raises_manifest_corrupt(tmp_path):
    """Two independently-published ``earnings_events`` fragments (different
    partition years) that happen to share one ``event_id`` at their boundary
    — a genuine cross-fragment duplicate, not a within-fragment one
    ``inspect_fragment`` would already refuse."""
    conn, clock, store = catalog_and_store(tmp_path)
    events_contract = contract_for("earnings_events")
    events_ref = contract_ref_for(events_contract)

    def _row(event_id, ticker, year):
        from datetime import datetime
        return dict(event_id=event_id, ticker=ticker, event_date=datetime(year, 1, 2), year=year,
                   session="BMO", session_src="orats", annc_tod=None, src_orats=True,
                   src_oquants=True, src_nasdaq=False, src_yfinance=False, date_agree=True,
                   date_conflict=False, updated_at=None, event_cluster_id=None, claim_count=None,
                   reconciliation=None)

    fragment_2024 = publish_and_inspect(
        store, events_contract, events_ref,
        [_row("AAA_2024-01-02", "AAA", 2024), _row("MID_SHARED", "MID", 2024)], "2024")
    fragment_2025 = publish_and_inspect(
        store, events_contract, events_ref,
        [_row("MID_SHARED", "MID", 2025), _row("ZZZ_2025-01-02", "ZZZ", 2025)], "2025")
    snap = commit_tables(conn, clock, {"earnings_events": [fragment_2024, fragment_2025]},
                         {"earnings_events": events_contract})
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=events_ref,
        columns=("event_id", "ticker"),
        key_filter=(KeyPredicate(column="ticker", operator="in", values=("AAA", "MID", "ZZZ")),),
        order_by=("event_id",), max_batch_rows=10, max_result_rows=10)
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="earnings_events"))
    assert err.value.code == "MANIFEST_CORRUPT"


# --------------------------------------------------------------------------
# timestamp-bound normalization (external review): a ``TimeInterval``/
# ``KeyPredicate`` bound given as a bare date, a naive timestamp, or a
# timezone-aware timestamp (``Z``, ``+00:00``, another offset) must compare
# identically -- before the fix, ``_normalize_bound`` only recognized a bare
# date or the exact naive wire form; any complete UTC timestamp got a bogus
# ``"T00:00:00.000000"`` appended, silently reversing inclusion at interval
# boundaries. Both fragment pruning (``_time_may_match``, reached from
# ``Repository.scan`` before any fragment is even opened) and row filtering
# (``_interval_matches``, ``_predicate_matches``) route through the one
# fixed ``_normalize_bound``, so all of this is exercised end to end through
# real temp storage.
# --------------------------------------------------------------------------


def _daily_market_row_at(ticker: str, when, year: int = 2026) -> dict:
    return dict(ticker=ticker, date=when, year=year, spot=100.0, iv10=30.0,
               iv30=32.0, exern_iv10=29.0, exern_iv30=31.0, implied_move=5.0,
               implied_reconstructed=False, rvol30=28.0, skew=1.1, contango=0.5, fwd90_30=33.0,
               fexern90_30=34.0, iee=0.2, mcap_usd=1e9, mcap_log=20.7,
               mcap_asof=when, mcap_age_days=0.0, src_spot="orats",
               src_iv="orats", src_mcap="orats")


def _boundary_snapshot(tmp_path):
    """One ``daily_market`` fragment with rows on 2026-09-10 and 2026-09-11
    -- the reviewer's own boundary."""
    from datetime import datetime
    conn, clock, store = catalog_and_store(tmp_path)
    record = publish_and_inspect(
        store, _DM, _DM_REF,
        [_daily_market_row_at("AAA", datetime(2026, 9, 10)),
         _daily_market_row_at("AAA", datetime(2026, 9, 11))], "2026")
    snap = commit_tables(conn, clock, {"daily_market": [record]}, {"daily_market": _DM})
    return conn, store, snap


def _scan_dates(repo, snap, *, start=None, end=None):
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(),
        time_interval=TimeInterval(column="date", start_inclusive=start, end_exclusive=end),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    rows = [row for batch in repo.scan(query, table_name="daily_market") for row in batch.to_pylist()]
    return [r["date"] for r in rows]


@pytest.mark.parametrize("start,end", [
    ("2026-09-10", "2026-09-11"),                                     # bare date -- unchanged
    ("2026-09-10T00:00:00.000000", "2026-09-11T00:00:00.000000"),     # naive wire form -- unchanged
    ("2026-09-10T00:00:00Z", "2026-09-11T00:00:00Z"),                 # aware, Z, no micros
    ("2026-09-10T00:00:00.000000Z", "2026-09-11T00:00:00.000000Z"),   # aware, Z, with micros
    ("2026-09-10T00:00:00+00:00", "2026-09-11T00:00:00+00:00"),       # aware, +00:00
])
def test_time_interval_boundary_matches_regardless_of_bound_form(tmp_path, start, end):
    """The reviewer's own repro: the half-open interval [09-10, 09-11) must
    return exactly the 09-10 row -- never the 09-11 row, never both, never
    neither -- whichever of the five equivalent spellings the bound uses."""
    from datetime import datetime
    conn, store, snap = _boundary_snapshot(tmp_path)
    repo = Repository(conn, store)
    assert _scan_dates(repo, snap, start=start, end=end) == [datetime(2026, 9, 10)]


def test_time_interval_non_utc_offset_converts_to_utc(tmp_path):
    """05:30 local at +05:30 is 00:00 UTC -- the same boundary as every form
    above, not shifted by the raw offset digits."""
    from datetime import datetime
    conn, store, snap = _boundary_snapshot(tmp_path)
    repo = Repository(conn, store)
    dates = _scan_dates(repo, snap, start="2026-09-10T05:30:00+05:30", end="2026-09-11T05:30:00+05:30")
    assert dates == [datetime(2026, 9, 10)]


def test_fragment_pruning_keeps_matching_fragment_for_utc_suffixed_interval(tmp_path):
    """Two fragments (2026 and 2027); only the 2026 fragment's time_min/
    time_max fall inside a Z-suffixed interval spanning September 2026.
    Before the fix, ``_time_may_match``'s malformed comparison could wrongly
    prune the fragment that legitimately matches, silently dropping its rows
    -- this exercises pruning, not just row filtering, since a wrongly-
    pruned fragment is never even opened."""
    from datetime import datetime
    conn, clock, store = catalog_and_store(tmp_path)
    sept_2026 = publish_and_inspect(
        store, _DM, _DM_REF,
        [_daily_market_row_at("AAA", datetime(2026, 9, 10)),
         _daily_market_row_at("AAA", datetime(2026, 9, 11))], "2026")
    jan_2027 = publish_and_inspect(
        store, _DM, _DM_REF, [_daily_market_row_at("AAA", datetime(2027, 1, 5), year=2027)], "2027")
    snap = commit_tables(conn, clock, {"daily_market": [sept_2026, jan_2027]}, {"daily_market": _DM})
    repo = Repository(conn, store)
    dates = _scan_dates(repo, snap, start="2026-09-01T00:00:00Z", end="2026-09-30T00:00:00Z")
    assert dates == [datetime(2026, 9, 10), datetime(2026, 9, 11)]


def test_key_predicate_timestamp_equality_matches_z_form(tmp_path):
    """A ``KeyPredicate`` equality on a timestamp column, given in ``Z``
    form, must match the same row a naive-form value matches -- ``Z`` bounds
    on ``KeyPredicate.values`` are not format-checked at document decode
    (that check only knows a column is a timestamp at scan time), so this
    exercises ``_predicate_matches``' own call into ``_normalize_bound``."""
    from datetime import datetime
    conn, store, snap = _boundary_snapshot(tmp_path)
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(KeyPredicate(column="date", operator="eq", values=("2026-09-10T00:00:00Z",)),),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    rows = [row for batch in repo.scan(query, table_name="daily_market") for row in batch.to_pylist()]
    assert [r["date"] for r in rows] == [datetime(2026, 9, 10)]


def test_normalize_bound_rejects_unparseable_value():
    with pytest.raises(DataError) as err:
        query_mod._normalize_bound("not-a-timestamp")
    assert err.value.code == "CONTRACT_MISMATCH"


def test_scan_with_unparseable_key_predicate_timestamp_refuses_contract_mismatch(tmp_path):
    """An unparseable ``KeyPredicate`` value on a timestamp column reaches
    ``_normalize_bound`` (it is not caught by document-decode, which cannot
    know a plain ``str`` field is a timestamp without the table contract)
    and refuses typed rather than comparing as a raw, mismatched string."""
    conn, store, snap = _boundary_snapshot(tmp_path)
    repo = Repository(conn, store)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(KeyPredicate(column="date", operator="eq", values=("not-a-timestamp",)),),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="daily_market"))
    assert err.value.code == "CONTRACT_MISMATCH"


# --------------------------------------------------------------------------
# query.py pure-function coverage
# --------------------------------------------------------------------------


def test_arrow_type_for_unsupported_physical_type_is_contract_mismatch():
    with pytest.raises(DataError) as err:
        query_mod.arrow_type_for("not_a_real_type")
    assert err.value.code == "CONTRACT_MISMATCH"


def test_null_array_for_returns_typed_nulls():
    array = query_mod.null_array_for("int64", 3)
    assert array.to_pylist() == [None, None, None]
    assert str(array.type) == "int64"


# --------------------------------------------------------------------------
# compile_row_matcher: a per-query-compiled row predicate whose normalization
# is O(values), not O(rows x values). Every case below asserts BOTH that the
# compiled matcher agrees with ``row_matches`` (they must never drift) AND
# the independently-known expected boolean -- including rows sitting exactly
# ON an interval bound, which pins the inclusive/exclusive edges.
#
# Perf note for the guard test at the bottom of this section: the OLD shape
# (``row_matches`` called once per row) rebuilt each predicate's ``wanted``
# set from scratch on every row, so N rows over a timestamp predicate of M
# values normalized M bounds N times (O(N*M) -- the chain-index scan blowup
# this fix removes). The compiled path normalizes the M values exactly once,
# no matter how many rows it then filters.
# --------------------------------------------------------------------------


_H = "sha256:" + "0" * 64


def _synthetic_contract() -> TableContract:
    """A tiny two-column contract (one string, one timestamp) built by hand
    -- ``compile_row_matcher`` is pure, so no data files are involved."""
    return TableContract(
        contract_id="tc_synth", definition_hash=_H, table_name="synthetic",
        semantic_version="1.0",
        columns=(ColumnContract(name="ticker", physical_type="string", nullable=False),
                 ColumnContract(name="obs_date", physical_type="timestamp[us]", nullable=False)),
        primary_key=("ticker", "obs_date"), duplicate_policy="reject", foreign_keys=(),
        partition_columns=(), filterable_columns=("ticker", "obs_date"),
        orderable_columns=("ticker", "obs_date"), observation_time_column="obs_date",
        finality_semantics="legacy_daily_close.v1", provenance_semantics="legacy_import.v1",
        coverage_semantics="legacy_full.v1", schema_evolution_policy="major_on_meaning_change.v1",
        maximum_batch_rows=1000, maximum_result_rows=1000)


def _synthetic_query(contract: TableContract, **overrides) -> DataQuery:
    base = dict(
        snapshot_id="snap_synth",
        table_contract_ref=TableContractRef(contract_id=contract.contract_id,
                                            definition_hash=contract.definition_hash),
        columns=("ticker", "obs_date"), key_filter=(),
        order_by=("ticker", "obs_date"), max_batch_rows=100, max_result_rows=100)
    base.update(overrides)
    return DataQuery(**base)


def _row(ticker, obs_date) -> dict:
    return {"ticker": ticker, "obs_date": obs_date}


def _representative_cases(contract: TableContract) -> tuple[dict, dict, dict]:
    """The shared ``compile_row_matcher``/``compile_batch_matcher`` fixture:
    every interesting row shape (``None`` column values, a missing key, exact
    boundary values, all three ``_normalize_bound`` forms) against every
    interesting query shape (string ``in``, timestamp ``in`` with one value
    per bound-normalization branch, and four interval bound combinations),
    plus the independently-known expected boolean for each pair."""
    from datetime import datetime
    ts_predicate = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="obs_date", operator="in",
        # one value per _normalize_bound branch: bare date, naive wire form,
        # tz-aware Z form.
        values=("2024-01-02", "2024-03-04T00:00:00.000000", "2024-05-06T00:00:00Z")),))
    queries = {
        "in_string": _synthetic_query(contract, key_filter=(KeyPredicate(
            column="ticker", operator="in", values=("AAA", "BBB")),)),
        "in_timestamp_three_bound_forms": ts_predicate,
        "interval_start_only": _synthetic_query(contract, time_interval=TimeInterval(
            column="obs_date", start_inclusive="2024-01-02")),
        "interval_end_only": _synthetic_query(contract, time_interval=TimeInterval(
            column="obs_date", end_exclusive="2024-01-02")),
        "interval_both": _synthetic_query(contract, time_interval=TimeInterval(
            column="obs_date", start_inclusive="2024-01-02", end_exclusive="2024-01-03")),
        "interval_neither": _synthetic_query(contract, time_interval=TimeInterval(
            column="obs_date")),
    }
    boundary = datetime(2024, 1, 2)  # exactly ON every interval bound above
    rows = {
        "AAA on boundary": _row("AAA", boundary),
        "CCC off list": _row("CCC", boundary),
        "AAA naive ts in values": _row("AAA", datetime(2024, 3, 4)),
        "AAA aware-form ts in values": _row("AAA", datetime(2024, 5, 6)),
        "AAA ts not in values": _row("AAA", datetime(2024, 2, 2)),
        "AAA before boundary": _row("AAA", datetime(2024, 1, 1)),
        "AAA after boundary": _row("AAA", datetime(2024, 1, 3)),
        "None ticker": _row(None, boundary),
        "missing ticker key": {"obs_date": boundary},
        "None obs_date": _row("AAA", None),
    }
    expected = {
        "in_string": {"AAA on boundary": True, "CCC off list": False,
                      "AAA naive ts in values": True, "AAA aware-form ts in values": True,
                      "AAA ts not in values": True, "AAA before boundary": True,
                      "AAA after boundary": True, "None ticker": False,
                      "missing ticker key": False, "None obs_date": True},
        "in_timestamp_three_bound_forms": {"AAA on boundary": True, "CCC off list": True,
                                           "AAA naive ts in values": True,
                                           "AAA aware-form ts in values": True,
                                           "AAA ts not in values": False,
                                           "AAA before boundary": False,
                                           "AAA after boundary": False, "None ticker": True,
                                           "missing ticker key": True, "None obs_date": False},
        # start_inclusive includes the boundary row itself; end_exclusive excludes it.
        "interval_start_only": {"AAA on boundary": True, "CCC off list": True,
                                "AAA naive ts in values": True, "AAA aware-form ts in values": True,
                                "AAA ts not in values": True, "AAA before boundary": False,
                                "AAA after boundary": True, "None ticker": True,
                                "missing ticker key": True, "None obs_date": False},
        "interval_end_only": {"AAA on boundary": False, "CCC off list": False,
                              "AAA naive ts in values": False, "AAA aware-form ts in values": False,
                              "AAA ts not in values": False, "AAA before boundary": True,
                              "AAA after boundary": False, "None ticker": False,
                              "missing ticker key": False, "None obs_date": False},
        "interval_both": {"AAA on boundary": True, "CCC off list": True,
                          "AAA naive ts in values": False, "AAA aware-form ts in values": False,
                          "AAA ts not in values": False, "AAA before boundary": False,
                          "AAA after boundary": False, "None ticker": True,
                          "missing ticker key": True, "None obs_date": False},
        "interval_neither": {"AAA on boundary": True, "CCC off list": True,
                             "AAA naive ts in values": True, "AAA aware-form ts in values": True,
                             "AAA ts not in values": True, "AAA before boundary": True,
                             "AAA after boundary": True, "None ticker": True,
                             "missing ticker key": True, "None obs_date": False},
    }
    return queries, rows, expected


def test_compile_row_matcher_agrees_with_row_matches_on_representative_cases():
    contract = _synthetic_contract()
    queries, rows, expected = _representative_cases(contract)
    for name, query in queries.items():
        matches = query_mod.compile_row_matcher(contract, query)
        for row_name, row in rows.items():
            compiled = matches(row)
            assert compiled == query_mod.row_matches(row, contract, query), (name, row_name)
            assert compiled is expected[name][row_name], (name, row_name)


def test_compile_row_matcher_none_columns_never_match():
    """Nulls keep today's "missing value never matches" semantics in both
    paths: a None predicate column AND a None interval column refuse."""
    from datetime import datetime
    contract = _synthetic_contract()
    query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="ticker", operator="in", values=("AAA",)),),
        time_interval=TimeInterval(column="obs_date", start_inclusive="2024-01-01"))
    matches = query_mod.compile_row_matcher(contract, query)
    null_predicate = _row(None, datetime(2024, 1, 2))
    null_interval = _row("AAA", None)
    assert query_mod.row_matches(null_predicate, contract, query) is False
    assert matches(null_predicate) is False
    assert query_mod.row_matches(null_interval, contract, query) is False
    assert matches(null_interval) is False


def test_compile_row_matcher_normalizes_predicate_values_once_per_query(monkeypatch):
    """The perf guard: ONE compile + N >= 50 matcher calls over a timestamp
    ``in`` predicate of M >= 20 values may touch ``_normalize_bound`` at most
    M times total (the old per-row shape was O(N*M) -- see the section
    comment above); row filtering itself must not re-normalize at all."""
    from datetime import datetime
    contract = _synthetic_contract()
    n_rows, m_values = 60, 25
    values = tuple(f"2024-01-{day:02d}" for day in range(1, m_values + 1))
    query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="obs_date", operator="in", values=values),))
    calls = []
    real = query_mod._normalize_bound

    def counting(value):
        calls.append(value)
        return real(value)

    monkeypatch.setattr(query_mod, "_normalize_bound", counting)
    matches = query_mod.compile_row_matcher(contract, query)
    rows = [_row("AAA", datetime(2024, 1, (i % m_values) + 1)) for i in range(n_rows)]
    assert all(matches(row) for row in rows)
    assert len(calls) <= m_values


# --------------------------------------------------------------------------
# compile_batch_matcher: ONLY key_filter equality/set-membership on a
# non-timestamp column, vectorized onto a decoded Arrow batch (task brief
# #286 follow-up narrowed the scope -- see query.py's docstring). Every
# case below asserts either: the batch mask agrees, index by index, with
# the row path for a predicate it DOES vectorize; or that it correctly
# REFUSES (returns None, falling back to compile_row_matcher) for a
# time_interval, a timestamp-typed predicate, or an unrepresentable value.
# --------------------------------------------------------------------------


def test_compile_batch_matcher_agrees_with_row_path_on_real_arrow_batch():
    """The full representative fixture, once as one real Arrow batch:
    ``compile_batch_matcher`` vectorizes ``in_string`` and
    ``in_timestamp_three_bound_forms`` (string/timestamp equality need no
    reinterpretation this package cannot prove identical) and each mask
    must equal ``expected`` (and ``compile_row_matcher``) index by index;
    every ``interval_*`` case here has a ``time_interval``, so task brief
    #286 follow-up's narrowed scope refuses all of them -- ``None``,
    falling back to ``compile_row_matcher`` entirely (already pinned
    correct by
    ``test_compile_row_matcher_agrees_with_row_matches_on_representative_cases``
    above)."""
    contract = _synthetic_contract()
    queries, rows, expected = _representative_cases(contract)
    row_order = list(rows)
    columns = {
        "ticker": pa.array([rows[name].get("ticker") for name in row_order], type=pa.string()),
        "obs_date": pa.array([rows[name].get("obs_date") for name in row_order],
                             type=pa.timestamp("us")),
    }
    vectorized_queries = {"in_string", "in_timestamp_three_bound_forms"}
    for query_name, query in queries.items():
        batch_matcher = query_mod.compile_batch_matcher(contract, query)
        if query_name in vectorized_queries:
            assert batch_matcher is not None, query_name
            mask = batch_matcher(columns, len(row_order)).to_pylist()
            row_matcher = query_mod.compile_row_matcher(contract, query)
            for index, row_name in enumerate(row_order):
                assert bool(mask[index]) is expected[query_name][row_name], (query_name, row_name)
                assert row_matcher(rows[row_name]) is expected[query_name][row_name], (
                    query_name, row_name)
        else:
            assert batch_matcher is None, query_name


def _timestamp_ns_contract() -> TableContract:
    """The synthetic contract again, but with ``obs_date`` declared
    ``timestamp[ns]`` — the resolution the legacy ``datetime64[ns]`` columns
    actually carry."""
    return TableContract(
        contract_id="tc_ns", definition_hash=_H, table_name="synthetic_ns",
        semantic_version="1.0",
        columns=(ColumnContract(name="ticker", physical_type="string", nullable=False),
                 ColumnContract(name="obs_date", physical_type="timestamp[ns]", nullable=False)),
        primary_key=("ticker", "obs_date"), duplicate_policy="reject", foreign_keys=(),
        partition_columns=(), filterable_columns=("ticker", "obs_date"),
        orderable_columns=("ticker", "obs_date"), observation_time_column="obs_date",
        finality_semantics="legacy_daily_close.v1", provenance_semantics="legacy_import.v1",
        coverage_semantics="legacy_full.v1", schema_evolution_policy="major_on_meaning_change.v1",
        maximum_batch_rows=1000, maximum_result_rows=1000)


def test_compile_batch_matcher_refuses_timestamp_interval_bounds_and_agrees_with_row_path():
    """#286 follow-up: a ``TimeInterval`` always falls back to
    ``compile_row_matcher`` now, regardless of how extreme its bound is --
    including the exact bounds a vectorized timestamp path once needed
    dedicated handling for (year 1, year 999, year 3000, and a pre-1970
    negative-epoch instant). Each must: (1) make ``compile_batch_matcher``
    refuse outright, and (2) still produce the correct, uncrashed result
    through ``compile_row_matcher`` alone. A 2024 row is after year 1,
    before year 3000, and inside 1000..3000; year 999 is the bound it is
    AFTER, which this platform's zero-padded ``%Y`` wire form compares
    chronologically, so that one row-path answer excludes the row."""
    import pandas as pd
    contract = _timestamp_ns_contract()
    instant = pd.Timestamp("2024-01-02T00:00:00.123456789").to_pydatetime()
    decoded_row = _row("AAA", instant)
    for bounds, matched in (
            (dict(start_inclusive="0001-01-01"), True),
            (dict(end_exclusive="0999-01-01"), False),
            (dict(end_exclusive="3000-01-01"), True),
            (dict(start_inclusive="1000-01-01", end_exclusive="3000-01-01"), True)):
        query = _synthetic_query(contract, time_interval=TimeInterval(column="obs_date", **bounds))
        assert query_mod.compile_batch_matcher(contract, query) is None, bounds
        row_matcher = query_mod.compile_row_matcher(contract, query)
        assert row_matcher(decoded_row) is matched, bounds

    negative_epoch_row = _row("AAA", pd.Timestamp("1969-12-31T23:59:59.999999999").to_pydatetime())
    interval_query = _synthetic_query(contract, time_interval=TimeInterval(
        column="obs_date", end_exclusive="1970-01-01T00:00:00.000000"))
    assert query_mod.compile_batch_matcher(contract, interval_query) is None
    assert query_mod.compile_row_matcher(contract, interval_query)(negative_epoch_row) is True


def test_compile_batch_matcher_truncates_nanoseconds_exactly_like_the_row_path():
    """A ns timestamp with a non-zero sub-microsecond remainder must MATCH
    the microsecond-truncated wire-form value on both paths: the row path
    because ``strftime("%f")`` drops the remainder, the batch path because
    ``_batch_comparable`` floors (never rounds) to microsecond resolution."""
    import pandas as pd
    contract = _timestamp_ns_contract()
    instant = pd.Timestamp("2024-01-02T00:00:00.123456789")
    array = pa.array([instant], type=pa.timestamp("ns"))
    query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="obs_date", operator="in", values=("2024-01-02T00:00:00.123456",)),))
    batch_matcher = query_mod.compile_batch_matcher(contract, query)
    assert batch_matcher is not None
    row_matcher = query_mod.compile_row_matcher(contract, query)
    decoded_row = _row("AAA", instant.to_pydatetime())
    assert row_matcher(decoded_row) is True
    mask = batch_matcher({"obs_date": array}, 1).to_pylist()
    assert bool(mask[0]) is True
    assert bool(mask[0]) == row_matcher(decoded_row)


def test_compile_batch_matcher_floors_negative_epoch_nanoseconds_like_the_row_path():
    """A pre-1970 (negative-epoch) ns timestamp with a non-zero sub-
    microsecond remainder must floor -- never truncate toward zero -- to
    the microsecond wire form on both paths: a toward-zero truncation of
    ``1969-12-31T23:59:59.999999999`` would wrongly round UP to
    ``1970-01-01T00:00:00.000000``, while ``_batch_comparable``'s exact
    ``int64`` arithmetic and the row path's ``strftime`` both give the
    one-microsecond-earlier ``1969-12-31T23:59:59.999999``."""
    import pandas as pd
    contract = _timestamp_ns_contract()
    instant = pd.Timestamp("1969-12-31T23:59:59.999999999")
    array = pa.array([instant], type=pa.timestamp("ns"))
    decoded_row = _row("AAA", instant.to_pydatetime())
    query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="obs_date", operator="in", values=("1969-12-31T23:59:59.999999",)),))
    batch_matcher = query_mod.compile_batch_matcher(contract, query)
    assert batch_matcher is not None
    row_matcher = query_mod.compile_row_matcher(contract, query)
    assert row_matcher(decoded_row) is True
    mask = batch_matcher({"obs_date": array}, 1).to_pylist()
    assert bool(mask[0]) is True
    assert bool(mask[0]) == row_matcher(decoded_row)


def test_compile_batch_matcher_near_minimum_ns_instant_floors_like_the_row_path():
    """A ns timestamp within a microsecond of the MINIMUM representable
    ``timestamp[ns]`` instant must floor to its own microsecond wire form
    on both paths. ``pc.floor_temporal`` silently wrapped exactly such a
    value around to the MAXIMUM representable instant (a real pyarrow bug:
    its internal arithmetic underflows near that boundary), which would
    make this ``in`` check compare against the wrong value;
    ``_floor_ns_to_us``'s exact ``int64`` arithmetic floors it correctly
    to ``1677-09-21T00:12:43.145224``."""
    import numpy as np
    import pandas as pd
    contract = _timestamp_ns_contract()
    instant = pd.Timestamp(np.iinfo(np.int64).min + 500)
    array = pa.array([instant], type=pa.timestamp("ns"))
    decoded_row = _row("AAA", instant.to_pydatetime())
    query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="obs_date", operator="in", values=("1677-09-21T00:12:43.145224",)),))
    batch_matcher = query_mod.compile_batch_matcher(contract, query)
    assert batch_matcher is not None
    row_matcher = query_mod.compile_row_matcher(contract, query)
    assert row_matcher(decoded_row) is True
    mask = batch_matcher({"obs_date": array}, 1).to_pylist()
    assert bool(mask[0]) is True
    assert bool(mask[0]) == row_matcher(decoded_row)


def _string_time_contract() -> TableContract:
    """A contract whose ``observation_time_column`` is declared ``string``
    -- e.g. the real ``price_history.retrieved_at`` -- the exact case that
    showed a vectorized interval comparing a STRING array against DATETIME
    bounds raises at evaluation time, outside any compile-time guard;
    task brief #286 follow-up's narrowed scope avoids this entirely by
    never vectorizing a ``TimeInterval`` at all, regardless of column
    type."""
    return TableContract(
        contract_id="tc_strtime", definition_hash=_H, table_name="synthetic_strtime",
        semantic_version="1.0",
        columns=(ColumnContract(name="ticker", physical_type="string", nullable=False),
                 ColumnContract(name="retrieved_at", physical_type="string", nullable=False)),
        primary_key=("ticker", "retrieved_at"), duplicate_policy="reject", foreign_keys=(),
        partition_columns=(), filterable_columns=("ticker", "retrieved_at"),
        orderable_columns=("ticker", "retrieved_at"), observation_time_column="retrieved_at",
        finality_semantics="legacy_daily_close.v1", provenance_semantics="legacy_import.v1",
        coverage_semantics="legacy_full.v1", schema_evolution_policy="major_on_meaning_change.v1",
        maximum_batch_rows=1000, maximum_result_rows=1000)


def test_compile_batch_matcher_refuses_interval_on_string_typed_time_column():
    """The exact gate-caught case: a ``TimeInterval`` on a column declared
    ``string`` (not a timestamp) must ALSO refuse at compile time -- it is
    a ``time_interval`` at all, so ``compile_batch_matcher`` returns
    ``None`` before ever looking at the column's physical type -- and the
    row path alone must still run without raising (whatever its own
    lexical-vs-chronological answer is for an extreme bound like this is
    unchanged, pre-existing row-path behavior, not something this test
    re-derives)."""
    contract = _string_time_contract()
    query = _synthetic_query(contract, columns=("ticker", "retrieved_at"), key_filter=(),
                             order_by=("ticker", "retrieved_at"),
                             time_interval=TimeInterval(column="retrieved_at",
                                                        end_exclusive="0999-01-01"))
    assert query_mod.compile_batch_matcher(contract, query) is None
    row_matcher = query_mod.compile_row_matcher(contract, query)
    row = {"ticker": "AAA", "retrieved_at": "2024-01-02T00:00:00.000000"}
    assert row_matcher(row) in (True, False)


def test_compile_batch_matcher_all_null_column_array_never_matches():
    """``null_array_for`` — what ``repository._filter_batch`` feeds the mask
    for a filter column the physical fragment lacks — must never match, for
    any predicate values and for both vectorized column kinds (string,
    timestamp), exactly like the row path's ``row.get(column) is None``
    -> never matches."""
    contract = _synthetic_contract()
    n = 4
    for values in (("AAA",), ("AAA", "ZZZ"), ("ZZZ", "ZZZZ")):
        query = _synthetic_query(contract, key_filter=(KeyPredicate(
            column="ticker", operator="in", values=values),))
        batch_matcher = query_mod.compile_batch_matcher(contract, query)
        assert batch_matcher is not None
        mask = batch_matcher({"ticker": query_mod.null_array_for("string", n)}, n).to_pylist()
        assert [bool(entry) for entry in mask] == [False] * n
    ts_query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="obs_date", operator="in", values=("2024-01-02",)),))
    ts_batch_matcher = query_mod.compile_batch_matcher(contract, ts_query)
    assert ts_batch_matcher is not None
    ts_mask = ts_batch_matcher({"obs_date": query_mod.null_array_for("timestamp[us]", n)},
                               n).to_pylist()
    assert [bool(entry) for entry in ts_mask] == [False] * n


def _mixed_types_contract() -> TableContract:
    """The synthetic contract plus one ``int64``, one ``bool``, and one
    ``float64`` column, so a predicate's ``values`` can be made
    unrepresentable in more than one declared Arrow type, and so a
    well-formed predicate on an excluded type (``bool``/``float64``) can be
    tested directly."""
    return TableContract(
        contract_id="tc_types", definition_hash=_H, table_name="synthetic_types",
        semantic_version="1.0",
        columns=(ColumnContract(name="ticker", physical_type="string", nullable=False),
                 ColumnContract(name="obs_date", physical_type="timestamp[us]", nullable=False),
                 ColumnContract(name="shares", physical_type="int64", nullable=False),
                 ColumnContract(name="flag", physical_type="bool", nullable=False),
                 ColumnContract(name="price", physical_type="float64", nullable=False)),
        primary_key=("ticker", "obs_date"), duplicate_policy="reject", foreign_keys=(),
        partition_columns=(), filterable_columns=("ticker", "obs_date", "shares", "flag", "price"),
        orderable_columns=("ticker", "obs_date"), observation_time_column="obs_date",
        finality_semantics="legacy_daily_close.v1", provenance_semantics="legacy_import.v1",
        coverage_semantics="legacy_full.v1", schema_evolution_policy="major_on_meaning_change.v1",
        maximum_batch_rows=1000, maximum_result_rows=1000)


def test_compile_batch_matcher_refuses_values_unrepresentable_in_column_type():
    """The documented ``None``: a ``key_filter`` whose ``values`` cannot be
    built into its column's declared Arrow type is not expressible
    vectorized, and the caller must fall back to the per-row path — which
    still compiles fine (it never performs the typed Arrow construction)
    and simply matches nothing."""
    contract = _mixed_types_contract()
    mismatched = (
        ("ticker", (1, 2)),
        ("shares", ("x", "y")),
        ("flag", ("yes",)),
    )
    for column, values in mismatched:
        query = _synthetic_query(contract, key_filter=(KeyPredicate(
            column=column, operator="in", values=values),))
        assert query_mod.compile_batch_matcher(contract, query) is None, (column, values)
        row_matcher = query_mod.compile_row_matcher(contract, query)
        assert row_matcher(_row("AAA", None)) is False


def test_compile_batch_matcher_refuses_int64_overflow_without_raising():
    """#286 follow-up: a Python ``int`` outside the C ``long`` range against
    an ``int64`` column makes ``pa.array(..., type=pa.int64())`` raise a
    plain ``OverflowError`` -- not a ``pyarrow.lib.ArrowException``
    subclass -- so the fallback guard must catch that too and return
    ``None`` (degrading to the per-row path, whose plain ``in`` set check
    never raises on this value) instead of crashing the scan."""
    contract = _mixed_types_contract()
    query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="shares", operator="in", values=(99999999999999999999999,)),))
    assert query_mod.compile_batch_matcher(contract, query) is None
    row_matcher = query_mod.compile_row_matcher(contract, query)
    assert row_matcher(_row("AAA", None)) is False


def test_compile_batch_matcher_refuses_float_and_bool_key_filter_and_agrees_with_row_path():
    """#286 follow-up (gate-caught regression): a ``float64`` column is
    excluded from vectorization even for a well-formed predicate --
    ``pyarrow.compute.is_in`` compares a float's raw bit pattern, so a
    ``-0.0`` row value never matches a ``0`` value_set entry, while the
    row path's plain Python ``==``/set membership correctly treats them
    equal (confirmed directly: a real divergence). ``bool`` is excluded
    too -- task brief #286 follow-up narrowed the vectorized set to
    ``string``/``int64``/timestamp only."""
    contract = _mixed_types_contract()
    float_query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="price", operator="in", values=(0,)),))
    assert query_mod.compile_batch_matcher(contract, float_query) is None
    row_matcher = query_mod.compile_row_matcher(contract, float_query)
    assert row_matcher({"ticker": "AAA", "price": -0.0}) is True

    bool_query = _synthetic_query(contract, key_filter=(KeyPredicate(
        column="flag", operator="in", values=(True,)),))
    assert query_mod.compile_batch_matcher(contract, bool_query) is None
    bool_row_matcher = query_mod.compile_row_matcher(contract, bool_query)
    assert bool_row_matcher({"ticker": "AAA", "flag": True}) is True


# --------------------------------------------------------------------------
# explain_dependencies — §5.5
# --------------------------------------------------------------------------


def test_explain_dependencies_names_surviving_fragments(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(
        key_filter=(KeyPredicate(column="year", operator="eq", values=(2024,)),)))
    plan = repo.explain_dependencies(query, table_name="securities")
    assert isinstance(plan, DependencyPlan)
    assert plan.snapshot_ref == snap
    assert len(plan.dependencies) == 1  # only the 2024 fragment survives pruning
    entry = plan.dependencies[0]
    assert entry.table_name == "securities"
    assert entry.maximum_rows == query.max_result_rows
    assert entry.estimated_rows > 0


def test_explain_dependencies_without_table_name_is_unsupported_contract(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query())
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(query)
    assert err.value.code == "UNSUPPORTED_CONTRACT"


def test_explain_dependencies_for_a_chain_query_is_unsupported_contract(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    chain_query = ChainQuery(security_id="sec", observation_ceiling="2024-01-01T00:00:00.000000Z",
                             session_date="2024-01-01", quote_policy_ref="legacy_stored_quote.v1",
                             max_contracts=10)
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(chain_query, table_name="option_chains")
    assert err.value.code == "UNSUPPORTED_CONTRACT"


def test_explain_dependencies_for_an_unrecognized_object_is_unsupported_contract(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(object(), table_name="securities")
    assert err.value.code == "UNSUPPORTED_CONTRACT"


# --------------------------------------------------------------------------
# Slice A: manifest membership bounds (``query.plan_scan_population`` and
# ``Repository.scan_population_bound``) and the footer-vs-manifest row-count
# integrity check in ``Repository._fragment_rows``.
# --------------------------------------------------------------------------


def _aaa_filter() -> tuple:
    return (KeyPredicate(column="ticker", operator="eq", values=("AAA",)),)


def test_scan_population_bound_of_empty_table_membership_is_zero(tmp_path):
    conn, clock, _store = catalog_and_store(tmp_path)
    snap = commit_tables(conn, clock, {"securities": []}, {"securities": _SEC})
    repo = Repository(conn)  # no ArtifactStore: metadata planning opens no bytes
    assert repo.scan_population_bound(
        snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
        key_filter=_aaa_filter()) == 0


def test_scan_population_bound_of_fully_pruned_selection_is_zero(tmp_path):
    conn, store, snap = _securities_snapshot(tmp_path)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("ZZZ",)),)))
    assert Repository(conn).scan_population_bound(
        snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
        key_filter=query.key_filter) == 0
    repo = Repository(conn, store)
    assert list(repo.scan(query, table_name="securities")) == []
    assert repo.explain_dependencies(query, table_name="securities").dependencies == ()

    unbounded = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query(
        key_filter=(), time_interval=None))
    with pytest.raises(DataError) as err:
        list(repo.scan(unbounded, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(unbounded, table_name="securities")
    assert err.value.code == "QUERY_NOT_BOUNDED"


def test_zero_row_fragment_metadata_has_zero_bound_and_scans_empty(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    record = hand_built_record(
        store, _SEC, _SEC_REF, table_from_rows(_SEC, []), partition_key="2024", row_count=0,
        primary_key_min=("AAA", 2024), primary_key_max=("ZZZ", 2024))
    snap = commit_tables(conn, clock, {"securities": [record]}, {"securities": _SEC})
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query())
    assert Repository(conn).scan_population_bound(
        snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
        key_filter=query.key_filter) == 0
    assert list(Repository(conn, store).scan(query, table_name="securities")) == []


def _pruning_case(case, tmp_path):
    from datetime import datetime

    conn, clock, store = catalog_and_store(tmp_path)
    if case == "partition":
        records = [
            publish_and_inspect(store, _SEC, _SEC_REF,
                                [_securities_row(t, 2024) for t in ("AAA", "BBB")], "2024"),
            publish_and_inspect(store, _SEC, _SEC_REF,
                                [_securities_row(t, 2025) for t in ("AAA", "BBB")], "2025"),
        ]
        snap = commit_tables(conn, clock, {"securities": records}, {"securities": _SEC})
        query = DataQuery(
            snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF,
            columns=("ticker", "year"),
            key_filter=(KeyPredicate(column="year", operator="eq", values=(2024,)),
                        KeyPredicate(column="ticker", operator="eq", values=("BBB",))),
            order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
        expected = (2, [{"ticker": "BBB", "year": 2024}], [records[0].fragment_id])
        return conn, store, snap, "securities", _SEC_REF, query, expected
    if case == "leading_key":
        records = [
            publish_and_inspect(store, _SEC, _SEC_REF,
                                [_securities_row(t, 2024) for t in ("AAA", "BBB")], "2024"),
            publish_and_inspect(store, _SEC, _SEC_REF,
                                [_securities_row(t, 2025) for t in ("CCC", "DDD")], "2025"),
        ]
        snap = commit_tables(conn, clock, {"securities": records}, {"securities": _SEC})
        query = DataQuery(
            snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF,
            columns=("ticker", "year"),
            key_filter=(KeyPredicate(column="ticker", operator="eq", values=("CCC",)),),
            order_by=("ticker", "year"), max_batch_rows=10, max_result_rows=10)
        expected = (2, [{"ticker": "CCC", "year": 2025}], [records[1].fragment_id])
        return conn, store, snap, "securities", _SEC_REF, query, expected
    records = [
        publish_and_inspect(store, _DM, _DM_REF, [_daily_market_row("AAA", 2024, 2)], "2024"),
        publish_and_inspect(store, _DM, _DM_REF,
                            [_daily_market_row("AAA", 2025, 2), _daily_market_row("AAA", 2025, 3)],
                            "2025"),
    ]
    snap = commit_tables(conn, clock, {"daily_market": records}, {"daily_market": _DM})
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(KeyPredicate(column="ticker", operator="eq", values=("AAA",)),),
        time_interval=TimeInterval(column="date", start_inclusive="2025-01-03",
                                   end_exclusive="2026-01-01"),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    expected = (2, [{"ticker": "AAA", "date": datetime(2025, 1, 3)}], [records[1].fragment_id])
    return conn, store, snap, "daily_market", _DM_REF, query, expected


@pytest.mark.parametrize("case", ["partition", "leading_key", "time"])
def test_pruning_bound_explain_and_opened_fragments_agree(tmp_path, monkeypatch, case):
    """The metadata bound equals the explain estimate sum equals the opened
    scan fragments, while a predicate inside the surviving fragment makes the
    candidate bound strictly larger than the matched rows."""
    conn, store, snap, table_name, ref, query, (bound, rows, ids) = _pruning_case(case, tmp_path)
    repo = Repository(conn, store)
    assert repo.scan_population_bound(
        snap.snapshot_id, table_name=table_name, table_contract_ref=ref,
        key_filter=query.key_filter, time_interval=query.time_interval) == bound
    plan = repo.explain_dependencies(query, table_name=table_name)
    assert [e.fragment_ref.fragment_id for e in plan.dependencies] == ids
    assert sum(e.estimated_rows for e in plan.dependencies) == bound
    opened = []
    real = Repository._fragment_rows

    def counting(self, record, *args, **kwargs):
        opened.append(record.fragment_id)
        return real(self, record, *args, **kwargs)

    monkeypatch.setattr(Repository, "_fragment_rows", counting)
    scanned = [row for batch in repo.scan(query, table_name=table_name)
               for row in batch.to_pylist()]
    assert opened == ids
    assert scanned == rows


@pytest.mark.parametrize("bad_count", [-1, True, "2"])
def test_plan_scan_population_refuses_invalid_surviving_row_count(tmp_path, bad_count):
    import dataclasses

    conn, _store, snap = _securities_snapshot(tmp_path)
    record = Repository(conn).fragment_records(snap, "securities")[0]
    with pytest.raises(DataError) as err:
        query_mod.plan_scan_population(_SEC, [dataclasses.replace(record, row_count=bad_count)])
    assert err.value.code == "MANIFEST_CORRUPT"
    assert err.value.problem.retryable is False


@pytest.mark.parametrize("bad_count", [-1, True, "2"])
def test_scan_and_explain_refuse_planted_invalid_row_count_before_streams(
        tmp_path, monkeypatch, bad_count):
    import dataclasses

    conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(conn, store)
    _contract, records = repo._table_records(snap, "securities", _SEC_REF)
    bad_record = dataclasses.replace(records[0], row_count=bad_count)
    original = repo._table_records

    def planted(snap_, table_name, table_contract_ref):
        contract, _records = original(snap_, table_name, table_contract_ref)
        return contract, [bad_record]

    monkeypatch.setattr(repo, "_table_records", planted)
    monkeypatch.setattr(Repository, "_fragment_rows",
                        lambda *args, **kwargs: pytest.fail("fragment stream opened"))
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query())
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "MANIFEST_CORRUPT" and err.value.problem.retryable is False
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(query, table_name="securities")
    assert err.value.code == "MANIFEST_CORRUPT" and err.value.problem.retryable is False


@pytest.mark.parametrize("physical_rows,recorded_count", [(2, 1), (1, 2)])
def test_footer_row_count_mismatch_refuses_before_iter_batches(
        tmp_path, monkeypatch, physical_rows, recorded_count):
    """Recorded below (a predicate hides one real row) and recorded above both
    refuse on the raw footer count, before ``_iter_batches`` -- so the below
    case cannot pass merely because filtering leaves fewer matches."""
    conn, clock, store = catalog_and_store(tmp_path)
    rows = [_daily_market_row("AAA", 2024, day) for day in range(2, 2 + physical_rows)]
    record = hand_built_record(
        store, _DM, _DM_REF, table_from_rows(_DM, rows), partition_key="2024",
        row_count=recorded_count,
        primary_key_min=("AAA", "2024-01-02T00:00:00.000000"),
        primary_key_max=("AAA", f"2024-01-{1 + physical_rows:02d}T00:00:00.000000"))
    snap = commit_tables(conn, clock, {"daily_market": [record]}, {"daily_market": _DM})
    repo = Repository(conn, store)

    def _no_batches(self, *args, **kwargs):
        raise AssertionError("_iter_batches reached despite footer row-count mismatch")

    monkeypatch.setattr(Repository, "_iter_batches", _no_batches)
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_DM_REF, columns=("ticker", "date"),
        key_filter=(KeyPredicate(column="date", operator="eq",
                                 values=("2024-01-02T00:00:00.000000",)),),
        order_by=("ticker", "date"), max_batch_rows=10, max_result_rows=10)
    with pytest.raises(DataError) as err:
        next(repo.scan(query, table_name="daily_market"))
    assert err.value.code == "MANIFEST_CORRUPT"
    assert err.value.problem.retryable is False


def test_result_limit_above_candidate_bound_succeeds_smaller_limit_fails(tmp_path):
    from engine.v2.foundation import content_hash, to_document

    conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id, **_basic_query())
    bound = repo.scan_population_bound(
        snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
        key_filter=query.key_filter)
    assert bound == 4 and query.max_result_rows == 10 > bound
    plan = repo.explain_dependencies(query, table_name="securities")
    assert plan.request_hash == content_hash(to_document(query))
    assert {e.maximum_rows for e in plan.dependencies} == {query.max_result_rows}
    rows = [row for batch in repo.scan(query, table_name="securities") for row in batch.to_pylist()]
    assert rows == [{"ticker": "AAA", "year": 2024}, {"ticker": "AAA", "year": 2025}]

    smaller = DataQuery(snapshot_id=snap.snapshot_id,
                        **_basic_query(max_batch_rows=1, max_result_rows=1))
    with pytest.raises(DataError) as err:
        list(repo.scan(smaller, table_name="securities"))
    assert err.value.code == "RESULT_LIMIT_EXCEEDED"
    above_cap = DataQuery(snapshot_id=snap.snapshot_id,
                          **_basic_query(max_result_rows=3_000_000))
    with pytest.raises(DataError) as err:
        list(repo.scan(above_cap, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(above_cap, table_name="securities")
    assert err.value.code == "QUERY_NOT_BOUNDED"


@pytest.mark.parametrize("case", ["empty_membership", "fully_pruned"])
def test_zero_population_bound_does_not_admit_zero_result_limit(tmp_path, case):
    if case == "empty_membership":
        conn, clock, store = catalog_and_store(tmp_path)
        snap = commit_tables(conn, clock, {"securities": []}, {"securities": _SEC})
        key_filter = _aaa_filter()
    else:
        conn, store, snap = _securities_snapshot(tmp_path)
        key_filter = (KeyPredicate(column="ticker", operator="eq", values=("ZZZ",)),)
    repo = Repository(conn, store)
    assert repo.scan_population_bound(
        snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
        key_filter=key_filter) == 0
    query = DataQuery(
        snapshot_id=snap.snapshot_id, table_contract_ref=_SEC_REF,
        columns=("ticker", "year"), key_filter=key_filter,
        order_by=("ticker", "year"), max_batch_rows=1, max_result_rows=0)
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED" and err.value.problem.retryable is False
    with pytest.raises(DataError) as err:
        repo.explain_dependencies(query, table_name="securities")
    assert err.value.code == "QUERY_NOT_BOUNDED" and err.value.problem.retryable is False


@pytest.mark.parametrize("key_filter", [
    None,
    1,
    True,
    "",
    "ticker",
    {},
    {"ticker": "AAA"},
    b"",
    set(),
])
def test_non_sequence_key_filter_container_refuses_query_not_bounded(tmp_path, key_filter):
    """PR 365 gate finding: metadata planning refuses any ``key_filter``
    that is not a list/tuple — None/integers used to raise a raw
    ``TypeError``, while empty strings/mappings/bytes/sets iterated nothing
    and silently selected ALL fragments. Both planning APIs refuse with a
    nonretryable ``QUERY_NOT_BOUNDED`` before touching a record; no raw
    ``TypeError`` escapes and no full-membership bound is ever returned."""
    conn, _store, snap = _securities_snapshot(tmp_path)
    records = Repository(conn).fragment_records(snap, "securities")
    with pytest.raises(DataError) as err:
        query_mod.plan_scan_population(_SEC, records, key_filter=key_filter)
    assert err.value.code == "QUERY_NOT_BOUNDED"
    assert err.value.problem.retryable is False
    with pytest.raises(DataError) as err:  # no store: metadata planning only
        Repository(conn).scan_population_bound(
            snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
            key_filter=key_filter)
    assert err.value.code == "QUERY_NOT_BOUNDED"
    assert err.value.problem.retryable is False


@pytest.mark.parametrize("key_filter", [(), []])
def test_empty_list_or_tuple_key_filter_still_yields_full_bound(tmp_path, key_filter):
    """Positive control for the refusal above: the two legitimate empty
    containers still plan whole-membership metadata (the full fixture bound
    of 4 = two 2-row fragments) through both APIs."""
    conn, _store, snap = _securities_snapshot(tmp_path)
    records = Repository(conn).fragment_records(snap, "securities")
    assert query_mod.plan_scan_population(_SEC, records, key_filter=key_filter).row_count == 4
    assert Repository(conn).scan_population_bound(
        snap.snapshot_id, table_name="securities", table_contract_ref=_SEC_REF,
        key_filter=key_filter) == 4
