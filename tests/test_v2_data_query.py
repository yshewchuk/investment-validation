"""D05, D06: bounded Arrow scans — phase-2 guide §5.3, §8.2, §12.

D05 drives the query validator's refusals; D06 drives real synthetic scans
(real Parquet, real catalog, real ``ArtifactStore``) for exact-match rows in
deterministic key order, typed-null synthesis for a declared-missing nullable
column, batch/result-row bounding, and fragment pruning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts.data import (  # noqa: E402
    ChainQuery,
    DataQuery,
    DependencyPlan,
    KeyPredicate,
    TimeInterval,
)
from engine.v2.data import objects, query as query_mod  # noqa: E402
from engine.v2.data.errors import DataError  # noqa: E402
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


def test_deadline_already_passed_is_query_not_bounded(tmp_path):
    _conn, store, snap = _securities_snapshot(tmp_path)
    repo = Repository(_conn, store)
    query = DataQuery(snapshot_id=snap.snapshot_id,
                      **_basic_query(deadline="2000-01-01T00:00:00.000000Z"))
    with pytest.raises(DataError) as err:
        list(repo.scan(query, table_name="securities"))
    assert err.value.code == "QUERY_NOT_BOUNDED"


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
