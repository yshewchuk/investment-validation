from __future__ import annotations

import datetime as dt

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.carried_set import (
    build_uncarried_exclusions,
    resolve_carried_set,
)
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

AS_OF = "2026-06-30"
LOWER = dt.date(2025, 1, 1)
UPPER = dt.date(2026, 6, 30)


def _daily(ticker, date, year, computed_at=None):
    if isinstance(date, dt.date) and not isinstance(date, dt.datetime):
        date = dt.datetime(date.year, date.month, date.day)
    row = {
        "ticker": ticker,
        "date": date,
        "year": year,
        "spot": 100.0,
        "iv10": 30.0,
        "iv30": 32.0,
        "exern_iv10": 29.0,
        "exern_iv30": 31.0,
        "implied_move": 5.0,
        "implied_reconstructed": False,
        "rvol30": 28.0,
        "skew": 1.1,
        "contango": 0.5,
        "fwd90_30": 33.0,
        "fexern90_30": 34.0,
        "iee": 0.2,
        "mcap_usd": 1e9,
        "mcap_log": 20.7,
        "mcap_asof": dt.datetime(year, 1, 2),
        "mcap_age_days": 0.0,
        "src_spot": "orats",
        "src_iv": "orats",
        "src_mcap": "orats",
    }
    if computed_at is not None:
        row["computed_at"] = computed_at
    return row


def _chain(ticker, obs_date, year, computed_at=None):
    if isinstance(obs_date, dt.date) and not isinstance(obs_date, dt.datetime):
        obs_date = dt.datetime(obs_date.year, obs_date.month, obs_date.day)
    row = {
        "ticker": ticker,
        "obs_date": obs_date,
        "year": year,
        "expiry": obs_date + dt.timedelta(days=30),
        "dte": 30,
        "strike": 100.0,
        "right": "C",
        "bid": 1.0,
        "ask": 1.2,
        "mid": 1.1,
        "iv": 30.0,
        "delta": 0.5,
        "spot": 100.0,
        "src": "orats",
        "src_file": "f.parquet",
        "chain_kind": "entry",
        "volume": None,
        "open_interest": None,
        "bid_size": None,
        "ask_size": None,
        "quote_repaired": False,
    }
    if computed_at is not None:
        row["computed_at"] = computed_at
    return row


def _snapshot(tmp_path, rows_by_table):
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {name: contract_for(name) for name in rows_by_table}
    fragments = {}
    for table, rows in rows_by_table.items():
        contract = contracts[table]
        ref = contract_ref_for(contract)
        by_year = {}
        for row in rows:
            by_year.setdefault(str(row["year"]), []).append(row)
        fragments[table] = [
            publish_and_inspect(store, contract, ref, year_rows, year)
            for year, year_rows in sorted(by_year.items())
        ]
    snapshot = commit_tables(conn, clock, fragments, contracts, store=store)
    return Repository(conn, store), snapshot


def test_individual_missing_tables_and_one_exclusion_per_ticker(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        {
            "daily_market": [
                _daily("AAA", dt.date(2026, 3, 2), 2026),
                _daily("BBB", dt.date(2026, 3, 2), 2026),
            ],
            "option_chains": [
                _chain("AAA", dt.date(2026, 3, 2), 2026),
                _chain("CCC", dt.date(2026, 3, 2), 2026),
            ],
        },
    )
    resolution = resolve_carried_set(repository, snapshot, as_of=AS_OF)
    assert resolution.daily_market_tickers == ("AAA", "BBB")
    assert resolution.option_chain_tickers == ("AAA", "CCC")
    assert resolution.carried_tickers == ("AAA",)

    exclusions = build_uncarried_exclusions(
        ["CCC", "BBB", "AAA", "BBB", "CCC", "DDD"], resolution
    )
    assert len(exclusions) == 3
    assert tuple(exclusion.ticker for exclusion in exclusions) == (
        "BBB",
        "CCC",
        "DDD",
    )
    by_ticker = {exclusion.ticker: exclusion for exclusion in exclusions}
    assert by_ticker["BBB"].missing_tables == ("option_chains",)
    assert by_ticker["CCC"].missing_tables == ("daily_market",)
    assert by_ticker["DDD"].missing_tables == ("daily_market", "option_chains")
    assert by_ticker["BBB"].reason_code == "UNCARRIED_TICKER"
    assert by_ticker["CCC"].reason_code == "UNCARRIED_TICKER"


def test_inclusive_data_date_bounds_for_each_required_column(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        {
            "daily_market": [
                _daily("LOW", LOWER, 2025),
                _daily("HIGH", UPPER, 2026),
            ],
            "option_chains": [
                _chain("LOW", UPPER, 2026),
                _chain("HIGH", LOWER, 2025),
            ],
        },
    )
    resolution = resolve_carried_set(repository, snapshot, as_of=AS_OF)
    assert resolution.carried_tickers == ("HIGH", "LOW")


def test_future_data_dates_ignored_and_computed_at_irrelevant(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        {
            "daily_market": [
                _daily(
                    "FUT",
                    dt.date(2026, 7, 15),
                    2026,
                    computed_at=dt.datetime(2026, 6, 30),
                ),
                _daily(
                    "OK",
                    dt.date(2026, 2, 2),
                    2026,
                    computed_at=dt.datetime(2024, 1, 1),
                ),
            ],
            "option_chains": [
                _chain(
                    "FUT",
                    dt.date(2026, 7, 15),
                    2026,
                    computed_at=dt.datetime(2026, 6, 30),
                ),
                _chain(
                    "OK",
                    dt.date(2026, 2, 2),
                    2026,
                    computed_at=dt.datetime(2027, 1, 1),
                ),
            ],
        },
    )
    resolution = resolve_carried_set(repository, snapshot, as_of=AS_OF)
    assert resolution.daily_market_tickers == ("OK",)
    assert resolution.option_chain_tickers == ("OK",)
    assert resolution.carried_tickers == ("OK",)


def test_new_listing_and_recently_delisted_name_are_carried(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        {
            "daily_market": [
                _daily("NEW", dt.date(2026, 5, 1), 2026),
                _daily("OLD", dt.date(2025, 2, 1), 2025),
            ],
            "option_chains": [
                _chain("NEW", dt.date(2026, 5, 1), 2026),
                _chain("OLD", dt.date(2025, 2, 1), 2025),
            ],
        },
    )
    resolution = resolve_carried_set(repository, snapshot, as_of=AS_OF)
    assert resolution.carried_tickers == ("NEW", "OLD")


def test_missing_option_chains_raises_contract_mismatch(tmp_path):
    repository, snapshot = _snapshot(
        tmp_path,
        {"daily_market": [_daily("AAA", dt.date(2026, 3, 2), 2026)]},
    )
    with pytest.raises(DataError) as excinfo:
        resolve_carried_set(repository, snapshot, as_of=AS_OF)
    assert excinfo.value.code == "CONTRACT_MISMATCH"
