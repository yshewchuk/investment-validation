"""Public fixtures for option-chain snapshot tests (no private cross-test imports)."""
from __future__ import annotations

from datetime import datetime

from engine.v2.data.price_history_table import PRICE_HISTORY_CONTRACT
from engine.v2.data.repository import Repository
from engine.v2.ops.native_board_universe import BoardRequest
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    fake_hash,
    publish_and_inspect,
)
import pandas as pd

CHAINS = contract_for("option_chains")
CHAINS_REF = contract_ref_for(CHAINS)
PH_REF = contract_ref_for(PRICE_HISTORY_CONTRACT)
CHAINS_NAME = "option_chains"
SESSION = "2024-01-05"
STRADDLE_KEY = BoardRequest("AAA", "STR-THRU", pd.Timestamp("2024-01-16"), "BMO")


def price_row(ticker: str, date: str) -> dict:
    return dict(
        ticker=ticker, date=date, close_adj=9.0, close_raw=10.0, high_raw=10.0,
        retrieved_at="2024-01-01T00:00:00Z", deleted=False, source_kind="yfinance",
        source_hash=fake_hash(f"px-{ticker}-{date}"), capture_id=f"cap-{ticker}-{date}",
    )


def chain_row(ticker: str, obs_date: datetime, expiry: datetime, strike: float, *,
              right="C", bid=1.0, ask=1.2) -> dict:
    return dict(
        ticker=ticker, obs_date=obs_date, year=obs_date.year, expiry=expiry,
        dte=(expiry - obs_date).days, strike=strike, right=right, bid=bid, ask=ask,
        mid=1.1, iv=30.0, delta=0.5, spot=100.0, src="orats", src_file="f.parquet",
        chain_kind="entry", volume=None, open_interest=None, bid_size=None,
        ask_size=None, quote_repaired=False,
    )


def domain_chain_rows() -> list[dict]:
    obs = datetime(2024, 1, 5)
    return [
        chain_row("AAA", obs, datetime(2024, 1, 19), 100.0, right="C", bid=0.0, ask=0.0),
        chain_row("AAA", obs, datetime(2024, 1, 19), 100.0, right="P", bid=0.0, ask=0.0),
        chain_row("AAA", obs, datetime(2024, 1, 19), 105.0, right="P", bid=None, ask=None),
        chain_row("AAA", obs, datetime(2024, 2, 16), 100.0, right="C"),
        chain_row("AAA", obs, datetime(2024, 3, 15), 90.0, right="P", bid=None, ask=None),
        chain_row("AAA", datetime(2024, 1, 4), datetime(2024, 1, 19), 100.0, right="C"),
        chain_row("AAA", datetime(2024, 1, 4), datetime(2024, 1, 19), 100.0, right="P"),
        chain_row("BBB", obs, datetime(2024, 1, 19), 100.0, right="C"),
        chain_row("BBB", obs, datetime(2024, 1, 19), 100.0, right="P"),
    ]


def _sorted_rows(contract, rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: tuple(row[field] for field in contract.primary_key))


def snapshot(tmp_path, *, price=None, chains=None):
    """A committed snapshot with optional ``price_history`` / ``option_chains``."""
    conn, clock, store = catalog_and_store(tmp_path)
    tables: dict[str, list] = {}
    contracts: dict[str, object] = {}
    if price is not None:
        tables["price_history"] = [
            publish_and_inspect(store, PRICE_HISTORY_CONTRACT, PH_REF,
                                _sorted_rows(PRICE_HISTORY_CONTRACT, rows), ticker)
            for ticker, rows in sorted(price.items())
        ]
        contracts["price_history"] = PRICE_HISTORY_CONTRACT
    if chains is not None:
        tables[CHAINS_NAME] = [
            publish_and_inspect(store, CHAINS, CHAINS_REF, _sorted_rows(CHAINS, chains), "2024")
        ]
        contracts[CHAINS_NAME] = CHAINS
    committed = commit_tables(conn, clock, tables, contracts, scope="test")
    return Repository(conn, store), committed