"""Small real snapshot fixture for replay CLI integration tests."""
from __future__ import annotations

import pandas as pd

from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)


def chain_rows() -> list[dict]:
    rows: list[dict] = []
    for obs_date in ("2024-05-02", "2024-05-03"):
        call = (2.0, 2.4) if obs_date == "2024-05-02" else (3.0, 3.4)
        put = (1.0, 1.4) if obs_date == "2024-05-02" else (2.0, 2.4)
        obs = pd.Timestamp(obs_date)
        for expiry in (pd.Timestamp("2024-05-03"), pd.Timestamp("2024-05-24")):
            dte = (expiry - obs).days
            for strike in (95.0, 100.0, 105.0):
                scale = 1.0 if dte < 10 else 2.0
                for right, (bid, ask) in (("C", call), ("P", put)):
                    rows.append({
                        "ticker": "TEST", "obs_date": obs, "expiry": expiry, "dte": dte,
                        "strike": strike, "right": right, "bid": bid * scale,
                        "ask": ask * scale, "spot": 100.0, "quote_repaired": False,
                        "year": 2024,
                    })
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"], row["strike"], row["right"]))
    return rows


def event_rows() -> list[dict]:
    return [{
        "event_id": "TEST_2024-05-02", "ticker": "TEST",
        "event_date": pd.Timestamp("2024-05-02"), "year": 2024,
        "session": "AMC", "session_src": "test", "annc_tod": None,
        "src_orats": True, "src_oquants": True, "src_nasdaq": False,
        "src_yfinance": False, "date_agree": True, "date_conflict": False,
        "updated_at": None, "event_cluster_id": None, "claim_count": 1,
        "reconciliation": "test",
    }]


def daily_market_rows() -> list[dict]:
    return [
        {"ticker": "MKT", "date": date, "year": int(date.year)}
        for date in pd.bdate_range("2024-03-01", periods=60)
    ]


def replay_snapshot(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    names_and_rows = {
        "option_chains": chain_rows(),
        "earnings_events": event_rows(),
        "daily_market": daily_market_rows(),
    }
    contracts = {name: contract_for(name) for name in names_and_rows}
    tables = {
        name: [publish_and_inspect(
            store, contracts[name], contract_ref_for(contracts[name]), rows, "2024"
        )]
        for name, rows in names_and_rows.items()
    }
    snapshot = commit_tables(
        conn, clock, tables, contracts, receipt_id="alpha-binding", store=store
    )
    conn.close()
    return snapshot
