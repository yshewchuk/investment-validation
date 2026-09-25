"""S4C: the native yfinance history/earnings edges.

The injectable seam is ``history_fn``/``earnings_fn`` (a canned DataFrame), the
library-call equivalent of ``http_get``; the default callables import
``yfinance`` lazily, so importing this module and constructing either fetcher
touches no network and no heavy dependency.
"""
from __future__ import annotations

import io

import pandas as pd

from engine.v2.ops.providers import PROVIDER_CREDENTIAL_VARIABLES, provider_credentials
from engine.v2.ops.providers.yfinance_edge import (
    BMO_CUTOFF,
    EARNINGS_LIMIT,
    _session_from_annc_tod,
    yfinance_earnings_fetcher,
    yfinance_history_fetcher,
)


def _history_frame() -> pd.DataFrame:
    days = pd.bdate_range("2026-04-27", "2026-05-05")
    return pd.DataFrame({"Open": 100.0, "Close": [100.0 + i for i in range(len(days))]},
                        index=days)


def test_history_fetcher_returns_the_csv_the_store_parses():
    fetcher = yfinance_history_fetcher(history_fn=lambda ticker: _history_frame())
    raw = fetcher("AAA")
    frame = pd.read_csv(io.BytesIO(raw))
    assert "Close" in frame.columns
    assert len(frame) == 7
    # the first CSV column is the index yfinance writes, exactly as before
    assert frame.columns[0] not in ("Open", "Close")


def test_history_fetcher_is_none_for_an_empty_frame():
    fetcher = yfinance_history_fetcher(history_fn=lambda ticker: pd.DataFrame())
    assert fetcher("AAA") is None


def test_earnings_fetcher_returns_the_parsed_columns_the_store_reads():
    frame = pd.DataFrame([{"ticker": "AAA", "event_date": "2026-05-01",
                           "annc_tod": "1650", "session": "AMC"}],
                         columns=["ticker", "event_date", "annc_tod", "session"])
    fetcher = yfinance_earnings_fetcher(earnings_fn=lambda ticker: frame)
    parsed = pd.read_csv(io.BytesIO(fetcher("AAA")))
    assert list(parsed.columns) == ["ticker", "event_date", "annc_tod", "session"]
    assert parsed.iloc[0]["session"] == "AMC"


def test_earnings_fetcher_is_none_for_an_empty_frame():
    fetcher = yfinance_earnings_fetcher(earnings_fn=lambda ticker: None)
    assert fetcher("AAA") is None


def test_the_ported_session_mapping_matches_the_legacy_cutoff():
    assert _session_from_annc_tod("0800") == "BMO"
    assert _session_from_annc_tod("1650") == "AMC"
    assert _session_from_annc_tod("1159") == "BMO"
    assert _session_from_annc_tod("1200") == "AMC"
    assert _session_from_annc_tod(None) is None
    assert _session_from_annc_tod(float("nan")) is None
    assert _session_from_annc_tod("time-not-supplied") is None
    assert BMO_CUTOFF == 1200
    assert EARNINGS_LIMIT == 12


def test_the_unmetered_account_needs_no_credentials(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "unused")
    assert PROVIDER_CREDENTIAL_VARIABLES["yfinance"] == ()
    assert provider_credentials({"provider_account": "yfinance"}) == {}
    assert callable(yfinance_history_fetcher())
    assert callable(yfinance_earnings_fetcher())
