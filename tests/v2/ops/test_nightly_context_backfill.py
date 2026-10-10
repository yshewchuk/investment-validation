"""Context-ticker history backfill in ``engine.dashboard.nightly``.

SPY has no earnings event, so the calendar-driven ticker list never includes it;
``history_backfill_tickers`` adds it, and the refresh's history pass must fetch
it (or skip it when cached) before the market-wide pull.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.dashboard import nightly
from engine.v2.features import panel_row_inputs
from engine.v2.features.panel_row_inputs import CONTEXT_TICKERS


class _StopAfterHistory(Exception):
    pass


class _Record:
    from_cache = False

    def json(self):
        return {"data": [{"x": 1}]}


class _Fetcher:
    def __init__(self, cached=()):
        self.cached = set(cached)
        self.calls = []

    def has(self, source, endpoint, params):
        return params.get("ticker") in self.cached

    def fetch(self, source, endpoint, params, **kw):
        self.calls.append((endpoint, dict(params)))
        if "ticker" not in params:
            raise _StopAfterHistory()
        return _Record()


@pytest.fixture(autouse=True)
def isolated_unknown(monkeypatch):
    monkeypatch.setattr(nightly, "_unknown_symbols", lambda: set())
    monkeypatch.setattr(nightly, "_remember_unknown_symbol", lambda ticker: None)


def test_history_backfill_tickers_adds_context():
    assert nightly.history_backfill_tickers(["AAPL"]) == ["AAPL", "SPY"]
    assert nightly.history_backfill_tickers(["SPY", "AAPL"]) == ["AAPL", "SPY"]
    assert set(CONTEXT_TICKERS) <= set(nightly.history_backfill_tickers([]))


def test_legacy_context_tickers_mirror_feature_owner():
    assert nightly.CONTEXT_TICKERS == panel_row_inputs.CONTEXT_TICKERS


def test_refresh_backfills_spy_without_earnings_event():
    fetcher = _Fetcher()
    with pytest.raises(_StopAfterHistory):
        nightly.refresh_calendar_data(["AAPL"], pd.Timestamp("2026-10-09"),
                                      fetcher=fetcher, forward=False)
    per_ticker = {params["ticker"] for _, params in fetcher.calls if "ticker" in params}
    assert per_ticker == {"AAPL", "SPY"}
    spy_endpoints = {ep for ep, params in fetcher.calls if params.get("ticker") == "SPY"}
    assert spy_endpoints == set(nightly.HISTORY_ENDPOINTS)


def test_refresh_cached_spy_costs_no_call():
    fetcher = _Fetcher(cached={"SPY"})
    with pytest.raises(_StopAfterHistory):
        nightly.refresh_calendar_data(["AAPL"], pd.Timestamp("2026-10-09"),
                                      fetcher=fetcher, forward=False)
    assert not [1 for _, params in fetcher.calls if params.get("ticker") == "SPY"]
    assert any(params.get("ticker") == "AAPL" for _, params in fetcher.calls)
