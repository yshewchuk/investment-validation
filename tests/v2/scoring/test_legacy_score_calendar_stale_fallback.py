"""score_calendar must preload the stale-quote fallback chain the scorer will ask for.

The scorer's fallback anchors on the entry/decision date; the board's chain
index used to be keyed on `as_of` only, so an early-entry row (STR-RUNUP) whose
entry-day chain was missing priced NO_CHAIN on the board but STALE_QUOTE when
selfcheck re-scored it on demand.
"""
from __future__ import annotations

import pandas as pd
import pytest

import engine.replay as replay
import engine.score as score_mod


class _Stop(Exception):
    pass


def _capture_keys(monkeypatch, chain_dates, plan_keys, as_of):
    events = pd.DataFrame([{"event_id": 1, "ticker": "SSNC",
                            "event_date": pd.Timestamp("2026-10-22"),
                            "session": "AMC"}])
    monkeypatch.setattr(score_mod.store, "read_table", lambda *a, **k: events)

    class Plan:
        chain_keys = set(plan_keys)

    monkeypatch.setattr(score_mod, "plan_events", lambda *a, **k: Plan())

    def latest(ticker, on_or_before):
        cutoff = pd.Timestamp(on_or_before).normalize()
        older = [d for d in chain_dates if d <= cutoff]
        return max(older) if older else None

    monkeypatch.setattr(replay, "latest_chain_date", latest)
    seen = {}

    def fake_load(keys, **kwargs):
        seen["keys"] = set(keys)
        raise _Stop

    monkeypatch.setattr(score_mod, "load_chain_index", fake_load)

    class Engine:
        calendar = object()

    with pytest.raises(_Stop):
        score_mod.score_calendar(as_of, strategies=["STR-RUNUP"],
                                 scorer=Engine(), tickers=["SSNC"])
    return seen["keys"]


def test_entry_date_fallback_chain_is_preloaded(monkeypatch):
    chains = [pd.Timestamp(d) for d in
              ("2026-10-01", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08")]
    keys = _capture_keys(monkeypatch, chains,
                         [("SSNC", pd.Timestamp("2026-10-02"))], "2026-10-08")
    assert ("SSNC", pd.Timestamp("2026-10-08")) in keys
    assert ("SSNC", pd.Timestamp("2026-10-01")) in keys


def test_a_chain_after_the_night_is_never_preloaded_as_a_fallback(monkeypatch):
    chains = [pd.Timestamp(d) for d in ("2026-10-01", "2026-10-09")]
    keys = _capture_keys(monkeypatch, chains,
                         [("SSNC", pd.Timestamp("2026-10-12"))], "2026-10-08")
    assert ("SSNC", pd.Timestamp("2026-10-09")) not in keys
    assert ("SSNC", pd.Timestamp("2026-10-01")) in keys


def test_the_board_row_shows_the_fallback_quote():
    from engine.dashboard.render import _BOARD_FIELDS

    for name in ("quote_date", "quote_age_sessions", "flags"):
        assert name in _BOARD_FIELDS