"""Session finality is positive evidence, not a permissive staleness check."""
from __future__ import annotations

import pandas as pd

from engine.data import finality
from engine import ledger


class _Entry:
    def __init__(self, endpoint, day):
        self.endpoint = endpoint
        self.params = {"tradeDate": day}
        self.meta = {"status": 200}


def _install(monkeypatch, daily, chains, entries):
    def read(table, **kwargs):
        return daily.copy() if table == "daily_market" else chains.copy()
    monkeypatch.setattr(finality.store, "read_table", read)
    monkeypatch.setattr(finality.fetch, "iter_cached", lambda source: list(entries))


def test_finality_requires_exact_date_and_all_three_proofs(monkeypatch):
    day = "2026-09-09"
    daily = pd.DataFrame({"ticker": ["A", "B"], "date": [day, day]})
    chains = pd.DataFrame({"ticker": ["A", "B"], "obs_date": [day, day]})
    _install(monkeypatch, daily, chains, [
        _Entry("hist/summaries", day), _Entry("hist/cores", day),
    ])

    result = finality.session_finality(day, ["A", "B"])

    assert result.is_final
    assert result.market_wide
    assert result.daily_share == 1.0
    assert result.chain_share == 1.0


def test_finality_refuses_a_missing_close_even_when_yesterday_exists(monkeypatch):
    day = "2026-09-10"
    daily = pd.DataFrame({"ticker": ["A", "B"], "date": ["2026-09-09", day]})
    chains = pd.DataFrame({"ticker": ["A", "B"], "obs_date": [day, day]})
    _install(monkeypatch, daily, chains, [
        _Entry("hist/summaries", day), _Entry("hist/cores", day),
    ])

    result = finality.session_finality(day, ["A", "B"])

    assert not result.is_final
    assert result.daily_share == 0.5
    assert "daily" in result.detail


def test_ledger_records_the_decision_session_not_the_later_entry(monkeypatch):
    monkeypatch.setattr(
        ledger, "_event_ids",
        lambda frame: frame.assign(event_id=["id-1"] * len(frame), session=["AMC"] * len(frame)),
    )
    scores = pd.DataFrame([{
        "ticker": "A", "strategy": "STR-THRU", "event_date": "2026-09-11",
        "as_of": "2026-09-09", "entry_date": "2026-09-10",
        "exit_date": "2026-09-14", "strike": None, "expiry": "2026-09-18",
    }])
    receipt = {"is_final": True, "date": "2026-09-09"}

    rows = ledger.build_prediction_rows(scores, as_of="2026-09-09", finality=receipt)

    assert len(rows) == 1
    assert rows[0]["as_of"] == "2026-09-09"
    assert rows[0]["structure"]["decision_date"] == "2026-09-09"
    assert rows[0]["structure"]["entry_date"] == "2026-09-10"
    assert rows[0]["finality"] == receipt


def test_a_ticker_the_store_never_carries_cannot_veto_a_session(monkeypatch):
    """The regression this gate shipped with.

    The nightly asks about its whole calendar, which includes illiquid names
    no data source carries. Counting those against a same-session check asks
    whether data arrived that was never going to arrive — measured 2026-09-11,
    81 of 213 requested names had no row on any date, so the gate read 62%
    against an 80% floor and refused every session it was shown.
    """
    day = "2026-09-10"
    daily = pd.DataFrame({"ticker": ["A", "B"], "date": [day, day]})
    chains = pd.DataFrame({"ticker": ["A", "B"], "obs_date": [day, day]})
    _install(monkeypatch, daily, chains, [
        _Entry("hist/summaries", day), _Entry("hist/cores", day),
    ])

    # Five names the store has never seen, alongside the two it has.
    result = finality.session_finality(
        day, ["A", "B", "AENT", "ALAR", "BTTC", "CMMB", "EONR"])

    assert result.is_final, result.detail
    assert result.daily_share == 1.0
    assert result.chain_share == 1.0
    assert result.tickers == 7        # asked about
    assert result.covered == 2        # actually carried — the real denominator


def test_a_store_carrying_none_of_the_universe_is_not_vacuously_final(monkeypatch):
    """Excluding uncovered names must not become 'divide by nothing, pass'.

    A store that lost the universe has an empty denominator, and 0/0 is
    exactly the shape that would report perfect freshness on no data at all.
    """
    day = "2026-09-10"
    daily = pd.DataFrame({"ticker": ["X", "Y"], "date": [day, day]})
    chains = pd.DataFrame({"ticker": ["X", "Y"], "obs_date": [day, day]})
    _install(monkeypatch, daily, chains, [
        _Entry("hist/summaries", day), _Entry("hist/cores", day),
    ])

    result = finality.session_finality(day, ["A", "B"])

    assert not result.is_final
    assert result.covered == 0
    assert "carried" in result.detail


def test_a_covered_name_missing_todays_close_still_fails(monkeypatch):
    """The exclusion is about names with NO rows, never about a missing close.

    A carried name that simply did not get today's pull must still count
    against the session — that is the whole point of the gate.
    """
    day = "2026-09-10"
    daily = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "date": [day, day, day, day, "2026-09-09"],
    })
    chains = pd.DataFrame({
        "ticker": ["A", "B", "C", "D", "E"],
        "obs_date": [day, day, day, day, day],
    })
    _install(monkeypatch, daily, chains, [
        _Entry("hist/summaries", day), _Entry("hist/cores", day),
    ])

    result = finality.session_finality(day, ["A", "B", "C", "D", "E", "GHOST"])

    assert result.covered == 5           # GHOST excluded, E is not
    assert result.daily_share == 0.8     # E's close is genuinely missing
    assert result.chain_share == 1.0
    assert result.is_final               # 80% is the floor, inclusive
