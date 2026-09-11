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
