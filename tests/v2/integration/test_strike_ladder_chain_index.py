"""engine.dashboard.nightly.strike_ladder: one chain index for the whole ladder.

# packages: foundation
"""
from __future__ import annotations

import pandas as pd

AS_OF = pd.Timestamp("2026-01-05")
PLAN_KEY = ("AAA", pd.Timestamp("2026-01-02"))


def _board_row(ticker, strategy, *, gate_pass=True, spot=100.0, session="AMC",
               event_date=AS_OF, quote_max_age_sessions=5):
    return {
        "ticker": ticker,
        "strategy": strategy,
        "event_date": event_date,
        "session": session,
        "gate_pass": gate_pass,
        "spot": spot,
        "quote_max_age_sessions": quote_max_age_sessions,
        "fill": 0.5,
        "variant": None,
        "structure_spec": None,
        "structure_params": None,
        "strike_offset": None,
    }


def _live_strategies():
    """STRUCTURES minus FORECAST_SIZED — the ones the ladder re-strikes."""
    from engine.forecast_sizing import FORECAST_SIZED
    from engine.structures import STRUCTURES

    return sorted(set(STRUCTURES) - set(FORECAST_SIZED))


class _FakeResult:
    def __init__(self, request):
        self._request = request

    def as_dict(self):
        return {"ticker": self._request.ticker, "strike": self._request.strike}


class _FakeScorer:
    """Records every (request, chain_index) the ladder asks for."""

    def __init__(self, calendar=None, snapshot="snap-test"):
        self.calendar = calendar
        self.snapshot = snapshot
        self.calls = []

    def score(self, request, chain_index=None):
        self.calls.append((request, chain_index))
        return _FakeResult(request)


def test_one_chain_index_serves_the_whole_ladder(monkeypatch):
    """load_chain_index runs once and every score sees that same index."""
    import engine.replay as replay_module
    import engine.score as score_module
    from engine.dashboard import nightly
    from engine.score import ladder_strike

    strategies = _live_strategies()[:2]
    s1, s2 = strategies

    board = pd.DataFrame([
        _board_row("AAA", s1, spot=100.0),
        _board_row("BBB", s1, spot=200.0),
        _board_row("CCC", s2, spot=50.0),
        _board_row("DDD", s1, gate_pass=False, spot=100.0),
        _board_row("EEE", s2, spot=float("nan")),
    ])
    finite_live = [board.iloc[0], board.iloc[1], board.iloc[2]]

    monkeypatch.setattr(
        score_module, "board_chain_keys",
        lambda *a, **k: {("AAA", pd.Timestamp("2026-01-05"))},
    )

    sentinel = replay_module.ChainIndex({})
    loaded = []

    def _fake_load_chain_index(keys, *, progress_every=1, years=None):
        loaded.append(set(keys))
        return sentinel

    monkeypatch.setattr(replay_module, "load_chain_index", _fake_load_chain_index)

    scorer = _FakeScorer(calendar=object())
    rows = nightly.strike_ladder(board, scorer=scorer, alt_strikes=1, as_of=AS_OF)

    assert len(loaded) == 1
    assert all(chain_index is sentinel for _, chain_index in scorer.calls)

    offsets = [-0.025, 0.025]
    assert len(scorer.calls) == len(finite_live) * len(offsets)

    expected = []
    for rec in finite_live:
        for offset in offsets:
            expected.append({
                "ticker": rec["ticker"],
                "strike": ladder_strike(rec["spot"], offset),
                "strike_offset": offset,
            })
    assert rows == expected


def test_empty_live_set_returns_empty_without_loading(monkeypatch):
    """No gate passers means no chain index is ever built."""
    import engine.replay as replay_module
    from engine.dashboard import nightly

    def _boom(*args, **kwargs):
        raise AssertionError("load_chain_index must not be called")

    monkeypatch.setattr(replay_module, "load_chain_index", _boom)

    board = pd.DataFrame([
        _board_row("AAA", _live_strategies()[0], gate_pass=False),
        _board_row("BBB", _live_strategies()[1], gate_pass=False),
    ])

    assert nightly.strike_ladder(
        board, scorer=_FakeScorer(calendar=object()), alt_strikes=1, as_of=AS_OF
    ) == []


def _plan_keys_for(events, strategy, *, quote_max_age_sessions):
    """board_chain_keys with plan_events and latest_chain_date both faked."""
    import engine.replay as replay_module
    import engine.score as score_module
    from engine.score import board_chain_keys

    called = []

    class _FakePlan:
        chain_keys = {PLAN_KEY}

    monkeypatch_target = score_module
    original = monkeypatch_target.plan_events
    monkeypatch_target.plan_events = lambda *a, **k: _FakePlan()

    original_latest = replay_module.latest_chain_date
    replay_module.latest_chain_date = lambda ticker, on_or_before: (
        called.append((ticker, on_or_before)) or PLAN_KEY[1]
    )
    try:
        keys = board_chain_keys(
            events, [strategy], {},
            calendar=object(), as_of=AS_OF,
            quote_max_age_sessions=quote_max_age_sessions,
        )
    finally:
        monkeypatch_target.plan_events = original
        replay_module.latest_chain_date = original_latest
    return keys, called


def test_board_chain_keys_refactor_equivalence():
    """With a quote bound the newest held chain joins the plan keys; without it
    the plan keys stand alone and latest_chain_date is never consulted."""
    from engine.forecast_sizing import FORECAST_SIZED
    from engine.score import DISABLED_STRATEGIES
    from engine.structures import STRUCTURES

    strategy = sorted(
        set(STRUCTURES) - set(FORECAST_SIZED) - set(DISABLED_STRATEGIES)
    )[0]
    events = pd.DataFrame([{
        "ticker": "AAA", "event_date": AS_OF, "session": "AMC",
    }])

    keys, called = _plan_keys_for(events, strategy, quote_max_age_sessions=5)
    assert keys == {PLAN_KEY}
    assert called

    keys, called = _plan_keys_for(events, strategy, quote_max_age_sessions=None)
    assert keys == {PLAN_KEY}
    assert called == []
