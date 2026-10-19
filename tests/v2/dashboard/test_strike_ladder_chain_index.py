"""strike_ladder scores against ONE pre-loaded chain index.

Without it every ladder request reloaded whole option_chains year partitions
(legacy nightly ladder phase peaked at the 9.5G cap).
"""
from __future__ import annotations

import pandas as pd

import engine.score as score_mod
from engine.dashboard import nightly


class _Result:
    def __init__(self, ticker):
        self.ticker = ticker

    def as_dict(self):
        return {"ticker": self.ticker}


class _Scorer:
    calendar = object()
    snapshot = None

    def __init__(self):
        self.indexes = []

    def score(self, request, chain_index=None):
        self.indexes.append(chain_index)
        return _Result(request.ticker)


def _board(offsets):
    rows = []
    for i, off in enumerate(offsets):
        rows.append({
            "ticker": f"T{i}", "strategy": "STR-THRU",
            "event_date": pd.Timestamp("2026-01-12"), "session": "AMC",
            "spot": 100.0, "gate_pass": True, "fill": 0.5,
            "structure_spec": {"decision_offset": off},
            "structure_params": None, "quote_max_age_sessions": 5,
        })
    return pd.DataFrame(rows)


def _patch(monkeypatch):
    calls = {"keys": [], "load": 0}
    sentinel = score_mod.ChainIndex({})

    def fake_keys(events, strategies, decision_offsets, **kw):
        calls["keys"].append((tuple(strategies), dict(decision_offsets)))
        return {("T0", pd.Timestamp("2026-01-05"))}

    def fake_load(keys, **kw):
        calls["load"] += 1
        return sentinel

    monkeypatch.setattr(score_mod, "board_chain_keys", fake_keys)
    monkeypatch.setattr(score_mod, "load_chain_index", fake_load)
    return calls, sentinel


def test_ladder_loads_one_index_and_shares_it(monkeypatch):
    calls, sentinel = _patch(monkeypatch)
    scorer = _Scorer()
    rows = nightly.strike_ladder(
        _board([None, None, None]), scorer=scorer, alt_strikes=2,
        as_of=pd.Timestamp("2026-01-05"),
    )
    assert calls["load"] == 1
    assert len(rows) == 3 * 4
    assert len(scorer.indexes) == 12
    assert all(ix is sentinel for ix in scorer.indexes)


def test_ladder_groups_keys_by_strategy_and_offset(monkeypatch):
    calls, _ = _patch(monkeypatch)
    nightly.strike_ladder(
        _board([None, 1, 1]), scorer=_Scorer(), alt_strikes=1,
        as_of=pd.Timestamp("2026-01-05"),
    )
    assert sorted(calls["keys"], key=lambda c: str(c)) == sorted(
        [(("STR-THRU",), {"STR-THRU": None}), (("STR-THRU",), {"STR-THRU": 1})],
        key=lambda c: str(c),
    )
    assert calls["load"] == 1


def test_board_chain_keys_adds_newest_chain(monkeypatch):
    import engine.replay as replay_mod

    monkeypatch.setattr(
        replay_mod, "latest_chain_date", lambda t, d: pd.Timestamp("2026-01-02")
    )
    events = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "event_date": [pd.Timestamp("2026-01-12")] * 2,
        "session": ["AMC", "BMO"],
    })
    monkeypatch.setattr(
        score_mod, "plan_events",
        lambda structure, events, calendar=None: type("P", (), {"chain_keys": {("AAA", pd.Timestamp("2026-01-09"))}})(),
    )
    keys = score_mod.board_chain_keys(
        events, ["STR-THRU"], {}, calendar=None,
        as_of=pd.Timestamp("2026-01-05"), quote_max_age_sessions=5,
    )
    assert ("AAA", pd.Timestamp("2026-01-02")) in keys
    assert ("BBB", pd.Timestamp("2026-01-02")) in keys
    assert ("AAA", pd.Timestamp("2026-01-09")) in keys