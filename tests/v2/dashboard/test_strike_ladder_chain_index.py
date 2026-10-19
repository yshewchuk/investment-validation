"""strike_ladder scores against ONE pre-loaded chain index.

Without it every ladder request reloaded whole option_chains year partitions
(legacy nightly ladder phase peaked at the 9.5G cap).
"""
from __future__ import annotations

import pandas as pd

import engine.score as score_mod
from engine.dashboard import nightly

AS_OF = pd.Timestamp("2026-01-05")
EVENT = pd.Timestamp("2026-01-12")


class _Result:
    def __init__(self, ticker):
        self.ticker = ticker

    def as_dict(self):
        return {"ticker": self.ticker}


class _Scorer:
    calendar = "CAL"
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
            "event_date": EVENT, "session": "AMC",
            "spot": 100.0, "gate_pass": True, "fill": 0.5,
            "structure_spec": {"decision_offset": off},
            "structure_params": None, "quote_max_age_sessions": 5,
        })
    return pd.DataFrame(rows)


def _patch(monkeypatch):
    calls = {"keys": [], "loaded": []}
    sentinel = score_mod.ChainIndex({})

    def fake_keys(events, strategies, decision_offsets, **kw):
        calls["keys"].append({
            "events": [tuple(r) for r in events.itertuples(index=False)],
            "strategies": tuple(strategies),
            "offsets": dict(decision_offsets),
            "calendar": kw["calendar"],
            "as_of": kw["as_of"],
            "age": kw["quote_max_age_sessions"],
        })
        t = events["ticker"].iloc[0]
        return {(t, pd.Timestamp("2026-01-05"))}

    def fake_load(keys, **kw):
        calls["loaded"].append(set(keys))
        return sentinel

    monkeypatch.setattr(score_mod, "board_chain_keys", fake_keys)
    monkeypatch.setattr(score_mod, "load_chain_index", fake_load)
    return calls, sentinel


def test_ladder_loads_one_index_and_shares_it(monkeypatch):
    calls, sentinel = _patch(monkeypatch)
    scorer = _Scorer()
    rows = nightly.strike_ladder(
        _board([None, None, None]), scorer=scorer, alt_strikes=2, as_of=AS_OF,
    )
    assert len(calls["loaded"]) == 1
    assert calls["loaded"][0] == {("T0", AS_OF)}
    assert len(rows) == 3 * 4
    assert len(scorer.indexes) == 12
    assert all(ix is sentinel for ix in scorer.indexes)
    assert len(calls["keys"]) == 1  # one (strategy, offset) group
    call = calls["keys"][0]
    assert call["as_of"] == AS_OF and call["age"] == 5 and call["calendar"] == "CAL"
    assert sorted(call["events"]) == [
        ("T0", EVENT, "AMC"), ("T1", EVENT, "AMC"), ("T2", EVENT, "AMC"),
    ]


def test_ladder_groups_keys_by_strategy_and_offset(monkeypatch):
    calls, _ = _patch(monkeypatch)
    nightly.strike_ladder(
        _board([None, 1, 1]), scorer=_Scorer(), alt_strikes=1, as_of=AS_OF,
    )
    offsets = sorted(
        (c["offsets"]["STR-THRU"] for c in calls["keys"]),
        key=lambda v: (v is not None, v),
    )
    assert offsets == [None, 1]
    assert all(
        v is None or type(v) is int for c in calls["keys"] for v in c["offsets"].values()
    )
    by_offset = {c["offsets"]["STR-THRU"]: c["events"] for c in calls["keys"]}
    assert by_offset[None] == [("T0", EVENT, "AMC")]
    assert sorted(by_offset[1]) == [("T1", EVENT, "AMC"), ("T2", EVENT, "AMC")]
    assert len(calls["loaded"]) == 1


def test_board_chain_keys_adds_newest_and_older_fallback_chains(monkeypatch):
    import engine.replay as replay_mod

    def latest(ticker, day):
        day = pd.Timestamp(day)
        if day == AS_OF:
            return pd.Timestamp("2026-01-02")
        if day == pd.Timestamp("2026-01-09"):
            return pd.Timestamp("2025-12-31")
        return None

    monkeypatch.setattr(replay_mod, "latest_chain_date", latest)
    monkeypatch.setattr(
        score_mod, "plan_events",
        lambda structure, events, calendar=None: type(
            "P", (), {"chain_keys": {("AAA", pd.Timestamp("2026-01-09"))}}
        )(),
    )
    events = pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "event_date": [EVENT] * 2,
        "session": ["AMC", "BMO"],
    })
    keys = score_mod.board_chain_keys(
        events, ["STR-THRU"], {}, calendar=None, as_of=AS_OF,
        quote_max_age_sessions=5,
    )
    assert keys == {
        ("AAA", pd.Timestamp("2026-01-09")),
        ("AAA", pd.Timestamp("2026-01-02")),
        ("BBB", pd.Timestamp("2026-01-02")),
        ("AAA", pd.Timestamp("2025-12-31")),
    }