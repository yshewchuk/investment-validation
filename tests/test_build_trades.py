"""build_trades — what a partial rebuild is allowed to replace.

The trades table holds three kinds of row: legacy S1/S2/S3, engine rows for the
strategy being rebuilt, and engine rows for every other strategy. Only the
middle kind may be replaced. Getting this wrong is a silent delete — the table
looks complete afterwards and the analog layer just has less to draw on.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine import build_trades


class _FakeResult:
    def __init__(self, strategy, trades):
        self.strategy = strategy
        self.trades = trades

    def as_dict(self):
        return {"strategy": self.strategy, "priced": len(self.trades)}


@pytest.fixture
def table():
    def row(strategy, provenance, event_id):
        return {"strategy": strategy, "provenance": provenance,
                "event_id": event_id, "ticker": "T",
                "event_date": pd.Timestamp("2024-05-02"), "ret": 0.1}
    return pd.DataFrame([
        row("S2", "legacy:s2", "L1"),
        row("STR-THRU", "engine.replay", "E1"),
        row("STR-RUNUP", "engine.replay", "E2"),
        row("CAL-P", "engine.replay", "E3"),
    ])


def _run(monkeypatch, table, rebuilt_strategy):
    written = {}
    fresh = pd.DataFrame([{
        "strategy": rebuilt_strategy, "provenance": "engine.replay",
        "event_id": "NEW", "ticker": "T",
        "event_date": pd.Timestamp("2024-08-02"), "ret": 0.2,
    }])
    monkeypatch.setattr(build_trades, "event_universe", lambda years=None: pd.DataFrame(
        {"event_id": [], "ticker": [], "event_date": [], "session": []}))
    monkeypatch.setattr(build_trades.replay, "available_chain_keys", lambda: set())
    monkeypatch.setattr(build_trades.replay, "replay",
                        lambda strategy, events: _FakeResult(strategy, fresh))
    monkeypatch.setattr(build_trades.replay, "to_trades_table", lambda results: fresh)
    monkeypatch.setattr(build_trades.store, "read_table", lambda name, **kw: table)
    monkeypatch.setattr(build_trades.store, "write_table",
                        lambda frame, name: written.update(frame=frame))
    monkeypatch.setattr(build_trades.store, "table_stats",
                        lambda name: type("S", (), {"rows": len(written["frame"])})())
    monkeypatch.setattr(build_trades, "coerce", lambda frame, name: frame)
    monkeypatch.setattr(build_trades.manifest, "write_snapshot", lambda: "snap")
    monkeypatch.setattr(build_trades.manifest, "write_manifest", lambda: None)
    report = build_trades.build([rebuilt_strategy])
    return written["frame"], report


class TestPartialRebuild:
    def test_a_partial_rebuild_keeps_the_other_strategies(self, monkeypatch, table):
        frame, _ = _run(monkeypatch, table, "CND-P")
        assert set(frame["event_id"]) == {"L1", "E1", "E2", "E3", "NEW"}

    def test_the_rebuilt_strategy_is_replaced_not_appended(self, monkeypatch, table):
        frame, _ = _run(monkeypatch, table, "STR-THRU")
        thru = frame[frame["strategy"] == "STR-THRU"]
        assert list(thru["event_id"]) == ["NEW"]
        assert set(frame["event_id"]) == {"L1", "E2", "E3", "NEW"}

    def test_legacy_rows_are_never_touched(self, monkeypatch, table):
        for strategy in ("STR-THRU", "CAL-P", "CND-P"):
            frame, _ = _run(monkeypatch, table, strategy)
            assert (frame["provenance"] == "legacy:s2").sum() == 1

    def test_the_report_says_what_was_replaced_and_what_survived(
            self, monkeypatch, table):
        _, report = _run(monkeypatch, table, "CND-P")
        assert report["rebuilt_strategies"] == ["CND-P"]
        assert report["legacy_rows"] == 1
        assert report["kept_engine_rows"] == 3
