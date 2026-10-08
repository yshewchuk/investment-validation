"""The cached finality callable must agree with per-call finality while
reading each coverage window once.

:func:`engine.data.finality.cached_session_finality` memoizes the
``{year - 1, year}`` coverage window per (table, exit-year). These tests pin
the two properties that make the optimization safe: identical
``SessionFinality`` evidence, and exactly one store read per (table, window)
no matter how many ``(date, ticker)`` checks share it -- including through
``engine.ledger.score_outcomes``.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from engine import ledger
from engine.data import finality
from engine.ledger_settlement import POLICY
from engine.structures import STRUCTURES

D1 = "2025-12-30"
D2 = "2026-01-06"
D3 = "2026-01-07"
D4 = "2026-02-03"
TICKERS = ["A", "B", "C", "D", "E"]


class _Entry:
    def __init__(self, endpoint, day):
        self.endpoint = endpoint
        self.params = {"tradeDate": day}
        self.meta = {"status": 200}


class _Counter:
    def __init__(self):
        self.reads = []

    def reset(self):
        self.reads.clear()


def _rows(column):
    rows = []
    for ticker in ("A", "B", "C", "D"):
        rows.append({column: D1, "ticker": ticker})
    for ticker in ("A", "B", "C", "D", "E"):
        rows.append({column: D2, "ticker": ticker})
    for ticker in ("A", "B", "E"):
        rows.append({column: D3, "ticker": ticker})
    rows.append({column: D2, "ticker": "A"})
    rows.append({column: f"{D2}T09:30:00", "ticker": "D"})
    rows.append({column: D1, "ticker": np.nan})
    return pd.DataFrame(rows)


def _store_frames():
    return _rows("date"), _rows("obs_date")


def _entries():
    out = []
    for day in (D1, D2, D3):
        out.append(_Entry("hist/summaries", day))
        out.append(_Entry("hist/cores", day))
    return out


@pytest.fixture()
def counting_store(monkeypatch):
    daily, chains = _store_frames()
    counter = _Counter()

    def read(table, **kwargs):
        counter.reads.append((table, tuple(sorted(kwargs.get("years") or []))))
        frame = daily if table == "daily_market" else chains
        years = kwargs.get("years")
        if years is not None:
            column = "date" if table == "daily_market" else "obs_date"
            frame = frame[pd.to_datetime(frame[column],
                                         errors="coerce").dt.year.isin(years)]
        columns = kwargs.get("columns")
        if columns is not None:
            frame = frame.loc[:, [c for c in columns if c in frame.columns]]
        return frame.copy()

    monkeypatch.setattr(finality.store, "read_table", read)
    monkeypatch.setattr(finality.fetch, "iter_cached", lambda source: list(_entries()))
    return counter


def _calls():
    calls = [(date, [ticker]) for date in (D1, D2, D3, D4) for ticker in TICKERS]
    calls.append((D3, ["C", "D"]))
    calls.append((D2, ["A", "B", "C"]))
    calls.append((D1, ["A", "ZZZ"]))
    return calls


def test_cached_equals_per_call(counting_store):
    run = finality.cached_session_finality()
    seen = set()
    for date, tickers in _calls():
        cached = run(date, tickers).as_dict()
        direct = finality.session_finality(date, tickers).as_dict()
        assert cached == direct, (date, tickers)
        seen.add(cached["is_final"])
    assert seen == {True, False}


def test_reads_once_per_year_window(counting_store):
    calls = _calls()
    for date, tickers in calls:
        finality.session_finality(date, tickers)
    assert len(counting_store.reads) == 2 * len(calls)

    counting_store.reset()
    run = finality.cached_session_finality()
    for date, tickers in calls:
        run(date, tickers)
    assert len(counting_store.reads) == 4

    reads = list(counting_store.reads)
    for _table, years in reads:
        assert list(years) == sorted(years)
        assert years[1] == years[0] + 1
    assert {years for _table, years in reads} == {(2024, 2025), (2025, 2026)}

    for date, tickers in calls:
        run(date, tickers)
    assert len(counting_store.reads) == 4


def test_failed_read_agrees(monkeypatch):
    def missing(table, **kwargs):
        raise FileNotFoundError(table)

    monkeypatch.setattr(finality.store, "read_table", missing)
    monkeypatch.setattr(finality.fetch, "iter_cached", lambda source: list(_entries()))

    run = finality.cached_session_finality()
    cached = run(D2, ["A"]).as_dict()
    direct = finality.session_finality(D2, ["A"]).as_dict()
    assert cached == direct
    assert cached["is_final"] is False


# --------------------------------------------------------------------------
# score_outcomes integration
# --------------------------------------------------------------------------


@pytest.fixture()
def ledger_root(tmp_path, monkeypatch):
    from engine import paths

    monkeypatch.setattr(paths, "LEDGER", tmp_path / "ledger")
    monkeypatch.setattr(ledger, "_settlement_calendar", lambda: pd.DataFrame(
        [{k: r[k] for k in ("event_id", "ticker", "event_date", "session")}
         for r in ledger.read_predictions()],
        columns=["event_id", "ticker", "event_date", "session"]))
    return tmp_path / "ledger"


def _prediction(ticker="A", strategy="STR-THRU", structure=None, **overrides) -> dict:
    as_of = "2026-10-06"
    rid = ledger.row_id(as_of, ticker, strategy, None, "2026-02-20")
    row = {
        "schema_version": ledger.SCHEMA_VERSION,
        "row_id": rid,
        "written_at": "2026-10-06T21:05:03+00:00",
        "as_of": as_of,
        "decision_ts": f"{as_of}T20:00:00+00:00",
        "ticker": ticker,
        "event_id": f"{ticker}-{D2}",
        "event_date": D2,
        "session": "AMC",
        "strategy": strategy,
        "settlement": {"policy": POLICY, "spec_version": 1,
                       "structure_spec": asdict(STRUCTURES[strategy]())},
        "structure": structure or {"strike": None, "expiry": "2026-02-20"},
        "intended_prices": {"alpha": 0.5, "entry_cost": 8.42},
        "score": {"win_model": 0.55, "exp_pnl_model": 0.04, "gate_pass": True},
        "model_versions": {"gate": "gate_midfill_str_thru@1"},
        "snapshot_hash": "dce985",
        "audit_receipt": None,
        "supersedes": None,
        "supersede_reason": None,
    }
    row.update(overrides)
    return row


def _fake_replay(trades: pd.DataFrame):
    class _Result:
        def __init__(self, frame):
            self.trades = frame

    def _replay(strategy, events, **kwargs):
        return _Result(trades[trades["strategy"] == strategy]
                       if "strategy" in trades.columns else trades)

    return _replay


_EXITS = {"A": D1, "B": D2, "C": D2, "D": D3, "E": D1}


def _scoring_rows():
    return [
        _prediction(ticker=ticker, event_id=f"{ticker}-{exit_date}",
                    event_date=exit_date,
                    structure={"strike": None, "expiry": "2026-02-20",
                               "exit_date": exit_date})
        for ticker, exit_date in _EXITS.items()
    ]


def test_score_outcomes_reads_once(ledger_root, counting_store, monkeypatch):
    from engine import replay as replay_mod

    rows = _scoring_rows()
    ledger.write_predictions(rows)
    baseline = {
        row["row_id"]: finality.session_finality(
            row["structure"]["exit_date"], [row["ticker"]]).is_final
        for row in rows
    }
    assert any(baseline.values()) and not all(baseline.values())
    counting_store.reset()

    monkeypatch.setattr(replay_mod, "replay", _fake_replay(pd.DataFrame()))
    result = ledger.score_outcomes(through="2026-02-10")
    reads = list(counting_store.reads)
    assert len(reads) == 4, reads

    assert result["deferred"] == sum(1 for final in baseline.values() if not final)

    outcomes = ledger.read_outcomes()
    assert outcomes
    assert any(o["exit_finality"]["is_final"] for o in outcomes)
    predictions = {row["row_id"]: row for row in ledger.read_predictions()}
    for outcome in outcomes:
        prediction = predictions[outcome["row_id"]]
        exit_date = prediction["structure"]["exit_date"]
        expected = finality.session_finality(exit_date, [prediction["ticker"]]).as_dict()
        assert outcome["exit_finality"] == expected
