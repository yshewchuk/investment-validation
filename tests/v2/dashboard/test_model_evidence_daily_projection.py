"""The projected ``daily_market`` read changes memory, not values (Refs #866)."""
from __future__ import annotations

import tracemalloc

import numpy as np
import pandas as pd

from engine.calendar import TradingCalendar
from engine.dashboard import model_evidence
from engine.features import DAILY_STATE_FIELDS, EVENT_HISTORY_FEATURES
from engine.models.training import implied_t1

TICKERS = ["AAA", "BBB", "CCC"]


def _daily() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=700)
    frames = []
    for ticker in TICKERS:
        n = len(dates)
        frame = pd.DataFrame({"ticker": ticker, "date": dates, "year": dates.year})
        for source in DAILY_STATE_FIELDS:
            frame[source] = rng.uniform(5.0, 60.0, n)
        frame["src_iv"] = "orats"
        frame.loc[frame.index[::7], "src_iv"] = None
        frame["src_spot"] = "x"
        frame["src_mcap"] = "x"
        frame["mcap_usd"] = rng.uniform(1e9, 5e9, n)
        frame["mcap_asof"] = dates
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _install_fake_store(monkeypatch, source: pd.DataFrame) -> None:
    parts = dict(tuple(source.groupby("year")))

    def fake(name, *, years=None, columns=None):
        for year, part in parts.items():
            if years is not None and year not in set(years):
                continue
            yield year, (part[list(columns)] if columns else part).copy()

    monkeypatch.setattr("engine.data.store.iter_table", fake)


def test_projection_keeps_requested_columns_and_values(monkeypatch):
    _install_fake_store(monkeypatch, _daily())
    cols = list(model_evidence._DAILY_STATE_READ)
    proj = model_evidence._daily_subset(["AAA", "BBB"], years=[2022, 2023], columns=cols)
    full = model_evidence._daily_subset(["AAA", "BBB"], years=[2022, 2023])
    assert list(proj.columns) == cols
    pd.testing.assert_frame_equal(proj, full[cols])

    source = _daily()
    expected = source[
        source["ticker"].isin(["AAA", "BBB"]) & source["year"].isin([2022, 2023])
    ].sort_values("year", kind="stable")[cols].reset_index(drop=True)
    assert set(proj["ticker"]) == {"AAA", "BBB"}
    pd.testing.assert_frame_equal(proj, expected)


def test_implied_t1_dataset_identical_with_projection(monkeypatch):
    _install_fake_store(monkeypatch, _daily())
    monkeypatch.setattr(implied_t1, "trading_calendar", lambda: TradingCalendar(pd.bdate_range("2021-01-04", "2024-06-28")))
    cols = list(model_evidence._DAILY_STATE_READ)
    events = pd.DataFrame(
        {
            "ticker": TICKERS,
            "event_date": pd.to_datetime(["2023-05-04", "2023-06-08", "2023-07-20"]),
            "session": ["AMC", "BMO", "AMC"],
        }
    )
    rng = np.random.default_rng(1)
    panel = pd.DataFrame({"ticker": TICKERS, "date": events["event_date"]})
    for feature in EVENT_HISTORY_FEATURES:
        panel[feature] = rng.normal(size=len(panel))
    full = model_evidence._daily_subset(TICKERS, years=[2022, 2023])
    proj = model_evidence._daily_subset(TICKERS, years=[2022, 2023], columns=cols)
    a = implied_t1.build_dataset(events, panel=panel, daily=full)
    b = implied_t1.build_dataset(events, panel=panel, daily=proj)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)


def test_projection_lowers_peak_memory(monkeypatch):
    _install_fake_store(monkeypatch, _daily())
    cols = list(model_evidence._DAILY_STATE_READ)

    def peak(**kwargs) -> int:
        tracemalloc.start()
        try:
            model_evidence._daily_subset(TICKERS, years=[2022, 2023], **kwargs)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    assert peak(columns=cols) < 0.9 * peak()


def _dispatch_calls(monkeypatch, role, module_name):
    import importlib

    calls = []

    def fake_subset(tickers, years=None, columns=None):
        calls.append(columns)
        return pd.DataFrame()

    events = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "event_date": pd.to_datetime(["2023-05-04"]),
            "session": ["AMC"],
        }
    )
    module = importlib.import_module(f"engine.models.training.{module_name}")
    monkeypatch.setattr(model_evidence, "_daily_subset", fake_subset)
    monkeypatch.setattr(
        "engine.models.training.train_all._events_with_session", lambda: events
    )
    monkeypatch.setattr(
        module, "build_dataset", lambda *a, **k: pd.DataFrame({"x": [1.0]})
    )
    model_evidence._dataset_for(role, "STR-RUNUP", panel=None, daily=None, trades=None)
    return calls


def test_dataset_for_implied_t1_projects_columns(monkeypatch):
    calls = _dispatch_calls(monkeypatch, "implied_t1", "implied_t1")
    assert calls == [model_evidence._DAILY_STATE_READ]


def test_dataset_for_runup_move_projects_columns(monkeypatch):
    calls = _dispatch_calls(monkeypatch, "runup_move", "runup_move")
    assert calls == [model_evidence._DAILY_STATE_READ]