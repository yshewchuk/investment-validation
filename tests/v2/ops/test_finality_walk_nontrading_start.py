"""``resolve_final_session`` must not walk one session too many when the
requested date is not itself a trading day.

``TradingCalendar.shift(date, -1)`` already snaps a non-trading date to the
prior session, so the first walk step must snap to that session instead of
subtracting one more.
"""
from __future__ import annotations

import pandas as pd
import pytest

_DAYS = [d for d in pd.bdate_range("2026-10-01", "2026-10-30")
         if d != pd.Timestamp("2026-10-12")]


class _Calendar:
    """Trading days as snap-prev semantics: shift = last day <= date, plus n."""

    def __init__(self):
        self.days = _DAYS

    def is_trading_day(self, date):
        return pd.Timestamp(date).normalize() in set(self.days)

    def shift(self, date, n):
        stamp = pd.Timestamp(date).normalize()
        prior = [d for d in self.days if d <= stamp]
        return self.days[len(prior) - 1 + n]


class _Result:
    def __init__(self, stamp, is_final):
        self.date = str(pd.Timestamp(stamp).date())
        self.is_final = is_final


def _last_trading_day(date):
    stamp = pd.Timestamp(date).normalize()
    return [d for d in _DAYS if d <= stamp][-1]


def _patch(monkeypatch, module, final_days, seen):
    def fake_finality(value, tickers, **kwargs):
        stamp = pd.Timestamp(value).normalize()
        seen.append(stamp)
        return _Result(stamp, stamp in final_days)

    monkeypatch.setattr(module, "session_finality", fake_finality)
    monkeypatch.setattr(module, "_coverage_frame", lambda *a, **k: None)


CASES = [
    ("2026-10-17", "2026-10-16", "2026-10-16"),  # Saturday request
    ("2026-10-18", "2026-10-16", "2026-10-16"),  # Sunday request
    ("2026-10-12", "2026-10-09", "2026-10-09"),  # holiday Monday request
    ("2026-10-16", "2026-10-16", "2026-10-16"),  # trading day, same day final
    ("2026-10-16", "2026-10-15", "2026-10-15"),  # trading day, walks back one
]


@pytest.mark.parametrize("impl", ["legacy", "native"])
@pytest.mark.parametrize("requested,final_day,expected", CASES)
def test_walk_snaps_nontrading_start(impl, requested, final_day, expected, monkeypatch):
    final_days = {pd.Timestamp(final_day).normalize()}
    seen = []
    calendar = _Calendar()

    if impl == "legacy":
        from engine.data import finality as module
        _patch(monkeypatch, module, final_days, seen)
        result = module.resolve_final_session(requested, ["A"], calendar=calendar)
    else:
        from engine.v2.ops import finality as module
        _patch(monkeypatch, module, final_days, seen)
        result = module._native_resolve_final_session(
            requested, ["A"], calendar=calendar, reads=object())

    assert result.date == expected
    # The first session tested is the snap target, never one session earlier.
    assert seen[0] == _last_trading_day(requested)
    assert all(calendar.is_trading_day(s) for s in seen)


def _resolve(impl, monkeypatch, requested, final_days, seen, **kw):
    calendar = _Calendar()
    if impl == "legacy":
        from engine.data import finality as module
        _patch(monkeypatch, module, final_days, seen)
        return module.resolve_final_session(requested, ["A"], calendar=calendar, **kw)
    from engine.v2.ops import finality as module
    _patch(monkeypatch, module, final_days, seen)
    return module._native_resolve_final_session(
        requested, ["A"], calendar=calendar, reads=object(), **kw)


@pytest.mark.parametrize("impl", ["legacy", "native"])
def test_nontrading_request_falls_back_past_nonfinal_snap(impl, monkeypatch):
    final_days = {pd.Timestamp("2026-10-15").normalize()}
    seen = []

    result = _resolve(impl, monkeypatch, "2026-10-18", final_days, seen)

    assert result.date == "2026-10-15"
    assert result.is_final is True
    assert seen == [pd.Timestamp("2026-10-16"), pd.Timestamp("2026-10-15")]


@pytest.mark.parametrize("impl", ["legacy", "native"])
def test_walk_budget_after_snap(impl, monkeypatch):
    seen = []

    with pytest.raises(RuntimeError):
        _resolve(impl, monkeypatch, "2026-10-18", set(), seen, max_sessions=3)

    assert seen == [pd.Timestamp("2026-10-16"), pd.Timestamp("2026-10-15"),
                    pd.Timestamp("2026-10-14")]
