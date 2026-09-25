"""Pure computed-move math, moved from ``engine.data.pulls.computed_moves``
(spec s4b Change 2).

``session_move``/``MIN_SCOREABLE``/``MAX_GAP_CALENDAR_DAYS`` are copied
verbatim from the untouched legacy pull (the source of record for the legacy
path); the legacy module itself is never edited. :func:`build_rows` keeps
``build_ticker``'s computation byte-for-byte and changes only its return
shape -- one row dict per event, matching
:mod:`engine.v2.data.computed_moves_table`'s columns, including one row per
SKIPPED event (``skipped=True``, ``realized_move_pct=None``). A skip must be
visible, not just counted (EXP-117 DEFINITION.md R3's materiality rule).

Layer 1 of ``system_rearchitecture.md`` §4.1: no ``engine.v2.ops`` and no
legacy ``engine.*`` imports -- this module is pure numpy/pandas math, the same
rule ``price_history_table.py`` states for its own layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "MAX_GAP_CALENDAR_DAYS",
    "MIN_SCOREABLE",
    "NativeTradingCalendar",
    "build_rows",
    "build_ticker",
    "native_trading_calendar",
    "projected_trading_days",
    "session_move",
    "us_market_holidays",
]

#: The scoreability bar the board's champion models impose (span-12 EMAs).
MIN_SCOREABLE = 12

#: P→Q windows wider than this are halts/gaps, excluded not guessed.
MAX_GAP_CALENDAR_DAYS = 5

#: Panel admission is k>=4, so fewer than this many computable events cannot score.
ADMISSION_EVENTS = 5


def session_move(sd, sc, t, session) -> float | None:
    if session == "BMO":
        j_pre = int(np.searchsorted(sd, t, side="left")) - 1
        j_post = int(np.searchsorted(sd, t, side="left"))
    else:
        j_pre = int(np.searchsorted(sd, t, side="right")) - 1
        j_post = int(np.searchsorted(sd, t, side="right"))
    if j_pre < 0 or j_post >= len(sd):
        return None
    if (sd[j_post] - sd[j_pre]) / np.timedelta64(1, "D") > MAX_GAP_CALENDAR_DAYS:
        return None
    p, q = sc[j_pre], sc[j_post]
    if not np.isfinite(p) or not np.isfinite(q) or p <= 0:
        return None
    return float((q / p - 1.0) * 100.0)


def _row(ticker: str, event_date: str, *, realized, implied, quarter: int, skipped: bool,
         computed_at: str, source_hash: str, capture_id: str) -> dict:
    return {
        "ticker": ticker,
        "event_date": event_date,
        "realized_move_pct": realized,
        "implied_move_pct": implied,
        "quarter_ordinal": quarter,
        "skipped": skipped,
        "computed_at": computed_at,
        "source_hash": source_hash,
        "capture_id": capture_id,
    }


def build_rows(ticker: str, events: pd.DataFrame, sd, sc, daily: pd.DataFrame, *,
               computed_at: str, source_hash: str, capture_id: str) -> list[dict]:
    """One row dict per event for ``ticker`` -- skipped events included.

    The computation is the legacy ``build_ticker`` body verbatim: the
    session-aware close-to-close realized move, the panel as-of implied move
    (last EOD row strictly before the print), and a per-calendar-year ordinal.
    The only changed thing is the return: a list of contract-shaped rows
    instead of one oquants document per ticker. Fewer than five computable
    events returns ``[]`` -- the same admission rule the legacy function
    applies by returning ``None``, never a partial write.
    """
    rows: list[dict] = []
    dm_dates = daily["date"].to_numpy()
    dm_im = daily["implied_move"].to_numpy(dtype=float)

    year_seen: dict[int, int] = {}
    computable = 0
    for r in events.itertuples():
        t = r.event_date.to_datetime64()
        year = int(str(r.event_date)[:4])
        m = session_move(sd, sc, t, r.session)
        if m is None:
            rows.append(_row(
                ticker, str(r.event_date)[:10], realized=None, implied=None,
                quarter=year_seen.get(year, 0) + 1, skipped=True, computed_at=computed_at,
                source_hash=source_hash, capture_id=capture_id))
            continue
        # panel as-of convention: the last EOD row strictly before the print
        j = int(np.searchsorted(dm_dates, t, side="left")) - 1
        im = float(dm_im[j]) if j >= 0 and np.isfinite(dm_im[j]) else None
        year_seen[year] = year_seen.get(year, 0) + 1
        computable += 1
        rows.append(_row(
            ticker, str(r.event_date)[:10], realized=m, implied=im,
            quarter=year_seen[year], skipped=False, computed_at=computed_at,
            source_hash=source_hash, capture_id=capture_id))

    if computable < ADMISSION_EVENTS:  # panel admission is k>=4, so fewer cannot score
        return []
    return rows


#: The legacy name of the same builder, kept so a reader of the moved code
#: finds the function it was moved from (spec s4b Change 2).
build_ticker = build_rows


# --------------------------------------------------------------------------
# the native forward trading calendar (spec s4c Rewrite 1)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeTradingCalendar:
    """A trading-day list whose observed prefix came from a pinned snapshot.

    ``observed_through`` is the last real session the snapshot carries;
    anything after it is rule-projected weekdays (holidays removed), exactly
    the split ``engine.calendar.TradingCalendar`` makes. Only ``days`` is
    consumed by ``horizon_dates``.
    """

    days: tuple[pd.Timestamp, ...]
    observed_through: pd.Timestamp


def _easter(year: int) -> pd.Timestamp:
    """Gregorian Easter Sunday (anonymous computus) — Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lam) // 451
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return pd.Timestamp(year=year, month=month, day=day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """``n``-th ``weekday`` (Mon=0) of a month; ``n=-1`` means the last one."""
    days = pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
    days = days[(days.month == month) & (days.weekday == weekday)]
    return days[n if n < 0 else n - 1]


def _observed(date: pd.Timestamp) -> pd.Timestamp:
    """NYSE observation rule: Saturday -> the Friday before, Sunday -> the Monday after."""
    if date.weekday() == 5:
        return date - pd.Timedelta(days=1)
    if date.weekday() == 6:
        return date + pd.Timedelta(days=1)
    return date


def _observed_new_year(year: int) -> pd.Timestamp | None:
    """New Year's Day, which does *not* follow the Saturday->Friday rule."""
    day = pd.Timestamp(year=year, month=1, day=1)
    if day.weekday() == 5:
        return None
    return _observed(day)


def us_market_holidays(year: int) -> set[pd.Timestamp]:
    """The scheduled NYSE holidays for ``year``, with observation rules applied.

    Moved verbatim (spec s4c Rewrite 1) from ``engine.calendar`` — pure
    arithmetic, no I/O — so the v2 layer never imports the legacy module.
    """
    out = {
        _nth_weekday(year, 1, 0, 3),  # MLK Day (from 1998)
        _nth_weekday(year, 2, 0, 3),  # Presidents' Day
        _easter(year) - pd.Timedelta(days=2),  # Good Friday
        _nth_weekday(year, 5, 0, -1),  # Memorial Day
        _observed(pd.Timestamp(year=year, month=7, day=4)),
        _nth_weekday(year, 9, 0, 1),  # Labor Day
        _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        _observed(pd.Timestamp(year=year, month=12, day=25)),
    }
    new_year = _observed_new_year(year)
    if new_year is not None:
        out.add(new_year)
    if year >= 2022:  # Juneteenth became a market holiday in 2022
        out.add(_observed(pd.Timestamp(year=year, month=6, day=19)))
    return out


def projected_trading_days(start, end) -> pd.DatetimeIndex:
    """Weekdays in ``(start, end]`` that are not scheduled market holidays.

    Moved verbatim (spec s4c Rewrite 1) from ``engine.calendar``; used only to
    extend the calendar *past* the last session the pinned snapshot observed.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return pd.DatetimeIndex([])
    days = pd.date_range(start + pd.Timedelta(days=1), end, freq="B")
    holidays: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        holidays |= us_market_holidays(year)
    return days[~days.isin(pd.DatetimeIndex(sorted(holidays)))]


def native_trading_calendar(daily_by_ticker, extend_days: int = 400) -> NativeTradingCalendar:
    """The forward calendar sourced from the pinned snapshot's own ``daily_market``.

    ``daily_by_ticker`` is one pass of the snapshot's ``daily_market`` rows
    grouped by ticker (``engine/v2/ops/*_store``'s single scan). The observed
    sessions are the DISTINCT ``date`` values across every ticker — the same
    "a day the index did not trade is a day no chain was observed" rule the
    legacy calendar derives from its S&P series, but read from the pinned
    snapshot instead of a mutable legacy CSV. Future dates are rule-projected
    (``projected_trading_days``), never observed.
    """
    observed = sorted({
        pd.Timestamp(value).normalize()
        for frame in daily_by_ticker.values()
        for value in frame["date"]
    })
    if not observed:
        raise ValueError("native trading calendar needs at least one daily_market session")
    last = observed[-1]
    future = projected_trading_days(last, last + pd.Timedelta(days=extend_days))
    return NativeTradingCalendar(tuple(observed) + tuple(future), last)
