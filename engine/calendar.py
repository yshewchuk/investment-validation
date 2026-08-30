"""The canonical earnings calendar, and trading-day arithmetic around a print.

Two jobs live here.

**The calendar itself.** ORATS ``/hist/earnings`` is authoritative: it carries
``anncTod``, which gives the BMO/AMC session directly, and it agreed with the
oquants panel on 99.52% of dates (EXP-038). The oquants panel is kept as a
cross-check — where the two disagree the row is flagged ``date_agree=False``
and ORATS wins, so a disagreement is visible in the data instead of being
silently resolved.

**Session-aware day arithmetic.** "Shortly before the print" is not one date.
A BMO announcement lands before the open of its event date, so the last
pre-print close is the *previous* trading day; an AMC announcement lands after
the close, so the last pre-print close is the event date itself. This is the
same convention the realized-move panel uses (BMO: close(t−1)→close(t), AMC:
close(t)→close(t+1), validated in EXP-000), and getting it wrong shifts every
entry and exit by a day.

Structure offsets are therefore expressed relative to the print:

===========  ==========================================================
offset        meaning
===========  ==========================================================
``0``         last pre-print close
``-1``        one trading day earlier
``+1``        first post-print close
===========  ==========================================================
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from engine import paths

__all__ = [
    "BMO",
    "AMC",
    "session_from_annc_tod",
    "TradingCalendar",
    "trading_calendar",
    "us_market_holidays",
    "projected_trading_days",
    "load_orats_earnings",
    "load_oquants_event_dates",
    "build_calendar",
    "detect_date_changes",
    "DateChange",
    "UNSCHEDULED_CLOSURES",
]

BMO = "BMO"
AMC = "AMC"

#: ORATS ``anncTod`` is an HHMM string. Anything before noon is a pre-market
#: announcement; 1600/1630 (the other mode) is after the close.
BMO_CUTOFF = 1200


def session_from_annc_tod(annc_tod) -> str | None:
    """Map ORATS ``anncTod`` to ``"BMO"`` / ``"AMC"``; ``None`` when unusable."""
    if annc_tod is None:
        return None
    if isinstance(annc_tod, float) and np.isnan(annc_tod):
        return None
    text = str(annc_tod).strip()
    if not text or text.lower() in ("none", "nan"):
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        return None
    try:
        hhmm = int(digits[-4:]) if len(digits) >= 3 else int(digits) * 100
    except ValueError:
        return None
    if not 0 <= hhmm <= 2359:
        return None
    return BMO if hhmm < BMO_CUTOFF else AMC


# --------------------------------------------------------------------------
# the forward trading calendar
# --------------------------------------------------------------------------

#: One-off NYSE closures that no holiday rule generates. Kept explicit so the
#: rule-based generator can be validated against real history exactly; the test
#: suite asserts these are the *only* discrepancies over 2006–today.
UNSCHEDULED_CLOSURES = {
    "2007-01-02",  # national day of mourning, President Ford
    "2012-10-29",  # Hurricane Sandy
    "2012-10-30",  # Hurricane Sandy
    "2018-12-05",  # national day of mourning, President G.H.W. Bush
    "2025-01-09",  # national day of mourning, President Carter
}


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
    """NYSE observation rule: Saturday → the Friday before, Sunday → the Monday after."""
    if date.weekday() == 5:
        return date - pd.Timedelta(days=1)
    if date.weekday() == 6:
        return date + pd.Timedelta(days=1)
    return date


def _observed_new_year(year: int) -> pd.Timestamp | None:
    """New Year's Day, which does *not* follow the Saturday→Friday rule.

    When January 1 falls on a Saturday the NYSE simply takes no holiday — it
    does not close the preceding Friday, unlike July 4 and Christmas. Getting
    this wrong closes the market on 2010-12-31 and 2021-12-31, both of which
    were full trading days.
    """
    day = pd.Timestamp(year=year, month=1, day=1)
    if day.weekday() == 5:
        return None
    return _observed(day)


def us_market_holidays(year: int) -> set[pd.Timestamp]:
    """The scheduled NYSE holidays for ``year``, with observation rules applied."""
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

    Used only to extend the calendar *past* the end of the price history, so
    upcoming events can be scored. Historical days always come from the actual
    index series, never from these rules.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if end <= start:
        return pd.DatetimeIndex([])
    days = pd.date_range(start + pd.Timedelta(days=1), end, freq="B")
    holidays: set[pd.Timestamp] = set()
    for year in range(start.year, end.year + 1):
        holidays |= us_market_holidays(year)
    return days[~days.isin(pd.DatetimeIndex(sorted(holidays)))]


# --------------------------------------------------------------------------
# trading-day arithmetic
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PrintWindow:
    """The dates a structure actually trades on, for one event."""

    event_date: pd.Timestamp
    session: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    last_pre_print: pd.Timestamp
    first_post_print: pd.Timestamp


class TradingCalendar:
    """Trading days, taken from the S&P 500 daily series (2006 → today).

    Using an index series rather than a holiday library keeps the calendar
    consistent with the price data the panel is built from: a day the index did
    not trade is a day no chain was observed, so it cannot be an entry or exit
    date by construction.
    """

    def __init__(self, days: Sequence[pd.Timestamp], *, observed_through=None):
        idx = pd.DatetimeIndex(pd.to_datetime(list(days))).normalize().unique().sort_values()
        if len(idx) == 0:
            raise ValueError("trading calendar cannot be empty")
        self.days = idx
        self._pos = {d: i for i, d in enumerate(idx)}
        #: Last day that came from real price history. Anything after it is
        #: rule-projected and could be wrong about an unscheduled closure.
        self.observed_through = (
            pd.Timestamp(observed_through).normalize() if observed_through is not None else idx[-1]
        )

    def is_projected(self, date) -> bool:
        return pd.Timestamp(date).normalize() > self.observed_through

    def __len__(self) -> int:
        return len(self.days)

    @property
    def first(self) -> pd.Timestamp:
        return self.days[0]

    @property
    def last(self) -> pd.Timestamp:
        return self.days[-1]

    def is_trading_day(self, date) -> bool:
        return pd.Timestamp(date).normalize() in self._pos

    def index_of(self, date, *, side: str = "exact") -> int:
        """Position of ``date``. ``side`` ∈ {exact, prev, next}."""
        d = pd.Timestamp(date).normalize()
        if side == "exact":
            if d not in self._pos:
                raise KeyError(f"{d.date()} is not a trading day")
            return self._pos[d]
        pos = int(self.days.searchsorted(d, side="left"))
        if side == "next":
            if pos >= len(self.days):
                raise KeyError(f"no trading day on or after {d.date()}")
            return pos
        if side == "prev":
            if pos < len(self.days) and self.days[pos] == d:
                return pos
            if pos == 0:
                raise KeyError(f"no trading day on or before {d.date()}")
            return pos - 1
        raise ValueError(f"unknown side {side!r}")

    def shift(self, date, n: int, *, side: str = "prev") -> pd.Timestamp:
        """``n`` trading days from ``date`` (negative = earlier)."""
        pos = self.index_of(date, side=side) + n
        if not 0 <= pos < len(self.days):
            raise KeyError(f"{n:+d} trading days from {pd.Timestamp(date).date()} is out of range")
        return self.days[pos]

    # -- session-aware anchors -------------------------------------------

    def last_pre_print(self, event_date, session: str) -> pd.Timestamp:
        """The last close that is strictly information-free about the print."""
        d = pd.Timestamp(event_date).normalize()
        if session == AMC:
            return self.days[self.index_of(d, side="prev")]
        if session == BMO:
            pos = self.index_of(d, side="prev")
            if self.days[pos] == d:
                pos -= 1
            if pos < 0:
                raise KeyError(f"no trading day before {d.date()}")
            return self.days[pos]
        raise ValueError(f"unknown session {session!r}")

    def first_post_print(self, event_date, session: str) -> pd.Timestamp:
        """The first close that already reflects the print."""
        d = pd.Timestamp(event_date).normalize()
        if session == BMO:
            return self.days[self.index_of(d, side="next")]
        if session == AMC:
            pos = self.index_of(d, side="next")
            if self.days[pos] == d:
                pos += 1
            if pos >= len(self.days):
                raise KeyError(f"no trading day after {d.date()}")
            return self.days[pos]
        raise ValueError(f"unknown session {session!r}")

    def resolve_offsets(
        self, event_date, session: str, entry_offset: int, exit_offset: int
    ) -> PrintWindow:
        """Map a structure's ``(entry_offset, exit_offset)`` onto real dates.

        Offsets are anchored on the last pre-print close (offset 0). Positive
        offsets step forward from the *first post-print* close, so ``+1`` is
        that close and ``+2`` the one after — there is no offset that lands on
        the print itself, because no chain is observed mid-event.
        """
        pre = self.last_pre_print(event_date, session)
        post = self.first_post_print(event_date, session)

        def resolve(offset: int) -> pd.Timestamp:
            if offset <= 0:
                return self.shift(pre, offset)
            return self.shift(post, offset - 1)

        return PrintWindow(
            event_date=pd.Timestamp(event_date).normalize(),
            session=session,
            entry_date=resolve(entry_offset),
            exit_date=resolve(exit_offset),
            last_pre_print=pre,
            first_post_print=post,
        )


@lru_cache(maxsize=2)
def trading_calendar(extend_days: int = 400) -> TradingCalendar:
    """Trading days from the cached S&P 500 daily series, extended forward.

    History comes from the index series (a day the index did not trade is a day
    no chain was observed). ``extend_days`` calendar days of rule-projected
    weekdays are appended so events past the end of the price history — the
    upcoming prints Phase 3 scores — still resolve to entry and exit dates.
    """
    path = paths.GSPC_DAILY
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — the trading calendar is derived from the S&P daily series"
        )
    # yfinance multi-header: row 0 is field names, rows 1-2 are ticker/blank.
    df = pd.read_csv(path, skiprows=3, header=None, usecols=[0], names=["date"])
    observed = pd.to_datetime(df["date"], errors="coerce").dropna()
    last = pd.Timestamp(observed.max()).normalize()
    future = projected_trading_days(last, last + pd.Timedelta(days=extend_days))
    return TradingCalendar(
        list(observed) + list(future), observed_through=last
    )


# --------------------------------------------------------------------------
# calendar sources
# --------------------------------------------------------------------------


def _read_gz_json(path: Path):
    with gzip.open(path, "rt") as fh:
        return json.load(fh)


def load_orats_earnings(tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Historical + forward earnings dates from the cached ORATS calendar.

    Returns ``ticker, event_date, annc_tod, session, updated_at``.
    """
    root = paths.RAW_ORATS_EARNINGS
    wanted = set(tickers) if tickers is not None else None
    records: list[dict] = []

    def take(row: dict, fallback_ticker: str | None = None) -> None:
        date = row.get("earnDate")
        if not date:
            return
        ticker = row.get("ticker") or fallback_ticker
        if wanted is not None and ticker not in wanted:
            return
        records.append(
            {
                "ticker": ticker,
                "event_date": date,
                "annc_tod": row.get("anncTod"),
                "updated_at": row.get("updatedAt"),
            }
        )

    if root.exists():
        for path in sorted(root.glob("*.json.gz")):
            ticker = path.name[: -len(".json.gz")]
            if wanted is not None and ticker not in wanted:
                continue
            for row in _read_gz_json(path) or []:
                take(row, ticker)

    # Anything the Tier-1 fetch wrapper has pulled since. Dates move, so a
    # refresh has to be able to add to the calendar — and on a restored machine
    # the legacy tree above does not exist at all, making this the only source.
    from engine.data.normalize.fetch_store import iter_orats_rows

    for source in iter_orats_rows("hist/earnings"):
        for row in source.rows:
            take(row)

    if not records and not root.exists():
        raise FileNotFoundError(
            f"no ORATS earnings data: neither {root} nor the Tier-1 fetch store "
            "holds any hist/earnings response"
        )
    if not records:
        return pd.DataFrame(
            columns=["ticker", "event_date", "annc_tod", "session", "updated_at"]
        )
    df = pd.DataFrame.from_records(records)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date"])
    df["session"] = [session_from_annc_tod(v) for v in df["annc_tod"]]
    df = df.drop_duplicates(["ticker", "event_date"]).sort_values(["ticker", "event_date"])
    return df.reset_index(drop=True)


def load_oquants_event_dates(tickers: Iterable[str] | None = None) -> pd.DataFrame:
    """Event dates from the cached oquants moves files (the cross-check source)."""
    root = paths.RAW_OQUANTS_MOVES
    if not root.exists():
        raise FileNotFoundError(f"{root} missing — oquants moves cache not present")
    wanted = set(tickers) if tickers is not None else None

    records: list[dict] = []
    for path in sorted(root.glob("moves_*.json")):
        ticker = path.name[len("moves_") : -len(".json")]
        if wanted is not None and ticker not in wanted:
            continue
        doc = json.loads(path.read_text())
        data = doc.get("data") or {}
        for date in data.get("dates") or []:
            records.append({"ticker": doc.get("ticker") or ticker, "event_date": date})
    if not records:
        return pd.DataFrame(columns=["ticker", "event_date"])
    df = pd.DataFrame.from_records(records)
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    return (
        df.dropna(subset=["event_date"])
        .drop_duplicates(["ticker", "event_date"])
        .sort_values(["ticker", "event_date"])
        .reset_index(drop=True)
    )


def build_calendar(
    orats: pd.DataFrame | None = None,
    oquants: pd.DataFrame | None = None,
    tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Merge the two sources into the canonical calendar.

    Output columns match the Tier-2 ``earnings_events`` schema: ``event_id``,
    ``ticker``, ``event_date``, ``session``, ``annc_tod``, ``src_orats``,
    ``src_oquants``, ``date_agree``, ``updated_at``.

    ``date_agree`` is False when only one source has the event. It is not a
    quality verdict — oquants only covers 2007+ and only names it tracked,
    while the ORATS calendar reaches back to the 1980s — but it is the flag any
    consumer needs in order to restrict to the doubly-confirmed subset.
    """
    orats = load_orats_earnings(tickers) if orats is None else orats.copy()
    oquants = load_oquants_event_dates(tickers) if oquants is None else oquants.copy()

    orats = orats.assign(src_orats=True)
    oquants = oquants.assign(src_oquants=True)

    merged = orats.merge(oquants, on=["ticker", "event_date"], how="outer")
    merged["src_orats"] = merged["src_orats"].fillna(False).astype(bool)
    merged["src_oquants"] = merged["src_oquants"].fillna(False).astype(bool)
    merged["date_agree"] = merged["src_orats"] & merged["src_oquants"]
    merged["event_id"] = (
        merged["ticker"].astype(str) + "_" + merged["event_date"].dt.strftime("%Y-%m-%d")
    )
    cols = [
        "event_id",
        "ticker",
        "event_date",
        "session",
        "annc_tod",
        "src_orats",
        "src_oquants",
        "date_agree",
        "updated_at",
    ]
    for col in cols:
        if col not in merged:
            merged[col] = pd.NA
    return (
        merged[cols].sort_values(["ticker", "event_date"]).reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# date-change detection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class DateChange:
    ticker: str
    kind: str  # "moved" | "added" | "removed" | "session_changed"
    old: str | None
    new: str | None

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.ticker}: {self.kind} {self.old} → {self.new}"


def detect_date_changes(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    horizon_days: int = 45,
) -> list[DateChange]:
    """Flag calendar drift between two refreshes.

    Announcement dates move, and a stale date is a known loss source: it puts
    the entry on the wrong day and can leave a short leg outstanding through an
    event that has not happened yet. Only the forward window matters, so
    ``horizon_days`` bounds the comparison to events near enough to trade.

    A "moved" event is one where a ticker's next scheduled date changed; the
    heuristic pairs each ticker's earliest upcoming event in the two snapshots.
    """
    changes: list[DateChange] = []
    if previous.empty and current.empty:
        return changes

    def upcoming(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.assign(event_date=pd.to_datetime([]))
        d = df.copy()
        d["event_date"] = pd.to_datetime(d["event_date"])
        anchor = pd.Timestamp.today().normalize()
        return d[
            (d["event_date"] >= anchor)
            & (d["event_date"] <= anchor + pd.Timedelta(days=horizon_days))
        ]

    prev_up, cur_up = upcoming(previous), upcoming(current)
    prev_first = prev_up.sort_values("event_date").groupby("ticker").first()
    cur_first = cur_up.sort_values("event_date").groupby("ticker").first()

    for ticker in sorted(set(prev_first.index) | set(cur_first.index)):
        in_prev, in_cur = ticker in prev_first.index, ticker in cur_first.index
        if in_prev and not in_cur:
            changes.append(
                DateChange(ticker, "removed", str(prev_first.loc[ticker, "event_date"].date()), None)
            )
            continue
        if in_cur and not in_prev:
            changes.append(
                DateChange(ticker, "added", None, str(cur_first.loc[ticker, "event_date"].date()))
            )
            continue
        old_date = prev_first.loc[ticker, "event_date"]
        new_date = cur_first.loc[ticker, "event_date"]
        if old_date != new_date:
            changes.append(
                DateChange(ticker, "moved", str(old_date.date()), str(new_date.date()))
            )
            continue
        old_sess = prev_first.loc[ticker].get("session")
        new_sess = cur_first.loc[ticker].get("session")
        if pd.notna(old_sess) and pd.notna(new_sess) and old_sess != new_sess:
            changes.append(DateChange(ticker, "session_changed", str(old_sess), str(new_sess)))
    return changes
