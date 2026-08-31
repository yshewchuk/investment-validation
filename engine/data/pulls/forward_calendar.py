"""Refresh the FORWARD earnings calendar — the thing ORATS cannot supply.

    python3 -m engine.data.pulls.forward_calendar --horizon 21

ORATS ``/hist/earnings`` is a history endpoint: its payloads stop at the last
print that already happened, so re-pulling it produces no upcoming events and
the monitoring board has nothing to score. This module is what fills the gap,
in two passes with different jobs:

**1. Nasdaq — who reports, and when.** One keyless call per DATE returns every
company reporting that day, so a three-week horizon costs ~15 trading-day calls
for the entire market. Roughly half of the forward rows also carry the session
(``time-pre-market`` / ``time-after-hours``), and that share rises as the print
approaches — 9% at 2–3 weeks out, 54% inside a week, 75% across the 1–10B and
>10B slices this program trades.

**2. yfinance — what time of day.** Per ticker at ~0.7–0.9s, so it runs only
for horizon names whose session is still unknown, ordered by how soon they
report. Its session is the checkable one: it keeps the announcement TIME on
historical rows, and agreed with ORATS ``anncTod`` on 99.72% of 716 overlapping
events. Nasdaq's cannot be graded after the fact at all, because it drops the
time once the date is past.

Both are unmetered, and both go through the Tier-1 fetch wrapper with
``live=True`` — one cache entry per source per day. So a second run on the same
day is free and idempotent, while tomorrow's run picks up every session that
firmed up overnight.

The session that survives into Tier 2 is chosen by
:data:`engine.calendar.SESSION_PRIORITY`, and every row records which source
gave it in ``session_src`` — so a forward guess is never confused with the
ORATS-confirmed article once the event is past.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

from engine.calendar import trading_calendar
from engine.data.sources.nasdaq import SESSION_BY_TIME

__all__ = ["refresh_forward_calendar", "CalendarRefresh", "horizon_dates"]

#: Per-run ceiling on the yfinance session-confirmation pass. It is a time
#: budget, not a quota: ~0.9s per ticker, so 400 is about six minutes. Names
#: left over are picked up on later nights, and they are by construction the
#: ones reporting furthest out.
MAX_SESSION_CONFIRMATIONS = 400


@dataclass
class CalendarRefresh:
    as_of: str
    horizon_days: int
    dates: list = field(default_factory=list)
    nasdaq_calls: int = 0
    nasdaq_cached: int = 0
    nasdaq_failed: list = field(default_factory=list)
    rows_seen: int = 0
    tickers_seen: int = 0
    session_from_nasdaq: int = 0
    session_missing: int = 0
    yfinance_calls: int = 0
    yfinance_cached: int = 0
    yfinance_resolved: int = 0
    truncated: bool = False
    rebuilt: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def horizon_dates(as_of, horizon_days: int) -> list[pd.Timestamp]:
    """Trading days in ``[as_of, as_of + horizon_days]``.

    Uses the program's trading calendar (which projects past its observed tail),
    so a holiday does not cost a call that can only ever return zero rows.
    """
    as_of = pd.Timestamp(as_of).normalize()
    end = as_of + pd.Timedelta(days=horizon_days)
    try:
        days = [d for d in trading_calendar().days if as_of <= d <= end]
    except Exception:  # a missing calendar must not stop the refresh
        days = []
    if not days:
        days = [d for d in pd.date_range(as_of, end, freq="D") if d.weekday() < 5]
    return list(days)


def _nasdaq_rows(record) -> list[dict]:
    try:
        payload = record.json()
    except (ValueError, AttributeError):
        return []
    return ((payload or {}).get("data") or {}).get("rows") or []


def refresh_forward_calendar(
    as_of=None,
    *,
    horizon_days: int = 21,
    fetcher=None,
    tickers: Iterable[str] | None = None,
    confirm_sessions: bool = True,
    max_confirmations: int = MAX_SESSION_CONFIRMATIONS,
    rebuild_events: bool = True,
) -> CalendarRefresh:
    """Pull the forward calendar and rebuild Tier-2 ``earnings_events``."""
    from engine.data.fetch import Fetcher

    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    fetcher = fetcher or Fetcher()
    wanted = set(tickers) if tickers is not None else None

    out = CalendarRefresh(as_of=str(as_of.date()), horizon_days=horizon_days)
    dates = horizon_dates(as_of, horizon_days)
    out.dates = [str(d.date()) for d in dates]

    # -- pass 1: Nasdaq, one call per date, whole market --------------------
    seen: dict[tuple[str, str], str | None] = {}
    for day in dates:
        try:
            record = fetcher.fetch(
                "nasdaq", "calendar/earnings", {"date": str(day.date())},
                live=True, note="forward calendar",
            )
        except Exception as exc:
            # One bad date must not cost the other twenty. The calendar is
            # additive: a date that fails today is re-tried tomorrow.
            out.nasdaq_failed.append({"date": str(day.date()),
                                      "error": f"{type(exc).__name__}: {exc}"[:200]})
            continue
        out.nasdaq_cached += 1 if record.from_cache else 0
        out.nasdaq_calls += 0 if record.from_cache else 1
        for row in _nasdaq_rows(record):
            ticker = str(row.get("symbol") or "").strip()
            if not ticker or (wanted is not None and ticker not in wanted):
                continue
            seen[(ticker, str(day.date()))] = SESSION_BY_TIME.get(row.get("time"))

    out.rows_seen = len(seen)
    out.tickers_seen = len({t for t, _ in seen})
    out.session_from_nasdaq = sum(1 for v in seen.values() if v)
    out.session_missing = out.rows_seen - out.session_from_nasdaq

    # -- pass 2: yfinance, only where the session is still unknown ----------
    if confirm_sessions and out.session_missing:
        from engine.calendar import load_yfinance_earnings

        known = load_yfinance_earnings()
        already = set()
        if len(known):
            already = {
                (str(r.ticker), str(pd.Timestamp(r.event_date).date()))
                for r in known[known["session"].notna()].itertuples(index=False)
            }
        # Soonest first: an event reporting on Tuesday needs its session now,
        # one three weeks out can wait for a night when Nasdaq has firmed up.
        pending = sorted(
            (date, ticker)
            for (ticker, date), session in seen.items()
            if not session and (ticker, date) not in already
        )
        out.truncated = len(pending) > max_confirmations
        for _, ticker in pending[:max_confirmations]:
            try:
                record = fetcher.fetch(
                    "yfinance", "earnings", {"ticker": ticker},
                    live=True, note="forward calendar session",
                )
            except Exception:
                continue  # a name yfinance cannot answer for stays session-less
            out.yfinance_cached += 1 if record.from_cache else 0
            out.yfinance_calls += 0 if record.from_cache else 1
            if record.meta.get("status") == 200:
                out.yfinance_resolved += 1

    # -- Tier 2 -------------------------------------------------------------
    if rebuild_events:
        from engine.data.rebuild import rebuild

        result = rebuild(tables=("events",))
        out.rebuilt = {"snapshot": result.snapshot, "elapsed_s": round(result.elapsed_s, 1)}
    return out


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--horizon", type=int, default=21)
    parser.add_argument("--tickers", default=None, help="comma-separated restriction")
    parser.add_argument("--no-sessions", action="store_true",
                        help="skip the yfinance session-confirmation pass")
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--max-confirmations", type=int, default=MAX_SESSION_CONFIRMATIONS)
    args = parser.parse_args(argv)

    result = refresh_forward_calendar(
        args.as_of,
        horizon_days=args.horizon,
        tickers=[t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None,
        confirm_sessions=not args.no_sessions,
        max_confirmations=args.max_confirmations,
        rebuild_events=not args.no_rebuild,
    )
    print(json.dumps(result.as_dict(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
