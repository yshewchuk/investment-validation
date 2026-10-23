"""Pure supplied-calendar pre-print horizon; no history reads or gap filling."""
from __future__ import annotations

from bisect import bisect_right
from typing import Any

from engine.v2.foundation.market_calendar import (
    CalendarInputError,
    CalendarSessions,
    _as_day,
    _calendar_days,
    _field,
    _pre_anchor,
)

__all__ = ["CalendarCoverageError", "trading_days_before_print"]


class CalendarCoverageError(CalendarInputError):
    """Valid calendar inputs cannot cover the requested pre-print interval."""


def trading_days_before_print(calendar: CalendarSessions, *, entry_date: Any,
                              event_date: Any, session: str) -> int:
    """Signed supplied-session distance to the pre-print close; zero is valid.

    Entry uses the previous listed session, matching legacy index_of(prev)
    within coverage. Neither endpoint may lie outside the supplied calendar;
    holes inside observed history are never filled. Malformed inputs retain
    CalendarInputError; only insufficient date/anchor coverage is its distinct
    CalendarCoverageError subclass. No I/O, request cache, retry or partial result;
    validation reuses only bounded immutable annual holiday sets.
    """
    days = _calendar_days(_field(calendar, "days"))
    observed = _as_day(_field(calendar, "observed_through"), "observed_through")
    if observed not in days:
        raise CalendarInputError("calendar observed_through must be a listed session")
    entry = _as_day(entry_date, "entry_date")
    event = _as_day(event_date, "event_date")
    if session not in ("BMO", "AMC"):
        raise CalendarInputError("unknown pre-print session")
    if not days[0] <= entry <= days[-1] or not days[0] <= event <= days[-1]:
        raise CalendarCoverageError("calendar does not cover the pre-print interval")
    try:
        pre = _pre_anchor(event, session, days)
    except CalendarInputError as exc:
        raise CalendarCoverageError("calendar has no pre-print close") from exc
    return days.index(pre) - (bisect_right(days, entry) - 1)
