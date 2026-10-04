"""Rule-based trading sessions and BMO/AMC exit anchoring (layer 0.5): projected
weekdays minus NYSE holidays, plus the first session after ``event_through``."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol

__all__ = ["CalendarEventKey", "CalendarSessions", "CalendarInputError",
           "build_calendar_sessions", "planned_exit_date"]


class CalendarInputError(ValueError):
    """R1: every missing or invalid input is refused whole, never partially."""
    code = "INVALID_REQUEST"


class CalendarEventKey(Protocol):
    strategy: str
    event_date: Any
    session: str


@dataclass(frozen=True, slots=True)
class CalendarSessions:
    days: tuple[str, ...]
    observed_through: str


_MISSING = object()
_EXIT_PRE = frozenset({"STR-RUNUP"})
_EXIT_POST = frozenset({"STR-THRU", "DYN-SV", "TWIN-P", "TWIN-P5", "CND-PS",
                        "BFLY-P", "BFLY-P5", "RAMP7", "CTR5"})


def _as_day(value: Any, field: str) -> date:
    """The v2 naive convention: a date, a naive datetime, or canonical text."""
    if isinstance(value, datetime):  # datetime subclasses date; check it first
        if value.utcoffset() is not None:
            raise CalendarInputError(f"{field} must be naive: {value!r} carries a timezone")
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError:
            parsed = None
        if parsed is None or parsed.isoformat() != value:
            raise CalendarInputError(f"{field} {value!r} is not a canonical YYYY-MM-DD date")
        return parsed
    raise CalendarInputError(f"{field} must be a date, naive datetime, or YYYY-MM-DD string")


def _observed(day: date) -> date:
    """Saturday to the prior Friday, Sunday to the next Monday."""
    return day + timedelta(days={5: -1, 6: 1}.get(day.weekday(), 0))


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year, 12, 31) if month == 12 else date(year, month + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous computus); Good Friday is two days earlier."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _holidays(year: int) -> set[date]:
    """The legacy scheduled NYSE rule set, observations applied."""
    jan1 = date(year, 1, 1)
    out = {_nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
           _easter(year) - timedelta(days=2), _last_weekday(year, 5, 0),
           _observed(date(year, 7, 4)), _nth_weekday(year, 9, 0, 1),
           _nth_weekday(year, 11, 3, 4), _observed(date(year, 12, 25))}
    if jan1.weekday() != 5:  # a Saturday New Year takes no Friday observance
        out.add(_observed(jan1))
    if year >= 2022:  # Juneteenth became a market holiday in 2022
        out.add(_observed(date(year, 6, 19)))
    return out


def _rule_sessions(start: date, end: date) -> list[date]:
    """Projected sessions in ``(start, end]`` — the legacy projection convention."""
    if end <= start:
        return []
    holidays: set[date] = set()
    for year in range(start.year, end.year + 1):
        holidays |= _holidays(year)
    out: list[date] = []
    day = start + timedelta(days=1)
    while day <= end:
        if day.weekday() < 5 and day not in holidays:
            out.append(day)
        day += timedelta(days=1)
    return out


def _first_rule_session_after(day: date) -> date:
    following = _rule_sessions(day, day + timedelta(days=15))
    if not following:
        raise CalendarInputError(f"no projected session within 15 days after {day}")
    return following[0]


def build_calendar_sessions(observed_sessions: tuple[str, ...], *, event_through: Any) -> CalendarSessions:
    """R6: sorted unique observed sessions projected through ``event_through``;
    ``observed_through`` is the max source session, never a projection, and
    ``days`` appends the first projected session after it for a next close.
    """
    if not isinstance(observed_sessions, tuple) or not observed_sessions:
        raise CalendarInputError("observed_sessions must be a non-empty tuple of YYYY-MM-DD strings")
    days: set[date] = set()
    for entry in observed_sessions:
        if not isinstance(entry, str):
            raise CalendarInputError(f"observed session {entry!r} is not a YYYY-MM-DD string")
        day = _as_day(entry, "observed session")
        if day.weekday() >= 5:
            raise CalendarInputError(f"observed session {entry} is a weekend")
        if day in _holidays(day.year):
            raise CalendarInputError(f"observed session {entry} is a scheduled market holiday")
        days.add(day)
    through = _as_day(event_through, "event_through")
    observed_through = max(days)
    projected = _rule_sessions(observed_through, through)
    tail = _first_rule_session_after(through)
    all_days = sorted(days.union(projected).union((tail,)))
    return CalendarSessions(tuple(d.isoformat() for d in all_days), observed_through.isoformat())


def _field(obj: Any, name: str) -> Any:
    value = getattr(obj, name, _MISSING)
    if value is _MISSING:
        raise CalendarInputError(f"{type(obj).__name__} is missing structural field {name!r}")
    return value


def _pre_anchor(day: date, session: str, days: list[date]) -> date:
    """Last information-free close: the AMC event day itself, else the prior session."""
    if not days or day > days[-1]:
        raise CalendarInputError(f"no calendar coverage for {day} ({session})")
    if session == "AMC" and day in days:
        return day
    earlier = [d for d in days if d < day]
    if not earlier:
        raise CalendarInputError(f"no pre-print session for {day} ({session})")
    return earlier[-1]


def _post_anchor(day: date, session: str, days: list[date]) -> date:
    """First print-reflecting close: the BMO event day itself, else the next session."""
    if session == "BMO" and day in days:
        return day
    later = [d for d in days if d > day]
    if not later:
        raise CalendarInputError(f"no post-print session for {day} ({session})")
    return later[0]


def planned_exit_date(key: CalendarEventKey, calendar: CalendarSessions) -> str:
    """Fixed foundation exit policy: STR-RUNUP the last pre-print, else first post."""
    strategy = _field(key, "strategy")
    session = _field(key, "session")
    event_date = _field(key, "event_date")
    raw_days = _field(calendar, "days")
    if session != "BMO" and session != "AMC":
        raise CalendarInputError(f"unknown session {session!r}")
    if not isinstance(strategy, str) or (strategy not in _EXIT_PRE and strategy not in _EXIT_POST):
        raise CalendarInputError(f"unknown strategy {strategy!r}")
    if not isinstance(raw_days, tuple) or not raw_days:
        raise CalendarInputError("calendar days must be a non-empty tuple of YYYY-MM-DD strings")
    days = [_as_day(text, "calendar day") for text in raw_days]
    if days != sorted(set(days)):
        raise CalendarInputError("calendar days must be sorted and duplicate-free")
    if any(d.weekday() > 4 or d in _holidays(d.year) for d in days):
        raise CalendarInputError("calendar days must be weekday non-holiday sessions")
    day = _as_day(event_date, "event_date")
    if day < days[0]:
        raise CalendarInputError(f"no calendar coverage before {day} ({days[0]} is first session)")
    if strategy in _EXIT_PRE:
        return _pre_anchor(day, session, days).isoformat()
    return _post_anchor(day, session, days).isoformat()
