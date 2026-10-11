"""Event-history primitives (design #651): pure, no I/O, no clock.

Dates are canonical ``YYYY-MM-DD`` strings and are the only time basis;
provenance timestamps such as ``retrieved_at`` or ``computed_at`` are never
read. Every input is checked against a type allow-list and a malformed one
refuses with ``CONTRACT_MISMATCH``, never a bare exception.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np

from engine.v2.data.errors import DataError, fail

__all__ = ["SequencedEvent", "event_sequence", "AsofSeries", "strict_prior_mean"]

_ON_MISSING = ("null", "refuse")
_POPULATIONS = ("admitted", "all")
_FLOATS = (float, np.floating)
_NUMBERS = (int, float, np.integer, np.floating)


def _refuse(message: str, **details: Any) -> DataError:
    return fail("CONTRACT_MISMATCH", message, details=details)


def _present(value: Any) -> bool:
    """False for None and any float NaN (numpy dtypes included)."""
    if value is None:
        return False
    return not (isinstance(value, _FLOATS) and math.isnan(value))


def _number(value: Any) -> float:
    """Allow-list: int, float, numpy integer/floating (not bool); finite only."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, _NUMBERS):
        raise _refuse("a numeric value is required")
    try:
        number = float(value)
    except OverflowError:
        raise _refuse("a finite numeric value is required") from None
    if not math.isfinite(number):
        raise _refuse("a finite numeric value is required")
    return number


def _count(value: Any, name: str) -> int:
    """A non-negative integer declaration (not bool, not fractional, not NaN)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise _refuse(f"{name} must be a non-negative integer")
    return int(value)


def _choice(value: Any, allowed: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise _refuse(f"{name} must be one of {allowed}")
    return value


def _names(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise _refuse(f"{what} must be a sequence of column names")
    if not all(isinstance(item, str) for item in value):
        raise _refuse(f"{what} must be a sequence of column names")
    return tuple(value)


def _day(text: Any) -> date:
    try:
        parsed = date.fromisoformat(text)
    except (TypeError, ValueError):
        raise _refuse("dates must be ISO YYYY-MM-DD strings") from None
    if parsed.isoformat() != text:
        raise _refuse("dates must be canonical YYYY-MM-DD strings")
    return parsed


@dataclass(frozen=True)
class SequencedEvent:
    """Validated on construction: canonical date, non-negative int ordinal, bool."""

    event_date: str
    ordinal: int
    admitted: bool

    def __post_init__(self) -> None:
        _day(self.event_date)
        _count(self.ordinal, "ordinal")
        if not isinstance(self.admitted, bool):
            raise _refuse("admitted must be a bool")


def event_sequence(event_dates: Sequence[str], *, warmup: int,
                   first_known: str | None = None) -> tuple[SequencedEvent, ...]:
    """One ticker's FULL ordered event list; ordinal counts every earlier event."""
    warmup = _count(warmup, "warmup")
    if isinstance(event_dates, str) or not isinstance(event_dates, Sequence):
        raise _refuse("event_dates must be a sequence of dates")
    days = list(event_dates)
    for day in days:
        _day(day)
    if first_known is not None:
        _day(first_known)
    for prev, cur in zip(days, days[1:]):
        if cur <= prev:
            raise _refuse("event dates must be strictly ascending")
    if first_known is not None and (not days or days[0] != first_known):
        raise _refuse("event list does not start at the ticker's first known event")
    return tuple(SequencedEvent(day, i, i >= warmup) for i, day in enumerate(days))


class AsofSeries:
    """Last eligible row strictly before an event date; never reaches back past it."""

    def __init__(self, rows: Sequence[Mapping[str, Any]], *, eligible: Sequence[str],
                 value: str, on_missing: str, max_staleness_days: int | None = None) -> None:
        self._on_missing = _choice(on_missing, _ON_MISSING, "on_missing")
        eligible = _names(eligible, "eligible")
        if not isinstance(value, str):
            raise _refuse("value must be a column name")
        if max_staleness_days is not None:
            max_staleness_days = _count(max_staleness_days, "max_staleness_days")
        if isinstance(rows, (str, Mapping)) or not isinstance(rows, Sequence):
            raise _refuse("rows must be a sequence of mappings")
        self._max_staleness = max_staleness_days
        self._dates: list[str] = []
        self._values: list[Any] = []
        prev: str | None = None
        for row in rows:
            if not isinstance(row, Mapping):
                raise _refuse("each row must be a mapping")
            for column in ("date", *eligible, value):
                if column not in row:
                    raise _refuse("rows lack a required column", column=column)
            day = row["date"]
            _day(day)
            if prev is not None and day <= prev:
                raise _refuse("rows must be strictly date-ascending")
            prev = day
            if all(_present(row[c]) for c in eligible):
                self._dates.append(day)
                self._values.append(row[value])

    def at(self, event_date: str) -> float | None:
        event_day = _day(event_date)
        i = bisect_left(self._dates, event_date) - 1
        found = None
        if i >= 0:
            age = (event_day - _day(self._dates[i])).days
            if self._max_staleness is None or age <= self._max_staleness:
                found = self._values[i]
        if _present(found):
            return _number(found)
        if self._on_missing == "refuse":
            raise _refuse("no usable observation before the event")
        return None


def _check_ascending(events: Sequence[SequencedEvent]) -> None:
    for prev, cur in zip(events, events[1:]):
        if cur.event_date <= prev.event_date:
            raise _refuse("event dates must be strictly ascending in ordinal order")


def _check_aggregate_inputs(events: Any, values: Any, population: Any) -> None:
    _choice(population, _POPULATIONS, "population")
    if isinstance(events, str) or not isinstance(events, Sequence):
        raise _refuse("events must be a sequence of SequencedEvent")
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise _refuse("values must be a sequence")
    if not all(isinstance(ev, SequencedEvent) for ev in events):
        raise _refuse("events must be SequencedEvent items")
    if len(events) != len(values):
        raise _refuse("events and values differ in length")
    if any(ev.ordinal != i for i, ev in enumerate(events)):
        raise _refuse("events must be the full sequence in ordinal order")
    _check_ascending(events)


def strict_prior_mean(events: Sequence[SequencedEvent], values: Sequence[Any], *,
                      population: str) -> tuple[tuple[float | None, int], ...]:
    """Per event: mean and count of non-null values of earlier-ordinal events.

    The mean is a running mean (``mean += x / n - mean / n``) so a finite input
    set never overflows an intermediate sum.
    """
    _check_aggregate_inputs(events, values, population)
    out: list[tuple[float | None, int]] = []
    mean, count = 0.0, 0
    for ev, val in zip(events, values):
        counted = population == "all" or ev.admitted
        out.append((mean if count else None, count) if counted else (None, 0))
        if counted and _present(val):
            count += 1
            number = _number(val)
            mean += number / count - mean / count
    return tuple(out)