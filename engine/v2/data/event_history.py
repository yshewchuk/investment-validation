"""Event-history primitives (design #651): pure, no I/O, no clock.

Dates are ISO ``YYYY-MM-DD`` strings and are the only time basis; provenance
timestamps such as ``retrieved_at`` or ``computed_at`` are never read.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from engine.v2.data.errors import DataError, fail

__all__ = ["SequencedEvent", "event_sequence", "AsofSeries", "strict_prior_mean"]

_ON_MISSING = ("null", "refuse")
_POPULATIONS = ("admitted", "all")


def _refuse(message: str, **details: Any) -> DataError:
    return fail("CONTRACT_MISMATCH", message, details=details)


def _present(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _day(text: Any) -> date:
    try:
        return date.fromisoformat(text)
    except (TypeError, ValueError):
        raise _refuse("dates must be ISO YYYY-MM-DD strings") from None


@dataclass(frozen=True)
class SequencedEvent:
    event_date: str
    ordinal: int
    admitted: bool


def event_sequence(event_dates: Sequence[str], *, warmup: int,
                   first_known: str | None = None) -> tuple[SequencedEvent, ...]:
    """One ticker's FULL ordered event list; ordinal counts every earlier event."""
    if warmup < 0:
        raise _refuse("warmup must be non-negative")
    days = list(event_dates)
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
        if on_missing not in _ON_MISSING:
            raise _refuse("on_missing must be 'null' or 'refuse'")
        self._on_missing = on_missing
        self._max_staleness = max_staleness_days
        self._dates: list[str] = []
        self._values: list[Any] = []
        prev: str | None = None
        for row in rows:
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
            try:
                return float(found)
            except (TypeError, ValueError):
                raise _refuse("anchor value is not numeric") from None
        if self._on_missing == "refuse":
            raise _refuse("no usable observation before the event")
        return None


def strict_prior_mean(events: Sequence[SequencedEvent], values: Sequence[Any], *,
                      population: str) -> tuple[tuple[float | None, int], ...]:
    """Per event: mean and count of non-null values of earlier-ordinal events."""
    if population not in _POPULATIONS:
        raise _refuse("population must be 'admitted' or 'all'")
    if len(events) != len(values):
        raise _refuse("events and values differ in length")
    if any(ev.ordinal != i for i, ev in enumerate(events)):
        raise _refuse("events must be the full sequence in ordinal order")
    out: list[tuple[float | None, int]] = []
    total, count = 0.0, 0
    for ev, val in zip(events, values):
        counted = population == "all" or ev.admitted
        out.append((total / count if count else None, count) if counted else (None, 0))
        if counted and _present(val):
            total += float(val)
            count += 1
    return tuple(out)