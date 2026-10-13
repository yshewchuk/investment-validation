"""Synthetic pre-print session intervals, independent expected counts."""
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from engine.v2.foundation.market_calendar import (
    CalendarInputError,
    CalendarSessions,
    build_calendar_sessions,
)
from engine.v2.foundation.market_horizon import CalendarCoverageError, trading_days_before_print


def _calendar():
    # MLK Day (Jan 15) is not a session; Jan 6/7 and 13/14 are weekends.
    return build_calendar_sessions(("2024-01-04", "2024-01-05"),
                                   event_through="2024-01-16")


@pytest.mark.parametrize("entry,event,session,expected", [
    ("2024-01-05", "2024-01-16", "BMO", 5),
    ("2024-01-05", "2024-01-16", "AMC", 6),
    ("2024-01-11", "2024-01-16", "BMO", 1),
    ("2024-01-12", "2024-01-16", "BMO", 0),
    ("2024-01-16", "2024-01-16", "AMC", 0),
    ("2024-01-16", "2024-01-16", "BMO", -1),
    ("2024-01-06", "2024-01-16", "BMO", 5),
    ("2024-01-05", "2024-01-13", "AMC", 5),
])
def test_counts_supplied_sessions_not_calendar_days(entry, event, session, expected):
    calendar = _calendar()
    assert trading_days_before_print(
        calendar, entry_date=entry, event_date=event, session=session) == expected
    assert calendar.observed_through == "2024-01-05"


def test_historical_gaps_are_not_synthesized_and_inputs_are_unchanged():
    calendar = CalendarSessions(("2024-01-05", "2024-01-10", "2024-01-16"), "2024-01-16")
    original = replace(calendar)
    for _ in range(2):
        assert trading_days_before_print(
            calendar, entry_date="2024-01-05", event_date="2024-01-16", session="BMO") == 1
    assert calendar == original


@pytest.mark.parametrize("changes", [
    {"entry_date": "2024-01-03"}, {"entry_date": "2024-01-18"},
    {"event_date": "2024-01-03"}, {"event_date": "2024-01-18"},
    {"event_date": "2024-01-04"},
])
def test_missing_bounds_or_pre_print_anchor_is_distinct_coverage_refusal(changes):
    kwargs = dict(entry_date="2024-01-05", event_date="2024-01-16", session="BMO")
    kwargs.update(changes)
    with pytest.raises(CalendarCoverageError) as error:
        trading_days_before_print(_calendar(), **kwargs)
    assert error.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("calendar", [
    CalendarSessions((), "2024-01-05"),
    CalendarSessions(("2024-01-05", "2024-01-04"), "2024-01-05"),
    CalendarSessions(("2024-01-05", "2024-01-05"), "2024-01-05"),
    CalendarSessions(("2024-01-05", "2024-01-06"), "2024-01-05"),
    CalendarSessions(("2024-01-05", "2024-01-15"), "2024-01-05"),
    CalendarSessions(("2024-01-05", "2024-01-16"), "bad"),
    CalendarSessions(("2024-01-05", "2024-01-16"), "2024-01-12"),
])
def test_malformed_calendar_is_not_retyped_as_coverage(calendar):
    with pytest.raises(CalendarInputError) as error:
        trading_days_before_print(
            calendar, entry_date="2024-01-03", event_date="2024-01-18", session="BMO")
    assert type(error.value) is CalendarInputError


@pytest.mark.parametrize("field,value", [
    ("entry_date", None), ("entry_date", True), ("entry_date", 20240105),
    ("entry_date", "2024-01-05T00:00:00Z"),
    ("event_date", datetime(2024, 1, 16, tzinfo=timezone.utc)),
    ("event_date", "bad"), ("session", "unknown"),
])
def test_invalid_arguments_keep_input_refusal(field, value):
    kwargs = dict(entry_date="2024-01-05", event_date="2024-01-16", session="BMO")
    kwargs[field] = value
    with pytest.raises(CalendarInputError) as error:
        trading_days_before_print(_calendar(), **kwargs)
    assert type(error.value) is CalendarInputError
