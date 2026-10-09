"""Golden shared membership checks independent of storage and arrival order."""
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timezone

import pytest

from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts


@pytest.mark.parametrize("event_id,expected", [("EVENT-0", False), ("EVENT-1", False),
    ("EVENT-2", False), ("EVENT-5", True), ("EVENT-30", True), ("EVENT-56", True)])
def test_random_membership_golden_vectors_and_month_independence(event_id, expected):
    assert ExperimentHoldouts("2024-10").random_membership(event_id) is expected
    assert ExperimentHoldouts("2025-05").random_membership(event_id) is expected


@pytest.mark.parametrize("month", [None, "", "2024-1", "2024-00", "2024-13", "0000-01",
                                       "2024-10-01", " 2024-10", 202410])
def test_as_of_month_is_explicit_and_strict(month):
    with pytest.raises(ValueError):
        ExperimentHoldouts(month)


def test_context_is_immutable():
    policy = ExperimentHoldouts("2024-10")
    with pytest.raises(FrozenInstanceError):
        policy.as_of_month = "2025-05"


@pytest.mark.parametrize("event_id", [None, "", " ", " EVENT-5", "EVENT-5 ", 5])
def test_random_identity_is_not_silently_normalized(event_id):
    with pytest.raises(ValueError):
        ExperimentHoldouts.random_membership(event_id)


@pytest.mark.parametrize("day,expected", [
    ("2024-04-30", frozenset()),
    ("2024-05-01", frozenset({"rolling"})),
    ("2024-10-31", frozenset({"rolling"})),
    ("2024-11-01", frozenset({"ambiguous"})),
])
def test_rolling_includes_as_of_month_and_exactly_five_preceding_months(day, expected):
    assert ExperimentHoldouts("2024-10").classify("EVENT-0", day) == expected


@pytest.mark.parametrize("month,day,expected", [
    ("2024-02", "2023-08-31", frozenset()),
    ("2024-02", "2023-09-01", frozenset({"rolling"})),
    ("2024-02", "2024-02-29", frozenset({"rolling"})),
    ("2025-01", "2024-07-31", frozenset()),
    ("2025-01", "2024-08-01", frozenset({"rolling"})),
])
def test_calendar_months_cross_year_and_leap_boundaries(month, day, expected):
    assert ExperimentHoldouts(month).classify("EVENT-0", day) == expected


def test_random_overlap_survives_monthly_release_and_order_changes():
    before, after = ExperimentHoldouts("2024-10"), ExperimentHoldouts("2024-11")
    assert before.classify("EVENT-5", "2024-05-15") == {"random", "rolling"}
    assert after.classify("EVENT-5", "2024-05-15") == {"random"}
    assert before.classify("EVENT-0", "2024-05-15") == {"rolling"}
    assert after.classify("EVENT-0", "2024-05-15") == set()
    keys = ["EVENT-0", "EVENT-5", "EVENT-30"]
    forward = {key: before.classify(key, "2024-05-15") for key in keys}
    assert {key: before.classify(key, "2024-05-15") for key in reversed(keys)} == forward


@pytest.mark.parametrize("day", [None, "bad", "20240501", "2024-02-30", 20240501,
    datetime(2024, 1, 15, 0, 0, 1), datetime(2024, 1, 15, tzinfo=timezone.utc)])
def test_ambiguous_dates_are_not_released(day):
    assert ExperimentHoldouts("2024-10").classify("EVENT-0", day) == {"ambiguous"}


def test_plain_dates_and_naive_midnight_datetimes_agree():
    policy = ExperimentHoldouts("2024-10")
    assert policy.classify("EVENT-0", date(2024, 5, 1)) == {"rolling"}
    assert policy.classify("EVENT-0", datetime(2024, 5, 1)) == {"rolling"}
