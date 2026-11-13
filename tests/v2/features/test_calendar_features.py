from __future__ import annotations

from datetime import date

import pytest

from engine.v2.data.errors import DataError
from engine.v2.features.calendar_features import calendar_features


def test_amc_friday_with_prior_week():
    result = calendar_features("AMC", "2025-03-14", "2025-03-07")

    assert result == {
        "cal_amc": 1,
        "cal_weekday": 4,
        "cal_month": 3,
        "cal_days_since_prior": 7,
    }


def test_bmo_monday_without_prior():
    result = calendar_features("BMO", "2025-12-01")

    assert result["cal_amc"] == 0
    assert result["cal_weekday"] == 0
    assert result["cal_month"] == 12
    assert result["cal_days_since_prior"] is None


def test_same_day_prior_is_zero_days():
    result = calendar_features("AMC", "2025-03-14", "2025-03-14")

    assert result["cal_days_since_prior"] == 0


def test_year_boundary_counts_calendar_days():
    result = calendar_features("BMO", "2026-01-02", "2025-12-31")

    assert result["cal_days_since_prior"] == 2


def test_result_has_exact_columns_and_int_values():
    result = calendar_features("AMC", "2025-03-14", "2025-03-07")

    assert set(result) == {
        "cal_amc",
        "cal_weekday",
        "cal_month",
        "cal_days_since_prior",
    }
    for value in result.values():
        assert value is None or type(value) is int


@pytest.mark.parametrize(
    ("session", "event_date", "prior_event_date", "message"),
    [
        ("amc", "2025-03-14", None, "session must be AMC or BMO"),
        ("DMH", "2025-03-14", None, "session must be AMC or BMO"),
        (None, "2025-03-14", None, "session must be AMC or BMO"),
        (1, "2025-03-14", None, "session must be AMC or BMO"),
        ("AMC", "2025-3-14", None, "event_date must be a canonical ISO date"),
        ("AMC", "20250314", None, "event_date must be a canonical ISO date"),
        ("AMC", None, None, "event_date must be a canonical ISO date"),
        ("AMC", date(2025, 3, 14), None, "event_date must be a canonical ISO date"),
        ("AMC", "2025-03-14", "bad", "prior_event_date must be None or a canonical ISO date"),
        ("AMC", "2025-03-14", "2025-3-7", "prior_event_date must be None or a canonical ISO date"),
        ("AMC", "2025-03-14", "2025-03-15", "prior_event_date is after event_date"),
    ],
    ids=[
        "lowercase-session",
        "unknown-session",
        "none-session",
        "non-string-session",
        "non-canonical-event-date",
        "compact-event-date",
        "none-event-date",
        "date-object-event-date",
        "bad-prior-date",
        "non-canonical-prior-date",
        "prior-after-event",
    ],
)
def test_invalid_inputs_are_refused(session, event_date, prior_event_date, message):
    with pytest.raises(DataError) as raised:
        calendar_features(session, event_date, prior_event_date)

    assert raised.value.code == "CONTRACT_MISMATCH"
    assert message in str(raised.value)
