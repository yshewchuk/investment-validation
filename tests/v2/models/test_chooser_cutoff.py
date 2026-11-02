from datetime import date, datetime, timezone

import pytest

from engine.v2.models.chooser_cutoff import (
    canonical_day,
    chooser_fold_start,
    require_causal_cutoff,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (date(2026, 9, 1), "2026-09-01"),
        (datetime(2026, 9, 1, 12, 30, 45), "2026-09-01"),
        ("2026-09-01", "2026-09-01"),
        ("2026-09-01T12:30:45", "2026-09-01"),
        ("2026-09-01 12:30:45", "2026-09-01"),
    ],
)
def test_canonical_day_accepts_canonical_values(value, expected):
    assert canonical_day(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 9, 1, tzinfo=timezone.utc),
        "2026-09-01T00:00:00+00:00",
        "2026-9-1",
        "2026-09-01T12:30:45.000",
        "2026-W36-2",
        "2026-09-01t12:30:45",
        20260901,
        None,
    ],
    ids=["aware-datetime", "aware-string", "short-date", "fractional-second",
         "week-date", "lowercase-separator", "integer", "none"],
)
def test_canonical_day_refuses_noncanonical_values(value):
    with pytest.raises(ValueError, match="^invalid chooser cutoff$") as exc:
        canonical_day(value)
    assert str(exc.value) == "invalid chooser cutoff"


@pytest.mark.parametrize(
    "training, fold, decision",
    [
        (20260901, "2026-10-01", "2026-11-01"),
        ("2026-09-01", 20261001, "2026-11-01"),
        ("2026-09-01", "2026-10-01", None),
        (date(2026, 9, 1), "2026-10-01", "2026-11-01"),
        ("2026-09-01", date(2026, 10, 1), "2026-11-01"),
        ("2026-09-01", "2026-10-01", date(2026, 11, 1)),
        ("2026-9-1", "2026-10-01", "2026-11-01"),
        ("2026-09-01", "2026-10-1", "2026-11-01"),
        ("2026-09-01", "2026-10-01", "2026-11-1"),
    ],
    ids=["training-not-string", "fold-not-string", "decision-not-string",
         "training-date-object", "fold-date-object", "decision-date-object",
         "training-noncanonical", "fold-noncanonical", "decision-noncanonical"],
)
def test_causal_cutoff_requires_three_canonical_date_strings(training, fold, decision):
    with pytest.raises(ValueError, match="^invalid chooser cutoff$"):
        require_causal_cutoff(training, fold, decision)


@pytest.mark.parametrize(
    "training, fold, decision, valid",
    [
        ("2026-09-10", "2026-09-10", "2026-10-01", False),
        ("2026-09-09", "2026-09-10", "2026-10-01", True),
        ("2026-10-01", "2026-11-01", "2026-10-01", False),
        ("2026-09-30", "2026-11-01", "2026-10-01", True),
    ],
    ids=["equal-fold", "day-before-fold", "equal-decision", "day-before-decision"],
)
def test_causal_cutoff_strict_boundaries(training, fold, decision, valid):
    if valid:
        assert require_causal_cutoff(training, fold, decision) is None
    else:
        with pytest.raises(ValueError, match="^invalid chooser cutoff$"):
            require_causal_cutoff(training, fold, decision)


@pytest.mark.parametrize(
    "event, decision, expected",
    [
        ("2026-05-20", "2026-06-10", "2026-05-01"),
        ("2026-06-20", "2026-06-10", "2026-06-01"),
        ("2026-07-20", "2026-06-10", "2026-06-01"),
        ("2025-12-31", "2026-01-10", "2025-12-01"),
    ],
    ids=["event-month-before", "same-month", "event-month-after", "year-boundary"],
)
def test_chooser_fold_start_uses_earlier_month(event, decision, expected):
    assert chooser_fold_start(event, decision) == expected


def test_repeated_inputs_return_stable_outputs():
    assert canonical_day("2026-09-01T12:30:45") == canonical_day("2026-09-01T12:30:45")
    assert chooser_fold_start("2026-05-20", "2026-06-10") == chooser_fold_start(
        "2026-05-20", "2026-06-10"
    )
    assert require_causal_cutoff("2026-09-01", "2026-10-01", "2026-11-01") is None
