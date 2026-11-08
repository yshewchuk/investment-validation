from __future__ import annotations

from datetime import date, timedelta

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research.season_heat import compute_season_heat


def _weekday_sessions(start: date, end: date) -> list[str]:
    sessions: list[str] = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


def _session_sequence(
    start: date, count: int, holiday: date | None = None
) -> list[str]:
    sessions: list[str] = []
    current = start
    while len(sessions) < count:
        if current.weekday() < 5 and current != holiday:
            sessions.append(current.isoformat())
        current += timedelta(days=1)
    return sessions


def _label(
    event_id: str,
    ticker: str,
    event_date: str,
    move: float,
    *,
    available_as_of_date: str | None = None,
) -> dict:
    if available_as_of_date is None:
        available_as_of_date = (
            date.fromisoformat(event_date) + timedelta(days=1)
        ).isoformat()
    return {
        "event_id": event_id,
        "ticker": ticker,
        "event_date": event_date,
        "available_as_of_date": available_as_of_date,
        "abs_move": move,
    }


def test_amc_d_label_is_unavailable_for_bmo_and_amc_d_plus_one():
    sessions = _weekday_sessions(date(2025, 1, 2), date(2025, 1, 31))
    anchors = sessions[:8]
    labels = [
        _label(
            f"E{index:02d}",
            f"L{index:02d}",
            anchors[index % len(anchors)],
            index + 1,
        )
        for index in range(20)
    ]
    labels.append(
        _label(
            "AMC-D",
            "AMC",
            "2025-01-15",
            100,
            available_as_of_date="2025-01-17",
        )
    )
    targets = [
        {
            "event_id": "BMO-D+1",
            "ticker": "BMO",
            "event_date": "2025-01-16",
            "decision_session": "2025-01-14",
        },
        {
            "event_id": "AMC-D+1",
            "ticker": "AMC",
            "event_date": "2025-01-16",
            "decision_session": "2025-01-15",
        },
    ]

    results = compute_season_heat(labels, targets, sessions)
    bmo = results[("BMO-D+1", "2025-01-14")]
    amc = results[("AMC-D+1", "2025-01-15")]

    assert bmo["season_heat_count_20"] == 20
    assert amc["season_heat_count_20"] == 20
    assert bmo["season_heat_mean_abs_20"] == pytest.approx(10.5)
    assert amc["season_heat_mean_abs_20"] == pytest.approx(10.5)
    assert bmo == amc


def test_weekend_availability_waits_for_next_decision_session():
    sessions = ["2025-03-07", "2025-03-10", "2025-03-11"]
    labels = [
        _label(
            "L1",
            "CCC",
            "2025-03-07",
            5,
            available_as_of_date="2025-03-09",
        )
    ]
    targets = [
        {
            "event_id": "FRI",
            "ticker": "AAA",
            "event_date": "2025-03-10",
            "decision_session": "2025-03-07",
        },
        {
            "event_id": "MON",
            "ticker": "BBB",
            "event_date": "2025-03-11",
            "decision_session": "2025-03-10",
        },
    ]

    results = compute_season_heat(labels, targets, sessions)
    friday = results[("FRI", "2025-03-07")]
    monday = results[("MON", "2025-03-10")]

    assert friday["season_heat_count_20"] == 0
    assert monday["season_heat_count_20"] == 1
    for result in (friday, monday):
        assert result["season_heat_mean_abs_5"] is None
        assert result["season_heat_mean_abs_20"] is None
        assert result["season_heat_mean_abs_60"] is None
        assert result["season_heat_ratio_20_252"] is None


def test_target_event_and_same_ticker_events_are_excluded():
    sessions = _weekday_sessions(date(2025, 1, 6), date(2025, 2, 7))
    assert len(sessions) == 25
    decision = sessions[20]
    labels = []
    for index in range(20):
        position = 1 + index % 19
        labels.append(
            _label(f"E{index:02d}", f"T{index:02d}", sessions[position], index + 1)
        )
    labels.append(_label("TARGET", "ZZZ", sessions[2], 500))
    labels.append(_label("OTHER", "AAA", sessions[3], 600))
    targets = [
        {
            "event_id": "TARGET",
            "ticker": "AAA",
            "event_date": sessions[21],
            "decision_session": decision,
        }
    ]

    result = compute_season_heat(labels, targets, sessions)[("TARGET", decision)]

    assert result["season_heat_count_20"] == 20
    assert result["season_heat_mean_abs_20"] == pytest.approx(10.5)


def test_all_window_warmups_counts_means_and_ratio():
    sessions = _session_sequence(date(2024, 1, 1), 252, date(2024, 2, 19))
    decision = sessions[-1]

    labels = []
    serial = 0
    for event_date in sessions[:-1]:
        labels.append(
            _label(f"E{serial:03d}", f"T{serial:03d}", event_date, serial + 1)
        )
        serial += 1
    for event_date in sessions[-5:-1]:
        for _ in range(4):
            labels.append(
                _label(f"E{serial:03d}", f"T{serial:03d}", event_date, serial + 1)
            )
            serial += 1
    assert serial == 267

    prior_four = set(sessions[-5:-1])
    prior_nineteen = set(sessions[-20:-1])
    prior_fifty_nine = set(sessions[-60:-1])

    def mean_for(dates: set[str]) -> float:
        moves = [
            label["abs_move"]
            for label in labels
            if label["event_date"] in dates
        ]
        return sum(moves) / len(moves)

    expected_mean_5 = mean_for(prior_four)
    expected_mean_20 = mean_for(prior_nineteen)
    expected_mean_60 = mean_for(prior_fifty_nine)
    expected_mean_252 = sum(range(1, 268)) / 267

    targets = [
        {
            "event_id": "A",
            "ticker": "AAA",
            "event_date": decision,
            "decision_session": decision,
        },
        {
            "event_id": "B",
            "ticker": "BBB",
            "event_date": decision,
            "decision_session": decision,
        },
    ]

    shuffled_labels = labels[::2] + labels[1::2]
    shuffled_targets = targets[::-1]
    results = compute_season_heat(shuffled_labels, shuffled_targets, sessions)
    first = results[("A", decision)]
    second = results[("B", decision)]

    for result in (first, second):
        assert result["season_heat_count_20"] == 35
        for key, expected in (
            ("season_heat_mean_abs_5", expected_mean_5),
            ("season_heat_mean_abs_20", expected_mean_20),
            ("season_heat_mean_abs_60", expected_mean_60),
            (
                "season_heat_ratio_20_252",
                expected_mean_20 / expected_mean_252,
            ),
        ):
            value = result[key]
            assert value is not None
            assert value == value
            assert value == pytest.approx(expected)
    assert first == second

    small_labels = [
        _label(f"S{index:02d}", f"S{index:02d}", event_date, index + 1)
        for index, event_date in enumerate(sessions[-20:-1])
    ]
    small_targets = [
        {
            "event_id": "S",
            "ticker": "AAA",
            "event_date": decision,
            "decision_session": decision,
        }
    ]
    small = compute_season_heat(small_labels, small_targets, sessions)[
        ("S", decision)
    ]

    assert small["season_heat_count_20"] == 19
    for key in (
        "season_heat_mean_abs_5",
        "season_heat_mean_abs_20",
        "season_heat_mean_abs_60",
        "season_heat_ratio_20_252",
    ):
        assert small[key] is None


def _valid_inputs():
    labels = [_label("L1", "CCC", "2025-03-07", 1)]
    targets = [
        {
            "event_id": "T1",
            "ticker": "AAA",
            "event_date": "2025-03-11",
            "decision_session": "2025-03-10",
        }
    ]
    sessions = ["2025-03-07", "2025-03-10", "2025-03-11"]
    return labels, targets, sessions


def _duplicate_target():
    labels, targets, sessions = _valid_inputs()
    targets.append(dict(targets[0]))
    return labels, targets, sessions


def _non_session_decision():
    labels, targets, sessions = _valid_inputs()
    targets[0]["decision_session"] = "2025-03-08"
    return labels, targets, sessions


def _availability_on_event_date():
    labels, targets, sessions = _valid_inputs()
    labels[0]["available_as_of_date"] = labels[0]["event_date"]
    return labels, targets, sessions


def _nan_move():
    labels, targets, sessions = _valid_inputs()
    labels[0]["abs_move"] = float("nan")
    return labels, targets, sessions


def _negative_move():
    labels, targets, sessions = _valid_inputs()
    labels[0]["abs_move"] = -1
    return labels, targets, sessions


@pytest.mark.parametrize(
    ("build", "expected_code"),
    [
        (_duplicate_target, "CONTRACT_MISMATCH"),
        (_non_session_decision, "CONTRACT_MISMATCH"),
        (_availability_on_event_date, "EXPERIMENT_VARIANT_FAILED"),
        (_nan_move, "EXPERIMENT_VARIANT_FAILED"),
        (_negative_move, "EXPERIMENT_VARIANT_FAILED"),
    ],
    ids=[
        "duplicate-target",
        "non-session-decision",
        "availability-not-after-event",
        "nan-move",
        "negative-move",
    ],
)
def test_season_heat_rejects_invalid_inputs_with_native_codes(
    build, expected_code
):
    labels, targets, sessions = build()
    with pytest.raises(DataError) as raised:
        compute_season_heat(labels, targets, sessions)
    assert raised.value.code == expected_code
