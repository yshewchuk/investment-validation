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
            sessions[5 + index % 3],
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
    assert bmo["season_heat_mean_abs_5"] == pytest.approx(10.5)
    assert amc["season_heat_mean_abs_5"] == pytest.approx(10.5)
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


def test_off_session_target_event_date_is_accepted():
    sessions = ["2025-03-07", "2025-03-10", "2025-03-11"]
    targets = [
        {
            "event_id": "WEEKEND",
            "ticker": "AAA",
            "event_date": "2025-03-08",
            "decision_session": "2025-03-07",
        }
    ]

    results = compute_season_heat([], targets, sessions)
    result = results[("WEEKEND", "2025-03-07")]

    assert result["season_heat_count_20"] == 0
    assert result["season_heat_mean_abs_5"] is None
    assert result["season_heat_mean_abs_20"] is None
    assert result["season_heat_mean_abs_60"] is None
    assert result["season_heat_ratio_20_252"] is None


def test_prior_session_label_available_after_decision_is_excluded():
    decision = "2025-01-15"
    sessions = _weekday_sessions(date(2025, 1, 13), date(2025, 1, 16))
    labels = [
        _label(
            f"E{index:02d}",
            f"T{index:02d}",
            "2025-01-14",
            index + 1,
            available_as_of_date=decision,
        )
        for index in range(20)
    ]
    labels.append(
        _label(
            "LATE-OTHER",
            "OTHER",
            "2025-01-14",
            100,
            available_as_of_date="2025-01-16",
        )
    )
    targets = [
        {
            "event_id": "TARGET",
            "ticker": "TARGET",
            "event_date": "2025-01-16",
            "decision_session": decision,
        }
    ]

    result = compute_season_heat(labels, targets, sessions)[("TARGET", decision)]

    assert result["season_heat_count_20"] == 20
    assert result["season_heat_mean_abs_20"] == pytest.approx(10.5)


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

    assert result["season_heat_mean_abs_5"] is None
    assert result["season_heat_count_20"] == 20
    assert result["season_heat_mean_abs_20"] == pytest.approx(10.5)
    assert result["season_heat_mean_abs_60"] == pytest.approx(10.5)
    assert result["season_heat_ratio_20_252"] == pytest.approx(1.0)


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


def test_rolling_windows_match_bruteforce_across_decisions():
    sessions = _session_sequence(date(2024, 1, 1), 300)
    position = {session: index for index, session in enumerate(sessions)}
    labels = [
        _label(f"E{i}-{k}", f"T{(i + k) % 7}", session, (i * 7 + k * 3) % 41 + 1)
        for i, session in enumerate(sessions[:-1])
        for k in range(8)
    ]
    labels.append(_label("LATE", "LATE", sessions[0], 41, available_as_of_date=sessions[47]))
    targets = [
        {"event_id": f"X{j}", "ticker": f"T{j % 7}", "event_date": day, "decision_session": day}
        for j, day in enumerate(sessions[30::17])
    ]
    results = compute_season_heat(labels, targets, sessions)
    for target in targets:
        decision = target["decision_session"]
        result = results[(target["event_id"], decision)]
        means = {}
        for window, key in ((5, "season_heat_mean_abs_5"), (20, "season_heat_mean_abs_20"), (60, "season_heat_mean_abs_60"), (252, None)):
            moves = [label["abs_move"] for label in labels
                     if label["ticker"] != target["ticker"]
                     and label["event_id"] != target["event_id"]
                     and label["available_as_of_date"] <= decision
                     and position[label["event_date"]] >= position[decision] - window + 1]
            if window == 20:
                assert result["season_heat_count_20"] == len(moves)
            means[window] = sum(moves) / len(moves) if len(moves) >= 20 else None
            if key is not None:
                if means[window] is None:
                    assert result[key] is None
                else:
                    assert result[key] == pytest.approx(means[window])
        if means[20] is None or means[252] in (None, 0):
            assert result["season_heat_ratio_20_252"] is None
        else:
            assert result["season_heat_ratio_20_252"] == pytest.approx(means[20] / means[252])


def test_252_session_cutoff_excludes_expired_label():
    sessions = _session_sequence(date(2024, 1, 1), 253)
    decision = sessions[-1]
    labels = [_label("ANCIENT", "ANCIENT", sessions[0], 1000)]
    serial = 0
    for event_date in sessions[-5:-1]:
        for _ in range(5):
            labels.append(
                _label(f"E{serial:02d}", f"T{serial:02d}", event_date, 2)
            )
            serial += 1
    targets = [
        {
            "event_id": "TARGET",
            "ticker": "TARGET",
            "event_date": decision,
            "decision_session": decision,
        }
    ]

    result = compute_season_heat(labels, targets, sessions)[("TARGET", decision)]

    assert result["season_heat_count_20"] == 20
    assert result["season_heat_mean_abs_20"] == pytest.approx(2)
    assert result["season_heat_mean_abs_60"] == pytest.approx(2)
    assert result["season_heat_ratio_20_252"] == pytest.approx(1)


def test_expiring_labels_reset_zero_reference_sum():
    sessions = _session_sequence(date(2024, 1, 1), 274)
    labels = [
        _label(f"OLD{index:02d}", f"O{index:02d}", sessions[0], 0.1)
        for index in range(20)
    ]
    serial = 0
    for event_date in sessions[269:273]:
        for _ in range(5):
            labels.append(
                _label(f"NEW{serial:02d}", f"N{serial:02d}", event_date, 0.0)
            )
            serial += 1
    targets = [
        {
            "event_id": f"TARGET{position}",
            "ticker": f"G{position}",
            "event_date": sessions[position],
            "decision_session": sessions[position],
        }
        for position in (251, 252, 273)
    ]

    results = compute_season_heat(labels, targets, sessions)
    final = results[("TARGET273", sessions[273])]

    assert final["season_heat_count_20"] == 20
    assert final["season_heat_mean_abs_20"] == 0.0
    assert final["season_heat_ratio_20_252"] is None


def test_zero_reference_mean_leaves_ratio_null():
    sessions = _session_sequence(date(2024, 1, 1), 252, date(2024, 2, 19))
    decision = sessions[-1]

    event_dates = sessions[-5:-1]
    labels = [
        _label(f"R{index:02d}", f"R{index:02d}", event_dates[index % 4], 0)
        for index in range(20)
    ]
    targets = [
        {
            "event_id": "ZERO",
            "ticker": "ZZZ",
            "event_date": decision,
            "decision_session": decision,
        }
    ]

    result = compute_season_heat(labels, targets, sessions)[("ZERO", decision)]

    assert result["season_heat_count_20"] == 20
    assert result["season_heat_mean_abs_5"] == 0
    assert result["season_heat_mean_abs_20"] == 0
    assert result["season_heat_mean_abs_60"] == 0
    assert result["season_heat_ratio_20_252"] is None


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


def _duplicate_label_event_id():
    labels, targets, sessions = _valid_inputs()
    labels.append(dict(labels[0]))
    return labels, targets, sessions


def _label_extra_key():
    labels, targets, sessions = _valid_inputs()
    labels[0]["extra"] = 1
    return labels, targets, sessions


def _label_event_date_outside_sessions():
    labels, targets, sessions = _valid_inputs()
    labels[0]["event_date"] = "2025-03-08"
    return labels, targets, sessions


def _malformed_availability():
    labels, targets, sessions = _valid_inputs()
    labels[0]["available_as_of_date"] = "not-a-date"
    return labels, targets, sessions


def _nan_move():
    labels, targets, sessions = _valid_inputs()
    labels[0]["abs_move"] = float("nan")
    return labels, targets, sessions


def _negative_move():
    labels, targets, sessions = _valid_inputs()
    labels[0]["abs_move"] = -1
    return labels, targets, sessions


def _empty_sessions():
    labels, targets, sessions = _valid_inputs()
    return labels, targets, []


def _descending_sessions():
    labels, targets, sessions = _valid_inputs()
    return labels, targets, ["2025-03-10", "2025-03-07"]


def _malformed_target_event_date():
    labels, targets, sessions = _valid_inputs()
    targets[0]["event_date"] = "2025-3-11"
    return labels, targets, sessions


def _label_event_id_whitespace():
    labels, targets, sessions = _valid_inputs()
    labels[0]["event_id"] = " L1 "
    return labels, targets, sessions


def _target_event_id_whitespace():
    labels, targets, sessions = _valid_inputs()
    targets[0]["event_id"] = " T1 "
    return labels, targets, sessions


def _mixed_integer_float_overflow():
    labels, targets, sessions = _valid_inputs()
    labels[0]["abs_move"] = 10**1000
    labels.append(_label("L2", "DDD", "2025-03-07", 1.5))
    return labels, targets, sessions


def _sum_overflow():
    labels = [
        _label(f"L{index:02d}", f"X{index:02d}", "2025-03-07", 1e308)
        for index in range(20)
    ]
    _, targets, sessions = _valid_inputs()
    return labels, targets, sessions


@pytest.mark.parametrize(
    ("build", "expected_code", "expected_message_fragment"),
    [
        (
            _duplicate_target,
            "CONTRACT_MISMATCH",
            "duplicate target key",
        ),
        (
            _non_session_decision,
            "CONTRACT_MISMATCH",
            "target decision_session is not a trading session",
        ),
        (
            _availability_on_event_date,
            "EXPERIMENT_VARIANT_FAILED",
            "label available_as_of_date must be strictly after event_date",
        ),
        (
            _nan_move,
            "EXPERIMENT_VARIANT_FAILED",
            "abs_move must be a finite nonnegative number",
        ),
        (
            _negative_move,
            "EXPERIMENT_VARIANT_FAILED",
            "abs_move must be a finite nonnegative number",
        ),
        (
            _duplicate_label_event_id,
            "CONTRACT_MISMATCH",
            "duplicate label event_id",
        ),
        (
            _label_extra_key,
            "CONTRACT_MISMATCH",
            "label row keys do not match the contract",
        ),
        (
            _label_event_date_outside_sessions,
            "CONTRACT_MISMATCH",
            "label event_date is not a trading session",
        ),
        (
            _malformed_availability,
            "EXPERIMENT_VARIANT_FAILED",
            "value is not a canonical ISO date",
        ),
        (
            _empty_sessions,
            "CONTRACT_MISMATCH",
            "trading_sessions must be a nonempty strictly ascending sequence"
            " of canonical ISO dates",
        ),
        (
            _descending_sessions,
            "CONTRACT_MISMATCH",
            "trading_sessions must be a nonempty strictly ascending sequence"
            " of canonical ISO dates",
        ),
        (
            _malformed_target_event_date,
            "CONTRACT_MISMATCH",
            "value is not a canonical ISO date",
        ),
        (
            _label_event_id_whitespace,
            "CONTRACT_MISMATCH",
            "event IDs and tickers must be nonblank canonical strings",
        ),
        (
            _target_event_id_whitespace,
            "CONTRACT_MISMATCH",
            "event IDs and tickers must be nonblank canonical strings",
        ),
        (
            _mixed_integer_float_overflow,
            "EXPERIMENT_VARIANT_FAILED",
            "abs_move must be a finite nonnegative number",
        ),
        (
            _sum_overflow,
            "EXPERIMENT_VARIANT_FAILED",
            "season heat sum is not finite",
        ),
    ],
    ids=[
        "duplicate-target",
        "non-session-decision",
        "availability-not-after-event",
        "nan-move",
        "negative-move",
        "duplicate-label-event-id",
        "label-extra-key",
        "label-event-date-outside-sessions",
        "malformed-availability",
        "empty-sessions",
        "descending-sessions",
        "malformed-target-event-date",
        "label-event-id-whitespace",
        "target-event-id-whitespace",
        "mixed-integer-float-overflow",
        "sum-overflow",
    ],
)
def test_season_heat_rejects_invalid_inputs_with_native_codes(
    build, expected_code, expected_message_fragment
):
    labels, targets, sessions = build()
    with pytest.raises(DataError) as raised:
        compute_season_heat(labels, targets, sessions)
    assert raised.value.code == expected_code
    assert expected_message_fragment in str(raised.value)


@pytest.mark.parametrize("ticker", ["", " ", "\t", " ABC "])
def test_blank_tickers_are_refused(ticker):
    labels = [_label("L1", ticker, "2025-03-07", 1)]
    targets = [{
        "event_id": "T1",
        "ticker": "AAA",
        "event_date": "2025-03-11",
        "decision_session": "2025-03-10",
    }]
    sessions = ["2025-03-07", "2025-03-10", "2025-03-11"]
    with pytest.raises(DataError) as raised:
        compute_season_heat(labels, targets, sessions)
    assert raised.value.code == "CONTRACT_MISMATCH"
    labels = [_label("L1", "CCC", "2025-03-07", 1)]
    targets[0]["ticker"] = ticker
    with pytest.raises(DataError) as raised:
        compute_season_heat(labels, targets, sessions)
    assert raised.value.code == "CONTRACT_MISMATCH"
