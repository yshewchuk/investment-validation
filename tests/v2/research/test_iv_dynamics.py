from __future__ import annotations

from datetime import date, timedelta

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research.iv_dynamics import compute_iv_dynamics

_START = date(2025, 1, 1)
_OUTPUT_KEYS = [
    "iv10_30", "iv_rv", "ex_ratio", "imp_chg5", "imp_chg10",
    "iv30_chg5", "iv30_chg20", "iv10_chg5",
]


def _surface(
    day: str,
    *,
    implied_move: float | None = None,
    iv10: float | None = None,
    iv30: float | None = None,
    exern_iv30: float | None = None,
    rvol30: float | None = None,
) -> dict:
    return {
        "date": day,
        "implied_move": implied_move,
        "iv10": iv10,
        "iv30": iv30,
        "exern_iv30": exern_iv30,
        "rvol30": rvol30,
    }


def _linear_surface(day: str, index: int) -> dict:
    return _surface(
        day,
        implied_move=float(index + 1),
        iv10=float(index + 10),
        iv30=float(index + 20),
        exern_iv30=float(index + 30),
        rvol30=float(index + 40),
    )


def _linear_series(count: int) -> list[dict]:
    return [
        _linear_surface((_START + timedelta(days=index)).isoformat(), index)
        for index in range(count)
    ]


def _reference_series(lag: int, gap: int) -> list[dict]:
    rows = [
        _linear_surface((_START + timedelta(days=index)).isoformat(), index)
        for index in range(lag)
    ]
    rows.append(_linear_surface((_START + timedelta(days=gap)).isoformat(), lag))
    return rows


def test_all_eight_values_on_hand_computed_series():
    result = compute_iv_dynamics(_linear_series(25))

    assert list(result) == _OUTPUT_KEYS
    assert result["iv10_30"] == pytest.approx(34.0 / 44.0)
    assert result["iv_rv"] == pytest.approx(44.0 / 64.0)
    assert result["ex_ratio"] == pytest.approx(54.0 / 44.0)
    assert result["imp_chg5"] == pytest.approx(25.0 / 20.0)
    assert result["imp_chg10"] == pytest.approx(25.0 / 15.0)
    assert result["iv30_chg5"] == pytest.approx(44.0 / 39.0)
    assert result["iv30_chg20"] == pytest.approx(44.0 / 24.0)
    assert result["iv10_chg5"] == pytest.approx(34.0 / 29.0)


@pytest.mark.parametrize(
    ("iv10", "iv30", "exern_iv30", "rvol30", "expected"),
    [
        (None, 2.0, 3.0, 4.0, {"iv10_30": None, "iv_rv": 0.5, "ex_ratio": 1.5}),
        (2.0, 0.0, 3.0, 4.0, {"iv10_30": None, "iv_rv": 0.0, "ex_ratio": None}),
        (2.0, 2.0, 3.0, None, {"iv10_30": 1.0, "iv_rv": None, "ex_ratio": 1.5}),
        (2.0, 2.0, 3.0, 0.0, {"iv10_30": 1.0, "iv_rv": None, "ex_ratio": 1.5}),
        (2.0, 2.0, None, 4.0, {"iv10_30": 1.0, "iv_rv": 0.5, "ex_ratio": None}),
    ],
)
def test_same_row_ratios_use_none_for_missing_or_zero_denominators(
    iv10, iv30, exern_iv30, rvol30, expected
):
    rows = [
        _surface(
            "2025-01-01",
            implied_move=1.0,
            iv10=iv10,
            iv30=iv30,
            exern_iv30=exern_iv30,
            rvol30=rvol30,
        )
    ]

    result = compute_iv_dynamics(rows)

    for key, value in expected.items():
        if value is None:
            assert result[key] is None
        else:
            assert result[key] == pytest.approx(value)


@pytest.mark.parametrize(
    ("key", "lag", "expected"),
    [
        ("imp_chg5", 5, 6.0 / 1.0),
        ("imp_chg10", 10, 11.0 / 1.0),
        ("iv30_chg5", 5, 25.0 / 20.0),
        ("iv30_chg20", 20, 40.0 / 20.0),
        ("iv10_chg5", 5, 15.0 / 10.0),
    ],
)
def test_lagged_ratio_needs_lag_plus_one_rows(key, lag, expected):
    assert compute_iv_dynamics(_linear_series(lag))[key] is None
    assert compute_iv_dynamics(_linear_series(lag + 1))[key] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("key", "lag"),
    [
        ("imp_chg5", 5),
        ("iv30_chg5", 5),
        ("iv10_chg5", 5),
        ("iv30_chg20", 20),
    ],
)
def test_stale_reference_boundary_is_two_k_plus_four_days(key, lag):
    fresh = compute_iv_dynamics(_reference_series(lag, 2 * lag + 4))
    stale = compute_iv_dynamics(_reference_series(lag, 2 * lag + 5))

    assert fresh[key] is not None
    assert stale[key] is None


def test_empty_input_returns_eight_none_values_in_order():
    result = compute_iv_dynamics([])

    assert list(result) == _OUTPUT_KEYS
    assert all(value is None for value in result.values())


def test_missing_current_values_leave_ratios_none():
    rows = _linear_series(21)
    for field in ("implied_move", "iv10", "iv30", "exern_iv30"):
        rows[-1][field] = None

    result = compute_iv_dynamics(rows)

    assert list(result) == _OUTPUT_KEYS
    assert all(value is None for value in result.values())


def test_computation_is_deterministic_and_does_not_mutate_input():
    rows = _linear_series(25)
    snapshot = [dict(row) for row in rows]

    first = compute_iv_dynamics(rows)
    second = compute_iv_dynamics(rows)

    assert first == second
    assert rows == snapshot


def _valid_rows() -> list[dict]:
    return [_linear_series(1)[0]]


def _unsorted_rows() -> list[dict]:
    rows = _valid_rows()
    rows.append(_linear_surface("2024-12-31", 1))
    return rows


def _duplicate_date_rows() -> list[dict]:
    rows = _valid_rows()
    rows.append(_linear_surface("2025-01-01", 1))
    return rows


def _bad_date_rows() -> list[dict]:
    rows = _valid_rows()
    rows[0]["date"] = "2025-1-1"
    return rows


def _extra_key_rows() -> list[dict]:
    rows = _valid_rows()
    rows[0]["extra"] = 1
    return rows


def _missing_key_rows() -> list[dict]:
    rows = _valid_rows()
    del rows[0]["iv10"]
    return rows


def _bool_value_rows() -> list[dict]:
    rows = _valid_rows()
    rows[0]["iv10"] = True
    return rows


def _nan_value_rows() -> list[dict]:
    rows = _valid_rows()
    rows[0]["iv10"] = float("nan")
    return rows


def _inf_value_rows() -> list[dict]:
    rows = _valid_rows()
    rows[0]["iv10"] = float("inf")
    return rows


def _string_input() -> str:
    return "not a sequence"


def _non_mapping_row() -> list:
    return [["not a mapping"]]


def _overflow_rows() -> list[dict]:
    return [
        _surface(
            "2025-01-01",
            implied_move=1.0,
            iv10=1e308,
            iv30=1e-308,
            exern_iv30=1.0,
            rvol30=1.0,
        )
    ]


@pytest.mark.parametrize(
    ("build", "expected_code", "expected_message_fragment"),
    [
        (
            _unsorted_rows,
            "CONTRACT_MISMATCH",
            "surface dates must be strictly ascending",
        ),
        (
            _duplicate_date_rows,
            "CONTRACT_MISMATCH",
            "surface dates must be strictly ascending",
        ),
        (
            _bad_date_rows,
            "CONTRACT_MISMATCH",
            "surface date is not a canonical ISO date",
        ),
        (
            _extra_key_rows,
            "CONTRACT_MISMATCH",
            "surface row keys do not match the contract",
        ),
        (
            _missing_key_rows,
            "CONTRACT_MISMATCH",
            "surface row keys do not match the contract",
        ),
        (
            _bool_value_rows,
            "CONTRACT_MISMATCH",
            "surface value must be a finite real number or None",
        ),
        (
            _nan_value_rows,
            "CONTRACT_MISMATCH",
            "surface value must be a finite real number or None",
        ),
        (
            _inf_value_rows,
            "CONTRACT_MISMATCH",
            "surface value must be a finite real number or None",
        ),
        (
            _string_input,
            "CONTRACT_MISMATCH",
            "surface_rows must be a sequence of mappings",
        ),
        (
            _non_mapping_row,
            "CONTRACT_MISMATCH",
            "surface_rows row is not a mapping",
        ),
        (
            _overflow_rows,
            "EXPERIMENT_VARIANT_FAILED",
            "iv dynamics ratio is not finite",
        ),
    ],
    ids=[
        "unsorted-dates",
        "duplicate-date",
        "bad-date",
        "extra-key",
        "missing-key",
        "bool-value",
        "nan-value",
        "inf-value",
        "string-input",
        "non-mapping-row",
        "overflow",
    ],
)
def test_iv_dynamics_rejects_invalid_inputs_with_native_codes(
    build, expected_code, expected_message_fragment
):
    with pytest.raises(DataError) as raised:
        compute_iv_dynamics(build())

    assert raised.value.code == expected_code
    assert expected_message_fragment in str(raised.value)
