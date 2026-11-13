from __future__ import annotations

import math
import statistics
from datetime import date, timedelta

import pytest

from engine.v2.data.errors import DataError
from engine.v2.features.own_vol import OWN_VOL_COLUMNS, compute_own_vol

_END = date(2025, 3, 31)


def _rows(closes, end=_END):
    dates = [end - timedelta(days=len(closes) - 1 - index) for index in range(len(closes))]
    return [(day.isoformat(), close) for day, close in zip(dates, closes)]


def _ceiling(end=_END):
    return end.isoformat()


def _doubling(count):
    return [2.0**index for index in range(count)]


def _assert_all_none(result):
    assert list(result) == list(OWN_VOL_COLUMNS)
    assert all(result[name] is None for name in OWN_VOL_COLUMNS)


def test_constant_growth_has_zero_volatility():
    result = compute_own_vol(_rows(_doubling(61)), _ceiling())

    assert result["own_rvol5"] == 0.0
    assert result["own_rvol10"] == 0.0
    assert result["own_rvol60"] == 0.0
    assert result["own_absret1"] == pytest.approx(math.log(2.0) * 100)
    assert result["own_rvol5_60"] is None


def test_known_values_match_sample_standard_deviation():
    factors = [1 + (((index * 3) % 11) - 5) / 100 for index in range(61)]
    closes = [100.0]
    for factor in factors[1:]:
        closes.append(closes[-1] * factor)
    returns = [
        math.log(current / previous)
        for previous, current in zip(closes, closes[1:])
    ]

    result = compute_own_vol(_rows(closes), _ceiling())

    for size in (5, 10, 60):
        expected = statistics.stdev(returns[-size:]) * math.sqrt(252) * 100
        assert result[f"own_rvol{size}"] == pytest.approx(expected)
    assert result["own_absret1"] == pytest.approx(abs(returns[-1]) * 100)
    expected_ratio = statistics.stdev(returns[-5:]) / statistics.stdev(returns[-60:])
    assert result["own_rvol5_60"] == pytest.approx(expected_ratio)


@pytest.mark.parametrize("size", (5, 10, 60))
def test_window_boundary_requires_one_more_close(size):
    enough = compute_own_vol(_rows(_doubling(size + 1)), _ceiling())
    assert enough[f"own_rvol{size}"] is not None

    short = compute_own_vol(_rows(_doubling(size)), _ceiling())
    assert short[f"own_rvol{size}"] is None


def test_abs_return_needs_two_closes():
    single = compute_own_vol(_rows([10.0]), _ceiling())
    assert single["own_absret1"] is None

    pair = compute_own_vol(_rows([10.0, 11.0]), _ceiling())
    assert pair["own_absret1"] == pytest.approx(math.log(1.1) * 100)


def test_staleness_boundary():
    fresh = _rows(_doubling(6), end=_END - timedelta(days=5))
    assert compute_own_vol(fresh, _ceiling())["own_rvol5"] is not None

    stale = _rows(_doubling(6), end=_END - timedelta(days=6))
    _assert_all_none(compute_own_vol(stale, _ceiling()))


@pytest.mark.parametrize("bad_close", (0.0, -1.0))
def test_non_positive_close_in_window_makes_all_missing(bad_close):
    closes = _doubling(61)
    closes[-10] = bad_close

    _assert_all_none(compute_own_vol(_rows(closes), _ceiling()))


def test_non_positive_close_before_window_is_ignored():
    closes = _doubling(70)
    closes[0] = -1.0
    window = _doubling(70)[9:]
    returns = [
        math.log(current / previous)
        for previous, current in zip(window, window[1:])
    ]

    result = compute_own_vol(_rows(closes), _ceiling())

    assert result["own_rvol60"] == pytest.approx(
        statistics.stdev(returns[-60:]) * math.sqrt(252) * 100
    )
    assert result["own_absret1"] is not None


def test_empty_input_returns_all_missing():
    _assert_all_none(compute_own_vol([], _ceiling()))


def test_single_close_returns_all_missing():
    _assert_all_none(compute_own_vol([(_END.isoformat(), 10.0)], _ceiling()))


def test_result_keys_are_the_contract_order():
    result = compute_own_vol(_rows(_doubling(6)), _ceiling())

    assert tuple(result) == OWN_VOL_COLUMNS


def test_ratio_is_none_when_short_window_is_missing():
    result = compute_own_vol(_rows(_doubling(5)), _ceiling())

    assert result["own_rvol5"] is None
    assert result["own_rvol5_60"] is None


def test_non_finite_math_from_finite_closes_raises_variant():
    closes = [1e-300, 1e300, 1e-300, 1e300, 1e-300, 1e300]

    with pytest.raises(DataError) as raised:
        compute_own_vol(_rows(closes), _ceiling())

    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"


def _valid_closes():
    return _rows(_doubling(6))


def _unsorted_rows():
    return [("2025-03-02", 1.0), ("2025-03-01", 1.1)]


def _duplicate_rows():
    return [("2025-03-01", 1.0), ("2025-03-01", 1.1)]


def _after_ceiling_rows():
    rows = _valid_closes()
    rows[-1] = ((_END + timedelta(days=1)).isoformat(), rows[-1][1])
    return rows


def _nan_close_rows():
    rows = _valid_closes()
    rows[-1] = (rows[-1][0], float("nan"))
    return rows


def _inf_close_rows():
    rows = _valid_closes()
    rows[-1] = (rows[-1][0], float("inf"))
    return rows


def _bool_close_rows():
    rows = _valid_closes()
    rows[-1] = (rows[-1][0], True)
    return rows


def _string_close_rows():
    rows = _valid_closes()
    rows[-1] = (rows[-1][0], "10.0")
    return rows


def _non_iso_date_rows():
    rows = _valid_closes()
    rows[-1] = ("2025/03/31", rows[-1][1])
    return rows


def _date_object_rows():
    rows = _valid_closes()
    rows[-1] = (_END, rows[-1][1])
    return rows


def _three_item_rows():
    rows = _valid_closes()
    rows[-1] = (rows[-1][0], rows[-1][1], 1.0)
    return rows


def _scalar_row():
    rows = _valid_closes()
    rows[-1] = 1.0
    return rows


@pytest.mark.parametrize(
    "build",
    [
        _unsorted_rows,
        _duplicate_rows,
        _after_ceiling_rows,
        _nan_close_rows,
        _inf_close_rows,
        _bool_close_rows,
        _string_close_rows,
        _non_iso_date_rows,
        _date_object_rows,
        _three_item_rows,
        _scalar_row,
    ],
)
def test_bad_closes_refuse_contract(build):
    with pytest.raises(DataError) as raised:
        compute_own_vol(build(), _ceiling())

    assert raised.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("value", (42, "not-a-series", {"2025-03-01": 1.0}))
def test_non_sequence_closes_refuse_contract(value):
    with pytest.raises(DataError) as raised:
        compute_own_vol(value, _ceiling())

    assert raised.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("ceiling", ("2025/03/31", "2025-3-31", 20250331, None))
def test_bad_ceiling_refuses_contract(ceiling):
    with pytest.raises(DataError) as raised:
        compute_own_vol(_valid_closes(), ceiling)

    assert raised.value.code == "CONTRACT_MISMATCH"
