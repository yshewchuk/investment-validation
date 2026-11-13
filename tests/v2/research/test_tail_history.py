from __future__ import annotations

import pytest

from engine.v2.data.errors import DataError
from engine.v2.research.tail_history import (
    TAIL_HISTORY_COLUMNS,
    compute_tail_history,
)


def test_empty_returns_all_none_with_exact_column_order():
    result = compute_tail_history([])

    assert list(result) == list(TAIL_HISTORY_COLUMNS)
    assert list(result.values()) == [None] * len(TAIL_HISTORY_COLUMNS)


def test_single_value():
    result = compute_tail_history([5.0])

    assert result["tail_max_abs"] == pytest.approx(5.0)
    assert result["tail_std_abs"] is None
    assert result["tail_p75_abs"] == pytest.approx(5.0)
    assert result["tail_p90_abs"] == pytest.approx(5.0)
    assert result["tail_share_gt10"] == pytest.approx(0.0)
    assert result["tail_max4_abs"] == pytest.approx(5.0)
    assert result["tail_std_over_mean"] is None
    assert result["tail_max_over_mean"] == pytest.approx(1.0)


def test_two_values():
    result = compute_tail_history([2, 12])

    assert result["tail_std_abs"] == pytest.approx(5.0)
    assert result["tail_share_gt10"] == pytest.approx(0.5)
    assert result["tail_p75_abs"] == pytest.approx(9.5)
    assert result["tail_p90_abs"] == pytest.approx(11.0)
    assert result["tail_std_over_mean"] == pytest.approx(5 / 7)
    assert result["tail_max_over_mean"] == pytest.approx(12 / 7)


def test_interpolated_percentile_and_last_four_max():
    result = compute_tail_history([1, 2, 3, 4, 5, 20])

    assert result["tail_p90_abs"] == pytest.approx(12.5)
    assert result["tail_max4_abs"] == pytest.approx(20.0)

    other = compute_tail_history([30, 1, 1, 1, 1])
    assert other["tail_max_abs"] == pytest.approx(30.0)
    assert other["tail_max4_abs"] == pytest.approx(1.0)


def test_zero_mean_leaves_ratios_none():
    result = compute_tail_history([0, 0])

    assert result["tail_std_abs"] == pytest.approx(0.0)
    assert result["tail_std_over_mean"] is None
    assert result["tail_max_over_mean"] is None


def test_exactly_ten_is_not_above_the_threshold():
    assert compute_tail_history([10.0])["tail_share_gt10"] == pytest.approx(0.0)
    assert compute_tail_history([10.0, 10.0, 20.0])["tail_share_gt10"] == pytest.approx(
        1 / 3
    )


def test_input_is_not_mutated_and_tuple_is_accepted():
    values = [3.0, 1.0, 2.0]
    snapshot = list(values)

    from_list = compute_tail_history(values)
    from_tuple = compute_tail_history(tuple(values))

    assert values == snapshot
    assert from_tuple == from_list


@pytest.mark.parametrize(
    "abs_moves",
    [
        "abc",
        None,
        [True],
        ["1"],
        [float("nan")],
        [float("inf")],
        [-1.0],
    ],
    ids=["string", "none", "bool-element", "str-element", "nan", "inf", "negative"],
)
def test_contract_mismatch(abs_moves):
    with pytest.raises(DataError) as raised:
        compute_tail_history(abs_moves)

    assert raised.value.code == "CONTRACT_MISMATCH"


def test_sum_overflow_is_a_variant_failure():
    with pytest.raises(DataError) as raised:
        compute_tail_history([1e308, 1e308])

    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"
