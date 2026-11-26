"""Tests for the carry and dividend pricer (fill model F, slice F7; design section 4). Synthetic inputs only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments import fair_value_carry as fvc
from experiments import fair_value_pricer as fvp


def _grid():
    s, k, t, v, r = np.meshgrid([100.0], [80.0, 100.0, 120.0], [0.1, 1.0], [0.2, 0.5], [-0.02, 0.0, 0.05], indexing="ij")
    return [a.ravel() for a in (s, k, t, v, r)]


def _arr(x):
    return np.array([x], dtype=float)


def _reference_escrowed(S, K, T, v, r, b, D, tD, is_put):
    """Independent per-node escrowed American CRR, pure python, TREE_STEPS steps."""
    n = fvp.TREE_STEPS
    dt = T / n
    v = max(v, fvp.PUT_VOL_FLOOR)
    u = math.exp(v * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(b * dt) - d) / (u - d)
    disc = math.exp(-r * dt)
    root = S - D * math.exp(-r * tD)
    sign = -1.0 if is_put else 1.0

    def payoff(i, j):
        t_i = T * (i / n)
        stock = root * u ** (2 * j - i) + (D * math.exp(-r * (tD - t_i)) if t_i < tD else 0.0)
        return max(sign * (stock - K), 0.0)

    vals = [payoff(n, j) for j in range(n + 1)]
    for i in range(n - 1, -1, -1):
        vals = [max(disc * (p * vals[j + 1] + (1.0 - p) * vals[j]), payoff(i, j)) for j in range(i + 1)]
    return vals[0]


def test_carry_rate_interpolates_between_borrow30_and_borrow2yr():
    days = np.array([1.0, 30.0, 31.0, 380.0, 730.0, 2000.0])
    b, fb = fvc.carry_rate(days, 0.02, 0.04, 0.03)
    np.testing.assert_allclose(b, [0.02, 0.02, 0.02 + 0.02 / 700, 0.03, 0.04, 0.04], rtol=0, atol=1e-15)
    assert not np.any(fb)


def test_carry_rate_one_missing_uses_the_other_and_both_missing_gives_rate():
    days = np.array([1.0, 30.0, 31.0, 380.0, 730.0, 2000.0])
    b, fb = fvc.carry_rate(days, np.nan, 0.04, 0.03)  # borrow30 missing: borrow2yr flat
    np.testing.assert_allclose(b, 0.04, rtol=0, atol=1e-15)
    assert not np.any(fb)
    b, fb = fvc.carry_rate(days, 0.02, np.nan, 0.03)  # borrow2yr missing: borrow30 flat
    np.testing.assert_allclose(b, 0.02, rtol=0, atol=1e-15)
    assert not np.any(fb)
    b, fb = fvc.carry_rate(days, np.nan, np.nan, 0.03)  # both missing: b = r, flagged
    np.testing.assert_allclose(b, 0.03, rtol=0, atol=1e-15)
    assert np.all(fb)


DIVIDEND_CASES = [
    # (div_amt, div_days, expiry_days, spot, rate) -> (fallback, applied)
    (np.nan, 10.0, 30.0, 100.0, 0.04, True, False),    # 1: missing amount
    (-1.0, 10.0, 30.0, 100.0, 0.04, True, False),      # 1: negative amount
    (-1.0, 45.0, 30.0, 100.0, 0.04, True, False),      # 1 beats 4: negative and after expiry is fallback
    (0.0, np.nan, 30.0, 100.0, 0.04, False, False),    # 2 beats 3: zero amount is known, even with no date
    (1.0, np.nan, 30.0, 100.0, 0.04, True, False),     # 3: missing date
    (1.0, 0.0, 30.0, 100.0, 0.04, True, False),        # 3: date on the obs date
    (1.0, -5.0, 30.0, 100.0, 0.04, True, False),       # 3: date before obs
    (1.0, 30.0, 30.0, 100.0, 0.04, False, True),       # divDate == expiry is NOT after expiry: applied
    (1.0, 31.0, 30.0, 100.0, 0.04, False, False),      # 4: after expiry, known
    (100.0, 10.0, 30.0, 100.0, 0.0, True, False),      # 5: PV exactly equal to spot
    (500.0, 60.0, 30.0, 100.0, 0.04, False, False),    # 4 beats 5: after expiry even though PV >= spot
    (1.0, 10.0, 30.0, 100.0, 0.04, False, True),       # 6: applied
]


@pytest.mark.parametrize("amt, dd, ex, spot, r, fallback, applied", DIVIDEND_CASES)
def test_dividend_cases_and_precedence(amt, dd, ex, spot, r, fallback, applied):
    D, fb, ap = fvc.dividend(amt, dd, ex, spot, r)
    assert bool(fb) is fallback and bool(ap) is applied
    assert float(D) == (amt if applied else 0.0)


def test_dividend_applied_amount_is_kept_as_given():
    D, _, ap = fvc.dividend(np.array([0.5, 2.0, np.nan]), np.array([5.0, 20.0, 5.0]), np.array([30.0] * 3),
                            np.array([100.0] * 3), np.array([0.04] * 3))
    np.testing.assert_array_equal(D, [0.5, 2.0, 0.0])
    np.testing.assert_array_equal(ap, [True, True, False])


def test_zero_dividend_and_b_equal_r_is_bit_identical_to_pricer():
    s, k, t, v, r = _grid()
    calls = fvc.value(s, k, t, v, r, r, 0.0, 0.0, False)
    puts = fvc.value(s, k, t, v, r, r, 0.0, 0.0, True)
    euro_rows = r >= 0  # with a negative rate a call takes the tree, so it is not the European formula
    np.testing.assert_array_equal(calls[euro_rows], fvp.european_call(s, k, t, v, r)[euro_rows])
    np.testing.assert_array_equal(puts, fvp.american_put(s, k, t, v, r))


def test_european_call_with_carry_above_rate_matches_independent_formula():
    S, K, T, v, r, b = 100.0, 95.0, 0.5, 0.3, 0.02, 0.05

    def N(x):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    d1 = (math.log(S / K) + (b + 0.5 * v * v) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    expected = S * math.exp((b - r) * T) * N(d1) - K * math.exp(-r * T) * N(d2)
    got = fvc.value(S, K, T, v, r, b, 0.0, 0.0, False)
    assert float(got) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("carry_of, div, div_years", [
    (lambda r: r - 0.03, 0.0, 0.0),  # carry below the rate: tree
    (lambda r: r, 0.0, 0.0),         # carry = rate: tree only where the rate is negative
    (lambda r: r, 2.0, 0.05),        # a dividend: tree
])
def test_tree_call_never_below_same_lattice_european(carry_of, div, div_years):
    s, k, t, v, r = _grid()
    carry = carry_of(r)
    dv, dy = np.full_like(s, div), np.full_like(s, div_years)
    am = fvc.value(s, k, t, v, r, carry, dv, dy, False)
    eu = fvc._tree(s, k, t, np.maximum(v, fvp.PUT_VOL_FLOOR), r, carry, dv, dy, np.ones_like(s), early=False)
    routed = ~((dv == 0) & (carry >= r) & (r >= 0))  # the rows that take the tree
    ok = np.isfinite(am) & np.isfinite(eu) & routed
    assert ok.sum() > 0
    assert np.all(am[ok] >= eu[ok] - 1e-10)


def test_early_exercise_strictly_pays_for_deep_itm_call_with_dividend():
    american = float(fvc.value(100.0, 20.0, 1.0, 0.2, 0.05, 0.05, 10.0, 0.5, False))
    european = float(fvc._tree(_arr(100.0), _arr(20.0), _arr(1.0), _arr(0.2), _arr(0.05), _arr(0.05),
                               _arr(10.0), _arr(0.5), _arr(1.0), early=False)[0])
    assert american > european + 1e-3


def test_negative_rate_call_with_b_equal_r_uses_the_tree():
    S, K, T, v, r = 100.0, 100.0, 0.5, 0.25, -0.02
    got = fvc.value(S, K, T, v, r, r, 0.0, 0.0, False)
    tree = fvc._tree(_arr(S), _arr(K), _arr(T), _arr(v), _arr(r), _arr(r), _arr(0.0), _arr(0.0), _arr(1.0))
    assert float(got) == pytest.approx(float(tree[0]), rel=1e-12)


@pytest.mark.parametrize("is_put", [False, True])
def test_escrowed_tree_matches_independent_reference(is_put):
    S, K, T, v, r, b, D = 100.0, 102.0, 30 / 365, 0.3, 0.03, 0.01, 1.5
    for tD in (10 / 365, 30 / 365):  # an ex-date inside the life, and div_days == expiry_days
        got = fvc.value(S, K, T, v, r, b, D, tD, is_put)
        expected = _reference_escrowed(S, K, T, v, r, b, D, tD, is_put)
        assert float(got) == pytest.approx(expected, rel=1e-9, abs=1e-9)


def test_expired_rows_ignore_carry_and_dividend():
    assert float(fvc.value(100.0, 90.0, 0.0, 0.2, 0.03, np.nan, 5.0, 0.1, False)) == 10.0
    assert float(fvc.value(100.0, 110.0, 0.0, 0.2, 0.03, 9.0, 5.0, 0.1, True)) == 10.0


@pytest.mark.parametrize("carry", [5.0, -5.0])
def test_tree_probability_outside_unit_interval_is_nan(carry):
    assert np.isnan(float(fvc.value(100.0, 100.0, 1.0, 0.01, 0.0, carry, 0.0, 0.0, True)))  # put: p > 1 or p < 0


@pytest.mark.parametrize("is_put", [False, True])
def test_non_finite_carry_gives_nan_on_live_rows(is_put):
    assert np.isnan(float(fvc.value(100.0, 100.0, 1.0, 0.2, 0.03, np.nan, 0.0, 0.0, is_put)))
    assert np.isnan(float(fvc.value(100.0, 100.0, 1.0, 0.2, 0.03, np.inf, 0.0, 0.0, is_put)))


def test_no_dividend_call_on_tree_is_strictly_above_same_lattice_european():
    """b far below r, no dividend: the call takes the tree, and early exercise pays on a deep ITM call."""
    S, K, T, v, r, b = 100.0, 50.0, 1.0, 0.2, 0.05, -0.5
    american = float(fvc.value(S, K, T, v, r, b, 0.0, 0.0, False))
    european = float(fvc._tree(_arr(S), _arr(K), _arr(T), _arr(v), _arr(r), _arr(b), _arr(0.0), _arr(0.0),
                               _arr(1.0), early=False)[0])
    assert american > european + 1e-3
