"""Reference tests for the fair-value pricer (fill model F, design section 4)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.pnl_sim import black_scholes_put
from experiments import fair_value_pricer as fvp


def _crr_put(spot, strike, years, vol, rate, steps=5000):
    """Plain CRR American put for one row, zero dividends; independent of the module."""
    dt = years / steps
    u = math.exp(vol * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(rate * dt) - d) / (u - d)
    disc = math.exp(-rate * dt)
    m = np.arange(steps + 1)
    value = np.maximum(strike - spot * u ** (2.0 * m - steps), 0.0)
    for i in range(steps - 1, -1, -1):
        cont = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        m = np.arange(i + 1)
        value = np.maximum(cont, np.maximum(strike - spot * u ** (2.0 * m - i), 0.0))
    return float(value[0])


def _euro_put(spot, strike, years, vol, rate):
    """European put by put-call parity from the module's call."""
    call = float(fvp.european_call(spot, strike, years, vol, rate))
    return call - spot + strike * math.exp(-rate * years)


def test_european_call_published_values():
    # Hull textbook: S=42, K=40, r=0.10, vol=0.20, T=0.5.
    assert float(fvp.european_call(42.0, 40.0, 0.5, 0.20, 0.10)) == pytest.approx(4.759, abs=0.005)
    assert float(fvp.european_call(100.0, 100.0, 1.0, 0.2, 0.05)) == pytest.approx(10.4506, abs=1e-4)


# Longstaff & Schwartz (2001) Table 1, finite-difference column: K=40, r=0.06.
# That column allows 50 exercise dates a year, so it is a lower bound on the
# continuously exercisable value (largest gap 0.027, at S=36, vol 0.2, T=2).
LS_ROWS = [
    (36.0, 0.2, 1.0, 4.478),
    (36.0, 0.2, 2.0, 4.821),
    (36.0, 0.4, 1.0, 7.101),
    (36.0, 0.4, 2.0, 8.508),
    (40.0, 0.2, 1.0, 2.314),
    (40.0, 0.4, 1.0, 5.312),
    (44.0, 0.2, 1.0, 1.110),
    (44.0, 0.4, 2.0, 5.647),
]


@pytest.mark.parametrize("spot, vol, years, expected", LS_ROWS)
def test_american_put_longstaff_schwartz(spot, vol, years, expected):
    value = float(fvp.american_put(spot, 40.0, years, vol, 0.06))
    assert -0.005 <= value - expected <= 0.03


def test_american_put_hull_textbook_value():
    # Hull's S=K=50, r=0.10, vol=0.40, T=5/12: the tree converges to about 4.28.
    value = float(fvp.american_put(50.0, 50.0, 5.0 / 12.0, 0.40, 0.10))
    assert value == pytest.approx(4.28, abs=0.01)


# spot, strike, years, vol, rate, tolerance (0.02 for rows at spot 100)
REF_ROWS = [
    (100.0, 100.0, 1.0, 0.20, 0.05, 0.02),
    (100.0, 110.0, 1.0, 0.25, 0.05, 0.02),
    (100.0, 90.0, 0.5, 0.30, 0.03, 0.02),
    (100.0, 100.0, 0.5, 0.20, -0.02, 0.02),
    (40.0, 40.0, 1.0, 0.30, 0.06, 0.01),
    (36.0, 40.0, 2.0, 0.40, 0.06, 0.01),
    (44.0, 40.0, 0.5, 0.20, 0.10, 0.01),
]


@pytest.mark.parametrize("spot, strike, years, vol, rate, tol", REF_ROWS)
def test_american_put_matches_fine_crr_reference(spot, strike, years, vol, rate, tol):
    value = float(fvp.american_put(spot, strike, years, vol, rate))
    assert abs(value - _crr_put(spot, strike, years, vol, rate)) < tol


def test_zero_rate_put_matches_black_scholes_put():
    for years in (0.02, 0.1, 0.5, 1.0):
        for strike in range(80, 121, 10):
            for vol in (0.2, 0.4, 0.6, 0.8):
                am = float(fvp.american_put(100.0, float(strike), years, vol, 0.0))
                eu = float(black_scholes_put(100.0, float(strike), years, vol))
                assert am == pytest.approx(eu, abs=0.05)


def test_zero_rate_call_is_put_plus_forward():
    for strike in (80.0, 100.0, 120.0):
        for vol in (0.2, 0.5):
            for years in (0.25, 1.0):
                call = float(fvp.european_call(100.0, strike, years, vol, 0.0))
                put = float(black_scholes_put(100.0, strike, years, vol))
                assert call == pytest.approx(put + 100.0 - strike, abs=1e-9)


def test_zero_rate_call_minus_american_put_is_forward():
    for strike in (80.0, 100.0, 120.0):
        for vol in (0.2, 0.5):
            call = float(fvp.european_call(100.0, strike, 1.0, vol, 0.0))
            put = float(fvp.american_put(100.0, strike, 1.0, vol, 0.0))
            assert (call - put) == pytest.approx(100.0 - strike, abs=0.05)


def test_american_put_at_least_european_put_on_grid():
    for rate in (0.0, 0.03, 0.10):
        for spot in (80.0, 100.0, 120.0):
            for strike in (90.0, 100.0, 110.0):
                for years in (0.25, 1.0):
                    for vol in (0.2, 0.4):
                        am = float(fvp.american_put(spot, strike, years, vol, rate))
                        # At rate 0 the premium is zero, so the slack covers the 200-step discretisation error.
                        assert am >= _euro_put(spot, strike, years, vol, rate) - 0.03


def test_early_exercise_premium_deep_itm_high_rate():
    am = float(fvp.american_put(60.0, 100.0, 1.0, 0.2, 0.10))
    assert am - _euro_put(60.0, 100.0, 1.0, 0.2, 0.10) > 0.1


def test_american_put_never_below_intrinsic():
    spots = np.linspace(50.0, 150.0, 9)
    for rate in (-0.05, 0.0, 0.10):
        values = fvp.american_put(spots, 100.0, 1.0, 0.3, rate)
        assert np.all(values >= np.maximum(100.0 - spots, 0.0) - 1e-12)


def test_negative_rate_put_close_to_european():
    for spot in (100.0, 105.0):
        am = float(fvp.american_put(spot, 100.0, 1.0, 0.3, -0.02))
        assert am == pytest.approx(_euro_put(spot, 100.0, 1.0, 0.3, -0.02), abs=0.05)


def test_monotonicity_at_atm_rate_three_percent():
    rate = 0.03
    vols = np.linspace(0.1, 0.6, 6)
    assert np.all(np.diff(fvp.american_put(100.0, 100.0, 1.0, vols, rate)) >= -1e-9)
    assert np.all(np.diff(fvp.european_call(100.0, 100.0, 1.0, vols, rate)) >= -1e-9)

    strikes = np.linspace(80.0, 120.0, 9)
    assert np.all(np.diff(fvp.american_put(100.0, strikes, 1.0, 0.25, rate)) >= -1e-9)
    assert np.all(np.diff(fvp.european_call(100.0, strikes, 1.0, 0.25, rate)) <= 1e-9)

    years = np.linspace(0.1, 2.0, 8)
    assert np.all(np.diff(fvp.american_put(100.0, 100.0, years, 0.25, rate)) >= -0.02)
    assert np.all(np.diff(fvp.european_call(100.0, 100.0, years, 0.25, rate)) >= -0.02)


@pytest.mark.parametrize("years", [0.0, -1.0, -np.inf])
def test_expired_rows_are_intrinsic_even_without_vol_or_rate(years):
    nan = np.nan
    assert float(fvp.american_put(100.0, 110.0, years, nan, nan)) == pytest.approx(10.0)
    assert float(fvp.american_put(100.0, 90.0, years, nan, nan)) == 0.0
    assert float(fvp.european_call(100.0, 90.0, years, nan, nan)) == pytest.approx(10.0)
    assert float(fvp.european_call(100.0, 110.0, years, nan, nan)) == 0.0


@pytest.mark.parametrize("func", [fvp.american_put, fvp.european_call])
def test_invalid_inputs_give_nan(func):
    for spot in (0.0, -5.0, np.nan, np.inf):
        assert np.isnan(float(func(spot, 100.0, 0.5, 0.2, 0.03)))
        # An invalid spot is NaN at expiry too.
        assert np.isnan(float(func(spot, 100.0, 0.0, 0.2, 0.03)))
    for strike in (0.0, -1.0, np.nan, np.inf):
        assert np.isnan(float(func(100.0, strike, 0.5, 0.2, 0.03)))
    for years in (np.nan, np.inf):
        assert np.isnan(float(func(100.0, 100.0, years, 0.2, 0.03)))
    for vol in (0.0, -0.2, np.nan, np.inf):
        assert np.isnan(float(func(100.0, 100.0, 0.5, vol, 0.03)))
    for rate in (np.nan, np.inf):
        assert np.isnan(float(func(100.0, 100.0, 0.5, 0.2, rate)))


def test_put_vol_floor_but_call_is_not_floored():
    low = fvp.american_put(100.0, 100.0, 0.5, 0.001, 0.0)
    floor = fvp.american_put(100.0, 100.0, 0.5, 0.01, 0.0)
    np.testing.assert_array_equal(low, floor)
    assert float(fvp.european_call(100.0, 100.0, 1.0, 0.001, 0.0)) < float(
        fvp.european_call(100.0, 100.0, 1.0, 0.01, 0.0)
    )


def test_tree_probability_outside_unit_interval_is_nan():
    # r*sqrt(dt) = 0.02 > vol: p > 1.
    assert np.isnan(float(fvp.american_put(100.0, 100.0, 2.0, 0.01, 0.20)))
    # Negative rate with the same step: p < 0.
    assert np.isnan(float(fvp.american_put(100.0, 100.0, 2.0, 0.01, -0.20)))
    assert np.isfinite(float(fvp.european_call(100.0, 100.0, 2.0, 0.01, 0.20)))


def test_pinned_constants():
    assert fvp.TREE_STEPS == 200
    assert fvp.PUT_VOL_FLOOR == 0.01


def test_batch_equals_single_rows_and_is_order_independent():
    spot = np.array([100.0, 100.0, 0.0, 120.0, 95.0, 100.0, 100.0])
    strike = np.array([100.0, 90.0, 100.0, 110.0, 100.0, np.nan, 100.0])
    years = np.array([0.5, 0.0, 1.0, -1.0, 0.25, 1.0, 1.0])
    vol = np.array([0.2, np.nan, 0.3, np.nan, 0.5, 0.2, 0.0])
    rate = np.array([0.03, np.nan, 0.05, 0.02, -0.01, 0.03, 0.03])
    for func in (fvp.american_put, fvp.european_call):
        batch = func(spot, strike, years, vol, rate)
        singles = np.array(
            [float(func(s, k, t, v, r)) for s, k, t, v, r in zip(spot, strike, years, vol, rate)]
        )
        np.testing.assert_array_equal(batch, singles)
        reverse = func(spot[::-1], strike[::-1], years[::-1], vol[::-1], rate[::-1])
        np.testing.assert_array_equal(batch[::-1], reverse)


def test_broadcast_shape_dtype_and_no_mutation():
    strikes = np.array([90.0, 100.0, 110.0])
    kept = strikes.copy()
    put = fvp.american_put(100.0, strikes, 0.5, 0.2, 0.03)
    call = fvp.european_call(100.0, strikes, 0.5, 0.2, 0.03)
    assert put.shape == (3,) and call.shape == (3,)
    assert put.dtype == np.float64 and call.dtype == np.float64
    for i, k in enumerate(strikes):
        assert put[i] == float(fvp.american_put(100.0, k, 0.5, 0.2, 0.03))
        assert call[i] == float(fvp.european_call(100.0, k, 0.5, 0.2, 0.03))
    assert np.ndim(fvp.american_put(100.0, 100.0, 0.5, 0.2, 0.03)) == 0
    np.testing.assert_array_equal(strikes, kept)
