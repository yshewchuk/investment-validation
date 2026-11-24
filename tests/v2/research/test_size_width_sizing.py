from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from engine.forecast_sizing import forecast_params
from experiments import size_width_sizing as sizing
from experiments.size_pnl_shortvol import NAMES

SPEC_PATH = Path(sizing.__file__).resolve().parent / "EXP-189_short_vol_width_tuning" / "spec.yaml"
FORECASTS = (1.0, 2.5, 5.0, 8.0)


def _design_grid():
    return yaml.safe_load(SPEC_PATH.read_text())["grid"]["k"]


# --------------------------------------------------------------------------
# k = 1 is production
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("forecast", FORECASTS)
def test_k_one_equals_production(name, forecast):
    assert sizing.k_params(name, forecast, 1.0) == forecast_params(name, forecast)


# --------------------------------------------------------------------------
# k scales the width; bounds apply after k
# --------------------------------------------------------------------------


def test_width_scales_linearly_with_k():
    base = sizing.k_params("TWIN-P", 3.0, 1.0)["width_moneyness"]
    doubled = sizing.k_params("TWIN-P", 3.0, 2.0)["width_moneyness"]
    assert doubled == pytest.approx(2 * base)


def test_unscaled_width_below_floor_is_brought_inside_by_k():
    # TWIN-P5 divisor is 1: f = 0.3 gives 0.003 at k = 1 (below WIDTH_MIN = 0.005), 0.009 at k = 3.
    assert sizing.k_params("TWIN-P5", 0.3, 1.0) is None
    assert sizing.k_params("TWIN-P5", 0.3, 3.0)["width_moneyness"] == pytest.approx(0.009)


def test_in_bounds_at_k_one_but_above_ceiling_at_k_three_is_declined():
    # TWIN-P5 divisor is 1: f = 6.0 gives 0.06 at k = 1 (in bounds), 0.18 at k = 3 (above WIDTH_MAX = 0.15).
    assert sizing.k_params("TWIN-P5", 6.0, 1.0)["width_moneyness"] == pytest.approx(0.06)
    assert sizing.k_params("TWIN-P5", 6.0, 3.0) is None


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("forecast", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), "abc", None])
def test_non_forecast_values_return_none(forecast):
    assert sizing.k_params("TWIN-P", forecast, 1.0) is None


def test_unknown_strategy_returns_none():
    assert sizing.k_params("NOT-A-STRUCTURE", 2.5, 1.0) is None


@pytest.mark.parametrize("k", [0.0, -1.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_k_raises(k):
    with pytest.raises(ValueError, match="positive finite"):
        sizing.k_params("TWIN-P", 2.5, k)


# --------------------------------------------------------------------------
# bounds are inclusive: TWIN-P5 (divisor 1.0) at k = 1 has width = forecast / 100
# --------------------------------------------------------------------------


def test_width_exactly_at_the_floor_is_accepted():
    # forecast 0.5 gives width 0.005 == WIDTH_MIN
    result = sizing.k_params("TWIN-P5", 0.5, 1.0)
    assert result is not None
    assert result["width_moneyness"] == pytest.approx(0.005)


def test_width_exactly_at_the_ceiling_is_accepted():
    # forecast 15.0 gives width 0.15 == WIDTH_MAX
    result = sizing.k_params("TWIN-P5", 15.0, 1.0)
    assert result is not None
    assert result["width_moneyness"] == pytest.approx(0.15)


def test_widths_just_outside_the_bounds_are_declined():
    assert sizing.k_params("TWIN-P5", 0.49, 1.0) is None
    assert sizing.k_params("TWIN-P5", 15.1, 1.0) is None


# --------------------------------------------------------------------------
# module constants
# --------------------------------------------------------------------------


def test_divisors_cover_exactly_the_seven_names():
    assert len(NAMES) == 7
    assert set(sizing.DIVISORS) == set(NAMES)


def test_grid_contains_one_and_equals_the_design_grid():
    assert 1.0 in sizing.GRID
    assert list(sizing.GRID) == _design_grid()
