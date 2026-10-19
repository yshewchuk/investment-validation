"""Closed exit declarations shared by planning and execution."""
from types import MappingProxyType

import pytest

from engine.v2.foundation.experiment_exits import valid_exit_recipe, valid_threshold_units


def target_stop():
    return {"kind": "target_stop", "trading_days": 3, "target_pnl": 0.5, "stop_pnl": -0.5}


@pytest.mark.parametrize("recipe", [
    {"kind": "fixed_day", "trading_days": 1}, target_stop(),
    {**target_stop(), "threshold_units": "quote_price_times_quantity"},
    {**target_stop(), "threshold_units": "entry_debit_fraction"},
])
def test_valid_recipe_is_not_normalized(recipe):
    before = dict(recipe)
    assert valid_exit_recipe(MappingProxyType(recipe)) is True
    assert recipe == before


@pytest.mark.parametrize("key,value", [
    ("kind", "unknown"), ("trading_days", True), ("trading_days", 0),
    ("trading_days", 3.0), ("target_pnl", True), ("target_pnl", 0),
    ("target_pnl", -1), ("target_pnl", "0.5"), ("target_pnl", None),
    ("target_pnl", float("nan")), ("target_pnl", float("inf")),
    ("target_pnl", 10 ** 400), ("stop_pnl", False), ("stop_pnl", 0),
    ("stop_pnl", 1), ("stop_pnl", float("-inf")), ("stop_pnl", -(10 ** 400)),
    ("threshold_units", "percent"), ("threshold_units", None),
    ("threshold_units", []), ("threshold_units", {}), ("event_thresholds", {}),
])
def test_invalid_recipe_values_return_false(key, value):
    assert valid_exit_recipe({**target_stop(), key: value}) is False


@pytest.mark.parametrize("recipe", [None, [], "target_stop", {},
    {"kind": "fixed_day", "trading_days": 1, "threshold_units": "entry_debit_fraction"},
    *[{key: value for key, value in target_stop().items() if key != missing}
      for missing in target_stop()],
])
def test_invalid_recipe_shapes_return_false(recipe):
    assert valid_exit_recipe(recipe) is False


@pytest.mark.parametrize("units,expected", [
    ("quote_price_times_quantity", True), ("entry_debit_fraction", True),
    (None, False), ("", False), ("percent", False), ("entry_credit_fraction", False),
    ([], False), ({}, False), (True, False),
])
def test_units_have_one_closed_vocabulary(units, expected):
    assert valid_threshold_units(units) is expected
