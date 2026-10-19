"""Pure exit-declaration admission shared by planning and execution."""
from collections.abc import Mapping
from math import isfinite


def valid_threshold_units(value) -> bool:
    """Admit only the declared absolute or net-opening-debit-relative unit."""
    return value in ("quote_price_times_quantity", "entry_debit_fraction")


def valid_exit_recipe(recipe) -> bool:
    """Check a closed fixed-day/target-stop declaration without normalizing it."""
    if not isinstance(recipe, Mapping):
        return False
    days = recipe.get("trading_days")
    if type(days) is not int or days <= 0:
        return False
    kind = recipe.get("kind")
    if kind == "fixed_day":
        return set(recipe) == {"kind", "trading_days"}
    required = {"kind", "trading_days", "target_pnl", "stop_pnl"}
    if (kind != "target_stop" or set(recipe) not in (required, required | {"threshold_units"})
            or not valid_threshold_units(recipe.get("threshold_units", "quote_price_times_quantity"))):
        return False
    try:
        return all(type(recipe[key]) in (int, float) and isfinite(recipe[key])
                   and sign * recipe[key] > 0
                   for key, sign in (("target_pnl", 1), ("stop_pnl", -1)))
    except OverflowError:
        return False
