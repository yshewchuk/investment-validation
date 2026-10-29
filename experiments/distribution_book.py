"""Explicit research-book policy and numeric validation; no execution or pricing."""
from math import isfinite

from engine.v2.ops.errors import fail

__all__ = ["validate_book_config"]


def _require(condition):
    if not condition:
        raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution book inputs or arithmetic")


def _number(value):
    _require(type(value) in (int, float))
    try:
        _require(isfinite(value))
    except OverflowError:
        _require(False)
    return value


def validate_book_config(config):
    """Validate every explicit limit; settlement delays are calendar days."""
    _require(type(config) is dict and set(config) == {"initial_cash", "contracts_per_trade",
        "max_trade_debit", "max_ticker_debit", "max_concurrent", "settlement_delay_days", "same_time_order"})
    for key in ("initial_cash", "max_trade_debit", "max_ticker_debit"):
        _require(_number(config[key]) > 0)
    for key in ("contracts_per_trade", "max_concurrent", "settlement_delay_days"):
        _require(type(config[key]) is int and config[key] >= (0 if key == "settlement_delay_days" else 1))
    _require(config["same_time_order"] in ("settlements_before_entries", "entries_before_settlements"))
    return dict(config)
