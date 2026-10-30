"""Shared legacy oracle for Phase 4 financial diagnostics parity.

Reproduces the frozen renderer's fair-premium and per-record financial
diagnostics arithmetic independently of checks, ops, and native scoring code.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

__all__ = ["expected_financial_diagnostics", "legacy_fair_premium"]


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


_RUNUP_COEFFICIENTS = (
    "intercept", "implied_move", "abs_moneyness",
    "moneyness_sq_div10", "signed_moneyness",
    "implied_x_abs_moneyness_div10",
)


def _runup_inputs(record: Mapping[str, Any], payoff: Mapping[str, Any],
                  driver: float | None):
    coefficients = payoff.get("coefficients") or {}
    move = _finite_number(record.get("runup_move_prediction"))
    spot = _finite_number(record.get("spot"))
    strike = _finite_number(record.get("strike"))
    if (not isinstance(coefficients, Mapping) or driver is None or move is None
            or spot is None or strike is None or spot <= 0.0 or strike <= 0.0
            or any(name not in coefficients for name in _RUNUP_COEFFICIENTS)):
        return None
    numeric = {}
    for name in _RUNUP_COEFFICIENTS:
        value = _finite_number(coefficients[name])
        if value is None:
            return None
        numeric[name] = value
    return driver, move, spot, strike, numeric


def _runup_fair_premium(record: Mapping[str, Any],
                        payoff: Mapping[str, Any],
                        driver: float | None) -> float | None:
    inputs = _runup_inputs(record, payoff, driver)
    if inputs is None:
        return None
    driver, move, spot, strike, coefficients = inputs
    values = []
    try:
        for direction in (-1.0, 1.0):
            exit_spot = spot * math.exp(direction * move / 100.0)
            money = 100.0 * math.log(exit_spot / strike)
            absolute = abs(money)
            terms = (
                1.0, driver, absolute, money * money / 10.0,
                money, driver * absolute / 10.0,
            )
            value = sum(
                coefficients[name] * term
                for name, term in zip(_RUNUP_COEFFICIENTS, terms)
            )
            if not math.isfinite(value):
                return None
            values.append(value)
    except (OverflowError, ValueError):
        return None
    fair = max(0.0, sum(values) / len(values) * 100.0)
    return fair if math.isfinite(fair) else None


def _linear_fair_premium(payoff: Mapping[str, Any],
                         driver: float | None) -> float | None:
    intercept = _finite_number(payoff.get("intercept"))
    slope = _finite_number(payoff.get("slope"))
    if driver is None or intercept is None or slope is None:
        return None
    fair = max(0.0, (intercept + slope * driver) * 100.0)
    return fair if math.isfinite(fair) else None


def legacy_fair_premium(record: Mapping[str, Any]) -> float | None:
    """Reproduce the frozen renderer oracle; invalid inputs are unavailable."""
    payoff = record.get("payoff") or {}
    if not isinstance(payoff, Mapping):
        return None
    driver = _finite_number(record.get("driver_prediction"))
    if payoff.get("kind") == "runup_payoff_surface":
        return _runup_fair_premium(record, payoff, driver)
    return _linear_fair_premium(payoff, driver)


def _ratio(numerator: float | None, denominator: float | None,
           multiplier: float = 1.0) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    result = numerator / denominator * multiplier
    return result if math.isfinite(result) else None


def expected_financial_diagnostics(
    record: Mapping[str, Any],
) -> dict[str, float | None]:
    """Compute expected diagnostics without importing native scoring code."""
    cost = _finite_number(record.get("entry_cost"))
    spot = _finite_number(record.get("spot"))
    entry_cost_pct = _ratio(cost, spot, 100.0)
    driver = _finite_number(record.get("driver_prediction"))
    implied = _finite_number(record.get("implied_move"))
    model_vs_market = (
        _ratio(driver, implied * 0.645)
        if record.get("driver_name") == "abs_move" and implied is not None
        else None
    )
    fair = legacy_fair_premium(record)
    premium_vs_fair = _ratio(entry_cost_pct, fair)
    width = _finite_number(record.get("structure_width"))
    cost_over_width = _ratio(cost, width)
    return {
        "entry_cost_pct": entry_cost_pct,
        "model_vs_market": model_vs_market,
        "fair_premium_pct": fair,
        "premium_vs_fair": premium_vs_fair,
        "cost_over_width": cost_over_width,
    }
