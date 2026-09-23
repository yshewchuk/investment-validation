"""Financial values owned by scoring, independent of display formatting."""
from __future__ import annotations

from math import exp, fsum, log
from typing import Any, Mapping

from engine.v2.domain.valuation import multi_expiry_refusal, planned_exit_label

__all__ = ["entry_cost_pct", "financial_diagnostics"]

ORATS_EMOVE_FACTOR = 0.645


def entry_cost_pct(cost: Any, spot: Any) -> float | None:
    """``entry_cost / spot * 100`` -- legacy's exact definition
    (``engine/score.py`` ``Scorer._features``, ~2562-2564: ``built[
    "entry_cost_pct"] = entry_cost / spot_entry * 100``). ``None`` when
    either input is missing or spot is zero: the same non-finite outcome a
    gate's frozen feature check treats as MISSING_FEATURES either way, so
    this stays the single source of the formula for both the pricing-time
    gate feature (R4-20 gap) and the post-hoc display diagnostic below.
    """
    if cost is None or spot is None or float(spot) == 0.0:
        return None
    return float(cost) / float(spot) * 100.0


def _base(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entry_cost_pct": None,
        "model_vs_market": None,
        "premium_vs_fair": None,
        "terminal_payoff": record.get("payoff") or None,
        "planned_exit": {
            "entry_date": record.get("entry_date"),
            "exit_date": record.get("exit_date"),
            "expiry": record.get("expiry"),
        },
    }


def _entry_cost(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    value = entry_cost_pct(record.get("entry_cost"), record.get("spot"))
    if value is not None:
        diagnostics["entry_cost_pct"] = value
def _model_vs_market(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    driver = record.get("driver_prediction")
    implied = record.get("implied_move")
    if (record.get("driver_name") == "abs_move" and driver is not None
            and implied not in (None, 0, 0.0)):
        diagnostics["model_vs_market"] = (
            float(driver) / (float(implied) * ORATS_EMOVE_FACTOR))
def _runup_fair_premium(record: Mapping[str, Any]) -> float | None:
    coefficients = (record.get("payoff") or {}).get("coefficients") or {}
    driver = record.get("driver_prediction")
    move = record.get("runup_move_prediction")
    spot = record.get("spot")
    strike = record.get("strike")
    names = ("intercept", "implied_move", "abs_moneyness",
             "moneyness_sq_div10", "signed_moneyness",
             "implied_x_abs_moneyness_div10")
    if (driver is None or move is None or spot in (None, 0, 0.0)
            or strike in (None, 0, 0.0)
            or any(name not in coefficients for name in names)):
        return None
    values = []
    for direction in (-1.0, 1.0):
        exit_spot = float(spot) * exp(direction * float(move) / 100.0)
        money = 100.0 * log(exit_spot / float(strike))
        absolute = abs(money)
        terms = (1.0, float(driver), absolute, money * money / 10.0,
                 money, float(driver) * absolute / 10.0)
        values.append(fsum(float(coefficients[name]) * term
                           for name, term in zip(names, terms)))
    return max(0.0, fsum(values) / len(values) * 100.0)


def _fair_premium(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    driver = record.get("driver_prediction")
    payoff = record.get("payoff") or {}
    if record.get("model_fair_pct") is not None:
        value = record.get("model_fair_pct")
    elif payoff.get("kind") == "runup_payoff_surface":
        value = _runup_fair_premium(record)
    elif driver is not None and payoff.get("intercept") is not None \
            and payoff.get("slope") is not None:
        value = max(0.0, (float(payoff["intercept"])
                          + float(payoff["slope"]) * float(driver)) * 100.0)
    else:
        value = None
    diagnostics["fair_premium_pct"] = value
def _premium_and_width(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    fair = diagnostics["fair_premium_pct"]
    cost = record.get("entry_cost")
    if diagnostics["entry_cost_pct"] is not None and fair not in (None, 0, 0.0):
        diagnostics["premium_vs_fair"] = diagnostics["entry_cost_pct"] / float(fair)
    width = record.get("structure_width")
    if cost is not None and width not in (None, 0, 0.0):
        diagnostics["cost_over_width"] = float(cost) / float(width)
    else:
        diagnostics["cost_over_width"] = None
def _horizon(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    if record.get("expiry") is not None and record.get("exit_date") is not None:
        diagnostics["payoff_horizon"] = planned_exit_label(record.get("expiry"), record.get("exit_date"))
    diagnostics["payoff_refusal"] = multi_expiry_refusal(record.get("legs") or ())
def financial_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive full-precision ratios from frozen score values."""
    diagnostics = _base(record)
    _entry_cost(record, diagnostics)
    _model_vs_market(record, diagnostics)
    _fair_premium(record, diagnostics)
    _premium_and_width(record, diagnostics)
    _horizon(record, diagnostics)
    return diagnostics
