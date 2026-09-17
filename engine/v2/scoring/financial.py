"""Financial values owned by scoring, independent of display formatting."""
from __future__ import annotations

from typing import Any, Mapping

from engine.v2.domain.valuation import multi_expiry_refusal, planned_exit_label

__all__ = ["financial_diagnostics"]

ORATS_EMOVE_FACTOR = 0.645


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
    spot = record.get("spot")
    cost = record.get("entry_cost")
    if spot is not None and cost is not None and float(spot) != 0.0:
        diagnostics["entry_cost_pct"] = float(cost) / float(spot) * 100.0
def _model_vs_market(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    driver = record.get("driver_prediction")
    implied = record.get("implied_move")
    if (record.get("driver_name") == "abs_move" and driver is not None
            and implied not in (None, 0, 0.0)):
        diagnostics["model_vs_market"] = (
            float(driver) / (float(implied) * ORATS_EMOVE_FACTOR))
def _fair_premium(record: Mapping[str, Any], diagnostics: dict[str, Any]) -> None:
    driver = record.get("driver_prediction")
    payoff = record.get("payoff") or {}
    if record.get("model_fair_pct") is not None:
        diagnostics["fair_premium_pct"] = record.get("model_fair_pct")
    elif driver is not None and payoff.get("intercept") is not None \
            and payoff.get("slope") is not None:
        diagnostics["fair_premium_pct"] = max(
            0.0, (float(payoff["intercept"]) + float(payoff["slope"]) * float(driver)) * 100.0)
    else:
        diagnostics["fair_premium_pct"] = None
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
