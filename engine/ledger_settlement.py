"""Recorded selection rules and comparisons for simulated ledger settlement."""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from typing import Mapping

import numpy as np
import pandas as pd

from engine.jsonio import json_safe
from engine.structures import ExpirySelector, LegSpec, StrikeSelector, Structure

POLICY = "recorded_selection_rule_v1"


def selection_key(settlement: Mapping, alpha) -> str:
    payload = json.dumps(json_safe([settlement, alpha]), sort_keys=True,
                         allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def recorded_structure(settlement: Mapping) -> Structure:
    """Rehydrate the entire spec without consulting mutable strategy factories."""
    if settlement.get("policy") != POLICY or settlement.get("spec_version") != 1:
        raise ValueError("missing or unsupported settlement policy/spec version")
    spec = settlement.get("structure_spec")
    if not isinstance(spec, dict):
        raise ValueError("structure specification was not recorded; cannot reproduce parameterization")
    required = {"name", "legs", "entry_offset", "exit_offset", "decision_offset",
                "description", "params"}
    if set(spec) != required:
        raise ValueError("incomplete or unsupported structure specification")
    values = dict(spec)
    legs = []
    for raw in values.pop("legs"):
        leg = dict(raw)
        for cls, data in ((LegSpec, leg), (ExpirySelector, leg.get("expiry")),
                          (StrikeSelector, leg.get("strike"))):
            if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
                raise ValueError("incomplete or unsupported leg/selector specification")
        leg["expiry"] = ExpirySelector(**leg["expiry"])
        leg["strike"] = StrikeSelector(**leg["strike"])
        legs.append(LegSpec(**leg))
    return Structure(**values, legs=tuple(legs))


def _date(value):
    return str(pd.Timestamp(value).date()) if value is not None and pd.notna(value) else None


def _number(value):
    return float(value) if value is not None and np.isfinite(float(value)) else None


def _contracts(legs):
    return sorted((leg["name"], leg["right"], leg["side"], float(leg["qty"]),
                   float(leg["strike"]), _date(leg["expiry"])) for leg in legs)


def comparison(row: Mapping, trade: Mapping) -> dict:
    """Drift is descriptive: a board estimate is never treated as an entry fill."""
    frozen = row.get("structure") or {}
    intended = row.get("intended_prices") or {}
    old_strike, new_strike = _number(frozen.get("strike")), _number(trade.get("strike"))
    old_expiry, new_expiry = _date(frozen.get("expiry")), _date(trade.get("expiry"))
    strike_match = None if old_strike is None or new_strike is None else bool(
        np.isclose(old_strike, new_strike, rtol=0, atol=1e-8))
    expiry_match = None if old_expiry is None or new_expiry is None else old_expiry == new_expiry
    frozen_legs, actual_legs = frozen.get("legs"), trade.get("entry_legs")
    if frozen_legs and actual_legs:
        matched = _contracts(frozen_legs) == _contracts(actual_legs)
        basis = "all_legs"
    else:
        matched = (strike_match and expiry_match
                   if strike_match is not None and expiry_match is not None else None)
        basis = "first_leg_only" if matched is not None else "unavailable"
    old_cost, new_cost = _number(intended.get("entry_cost")), _number(trade.get("entry_cost"))
    delta = new_cost - old_cost if old_cost is not None and new_cost is not None else None
    return {
        "settlement_source": "orats_quote_simulation",
        "frozen_structure": frozen,
        "realized_structure": {"strike": new_strike, "expiry": new_expiry,
                               "legs": actual_legs,
                               "entry_date": _date(trade.get("entry_date")),
                               "exit_date": _date(trade.get("exit_date"))},
        "strike_matched": strike_match, "expiry_matched": expiry_match,
        "contract_matched": matched, "contract_comparison_basis": basis,
        "intended_entry_cost": old_cost,
        "entry_cost_drift": delta,
        "entry_cost_drift_fraction": delta / abs(old_cost) if delta is not None and old_cost else None,
        "entry_quote_date": intended.get("quote_date"),
    }
