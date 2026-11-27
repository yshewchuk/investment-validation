"""Fill model F, slice F6c part 2: applies the fitted model A (mid-anchored) constants to leg fills, structure entry and exit.

Not in fvf_id; F's own code, id and outputs are unchanged. No experiment uses it yet. It stores nothing.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pandas as pd

from engine.fills import MIN_MEANINGFUL_COST
from experiments import fair_value_constants as fvk, fair_value_fills as fvf, fair_value_legs as fvl
from experiments.fair_value_legs import FairValueRefusal

PRIMARY = "primary"
BINDING_COLUMNS = ["constants_id", "variant"]


def load_constants(document, expected_id, variant) -> dict:
    """Constants for ``variant`` from a pinned, self-consistent document. Refusals come in the design section 9 order."""
    if not isinstance(document, dict) or not isinstance(document.get("constants"), dict):
        raise FairValueRefusal("constants_invalid")
    constants = document["constants"]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= sys.float_info.max
               for v in (constants.get(name) for name in fvk.FITTED)):
        raise FairValueRefusal("constants_invalid")
    rebuilt = fvk.constants_document(fvk.primary_constants(*[constants[name] for name in fvk.FITTED]))
    if {k: v for k, v in document.items() if k != "constants_id"} != {k: v for k, v in rebuilt.items() if k != "constants_id"}:
        raise FairValueRefusal("constants_invalid")
    if not (rebuilt["constants_id"] == expected_id and document.get("constants_id") == rebuilt["constants_id"]):
        raise FairValueRefusal("constants_id_mismatch")
    if variant != PRIMARY and (not isinstance(variant, str) or variant not in fvk.SPREAD_SENSITIVITIES):
        raise FairValueRefusal("variant_unknown")
    out = dict(rebuilt["constants"])
    if variant != PRIMARY:
        out["C_SPREAD"] = fvk.SPREAD_SENSITIVITIES[variant]
    return out


def _concession(legs: pd.DataFrame, k: dict, buy: bool, fixed, spread_term) -> np.ndarray:
    """Model A concession for one side; unknown liquidity terms take ``fixed`` (None keeps the stored thin values)."""
    size_col, unknown_col = ("thin_size_buy", "unknown_ask_size") if buy else ("thin_size_sell", "unknown_bid_size")
    size, oi = fvf._f(legs, size_col), fvf._f(legs, "thin_oi")
    if fixed is not None:
        size = np.where(legs[unknown_col].to_numpy(dtype=bool), fixed, size)
        oi = np.where(legs["unknown_oi"].to_numpy(dtype=bool), fixed, oi)
    return fvk.concession(k["C_BASE"], k["C_SPREAD"], k["C_THIN_SIZE"], k["C_THIN_OI"], spread_term, size, oi)


def _model_fills(legs: pd.DataFrame, k: dict) -> dict:
    """Model A concessions and fills from the effective quote and liquidity, with no V. A NaN spread gives NaN fills."""
    lo, hi = fvf._f(legs, "eff_bid"), fvf._f(legs, "eff_ask")
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    with np.errstate(invalid="ignore"):
        spread_term = np.minimum(fvf._f(legs, "s") / k["SPREAD_SCALE"], 1.0)
        out = {"c_buy": _concession(legs, k, True, None, spread_term), "c_sell": _concession(legs, k, False, None, spread_term)}
        out["fill_buy"], out["fill_sell"] = mid + out["c_buy"] * half, mid - out["c_sell"] * half
        for v, fixed in (("unknown0", 0.0), ("unknown1", 1.0)):
            out[f"fill_buy_{v}"] = mid + _concession(legs, k, True, fixed, spread_term) * half
            out[f"fill_sell_{v}"] = mid - _concession(legs, k, False, fixed, spread_term) * half
        out["fill_buy_midfloor"] = np.maximum(out["fill_buy"], mid)
        out["fill_sell_midfloor"] = np.minimum(out["fill_sell"], mid)
    return out


def leg_fills(quotes, vol_table, document, expected_id, variant) -> pd.DataFrame:
    """F's leg table with model A fills (NaN off ok rows) and the pin and variant appended. Refuses before pricing."""
    k = load_constants(document, expected_id, variant)
    table = fvf.leg_fills(quotes, vol_table)
    stale = (table["status"] != "ok").to_numpy()
    new = {name: np.where(stale, np.nan, value) if name.startswith("fill_") else value for name, value in _model_fills(table, k).items()}
    print(f"[fv_apply] constants_id={expected_id} variant={variant} rows={len(table)}", file=sys.stderr, flush=True)
    return table.assign(**new, constants_id=expected_id, variant=variant)


def _guard(debit_mid: float, debit_fair: float, debit: float, spread_sum: float) -> str | None:
    if debit_mid <= MIN_MEANINGFUL_COST or debit_fair <= MIN_MEANINGFUL_COST:
        return "not_a_debit"
    if debit - debit_mid > fvf.GUARD_FILL_FRACTION * debit_mid:
        return "fill_over_mid"
    if spread_sum > debit:
        return "spread_over_debit"
    return None


def structure_entry(legs: pd.DataFrame) -> fvf.EntryResult:
    """Entry debits from the model A fill columns, with the model A guard (design section 6). Entry reads no pin."""
    got = fvf.structure_entry(legs)
    if got.outcome == "undetermined":
        return got
    reason = _guard(got.debit_mid, got.debit_fair, got.debit_F, got.spread_sum)
    return dataclasses.replace(got, outcome="skip" if reason else "enter", skip_reason=reason)


def structure_exit(legs: pd.DataFrame, document, expected_id, variant) -> fvf.ExitResult:
    """Exit proceeds under model A; fallback legs take the model A fills at their effective mid (design section 7)."""
    k = load_constants(document, expected_id, variant)
    fvf._check_legs(legs)
    unbound = [c for c in BINDING_COLUMNS if c not in legs.columns or legs[c].isna().any()]
    if unbound:
        raise FairValueRefusal("legs_unbound", columns=unbound)
    if not ((legs["constants_id"] == expected_id).all() and (legs["variant"] == variant).all()):
        raise FairValueRefusal("legs_constants_mismatch")
    fb = legs["status"].isin(fvf.FALLBACK_STATUSES).to_numpy()
    flags = fvf._leg_flags(legs, at_exit=True, fallback=fb)
    if (legs["status"] == "unpriceable").any():
        return fvf.ExitResult("unpriceable", None, {v: None for v in fvf.VARIANTS}, bool(fb.any()), False, flags)
    qty, buy = fvl.float_array(legs["qty"]), (legs["side"] == "buy").to_numpy()
    fresh = _model_fills(legs, k)

    def pick(name):
        return np.where(fb, fresh[name], fvf._f(legs, name))

    base = fvf._proceeds(qty, buy, pick("fill_buy"), pick("fill_sell"))
    sens = {v: fvf._proceeds(qty, buy, pick(f"fill_buy_{v}"), pick(f"fill_sell_{v}")) for v in fvf.VARIANTS}
    return fvf.ExitResult("ok", base, sens, bool(fb.any()), base < 0, flags)
