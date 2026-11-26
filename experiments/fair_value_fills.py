"""Fill model F, slice F4a: concession, unknown and repaired rules, sensitivities, per-quote leg fills
(design sections 5, 7, 10). Structure entry, guard and exit are slice F4b.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments import fair_value_legs as fvl
from experiments.fair_value_legs import FairValueRefusal

C_BASE = 0.25
C_SPREAD = 0.50
SPREAD_SCALE = 0.50
C_THIN_SIZE = 0.15
C_THIN_OI = 0.10
THIN_SIZE_BELOW = 10
THIN_OI_BELOW = 100
UNKNOWN_THIN = 0.5
VARIANTS = ("unknown0", "unknown1", "midfloor")
LIQUIDITY_COLUMNS = ["open_interest", "bid_size", "ask_size"]
FILL_COLUMNS = ["s", "thin_size_buy", "thin_size_sell", "thin_oi", "unknown_bid_size", "unknown_ask_size",
                "unknown_oi", "c_buy", "c_sell", "fill_buy", "fill_sell"] + [
    f"fill_{side}_{v}" for v in VARIANTS for side in ("buy", "sell")]
_NUM_INPUTS = ["V", "eff_bid", "eff_ask", "s", "thin_size_buy", "thin_size_sell", "thin_oi"]
_BOOL_INPUTS = ["unknown_ask_size", "unknown_bid_size", "unknown_oi"]


def _concession(s, thin_size, thin_oi):
    with np.errstate(invalid="ignore"):
        raw = C_BASE + C_SPREAD * np.minimum(s / SPREAD_SCALE, 1.0) + C_THIN_SIZE * thin_size + C_THIN_OI * thin_oi
        return np.minimum(raw, 1.0)


def _resolve(a: dict, u: dict, fixed):
    """(c_buy, c_sell, fill_buy, fill_sell); unknown terms take `fixed`, None keeps 0.5."""
    tb, ts, to = a["thin_size_buy"], a["thin_size_sell"], a["thin_oi"]
    if fixed is not None:
        tb = np.where(u["unknown_ask_size"], fixed, tb)
        ts = np.where(u["unknown_bid_size"], fixed, ts)
        to = np.where(u["unknown_oi"], fixed, to)
    c_buy, c_sell = _concession(a["s"], tb, to), _concession(a["s"], ts, to)
    V, lo, hi = a["V"], a["eff_bid"], a["eff_ask"]
    return c_buy, c_sell, V + c_buy * (hi - V), V - c_sell * (V - lo)


def _fills(f: pd.DataFrame) -> dict:
    """Concession and fills for the base and each sensitivity."""
    a = {k: f[k].to_numpy(dtype=float) for k in _NUM_INPUTS}
    u = {k: f[k].to_numpy(dtype=bool) for k in _BOOL_INPUTS}
    c_buy, c_sell, fill_buy, fill_sell = _resolve(a, u, None)
    out = {"c_buy": c_buy, "c_sell": c_sell, "fill_buy": fill_buy, "fill_sell": fill_sell}
    for v, fixed in (("unknown0", 0.0), ("unknown1", 1.0)):
        out[f"fill_buy_{v}"], out[f"fill_sell_{v}"] = _resolve(a, u, fixed)[2:]
    mid = (a["eff_bid"] + a["eff_ask"]) / 2.0
    out["fill_buy_midfloor"] = np.maximum(fill_buy, mid)
    out["fill_sell_midfloor"] = np.minimum(fill_sell, mid)
    return out


def _log(message: str) -> None:
    print(f"[fv_fills] {message}", file=sys.stderr, flush=True)


def _thin(values: np.ndarray, below: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(values), UNKNOWN_THIN, (values < below).astype(float))


def leg_fills(quotes: pd.DataFrame, vol_table: pd.DataFrame) -> pd.DataFrame:
    """Leg table (fair_value_legs.leg_values) with spread, liquidity flags, concession and fills appended."""
    missing = sorted(set(LIQUIDITY_COLUMNS) - set(quotes.columns))
    if missing:
        raise FairValueRefusal("column_missing", table="quotes", columns=missing)
    table = fvl.leg_values(quotes, vol_table)
    lo, hi = table["eff_bid"].to_numpy(dtype=float), table["eff_ask"].to_numpy(dtype=float)
    unpriceable = (table["status"] == "unpriceable").to_numpy()
    with np.errstate(all="ignore"):
        s = np.where(unpriceable, np.nan, (hi - lo) / ((lo + hi) / 2.0))
    bid_size, ask_size = fvl.float_array(quotes["bid_size"]), fvl.float_array(quotes["ask_size"])
    oi = fvl.float_array(quotes["open_interest"])
    unk_bid, unk_ask, unk_oi = np.isnan(bid_size), np.isnan(ask_size), np.isnan(oi)
    _log(f"unknown liquidity rows={len(table)} bid_size={int(unk_bid.sum())} "
             f"ask_size={int(unk_ask.sum())} open_interest={int(unk_oi.sum())}")
    cols = {"s": s, "thin_size_buy": _thin(ask_size, THIN_SIZE_BELOW),
            "thin_size_sell": _thin(bid_size, THIN_SIZE_BELOW), "thin_oi": _thin(oi, THIN_OI_BELOW),
            "unknown_bid_size": unk_bid, "unknown_ask_size": unk_ask, "unknown_oi": unk_oi}
    frame = pd.DataFrame({"V": table["V"].to_numpy(dtype=float), "eff_bid": lo, "eff_ask": hi, **cols})
    new = {**cols, **_fills(frame)}
    return table.assign(**{name: new[name] for name in FILL_COLUMNS})
