"""Fill model F, slice F4: concession, unknown and repaired rules, sensitivities, structure entry and exit
(design sections 5-7, 10).

Callers pass legs they resolved. Entry and exit are separate calls, so entry values never read exit-date rows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.fills import MIN_MEANINGFUL_COST
from experiments import fair_value_legs as fvl
from experiments.fair_value_legs import OUT_COLUMNS, FairValueRefusal

C_BASE = 0.25
C_SPREAD = 0.50
SPREAD_SCALE = 0.50
C_THIN_SIZE = 0.15
C_THIN_OI = 0.10
THIN_SIZE_BELOW = 10
THIN_OI_BELOW = 100
UNKNOWN_THIN = 0.5
GUARD_FILL_FRACTION = 0.25
VARIANTS = ("unknown0", "unknown1", "midfloor")
LIQUIDITY_COLUMNS = ["open_interest", "bid_size", "ask_size"]
LEG_STATUSES = ("ok", "unpriceable", "no_fair_value", "rate_unknown")
FALLBACK_STATUSES = ("no_fair_value", "rate_unknown")
FILL_COLUMNS = ["s", "thin_size_buy", "thin_size_sell", "thin_oi", "unknown_bid_size", "unknown_ask_size",
                "unknown_oi", "c_buy", "c_sell", "fill_buy", "fill_sell"] + [
    f"fill_{side}_{v}" for v in VARIANTS for side in ("buy", "sell")]
LEG_REQUIRED = OUT_COLUMNS + FILL_COLUMNS + ["side", "qty"]
_NUM_INPUTS = ["V", "eff_bid", "eff_ask", "s", "thin_size_buy", "thin_size_sell", "thin_oi"]
_BOOL_INPUTS = ["unknown_ask_size", "unknown_bid_size", "unknown_oi"]


@dataclass(frozen=True)
class EntryResult:
    outcome: str
    skip_reason: str | None
    debit_mid: float | None
    debit_fair: float | None
    debit_F: float | None
    spread_sum: float | None
    sensitivity_debits: dict
    leg_flags: tuple


@dataclass(frozen=True)
class ExitResult:
    outcome: str
    proceeds_F: float | None
    sensitivity_proceeds: dict
    exit_fair_mid: bool
    exit_negative: bool
    leg_flags: tuple


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
    """Concession and fills for the base and each sensitivity. Shared by leg_fills and the exit fallback."""
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


def _f(legs: pd.DataFrame, name: str) -> np.ndarray:
    return legs[name].to_numpy(dtype=float)


def _check_legs(legs: pd.DataFrame) -> None:
    missing = sorted(set(LEG_REQUIRED) - set(legs.columns))
    if missing:
        raise FairValueRefusal("column_missing", table="legs", columns=missing)
    if len(legs) == 0 or not legs["side"].isin(["buy", "sell"]).all():
        raise FairValueRefusal("leg_invalid")
    try:
        qty = fvl.float_array(legs["qty"])
    except (TypeError, ValueError):
        raise FairValueRefusal("leg_invalid") from None
    with np.errstate(invalid="ignore"):
        if not (np.isfinite(qty) & (qty > 0)).all():
            raise FairValueRefusal("leg_invalid")
    if not legs["status"].isin(LEG_STATUSES).all():
        raise FairValueRefusal("leg_invalid")


def _leg_flags(legs: pd.DataFrame, at_exit: bool, fallback: np.ndarray) -> tuple:
    """Flags for the side HIT: entry buys hit the ask, sells the bid; exit is flipped."""
    def col(name):
        return legs[name].to_numpy()

    hit_ask = (col("side") == "buy") != at_exit
    thin_size = np.where(hit_ask, col("thin_size_buy"), col("thin_size_sell"))
    unknown_size = np.where(hit_ask, col("unknown_ask_size"), col("unknown_bid_size"))
    return tuple(
        {"status": str(st), "clamp_side": cs, "repaired": bool(rp), "thin_size": float(ts),
         "thin_oi": float(to), "unknown_size": bool(us), "unknown_oi": bool(uo), "exit_fair_mid": bool(fb)}
        for st, cs, rp, ts, to, us, uo, fb in zip(
            col("status"), col("clamp_side"), col("repaired"), thin_size, col("thin_oi"),
            unknown_size, col("unknown_oi"), fallback))


def _debit(qty, buy, buy_px, sell_px) -> float:
    return float(np.sum(qty * np.where(buy, buy_px, -sell_px)))


def _proceeds(qty, buy, buy_px, sell_px) -> float:
    """Opening buys close at the sell fill; opening sells close at the buy fill."""
    return float(np.sum(qty * np.where(buy, sell_px, -buy_px)))


def _guard(debit_mid: float, debit_fair: float, debit_F: float, spread_sum: float) -> str | None:
    if debit_mid <= MIN_MEANINGFUL_COST or debit_fair <= MIN_MEANINGFUL_COST:
        return "not_a_debit"
    if debit_F - debit_fair > GUARD_FILL_FRACTION * debit_fair:
        return "fill_over_fair"
    if spread_sum > debit_F:
        return "spread_over_debit"
    return None


def structure_entry(legs: pd.DataFrame) -> EntryResult:
    """Entry debits and the guard (design section 6). Legs carry the opening side and quantity."""
    _check_legs(legs)
    flags = _leg_flags(legs, at_exit=False, fallback=np.zeros(len(legs), dtype=bool))
    if (legs["status"] != "ok").any():
        return EntryResult("undetermined", None, None, None, None, None, {v: None for v in VARIANTS}, flags)
    qty, buy = fvl.float_array(legs["qty"]), (legs["side"] == "buy").to_numpy()
    lo, hi, mid = _f(legs, "eff_bid"), _f(legs, "eff_ask"), (_f(legs, "eff_bid") + _f(legs, "eff_ask")) / 2.0
    debit_mid = _debit(qty, buy, mid, mid)
    debit_fair = _debit(qty, buy, _f(legs, "V"), _f(legs, "V"))
    debit_F = _debit(qty, buy, _f(legs, "fill_buy"), _f(legs, "fill_sell"))
    spread_sum = float(np.sum(qty * (hi - lo)))
    sens = {v: _debit(qty, buy, _f(legs, f"fill_buy_{v}"), _f(legs, f"fill_sell_{v}")) for v in VARIANTS}
    reason = _guard(debit_mid, debit_fair, debit_F, spread_sum)
    return EntryResult("skip" if reason else "enter", reason, debit_mid, debit_fair, debit_F, spread_sum,
                       sens, flags)


def structure_exit(legs: pd.DataFrame) -> ExitResult:
    """Exit proceeds at F's fills, with no guard (design section 7). Legs are exit-date rows, opening side."""
    _check_legs(legs)
    fb = legs["status"].isin(FALLBACK_STATUSES).to_numpy()
    exit_fair_mid = bool(fb.any())
    flags = _leg_flags(legs, at_exit=True, fallback=fb)
    if (legs["status"] == "unpriceable").any():
        return ExitResult("unpriceable", None, {v: None for v in VARIANTS}, exit_fair_mid, False, flags)
    qty, buy = fvl.float_array(legs["qty"]), (legs["side"] == "buy").to_numpy()
    # Fallback legs take V = mid and pay the concession; their stored fills are ignored.
    frame = legs[_NUM_INPUTS + _BOOL_INPUTS].copy()
    frame["V"] = np.where(fb, (_f(legs, "eff_bid") + _f(legs, "eff_ask")) / 2.0, _f(legs, "V"))
    fresh = _fills(frame)

    def pick(name):
        return np.where(fb, fresh[name], _f(legs, name))

    base = _proceeds(qty, buy, pick("fill_buy"), pick("fill_sell"))
    sens = {v: _proceeds(qty, buy, pick(f"fill_buy_{v}"), pick(f"fill_sell_{v}")) for v in VARIANTS}
    return ExitResult("ok", base, sens, exit_fair_mid, base < 0, flags)


def return_on_debit(proceeds_F: float, debit_F: float) -> float:
    """(proceeds_F - debit_F) / debit_F, unclipped. Precondition: debit_F is from an entry with outcome "enter"."""
    return (proceeds_F - debit_F) / debit_F
