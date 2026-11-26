"""Fill model F, slice F3: fit set, fallback smile, vol source, leg values (design sections 3, 4, 7, 10).

Callers pass WHOLE listed expiries (a row's source depends on its whole expiry; memory scales with
tree rows (puts plus tree-routed calls), so the caller chunks by whole expiries). Values never depend on row order.
"""

from __future__ import annotations

import resource
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

from experiments import fair_value_carry

FIT_MIN_STRIKES = 5
WEIGHT_SPREAD_FLOOR = 0.02
REPAIRED_BID = 0.75
REPAIRED_ASK = 1.25
EXPIRY_KEY = ["ticker", "obs_date", "expiry"]
VOL_KEY = EXPIRY_KEY + ["strike"]
QUOTE_KEY = VOL_KEY + ["right"]
PROGRESS_S = 30.0
QUOTE_COLUMNS = QUOTE_KEY + ["bid", "ask", "iv", "spot", "quote_repaired", "rate", "borrow30", "borrow2yr", "divAmt", "divDate"]
VOL_COLUMNS = VOL_KEY + ["smvVol"]
OUT_COLUMNS = ["ticker", "obs_date", "expiry", "strike", "right", "status", "V", "vol", "source",
               "rate", "b", "carry_fallback", "dividend_fallback", "dividend_applied", "clamp_side", "repaired", "eff_bid", "eff_ask"]


class FairValueRefusal(ValueError):
    code = "EXPERIMENT_IDENTITY_CONFLICT"

    def __init__(self, reason, **extra):
        self.details = {"reason": reason, **extra}
        super().__init__(f"{self.code}: {reason}")


def _log(message: str) -> None:
    print(f"[fv_legs] {message}", file=sys.stderr, flush=True)


def _peak_rss_kb() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def float_array(values) -> np.ndarray:
    """Float array; NaN, None and pd.NA all become NaN."""
    return pd.Series(values).astype("Float64").to_numpy(dtype=float, na_value=np.nan)


def _validate(quotes: pd.DataFrame, vols: pd.DataFrame) -> None:
    for table, frame, required in (("quotes", quotes, QUOTE_COLUMNS), ("vols", vols, VOL_COLUMNS)):
        missing = sorted(set(required) - set(frame.columns))
        if missing:
            raise FairValueRefusal("column_missing", table=table, columns=missing)
    if quotes.duplicated(QUOTE_KEY).any():
        raise FairValueRefusal("duplicate_quote_key")
    if vols.duplicated(VOL_KEY).any():
        raise FairValueRefusal("duplicate_vol_key")
    if not quotes["right"].isin(["C", "P"]).all():
        raise FairValueRefusal("right_invalid")
    rep = quotes["quote_repaired"]
    is_bool = pd.api.types.is_bool_dtype(rep.dtype) or rep.map(lambda v: isinstance(v, (bool, np.bool_))).all()
    if rep.isna().any() or not is_bool:
        raise FairValueRefusal("quote_repaired_invalid")


def effective_quote(bid, ask, repaired):
    """(lo, hi): a repaired row (stored bid = ask = P) gets [0.75 P, 1.25 P]; others keep bid, ask."""
    bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    rep = np.asarray(repaired, dtype=bool)
    mid = (bid + ask) / 2.0
    return np.where(rep, REPAIRED_BID * mid, bid), np.where(rep, REPAIRED_ASK * mid, ask)


def fit_smile(strike, k, iv, rel_spread):
    """(coef, k_min, k_max) of the weighted quadratic iv(k), or None when no valid fit exists."""
    strike, k, iv, rel_spread = (np.asarray(a, dtype=float) for a in (strike, k, iv, rel_spread))
    if np.unique(strike).size < FIT_MIN_STRIKES or not (np.any(k < 0) and np.any(k > 0)):
        return None
    try:
        coef = np.polyfit(k, iv, 2, w=1.0 / np.maximum(rel_spread, WEIGHT_SPREAD_FLOOR))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(coef)):
        return None
    k_min, k_max = float(k.min()), float(k.max())
    c, b, _ = coef
    vertex = [-b / (2.0 * c)] if c > 0 and k_min <= -b / (2.0 * c) <= k_max else []
    if np.any(np.polyval(coef, [k_min, k_max] + vertex) <= 0):
        return None
    return coef, k_min, k_max


def smile_vol(fit, k):
    """Smile vol at k, edge value held outside the fitted range; non-finite k gives NaN."""
    coef, k_min, k_max = fit
    k = np.asarray(k, dtype=float)
    return np.where(np.isfinite(k), np.polyval(coef, np.clip(k, k_min, k_max)), np.nan)


def _inputs(quotes: pd.DataFrame, vols: pd.DataFrame) -> dict:
    """Float arrays and masks per quote row, in input order."""
    n = len(quotes)
    c = {name: float_array(quotes[name]) for name in ("strike", "spot", "bid", "ask", "iv", "rate")}
    days = (pd.to_datetime(quotes["expiry"]) - pd.to_datetime(quotes["obs_date"])).dt.days
    c["T"] = np.maximum(float_array(days), 0.0) / 365.0
    left = quotes[VOL_KEY].reset_index(drop=True)
    left["_pos"] = np.arange(n)
    merged = left.merge(vols[VOL_COLUMNS], on=VOL_KEY, how="left", sort=False)
    smv = np.full(n, np.nan)
    smv[merged["_pos"].to_numpy()] = float_array(merged["smvVol"])
    c["smv"] = np.where(np.isfinite(smv) & (smv > 0), smv, np.nan)
    right = quotes["right"].to_numpy()
    c["put"], c["call"] = right == "P", right == "C"
    c["rep"] = quotes["quote_repaired"].to_numpy(dtype=bool)
    s, spot, b, a, iv = c["strike"], c["spot"], c["bid"], c["ask"], c["iv"]
    with np.errstate(all="ignore"):
        mid = (b + a) / 2.0
        otm = (c["put"] & (s <= spot)) | (c["call"] & (s > spot))
        c["fit"] = (np.isfinite(s) & (s > 0) & np.isfinite(spot) & (spot > 0) & otm
                    & np.isfinite(iv) & (iv > 0) & np.isfinite(b) & np.isfinite(a)
                    & (b >= 0) & (b <= a) & (mid > 0) & ~c["rep"])
        c["rel"] = (a - b) / mid
    return c


def _expiry_vols(c: dict, idx: np.ndarray):
    """(source, vol per row) for one listed expiry, given by its row positions idx."""
    rows, T = len(idx), c["T"][idx[0]]
    if np.isnan(T):
        return None, np.full(rows, np.nan)
    if T == 0:
        return "intrinsic", np.full(rows, np.nan)
    fit, smv = c["fit"][idx], c["smv"][idx]
    if fit.any() and np.isfinite(smv[fit]).all():
        return "smvvol", smv
    strike = c["strike"][idx]
    with np.errstate(all="ignore"):
        k = np.log(strike / c["spot"][idx]) / np.sqrt(T)
    sel = np.flatnonzero(fit)
    sel = sel[np.lexsort((c["put"][idx][sel], strike[sel]))]  # fixed (strike, right) order
    fitted = fit_smile(strike[sel], k[sel], c["iv"][idx][sel], c["rel"][idx][sel])
    return "smile", (np.full(rows, np.nan) if fitted is None else smile_vol(fitted, k))


def leg_values(quotes: pd.DataFrame, vol_table: pd.DataFrame) -> pd.DataFrame:
    """Leg table with the input index and order. Pass WHOLE listed expiries; inputs are not mutated."""
    _validate(quotes, vol_table)
    started, n = time.monotonic(), len(quotes)
    c = _inputs(quotes, vol_table)
    gid = quotes.groupby(EXPIRY_KEY, sort=False, dropna=False).ngroup().to_numpy()
    order = np.argsort(gid, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(gid[order])) + 1) if n else []
    _log(f"start rows={n} expiries={len(groups)} peak_rss_kb={_peak_rss_kb()}")

    vol_in, source = np.full(n, np.nan), np.full(n, None, dtype=object)
    counts, last = Counter(), started
    for done, idx in enumerate(groups, 1):
        src, vol = _expiry_vols(c, idx)
        vol_in[idx], source[idx] = vol, src
        counts[src] += 1
        now = time.monotonic()
        if now - last >= PROGRESS_S:
            last, rate = now, done / (now - started)
            _log(f"expiries {done}/{len(groups)} rate={rate:.2f}/s eta={(len(groups) - done) / rate:.0f}s")

    with np.errstate(all="ignore"):
        bad = (~np.isfinite(c["bid"]) | ~np.isfinite(c["ask"]) | (c["bid"] < 0) | (c["ask"] <= 0)
               | (c["bid"] > c["ask"]))
        rate_unknown = (c["T"] > 0) & ~np.isfinite(c["rate"])
    priced = ~bad & ~rate_unknown
    put_rows, call_rows = priced & c["put"], priced & c["call"]
    run = priced & (c["T"] > 0) & np.isfinite(c["spot"]) & (c["spot"] > 0)  # carry and dividend resolved only here
    dates = pd.to_datetime(quotes["obs_date"])
    exp_days = float_array((pd.to_datetime(quotes["expiry"]) - dates).dt.days)
    div_days = float_array((pd.to_datetime(quotes["divDate"]) - dates).dt.days)
    b, carry_fb = fair_value_carry.carry_rate(exp_days, float_array(quotes["borrow30"]), float_array(quotes["borrow2yr"]), c["rate"])
    div, div_fb, div_ok = fair_value_carry.dividend(float_array(quotes["divAmt"]), div_days, exp_days, c["spot"], c["rate"])
    b, carry_fb, div_fb, div_ok = np.where(run, b, np.nan), carry_fb & run, div_fb & run, div_ok & run
    div, div_years = np.where(div_ok, div, 0.0), np.where(div_ok, div_days / 365.0, 0.0)
    _log(f"sources per expiry: smvvol={counts['smvvol']} smile={counts['smile']} "
         f"intrinsic={counts['intrinsic']} none={counts[None]}; pricing {put_rows.sum()} puts, {call_rows.sum()} calls; "
         f"carry_fallback={carry_fb.sum()} dividend_fallback={div_fb.sum()} dividend_applied={div_ok.sum()}")
    model = np.full(n, np.nan)
    model[priced] = fair_value_carry.value(c["spot"][priced], c["strike"][priced], c["T"][priced], vol_in[priced],
                                           c["rate"][priced], b[priced], div[priced], div_years[priced], c["put"][priced])

    lo, hi = effective_quote(c["bid"], c["ask"], c["rep"])
    ok = priced & np.isfinite(model)
    with np.errstate(all="ignore"):
        value = np.where(ok, np.clip(model, lo, hi), np.nan)
        side = np.where(model < lo, "bid", np.where(model > hi, "ask", "none")).astype(object)
    side[~ok] = None
    status = np.where(bad, "unpriceable", np.where(
        rate_unknown, "rate_unknown", np.where(ok, "ok", "no_fair_value"))).astype(object)
    passthrough = {name: quotes[name].to_numpy() for name in ("ticker", "obs_date", "expiry", "right")}
    out = pd.DataFrame(
        {**passthrough, "strike": c["strike"],
         "status": pd.Series(status, index=quotes.index, dtype=object), "V": value,
         "vol": np.where(ok, vol_in, np.nan),
         "source": pd.Series(source, index=quotes.index, dtype=object), "rate": c["rate"],
         "b": b, "carry_fallback": carry_fb, "dividend_fallback": div_fb, "dividend_applied": div_ok,
         "clamp_side": pd.Series(side, index=quotes.index, dtype=object),
         "repaired": c["rep"], "eff_bid": lo, "eff_ask": hi},
        index=quotes.index, columns=OUT_COLUMNS,
    )
    _log(f"end status={dict(Counter(status.tolist()))} elapsed={time.monotonic() - started:.1f}s "
         f"peak_rss_kb={_peak_rss_kb()}")
    return out
