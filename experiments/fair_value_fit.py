"""Fill model F, slice F6a of design section 9: validate fills, price them at the quote snapshot's own spot (user decision B1;
60-minute window), write fit_receipt.json. The fit is slice F6b. No F constant changes; schema: experiments/README.md."""
import argparse
import hashlib
import io
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from experiments import fair_value_carry as fvc, fair_value_fills as fvf, fair_value_inputs as fvi, fair_value_legs as fvl, fair_value_run as fvr

NEVER_NULL = ["fill_time", "snapshot_time", "ticker", "expiry", "strike", "right", "side", "qty", "price", "quote_repaired"]


def _check(reason: str, column: str, mask) -> None:
    if n := int(np.sum(mask)):
        raise fvi.FairValueRefusal(reason, "rows fail the rule", column=column, rows=n)


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    required = NEVER_NULL + ["bid", "ask", "bid_size", "ask_size", "open_interest", "spot", "vol_date", "smvVol", "rates_date",
                             "iRate5wk", "iRateLt", "borrow30", "borrow2yr", "divAmt", "divDate"]
    if missing := [c for c in required if c not in frame.columns]:
        raise fvi.FairValueRefusal("column_missing", "fills lack required columns", columns=missing)
    if len(frame) == 0:
        raise fvi.FairValueRefusal("fills_empty", "the fills file has no rows")
    for c in NEVER_NULL:
        _check("value_invalid", c, frame[c].isna())
    t = pd.DataFrame(index=frame.index)
    for c, name in (("fill_time", "fill"), ("snapshot_time", "snap")):  # a trailing offset and a time of day are required
        ok = frame[c].str.contains(r"[T ]\d\d:\d\d.*(?:Z|[+-]\d\d:?\d\d)$")
        t[name] = pd.to_datetime(frame[c], format="ISO8601", utc=True, errors="coerce").where(ok)
        _check("value_invalid", c, t[name].isna())
    t["order"] = t["fill"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    for c in ["expiry", "vol_date", "rates_date", "divDate"]:
        t[c] = pd.to_datetime(frame[c], format="%Y-%m-%d", errors="coerce")
        _check("value_invalid", c, frame[c].notna() & t[c].isna())
    _check("value_invalid", "expiry", t["expiry"] < t["order"])
    for c in ["strike", "qty", "price", "bid", "ask", "bid_size", "ask_size", "open_interest", "spot", "smvVol", "iRate5wk",
              "iRateLt", "borrow30", "borrow2yr", "divAmt"]:
        vals = t[c] = pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
        strict, nonneg = c in ("strike", "qty", "price"), c in ("bid_size", "ask_size", "open_interest")
        rule = np.isfinite(vals) & ((vals > 0) if strict else (vals >= 0) if nonneg else True)
        _check("value_invalid", c, frame[c].notna().to_numpy() & ~rule)
    for c, allowed in (("right", ("C", "P")), ("side", ("buy", "sell"))):
        _check("value_invalid", c, ~frame[c].isin(allowed))
    _check("value_invalid", "quote_repaired", ~(rep := frame["quote_repaired"].str.lower()).isin(["true", "false"]))
    t["put"], t["buy"], t["repaired"] = (frame["right"] == "P").to_numpy(), (frame["side"] == "buy").to_numpy(), (rep == "true").to_numpy()
    for c in ("vol_date", "rates_date"):
        _check("not_causal", c, t[c] >= t["order"])
    _check("not_causal", "vol_date", t["smvVol"].notna() & t["vol_date"].isna())
    carry = t[["iRate5wk", "iRateLt", "borrow30", "borrow2yr", "divAmt", "divDate"]].notna().any(axis=1)
    _check("not_causal", "rates_date", carry & t["rates_date"].isna())
    t["smvVol"] = np.where(t["smvVol"] > 0, t["smvVol"], np.nan)
    if np.isfinite(t["smvVol"]).any() and not fvi.VOL_MEDIAN_BOUNDS[0] < float(np.nanmedian(t["smvVol"])) < fvi.VOL_MEDIAN_BOUNDS[1]:
        raise fvi.FairValueRefusal("vol_unit", "median positive smvVol is outside the bounds")
    return t


def _price(t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n, counts = len(t), {"rows": len(t)}
    for name, cols, bounds in (("rate_cells_out_of_range", ["iRate5wk", "iRateLt"], fvi.RATE_BOUNDS),
                               ("borrow_cells_out_of_range", ["borrow30", "borrow2yr"], fvi.BORROW_BOUNDS)):
        counts[name] = sum(int((t[c].notna() & ~t[c].between(*bounds)).sum()) for c in cols)
        for c in cols:
            t[c] = t[c].mask(t[c].notna() & ~t[c].between(*bounds))
    sel = (t["rates_date"] >= t["order"] - pd.Timedelta(days=fvi.RATE_LOOKBACK_DAYS)).to_numpy()
    rates = t.loc[sel, ["iRate5wk", "iRateLt", "borrow30", "borrow2yr", "divAmt", "divDate", "order"]].rename(columns={"order": "obs_date"})
    rates = rates.assign(annIdiv=np.nan, annActDiv=np.nan, assetType=np.nan, ticker=np.flatnonzero(sel).astype(str))
    rows = pd.DataFrame({"ticker": np.arange(n).astype(str), "obs_date": t["order"].to_numpy(), "expiry": t["expiry"].to_numpy()})
    look = fvi.lookup_inputs(rows, rates)
    with np.errstate(all="ignore"):
        spot, rate, buy, price = t["spot"].to_numpy(), look["rate"].to_numpy(dtype=float), t["buy"].to_numpy(), t["price"].to_numpy()
        exp_days = (t["expiry"] - t["order"]).dt.days.to_numpy(dtype=float)
        div_days = fvl.float_array((look["divDate"] - t["order"]).dt.days)
        years = np.maximum(exp_days, 0.0) / 365.0
        b, _ = fvc.carry_rate(exp_days, look["borrow30"], look["borrow2yr"], rate)
        div, _, applied = fvc.dividend(look["divAmt"], div_days, exp_days, spot, rate)
        model = fvc.value(spot, t["strike"], years, t["smvVol"], rate, b, div, np.where(applied, div_days / 365.0, 0.0), t["put"])
        lo, hi = fvl.effective_quote(t["bid"], t["ask"], t["repaired"])
        V = np.clip(model, lo, hi)
        denom = np.where(buy, hi - V, V - lo)
        c = np.where(buy, (price - V) / denom, (V - price) / denom)
        s, size, oi = (hi - lo) / ((hi + lo) / 2.0), np.where(buy, t["ask_size"], t["bid_size"]), t["open_interest"].to_numpy()
        masks = ((t["fill"] - t["snap"]).abs().to_numpy() > pd.Timedelta(minutes=60), ~fvi.usable_quote(t["bid"], t["ask"]),
                 (years > 0) & ~np.isfinite(rate), ~np.isfinite(model) | ~(np.isfinite(spot) & (spot > 0)), denom < 0.005)
        left = np.ones(n, dtype=bool)
        for name, mask in zip(("outside_window", "unpriceable", "rate_unknown", "no_fair_value", "small_denominator"), masks):
            counts[name], left = int((left & mask).sum()), left & ~mask
        ts, to = fvf._thin(size, fvf.THIN_SIZE_BELOW), fvf._thin(oi, fvf.THIN_OI_BELOW)
        group = np.select([np.isnan(size) | np.isnan(oi), (ts == 1) & (to == 1), ts == 1, to == 1],
                          ["unknown", "both", "size_only", "oi_only"], default="unflagged")
        d = pd.DataFrame({"c": c, "x": np.minimum(s / fvf.SPREAD_SCALE, 1.0), "thin_size": ts, "thin_oi": to, "group": group,
                          "order": t["order"].to_numpy()})[left]
    return d, {**counts, "used": len(d), **{g: int((d["group"] == g).sum()) for g in ("unflagged", "size_only", "oi_only", "both", "unknown")}}


def price_fills(blob: bytes) -> tuple[pd.DataFrame, dict]:
    started = time.monotonic()
    logging.info("[fv_fit] load+price start")
    try:
        frame = pd.read_csv(io.BytesIO(blob), dtype=str, keep_default_na=False, na_values=[""])
    except ValueError as err:  # parse and decode errors; only the exception's type name is kept
        raise fvi.FairValueRefusal("fills_unreadable", "the fills file is not readable CSV", error_type=type(err).__name__) from None
    d, counts = _price(_typed(frame))
    logging.info("[fv_fit] load+price end: %.1f s, peak rss %.0f MB; counts %s", time.monotonic() - started,
                 fvr._peak_mb(), json.dumps(counts, sort_keys=True))
    dates = d["order"].dt.strftime("%Y-%m-%d")
    receipt = {"input_sha256": hashlib.sha256(blob).hexdigest(), "base_fvf_id": fvr.fvf_id(), "window_minutes": 60,
               "fit_source_sha256": hashlib.sha256(fvi._read_capped(Path(__file__), 1 << 20, "fair_value_fit")).hexdigest(), "counts": counts,
               "first_order_date": dates.min() if len(d) else None, "last_order_date": dates.max() if len(d) else None}
    return d, receipt


def run_fit(fills_path, out_dir) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise fvi.FairValueRefusal("out_dir_exists", "the output directory already exists")
    receipt = price_fills(fvi._read_capped(Path(fills_path), 16 << 20, "fills"))[1]
    with tempfile.TemporaryDirectory(dir=out_dir.parent, prefix=".fit-") as hold:
        Path(hold, "out").mkdir()
        (Path(hold, "out") / "fit_receipt.json").write_bytes(json.dumps(receipt, sort_keys=True, allow_nan=False, indent=1).encode())
        Path(hold, "out").rename(out_dir)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fill model F pricing, slice F6a; one JSON line on stdout.")
    parser.add_argument("--fills", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    try:
        receipt = run_fit(args.fills, args.out_dir)
    except (OSError, ValueError) as err:  # refusals, and read or write errors; repr(err) is never printed
        refusal = err if hasattr(err, "details") else fvi.FairValueRefusal("input_unreadable", "input unreadable", error_type=type(err).__name__)
        print(json.dumps({"refusal": refusal.code, "details": refusal.details}, default=str))
        return 2
    print(json.dumps({"input_sha256": receipt["input_sha256"], "counts": receipt["counts"]}, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
