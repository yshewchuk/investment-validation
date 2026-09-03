#!/usr/bin/env python3
"""EXP-122 — the 1-DTE straddle as a model-free implied move.

Run:  python3 experiments/EXP-122_.../run.py

Stage 0 (feasibility) then Stage 1 (Q1 winner rule, Q2 convention test,
positive control). Registered in spec.yaml before any of it ran. Zero API calls
— every quote comes from the stored chains.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine import replay as R  # noqa: E402
from engine.build_trades import event_universe  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.fills import WIDE_MARKET_RATIO  # noqa: E402
from engine.replay import _clean, load_chain_index  # noqa: E402
from engine.structures import (ChainSnapshot, ExpirySelector,  # noqa: E402
                               StructureError, straddle_through)
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MAX_STRIKE_OFFSET = 0.01      # |K - F| / F, registered
BAD_QUOTE_PCT = 30.0          # registered, EXP-117's ceiling
STAGE0_GATE = 1_500


def straddle_row(snap: ChainSnapshot, max_dte: int) -> dict | None:
    """The ATM straddle at the first post-print expiry, if it is usable."""
    try:
        expiry = ExpirySelector(kind="first_post_event").select(
            snap.rows, snap.event_date, snap.session)
    except StructureError:
        return None
    at = snap.rows[snap.rows["expiry"] == expiry]
    if at.empty:
        return None
    dte = int(at["dte"].iloc[0])
    if dte > max_dte:
        return None
    spot = snap.spot_price
    calls = at[at["right"] == "C"].set_index("strike")
    puts = at[at["right"] == "P"].set_index("strike")
    both = sorted(set(calls.index) & set(puts.index))
    if not both:
        return None

    def quote(k):
        c, p = calls.loc[k], puts.loc[k]
        return (float(c["bid"]), float(c["ask"]), float(p["bid"]), float(p["ask"]))

    k0 = min(both, key=lambda k: abs(k - spot))
    cb, ca, pb, pa = quote(k0)
    if min(ca, pa) <= 0:
        return None
    fwd = k0 + 0.5 * (cb + ca) - 0.5 * (pb + pa)      # put-call parity
    k = min(both, key=lambda x: abs(x - fwd))          # straddle nearest the forward
    cb, ca, pb, pa = quote(k)
    if min(ca, pa) <= 0 or cb > ca or pb > pa:
        return None
    fwd = k + 0.5 * (cb + ca) - 0.5 * (pb + pa)
    if fwd <= 0:
        return None
    cmid, pmid = 0.5 * (cb + ca), 0.5 * (pb + pa)
    wide = (max((ca - cb) / cmid if cmid > 0 else np.inf,
                (pa - pb) / pmid if pmid > 0 else np.inf) > WIDE_MARKET_RATIO)
    return {
        "ticker": snap.ticker, "event_date": snap.event_date, "dte": dte,
        "spot": spot, "strike": k, "forward": fwd,
        "offset": abs(k - fwd) / fwd,
        "im_mid": 100.0 * (cmid + pmid) / fwd,
        "im_bid": 100.0 * (cb + pb) / fwd,
        "im_ask": 100.0 * (ca + pa) / fwd,
        "wide": bool(wide),
    }


def collect(max_dte: int) -> pd.DataFrame:
    cache = RESULTS / f"straddles_dte{max_dte}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    started = time.time()
    events = event_universe()
    plan = R.plan_events(straddle_through(), events)
    avail = R.available_chain_keys()
    f = plan.frame
    f = f[[(t, d) in avail for t, d in zip(f["ticker"], f["entry_date"])]].reset_index(drop=True)

    rows = []
    for year, block in f.groupby(f["entry_date"].dt.year, sort=True):
        idx = load_chain_index({(t, d) for t, d in zip(block["ticker"], block["entry_date"])},
                               progress_every=0)
        for pr in block.to_dict("records"):
            ch = idx.get(pr["ticker"], pr["entry_date"])
            if ch is None or ch.empty:
                continue
            ch = _clean(ch)
            if ch.empty:
                continue
            try:
                snap = ChainSnapshot(ticker=pr["ticker"], obs_date=pr["entry_date"],
                                     event_date=pr["event_date"], rows=ch,
                                     session=pr["session"])
                row = straddle_row(snap, max_dte)
            except StructureError:
                continue
            if row:
                rows.append(row | {"year": int(year)})
        del idx
        print(f"  [dte<={max_dte}] {year}: {len(rows):,} straddles, "
              f"{time.time()-started:.0f}s", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(cache, index=False)
    return out


def join_vendors(df: pd.DataFrame) -> pd.DataFrame:
    p = load_panel()[["ticker", "date", "implied_move", "or_implied"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    df = df.merge(p, left_on=["ticker", "event_date"], right_on=["ticker", "date"], how="left")
    return df


def apply_exclusions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    counts, cur = {}, df
    for label, mask in [
        ("strike more than 1% from the forward", cur["offset"] <= MAX_STRIKE_OFFSET),
        ("a leg quoted through a wide market", ~cur["wide"]),
        ("implied move above the 30%-of-spot ceiling", cur["im_mid"] <= BAD_QUOTE_PCT),
    ]:
        before = len(cur)
        cur = cur[mask.reindex(cur.index).fillna(False)]
        counts[label] = before - len(cur)
    return cur, counts


def wilcoxon(a, b):
    from scipy.stats import wilcoxon as w
    try:
        return float(w(a, b).pvalue)
    except Exception:
        return float("nan")


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    RESULTS.mkdir(exist_ok=True)
    report: dict = {"spec": spec["id"], "generated_at": pd.Timestamp.now("UTC").isoformat()}

    # ---- stage 0 ---------------------------------------------------------
    one = join_vendors(collect(1))
    kept, excl = apply_exclusions(one)
    usable = kept[kept["implied_move"].notna() & kept["or_implied"].notna()]
    report["stage0"] = {
        "dte1_straddles_priced": int(len(one)),
        "excluded": excl,
        "after_exclusions": int(len(kept)),
        "with_both_vendors": int(len(usable)),
        "gate": STAGE0_GATE,
        "gate_met": bool(len(usable) >= STAGE0_GATE),
        "primary_band": "dte == 1" if len(usable) >= STAGE0_GATE else "dte <= 2 (registered fallback)",
    }
    print(json.dumps(report["stage0"], indent=1), flush=True)

    if len(usable) < STAGE0_GATE:
        two = join_vendors(collect(2))
        kept2, _ = apply_exclusions(two)
        usable = kept2[kept2["implied_move"].notna() & kept2["or_implied"].notna()]
        report["stage0"]["fallback_n"] = int(len(usable))

    d = usable.copy()
    d["err_oq"] = (d["implied_move"] - d["im_mid"]).abs()
    d["err_or"] = (d["or_implied"] - d["im_mid"]).abs()

    # ---- Q1: which vendor is closer, on the registered consistency rule ----
    years = sorted(d["year"].unique())
    per_year = {int(y): {"n": int(len(g)),
                         "oquants": float(g["err_oq"].median()),
                         "orats": float(g["err_or"].median())}
                for y, g in d.groupby("year")}
    oq_wins = sum(1 for v in per_year.values() if v["oquants"] < v["orats"])
    rng = np.random.default_rng(0)
    diffs = []
    idx = np.arange(len(d))
    for _ in range(10_000):
        s = rng.choice(idx, len(idx), replace=True)
        diffs.append(d["err_oq"].to_numpy()[s].mean() - d["err_or"].to_numpy()[s].mean())
    report["q1"] = {
        "n": int(len(d)),
        "median_abs_err_oquants_pp": float(d["err_oq"].median()),
        "median_abs_err_orats_pp": float(d["err_or"].median()),
        "oquants_closer_in_years": f"{oq_wins} of {len(years)}",
        "consistency_bar": "70% of years",
        "consistency_met": bool(oq_wins / len(years) >= 0.70 or (len(years) - oq_wins) / len(years) >= 0.70),
        "wilcoxon_p": wilcoxon(d["err_oq"], d["err_or"]),
        "bootstrap_ci95_mean_err_diff_oq_minus_or": [float(np.percentile(diffs, 2.5)),
                                                     float(np.percentile(diffs, 97.5))],
        "neither_usable_rule": bool(d["err_oq"].median() > 2.0 and d["err_or"].median() > 2.0),
        "per_year": per_year,
    }

    # ---- Q2: the registered convention prediction -------------------------
    r_oq = (d["im_mid"] / d["implied_move"]).replace([np.inf, -np.inf], np.nan).dropna()
    r_or = (d["im_mid"] / d["or_implied"]).replace([np.inf, -np.inf], np.nan).dropna()
    med_oq, med_or = float(r_oq.median()), float(r_or.median())
    report["q2"] = {
        "im_1dte_over_oquants": med_oq,
        "im_1dte_over_orats": med_or,
        "prediction": "oquants ~1.00 (E|move|), ORATS ~0.7979 (1 SD)",
        "confirmed_orats_is_sigma": bool(0.95 <= med_oq <= 1.05 and 0.76 <= med_or <= 0.84),
        "confirmed_reversed": bool(0.95 <= med_or <= 1.05 and 0.76 <= med_oq <= 0.84),
    }
    report["q2"]["reading"] = (
        "CONFIRMED: ORATS is the 1-sigma convention" if report["q2"]["confirmed_orats_is_sigma"]
        else "CONFIRMED with roles REVERSED" if report["q2"]["confirmed_reversed"]
        else "NOT a convention difference — no convention claim is made")

    # ---- positive control -------------------------------------------------
    control = {}
    for band in (1, 2, 4):
        b = join_vendors(collect(band))
        b, _ = apply_exclusions(b)
        b = b[b["implied_move"].notna() & b["or_implied"].notna()]
        if len(b):
            control[f"dte<={band}"] = {
                "n": int(len(b)),
                "median_im_1dte": float(b["im_mid"].median()),
                "gap_vs_oquants_pp": float((b["im_mid"] - b["implied_move"]).median()),
                "gap_vs_orats_pp": float((b["im_mid"] - b["or_implied"]).median()),
            }
    gaps = [v["gap_vs_oquants_pp"] for v in control.values()]
    report["positive_control"] = {
        "bands": control,
        "monotone_increasing": bool(all(b > a for a, b in zip(gaps, gaps[1:]))),
        "note": ("the diffusive overhang must GROW with the DTE band, or the "
                 "measurement is not reading what it claims and the headline is withdrawn"),
    }

    # ---- representativeness + fill bracket ---------------------------------
    report["fill_bracket"] = {
        "median_im_at_bid": float(d["im_bid"].median()),
        "median_im_at_mid": float(d["im_mid"].median()),
        "median_im_at_ask": float(d["im_ask"].median()),
    }
    (RESULTS / "exp122_results.json").write_text(json.dumps(report, indent=1, default=str))
    d.to_parquet(RESULTS / "exp122_events.parquet", index=False)
    print(json.dumps({k: v for k, v in report.items() if k != "stage0"}, indent=1,
                     default=str)[:4000], flush=True)


if __name__ == "__main__":
    main()
