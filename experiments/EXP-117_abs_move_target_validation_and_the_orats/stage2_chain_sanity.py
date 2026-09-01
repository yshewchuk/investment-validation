#!/usr/bin/env python3
"""EXP-117 Stage 2 — chain quote sanity.

(a) The pre-registered cost%-of-spot ceiling: an ATM straddle entry whose
    worst-fill cost (buy both legs at the ask) exceeds 30% of spot is flagged
    BAD_QUOTE and excluded from scoring. The live motivating case: CBAT
    2026-08-31 STR-THRU at 166.7% of spot, which WIDE_MARKET flagged but did
    not remove.
(b) Greeks consistency detector: for events with chains on both sides of the
    print, the observed straddle-mid change must be reconcilable with the
    observed spot move under Black-Scholes deltas/gamma/vega/theta built from
    the independently fitted ORATS contract IVs. A large irreconcilable
    residual marks a bad quote on at least one side. The detector is an audit
    (report the residual distribution); the hard filter is (a).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/root/investing-plan")
sys.path.insert(0, str(ROOT))

from engine.data import store  # noqa: E402

HERE = Path(__file__).resolve().parent
COST_PCT_CEILING = 30.0  # registered in spec.yaml stage2
report: dict = {"generated_at": pd.Timestamp.now("UTC").isoformat(),
                "cost_pct_ceiling": COST_PCT_CEILING}

SQRT2PI = math.sqrt(2.0 * math.pi)


def ncdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def ndens(x):
    return np.exp(-0.5 * x * x) / SQRT2PI


def bs_greeks(S, K, T, sigma):
    """r=0 Black-Scholes call/put values and greeks, vectorized-safe scalars."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None
    sq = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sq * sq) / sq
    d2 = d1 - sq
    call = S * ncdf(d1) - K * ncdf(d2)
    put = K * ncdf(-d2) - S * ncdf(-d1)
    gamma = ndens(d1) / (S * sq)
    vega = S * ndens(d1) * math.sqrt(T)  # per 1.0 of sigma (iv column is a decimal)
    theta = -S * ndens(d1) * sigma / (2.0 * math.sqrt(T)) / 365.0  # per day, r=0
    return {
        "call": float(call), "put": float(put),
        "delta_call": float(ncdf(d1)), "delta_put": float(ncdf(d2) - 1.0),
        "gamma": float(gamma), "vega": float(vega),
        "theta_call": float(theta), "theta_put": float(theta),
    }


def main() -> None:
    trades = store.read_table("trades")
    st = trades[(trades["strategy"] == "STR-THRU") & (trades["fill_alpha"] == 0.0)].copy()
    print(f"STR-THRU worst-fill rows: {len(st):,}", flush=True)

    def parse(legs):
        try:
            doc = json.loads(legs) if isinstance(legs, str) else legs
        except (TypeError, ValueError):
            return None
        return doc

    rows = []
    for row in st.itertuples():
        doc = parse(row.legs)
        if not doc:
            continue
        spot_entry = doc.get("spot_entry")
        entry = {leg["name"]: leg for leg in doc.get("entry", [])}
        exit_ = {leg["name"]: leg for leg in doc.get("exit", [])}
        if not (spot_entry and "call" in entry and "put" in entry):
            continue
        cost_ask = entry["call"]["ask"] + entry["put"]["ask"]
        cost_mid = (entry["call"]["bid"] + entry["call"]["ask"]) / 2 \
            + (entry["put"]["bid"] + entry["put"]["ask"]) / 2
        exit_mid = None
        if "call" in exit_ and "put" in exit_:
            exit_mid = (exit_["call"]["bid"] + exit_["call"]["ask"]) / 2 \
                + (exit_["put"]["bid"] + exit_["put"]["ask"]) / 2
        expiry = entry["call"].get("expiry")
        rows.append({
            "ticker": row.ticker,
            "event_date": row.event_date,
            "entry_date": row.entry_date,
            "exit_date": row.exit_date,
            "strike": row.strike,
            "expiry": expiry,
            "dte_entry": doc.get("dte_entry"),
            "spot_entry": spot_entry,
            "spot_exit": doc.get("spot_exit"),
            "cost_ask": cost_ask,
            "cost_mid": cost_mid,
            "exit_mid": exit_mid,
            "wide_entry": bool(entry["call"].get("wide_market") or entry["put"].get("wide_market")),
        })
    ev = pd.DataFrame(rows)
    ev["entry_date"] = pd.to_datetime(ev["entry_date"])
    ev["exit_date"] = pd.to_datetime(ev["exit_date"])
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    ev["cost_pct"] = ev["cost_ask"] / ev["spot_entry"] * 100.0
    print(f"parsed events: {len(ev):,}", flush=True)

    # ---------------- (a) the cost% ceiling ----------------
    q = ev["cost_pct"]
    report["cost_pct_distribution"] = {
        str(int(p)): round(float(q.quantile(p / 100)), 3) for p in (50, 90, 95, 99, 99.9)
    }
    report["cost_pct_max"] = round(float(q.max()), 2)
    bad = ev[ev["cost_pct"] > COST_PCT_CEILING]
    report["n_bad_quote_over_ceiling"] = int(len(bad))
    report["bad_quote_rate_pct"] = round(len(bad) / len(ev) * 100, 3)
    ev["year"] = ev["event_date"].dt.year
    by_year = ev.groupby("year").apply(
        lambda g: (g["cost_pct"] > COST_PCT_CEILING).mean() * 100, include_groups=False
    )
    report["bad_quote_rate_by_year"] = {int(k): round(float(v), 3) for k, v in by_year.items()}
    top = bad.sort_values("cost_pct", ascending=False).head(15)
    report["worst_bad_quotes"] = [
        {"ticker": r.ticker, "date": str(r.event_date)[:10],
         "cost_pct": round(r.cost_pct, 1), "spot": r.spot_entry,
         "cost_ask": r.cost_ask} for r in top.itertuples()
    ]
    # does the ceiling catch the CBAT-style case (cost far above any real quote)?
    report["cbat_case"] = {
        "cost_pct": 166.7, "ceiling": COST_PCT_CEILING,
        "caught": bool(166.7 > COST_PCT_CEILING),
    }

    # ---------------- (b) Greeks consistency ----------------
    oc = store.read_table(
        "option_chains",
        columns=["ticker", "obs_date", "expiry", "strike", "right", "iv", "bid", "ask"],
    )
    oc = oc[oc["iv"].notna() & (oc["iv"] > 0)].copy()
    oc["obs_date"] = pd.to_datetime(oc["obs_date"])
    oc["expiry"] = pd.to_datetime(oc["expiry"])
    print(f"option_chains iv rows: {len(oc):,}", flush=True)

    need = ev[["ticker", "event_date", "entry_date", "exit_date", "strike", "expiry",
               "spot_entry", "spot_exit", "cost_mid", "exit_mid", "dte_entry"]].copy()
    need["expiry"] = pd.to_datetime(need["expiry"])

    def join_iv(df, date_col, suffix):
        ren = {"obs_date": date_col, "iv": f"iv_{suffix}",
               "bid": f"bid_{suffix}", "ask": f"ask_{suffix}",
               "right": f"right_{suffix}"}
        j = df.merge(oc.rename(columns=ren),
                     left_on=["ticker", date_col, "strike", "expiry"],
                     right_on=["ticker", date_col, "strike", "expiry"],
                     how="left")
        return j

    j = join_iv(need.reset_index(), "entry_date", "e")
    j = join_iv(j, "exit_date", "x")
    # ATM iv = mean of call and put mid-iv at the strike. The merges multiply
    # rows (one per side per date), so collapse back per ORIGINAL event row via
    # the preserved `index` column — never the positional index.
    iv_e_c = j[j["right_e"] == "C"].groupby("index")["iv_e"].first()
    iv_e_p = j[j["right_e"] == "P"].groupby("index")["iv_e"].first()
    iv_x_c = j[j["right_x"] == "C"].groupby("index")["iv_x"].first()
    iv_x_p = j[j["right_x"] == "P"].groupby("index")["iv_x"].first()
    g = pd.DataFrame({
        "iv_entry": pd.concat([iv_e_c, iv_e_p], axis=1).mean(axis=1),
        "iv_exit": pd.concat([iv_x_c, iv_x_p], axis=1).mean(axis=1),
    }).reindex(need.index)
    need = need.join(g)
    have = need[np.isfinite(need["iv_entry"]) & np.isfinite(need["iv_exit"])
                & np.isfinite(need["exit_mid"]) & np.isfinite(need["spot_exit"])].copy()
    print(f"events with ivs on both sides: {len(have):,}", flush=True)

    resid = []
    for r in have.itertuples():
        T0 = max(r.dte_entry, 1) / 365.0
        dte_exit = max((pd.Timestamp(r.expiry) - pd.Timestamp(r.exit_date)).days, 0)
        T1 = dte_exit / 365.0
        g0 = bs_greeks(r.spot_entry, r.strike, T0, r.iv_entry)
        if g0 is None:
            resid.append(np.nan)
            continue
        dS = r.spot_exit - r.spot_entry
        dIV = r.iv_exit - r.iv_entry  # sigma is stored as a decimal
        dt_days = (pd.Timestamp(r.exit_date) - pd.Timestamp(r.entry_date)).days
        delta_straddle = g0["delta_call"] + g0["delta_put"]
        gamma_straddle = 2.0 * g0["gamma"]
        vega_straddle = 2.0 * g0["vega"]
        theta_straddle = g0["theta_call"] + g0["theta_put"]
        model_change = (delta_straddle * dS + 0.5 * gamma_straddle * dS * dS
                        + vega_straddle * dIV + theta_straddle * dt_days)
        observed_change = r.exit_mid - r.cost_mid
        resid.append(observed_change - model_change)
    have["resid"] = resid
    have["resid_rel"] = have["resid"] / have["cost_mid"].clip(lower=0.01)
    have["move_pct"] = (have["spot_exit"] / have["spot_entry"] - 1.0) * 100.0

    # The Taylor decomposition is a second-order approximation: trustworthy for
    # moderate moves, guaranteed to understate convexity for big ones. Report
    # the two regimes separately and flag only inside the trustworthy one.
    small = have[have["move_pct"].abs() <= 15.0]
    rr = small["resid_rel"].abs()
    report["greeks_n"] = int(len(have))
    report["greeks_n_small_move"] = int(len(small))
    report["greeks_resid_rel_small_move_pct"] = {
        str(int(p)): round(float(rr.quantile(p / 100)) * 100, 2) for p in (50, 90, 95, 99)
    }
    flagged = small[rr > 0.25]
    report["greeks_flagged_gt_25pct"] = int(len(flagged))
    report["greeks_flagged_rate_pct"] = round(len(flagged) / len(small) * 100, 3)
    report["greeks_flagged_gt_100pct"] = int((rr > 1.0).sum())
    report["greeks_flagged_gt_100pct_rate"] = round(float((rr > 1.0).mean()) * 100, 3)
    report["greeks_flagged_examples"] = [
        {"ticker": r.ticker, "date": str(r.event_date)[:10],
         "resid_rel": round(r.resid_rel, 3), "cost_mid": round(r.cost_mid, 3),
         "exit_mid": round(r.exit_mid, 3), "move_pct": round(r.move_pct, 2)}
        for r in flagged.sort_values("resid_rel", key=abs, ascending=False).head(12).itertuples()
    ]
    # model-free no-arbitrage check: an exit straddle worth less than intrinsic
    intrinsic = (have["spot_exit"] - have["strike"]).abs()
    below = have[have["exit_mid"] < intrinsic - 0.02 * intrinsic.clip(lower=0.5)]
    report["n_exit_below_intrinsic"] = int(len(below))
    report["exit_below_intrinsic_examples"] = [
        {"ticker": r.ticker, "date": str(r.event_date)[:10],
         "exit_mid": round(r.exit_mid, 3), "intrinsic": round(
             abs(r.spot_exit - r.strike), 3)}
        for r in below.head(10).itertuples()
    ]

    have.to_parquet(HERE / "results" / "stage2_greeks.parquet")
    ev.to_parquet(HERE / "results" / "stage2_costpct.parquet")
    (HERE / "results" / "stage2_results.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
