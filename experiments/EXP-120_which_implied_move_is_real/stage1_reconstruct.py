"""EXP-120 Stage 1 — what IS the ORATS number, and what does the chain say?

Stage 0 closed the registered primary: only 406 of 52,390 events carry both a
pre- and post-print expiry, so the chain-only reconstruction cannot carry a
headline. Stage 1 answers the two questions the stored data CAN answer, both
registered in spec.yaml before this ran.

Q1  Does the iv30/exErnIv30 decomposition REPRODUCE ORATS impliedMove?
    It cannot adjudicate between vendors - it is built from ORATS fields. That
    is the point: if it reproduces ORATS, we learn what the board's denominator
    actually is, and that oquants is measuring something else.
Q2  On the 406 two-expiry events, which vendor does the chain-only
    reconstruction sit closer to? Evidence, explicitly not a verdict.

Zero API calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.calendar import trading_calendar  # noqa: E402
from engine.data import store  # noqa: E402
from engine.features import load_panel  # noqa: E402

SQRT_2_PI = float(np.sqrt(2.0 / np.pi))   # 0.7979
IV30_HORIZON_YEARS = 30.0 / 365.0
MAX_E2_GAP_DAYS = 45
OUT = Path(__file__).resolve().parent / "results"

# Registered readings for Q1 (spec.yaml stage1.q1_what_is_the_orats_number)
Q1_SAME = 0.2
Q1_DIFFERENT = 1.0


def anchors() -> pd.DataFrame:
    ev = store.read_table("earnings_events", columns=["ticker", "event_date", "session"])
    ev = ev[ev["session"].notna()].copy()
    ev["ticker"] = ev["ticker"].astype(str)
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    cal = trading_calendar()
    obs = []
    for d, s in zip(ev["event_date"], ev["session"]):
        try:
            obs.append(cal.last_pre_print(d, str(s)))
        except Exception:
            obs.append(pd.NaT)
    ev["obs_date"] = pd.to_datetime(pd.Series(obs, index=ev.index))
    return ev[ev["obs_date"].notna()]


def chain_expiries(wanted: set) -> pd.DataFrame:
    rows = []
    for item in store.iter_table(
        "option_chains", columns=["ticker", "obs_date", "expiry", "dte", "iv", "right", "strike", "spot"]
    ):
        chunk = item[1] if isinstance(item, tuple) else item
        chunk["ticker"] = chunk["ticker"].astype(str)
        chunk["obs_date"] = pd.to_datetime(chunk["obs_date"])
        key = list(zip(chunk["ticker"], chunk["obs_date"]))
        chunk = chunk[[k in wanted for k in key]]
        if len(chunk):
            rows.append(chunk)
    return pd.concat(rows) if rows else pd.DataFrame()


def atm_iv(group: pd.DataFrame) -> float:
    """ATM implied vol at one expiry: the strike nearest spot, both rights averaged."""
    g = group.dropna(subset=["strike", "spot", "iv"])
    if g.empty:
        return float("nan")
    spot = float(g["spot"].iloc[0])
    k = g.loc[(g["strike"] - spot).abs().idxmin(), "strike"]
    at = g[g["strike"] == k]
    return float(pd.to_numeric(at["iv"], errors="coerce").mean())


def main() -> int:
    ev = anchors()
    wanted = set(zip(ev["ticker"], ev["obs_date"]))
    print(f"event anchors: {len(ev):,}", flush=True)

    ch = chain_expiries(wanted)
    print(f"chain rows at an anchor: {len(ch):,}", flush=True)
    ch["expiry"] = pd.to_datetime(ch["expiry"])
    ch = ch.merge(ev[["ticker", "obs_date", "event_date"]], on=["ticker", "obs_date"])

    # -- the post-print expiry E2 every event needs -------------------------
    post = ch[(ch["expiry"] >= ch["event_date"])
              & ((ch["expiry"] - ch["event_date"]).dt.days <= MAX_E2_GAP_DAYS)]
    e2 = (post.groupby(["ticker", "obs_date"])["expiry"].min()
              .rename("e2").reset_index())

    # -- Q1: the ORATS-field decomposition ----------------------------------
    dm = store.read_table("daily_market", columns=["ticker", "date", "iv30", "exern_iv30", "implied_move"])
    dm["ticker"] = dm["ticker"].astype(str)
    dm["date"] = pd.to_datetime(dm["date"])
    q = e2.merge(dm, left_on=["ticker", "obs_date"], right_on=["ticker", "date"], how="inner")
    iv30 = pd.to_numeric(q["iv30"], errors="coerce") / 100.0
    exern = pd.to_numeric(q["exern_iv30"], errors="coerce") / 100.0
    var_event = (iv30**2 - exern**2) * IV30_HORIZON_YEARS
    q["fallback_im"] = SQRT_2_PI * np.sqrt(var_event.clip(lower=0)) * 100.0
    q["orats_im"] = pd.to_numeric(q["implied_move"], errors="coerce")
    ok = q["fallback_im"].notna() & q["orats_im"].notna() & (q["orats_im"] > 0)
    q1 = q[ok]
    d1 = (q1["fallback_im"] - q1["orats_im"]).abs()
    med = float(d1.median())
    reading = ("ORATS IS the decomposition" if med <= Q1_SAME
               else "ORATS is something else" if med >= Q1_DIFFERENT
               else "inconclusive")
    print()
    print(f"Q1  fallback vs ORATS impliedMove, n={len(q1):,}")
    print(f"    median |fallback - ORATS| = {med:.3f} pp   (<= {Q1_SAME} same, >= {Q1_DIFFERENT} different)")
    print(f"    corr {q1['fallback_im'].corr(q1['orats_im']):.4f}   "
          f"within 0.5pp {100*(d1<=0.5).mean():.1f}%   medians "
          f"fallback {q1['fallback_im'].median():.2f} / ORATS {q1['orats_im'].median():.2f}")
    print(f"    REGISTERED READING: {reading}")

    # -- Q2: chain-only, on whatever has two expiries -----------------------
    pre = ch[ch["expiry"] < ch["event_date"]]
    e1 = (pre.groupby(["ticker", "obs_date"])["expiry"].max().rename("e1").reset_index())
    two = e2.merge(e1, on=["ticker", "obs_date"], how="inner")
    print(f"\nQ2  two-expiry events: {len(two):,}", flush=True)

    recs = []
    idx = ch.set_index(["ticker", "obs_date"]).sort_index()
    for t, o, x2, x1 in zip(two["ticker"], two["obs_date"], two["e2"], two["e1"]):
        try:
            g = idx.loc[(t, o)]
        except KeyError:
            continue
        if isinstance(g, pd.Series):
            continue
        s2, s1 = atm_iv(g[g["expiry"] == x2]), atm_iv(g[g["expiry"] == x1])
        if not (np.isfinite(s2) and np.isfinite(s1)):
            continue
        T2 = (pd.Timestamp(x2) - pd.Timestamp(o)).days / 365.0
        if T2 <= 0:
            continue
        v = (s2 / 100.0) ** 2 * T2 - (s1 / 100.0) ** 2 * T2
        recs.append({"ticker": t, "obs_date": o, "e1": x1, "e2": x2,
                     "chain_im": SQRT_2_PI * np.sqrt(max(v, 0.0)) * 100.0,
                     "inverted": v <= 0})
    rec = pd.DataFrame(recs)
    print(f"    reconstructed: {len(rec):,}  (inverted var_event, kept: "
          f"{int(rec['inverted'].sum()) if len(rec) else 0})")

    result = {"q1": {"n": int(len(q1)), "median_abs_diff_pp": med, "reading": reading,
                     "corr": float(q1["fallback_im"].corr(q1["orats_im"])),
                     "within_0.5pp": float((d1 <= 0.5).mean())}}

    if len(rec):
        p = load_panel()
        p["date"] = pd.to_datetime(p["date"]); p["ticker"] = p["ticker"].astype(str)
        pe = p.merge(ev[["ticker", "event_date", "obs_date"]],
                     left_on=["ticker", "date"], right_on=["ticker", "event_date"], how="inner")
        j = rec.merge(pe[["ticker", "obs_date", "implied_move", "or_implied"]],
                      on=["ticker", "obs_date"], how="inner")
        j = j[(pd.to_numeric(j["implied_move"], errors="coerce") > 0)
              & (pd.to_numeric(j["or_implied"], errors="coerce") > 0)]
        if len(j):
            doq = (pd.to_numeric(j["implied_move"], errors="coerce") - j["chain_im"]).abs()
            dor = (pd.to_numeric(j["or_implied"], errors="coerce") - j["chain_im"]).abs()
            rng = np.random.default_rng(120)
            diffs = [float(np.median(doq.values[i]) - np.median(dor.values[i]))
                     for i in (rng.integers(0, len(j), len(j)) for _ in range(2000))]
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            print(f"    joined with both vendors: {len(j):,}")
            print(f"    median |oquants - chain| = {doq.median():.3f} pp")
            print(f"    median |ORATS   - chain| = {dor.median():.3f} pp")
            print(f"    difference (oq - orats)  = {doq.median()-dor.median():+.3f} pp"
                  f"   bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]")
            print(f"    chain-reconstructed median implied move: {j['chain_im'].median():.2f}%")
            print("    NOTE: n is far below the registered consistency bar. EVIDENCE, NOT A VERDICT.")
            result["q2"] = {"n": int(len(j)),
                            "median_abs_oquants": float(doq.median()),
                            "median_abs_orats": float(dor.median()),
                            "diff": float(doq.median() - dor.median()),
                            "ci": [float(lo), float(hi)],
                            "verdict": "EVIDENCE ONLY - below the registered consistency bar"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage1.json").write_text(json.dumps(result, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
