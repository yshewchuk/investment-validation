"""EXP-120 Stage 0 — can the chain-only reconstruction carry the headline?

Counts events whose stored chain, at the last pre-print close, holds BOTH a
pre-print expiry (no event inside it) and a post-print expiry. The registered
gate is 3,000; below it the two-expiry reconstruction becomes a limited-sample
secondary and the headline falls to the ORATS-dependent fallback.

Zero API calls. Reads option_chains, earnings_events and the calendar only.
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

#: Registered in spec.yaml. An E2 further out than this makes the event a
#: vanishing share of the straddle and the residual mostly noise.
MAX_E2_GAP_DAYS = 45

OUT = Path(__file__).resolve().parent / "results"


def main() -> int:
    ev = store.read_table("earnings_events", columns=["ticker", "event_date", "session"])
    ev = ev[ev["session"].notna()].copy()
    ev["ticker"] = ev["ticker"].astype(str)
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    cal = trading_calendar()

    # The chain we would read is the one at the last pre-print close.
    obs = []
    for t, d, s in zip(ev["ticker"], ev["event_date"], ev["session"]):
        try:
            obs.append(cal.last_pre_print(d, str(s)))
        except Exception:
            obs.append(pd.NaT)
    ev["obs_date"] = pd.to_datetime(pd.Series(obs, index=ev.index))
    ev = ev[ev["obs_date"].notna()]
    print(f"events with a session and an anchor: {len(ev):,}", flush=True)

    wanted = set(zip(ev["ticker"], ev["obs_date"]))

    # One pass over the chains, keeping only the expiry list per (ticker, obs).
    rows = []
    for item in store.iter_table("option_chains", columns=["ticker", "obs_date", "expiry"]):
        chunk = item[1] if isinstance(item, tuple) else item
        chunk = chunk.drop_duplicates()
        chunk["ticker"] = chunk["ticker"].astype(str)
        chunk["obs_date"] = pd.to_datetime(chunk["obs_date"])
        key = list(zip(chunk["ticker"], chunk["obs_date"]))
        chunk = chunk[[k in wanted for k in key]]
        if len(chunk):
            rows.append(chunk)
    chains = pd.concat(rows).drop_duplicates() if rows else pd.DataFrame()
    print(f"chain rows matching an event anchor: {len(chains):,}", flush=True)

    j = chains.merge(ev[["ticker", "obs_date", "event_date"]], on=["ticker", "obs_date"])
    j["expiry"] = pd.to_datetime(j["expiry"])
    j["pre"] = j["expiry"] < j["event_date"]
    j["post"] = (j["expiry"] >= j["event_date"]) & (
        (j["expiry"] - j["event_date"]).dt.days <= MAX_E2_GAP_DAYS
    )

    g = j.groupby(["ticker", "obs_date"]).agg(
        n_expiries=("expiry", "nunique"),
        has_pre=("pre", "any"),
        has_post=("post", "any"),
        event_date=("event_date", "first"),
    ).reset_index()
    g["year"] = pd.to_datetime(g["event_date"]).dt.year

    n = len(g)
    both = int((g["has_pre"] & g["has_post"]).sum())
    post_only = int((~g["has_pre"] & g["has_post"]).sum())
    report = {
        "events_with_a_chain": n,
        "both_expiries": both,
        "post_only_fallback_needed": post_only,
        "neither": int((~g["has_post"]).sum()),
        "median_expiries_per_event": float(g["n_expiries"].median()),
        "gate_threshold": 3000,
        "gate_met": bool(both >= 3000),
        "both_by_year": {
            int(y): int(v) for y, v in
            g[g["has_pre"] & g["has_post"]].groupby("year").size().items()
        },
        "events_by_year": {int(y): int(v) for y, v in g.groupby("year").size().items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stage0_coverage.json").write_text(json.dumps(report, indent=1))

    print()
    print(f"events with any chain at the anchor : {n:,}")
    print(f"  BOTH pre- and post-print expiry   : {both:,}  ({100*both/max(n,1):.1f}%)")
    print(f"  post-print only (fallback needed) : {post_only:,}")
    print(f"  no usable post-print expiry       : {report['neither']:,}")
    print(f"  median expiries per event         : {report['median_expiries_per_event']:.0f}")
    print()
    print(f"REGISTERED GATE (>= 3,000 both): {'MET' if report['gate_met'] else 'NOT MET'}")
    if report["both_by_year"]:
        print("\nboth-expiry events by year:")
        for y in sorted(report["both_by_year"]):
            print(f"  {y}: {report['both_by_year'][y]:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
