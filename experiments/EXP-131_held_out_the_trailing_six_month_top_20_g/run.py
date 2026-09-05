#!/usr/bin/env python3
"""EXP-131 — the trailing six-month top-20% gate, on data its choice never saw.

    python3 experiments/EXP-131_held_out_the_trailing_six_month_top_20_g/run.py

The window and quantile were fixed on 2018-2022. Everything reported here is
2023-2026. The trailing cutoff is allowed to reach back across the boundary,
because a live deployment in January 2023 would have had 2022's events in hand
and withholding them would model a constraint that does not exist.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "experiments" / "EXP-130_a_monthly_trailing_cutoff_for_the_p_l_ga"))

from experiments import lib  # noqa: E402
import run as gate  # noqa: E402  — EXP-130's cutoff rules and scorer

RESULTS = HERE / "results"
SELECT_END, HOLD_START = 2022, 2023


def bootstrap(a: np.ndarray, b: np.ndarray, B: int = 2000, block: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    def draw(x):
        n = len(x); nb = int(np.ceil(n / block)); out = np.empty(B)
        for i in range(B):
            starts = rng.integers(0, max(n - block, 1), nb)
            out[i] = np.concatenate([x[j:j + block] for j in starts])[:n].mean()
        return out
    diff = draw(a) - draw(b)
    lo, hi = np.percentile(diff, [5, 95])
    return {"median": float(np.median(diff)), "lo": float(lo), "hi": float(hi),
            "p_gt_0": float((diff > 0).mean()),
            "distinguishable": bool(lo > 0 or hi < 0)}


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    RESULTS.mkdir(exist_ok=True)
    sim = pd.read_csv(gate.EXP129 / "results" / "simulated.csv")
    sim["event_date"] = pd.to_datetime(sim["event_date"])
    trades = pd.read_parquet(gate.TRADES)
    trades["event_date"] = pd.to_datetime(trades["event_date"])

    out: dict = {"spec_hash": lib.spec_hash(spec), "books": {}, "bootstrap": {}, "shares": {}}
    hold_ids = set(sim.loc[sim.year >= HOLD_START, "event_id"])
    hold_trades = trades[trades.event_id.isin(hold_ids)]
    print(f"[EXP-131] holdout {HOLD_START}-2026: {len(hold_ids):,} candidates", flush=True)

    def score(label, ids, cuts=None):
        ids = set(ids) & hold_ids
        out["books"][label] = gate.book(hold_trades, ids, label)
        if cuts is not None and not cuts.empty:
            c = cuts[cuts.month.str.slice(0, 4).astype(int) >= HOLD_START].copy()
            c["year"] = c.month.str.slice(0, 4)
            per = c.groupby("year").apply(
                lambda g: g.admitted.sum() / max(g.candidates.sum(), 1), include_groups=False)
            out["shares"][label] = {k: float(v) for k, v in per.items()}
        return ids

    # The gate: cutoffs computed over the WHOLE series so the trailing window can
    # cross the boundary, then restricted to held-out events.
    for label, w, q in (("primary_6m_top20", 6, 0.20), ("window_9m", 9, 0.20),
                        ("window_12m", 12, 0.20), ("quantile_10", 6, 0.10),
                        ("quantile_30", 6, 0.30)):
        ids, cuts, _ = gate.monthly_trailing(sim, window_months=w, quantile=q)
        score(label, ids, cuts)

    ratio = sim.cost / sim.peak
    score("incumbent_cost_lt_half_peak", sim.loc[ratio < 0.5, "event_id"])

    mid = hold_trades[np.isclose(hold_trades.fill_alpha, 0.5)]
    ids_p, _, _ = gate.monthly_trailing(sim, window_months=6, quantile=0.20)
    a = mid[mid.event_id.isin(set(ids_p) & hold_ids)]["ret"].to_numpy(float)
    b = mid[mid.event_id.isin(set(sim.loc[ratio < 0.5, "event_id"]) & hold_ids)]["ret"].to_numpy(float)
    if len(a) > 40 and len(b) > 40:
        out["bootstrap"]["primary_minus_incumbent"] = bootstrap(a, b)

    # Selection-period performance for the same rule — the degradation is the finding.
    sel_ids = set(sim.loc[sim.year <= SELECT_END, "event_id"])
    sel_trades = trades[trades.event_id.isin(sel_ids)]
    out["books"]["primary_6m_top20_SELECTION_PERIOD"] = gate.book(
        sel_trades, set(ids_p) & sel_ids, "selection period")

    (RESULTS / "metrics.json").write_text(json.dumps(out, indent=1, default=str))

    print(f"\n{'rule':38}{'n':>6}{'mean':>8}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}{'be_a':>7}{'yrs+':>7}")
    for k, v in out["books"].items():
        if not v.get("n"):
            continue
        print(f"{k:38}{v['n']:>6,}{100*v['mean']:>7.2f}%{100*v['cagr']:>8.2f}%"
              f"{v['sharpe_trade']:>8.2f}{100*v['max_drawdown']:>7.1f}%"
              f"{(v['breakeven_alpha'] or float('nan')):>7.3f}{v['years_positive']:>4}/{v['years']}")
    bs = out["bootstrap"].get("primary_minus_incumbent")
    if bs:
        print(f"\nheld-out bootstrap, primary - incumbent: median {100*bs['median']:+.2f}pp  "
              f"90% CI [{100*bs['lo']:+.2f}, {100*bs['hi']:+.2f}]  P(>0)={bs['p_gt_0']:.0%}  "
              f"-> {'DISTINGUISHABLE' if bs['distinguishable'] else 'NOT distinguishable'}")
    print("\nadmitted share by held-out year (target 0.20):")
    for k, v in out["shares"].items():
        print(f"  {k:20} " + "  ".join(f"{y} {s:.2f}" for y, s in sorted(v.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
