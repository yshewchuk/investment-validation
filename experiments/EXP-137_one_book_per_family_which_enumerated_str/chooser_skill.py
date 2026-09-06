#!/usr/bin/env python3
"""When a structure was PREDICTED best for an event, was it?

    python3 experiments/EXP-137_.../chooser_skill.py

A diagnostic on artifacts that already exist, not a new registered hypothesis.
EXP-137 priced all eight families on every event, so for each event there are
eight predicted expected returns and eight realized ones — and the question
EXP-134 could not answer is simply whether the prediction picks the right one.

Restricted to the COMMON UNIVERSE, where all eight families are tradeable, so
"best of eight" means the same thing on every event. Anywhere else the chooser
is picking from a varying menu and a hit rate would not be comparable across
events.

The number to beat is **12.5%** — one in eight. Everything here is measured
against that and against two bounds that make the hit rate readable:

    ORACLE   take the realized best. The ceiling.
    RANDOM   take a uniformly random family. The floor, and the thing a
             chooser has to beat to have done anything at all.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
E134 = ROOT / "experiments" / "EXP-134_priced_right_funded_and_held_structure_s"
for p in (ROOT, E133, E134, HERE):
    sys.path.insert(0, str(p))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


e137 = _load("cs_build", HERE / "build.py")
e133run = _load("cs_e133run", E133 / "run.py")
e134run = _load("cs_e134run", E134 / "run.py")
import family as fammod  # noqa: E402

LABEL = {f.key: f for f in fammod.FAMILIES}
FAMILIES = list(e137.FAMILIES)
RESULTS = HERE / "results"


def panel() -> pd.DataFrame:
    """One row per (event, family) on the common universe: predicted and realized."""
    built = e137.build_all()
    trades = e133run.attach(built["trades"], built["tallies"])
    parts = []
    for fam in FAMILIES:
        rows = e134run.apply_exit(trades[trades["arm"] == fam], "conditional")
        mid = rows[np.isclose(rows["fill_alpha"].astype(float), 0.5)]
        mid = mid[mid["common_universe"].fillna(False).astype(bool)]
        parts.append(mid[["event_id", "ticker", "event_date", "arm",
                          "exp_pnl_sim_select", "ret", "entry_cost", "pnl"]])
    p = pd.concat(parts, ignore_index=True)
    # Only events carrying all eight, so "best of eight" is the same question
    # everywhere. The common-universe flag is set at build time on
    # tradeability; a family can still drop out here if its sim was undefined.
    full = p.groupby("event_id")["arm"].nunique() == len(FAMILIES)
    return p[p["event_id"].isin(full[full].index)].copy()


def main() -> None:
    p = panel()
    n_events = p["event_id"].nunique()
    print(f"common universe with all {len(FAMILIES)} families priced: {n_events:,} events\n")

    pred = p.pivot(index="event_id", columns="arm", values="exp_pnl_sim_select")
    real = p.pivot(index="event_id", columns="arm", values="ret")
    pred, real = pred[FAMILIES], real[FAMILIES]

    chosen = pred.values.argmax(axis=1)
    truth = real.values.argmax(axis=1)
    rank_of_choice = (-real.values).argsort(axis=1).argsort(axis=1)[
        np.arange(len(chosen)), chosen] + 1

    hit = (chosen == truth)
    print("=== was the predicted-best structure the realized-best? ===")
    print(f"  top-1 hit rate      {100*hit.mean():5.1f}%   (chance = {100/len(FAMILIES):.1f}%)")
    for k in (2, 3, 4):
        print(f"  predicted best in the realized top {k}: "
              f"{100*(rank_of_choice <= k).mean():5.1f}%   "
              f"(chance = {100*k/len(FAMILIES):.1f}%)")
    print(f"  mean realized RANK of the chosen structure: "
          f"{rank_of_choice.mean():.2f} of {len(FAMILIES)}  (chance = "
          f"{(len(FAMILIES)+1)/2:.1f})")

    rho = np.array([spearmanr(pred.values[i], real.values[i]).statistic
                    for i in range(len(pred))])
    rho = rho[np.isfinite(rho)]
    print(f"  per-event Spearman rho(predicted, realized): mean {rho.mean():+.3f}, "
          f"median {np.median(rho):+.3f}, share > 0 {100*(rho > 0).mean():.1f}%")

    print("\n=== what the choice was worth ===")
    picked = real.values[np.arange(len(chosen)), chosen]
    oracle = real.values.max(axis=1)
    rnd = real.values.mean(axis=1)
    worst = real.values.min(axis=1)
    for name, v in (("chooser", picked), ("oracle (realized best)", oracle),
                    ("random family (mean of 8)", rnd), ("worst family", worst)):
        print(f"  {name:26s} mean {100*v.mean():+7.2f}%   median {100*np.median(v):+7.2f}%")
    span = oracle.mean() - rnd.mean()
    print(f"  captured {100*(picked.mean() - rnd.mean()) / span:.1f}% of the "
          f"available spread between random and oracle")

    print("\n=== per family: precision and recall ===")
    print(f"  {'family':16s} {'payoff':7s} {'predicted':>9} {'was best':>9} "
          f"{'precision':>10} {'recall':>7} {'mean when picked':>17}")
    for i, fam in enumerate(FAMILIES):
        sel, tru = chosen == i, truth == i
        prec = 100 * (sel & tru).sum() / max(sel.sum(), 1)
        rec = 100 * (sel & tru).sum() / max(tru.sum(), 1)
        mean_when = 100 * picked[sel].mean() if sel.sum() else float("nan")
        print(f"  {fam:16s} {'twin' if LABEL[fam].twin_peaked else 'centre':7s} "
              f"{sel.sum():9,} {tru.sum():9,} {prec:9.1f}% {rec:6.1f}% {mean_when:16.2f}%")

    out = {
        "events": int(n_events),
        "top1_hit_rate": float(hit.mean()),
        "chance": 1.0 / len(FAMILIES),
        "mean_realized_rank": float(rank_of_choice.mean()),
        "spearman_mean": float(rho.mean()),
        "spearman_share_positive": float((rho > 0).mean()),
        "mean_ret_chooser": float(picked.mean()),
        "mean_ret_oracle": float(oracle.mean()),
        "mean_ret_random": float(rnd.mean()),
        "share_of_spread_captured": float((picked.mean() - rnd.mean()) / span),
        "per_family": {
            fam: {
                "predicted_best": int((chosen == i).sum()),
                "was_best": int((truth == i).sum()),
                "precision": float((chosen == i)[(truth == i) & (chosen == i)].sum()
                                   / max((chosen == i).sum(), 1)),
            } for i, fam in enumerate(FAMILIES)},
    }
    (RESULTS / "chooser_skill.json").write_text(json.dumps(out, indent=1))
    print(f"\nwritten to {RESULTS / 'chooser_skill.json'}")


if __name__ == "__main__":
    main()
