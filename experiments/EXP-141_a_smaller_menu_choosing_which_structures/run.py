#!/usr/bin/env python3
"""EXP-141 — does a smaller menu survive being chosen out of sample?

    python3 experiments/EXP-141_a_smaller_menu_choosing_which_structures/run.py

Nothing is repriced. EXP-137 already priced all eight families on every event;
this file only changes which of them the chooser is allowed to offer, and
chooses that menu on data the evaluation never sees.

The measures that decide it are per-trade Sharpe and **profit per dollar of
posted collateral** — not return on capital. Return on capital divides by
premium, and premium is not the binding resource: a cash-secured account is
limited by the strike value it must post, which is why a two-short-put condor
at $24,400 and a four-short-put twin peak at $48,800 are not comparable on a
per-premium basis however similar their percentages look.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments/EXP-133_every_symmetric_put_structure_the_ladder"
E134 = ROOT / "experiments/EXP-134_priced_right_funded_and_held_structure_s"
E137 = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str"
for p in (ROOT, E133, E134, E137, HERE):
    sys.path.insert(0, str(p))

from engine import pnl_sim  # noqa: E402
from engine.evaluate import evaluate, trade_stats  # noqa: E402
from experiments import common as ecommon, lib  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


e137 = _load("e141_build", E137 / "build.py")
e133run = _load("e141_e133run", E133 / "run.py")
e134run = _load("e141_e134run", E134 / "run.py")
margin = _load("e141_margin", E134 / "margin.py")
import family as fammod  # noqa: E402

LABEL = {f.key: f for f in fammod.FAMILIES}
F = list(e137.FAMILIES)
RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
#: Registered split. Train chooses the menu; holdout scores it.
TRAIN_END, MENU_SIZES = 2022, (3, 4, 5, 6, 8)
GATES = (0.20, 0.30)


def load_panel():
    b = e137.build_all()
    rows = e134run.apply_exit(e133run.attach(b["trades"], b["tallies"]), "conditional")
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), 0.5)]
    mid = mid[mid["common_universe"].fillna(False).astype(bool)]
    full = mid.groupby("event_id")["arm"].nunique() == len(F)
    mid = mid[mid["event_id"].isin(full[full].index)].copy()
    mid["secured"] = mid["legs"].map(margin.secured_per_contract)
    return mid


def menu_from(mid: pd.DataFrame, k: int, criterion: str) -> list[str]:
    """Rank families on a period and take the top k.

    ``realized_best_count`` measures the thing a menu controls — how often a
    family can win at all. ``own_return_on_capital`` measures how well it does
    when always traded, which is a different question and is registered as the
    alternative rather than assumed equivalent.
    """
    real = mid.pivot(index="event_id", columns="arm", values="ret")[F]
    if criterion == "realized_best_count":
        score = pd.Series(np.bincount(real.values.argmax(1), minlength=len(F)), index=F)
    else:
        g = mid.groupby("arm")
        score = (g["pnl"].sum() / g["entry_cost"].sum()).reindex(F)
    return list(score.sort_values(ascending=False).index[:k])


def evaluate_menu(mid: pd.DataFrame, menu: list[str], q: float, label: str) -> dict:
    """One menu, gated, on whatever events `mid` holds."""
    piv = lambda c: mid.pivot(index="event_id", columns="arm", values=c)
    pred, real, cost, pnl, sim, sec = (piv("exp_pnl_sim_select"), piv("ret"),
                                       piv("entry_cost"), piv("pnl"),
                                       piv("exp_pnl_sim"), piv("secured"))
    dates = pd.to_datetime(mid.drop_duplicates("event_id")
                           .set_index("event_id")["event_date"].reindex(pred.index))
    sub = [F.index(m) for m in menu]
    n = len(pred)
    col = np.array(sub)[pred.values[:, sub].argmax(1)] if len(sub) > 1 \
        else np.full(n, sub[0])
    ix = np.arange(n)
    truth_full = real.values.argmax(1)
    truth_menu = np.array(sub)[real.values[:, sub].argmax(1)]

    d = pd.DataFrame({"event_date": dates.values, "exp_pnl_sim": sim.values[ix, col]})
    if q >= 1.0:
        keep = np.ones(n, dtype=bool)
    else:
        months = d["event_date"].dt.to_period("M").unique()
        cut = {m: pnl_sim.trailing_cutoff(d, m.to_timestamp(), quantile=q) for m in months}
        bar = d["event_date"].dt.to_period("M").map(cut).astype(float)
        keep = (d["exp_pnl_sim"].notna() & bar.notna()
                & (d["exp_pnl_sim"] >= bar)).to_numpy()
    if keep.sum() < 20:
        return {}
    r, c, p, s = (real.values[ix, col][keep], cost.values[ix, col][keep],
                  pnl.values[ix, col][keep], sec.values[ix, col][keep])
    yr = pd.Series(r).groupby(pd.DatetimeIndex(dates.values[keep]).year.values).mean()
    hit = float((col == truth_menu).mean())
    return {
        "label": label, "menu": menu, "k": len(menu), "gate": q,
        "n": int(keep.sum()),
        "hit": hit, "chance": 1.0 / len(menu), "lift": hit * len(menu),
        "hit_vs_all8": float((col == truth_full).mean()),
        "sharpe": float(trade_stats(r, pd.Series(dates.values[keep]))["sharpe_trade"]),
        "win": float((r > 0).mean()), "mean": float(r.mean()),
        "median": float(np.median(r)),
        "on_capital": float(p.sum() / c.sum()),
        # The measure that matters under a cash-secured constraint, as a
        # PERCENT: mean dollar P&L per contract (option points x 100) over the
        # median dollars of strike value posted. Printed at three decimals it
        # was rounding to 0.001 and carrying no information.
        "per_collateral": float(100 * (100 * p.mean()) / np.median(s)),
        "collateral_median": float(np.median(s)),
        "years_positive": int((yr > 0).sum()), "years": int(yr.size),
        "worse_than_debit": int((p < -c - 1e-9).sum()),
    }


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    mid = load_panel()
    mid["year"] = pd.to_datetime(mid["event_date"]).dt.year
    train, hold = mid[mid["year"] <= TRAIN_END], mid[mid["year"] > TRAIN_END]
    print(f"[EXP-141] train {train['event_id'].nunique():,} events "
          f"(<= {TRAIN_END}), holdout {hold['event_id'].nunique():,}", flush=True)

    menus, out = {}, []
    for crit in ("realized_best_count", "own_return_on_capital"):
        full_menu = menu_from(mid, 5, crit)
        for k in MENU_SIZES:
            m_train = menu_from(train, k, crit)
            menus[f"{crit}|{k}"] = {"train": m_train,
                                    "full_sample": menu_from(mid, k, crit)}
            for q in GATES:
                r = evaluate_menu(hold, m_train, q, f"menu k={k} [{crit}]")
                if r:
                    r["criterion"] = crit
                    out.append(r)
        print(f"[EXP-141] {crit}: train-period top-5 = {menu_from(train,5,crit)}",
              flush=True)
        print(f"[EXP-141] {crit}: full-sample top-5 = {full_menu}", flush=True)

    for q in GATES:
        for label, menu in (("chooser all 8", F),
                            ("always condor", ["N4q0_-1_1"]),
                            ("always TWIN-P", ["N7q2_-1_-1_1"])):
            r = evaluate_menu(hold, menu, q, label)
            if r:
                r["criterion"] = "benchmark"
                out.append(r)

    t = pd.DataFrame(out)
    t.to_csv(RESULTS / "menu_holdout.csv", index=False)
    cols = ["label", "gate", "n", "lift", "sharpe", "win", "mean", "median",
            "on_capital", "per_collateral"]
    for q in GATES:
        print(f"\n=== HOLDOUT {TRAIN_END+1}-2026, top {int(100*q)}% gate ===")
        s = t[(t["gate"] == q) & (t["criterion"] != "own_return_on_capital")]
        print(s[cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    base = t[(t["label"] == "chooser all 8") & (t["gate"] == 0.20)]
    prim = t[(t["label"] == "menu k=5 [realized_best_count]") & (t["gate"] == 0.20)]
    cond = t[(t["label"] == "always condor") & (t["gate"] == 0.20)]
    tm = menus["realized_best_count|5"]
    acceptance = {
        "beats_all_eight": bool(len(prim) and len(base)
                                and prim.iloc[0]["sharpe"] > base.iloc[0]["sharpe"]
                                and prim.iloc[0]["per_collateral"] > base.iloc[0]["per_collateral"]),
        "beats_the_condor": bool(len(prim) and len(cond)
                                 and prim.iloc[0]["sharpe"] >= cond.iloc[0]["sharpe"]
                                 and prim.iloc[0]["per_collateral"] >= cond.iloc[0]["per_collateral"]),
        "stable_menu": len(set(tm["train"]) & set(tm["full_sample"])) >= 4,
        "universe_floor": bool(len(prim) and prim.iloc[0]["n"] >= 200),
    }
    print(f"\n[EXP-141] acceptance: {acceptance}", flush=True)
    (RESULTS / "menus.json").write_text(json.dumps(menus, indent=1))
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
