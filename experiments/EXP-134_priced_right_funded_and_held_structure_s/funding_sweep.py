#!/usr/bin/env python3
"""POST-HOC — what a different funding policy would have bought.

    python3 experiments/EXP-134_.../funding_sweep.py

**Not a registered arm.** The structure selection, the no-arbitrage filter, the
gate and the exit convention are all exactly as registered and none of them
moves here. What moves is only how the account funds what the strategy already
chose. The PER-TRADE edge is therefore untouched — every trade's return on its
own debit is what it was. Aggregate return on capital still moves between rows,
and not because the edge moved: a different policy funds a different SUBSET, so
the mix changes. Saying it "cannot change" would be wrong, and it does change,
by up to 6pp below.

That is why this can be reported without a new pre-registration and why nothing
in it may be promoted: it is a capital-allocation sensitivity on a fixed book,
not a second test of the strategy.

Four changes were requested together, so they are run one at a time as well as
together. A bundle that improves the outcome tells you nothing about which of
its parts did the work, and three of these four could plausibly do nothing:

    account      $100,000 -> $200,000
    cap          50% -> 66% of current equity
    sizing       5% of equity -> aim at 1/4 of current headroom
    ordering     chronological -> most expensive first within an entry date
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
for p in (ROOT, E133, HERE):
    sys.path.insert(0, str(p))

import margin  # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


e134 = _load("exp134_build", HERE / "build.py")
e133run = _load("exp133_run", E133 / "run.py")
e134run = _load("exp134_run", HERE / "run.py")

BASE = dict(start_equity=100_000.0, cap=0.50, target_share=None, expensive_first=False)

POLICIES = {
    "registered ($100k, 50%, 5% of equity, chronological)": {},
    "+ account $200k only": dict(start_equity=200_000.0),
    "+ cap 66% only": dict(cap=0.66),
    "+ size at 1/4 of headroom only": dict(target_share=0.25),
    "+ most expensive first only": dict(expensive_first=True),
    "ALL FOUR ($200k, 66%, 1/4 headroom, expensive first)": dict(
        start_equity=200_000.0, cap=0.66, target_share=0.25, expensive_first=True),
    # The combinations the first sweep showed were missing. Sizing at a share
    # of headroom is the one change that clearly hurt, so it is swept rather
    # than dropped: if the direction is right and only the magnitude is wrong,
    # 1/2 will show it.
    "$200k + 66%": dict(start_equity=200_000.0, cap=0.66),
    "$200k + 66% + expensive first": dict(
        start_equity=200_000.0, cap=0.66, expensive_first=True),
    "$200k + 66% + expensive first + 1/2 headroom": dict(
        start_equity=200_000.0, cap=0.66, target_share=0.50, expensive_first=True),
}


def gated_book(arm: str, convention: str = "conditional") -> pd.DataFrame:
    """The trades the registered strategy chose and the gate admitted."""
    built = e134.build_all()
    trades = e133run.attach(built["trades"], built["tallies"])
    rows = e134run.apply_exit(trades[trades["arm"] == arm], convention)
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), 0.5)]
    decided = e133run.apply_gate(mid)
    keep = set(decided.loc[decided["traded"], "event_id"])
    book = mid[mid["event_id"].isin(keep)].copy()
    book["secured_per_contract"] = book["legs"].map(margin.secured_per_contract)
    return book


def run(arm: str = "best_all") -> pd.DataFrame:
    book = gated_book(arm)
    out = []
    for label, override in POLICIES.items():
        kw = dict(BASE, **override)
        f = margin.simulate(book, **kw)
        ok = f[f["funded"]]
        start = kw["start_equity"]
        final = f.attrs.get("final_equity", float("nan"))
        days = (pd.to_datetime(ok["exit_date"]) - pd.to_datetime(ok["entry_date"])).dt.days
        span = (pd.to_datetime(f["exit_date"]).max()
                - pd.to_datetime(f["entry_date"]).min()).days
        years = span / 365.25
        out.append({
            "policy": label,
            "wanted": len(f),
            "funded": int(f["funded"].sum()),
            "funded_pct": 100 * f["funded"].mean(),
            "contracts_median": float(ok["contracts"].median()) if len(ok) else 0.0,
            "avg_open": days.sum() / span if span else float("nan"),
            "peak_conc": int(f["concurrency"].max()),
            "final": final,
            "total_pct": 100 * (final / start - 1),
            "cagr_pct": 100 * ((final / start) ** (1 / years) - 1) if years > 0 else float("nan"),
            # Moves only through the funded MIX, never through the per-trade
            # edge: each trade's return on its own debit is fixed by the book.
            "on_capital_pct": 100 * ((ok["exit_value"] - ok["entry_cost"]).sum()
                                     / ok["entry_cost"].sum()) if len(ok) else float("nan"),
        })
    return pd.DataFrame(out)


def main() -> None:
    for arm in ("best_all", "incumbent"):
        table = run(arm)
        print(f"\n=== {arm} | conditional exit — funding policy sweep ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
        table.to_csv(HERE / "results" / f"funding_sweep_{arm}.csv", index=False)
    print(f"\nwritten to {HERE / 'results'}")


if __name__ == "__main__":
    main()
