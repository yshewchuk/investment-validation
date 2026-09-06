#!/usr/bin/env python3
"""EXP-136 — funding the book: account size, cap, sizing and fill order.

    python3 experiments/EXP-136_funding_the_book_account_size_the_secure/run.py

Nothing is repriced and nothing is re-selected. EXP-134's registered
configuration — the enumerated candidates, the width rules, the no-arbitrage
filter, the objective, the gate, the exit convention — is held fixed, and the
set of trades the strategy WANTS is identical in every cell below. The only
thing that varies is which of them a cash-secured account can actually place,
and at what size.

**The registered primary is expected to lose.** A post-hoc sweep of these
policies was run on this same book before the spec was written, at the user's
request, and it said the requested bundle (4.24% CAGR) is beaten by two of its
own components ($200k + 66% cap: 10.73%). It is registered as the primary
anyway because it is the policy that was asked for, and registering a
hypothesis you expect to lose is the point of registering it. Every number here
was visible first; nothing in this run is independent confirmation.

The one non-obvious finding this exists to nail down: **sizing to a share of
headroom, to "leave room for concurrency", is backwards for this book.** The
account averages a quarter of one open position and its median secured-at-entry
is zero. Reserving headroom trades away size that is certain for concurrency
that mostly never arrives.
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
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
E134 = ROOT / "experiments" / "EXP-134_priced_right_funded_and_held_structure_s"
for p in (ROOT, E133, E134, HERE):
    sys.path.insert(0, str(p))

from engine.evaluate import evaluate  # noqa: E402
from experiments import common, lib  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _load(name: str, path: Path):
    """Import by PATH. Three experiments here have a `run.py` and two a
    `build.py`; a bare import resolves to whichever directory sits first on
    `sys.path`, which silently loaded the wrong module once already."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


margin = _load("exp136_margin", E134 / "margin.py")
e134build = _load("exp136_e134build", E134 / "build.py")
e133run = _load("exp136_e133run", E133 / "run.py")
e134run = _load("exp136_e134run", E134 / "run.py")

#: EXP-134's registered funding policy — the reference every cell is read
#: against, and the only one already in the ledger.
BASE = dict(start_equity=100_000.0, cap=0.50, target_share=None,
            expensive_first=False)

POLICIES = {
    "registered_exp134": {},
    "account_200k_only": dict(start_equity=200_000.0),
    "cap_66_only": dict(cap=0.66),
    "quarter_headroom_only": dict(target_share=0.25),
    "expensive_first_only": dict(expensive_first=True),
    "account_200k_and_cap_66": dict(start_equity=200_000.0, cap=0.66),
    "account_200k_cap_66_expensive_first": dict(
        start_equity=200_000.0, cap=0.66, expensive_first=True),
    "account_200k_cap_66_expensive_first_half_headroom": dict(
        start_equity=200_000.0, cap=0.66, target_share=0.50, expensive_first=True),
    # The registered primary, listed last so the table reads as a build-up.
    "PRIMARY_all_four": dict(start_equity=200_000.0, cap=0.66,
                             target_share=0.25, expensive_first=True),
}
PRIMARY = "PRIMARY_all_four"
REFERENCE = "registered_exp134"

ARMS = ("best_all", "best_twin_only", "incumbent")
EXITS = ("conditional", "close_x1", "hold_expiry")


def gated_book(trades: pd.DataFrame, arm: str, convention: str) -> pd.DataFrame:
    """EXP-134's wanted trades for one arm and exit convention."""
    rows = e134run.apply_exit(trades[trades["arm"] == arm], convention)
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), 0.5)]
    decided = e133run.apply_gate(mid)
    keep = set(decided.loc[decided["traded"], "event_id"])
    book = mid[mid["event_id"].isin(keep)].copy()
    book["secured_per_contract"] = book["legs"].map(margin.secured_per_contract)
    return book


def funded_rows(all_alphas: pd.DataFrame, book: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Every alpha of the trades one policy could actually place.

    ``score`` works on the mid rows because the funding decision is made once,
    at the entry, on the mid debit. The REPORT needs the whole alpha grid for
    the same events, which is this.
    """
    f = margin.simulate(book, **dict(BASE, **policy))
    keep = set(f.loc[f["funded"], "event_id"])
    return all_alphas[all_alphas["event_id"].isin(keep)].copy()


def score(book: pd.DataFrame, policy: dict) -> dict:
    """One funding policy over one fixed book."""
    kw = dict(BASE, **policy)
    f = margin.simulate(book, **kw)
    ok = f[f["funded"]]
    if ok.empty:
        return {}
    start = kw["start_equity"]
    final = f.attrs.get("final_equity", float("nan"))
    days = (pd.to_datetime(ok["exit_date"]) - pd.to_datetime(ok["entry_date"])).dt.days
    span = (pd.to_datetime(f["exit_date"]).max()
            - pd.to_datetime(f["entry_date"]).min()).days
    years = span / 365.25
    unf = f[~f["funded"]]
    # Which refusals a bigger account would fix, and which only scheduling could.
    too_big = int((unf["secured_per_contract"] > kw["cap"] * unf["equity_at_entry"]).sum())
    cost = pd.to_numeric(ok["entry_cost"], errors="coerce")
    return {
        "wanted": int(len(f)),
        "funded": int(len(ok)),
        "funded_pct": 100 * float(f["funded"].mean()),
        "contracts_median": float(ok["contracts"].median()),
        "avg_open": float(days.sum() / span) if span else float("nan"),
        "peak_concurrency": int(f["concurrency"].max()),
        "refused_too_expensive": too_big,
        "refused_blocked": int(len(unf)) - too_big,
        "start": start,
        "final": float(final),
        "cagr_pct": 100 * ((final / start) ** (1 / years) - 1) if years > 0 else float("nan"),
        "on_capital_pct": 100 * float(
            (ok["exit_value"] - ok["entry_cost"]).sum() / cost.sum()),
        "worse_than_debit": int(((ok["exit_value"] - ok["entry_cost"]) < -cost - 1e-9).sum()),
        "trades_per_year": float(len(ok) / years) if years > 0 else float("nan"),
    }


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    built = e134build.build_all()
    trades = e133run.attach(built["trades"], built["tallies"])

    results: dict[str, dict] = {}
    primary_rows = primary_book = None
    for arm in ARMS:
        for convention in EXITS:
            book = gated_book(trades, arm, convention)
            if book.empty:
                continue
            if arm == "best_all" and convention == "conditional":
                primary_book = book
                primary_rows = e134run.apply_exit(trades[trades["arm"] == arm], convention)
            for name, policy in POLICIES.items():
                s = score(book, policy)
                if s:
                    results[f"{arm}|{convention}|{name}"] = s
        print(f"[EXP-136] {arm}: {len(EXITS) * len(POLICIES)} cells scored", flush=True)

    (RESULTS / "policy_sweep.json").write_text(json.dumps(results, indent=1, default=str))
    frame = pd.DataFrame(results).T
    frame.index.name = "arm|exit|policy"
    frame.to_csv(RESULTS / "policy_sweep.csv")

    key = f"best_all|conditional|{PRIMARY}"
    ref = f"best_all|conditional|{REFERENCE}"
    p, r = results.get(key, {}), results.get(ref, {})
    acceptance = {
        "beats_the_registered_policy": bool(
            p.get("cagr_pct", -9) > r.get("cagr_pct", -9)),
        "funds_more": bool(p.get("funded", 0) > r.get("funded", 0)),
        "no_new_defined_risk_failures": p.get("worse_than_debit", 1) == 0,
    }
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))

    print("\n=== best_all | conditional — the registered comparison ===", flush=True)
    show = [k for k in POLICIES]
    tab = pd.DataFrame(
        [dict(policy=k, **results[f"best_all|conditional|{k}"]) for k in show
         if f"best_all|conditional|{k}" in results])
    print(tab[["policy", "funded", "funded_pct", "contracts_median", "avg_open",
               "refused_too_expensive", "refused_blocked", "final", "cagr_pct",
               "on_capital_pct"]].to_string(index=False,
                                            float_format=lambda v: f"{v:,.2f}"), flush=True)
    print(f"\n[EXP-136] acceptance: {acceptance}", flush=True)

    best = tab.loc[tab["cagr_pct"].idxmax()]
    print(f"[EXP-136] best cell is '{best['policy']}' at {best['cagr_pct']:.2f}% CAGR; "
          f"the registered primary is "
          f"{results[key]['cagr_pct']:.2f}%", flush=True)

    # A policy sweep is still an experiment, and the program's rule is that an
    # experiment without a REPORT.md generated by engine.report does not exist.
    # The primary cell's FUNDED book goes through the standard evaluation so
    # the record carries a provenance block, an alpha sweep and real headline
    # numbers -- and so the ledger gets a genuine mean and Sharpe instead of
    # the CAGR-in-the-Sharpe-column that the first run of this file wrote.
    spy = common.load_spy_daily()
    for name, is_primary in ((PRIMARY, True), ("account_200k_and_cap_66", False)):
        rows = funded_rows(primary_rows, primary_book, POLICIES[name])
        if rows.empty:
            continue
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["policy"] = f"grid cell: {name}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / name
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, rows, gate=None, run_dir=run_dir, repricer=None,
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[E134 / "results" / "candidates.parquet"],
            extra_sections=lambda rr, k=name: policy_sections(results, k),
            write_report=True)
        print(f"[EXP-136] {name}: report {result.report_path}", flush=True)
        if record and is_primary:
            # Appended even though a RAN row for this hash exists: that row
            # carries CAGR in the sharpe_trade column and a percentage in
            # oos_mean_mid. The ledger is append-only, so the correction is a
            # new row and a note in the guide, not an edit.
            lib.record_evaluation(HERE, cell, result.results)
            print("[EXP-136] corrected ledger row recorded", flush=True)


def policy_sections(results: dict, cell: str) -> list[dict]:
    """The sweep itself, as the report's own table."""
    rows = []
    for name in POLICIES:
        s = results.get(f"best_all|conditional|{name}")
        if not s:
            continue
        rows.append([
            f"**{name}**" if name == cell else name,
            f"{s['funded']:,}", f"{s['funded_pct']:.1f}%",
            f"{s['contracts_median']:.0f}", f"{s['avg_open']:.2f}",
            f"{s['refused_too_expensive']:,}", f"{s['refused_blocked']:,}",
            f"${s['final']:,.0f}", f"{s['cagr_pct']:.2f}%",
            f"{s['on_capital_pct']:.1f}%",
        ])
    return [{
        "title": "Funding policy sweep — one change at a time",
        "note": (
            "Identical book in every row: same candidates, same no-arbitrage "
            "filter, same objective, same gate, same exit convention, same "
            "1,428 wanted trades. Only funding moves. The registered primary "
            "is all four changes together; it is beaten by two of its own "
            "components, and the single change responsible is sizing to a "
            "share of headroom. These cells reuse one book, so they are a "
            "deterministic comparison and carry no error bar."),
        "columns": ["policy", "funded", "funded %", "contracts", "avg open",
                    "refused: too big", "refused: blocked", "final", "CAGR",
                    "on capital"],
        "align": ["---"] + ["---:"] * 9,
        "rows": rows,
    }]


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
