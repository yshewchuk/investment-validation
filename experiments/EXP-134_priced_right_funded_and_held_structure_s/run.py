#!/usr/bin/env python3
"""EXP-134 — the search under three constraints the account actually has.

    python3 experiments/EXP-134_priced_right_funded_and_held_structure_s/run.py

``build.py`` has already refused every candidate whose own strikes carry a
no-arbitrage violation. Two things are left, and neither can be decided one
event at a time:

  THE EXIT    is a choice between three conventions, and the registered
              primary is CONDITIONAL: close at the first post-print close when
              that curve is arithmetically consistent, hold to expiry when it
              is not. Both branches are observable at the moment of the
              decision, which is what makes it a rule rather than a hindsight
              repair.
  THE MARGIN  depends on the whole book in date order — what is already open,
              what equity is now — so it runs last, after the gate, over the
              trades the strategy actually wanted to place.

Order matters and is registered: choose the most profitable structure, gate it,
then ask whether the account can fund it. A trade the account cannot secure
does not happen and is NOT re-selected to something cheaper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
E133 = ROOT / "experiments" / "EXP-133_every_symmetric_put_structure_the_ladder"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E133))
sys.path.insert(0, str(HERE))

from engine.evaluate import evaluate, trade_stats               # noqa: E402
from experiments import common, lib                             # noqa: E402

import margin                                                   # noqa: E402


def _load(name: str, path: Path):
    """Import a module by PATH, not by name.

    Both experiments have a `build.py` and a `run.py`, and both directories are
    on `sys.path` because this run reuses EXP-133's pricing engine. A bare
    `import build` then resolves to whichever directory happens to sit first —
    it resolved to EXP-133's, so this file silently evaluated EXP-133's cached
    candidates and failed on the first column EXP-134 adds. Loading by path
    makes which module is which a statement rather than an accident.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


e134 = _load("exp134_build", HERE / "build.py")
e133run = _load("exp133_run", E133 / "run.py")

RESULTS = HERE / "results"
MID = 0.5
PRIMARY = "best_all"
PRIMARY_EXIT = "conditional"
EXITS = ("conditional", "close_x1", "hold_expiry")
MCAP_FLOOR = 10e9


def apply_exit(trades: pd.DataFrame, convention: str) -> pd.DataFrame:
    """Rewrite exit value, P&L and return under one exit convention.

    ``close_x1``     the first post-print close, at the quoted fill. What every
                     prior twin-peak experiment assumed without testing.
    ``hold_expiry``  the terminal payoff at the expiry close. Defined risk is
                     then exact — the enumerated payoff floor is zero — at the
                     cost of the post-print vol crush and several more days of
                     directional exposure.
    ``conditional``  close when the post-print curve satisfies the same three
                     no-arbitrage conditions the entry had to satisfy, hold
                     when it does not. A position marked at an inconsistent
                     curve is not being valued, it is being guessed at, and
                     nobody would sell into that quote.
    """
    t = trades.copy()
    if convention == "close_x1":
        pass
    elif convention == "hold_expiry":
        t["exit_value"] = t["exit_value_expiry"]
    elif convention == "conditional":
        hold = ~t["exit_arb_ok"].fillna(True).astype(bool)
        t["exit_value"] = np.where(hold, t["exit_value_expiry"], t["exit_value"])
        t["held_to_expiry"] = hold
    else:
        raise ValueError(convention)
    if "held_to_expiry" not in t:
        t["held_to_expiry"] = convention == "hold_expiry"
    # A held position settles at intrinsic and the alpha grid has nothing to
    # say about it: there is no spread to cross at expiry. The row keeps its
    # alpha label so the sweep stays one sample, but its value does not move.
    t = t[t["exit_value"].notna()].copy()
    t["pnl"] = t["exit_value"] - t["entry_cost"]
    t["ret"] = np.where(t["entry_cost"] > 0, t["pnl"] / t["entry_cost"], np.nan)
    return t


def fund(mid: pd.DataFrame) -> pd.DataFrame:
    """Walk the gated book in date order and record what the account could do."""
    m = mid.copy()
    m["secured_per_contract"] = m["legs"].map(margin.secured_per_contract)
    return margin.simulate(m)


def book_stats(mid: pd.DataFrame, funded: pd.DataFrame) -> dict:
    """Trade statistics on the FUNDED book, plus what funding cost."""
    f = funded[funded["funded"]]
    if f.empty:
        return {}
    cost = pd.to_numeric(f["entry_cost"], errors="coerce")
    pnl = pd.to_numeric(f["pnl"], errors="coerce")
    stats = trade_stats(f["ret"], f["event_date"])
    per_year = f.groupby(pd.to_datetime(f["event_date"]).dt.year)["ret"].mean()
    return {
        "wanted": int(len(funded)),
        "n": int(len(f)),
        "tickers": int(f["ticker"].nunique()),
        "unfunded": int((~funded["funded"]).sum()),
        "mean": float(f["ret"].mean()),
        "median": float(f["ret"].median()),
        "win": float((f["ret"] > 0).mean()),
        "sharpe_trade": float(stats["sharpe_trade"]),
        "return_on_capital": float(pnl.sum() / cost.sum()) if cost.sum() else float("nan"),
        "years_positive": int((per_year > 0).sum()),
        "years": int(per_year.size),
        "worse_than_debit": int((pnl < -cost - 1e-9).sum()),
        "held_share": float(f["held_to_expiry"].mean()),
        "centre_share": float((~f["twin_peaked"].fillna(False).astype(bool)).mean()),
        # What the account actually did, as opposed to what sizing asked for.
        "contracts_median": float(f["contracts"].median()),
        "secured_median": float(f["secured_per_contract"].median()),
        "concurrency_max": int(funded["concurrency"].max()),
        "final_equity": float(funded.attrs.get("final_equity", float("nan"))),
        "margin_bound_share": float(
            (f["contracts"] < (0.05 * f["equity_at_entry"] /
                               (f["entry_cost"] * margin.CONTRACT_MULTIPLIER))).mean()),
    }


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    built = e134.build_all()
    trades = e133run.attach(built["trades"], built["tallies"])

    summary, funded_books, books = {}, {}, {}
    for convention in EXITS:
        for arm in e134.ARMS:
            rows = trades[trades["arm"] == arm]
            if rows.empty:
                continue
            rows = apply_exit(rows, convention)
            mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
            decided = e133run.apply_gate(mid)
            traded_ids = set(decided.loc[decided["traded"], "event_id"])
            gated_mid = mid[mid["event_id"].isin(traded_ids)].copy()
            if gated_mid.empty:
                continue
            f = fund(gated_mid)
            key = f"{arm}|{convention}"
            funded_books[key] = f
            books[key] = rows[rows["event_id"].isin(
                set(f.loc[f["funded"], "event_id"]))]
            summary[key] = book_stats(gated_mid, f)
            s = summary[key]
            if s:
                print(f"[EXP-134] {key:28s} wanted {s['wanted']:5d} funded {s['n']:5d} "
                      f"({s['tickers']:4d} tickers)  mean {100*s['mean']:+7.1f}%  "
                      f"on capital {100*s['return_on_capital']:+6.1f}%  "
                      f"{s['years_positive']}/{s['years']} yrs  "
                      f"held {100*s['held_share']:.0f}%  "
                      f"worse-than-debit {s['worse_than_debit']}", flush=True)

    (RESULTS / "arm_summary.json").write_text(
        json.dumps({"arms": summary, "meta": built["meta"]}, indent=1, default=str))

    prim = f"{PRIMARY}|{PRIMARY_EXIT}"
    inc = f"incumbent|{PRIMARY_EXIT}"
    rnd = f"random_pick|{PRIMARY_EXIT}"
    p, i, r = summary.get(prim, {}), summary.get(inc, {}), summary.get(rnd, {})
    acceptance = {
        "beats_the_incumbent": bool(
            p.get("return_on_capital", -9) >= i.get("return_on_capital", -9)
            and p.get("years_positive", -1) >= i.get("years_positive", -1)),
        "choosing_beats_not_choosing": bool(
            p.get("return_on_capital", -9) > r.get("return_on_capital", -9)),
        "defined_risk_holds": p.get("worse_than_debit", 1) == 0,
        "universe_floor": p.get("n", 0) >= 250,
    }
    print(f"[EXP-134] acceptance: {acceptance}", flush=True)
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))

    already = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])
    for key, book in books.items():
        arm, convention = key.split("|")
        is_primary = key == prim
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["structure"] = f"grid cell: {arm} / {convention}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / arm / convention
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, book, gate=None, run_dir=run_dir, repricer=None,
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[RESULTS / "candidates.parquet"],
            extra_sections=lambda rr, k=key: sections(summary, funded_books,
                                                      built, acceptance, k),
            write_report=True)
        if record and lib.spec_hash(cell) not in already:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-134] {key}: report {result.report_path}", flush=True)


def sections(summary, funded_books, built, acceptance, key):
    """The required outputs: the funnels, the exit comparison, the margin."""
    ev = built["events"]
    order = [f"{a}|{c}" for c in EXITS for a in e134.ARMS if f"{a}|{c}" in summary]

    arm_rows = [[
        f"**{k}**" if k == key else k, f"{summary[k]['wanted']:,}",
        f"{summary[k]['n']:,}", f"{summary[k]['tickers']:,}",
        f"{100*summary[k]['mean']:+.1f}%", f"{100*summary[k]['median']:+.1f}%",
        f"{100*summary[k]['return_on_capital']:+.1f}%",
        f"{summary[k]['sharpe_trade']:.2f}",
        f"{summary[k]['years_positive']}/{summary[k]['years']}",
        f"{100*summary[k]['held_share']:.0f}%",
        f"{100*summary[k]['centre_share']:.0f}%",
        f"{summary[k]['worse_than_debit']}",
    ] for k in order if summary.get(k)]

    f = funded_books.get(key, pd.DataFrame())
    margin_rows = []
    if len(f):
        ok = f[f["funded"]]
        margin_rows = [
            ["trades the strategy wanted", f"{len(f):,}"],
            ["... the account could secure", f"{int(f['funded'].sum()):,}"],
            ["... refused: one contract exceeded the headroom",
             f"{int((~f['funded']).sum()):,}"],
            ["median cash secured per contract",
             f"${ok['secured_per_contract'].median():,.0f}" if len(ok) else "—"],
            ["median contracts bought", f"{ok['contracts'].median():.0f}" if len(ok) else "—"],
            ["peak concurrent positions", f"{int(f['concurrency'].max())}"],
            ["final equity from $100,000",
             f"${f.attrs.get('final_equity', float('nan')):,.0f}"],
        ]

    return [
        {
            "title": "Acceptance — the registered criteria",
            "note": ("Return on capital, not CAGR: with margin capping size at a "
                     "contract or two, a compounding curve is a statement about "
                     "the cap rather than about the edge."),
            "columns": ["criterion", "result"],
            "align": ["---", "---"],
            "rows": [[k, "**PASS**" if v else "**FAIL**"] for k, v in acceptance.items()],
        },
        {
            "title": "Every arm under every exit convention",
            "note": ("`wanted` is what the strategy chose and the gate admitted; "
                     "`funded` is what a cash-secured $100,000 account could "
                     "actually place. `held` is the share closed at expiry rather "
                     "than at the post-print close, and under `conditional` it is "
                     "exactly the share whose post-print curve violated "
                     "no-arbitrage."),
            "columns": ["arm | exit", "wanted", "funded", "tickers", "mean",
                        "median", "on capital", "Sharpe", "years+", "held",
                        "centre", "loses>debit"],
            "align": ["---"] + ["---:"] * 11,
            "rows": arm_rows,
        },
        {
            "title": "The no-arbitrage funnel — what the filter costs",
            "note": ("Per event, medians over the priced universe. A candidate is "
                     "judged on the sub-curve it trades: a violation four strikes "
                     "away is somebody else's problem."),
            "columns": ["stage", "median per event", "mean per event"],
            "align": ["---", "---:", "---:"],
            "rows": [
                ["candidates admissible on geometry and width",
                 f"{ev['n_admissible'].median():,.0f}", f"{ev['n_admissible'].mean():,.0f}"],
                ["... surviving the three no-arbitrage conditions",
                 f"{ev['n_arb_ok'].median():,.0f}", f"{ev['n_arb_ok'].mean():,.0f}"],
                ["... refused for a violation on their own strikes",
                 f"{ev['n_arb_rejected'].median():,.0f}", f"{ev['n_arb_rejected'].mean():,.0f}"],
                ["... also inside the 25% spread filter",
                 f"{ev['n_tradeable'].median():,.0f}", f"{ev['n_tradeable'].mean():,.0f}"],
            ],
            "body": ["", f"EXP-133 measured 44.6% of its TRADED entries sitting on a "
                     "violation against the fixed-shape incumbent's 1.0%. This "
                     "filter is why that number should now be zero by construction."],
        },
        {
            "title": f"What the account could fund ({key})",
            "note": ("Every short put secures `strike x 100` in cash with no spread "
                     "offset, held until the position closes, and total secured "
                     "across all open positions may not exceed 50% of current "
                     "marked equity. A trade that cannot be secured does not "
                     "happen and is not re-selected to something cheaper."),
            "columns": ["", "count"],
            "align": ["---", "---:"],
            "rows": margin_rows,
            "body": ["", "Fixed-fraction sizing does not bind here. 5% of equity "
                     "over a dollar debit asks for tens of contracts; margin "
                     "allows one or two. Every size in this table was set by the "
                     "secured requirement."],
        },
        {
            "title": "Pricer equivalence receipt",
            "columns": ["check", "comparisons", "max |difference|"],
            "align": ["---", "---:", "---:"],
            "rows": [
                ["entry cost and exit value, every alpha",
                 f"{built['meta']['equivalence'].get('price_comparisons', 0):,}",
                 f"{built['meta']['equivalence'].get('price_max_abs_diff', float('nan')):.2e}"],
                ["expected P&L against engine.pnl_sim",
                 f"{built['meta']['equivalence'].get('sim_comparisons', 0):,}",
                 f"{built['meta']['equivalence'].get('sim_max_abs_diff', float('nan')):.2e}"],
            ],
            "body": ["", f"Sampled over "
                     f"{built['meta']['equivalence'].get('events_checked', 0)} events "
                     "spread across the whole run."],
        },
    ]


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
