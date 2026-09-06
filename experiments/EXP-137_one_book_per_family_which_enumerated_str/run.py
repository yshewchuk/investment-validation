#!/usr/bin/env python3
"""EXP-137 — eight books, compared like for like.

    python3 experiments/EXP-137_one_book_per_family_which_enumerated_str/run.py

Four readings of every family, because each answers a different question and
the differences between them are the result:

  UNGATED, all events        is this family alive at all? The unselected
                             population — no gate, no chooser between families.
  GATED, all events          the book anyone would actually run.
  UNGATED, common universe   the eight families on IDENTICAL events, so the
                             comparison is like for like rather than eight
                             different samples.
  GATED, common universe     the same, gated.

A family positive ungated and negative gated has a GATE problem, not a
structure problem, and nothing but the pair distinguishes them. A family that
looks good on all events and bad on the common universe was being carried by
the events its rivals could not reach.

Families are never presented as one flat ranking. Six of the eight are
centre-peaked and want a quiet print; two are twin-peaked and want the modal
move. Those are two theses, not eight variants of one.
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

from engine.evaluate import evaluate, trade_stats  # noqa: E402
from experiments import common as ecommon, lib  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


margin = _load("e137_margin", E134 / "margin.py")
e137build = _load("e137_build", HERE / "build.py")
e133run = _load("e137_e133run", E133 / "run.py")
e134run = _load("e137_e134run", E134 / "run.py")
import family as fammod  # noqa: E402

RESULTS = HERE / "results"
MID = 0.5
PRIMARY = "N4q0_-1_1"
FAMILIES = e137build.FAMILIES
LABEL = {f.key: f for f in fammod.FAMILIES}


def book(trades: pd.DataFrame, fam: str, *, gated: bool, common: bool) -> pd.DataFrame:
    """One family's mid rows under one gate setting and one slice."""
    rows = e134run.apply_exit(trades[trades["arm"] == fam], "conditional")
    if common:
        rows = rows[rows["common_universe"].fillna(False).astype(bool)]
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
    if mid.empty:
        return mid
    if not gated:
        return mid.copy()
    decided = e133run.apply_gate(mid)
    return mid[mid["event_id"].isin(set(decided.loc[decided["traded"], "event_id"]))].copy()


def stats(mid: pd.DataFrame) -> dict:
    """Per-family statistics, before and after the account can pay for it."""
    if mid.empty:
        return {}
    cost = pd.to_numeric(mid["entry_cost"], errors="coerce")
    pnl = pd.to_numeric(mid["pnl"], errors="coerce")
    per_year = mid.groupby(pd.to_datetime(mid["event_date"]).dt.year)["ret"].mean()
    m = mid.copy()
    m["secured_per_contract"] = m["legs"].map(margin.secured_per_contract)
    funded = margin.simulate(m)
    ok = funded[funded["funded"]]
    return {
        "n": int(len(mid)),
        "tickers": int(mid["ticker"].nunique()),
        "mean": float(mid["ret"].mean()),
        "median": float(mid["ret"].median()),
        "win": float((mid["ret"] > 0).mean()),
        "sharpe_trade": float(trade_stats(mid["ret"], mid["event_date"])["sharpe_trade"]),
        "return_on_capital": float(pnl.sum() / cost.sum()) if cost.sum() else float("nan"),
        "years_positive": int((per_year > 0).sum()),
        "years": int(per_year.size),
        "worse_than_debit": int((pnl < -cost - 1e-9).sum()),
        "width_over_forecast": float(mid["width_over_forecast"].median()),
        "beyond_wing": float((mid["landed_over_half"] >= 1.0).mean()),
        "held_share": float(mid["held_to_expiry"].mean()),
        "funded": int(len(ok)),
        "secured_median": float(ok["secured_per_contract"].median()) if len(ok) else float("nan"),
        "final_equity": float(funded.attrs.get("final_equity", float("nan"))),
    }


def main(record: bool = True) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    built = e137build.build_all()
    trades = e133run.attach(built["trades"], built["tallies"])
    events = built["events"]

    results: dict[str, dict] = {}
    for fam in FAMILIES:
        for gated in (False, True):
            for cu in (False, True):
                s = stats(book(trades, fam, gated=gated, common=cu))
                if s:
                    tag = f"{fam}|{'gated' if gated else 'ungated'}|{'common' if cu else 'all'}"
                    s["offered_events"] = int(events[f"offers_{fam}"].sum())
                    results[tag] = s
        print(f"[EXP-137] {fam}: scored", flush=True)

    (RESULTS / "family_books.json").write_text(json.dumps(results, indent=1, default=str))
    pd.DataFrame(results).T.to_csv(RESULTS / "family_books.csv")

    prim = f"{PRIMARY}|gated|common"
    twin = f"N5q2_-2_1|gated|common"
    p, t = results.get(prim, {}), results.get(twin, {})
    acceptance = {
        "beats_twin_p5_family": bool(
            p.get("return_on_capital", -9) > t.get("return_on_capital", -9)
            and p.get("years_positive", -1) >= t.get("years_positive", -1)),
        "alive_ungated": bool(
            results.get(f"{PRIMARY}|ungated|all", {}).get("return_on_capital", -9) > 0),
        "defined_risk_holds": results.get(f"{PRIMARY}|gated|all", {}).get(
            "worse_than_debit", 1) == 0,
        "universe_floor": results.get(f"{PRIMARY}|gated|all", {}).get("n", 0) >= 250,
    }
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))
    print(f"[EXP-137] acceptance: {acceptance}", flush=True)

    spy = ecommon.load_spy_daily()
    already = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])
    for fam in FAMILIES:
        rows = e134run.apply_exit(trades[trades["arm"] == fam], "conditional")
        mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
        if mid.empty:
            continue
        decided = e133run.apply_gate(mid)
        keep = set(decided.loc[decided["traded"], "event_id"])
        full = rows[rows["event_id"].isin(keep)]
        if full.empty:
            continue
        is_primary = fam == PRIMARY
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["structure"] = f"grid cell: family {fam}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / fam
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, full, gate=None, run_dir=run_dir, repricer=None,
            tail_shock=ecommon.abs_move_tail_shock, spy_daily=spy,
            input_files=[RESULTS / "candidates.parquet"],
            extra_sections=lambda rr, k=fam: sections(results, events, built, k),
            write_report=True)
        if record and lib.spec_hash(cell) not in already:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-137] {fam}: report {result.report_path}", flush=True)


def sections(results, events, built, cell):
    """The four readings, grouped by payoff shape rather than ranked flat."""
    def rows_for(gated, cu):
        out = []
        for shape in (False, True):                      # centre first, then twin
            group = [f for f in FAMILIES if LABEL[f].twin_peaked == shape]
            for fam in group:
                s = results.get(f"{fam}|{'gated' if gated else 'ungated'}|"
                                f"{'common' if cu else 'all'}")
                if not s:
                    continue
                out.append([
                    ("**" + fam + "**") if fam == cell else fam,
                    "twin" if shape else "centre",
                    f"{s['n']:,}", f"{s['tickers']:,}",
                    f"{100*s['mean']:+.1f}%", f"{100*s['median']:+.1f}%",
                    f"{100*s['return_on_capital']:+.1f}%",
                    f"{s['sharpe_trade']:.2f}",
                    f"{s['years_positive']}/{s['years']}",
                    f"{s['worse_than_debit']}",
                ])
        return out

    cols = ["family", "payoff", "n", "tickers", "mean", "median", "on capital",
            "Sharpe", "years+", "loses>debit"]
    align = ["---", "---"] + ["---:"] * 8

    offered = [[fam, "twin" if LABEL[fam].twin_peaked else "centre",
                LABEL[fam].label,
                f"{int(events[f'offers_{fam}'].sum()):,}",
                f"{100*events[f'offers_{fam}'].mean():.1f}%"]
               for fam in FAMILIES]

    return [
        {
            "title": "Is this family alive? — every event it can carry, NO gate",
            "note": ("The unselected population. No gate and no chooser between "
                     "families: each trades everything its own rules admit. A "
                     "family that cannot make money here is dead, whatever its "
                     "share of EXP-134's argmax was."),
            "columns": cols, "align": align, "rows": rows_for(False, False),
        },
        {
            "title": "The book anyone would run — trailing six-month top 20%",
            "note": ("A family positive above and negative here has a GATE "
                     "problem, not a structure problem. The pair is what tells "
                     "them apart."),
            "columns": cols, "align": align, "rows": rows_for(True, False),
        },
        {
            "title": "Like for like — the common universe, ungated",
            "note": (f"The {built['meta'].get('common_universe_events', 0):,} events "
                     "on which ALL EIGHT families are tradeable, so every row "
                     "covers identical events. This is the comparison the "
                     "primary is read on; the tables above compare eight "
                     "different samples however carefully each was run."),
            "columns": cols, "align": align, "rows": rows_for(False, True),
        },
        {
            "title": "Like for like — the common universe, gated",
            "columns": cols, "align": align, "rows": rows_for(True, True),
        },
        {
            "title": "How often each family is offered at all",
            "note": ("Share-of-argmax and ability-to-trade are different things "
                     "and EXP-134 could not separate them. Three strikes fit "
                     "almost any ladder; seven rarely do."),
            "columns": ["family", "payoff", "shape", "events offered", "share"],
            "align": ["---", "---", "---", "---:", "---:"],
            "rows": offered,
        },
    ]


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
