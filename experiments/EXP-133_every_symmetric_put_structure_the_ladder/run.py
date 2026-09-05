#!/usr/bin/env python3
"""EXP-133 — every symmetric put structure the ladder can carry.

    python3 experiments/EXP-133_every_symmetric_put_structure_the_ladder/run.py

``build.py`` prices 12,600 candidate structures against real quotes on every
event and records, per arm, the one that arm would have chosen. This file does
the two things that cannot be done one event at a time:

  THE GATE     is a trailing quantile, so it needs the whole history in date
               order. Each arm computes its OWN bar from its OWN chosen-
               candidate history — see the spec: an argmax over a thousand
               candidates has a different exp_pnl_sim distribution than a fixed
               shape, and applying one bar to both compares two quantities
               rather than two rules.
  THE TALLIES  are the three counts the experiment was asked for — admissible,
               chosen, traded — and the third only exists after gating.

Primary is ``best_all``. Every other arm is a labelled grid cell that isolates
one thing the primary confounds, and ``oracle_realized`` reads the outcome and
is not promotable under any result.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from engine import pnl_sim  # noqa: E402
from engine.data import store  # noqa: E402
from engine.evaluate import (  # noqa: E402
    _max_drawdown, build_equity, cagr, evaluate, trade_stats,
)
from experiments import common, lib  # noqa: E402

import build as build_mod  # noqa: E402
import family as fam  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MID = 0.5
MCAP_FLOOR = build_mod.MCAP_FLOOR

#: What each registered acceptance criterion is actually asserting, in the
#: report, next to its verdict — so a PASS/FAIL is readable without the spec.
ACCEPTANCE_TEXT = {
    "more_names": "the primary trades at least twice the incumbent's events — the motivating claim",
    "beats_the_incumbent": "the primary's CAGR and per-trade Sharpe are at or above the incumbent's",
    "choosing_beats_not_choosing": "the primary beats a uniformly random admissible candidate on CAGR",
    "still_pays": "breakeven alpha at or below 0.45 against a mid assumption of 0.50",
    "defined_risk_holds": "no chosen trade loses more than its debit",
    "universe_floor": "at least 250 traded events",
}
#: Matches EXP-129/131 so the CAGR numbers here are the ones those reports mean.
FRACTION = 0.05
PRIMARY = "best_all"
#: Reported alongside, never as a benchmark this run reproduces: a different
#: snapshot and a different universe. EXP-126/131.
EXP131_REFERENCE = {"n": 393, "tickers": 217, "cagr": 0.1298, "sharpe": 1.10,
                    "years_positive": "9/9", "breakeven_alpha": 0.467}


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


def apply_gate(rows: pd.DataFrame) -> pd.DataFrame:
    """Mark each event traded / rejected / undetermined on a trailing bar.

    EXP-131's rule verbatim, on this arm's own history: an event in month M is
    ranked against the 80th percentile of ``exp_pnl_sim`` over the arm's
    gateable events dated in ``[M - 6 months, M)``. A window holding fewer than
    100 events leaves the event UNDETERMINED — not admitted, not rejected —
    because a bar computed from forty numbers is not the rule the experiment
    registered, and defaulting either way turns a cold start into a decision.
    """
    out = rows.copy()
    out["event_date"] = pd.to_datetime(out["event_date"])
    history = out.loc[out["exp_pnl_sim"].notna(), ["event_date", "exp_pnl_sim"]]
    months = out["event_date"].dt.to_period("M").unique()
    cutoffs = {m: pnl_sim.trailing_cutoff(history, m.to_timestamp()) for m in months}
    bar = out["event_date"].dt.to_period("M").map(cutoffs)
    out["gate_cutoff"] = bar.astype(float)
    out["gateable"] = out["exp_pnl_sim"].notna() & out["gate_cutoff"].notna()
    out["traded"] = out["gateable"] & (out["exp_pnl_sim"] >= out["gate_cutoff"])
    return out


def arm_rows(trades: pd.DataFrame, arm: str) -> pd.DataFrame:
    """This arm's chosen candidate per event, at every alpha, gated."""
    rows = trades[trades["arm"] == arm]
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)]
    decided = apply_gate(mid)
    traded_ids = set(decided.loc[decided["traded"], "event_id"])
    kept = rows[rows["event_id"].isin(traded_ids)].copy()
    return kept, decided


# --------------------------------------------------------------------------
# scoring a book
# --------------------------------------------------------------------------


def book_stats(mid: pd.DataFrame) -> dict:
    """The numbers every acceptance criterion is written against."""
    if mid.empty:
        return {}
    equity = build_equity(mid, FRACTION, mode="cashflow", max_deployed=1.0)
    curve = equity.get("equity")
    curve = (pd.Series(np.asarray(curve, dtype=float), index=pd.to_datetime(curve.index))
             if curve is not None and len(curve) >= 2 else None)
    stats = trade_stats(mid["ret"], mid["event_date"])
    per_year = mid.groupby(mid["event_date"].dt.year)["ret"].mean()
    cost = pd.to_numeric(mid["entry_cost"], errors="coerce")
    pnl = pd.to_numeric(mid["pnl"], errors="coerce")
    centre = ~mid["twin_peaked"].fillna(False).astype(bool)
    return {
        "n": int(len(mid)),
        "tickers": int(mid["ticker"].nunique()),
        "mean": float(mid["ret"].mean()),
        "median": float(mid["ret"].median()),
        "win": float((mid["ret"] > 0).mean()),
        "sharpe_trade": float(stats["sharpe_trade"]),
        "cagr": float(cagr(curve)) if curve is not None else float("nan"),
        "max_dd": float(_max_drawdown(curve)) if curve is not None else float("nan"),
        "return_on_capital": float(pnl.sum() / cost.sum()) if cost.sum() else float("nan"),
        "years_positive": int((per_year > 0).sum()),
        "years": int(per_year.size),
        "centre_share": float(centre.mean()),
        "n_strikes_median": float(mid["n_strikes"].median()),
        "width_over_forecast_median": float(mid["width_over_forecast"].median()),
        "half_width_pct_spot_median": float(mid["half_width_pct_spot"].median()),
        "rel_spread_median": float(mid["rel_spread"].median()),
        "landed_over_half_median": float(mid["landed_over_half"].median()),
        "beyond_wing": float((mid["landed_over_half"] >= 1.0).mean()),
        "sim_minus_real": float((mid["exp_pnl_sim"] - mid["ret"]).mean()),
        "n_admissible_median": float(mid["n_admissible"].median()),
        "worse_than_debit": int((pnl < -cost - 1e-9).sum()),
    }


def breakeven(rows: pd.DataFrame) -> float | None:
    """Lowest alpha on the grid whose mean return is positive — linear between."""
    by = rows.groupby("fill_alpha")["ret"].mean().sort_index()
    if by.empty or (by > 0).all():
        return 0.0
    if (by <= 0).all():
        return None
    below = by[by <= 0]
    above = by[by > 0]
    if below.empty or above.empty:
        return None
    a0, v0 = float(below.index[-1]), float(below.iloc[-1])
    a1, v1 = float(above.index[0]), float(above.iloc[0])
    return a0 if v1 == v0 else a0 + (a1 - a0) * (-v0) / (v1 - v0)


# --------------------------------------------------------------------------
# the consistency test the spec registers
# --------------------------------------------------------------------------


def optimism_consistency(mid: pd.DataFrame) -> dict:
    """Does the chooser get MORE optimistic the more candidates it is given?

    A chooser mining Black-Scholes error should: more draws from a biased
    estimator means a higher maximum, and the excess is not in the realized
    P&L. A chooser reading real structure should not. Spearman rho between an
    event's admissible-candidate count and its (simulated minus realized)
    return, with a two-sided p — a consistency test rather than a magnitude
    cutoff, per the program's promotion standard.
    """
    from scipy.stats import spearmanr

    frame = mid[["n_admissible", "exp_pnl_sim", "ret"]].dropna()
    if len(frame) < 30:
        return {"n": int(len(frame)), "rho": None, "p": None,
                "verdict": "too few trades to test"}
    gap = frame["exp_pnl_sim"] - frame["ret"]
    rho, p = spearmanr(frame["n_admissible"], gap)
    verdict = ("optimism GROWS with the candidate count — consistent with the "
               "chooser reading model error" if (rho > 0 and p < 0.05) else
               "no detectable relationship between candidate count and optimism")
    return {"n": int(len(frame)), "rho": float(rho), "p": float(p), "verdict": verdict}


# --------------------------------------------------------------------------
# the tallies the experiment was asked for
# --------------------------------------------------------------------------


def tally_frame(tallies: pd.DataFrame, decided: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per pattern: admissible, chosen, traded — the three counts, plus the arms.

    ``traded`` can only be counted here: it is ``chosen AND through the gate``,
    and the gate is a trailing quantile no single event can see.
    """
    out = tallies.copy()
    for arm, frame in decided.items():
        counts = (frame.loc[frame["traded"], "pattern_index"]
                  .value_counts().reindex(out["pattern_index"]).fillna(0))
        out[f"traded_{arm}"] = counts.to_numpy(dtype=np.int64)
    return out


def _group(tallies: pd.DataFrame, by: str, arm: str) -> pd.DataFrame:
    agg = tallies.groupby(by, dropna=False).agg(
        listed=("listed", "sum"), admissible=("admissible", "sum"),
        tradeable=("tradeable", "sum"),
        chosen=(f"argmax_{arm}", "sum") if f"argmax_{arm}" in tallies else ("listed", "size"),
        traded=(f"traded_{arm}", "sum"),
        patterns=("pattern_index", "size"),
    )
    return agg.sort_values("chosen", ascending=False)


def posthoc_section() -> dict:
    """Appendix P, if it has been generated — the post-hoc debit sweep.

    Folded into the report rather than left as a loose file, because a reader
    who reaches the acceptance table needs the answer to "does anything survive
    once the price is real" in the same document. It is generated by
    ``posthoc.py`` AFTER the primary ran and is labelled post-hoc everywhere it
    appears; when the file is absent this section says so rather than vanishing.
    """
    path = RESULTS / "posthoc_debit_floor.md"
    if not path.exists():
        return {"title": "POST-HOC: the chooser with a floor under the net debit",
                "note": "Not generated. Run `posthoc.py` after this file.",
                "body": []}
    lines = [ln for ln in path.read_text().splitlines() if not ln.startswith("## ")]
    return {"title": "POST-HOC: the chooser with a floor under the net debit",
            "note": "", "body": lines}


def sections(summary, funnels, tallies, decided, events, meta, arm, consistency,
             acceptance, defended):
    """Every required output in spec.yaml that engine.report does not produce."""
    fams = [
        [f"N={f.n_strikes}", f"({f.q0},{','.join(str(q) for q in f.tail)})",
         "**twin**" if f.twin_peaked else "centre", f.label]
        for f in fam.FAMILIES
    ]

    order = [a for a in build_mod.ARMS if a in summary and summary[a]]
    arm_rows_out = []
    for key in order:
        s = summary[key]
        arm_rows_out.append([
            f"**{key}**" if key == arm else key,
            f"{s['n']:,}", f"{s['tickers']:,}",
            f"{100*s['cagr']:+.2f}%" if s["cagr"] == s["cagr"] else "n/a",
            f"{s['sharpe_trade']:.2f}" if s["sharpe_trade"] == s["sharpe_trade"] else "n/a",
            f"{100*s['mean']:+.2f}%", f"{100*s['return_on_capital']:+.2f}%",
            f"{100*s['max_dd']:.1f}%" if s["max_dd"] == s["max_dd"] else "n/a",
            f"{s['years_positive']}/{s['years']}",
            f"{s['breakeven_alpha']:.3f}" if s.get("breakeven_alpha") is not None else "never",
            f"{100*s['centre_share']:.0f}%",
        ])

    fun_rows = [[k, f"{v['priced']:,}", f"{v['spread_mcap']:,}", f"{v['gateable']:,}",
                 f"{v['traded']:,}", f"{100*v['admit_rate']:.1f}%"]
                for k, v in funnels.items()]

    by_family = _group(tallies, "family_key", arm).reset_index()
    fam_label = {f.key: f for f in fam.FAMILIES}
    fam_rows = [[
        f"N={fam_label[r.family_key].n_strikes} "
        f"({fam_label[r.family_key].q0},{','.join(str(q) for q in fam_label[r.family_key].tail)})",
        "twin" if fam_label[r.family_key].twin_peaked else "centre",
        f"{r.patterns:,}", f"{r.listed:,}", f"{r.admissible:,}", f"{r.tradeable:,}",
        f"{r.chosen:,}", f"{r.traded:,}",
    ] for r in by_family.itertuples()]

    by_shape = _group(tallies, "shape_key", arm).reset_index().head(20)
    shape_rows = [[r.shape_key, f"{r.listed:,}", f"{r.admissible:,}", f"{r.tradeable:,}",
                   f"{r.chosen:,}", f"{r.traded:,}"] for r in by_shape.itertuples()]

    by_anchor = _group(tallies, "anchor_offset", arm).reset_index()
    anchor_rows = [[f"ATM+{int(r.anchor_offset)}", f"{r.listed:,}", f"{r.admissible:,}",
                    f"{r.tradeable:,}", f"{r.chosen:,}", f"{r.traded:,}"]
                   for r in by_anchor.itertuples()]

    ev = events
    bound_rows = [
        ["candidates listed by the ladder", f"{ev['n_listed'].median():,.0f}",
         f"{ev['n_listed'].mean():,.0f}"],
        ["... too narrow (fails 'spans the predicted move')",
         f"{ev['n_too_narrow'].median():,.0f}", f"{ev['n_too_narrow'].mean():,.0f}"],
        ["... too wide (fails 'within predicted move + 3sd')",
         f"{ev['n_too_wide'].median():,.0f}", f"{ev['n_too_wide'].mean():,.0f}"],
        ["... floor below zero on this ladder", f"{ev['n_floor_fail'].median():,.0f}",
         f"{ev['n_floor_fail'].mean():,.0f}"],
        ["... mean leg spread above 25%", f"{ev['n_spread_fail'].median():,.0f}",
         f"{ev['n_spread_fail'].mean():,.0f}"],
        ["candidates surviving all of it", f"{ev['n_tradeable'].median():,.0f}",
         f"{ev['n_tradeable'].mean():,.0f}"],
    ]

    mid = decided[arm]
    traded_mid = mid[mid["traded"]]
    wing_bucket = pd.cut(traded_mid["half_width_pct_spot"],
                         [0, 5, 10, 15, 100],
                         labels=["<5% of spot", "5-10%", "10-15%", ">15%"])
    opt = traded_mid.assign(bucket=wing_bucket).groupby("bucket", observed=True).agg(
        n=("ret", "size"), sim=("exp_pnl_sim", "mean"), real=("ret", "mean"))
    opt_rows = [[str(i), f"{r.n:,}", f"{100*r.sim:+.1f}%", f"{100*r.real:+.1f}%",
                 f"{100*(r.sim-r.real):+.1f}pp"] for i, r in opt.iterrows()]

    half_rows = [[
        key,
        f"{summary[key]['n']:,}",
        f"{100*decided[key].loc[decided[key]['traded'], 'exp_pnl_sim_select'].mean():+.2f}%",
        f"{100*decided[key].loc[decided[key]['traded'], 'exp_pnl_sim'].mean():+.2f}%",
        f"{100*(decided[key].loc[decided[key]['traded'], 'exp_pnl_sim_select'].mean() - decided[key].loc[decided[key]['traded'], 'exp_pnl_sim'].mean()):+.2f}pp",
    ] for key in order if len(decided[key][decided[key]["traded"]])]

    acc_rows = [[k, "**PASS**" if v else "**FAIL**", ACCEPTANCE_TEXT[k]]
                for k, v in acceptance.items()]

    return [
        {
            "title": "Acceptance — the registered criteria, against §0's verdict",
            "note": (
                "**§0 above is measuring the wrong thing here, and this table is "
                "the verdict.** The auto-verdict turns on mean return per trade. "
                "EXP-126 removed mean per trade from the promotion decision for "
                "exactly this reason: its chooser arm reported a mean of +0.49% "
                "while the book it managed lost 0.5% a year and drew down 79%. "
                "`ret` is P&L divided by the net debit, and this experiment's "
                "chooser maximises a quantity that is unbounded as that debit "
                "goes to zero — so a mean of +1,188% per trade is a statement "
                "about the denominator. Read CAGR, the alpha sweep and this "
                "table instead."
            ),
            "columns": ["registered criterion", "result", "what it says"],
            "align": ["---", "---", "---"],
            "rows": acc_rows,
            "body": ["", "**Where the debit went.** The chooser's median net "
                     f"debit is ${defended['cost_median_primary']:.2f} against the "
                     f"incumbent's ${defended['cost_median_incumbent']:.2f}, and "
                     f"{100*defended['cheap_share']:.1f}% of its picks net under "
                     "$0.25 across eight legs against "
                     f"{100*defended['cheap_share_incumbent']:.1f}% of the "
                     "incumbent's. US options quote on a one-cent grid below "
                     "$3.00, so a few cents net across eight mid-quotes is "
                     "arithmetic residue rather than a fillable price — which is "
                     "what the alpha sweep says out loud: the mean is "
                     f"{100*defended['mean_at_25_primary']:+.0f}% at alpha 0.25 "
                     f"against {100*defended['mean_at_25_incumbent']:+.0f}% for "
                     "the incumbent. `Appendix P` sweeps a floor under the debit."],
        },
        {
            "title": "The eight families — enumerated, not chosen",
            "note": (
                "Every all-put structure that is mirror-symmetric about a listed "
                "strike, whose contracts sum to zero, whose payoff floor is zero, "
                "and which fits in eight contracts, on 3/4/5/7 strikes. Both "
                "incumbents fall out of the enumeration rather than being inserted "
                "into it. **Only two are twin-peaked, and neither has fewer than "
                "five strikes** — so dropping to four strikes or three does not buy "
                "a cheaper twin peak, it buys a tent that wants a quiet print."
            ),
            "columns": ["strikes", "contracts q", "payoff", "shape"],
            "align": ["---", "---", "---", "---"],
            "rows": fams,
        },
        {
            "title": "Every arm, one universe, one gate each",
            "note": (
                "`centre` is the share of the arm's trades taken in a "
                "centre-peaked family — a book that is mostly condors is not a "
                "twin-peak programme with a wider universe, whatever its CAGR. "
                "`random_pick` is the null: a uniformly random admissible "
                "candidate. `oracle_realized` reads the outcome and is the "
                "ceiling, not a result."
            ),
            "columns": ["arm", "n", "tickers", "CAGR", "Sharpe", "mean/trade",
                        "on capital", "max DD", "years+", "breakeven a", "centre"],
            "align": ["---"] + ["---:"] * 10,
            "rows": arm_rows_out,
            "body": ["", f"EXP-131's published TWIN-P5 book, on an earlier snapshot "
                     f"and a different universe, for orientation only: "
                     f"{EXP131_REFERENCE['n']} trades, CAGR "
                     f"{100*EXP131_REFERENCE['cagr']:.2f}%, Sharpe "
                     f"{EXP131_REFERENCE['sharpe']:.2f}, "
                     f"{EXP131_REFERENCE['years_positive']} years positive."],
        },
        {
            "title": "Funnel — what each arm loses, and where",
            "columns": ["arm", "events with a candidate", "spread + mcap", "gateable",
                        "traded", "gate admit rate"],
            "align": ["---"] + ["---:"] * 5,
            "rows": fun_rows,
            "body": ["", "`gateable` needs a residual pool of at least 250 paired "
                     "errors AND a trailing window of at least 100 events; below "
                     "either, the event is UNDETERMINED rather than rejected."],
        },
        {
            "title": f"The three tallies, per family ({arm})",
            "note": (
                "`listed` = the ticker's ladder carries it. `admissible` = it also "
                "spans the predicted move without exceeding three SDs, with a zero "
                "floor in dollars. `tradeable` = its legs also quote inside 25%. "
                "`chosen` = it was the highest simulated expected P&L for its "
                "event. `traded` = it was chosen AND cleared the trailing "
                "top-20% gate. Counts are candidate-events, so one event "
                "contributes to `listed` once per pattern it can carry."
            ),
            "columns": ["strikes / q", "payoff", "patterns", "listed", "admissible",
                        "tradeable", "chosen", "traded"],
            "align": ["---", "---"] + ["---:"] * 6,
            "rows": fam_rows,
        },
        {
            "title": f"The twenty most-chosen shapes ({arm})",
            "note": ("A shape is a family plus a spacing RATIO, so `N5q2_-2_1@1:3` "
                     "is TWIN-P5 with its wing at three times its peak spacing — "
                     "the incumbent geometry — at whatever absolute width the "
                     "event's ladder and the width rules allowed."),
            "columns": ["shape", "listed", "admissible", "tradeable", "chosen", "traded"],
            "align": ["---"] + ["---:"] * 5,
            "rows": shape_rows,
        },
        {
            "title": f"By anchor offset ({arm})",
            "note": "Where the axis of symmetry sat, in ladder positions above the strike at or below spot.",
            "columns": ["anchor", "listed", "admissible", "tradeable", "chosen", "traded"],
            "align": ["---"] + ["---:"] * 5,
            "rows": anchor_rows,
        },
        {
            "title": "What bounds the search, per event",
            "note": ("Medians and means over priced events. The two width rules are "
                     "doing the work: the ladder lists far more than the rules admit, "
                     "which is the opposite of TWIN-P's problem, where the ladder "
                     "was the binding constraint."),
            "columns": ["stage", "median per event", "mean per event"],
            "align": ["---", "---:", "---:"],
            "rows": bound_rows,
            "body": ["", f"The 20-position ladder bound was reachable on "
                     f"{100*events['ladder_bound_binds'].mean():.1f}% of events; "
                     f"the median ladder carried {events['ladder_steps'].median():.0f} "
                     f"quoted strikes at the traded expiry."],
        },
        {
            "title": "Defined risk — where the claim stops being true",
            "note": (
                "Max loss is the debit **at expiry**: the contracts sum to zero "
                "over strikes that mirror exactly, so the deep-ITM tail is flat "
                "and the floor is zero. The exit is not at expiry — median 9 DTE "
                "remain — and it is priced by selling the longs and buying back "
                "the shorts at real quotes, which can net below zero. That is "
                "the honest boundary of the claim, and a search maximising "
                "return-on-debit finds the events where it bites."
            ),
            "columns": ["arm", "trades", "worse than a total loss", "worst return",
                        "median debit of those"],
            "align": ["---"] + ["---:"] * 4,
            "rows": [[
                k, f"{len(decided[k]):,}",
                f"{int((decided[k]['ret'] < -1.0000001).sum()):,} "
                f"({100*(decided[k]['ret'] < -1.0000001).mean():.1f}%)",
                f"{100*decided[k]['ret'].min():,.0f}%",
                f"${decided[k].loc[decided[k]['ret'] < -1.0000001, 'entry_cost'].median():.2f}"
                if (decided[k]["ret"] < -1.0000001).any() else "—",
            ] for k in order],
            "body": ["", "Counted over every candidate each arm CHOSE, before the "
                     "gate, at mid fills — the gate would otherwise hide the "
                     "problem behind its own selectivity."],
        },
        {
            "title": "Simulated against realized, by how wide the chooser went",
            "note": ("EXP-129 measured Black-Scholes at 3.88% from the mid at the "
                     "median but 14.41% for |delta| < 0.10. The widest candidates "
                     "are made of those contracts, so this table is where a "
                     "chooser mining repricing error shows up."),
            "columns": ["wing distance", "n", "simulated", "realized", "gap"],
            "align": ["---"] + ["---:"] * 4,
            "rows": opt_rows,
            "body": ["", f"**Consistency test (registered):** Spearman rho between "
                     f"an event's admissible-candidate count and its (simulated − "
                     f"realized) return over {consistency['n']:,} traded events: "
                     + (f"rho = {consistency['rho']:+.3f}, p = {consistency['p']:.4f}. "
                        f"{consistency['verdict']}."
                        if consistency["rho"] is not None else consistency["verdict"] + ".")],
        },
        {
            "title": "Split-half — how much of the argmax is Monte-Carlo luck",
            "note": ("Draws 1-2,000 chose the candidate; draws 2,001-4,000 supplied "
                     "the number the gate ranked. The gap between them is the "
                     "winner's curse in the ESTIMATE, and it is the part this "
                     "design removes. It is not the model error, which both halves "
                     "share and neither can see."),
            "columns": ["arm", "n", "selection half", "gate half", "gap"],
            "align": ["---"] + ["---:"] * 4,
            "rows": half_rows,
        },
        posthoc_section(),
        {
            "title": "Pricer equivalence receipt",
            "note": ("A second pricing path is only tolerable while it is "
                     "continuously proved to be the first one. A sample of "
                     "candidates was priced through "
                     "`engine.structures.price_structure` and their expected P&L "
                     "through `engine.pnl_sim.expected_pnl`; the run aborts above "
                     "1e-9."),
            "columns": ["check", "comparisons", "max |difference|"],
            "align": ["---", "---:", "---:"],
            "rows": [
                ["entry cost and exit value, every alpha",
                 f"{meta['equivalence']['price_comparisons']:,}",
                 f"{meta['equivalence']['price_max_abs_diff']:.2e}"],
                ["expected P&L against engine.pnl_sim",
                 f"{meta['equivalence']['sim_comparisons']:,}",
                 f"{meta['equivalence']['sim_max_abs_diff']:.2e}"],
            ],
            "body": ["", f"{meta['equivalence']['price_skipped_no_structure']} of "
                     f"{meta['equivalence']['candidates_sampled']} sampled candidates "
                     "could not take the first path: a family with no contract at its "
                     "own axis (the four-strike condor) has no `Structure` to mirror "
                     "about, because `LegSpec` requires a positive qty. Those are "
                     "covered by the second check and by identical code, not by the "
                     "first."],
        },
    ]


# --------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------


def attach(trades: pd.DataFrame, tallies: pd.DataFrame) -> pd.DataFrame:
    """Market cap, the shape's identity, and the mcap filter — once, for every arm."""
    t = trades.copy()
    t["event_date"] = pd.to_datetime(t["event_date"])
    t["year"] = t["event_date"].dt.year
    sec = store.read_table("securities", years=range(2017, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    t = t.merge(sec, on=["ticker", "year"], how="left")
    meta_cols = ["pattern_index", "family_key", "shape_key", "n_strikes", "twin_peaked",
                 "anchor_offset"]
    t = t.merge(tallies[meta_cols], on="pattern_index", how="left")
    # The incumbent is not in the pattern grid (pattern_index -1): its spacing
    # comes from a share of spot, not a ladder position. Its identity is filled
    # in by hand rather than left null, so every table that groups by shape
    # reports it instead of dropping it.
    inc = t["pattern_index"] < 0
    t.loc[inc, ["family_key", "shape_key"]] = build_mod.TWIN_P5_KEY, f"{build_mod.TWIN_P5_KEY}@1:3"
    t.loc[inc, ["n_strikes", "anchor_offset"]] = 5, 0
    t.loc[inc, "twin_peaked"] = True
    t["twin_peaked"] = t["twin_peaked"].fillna(False).astype(bool)
    return t[t["mcap_usd"] >= MCAP_FLOOR]


def main(record: bool = True) -> None:
    """``record=False`` evaluates and reports without touching LEDGER.csv.

    The ledger is a multiple-testing record, not a run log, and a pipeline
    smoke test on one year is not a hypothesis test. This flag exists because
    an early subset run of this file wrote seven RAN rows before the full
    universe had ever been priced; those rows are disclosed in REPORT.md rather
    than removed, and this is how it does not happen again.
    """
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    built = build_mod.build_all()
    trades = attach(built["trades"], built["tallies"])
    events, meta = built["events"], built["meta"]

    priced_per_arm = built["trades"].groupby("arm")["event_id"].nunique().to_dict()

    books, decided, summary, funnels = {}, {}, {}, {}
    for arm in build_mod.ARMS:
        kept, marked = arm_rows(trades, arm)
        books[arm] = kept
        decided[arm] = marked
        mid = kept[np.isclose(kept["fill_alpha"].astype(float), MID)]
        stats = book_stats(mid)
        if stats:
            stats["breakeven_alpha"] = breakeven(kept)
        summary[arm] = stats
        funnels[arm] = {
            "priced": int(priced_per_arm.get(arm, 0)),
            "spread_mcap": int(marked["event_id"].nunique()),
            "gateable": int(marked["gateable"].sum()),
            "traded": int(marked["traded"].sum()),
            "admit_rate": float(marked.loc[marked["gateable"], "traded"].mean())
            if marked["gateable"].any() else float("nan"),
        }
        if stats:
            print(f"[EXP-133] {arm}: {stats['n']:,} trades on {stats['tickers']:,} "
                  f"tickers, CAGR {100*stats['cagr']:+.2f}%, Sharpe "
                  f"{stats['sharpe_trade']:.2f}, {stats['years_positive']}/"
                  f"{stats['years']} years, centre {100*stats['centre_share']:.0f}%",
                  flush=True)
        else:
            print(f"[EXP-133] {arm}: nothing traded", flush=True)

    tallies = tally_frame(built["tallies"], decided)
    tallies.to_parquet(RESULTS / "tallies_gated.parquet", index=False)
    consistency = optimism_consistency(
        decided[PRIMARY][decided[PRIMARY]["traded"]])
    prim, inc = decided[PRIMARY], decided["incumbent"]
    prim_all = trades[trades["arm"] == PRIMARY]
    inc_all = trades[trades["arm"] == "incumbent"]
    defended = {
        "cost_median_primary": float(prim["entry_cost"].median()),
        "cost_median_incumbent": float(inc["entry_cost"].median()),
        "cheap_share": float((prim["entry_cost"] < 0.25).mean()),
        "cheap_share_incumbent": float((inc["entry_cost"] < 0.25).mean()),
        "mean_at_25_primary": float(
            prim_all.loc[np.isclose(prim_all["fill_alpha"], 0.25), "ret"].mean()),
        "mean_at_25_incumbent": float(
            inc_all.loc[np.isclose(inc_all["fill_alpha"], 0.25), "ret"].mean()),
    }
    (RESULTS / "arm_summary.json").write_text(json.dumps(
        {"arms": summary, "funnel": funnels, "consistency": consistency,
         "equivalence": meta["equivalence"], "skips": meta["skips"]},
        indent=1, default=str))

    acceptance = {
        "more_names": (summary.get(PRIMARY, {}).get("n", 0)
                       >= 2 * max(summary.get("incumbent", {}).get("n", 0), 1)),
        "beats_the_incumbent": bool(
            summary.get(PRIMARY, {}).get("cagr", float("-inf"))
            >= summary.get("incumbent", {}).get("cagr", float("-inf"))
            and summary.get(PRIMARY, {}).get("sharpe_trade", float("-inf"))
            >= summary.get("incumbent", {}).get("sharpe_trade", float("-inf"))),
        "choosing_beats_not_choosing": bool(
            summary.get(PRIMARY, {}).get("cagr", float("-inf"))
            > summary.get("random_pick", {}).get("cagr", float("-inf"))),
        "still_pays": bool((summary.get(PRIMARY, {}).get("breakeven_alpha") or 1.0) <= 0.45),
        "defined_risk_holds": summary.get(PRIMARY, {}).get("worse_than_debit", 1) == 0,
        "universe_floor": summary.get(PRIMARY, {}).get("n", 0) >= 250,
    }
    print(f"[EXP-133] acceptance: {acceptance}", flush=True)
    (RESULTS / "acceptance.json").write_text(json.dumps(acceptance, indent=1))

    already = set(lib.ledger_read().query("stage == 'ran'")["spec_hash"])
    for arm in build_mod.ARMS:
        book = books[arm]
        if book is None or book.empty:
            print(f"[EXP-133] {arm}: nothing traded, no evaluation", flush=True)
            continue
        is_primary = arm == PRIMARY
        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["structure"] = f"grid cell: {arm}"
            cell["grid_cell"] = True
        run_dir = HERE if is_primary else HERE / "arms" / arm
        for sub in ("results", "figures"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        result = evaluate(
            cell, book, gate=None, run_dir=run_dir,
            # A shifted-date reprice would need the forecast, the ladder and the
            # width rules all re-derived at the shifted date — the structure is
            # a function of the decision, not a fixed contract set — so the same
            # exemption EXP-125/126 took.
            repricer=None,
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[RESULTS / "candidates.parquet"],
            extra_sections=lambda r, a=arm: sections(
                summary, funnels, tallies, decided, events, meta, a, consistency,
                acceptance, defended),
            write_report=True,
        )
        if not record:
            print(f"[EXP-133] {arm}: --no-ledger, no row written", flush=True)
        elif lib.spec_hash(cell) in already:
            print(f"[EXP-133] {arm}: ledger row already recorded", flush=True)
        else:
            lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-133] {arm}: report {result.report_path}", flush=True)


if __name__ == "__main__":
    main(record="--no-ledger" not in sys.argv)
