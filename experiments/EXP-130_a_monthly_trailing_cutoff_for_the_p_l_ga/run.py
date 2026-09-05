#!/usr/bin/env python3
"""EXP-130 — a monthly trailing cutoff for the P&L gate.

    python3 experiments/EXP-130_a_monthly_trailing_cutoff_for_the_p_l_ga/run.py

Only the CLOCK changes. ``exp_pnl_sim`` is read from EXP-129's stored output —
not recomputed — so any difference between these books is the cutoff rule and
nothing else. Re-simulating would reintroduce Monte Carlo noise into a
comparison whose whole point is to isolate one variable.

The annual rule and the incumbent are rerun HERE on identical rows rather than
quoted from EXP-129, because a comparison against a number computed elsewhere
is a comparison against a number, not against a rule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent

from engine.evaluate import (  # noqa: E402
    alpha_sweep, breakeven_alpha_from_sweep, build_equity, by_year_table,
    cagr, trade_stats, years_positive,
)
from experiments import lib  # noqa: E402

RESULTS = HERE / "results"
MID, FRACTION = 0.5, 0.05
EXP129 = ROOT / "experiments" / "EXP-129_predicted_p_l_by_simulation_repricing_a"
TRADES = (ROOT / "experiments" / "EXP-126_five_strikes_or_seven_letting_each_event"
          / "results" / "trades_five_wide.parquet")


# --------------------------------------------------------------------------
# the cutoff rules
# --------------------------------------------------------------------------


def monthly_trailing(sim: pd.DataFrame, *, window_months: int = 12,
                     quantile: float = 0.20, min_window: int = 100):
    """Admitted ids, plus the cutoff series, for a trailing-window rule.

    At month M the bar is the ``1 - quantile`` percentile of every gateable
    event dated in ``[M - window, M)`` — strictly before M begins, which is the
    same month boundary Tier 4 refits its folds on. A month whose window holds
    fewer than ``min_window`` events is not gated at all: its events are
    ungateable, never admitted by default.
    """
    ok = sim.dropna(subset=["exp_pnl_sim"]).sort_values("event_date")
    dates = ok["event_date"].to_numpy()
    values = ok["exp_pnl_sim"].to_numpy(dtype=float)
    months = ok["event_date"].dt.to_period("M")

    admitted, cutoffs, ungated = [], [], 0
    for month in sorted(months.unique()):
        start = month.to_timestamp()
        window_start = start - pd.DateOffset(months=window_months)
        prior = (dates >= np.datetime64(window_start)) & (dates < np.datetime64(start))
        rows = ok[months == month]
        if int(prior.sum()) < min_window:
            ungated += len(rows)
            continue
        cut = float(np.quantile(values[prior], 1.0 - quantile))
        cutoffs.append({"month": str(month), "cutoff": cut,
                        "window_n": int(prior.sum()), "candidates": int(len(rows)),
                        "admitted": int((rows["exp_pnl_sim"] >= cut).sum())})
        admitted.append(rows.loc[rows["exp_pnl_sim"] >= cut, "event_id"])
    ids = pd.concat(admitted) if admitted else ok["event_id"].head(0)
    return ids, pd.DataFrame(cutoffs), ungated


def annual_prior(sim: pd.DataFrame, *, quantile: float = 0.20, min_prior: int = 100):
    """EXP-129's rule: the bar comes from every PRIOR CALENDAR YEAR."""
    ok = sim.dropna(subset=["exp_pnl_sim"])
    admitted, cutoffs, ungated = [], [], 0
    for year in sorted(ok["year"].unique()):
        prior = ok[ok["year"] < year]["exp_pnl_sim"]
        rows = ok[ok["year"] == year]
        if len(prior) < min_prior:
            ungated += len(rows)
            continue
        cut = float(np.quantile(prior, 1.0 - quantile))
        cutoffs.append({"month": f"{int(year)}-01", "cutoff": cut,
                        "window_n": int(len(prior)), "candidates": int(len(rows)),
                        "admitted": int((rows["exp_pnl_sim"] >= cut).sum())})
        admitted.append(rows.loc[rows["exp_pnl_sim"] >= cut, "event_id"])
    ids = pd.concat(admitted) if admitted else ok["event_id"].head(0)
    return ids, pd.DataFrame(cutoffs), ungated


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def book(trades: pd.DataFrame, ids, label: str) -> dict:
    rows = trades[trades["event_id"].isin(set(ids))]
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)].sort_values("entry_date")
    if mid.empty:
        return {"label": label, "n": 0}
    stats = trade_stats(mid["ret"].to_numpy(dtype=float), mid["event_date"])
    equity = build_equity(mid, FRACTION, mode="cashflow", record=True)
    curve = equity.get("equity")
    yearly = by_year_table(mid)
    pos, total = years_positive(yearly)
    sweep = alpha_sweep(rows)
    out = {
        "label": label,
        "n": int(len(mid)),
        "tickers": int(mid["ticker"].nunique()),
        "mean": float(mid["ret"].mean()),
        "win": float((mid["ret"] > 0).mean()),
        "sharpe_trade": stats.get("sharpe_trade"),
        "years_positive": int(pos),
        "years": int(total),
        "return_on_capital": float(
            pd.to_numeric(mid["pnl"], errors="coerce").sum()
            / pd.to_numeric(mid["cost"], errors="coerce").sum()),
        "max_drawdown": float(equity.get("max_dd", float("nan"))),
        "breakeven_alpha": breakeven_alpha_from_sweep(sweep),
        "by_year": {str(k): {"n": int(v.get("n", 0)), "mean": float(v.get("mean", float("nan")))}
                    for k, v in yearly.items()},
    }
    if curve is not None and len(curve) > 1:
        out["cagr"] = cagr(pd.Series(np.asarray(curve, dtype=float),
                                     index=pd.to_datetime(curve.index)))
    return out


def within_year_movement(cutoffs: pd.DataFrame) -> float | None:
    """Mean within-calendar-year SD of the cutoff — the adaptation criterion.

    An annual rule is constant inside a year by construction, so this is exactly
    zero for it. Reporting it makes "adapts faster" a measured claim rather than
    a description of the mechanism.
    """
    if cutoffs.empty:
        return None
    c = cutoffs.copy()
    c["year"] = c["month"].str.slice(0, 4)
    per = c.groupby("year")["cutoff"].std()
    return float(per.mean(skipna=True)) if len(per) else None


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    RESULTS.mkdir(exist_ok=True)

    sim = pd.read_csv(EXP129 / "results" / "simulated.csv")
    sim["event_date"] = pd.to_datetime(sim["event_date"])
    trades = pd.read_parquet(TRADES)
    trades["event_date"] = pd.to_datetime(trades["event_date"])
    print(f"[EXP-130] {len(sim):,} simulated candidates from EXP-129, "
          f"{int(sim.exp_pnl_sim.notna().sum()):,} gateable", flush=True)

    results: dict = {"spec_hash": lib.spec_hash(spec), "books": {}, "cutoffs": {},
                     "ungateable": {}, "adaptation": {}}

    def record(label, ids, cuts, ungated):
        results["books"][label] = book(trades, ids, label)
        results["cutoffs"][label] = cuts.to_dict(orient="records")
        results["ungateable"][label] = int(ungated)
        results["adaptation"][label] = within_year_movement(cuts)

    ids, cuts, un = monthly_trailing(sim, window_months=12, quantile=0.20)
    record("primary_monthly_12m_top20", ids, cuts, un)
    for months in spec["grid"]["window_months"]:
        ids, cuts, un = monthly_trailing(sim, window_months=int(months), quantile=0.20)
        record(f"monthly_{months}m_top20", ids, cuts, un)
    for q in spec["grid"]["quantile"]:
        ids, cuts, un = monthly_trailing(sim, window_months=12, quantile=float(q))
        record(f"monthly_12m_top{int(float(q)*100)}", ids, cuts, un)

    ids, cuts, un = annual_prior(sim, quantile=0.20)
    record("annual_prior_years_top20", ids, cuts, un)

    ratio = sim["cost"] / sim["peak"]
    record("incumbent_cost_lt_half_peak",
           sim.loc[ratio < 0.5, "event_id"], pd.DataFrame(), 0)

    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=1, default=str))

    print(f"\n{'rule':30}{'n':>6}{'mean':>8}{'CAGR':>9}{'Sharpe':>8}{'maxDD':>8}"
          f"{'RoC':>8}{'be_a':>7}{'yrs+':>7}{'cutoff SD':>11}")
    for label, b in results["books"].items():
        if not b.get("n"):
            continue
        adapt = results["adaptation"].get(label)
        print(f"{label:30}{b['n']:>6,}{100*b['mean']:>7.2f}%"
              f"{(100*b['cagr']):>8.2f}%{b['sharpe_trade']:>8.2f}"
              f"{100*b['max_drawdown']:>7.1f}%{100*b['return_on_capital']:>7.2f}%"
              f"{(b['breakeven_alpha'] or float('nan')):>7.3f}"
              f"{b['years_positive']:>4}/{b['years']}"
              f"{('n/a' if adapt is None else f'{adapt:.4f}'):>11}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
