#!/usr/bin/env python3
"""EXP-129 — a P&L gate for the twin peak, against the constant it replaces.

    python3 experiments/EXP-129_predicted_p_l_by_simulation_repricing_a/run.py

Four stages, and the first two decide whether the rest means anything.

  STAGE 0  repricing validation. Feed the REALIZED move and the REALIZED crush
           into the pricer and compare its exit value to the one the chains
           actually quoted. This isolates repricing error from distributional
           error, and it is the only stage that can tell "the simulation is
           wrong about the world" apart from "the world was uncertain". If the
           pricer cannot reproduce a known outcome it cannot integrate over an
           unknown one, and nothing below is worth reading.
  STAGE 1  simulate every candidate: paired draws, repriced legs, expected
           return and win probability.
  STAGE 2  apply every gate — the primary, the P&L sweep, and the arithmetic
           sweep of the incumbent's own constant — and score each resulting
           book on CAGR, Sharpe, years positive.
  STAGE 3  the two curves, read at MATCHED SELECTIVITY. A gate taking 200
           trades and one taking 2,000 are not comparable on CAGR, and the
           honest comparison is at equal gated-event count.

Nothing here promotes anything. The primary is exp_pnl_sim > 0 and every grid
cell is secondary; the threshold families are reported as curves with the
primary and incumbent marked, never as a leaderboard.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from engine.data.features import tier4  # noqa: E402
from engine.evaluate import (  # noqa: E402
    _max_drawdown,
    build_equity,
    by_year_table,
    cagr,
    trade_stats,
    years_positive,
)
from engine.features import load_panel  # noqa: E402
from engine.models.training import iv_crush  # noqa: E402
from experiments import lib  # noqa: E402

from simulate import DRAWS, ResidualPool, black_scholes_put, parse_legs, simulate_event  # noqa: E402

RESULTS = HERE / "results"
MID = 0.5
#: Fixed-fraction sizing, matching EXP-126's equity settings so the two books
#: are comparable without anyone re-deriving what "CAGR" meant there.
FRACTION = 0.05

#: EXP-126 priced this universe; re-pricing it from chains would take hours and
#: change nothing. The artifact carries the entry-rule terms as SEPARATE flags,
#: which is what makes a universe without the reward term recoverable at all.
SOURCE = (
    ROOT / "experiments" / "EXP-126_five_strikes_or_seven_letting_each_event"
    / "results" / "trades_five_wide.parquet"
)


def _curve(equity) -> pd.Series | None:
    """The date-indexed equity series ``build_equity`` returns under "equity".

    Named "equity", not "curve" — reading the wrong key returned None silently
    and every CAGR in the first run came back n/a, which is exactly the kind of
    missing number a reader skims past.
    """
    series = equity.get("equity") if isinstance(equity, dict) else None
    if series is None or len(series) < 2:
        return None
    return pd.Series(np.asarray(series, dtype=float), index=pd.to_datetime(series.index))


# --------------------------------------------------------------------------
# inputs
# --------------------------------------------------------------------------


def candidate_pool() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """``(all alphas, mid rows, funnel)`` for events passing spread AND mcap.

    The reward term is deliberately NOT applied: it is the rule under test, so
    it cannot also define the sample it is tested on.
    """
    trades = pd.read_parquet(SOURCE)
    trades["event_date"] = pd.to_datetime(trades["event_date"])
    mid = trades[np.isclose(trades["fill_alpha"].astype(float), MID)]
    keep = mid["f_spread"].fillna(False).astype(bool) & mid["f_mcap"].fillna(False).astype(bool)
    pool = mid[keep].copy()
    funnel = {
        "priced": int(mid["event_id"].nunique()),
        "f_spread": int(mid["f_spread"].fillna(False).sum()),
        "f_mcap": int(mid["f_mcap"].fillna(False).sum()),
        "candidates": int(len(pool)),
        "candidate_tickers": int(pool["ticker"].nunique()),
        "incumbent_passes": int(mid["passes"].fillna(False).sum()),
    }
    return trades, pool, funnel


def attach_forecasts(pool: pd.DataFrame, log=print) -> pd.DataFrame:
    """Join Tier-4's two forecasts and the pre-print vol the crush multiplies."""
    forecasts = tier4.load_forecasts()[
        ["ticker", "event_date", "pred_abs_move", "pred_iv_crush_30"]
    ]
    crush = iv_crush.crush_frame()[["ticker", "event_date", "pre_iv30", "crush_pct_iv30"]]
    out = pool.merge(forecasts, on=["ticker", "event_date"], how="left")
    out = out.merge(crush, on=["ticker", "event_date"], how="left")
    have = out["pred_abs_move"].notna() & out["pred_iv_crush_30"].notna() & out["pre_iv30"].notna()
    log(f"[EXP-129] {int(have.sum()):,} of {len(out):,} candidates carry both forecasts "
        f"and a pre-print vol ({have.mean():.1%})")
    return out


def residual_history(log=print) -> pd.DataFrame:
    """Paired out-of-sample errors for every event both models scored.

    Both residuals come from the same event, which is the entire mechanism by
    which the simulation inherits the dependence between the move and the crush
    without anyone estimating it.
    """
    panel = load_panel()[["ticker", "date", "abs_move"]].rename(columns={"date": "event_date"})
    forecasts = tier4.load_forecasts()[
        ["ticker", "event_date", "pred_abs_move", "pred_iv_crush_30"]
    ]
    crush = iv_crush.crush_frame()[["ticker", "event_date", "crush_pct_iv30"]]
    h = forecasts.merge(panel, on=["ticker", "event_date"], how="inner")
    h = h.merge(crush, on=["ticker", "event_date"], how="inner")
    h["err_move"] = h["abs_move"] - h["pred_abs_move"]
    h["err_crush"] = h["crush_pct_iv30"] - h["pred_iv_crush_30"]
    h = h.dropna(subset=["err_move", "err_crush", "pred_abs_move"])
    log(f"[EXP-129] residual history: {len(h):,} paired (move, crush) errors "
        f"{h.event_date.min().date()}..{h.event_date.max().date()}")
    return h


# --------------------------------------------------------------------------
# stage 0 — can the pricer reproduce an outcome it is told?
# --------------------------------------------------------------------------


def stage_0(rows: pd.DataFrame, log=print) -> dict:
    """Reprice at the REALIZED move and crush; compare to the quoted exit value.

    Any error here is the pricer's, not the forecast's. It is the ceiling on
    how good the simulated expectation can be, and it is reported before any
    expectation is computed so the two cannot be conflated later.
    """
    have = rows.dropna(subset=["pre_iv30", "crush_pct_iv30", "spot_entry", "spot_exit",
                               "exit_value", "entry_cost"])
    have = have[have["entry_cost"] > 0]
    modelled, actual = [], []
    for row in have.itertuples(index=False):
        legs = parse_legs(row.legs)
        exits = legs.get("exit") or []
        if not exits:
            continue
        dte = float(np.median([float(leg.get("dte", np.nan)) for leg in exits]))
        if not np.isfinite(dte) or dte < 0:
            continue
        vol = (float(row.pre_iv30) / 100.0) * (1.0 + float(row.crush_pct_iv30) / 100.0)
        value = 0.0
        for leg in exits:
            strike, qty = float(leg.get("strike", np.nan)), float(leg.get("qty", 0.0))
            if not np.isfinite(strike) or qty == 0:
                continue
            side = 1.0 if str(leg.get("side", "")).lower() == "sell" else -1.0
            value += side * qty * float(
                black_scholes_put(float(row.spot_exit), strike, dte / 365.0, vol)
            )
        modelled.append(value)
        actual.append(float(row.exit_value))
    modelled, actual = np.asarray(modelled), np.asarray(actual)
    if modelled.size == 0:
        return {"n": 0}
    err = modelled - actual
    out = {
        "n": int(modelled.size),
        "median_abs_err": float(np.median(np.abs(err))),
        "median_actual_exit": float(np.median(actual)),
        "bias": float(np.mean(err)),
        "r": float(np.corrcoef(modelled, actual)[0, 1]),
        "within_10pct_of_entry_cost": float(
            np.mean(np.abs(err) < 0.10 * have["entry_cost"].to_numpy()[: modelled.size])
        ),
    }
    log(f"[EXP-129] stage 0: repriced {out['n']:,} known outcomes — median |err| "
        f"${out['median_abs_err']:.3f} against a median exit of "
        f"${out['median_actual_exit']:.2f}, r={out['r']:.3f}, bias {out['bias']:+.3f}")
    return out


# --------------------------------------------------------------------------
# stage 1 — simulate
# --------------------------------------------------------------------------


def stage_1(rows: pd.DataFrame, pool: ResidualPool, *, paired=True, use_crush=True,
            use_move=True, draws=DRAWS, tag="primary", log=print) -> pd.DataFrame:
    started = time.time()
    out = []
    for row in rows.itertuples(index=False):
        sim = simulate_event(
            pd.Series(row._asdict()), pool, draws=draws, paired=paired,
            use_crush=use_crush, use_move=use_move, seed_key=tag,
        )
        out.append(sim or {})
    frame = pd.DataFrame(out, index=rows.index)
    done = frame["exp_pnl_sim"].notna() if "exp_pnl_sim" in frame else pd.Series(False, index=rows.index)
    log(f"[EXP-129] stage 1 [{tag}]: simulated {int(done.sum()):,} of {len(rows):,} "
        f"candidates [{time.time()-started:.0f}s]")
    return pd.concat([rows, frame], axis=1)


# --------------------------------------------------------------------------
# stage 2 — gate and score
# --------------------------------------------------------------------------


def book(all_alphas: pd.DataFrame, gated_ids, label: str) -> dict:
    """Score one gate's book on the measures promote.decide turns on."""
    rows = all_alphas[all_alphas["event_id"].isin(set(gated_ids))]
    mid = rows[np.isclose(rows["fill_alpha"].astype(float), MID)].copy()
    if mid.empty:
        return {"label": label, "n": 0}
    mid = mid.sort_values("entry_date")
    stats = trade_stats(mid["ret"].to_numpy(dtype=float), mid["event_date"])
    equity = build_equity(mid, FRACTION, mode="cashflow", record=True)
    curve = _curve(equity)
    yearly = by_year_table(mid)
    pos, total = years_positive(yearly)
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
            / pd.to_numeric(mid["cost"], errors="coerce").sum()
        ),
    }
    if curve is not None and len(curve) > 1:
        out["cagr"] = cagr(curve)
        out["max_drawdown"] = float(equity.get("max_dd", _max_drawdown(curve)))
        out["final_equity"] = float(equity.get("final", curve.iloc[-1]))
    return out


def gates(sim: pd.DataFrame, spec: dict) -> dict[str, pd.Index]:
    """Every registered gate, as a map from label to the event ids it admits."""
    ok = sim[sim["exp_pnl_sim"].notna()]
    out: dict[str, pd.Index] = {"primary_exp_pnl_gt_0": ok.loc[ok["exp_pnl_sim"] > 0, "event_id"]}
    for t in spec["grid"]["threshold"]:
        out[f"exp_pnl_gt_{t}"] = ok.loc[ok["exp_pnl_sim"] > float(t), "event_id"]
    # Quantile cutoffs from PRIOR YEARS only, applied forward.
    for q in spec["grid"]["quantile"]:
        picks = []
        for year in sorted(ok["year"].unique()):
            prior = ok[ok["year"] < year]["exp_pnl_sim"]
            if len(prior) < 100:
                continue
            cut = float(np.quantile(prior, 1.0 - float(q)))
            this = ok[(ok["year"] == year) & (ok["exp_pnl_sim"] >= cut)]
            picks.append(this["event_id"])
        out[f"top_{int(float(q)*100)}pct"] = (
            pd.concat(picks) if picks else ok["event_id"].head(0)
        )
    out["win_sim_gt_50"] = ok.loc[ok["win_sim"] > 0.5, "event_id"]
    # The incumbent's own constant, swept over the SAME candidate pool.
    ratio = sim["cost"] / sim["peak"]
    for r in spec["grid"]["reward_ratio"]:
        out[f"cost_lt_{r}_peak"] = sim.loc[ratio < float(r), "event_id"]
    return out


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    RESULTS.mkdir(exist_ok=True)
    results: dict = {"spec_hash": lib.spec_hash(spec)}

    trades, pool_rows, funnel = candidate_pool()
    results["funnel"] = funnel
    print(f"[EXP-129] candidates {funnel['candidates']:,} on "
          f"{funnel['candidate_tickers']} tickers; incumbent admits "
          f"{funnel['incumbent_passes']:,}", flush=True)

    rows = attach_forecasts(pool_rows)
    results["stage_0_repricing"] = stage_0(rows)

    history = residual_history()
    residuals = ResidualPool(history)

    ready = rows.dropna(subset=["pred_abs_move", "pred_iv_crush_30", "pre_iv30"]).copy()
    results["coverage"] = {
        "candidates": int(len(rows)),
        "with_both_forecasts": int(len(ready)),
        "share": float(len(ready) / max(len(rows), 1)),
    }

    sim = stage_1(ready, residuals, tag="primary")
    results["simulated"] = int(sim["exp_pnl_sim"].notna().sum())

    books = {}
    for label, ids in gates(sim, spec).items():
        books[label] = book(trades, ids, label)
    results["books"] = books

    # The arms that change the simulation rather than the threshold.
    arm_books = {}
    # The full 2x2 over which variable carries its uncertainty, plus the two
    # registered arms. `no_variance` is the control that says whether ANY of the
    # simulation is load-bearing: with both variables at their point forecasts
    # the shape sits on its own peak and the gate degenerates toward reward:risk.
    for tag, kwargs in (
        ("independent", {"paired": False}),
        ("move_only", {"use_crush": False}),
        ("crush_only", {"use_move": False}),
        ("no_variance", {"use_move": False, "use_crush": False}),
        ("draws_500", {"draws": 500}),
    ):
        alt = stage_1(ready, residuals, tag=tag, **kwargs)
        ids = alt.loc[alt["exp_pnl_sim"].notna() & (alt["exp_pnl_sim"] > 0), "event_id"]
        arm_books[tag] = book(trades, ids, tag)
        arm_books[tag]["mean_exp_pnl_sim"] = float(alt["exp_pnl_sim"].mean())
        arm_books[tag]["sd_exp_pnl_sim"] = float(alt["exp_pnl_sim"].std())
        # How close does this arm's ordering sit to the incumbent's? A gate that
        # reproduces reward:risk is reward:risk with extra steps.
        ok = alt.dropna(subset=["exp_pnl_sim", "rr"])
        arm_books[tag]["spearman_vs_reward_risk"] = float(
            ok["exp_pnl_sim"].corr(ok["rr"], method="spearman")
        ) if len(ok) > 100 else None
    results["arms"] = arm_books
    results["primary_sim_summary"] = {
        "mean_exp_pnl_sim": float(sim["exp_pnl_sim"].mean()),
        "sd_exp_pnl_sim": float(sim["exp_pnl_sim"].std()),
        "median_win_sim": float(sim["win_sim"].median()),
    }

    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=1, default=str))
    sim[["event_id", "ticker", "event_date", "year", "cost", "peak", "rr", "ret",
         "exp_pnl_sim", "win_sim", "sim_p10", "sim_p90", "dte_exit", "pool_n"]].to_csv(
        RESULTS / "simulated.csv", index=False)

    print("\n{:<26} {:>6} {:>8} {:>9} {:>8} {:>7}".format(
        "gate", "n", "mean", "CAGR", "Sharpe", "yrs+"))
    for label, b in books.items():
        if not b.get("n"):
            continue
        print("{:<26} {:>6,} {:>7.2f}% {:>8} {:>8} {:>4}/{}".format(
            label, b["n"], 100 * b["mean"],
            f"{100*b['cagr']:.2f}%" if b.get("cagr") is not None else "n/a",
            f"{b['sharpe_trade']:.2f}" if b.get("sharpe_trade") is not None else "n/a",
            b["years_positive"], b["years"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
