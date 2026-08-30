"""The Phase 2 evaluation suite: one standardized path from candidate to evidence.

Every candidate — a model, a gate, a structure variant, a parameter change —
runs through the same stages in the same order, so two experiments' reports
are comparable and a promotion decision is a diff, not an argument:

1. **Backtest** — the candidate's trades at the fill-alpha grid
   {0, 0.25, 0.5, 0.75, 1.0}; worst / mid / best side by side plus the
   **breakeven alpha**, the margin of safety on the mid-fill assumption.
2. **Walk-forward** — expanding window by calendar year. Anything tunable
   refits inside the loop on years < Y only; year Y is traded once, OOS.
   Headline numbers come from this stage and no other.
3. **Monte Carlo** — block bootstrap (block = 20 trades, preserving
   earnings-week clustering) on the walk-forward OOS sequence: P(final
   loss), drawdown percentiles, terminal-equity distribution, and the sizing
   curve at {2%, 5%, 10%, 20%} so position size is chosen from MC, not vibes.
4. **Stress battery** — crisis replays, tail injection for short legs,
   entry/exit slippage days, stale-earnings-date simulation, IV-regime split.
5. **Metrics dict** — one set of canonical keys, identical across all
   strategies, so the leaderboard is comparable.

The entry point is :func:`evaluate`. It takes a *spec* (the pre-registered
description of what is being tried, parsed from an experiment's ``spec.yaml``)
and a *trade set* (a priced frame in :mod:`engine.replay` output shape, one
row per event × fill alpha), runs the stages, writes the run artifacts under
the experiment's ``results/`` directory when one is given, and renders the
Phase 4 report. An experiment without a report does not exist.

**Pre-registration is enforced here, not by convention.** When the run is
attached to an experiment directory, every invocation is appended to
``results/run_log.jsonl`` and the OOS stage refuses to run if the spec carries
no ``preregistered_at`` or one later than the first recorded run — a spec
edited after seeing results is a spec that has never been tested.

**Equity construction.** Two documented modes:

``cashflow`` (default)
    Chronological by entry date; at each entry a fraction of *current* equity
    is committed (``contracts = fraction × equity / entry_cost``), the debit is
    paid at entry and the exit value credited at exit. Overlapping positions
    are allowed and counted (max concurrency is reported). This is the mode
    new experiments report.

``sequential``
    Each trade compounds equity by ``(1 + fraction × ret)`` in entry order,
    ignoring overlap. This is how the pre-engine EXP-050 equity curve was
    built; the mode exists so the harness can reproduce that evidence exactly
    (the regression in ``checks/phase2_checks.py``). Using it for a new
    experiment overstates compounding whenever trades overlap, so the report
    flags it.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.calibrate import brier as brier_score
from engine.calibrate import brier_skill as brier_skill_score

__all__ = [
    "METRIC_KEYS",
    "ALPHA_GRID",
    "SIZING_FRACTIONS",
    "MC_BLOCK",
    "MC_PATHS",
    "REGIME_WINDOWS",
    "EvaluationError",
    "PreregistrationError",
    "Gate",
    "EvalResult",
    "spec_hash",
    "trade_stats",
    "alpha_sweep",
    "breakeven_alpha_from_sweep",
    "build_equity",
    "dollar_weighted_return",
    "transaction_log",
    "reconcile_transaction_log",
    "walk_forward",
    "monte_carlo",
    "stress_regimes",
    "stress_iv_regime",
    "stress_tail_injection",
    "stress_slippage",
    "stress_stale_dates",
    "evaluate",
]

#: Canonical metric keys — identical across all strategies so the leaderboard
#: is comparable. A results dict that misses one of these is a bug, and
#: evaluate() asserts it rather than describing it.
METRIC_KEYS = (
    "n", "mean", "median", "std", "win_rate", "profit_factor",
    "sharpe_trade", "sharpe_equity", "sortino", "max_dd", "tail_ratio",
    "dollar_weighted", "by_year", "breakeven_alpha", "capacity", "deployment", "mc",
)

#: The fill alphas every result is reported at (worst/mid/best plus the quarter
#: points that make the degradation curve a lookup). Mirrors engine.replay.
ALPHA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

#: The sizing curve fractions. 5% is the program's base case (the plan's
#: go-live sizing); 2/10/20% bracket it so the curve shape is visible.
SIZING_FRACTIONS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.20)

#: Block length for the block bootstrap, in trades. Twenty trades is about one
#: earnings week of a full-universe strategy — long enough that a block keeps
#: the cross-name clustering a real print produces.
MC_BLOCK = 20

#: Bootstrap paths. 1,000 matches the existing evidence; P(loss) estimates
#: below ~1% are noise at this resolution, and the reports say so.
MC_PATHS = 1000

#: Crisis windows for the regime-replay stress, keyed by name. Dates are
#: inclusive. The 2018Q4 unwind, the 2020 crash, and the 2022 vol year are the
#: three regimes the edge is most suspected of leaning on; the worst realized
#: earnings weeks are computed from the SPY series at run time.
REGIME_WINDOWS = {
    "2018Q4": (pd.Timestamp("2018-10-01"), pd.Timestamp("2018-12-31")),
    "2020-02_04": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    "2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}

#: Column contract for the priced trade frame evaluate() consumes. This is the
#: engine.replay output shape minus the legs blob; extra columns (features,
#: realized_move, spy_vol20, source) are passed through to gates and stress.
REQUIRED_TRADE_COLUMNS = (
    "event_id", "ticker", "event_date", "entry_date", "exit_date",
    "fill_alpha", "entry_cost", "exit_value", "ret",
)


class EvaluationError(ValueError):
    """The candidate or the spec could not be evaluated as given."""


class PreregistrationError(RuntimeError):
    """The OOS stage refused to run: the spec was not pre-registered in time."""


# --------------------------------------------------------------------------
# specs
# --------------------------------------------------------------------------


def spec_hash(spec: Mapping[str, Any]) -> str:
    """Stable identity of an evaluated spec, for the ledger and the cache.

    Everything except ``id`` and ``preregistered_at`` goes into the hash: two
    runs of the same hypothesis on the same snapshot are the same spec even if
    the scaffolder stamped them on different days, and a grid cell differs from
    the primary spec exactly when its parameters do.
    """
    doc = {k: v for k, v in spec.items() if k not in ("id", "preregistered_at")}
    payload = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


# --------------------------------------------------------------------------
# per-trade statistics
# --------------------------------------------------------------------------


def _annual_trades_per_year(event_dates: pd.Series) -> float:
    """Average trades per year over the span actually covered.

    The span is the time between the first and the last event, not the count of
    distinct years: a strategy that traded two busy years out of eight is a
    two-trades-per-year strategy for Sharpe purposes, and counting calendar
    years it skipped would inflate its annualized Sharpe by 4x.
    """
    if len(event_dates) < 2:
        return float(len(event_dates))
    span_days = max((event_dates.max() - event_dates.min()).days, 1)
    return len(event_dates) / (span_days / 365.25)


def trade_stats(rets: Sequence[float], event_dates: pd.Series | None = None) -> dict[str, float]:
    """The per-trade block of the canonical metrics dict.

    ``event_dates`` is only needed for the trades-per-year annualization; when
    absent, Sharpe is annualized assuming the trades span one year each.
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = int(r.size)
    out: dict[str, float] = {
        "n": n,
        "mean": float(np.nan) if n == 0 else float(r.mean()),
        "median": float(np.nan) if n == 0 else float(np.median(r)),
        "std": float(np.nan) if n < 2 else float(r.std(ddof=1)),
        "win_rate": float(np.nan) if n == 0 else float((r > 0).mean()),
    }
    wins = r[r > 0]
    losses = r[r < 0]
    out["profit_factor"] = (
        float(wins.sum() / abs(losses.sum())) if wins.size and losses.size and losses.sum() != 0
        else float(np.nan)
    )
    tpy = _annual_trades_per_year(pd.Series(event_dates)) if event_dates is not None and len(event_dates) else 1.0
    tpy = max(tpy, 1e-9)
    if n >= 2 and r.std(ddof=1) > 0:
        out["sharpe_trade"] = float(r.mean() / r.std(ddof=1) * np.sqrt(tpy))
    else:
        out["sharpe_trade"] = float(np.nan)
    downside = r[r < 0]
    if n >= 2 and downside.size and np.sqrt((downside ** 2).mean()) > 0:
        out["sortino"] = float(r.mean() / np.sqrt((downside ** 2).mean()) * np.sqrt(tpy))
    else:
        out["sortino"] = float(np.nan)
    if wins.size and losses.size:
        out["tail_ratio"] = float(np.percentile(wins, 95) / abs(np.percentile(losses, 5)))
    else:
        out["tail_ratio"] = float(np.nan)
    return out


def dollar_weighted_return(trades: pd.DataFrame) -> float:
    """Total P&L over total premium paid — the capital-weighted return.

    The equal-weighted mean answers "what did the average TRADE return"; this
    answers "what did the average DOLLAR return". They diverge exactly when the
    edge sits in the cheapest contracts, and the divergence is the thing worth
    seeing: a $1.10 straddle and a $9.60 straddle count the same in the mean,
    while fixed-fraction sizing buys nine times as many of the former —
    a capacity claim disguised as a return.
    """
    if not len(trades) or not {"entry_cost", "exit_value"} <= set(trades.columns):
        return float("nan")
    cost = pd.to_numeric(trades["entry_cost"], errors="coerce")
    value = pd.to_numeric(trades["exit_value"], errors="coerce")
    ok = cost.notna() & value.notna() & (cost > 0)
    if not ok.any():
        return float("nan")
    return float((value[ok] - cost[ok]).sum() / cost[ok].sum())


def by_year_table(trades: pd.DataFrame, ret_col: str = "ret") -> dict[str, dict[str, float]]:
    """Per-year {n, mean, win_rate} — the lumpiness view every report carries."""
    out: dict[str, dict[str, float]] = {}
    if trades.empty:
        return out
    years = pd.to_datetime(trades["event_date"]).dt.year
    for year, idx in years.groupby(years).groups.items():
        r = trades.loc[idx, ret_col].to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        out[str(int(year))] = {
            "n": int(r.size),
            "mean": float(r.mean()) if r.size else float(np.nan),
            "win_rate": float((r > 0).mean()) if r.size else float(np.nan),
        }
    return dict(sorted(out.items()))


def capacity_note(trades: pd.DataFrame, alpha: float = 0.5) -> dict[str, Any]:
    """Capacity notes: spread width at the traded strikes (§P2.2.5).

    Sizing decisions must not be made on mean return alone: a +5%/trade edge
    quoted through a 4% wide spread on a thinly-traded name is not the same
    asset as the same edge through a 0.4% spread. This reports the relative
    spread of the traded legs (the worst leg governs executability) and the
    wide-market fraction. Volume at the traded strikes is NOT verifiable from
    the chain source (ORATS /hist/strikes carries bid/ask but no volume), and
    the note says so instead of implying otherwise.
    """
    rows = trades[np.isclose(trades["fill_alpha"], alpha)] if "fill_alpha" in trades.columns else trades
    out: dict[str, Any] = {"available": False}
    if rows.empty or "legs" not in rows.columns:
        out["note"] = "no legs blob on this trade set; capacity not measurable"
        return out

    worst_rel: list[float] = []
    for blob in rows["legs"].to_numpy():
        if not isinstance(blob, str):
            continue
        try:
            doc = json.loads(blob)
        except ValueError:
            continue
        rels = []
        for leg in (doc.get("entry") or []):
            bid, ask = float(leg.get("bid", 0.0)), float(leg.get("ask", 0.0))
            mid = (bid + ask) / 2.0
            if mid > 0 and ask >= bid >= 0:
                rels.append((ask - bid) / mid)
        if rels:
            worst_rel.append(max(rels))

    if not worst_rel:
        out["note"] = "legs blob carried no usable entry quotes"
        return out
    arr = np.asarray(worst_rel)
    # Where does the money come from, by how wide the market was? An edge that
    # lives in the widest-quoted names is an edge in the fill assumption, not
    # in the trade: mid is a real price only where the market is tight enough
    # for mid to mean something.
    pnl_by_spread: dict[str, Any] = {}
    if len(arr) == len(rows) and {"entry_cost", "exit_value"} <= set(rows.columns):
        frame = pd.DataFrame({
            "rel": arr,
            "pnl": pd.to_numeric(rows["exit_value"], errors="coerce").to_numpy()
                   - pd.to_numeric(rows["entry_cost"], errors="coerce").to_numpy(),
        }).dropna()
        if len(frame) >= 25 and frame["rel"].nunique() >= 5:
            try:
                buckets = pd.qcut(frame["rel"], 5, labels=False, duplicates="drop")
            except ValueError:
                buckets = None
            if buckets is not None and frame["pnl"].sum() != 0:
                totals = frame.groupby(buckets)["pnl"].sum()
                net = totals.sum()
                pnl_by_spread = {
                    "widest_quintile_share": float(totals.iloc[-1] / net),
                    "tightest_two_quintiles_share": float(totals.iloc[:2].sum() / net),
                    "median_rel_spread_widest": float(
                        frame.loc[buckets == buckets.max(), "rel"].median()),
                    "median_rel_spread_tightest": float(
                        frame.loc[buckets == buckets.min(), "rel"].median()),
                }
    out.update({
        "available": True,
        "pnl_by_spread": pnl_by_spread,
        "n": int(arr.size),
        "mean_rel_spread": float(arr.mean()),
        "p95_rel_spread": float(np.percentile(arr, 95)),
        "wide_market_frac": float(rows["wide_market"].mean())
        if "wide_market" in rows.columns else None,
        "note": ("spread-based capacity only: the chain source carries no volume "
                 "at the traded strikes"),
    })
    return out


def calibration_block(proba: Sequence[float], outcome: Sequence[float],
                      n_bins: int = 10, min_n: int = 50) -> dict[str, Any]:
    """Reliability of a gate's P(win) against realized outcomes, OOS.

    ``proba`` and ``outcome`` (0/1) must already be out-of-sample — the
    walk-forward collects them.

    **The skill is imported, not re-derived.** ``engine.calibrate.brier_skill``
    is the program's one definition — the SKILL SCORE ``1 - brier/reference``,
    normalized by the base-rate forecaster's own Brier. An earlier version of
    this function computed the unnormalized difference ``reference - brier``
    instead, which is the same number scaled by ``base*(1-base) ~ 0.235``. That
    matters because promotion applies the decision record's floor
    (``MIN_BRIER_SKILL = -0.05``) to whatever this returns: on the unnormalized
    scale that floor is really -0.21, and the worst anti-calibration the record
    ever measured was -0.204 — so the gate would have passed the exact failure
    it exists to block. Two spellings of one metric is the whole bug; there is
    now one spelling, and it lives in engine.calibrate.
    """
    p = np.asarray(proba, dtype=float)
    y = np.asarray(outcome, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    n = int(p.size)
    if n < min_n:
        return {"available": False,
                "reason": f"only {n} OOS scored rows (< {min_n}); calibration not reliable"}

    base = float(y.mean())
    brier = brier_score(p, y)
    # The base-rate forecaster's Brier: mean((base - y)^2) == base*(1 - base)
    # for 0/1 outcomes. This is the `reference` engine.calibrate divides by.
    brier_base = float(base * (1.0 - base))

    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return {"available": False,
                "reason": "predicted probabilities degenerate (no spread to bin)"}
    bin_idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)

    deciles: list[dict[str, Any]] = []
    for b in range(len(edges) - 1):
        sel = bin_idx == b
        if not sel.any():
            continue
        deciles.append({
            "predicted": float(p[sel].mean()),
            "realized": float(y[sel].mean()),
            "n": int(sel.sum()),
        })
    preds = np.array([d["predicted"] for d in deciles])
    reals = np.array([d["realized"] for d in deciles])
    if len(deciles) >= 3 and preds.std() > 0 and reals.std() > 0:
        monotonicity = float(np.corrcoef(preds, reals)[0, 1])
    else:
        monotonicity = float(np.nan)
    return {
        "available": True,
        "n": n,
        "base_rate": base,
        "brier": brier,
        "brier_base_rate": brier_base,
        "brier_skill": brier_skill_score(p, y),
        "reliability_monotonicity": monotonicity,
        "deciles": deciles,
    }


# --------------------------------------------------------------------------
# fill-alpha sweep
# --------------------------------------------------------------------------


def alpha_sweep(trades: pd.DataFrame, alphas: Sequence[float] = ALPHA_GRID) -> dict[str, dict]:
    """Per-alpha per-trade stats on the full (unselected) trade set.

    The anti-selection guard lives here: this sweep is computed on the whole
    universe the candidate priced, so the report can show the unselected base
    next to whatever subset the gate keeps — the S4 lesson, made structural.
    """
    out: dict[str, dict] = {}
    if trades.empty or "fill_alpha" not in trades.columns:
        return out
    for alpha in alphas:
        rows = trades[np.isclose(trades["fill_alpha"], float(alpha))]
        if rows.empty:
            continue
        stats = trade_stats(rows["ret"].to_numpy(), rows["event_date"])
        stats.pop("sharpe_trade", None)
        stats.pop("sortino", None)
        out[f"{float(alpha):.2f}"] = stats
    return out


def breakeven_alpha_from_sweep(sweep: Mapping[str, Mapping[str, float]]) -> float | None:
    """Alpha at which the interpolated mean return crosses zero.

    Per-trade P&L is linear in alpha for fixed legs, but the *return* (P&L over
    a debit that also moves with alpha) is not, so this interpolates the swept
    points rather than assuming the endpoints determine the curve — the guide's
    "linear interpolation of mean return across the sweep". Returns None when
    the curve never crosses zero inside the swept range (always positive or
    always negative): the caller reports 0.0 or 1.0 semantics explicitly.
    """
    points = sorted((float(a), float(s["mean"])) for a, s in sweep.items() if np.isfinite(s.get("mean", np.nan)))
    if len(points) < 2:
        return None
    for (a0, m0), (a1, m1) in zip(points, points[1:]):
        if m0 == 0.0:
            return a0
        if (m0 < 0 < m1) or (m1 < 0 < m0):
            return float(a0 + (0.0 - m0) * (a1 - a0) / (m1 - m0))
    return None


# --------------------------------------------------------------------------
# equity curves
# --------------------------------------------------------------------------


def build_equity(
    trades: pd.DataFrame,
    fraction: float,
    *,
    mode: str = "cashflow",
    ret_col: str = "ret",
    max_deployed: float | None = None,
    record: bool = False,
) -> dict[str, Any]:
    """Equity curve of a fixed-fraction sizing rule over priced trades.

    ``cashflow`` (default): chronological by entry date; each entry sizes off
    the current **marked** equity — ``cash + Σ contracts × entry_cost`` over
    the positions still open — i.e. net liquidation value with open positions
    marked at cost (the honest floor: no interpolation, no mid-life quotes
    needed). The debit is paid at entry, the exit value credited at exit, and
    the reported series is the marked equity, so drawdown measures P&L rather
    than how much capital happens to be deployed. Overlapping positions are
    allowed; ``max_concurrency`` reports how many were open at once.

    Marking at cost, not just tracking cash, is load-bearing: with a hundred
    concurrent positions a cash-only curve reports a ~100% "drawdown" on a
    trade set that cannot lose money, because everything deployed reads as
    money gone. It also fixes sizing — a broker sizes the next trade off net
    liquidation value, not off whatever cash remains after the open trades'
    debits.

    **Deployment cap.** Fixed-fraction sizing PER TRADE with many concurrent
    positions deploys ``fraction × concurrency`` of equity — "5% per trade"
    with 133 simultaneous positions is 665% notional financed by an implicit,
    uncharged margin loan. ``max_deployed`` (a fraction of marked equity,
    e.g. 1.0 = no leverage) caps total deployed notional: once the open
    positions reach the cap, new entries are skipped (counted in
    ``constrained_entries``) rather than levered up. The curve also reports
    ``peak_deployment`` and ``worst_cash`` so the leverage a run actually used
    is visible next to its returns — with or without the cap.

    ``sequential``: ``equity *= (1 + fraction × ret)`` per trade in entry
    order — the EXP-050 reference construction. It ignores overlap, so a new
    experiment reporting it overstates compounding; evaluate() flags the mode
    in the report.

    ``record=True`` additionally returns ``ledger``: one row per trade with the
    contracts bought, the equity it was sized off, the cash and deployment
    after it, and its contribution to the final equity. That is the audit trail
    behind the plotted curve — without it the chart is an assertion, and a
    reader who wants to check one trade has nowhere to look.
    """
    if mode not in ("cashflow", "sequential"):
        raise EvaluationError(f"unknown equity mode {mode!r}")
    if max_deployed is not None and not 0 < max_deployed:
        raise EvaluationError(f"max_deployed must be positive, got {max_deployed!r}")
    if trades.empty:
        return {"equity": pd.Series(dtype=float), "final": 1.0, "max_dd": 0.0,
                "max_concurrency": 0, "mode": mode, "fraction": fraction,
                "peak_deployment": 0.0, "worst_cash": 1.0, "constrained_entries": 0,
                "ledger": pd.DataFrame() if record else None}

    t = trades.sort_values(["entry_date", "exit_date"], kind="stable").reset_index(drop=True)
    entry_dates = pd.to_datetime(t["entry_date"])
    exit_dates = pd.to_datetime(t["exit_date"])
    rets = t[ret_col].to_numpy(dtype=float)

    if mode == "sequential":
        eq = 1.0
        marks: list[tuple[pd.Timestamp, float]] = [(entry_dates.iloc[0] - pd.Timedelta(days=1), 1.0)]
        seq_before: list[float] = []
        seq_after: list[float] = []
        for i, r in enumerate(rets):
            seq_before.append(eq)
            eq *= 1.0 + fraction * r
            seq_after.append(eq)
            marks.append((exit_dates.iloc[i], eq))
        curve = pd.Series(dict(marks)).sort_index()
        curve = curve[~curve.index.duplicated(keep="last")]
        return {
            "equity": curve,
            "final": float(curve.iloc[-1]),
            "max_dd": _max_drawdown(curve),
            "max_concurrency": int(_max_concurrency(entry_dates, exit_dates)),
            "mode": mode,
            "fraction": fraction,
            # Sequential compounding models no balance at all, so deployment
            # accounting does not apply; the keys stay for a uniform shape.
            "peak_deployment": float(np.nan),
            "worst_cash": float(np.nan),
            "constrained_entries": 0,
            "ledger": (pd.DataFrame({
                "seq": np.arange(len(t)),
                "equity_at_entry": seq_before,
                "equity_after_exit": seq_after,
                "pnl_contribution": np.asarray(seq_after) - np.asarray(seq_before),
                # No position sizing exists in this mode; the columns stay so
                # the log has one shape, and NaN says "not modelled" rather
                # than implying a contract count nobody computed.
                "contracts": np.nan, "notional_at_entry": np.nan,
                "cash_after_entry": np.nan, "deployed_after_entry": np.nan,
                "concurrency_at_entry": np.nan, "constrained": False,
                "proceeds_at_exit": np.nan,
            }, index=t.index) if record else None),
        }

    # cashflow mode: process exits before entries on the same day, so a closing
    # trade's credit is available to size an opening one — the conservative-
    # capital reading, and the one that matches a broker account.
    events: list[tuple[pd.Timestamp, int, int]] = []
    for i in range(len(t)):
        events.append((exit_dates.iloc[i], 0, i))
        events.append((entry_dates.iloc[i], 1, i))
    events.sort(key=lambda e: (e[0], e[1]))

    costs = t["entry_cost"].to_numpy(dtype=float)
    exits = t["exit_value"].to_numpy(dtype=float)

    def mark(cash: float, open_pos: dict[int, float]) -> float:
        # Net liquidation value with open positions marked at cost.
        return cash + sum(c * costs[i] for i, c in open_pos.items())

    cash = 1.0
    open_pos: dict[int, float] = {}
    marks = [(events[0][0] - pd.Timedelta(days=1), 1.0)]
    n = len(t)
    # Per-trade accounting, filled as the event loop passes each trade. These
    # are the columns that let a reader re-derive the plotted curve by hand.
    rec_contracts = np.zeros(n)
    rec_equity_entry = np.full(n, np.nan)
    rec_cash_entry = np.full(n, np.nan)
    rec_deployed_entry = np.full(n, np.nan)
    rec_concurrency = np.zeros(n, dtype=int)
    rec_constrained = np.zeros(n, dtype=bool)
    rec_proceeds = np.zeros(n)
    rec_equity_exit = np.full(n, np.nan)
    concurrency = 0
    max_concurrency = 0
    peak_deployment = 0.0
    worst_cash = 1.0
    constrained_entries = 0
    for date, kind, i in events:
        if kind == 0:  # exit
            contracts = open_pos.pop(i, 0.0)
            cash += contracts * exits[i]
            rec_proceeds[i] = contracts * exits[i]
            concurrency -= 1
        else:  # entry — size off the marked equity, not the cash balance
            equity_now = mark(cash, open_pos)
            rec_equity_entry[i] = equity_now
            if equity_now > 0 and costs[i] > 0:
                contracts = fraction * equity_now / costs[i]
                if max_deployed is not None:
                    deployed = sum(c * costs[j] for j, c in open_pos.items())
                    headroom = max_deployed * equity_now - deployed
                    if headroom <= 0:
                        contracts = 0.0
                        constrained_entries += 1
                        rec_constrained[i] = True
                    else:
                        capped = headroom / costs[i]
                        if capped < contracts:
                            contracts = capped
                            constrained_entries += 1
                            rec_constrained[i] = True
                if contracts > 0:
                    cash -= contracts * costs[i]
                    open_pos[i] = contracts
                rec_contracts[i] = contracts
            concurrency += 1
            rec_concurrency[i] = concurrency
        max_concurrency = max(max_concurrency, concurrency)
        equity_now = mark(cash, open_pos)
        deployed_now = sum(c * costs[j] for j, c in open_pos.items())
        if kind == 0:
            rec_equity_exit[i] = equity_now
        else:
            rec_cash_entry[i] = cash
            rec_deployed_entry[i] = deployed_now
        if equity_now > 0:
            peak_deployment = max(peak_deployment, deployed_now / equity_now)
            worst_cash = min(worst_cash, cash / equity_now)
        marks.append((date, equity_now))
    curve = pd.Series(dict(marks)).sort_index()
    curve = curve[~curve.index.duplicated(keep="last")]
    return {
        "equity": curve,
        "final": float(curve.iloc[-1]),
        "max_dd": _max_drawdown(curve),
        "max_concurrency": max_concurrency,
        "mode": mode,
        "fraction": fraction,
        "peak_deployment": peak_deployment,
        "worst_cash": worst_cash,
        "constrained_entries": constrained_entries,
        "ledger": (pd.DataFrame({
            "seq": np.arange(n),
            "contracts": rec_contracts,
            "notional_at_entry": rec_contracts * costs,
            "equity_at_entry": rec_equity_entry,
            "cash_after_entry": rec_cash_entry,
            "deployed_after_entry": rec_deployed_entry,
            "concurrency_at_entry": rec_concurrency,
            "constrained": rec_constrained,
            "proceeds_at_exit": rec_proceeds,
            "equity_after_exit": rec_equity_exit,
            # What this trade added to (or took from) final equity, in units of
            # starting equity. These sum to final - 1 by construction, which is
            # the reconciliation the report prints.
            "pnl_contribution": rec_proceeds - rec_contracts * costs,
        }, index=t.index) if record else None),
    }


#: Columns that identify a trade well enough to look it up in a chain file.
_LOG_IDENTITY = ("trade_id", "event_id", "ticker", "event_date", "session", "strategy",
                 "variant", "entry_date", "exit_date", "strike", "expiry", "dte_entry",
                 "fill_alpha", "entry_cost", "exit_value", "ret", "provenance")


def _flatten_legs(blob: Any, max_legs: int = 2) -> dict[str, Any]:
    """One trade's legs blob → flat ``entry_leg1_bid`` style columns.

    The point of the transaction log is that a person can take a row to the
    chain file it came from, so the quotes travel WITH the row: right, strike,
    expiry, dte, bid and ask for every leg at both ends. Recomputing
    ``entry_cost`` from those columns at the row's own alpha is the spot check.
    """
    out: dict[str, Any] = {}
    try:
        doc = json.loads(blob) if isinstance(blob, str) else (blob or {})
    except ValueError:
        return out
    for phase in ("entry", "exit"):
        legs = list(doc.get(phase) or [])
        for i in range(max_legs):
            prefix = f"{phase}_leg{i + 1}_"
            leg = legs[i] if i < len(legs) else {}
            for field_name in ("name", "right", "side", "qty", "strike", "expiry",
                               "dte", "bid", "ask", "price"):
                out[prefix + field_name] = leg.get(field_name)
            out[prefix + "wide"] = leg.get("wide_market")
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _leg_count(blobs: Sequence[Any], cap: int = 8) -> int:
    """Widest leg count in a trade set, so no leg is silently dropped.

    Today's three structures are two-legged; Phase 6's put diagonals and
    spreads need not be. Hard-coding 2 would drop a third leg from the log
    while the file still looked complete — the failure mode a log must not
    have.
    """
    widest = 0
    for blob in blobs:
        try:
            doc = json.loads(blob) if isinstance(blob, str) else (blob or {})
        except ValueError:
            continue
        widest = max(widest, len(doc.get("entry") or []), len(doc.get("exit") or []))
        if widest >= cap:
            return cap
    return max(widest, 1)


def transaction_log(trades: pd.DataFrame, equity: Mapping[str, Any],
                    scores: pd.DataFrame | None = None,
                    max_legs: int | None = None) -> pd.DataFrame:
    """Every trade behind a plotted equity curve, one row each.

    Identity + the quotes it was priced from + the equity accounting that put
    it on the chart. Ordered exactly as the equity engine processed it, so the
    curve can be re-derived from this file alone:
    ``equity_after_exit`` of the last row equals the curve's final value, and
    ``pnl_contribution`` sums to ``final - 1``.

    ``scores`` (the walk-forward gate's out-of-sample probabilities) is joined
    where present, because "why was this trade selected" is the second question
    anyone spot-checking a gated curve asks.
    """
    ledger = equity.get("ledger")
    if ledger is None:
        raise EvaluationError("build_equity(record=True) is required for a transaction log")
    ordered = trades.sort_values(["entry_date", "exit_date"], kind="stable")
    keep = [c for c in _LOG_IDENTITY if c in ordered.columns]
    log = ordered[keep].reset_index(drop=True)

    if "legs" in ordered.columns:
        legs = max_legs if max_legs is not None else _leg_count(ordered["legs"].to_numpy())
        flat = pd.DataFrame([_flatten_legs(b, legs) for b in ordered["legs"]])
        log = pd.concat([log, flat.reset_index(drop=True)], axis=1)

    accounting = ledger.reset_index(drop=True)
    log = pd.concat([log, accounting], axis=1)

    if scores is not None and len(scores) and "event_id" in log.columns:
        cols = [c for c in ("event_id", "proba") if c in scores.columns]
        if len(cols) == 2:
            log = log.merge(scores[cols].drop_duplicates("event_id").rename(
                columns={"proba": "gate_proba"}), on="event_id", how="left")

    log.insert(0, "row", np.arange(1, len(log) + 1))
    return log


def reconcile_transaction_log(log: pd.DataFrame, equity: Mapping[str, Any]) -> dict[str, Any]:
    """Does the log reproduce the curve it claims to explain?

    Computed and printed rather than assumed: a log that does not add up is
    worse than no log, because it looks like evidence.
    """
    final = float(equity.get("final", np.nan))
    series = log["pnl_contribution"] if "pnl_contribution" in log else pd.Series(dtype=float)
    contributions = float(series.sum()) if len(series) else float(np.nan)
    out = {
        "rows": int(len(log)),
        "final_equity": final,
        "sum_pnl_contribution": contributions,
        "implied_final": 1.0 + contributions,
        "abs_error": float(abs(1.0 + contributions - final)),
        "reconciles": bool(abs(1.0 + contributions - final) < 1e-6),
    }

    # Concentration, computed here because the log is the only place it CAN be
    # computed: a mean return says nothing about whether ten trades carried the
    # curve. Both readings are reported — the net share is the intuitive one
    # and is unstable when the net is near zero, so the gross reading (how few
    # trades make half the gains) is printed beside it.
    if len(series):
        ordered = series.reindex(series.abs().sort_values(ascending=False).index)
        gains = series[series > 0].sort_values(ascending=False)
        half = int((gains.cumsum() < gains.sum() / 2).sum() + 1) if len(gains) else 0
        out["concentration"] = {
            "top10_net_share": (float(ordered.head(10).sum() / contributions)
                                if abs(contributions) > 1e-9 else None),
            "top1_net_share": (float(ordered.head(1).sum() / contributions)
                               if abs(contributions) > 1e-9 else None),
            "trades_for_half_the_gains": half,
            "n_winners": int(len(gains)),
        }
    return out


def _max_concurrency(entry_dates: pd.Series, exit_dates: pd.Series) -> int:
    events = [(d, 1) for d in entry_dates] + [(d, -1) for d in exit_dates]
    events.sort(key=lambda e: (e[0], e[1]))  # exits before entries on a tie
    peak = cur = 0
    for _, delta in events:
        cur += delta
        peak = max(peak, cur)
    return peak


def _max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(abs(dd.min()))


def sharpe_equity(equity_curve: pd.Series) -> float:
    """Annualized Sharpe of an event-driven equity curve, on a daily grid.

    The curve only moves on entry/exit dates; between events it is flat, so the
    daily series carries real zero-return days. Including them (rather than
    pretending every day was an event day) is what keeps the annualization
    honest for a sparse strategy — a straddle program trades ~1.3 days a week,
    and Sharpe computed only on event days would read as if it traded daily.
    """
    if len(equity_curve) < 2:
        return float(np.nan)
    daily = equity_curve.reindex(
        pd.date_range(equity_curve.index.min(), equity_curve.index.max(), freq="D")
    ).ffill()
    rets = daily.pct_change().dropna().to_numpy(dtype=float)
    if rets.size < 2 or rets.std(ddof=1) == 0:
        return float(np.nan)
    return float(rets.mean() / rets.std(ddof=1) * np.sqrt(252))


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------


@dataclass
class Gate:
    """A walk-forward selector: ``fit`` on history, ``select`` the next year.

    ``fit(train)`` sees ONLY rows from years strictly before the year being
    traded — the harness enforces that, a gate cannot break it. ``select(rows)``
    returns a boolean mask of the trades it keeps for the test year. Both get
    the alpha = 0.5 view of the trades; fill convention is not a tunable.

    ``seen`` is the audit hook: the harness appends the max year every ``fit``
    received, and the leak-poison acceptance test asserts no fit ever saw the
    year it was selecting.
    """

    fit: Callable[[pd.DataFrame], None]
    select: Callable[[pd.DataFrame], pd.Series]
    name: str = "gate"
    seen: list = field(default_factory=list)
    #: Optional P(win) ∈ [0, 1] per row. When present, the walk-forward
    #: collects out-of-sample probabilities for every traded year and
    #: evaluate() turns them into the calibration block — which is what makes
    #: the promotion Brier-skill rule live instead of a permanent WARN. Gates
    #: without one are still evaluated; their calibration reports unavailable.
    predict_proba: Callable[[pd.DataFrame], np.ndarray] | None = None


def walk_forward(
    trades: pd.DataFrame,
    gate: Gate | None,
    *,
    min_train_years: int = 2,
    alpha: float = 0.5,
) -> dict[str, Any]:
    """Expanding-window OOS evaluation. Train ≤ Y−1, trade Y, concatenate.

    Without a gate every OOS year is kept — the stage still matters, because it
    fixes the year-by-year accounting the headline numbers come from. Years
    with fewer than ``min_train_years`` preceding years are traded ungated (a
    gate with no training history has no business selecting) and are flagged as
    ``ungated`` in the diagnostics so the report says so.

    Returns ``{selected, diagnostics, audit}`` where ``selected`` carries the
    kept rows at every alpha (selection is decided at mid and applied to the
    whole alpha grid — the contracts a structure selects must not depend on the
    fill assumption) and ``audit`` is the leak receipt the report checklist
    cites.
    """
    if trades.empty:
        return {"selected": trades, "diagnostics": [], "audit": {"years": [], "leak_free": True}}

    mid = trades[np.isclose(trades["fill_alpha"], alpha)].copy()
    mid["year"] = pd.to_datetime(mid["event_date"]).dt.year
    years = sorted(mid["year"].unique())

    kept_ids: list = []
    diagnostics: list[dict] = []
    fit_years_seen: list[int] = []
    score_rows: list[pd.DataFrame] = []

    for year in years:
        train = mid[mid["year"] < year]
        test = mid[mid["year"] == year]
        train_years = int(train["year"].nunique())
        row: dict[str, Any] = {"year": int(year), "n_train": int(len(train)),
                               "n_test": int(len(test)), "train_years": train_years}
        if gate is not None and gate.predict_proba is not None and len(test):
            # OOS probabilities for the calibration block — collected for EVERY
            # traded year (ungated ones included): calibration is measured on
            # the whole out-of-sample universe, never on the selected subset.
            score_rows.append(pd.DataFrame({
                "event_id": test["event_id"].to_numpy(),
                "proba": np.asarray(gate.predict_proba(test), dtype=float),
                "year": int(year),
            }))
        if gate is None or train_years < min_train_years or test.empty:
            row["n_selected"] = int(len(test))
            row["ungated"] = gate is not None and train_years < min_train_years
            kept_ids.extend(test["event_id"].tolist())
            diagnostics.append(row)
            continue

        # The leak guard: fit receives only prior years, and the harness
        # records what it handed over so the poison test can prove it. An
        # empty train frame is legitimate (min_train_years=0 with a gate
        # trained upstream) and carries no year to record.
        if len(train):
            assert int(train["year"].max()) < year, "walk-forward handed the test year to fit()"
            fit_years_seen.append(int(train["year"].max()))
            gate.seen.append(int(train["year"].max()))
        gate.fit(train)
        mask = gate.select(test)
        mask = pd.Series(np.asarray(mask, dtype=bool), index=test.index)
        row["n_selected"] = int(mask.sum())
        row["ungated"] = False
        kept_ids.extend(test.loc[mask, "event_id"].tolist())
        diagnostics.append(row)

    kept_set = set(kept_ids)
    selected = trades[trades["event_id"].isin(kept_set)].copy()
    # Leak discipline is enforced structurally — fit never receives the test
    # year (the assert above) — and the receipt records the max year every fit
    # saw so the poison test and the report auditor can verify it.
    gated_years = [d["year"] for d in diagnostics if not d.get("ungated") and d["n_train"]]
    leak_free = all(seen < year for seen, year in zip(fit_years_seen, gated_years))
    # The receipt is what the report's checklist renders: counts, the latest
    # information any fit saw, and the margin to the year it then traded — a
    # statement OUT of the audit rather than about it.
    receipt = None
    if fit_years_seen and gated_years:
        margin_years = min(year - seen for seen, year in zip(fit_years_seen, gated_years))
        receipt = {
            "n_rows_checked": int(len(mid)),
            "n_folds_checked": len(gated_years),
            "max_fit_year": int(max(fit_years_seen)),
            "min_margin_years": int(margin_years),
            "paths": ["evaluate.walk_forward"],
        }
    audit = {"years": years, "fit_years_seen": fit_years_seen,
             "leak_free": bool(leak_free), "receipt": receipt}
    if not leak_free:
        raise EvaluationError(
            f"walk-forward leak: a fit saw data from its own test year "
            f"(fits saw {fit_years_seen}, test years {gated_years})"
        )
    scores = pd.concat(score_rows, ignore_index=True) if score_rows else None
    return {"selected": selected, "diagnostics": diagnostics, "audit": audit,
            "scores": scores}


# --------------------------------------------------------------------------
# Monte Carlo
# --------------------------------------------------------------------------


def monte_carlo(
    rets: Sequence[float],
    *,
    fractions: Sequence[float] = SIZING_FRACTIONS,
    block: int = MC_BLOCK,
    paths: int = MC_PATHS,
    seed: int = 0,
    mode: str = "sequential",
    draw_order: str = "shared",
    path_bands_for: Sequence[float] = (),
) -> dict[str, Any]:
    """Block-bootstrap MC on a trade sequence → the sizing curve.

    Blocks of ``block`` consecutive trades are drawn with replacement and
    concatenated until the path is as long as the original sequence, which is
    what preserves earnings-week clustering — an iid resample would scatter one
    bad week across a year and understate ruin.

    ``draw_order="shared"`` (default for new experiments) draws one bootstrap
    sequence per path and prices every fraction on it, so the sizing curve
    compares fractions on identical scenarios. ``draw_order="per_fraction"``
    consumes fresh draws per fraction in the order the fractions are given —
    the EXP-050 reference behaviour, kept for the regression.

    ``path_bands_for`` lists the fractions for which the equity-path percentile
    bands (p05/p50/p95 over the trade index) are computed — the data for the
    report's MC fan chart, as distinct from the sizing curve.
    """
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out: dict[str, Any] = {"block": block, "paths": paths, "seed": seed, "n_trades": n,
                           "mode": mode, "by_fraction": {}, "path_bands": {}}
    if n == 0:
        return out
    if n <= block:
        # A bootstrap needs at least two distinct block starts; below that the
        # MC is degenerate and the honest output is the deterministic curve.
        for f in fractions:
            eq = _compound(r, f, mode)
            out["by_fraction"][f"{f:.2f}"] = {
                "p_loss": float(eq[-1] < 1.0), "terminal_p05": float(eq[-1]),
                "terminal_p50": float(eq[-1]), "terminal_p95": float(eq[-1]),
                "dd_p50": _max_drawdown(pd.Series(eq)), "dd_p95": _max_drawdown(pd.Series(eq)),
                "degenerate": True,
            }
        return out

    rng = np.random.default_rng(seed)

    def draw_index() -> np.ndarray:
        idx: list[int] = []
        while len(idx) < n:
            # high is exclusive: +1 so the final block (the most recent trades,
            # the ones most representative of the current regime) is reachable.
            s = int(rng.integers(0, n - block + 1))
            idx.extend(range(s, s + block))
        return np.array(idx[:n])

    if draw_order not in ("shared", "per_fraction"):
        raise EvaluationError(f"unknown draw_order {draw_order!r}")
    shared_indices = [draw_index() for _ in range(paths)] if draw_order == "shared" else None

    bands_wanted = {float(f) for f in path_bands_for}
    for f in fractions:
        finals = np.empty(paths)
        dds = np.empty(paths)
        collect = float(f) in bands_wanted
        eq_paths = np.empty((paths, n + 1)) if collect else None
        for p in range(paths):
            idx = shared_indices[p] if shared_indices is not None else draw_index()
            eq = _compound(r[idx], f, mode)
            finals[p] = eq[-1]
            dds[p] = _max_drawdown(pd.Series(eq))
            if collect:
                eq_paths[p] = eq
        out["by_fraction"][f"{f:.2f}"] = {
            "p_loss": float((finals < 1.0).mean()),
            "terminal_p05": float(np.percentile(finals, 5)),
            "terminal_p50": float(np.percentile(finals, 50)),
            "terminal_p95": float(np.percentile(finals, 95)),
            "dd_p50": float(np.percentile(dds, 50)),
            "dd_p95": float(np.percentile(dds, 95)),
        }
        if collect:
            out["path_bands"][f"{f:.2f}"] = {
                "p05": [float(v) for v in np.percentile(eq_paths, 5, axis=0)],
                "p50": [float(v) for v in np.percentile(eq_paths, 50, axis=0)],
                "p95": [float(v) for v in np.percentile(eq_paths, 95, axis=0)],
            }
    return out


def _compound(rets: np.ndarray, fraction: float, mode: str) -> np.ndarray:
    """Compounded equity path. ``cashflow`` on a bare return sequence reduces
    to sequential compounding — overlap information lives in the trade frame,
    not in the returns, so MC paths use the sequential arithmetic in both
    modes and the cashflow/sequential distinction is reported from the
    deterministic curve."""
    return np.concatenate([[1.0], np.cumprod(1.0 + fraction * rets)])


# --------------------------------------------------------------------------
# stress battery
# --------------------------------------------------------------------------


def stress_regimes(trades: pd.DataFrame, alpha: float = 0.5,
                   spy_daily: pd.DataFrame | None = None) -> dict[str, dict]:
    """Per-regime P&L: the fixed crisis windows plus the worst earnings weeks.

    ``worst_earnings_weeks`` are the 10 ISO weeks with the worst SPY weekly
    return inside the trade sample's date span — a market-wide stress definition
    that needs no lookahead (the weeks are selected from the realized series and
    the trades inside them are then measured, which is the replay question:
    "what would this have done in those weeks", not "can we find the weeks").
    """
    rows = trades[np.isclose(trades["fill_alpha"], alpha)] if "fill_alpha" in trades.columns else trades
    out: dict[str, dict] = {}
    event_dates = pd.to_datetime(rows["event_date"])
    for name, (start, end) in REGIME_WINDOWS.items():
        hit = rows[(event_dates >= start) & (event_dates <= end)]
        out[name] = _regime_stats(hit)

    weeks_hit = _worst_week_trades(rows, spy_daily, n_weeks=10)
    out["worst_earnings_weeks"] = weeks_hit
    return out


def _regime_stats(rows: pd.DataFrame) -> dict[str, float]:
    r = rows["ret"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    return {
        "n": int(r.size),
        "mean": float(r.mean()) if r.size else float(np.nan),
        "win_rate": float((r > 0).mean()) if r.size else float(np.nan),
    }


def _worst_week_trades(rows: pd.DataFrame, spy_daily: pd.DataFrame | None,
                       n_weeks: int = 10) -> dict[str, Any]:
    if rows.empty or spy_daily is None or spy_daily.empty:
        return {"n": 0, "mean": float(np.nan), "win_rate": float(np.nan),
                "note": "no SPY series supplied; worst-week replay skipped"}
    spy = spy_daily.sort_values("date").copy()
    spy["date"] = pd.to_datetime(spy["date"])
    span_lo = pd.to_datetime(rows["event_date"]).min()
    span_hi = pd.to_datetime(rows["event_date"]).max()
    spy = spy[(spy["date"] >= span_lo - pd.Timedelta(days=14)) & (spy["date"] <= span_hi)]
    if len(spy) < 10:
        return {"n": 0, "mean": float(np.nan), "win_rate": float(np.nan),
                "note": "SPY series does not cover the trade span"}
    spy["week"] = spy["date"].dt.to_period("W")
    weekly = spy.groupby("week")["close"].agg(["first", "last"])
    weekly["ret"] = weekly["last"] / weekly["first"] - 1.0
    worst = weekly.nsmallest(n_weeks, "ret").index
    event_dates = pd.to_datetime(rows["event_date"])
    hit = rows[event_dates.dt.to_period("W").isin(worst)]
    stats = _regime_stats(hit)
    stats["weeks"] = [str(w) for w in sorted(worst)]
    return stats


def stress_iv_regime(trades: pd.DataFrame, alpha: float = 0.5,
                     spy_daily: pd.DataFrame | None = None) -> dict[str, Any]:
    """High-vol vs low-vol split, so the report quantifies how much of the edge
    lives in 2022/2024-style regimes.

    Uses the trades' own ``spy_vol20`` column when present (the panel carries
    it); otherwise computes a per-year median 20-day SPY vol from the cached
    index series and splits years at the median of those medians.
    """
    rows = trades[np.isclose(trades["fill_alpha"], alpha)] if "fill_alpha" in trades.columns else trades
    if rows.empty:
        return {"split_by": None, "high": _regime_stats(rows), "low": _regime_stats(rows)}

    if "spy_vol20" in rows.columns and rows["spy_vol20"].notna().any():
        med = rows["spy_vol20"].median()
        high = rows[rows["spy_vol20"] >= med]
        low = rows[rows["spy_vol20"] < med]
        return {"split_by": "spy_vol20 (per trade)", "threshold": float(med),
                "high": _regime_stats(high), "low": _regime_stats(low)}

    if spy_daily is None or spy_daily.empty:
        return {"split_by": None, "high": _regime_stats(rows), "low": _regime_stats(rows),
                "note": "no spy_vol20 column and no SPY series; IV-regime split skipped"}

    spy = spy_daily.sort_values("date").copy()
    spy["date"] = pd.to_datetime(spy["date"])
    spy["vol20"] = spy["close"].pct_change().rolling(20).std(ddof=1) * np.sqrt(252)
    spy["year"] = spy["date"].dt.year
    yearly = spy.groupby("year")["vol20"].median().dropna()
    if yearly.empty:
        return {"split_by": None, "high": _regime_stats(rows), "low": _regime_stats(rows)}
    med = yearly.median()
    high_years = set(yearly[yearly >= med].index)
    years = pd.to_datetime(rows["event_date"]).dt.year
    return {"split_by": "yearly median SPY vol20", "threshold": float(med),
            "high_years": sorted(high_years),
            "high": _regime_stats(rows[years.isin(high_years)]),
            "low": _regime_stats(rows[~years.isin(high_years)])}


def stress_tail_injection(
    trades: pd.DataFrame,
    shock: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    alpha: float = 0.5,
    fractions: Sequence[float] = SIZING_FRACTIONS,
    block: int = MC_BLOCK,
    paths: int = MC_PATHS,
    seed: int = 0,
) -> dict[str, Any]:
    """Double the worst 1% of realized |moves| and re-price.

    The harness does not know each structure's payoff, so the caller supplies
    ``shock(trades) -> trades``: a function that applies the doubled-move
    re-pricing and returns the frame with the shocked rows' ``ret`` updated.
    This stage is MANDATORY for any structure with a short leg — the report
    checklist fails a short-leg spec that arrives without it, because a
    defined-risk claim that has never met a doubled tail is an assumption.
    """
    rows = trades[np.isclose(trades["fill_alpha"], alpha)] if "fill_alpha" in trades.columns else trades
    shocked = shock(rows.copy())
    base_rets = rows["ret"].to_numpy(dtype=float)
    new_rets = shocked["ret"].to_numpy(dtype=float)
    out = {
        "base_worst_trade": float(np.nanmin(base_rets)) if base_rets.size else float(np.nan),
        "shocked_worst_trade": float(np.nanmin(new_rets)) if new_rets.size else float(np.nan),
        "n_shocked": int((np.abs(new_rets - base_rets) > 1e-12).sum()) if new_rets.size == base_rets.size else -1,
        "mc": monte_carlo(new_rets, fractions=fractions, block=block, paths=paths, seed=seed)["by_fraction"],
    }
    return out


def stress_slippage(
    trades: pd.DataFrame,
    repricer: Callable[[pd.DataFrame, int], pd.DataFrame] | None,
    *,
    alpha: float = 0.5,
    days: Sequence[int] = (-1, 1),
) -> dict[str, Any]:
    """Shift entry/exit by ±1 trading day where the adjacent chain is cached.

    ``repricer(trades, shift_days)`` returns the re-priced frame for the subset
    of trades whose shifted chains exist — never fabricating a missing chain —
    plus a ``coverage`` attr: the fraction of trades it could re-price. Without
    a repricer (a pre-priced dataset with no chain access) the stage reports
    N/A rather than inventing numbers.
    """
    if repricer is None:
        return {"available": False,
                "note": "no repricer supplied; slippage stress needs chain access"}
    rows = trades[np.isclose(trades["fill_alpha"], alpha)] if "fill_alpha" in trades.columns else trades
    base_mean = float(rows["ret"].mean()) if len(rows) else float(np.nan)
    out: dict[str, Any] = {"available": True, "base_mean": base_mean, "shifts": {}}
    for shift in days:
        repriced = repricer(rows.copy(), int(shift))
        coverage = float(getattr(repriced, "attrs", {}).get("coverage", np.nan))
        r = repriced["ret"].to_numpy(dtype=float)
        out["shifts"][f"{int(shift):+d}d"] = {
            "coverage": coverage,
            "n": int(r.size),
            "mean": float(np.nanmean(r)) if r.size else float(np.nan),
            "delta_mean": float(np.nanmean(r) - base_mean) if r.size else float(np.nan),
        }
    return out


def stress_stale_dates(
    trades: pd.DataFrame,
    repricer: Callable[[pd.DataFrame, int], pd.DataFrame] | None,
    *,
    alpha: float = 0.5,
    fraction: float = 0.01,
    seed: int = 0,
) -> dict[str, Any]:
    """Mis-date 1% of events by one day — the stale-calendar failure mode.

    Earnings dates move; a live system that trades yesterday's date buys the
    wrong print. This mis-dates a random 1% of events (seeded) and re-prices
    them through the same repricer the slippage stage uses, leaving the rest
    untouched, then reports the P&L impact on the affected subset.
    """
    if repricer is None:
        return {"available": False,
                "note": "no repricer supplied; stale-date stress needs chain access"}
    rows = trades[np.isclose(trades["fill_alpha"], alpha)] if "fill_alpha" in trades.columns else trades
    if rows.empty:
        return {"available": True, "n_misdated": 0}
    rng = np.random.default_rng(seed)
    k = max(int(round(len(rows) * fraction)), 1)
    idx = rng.choice(rows.index.to_numpy(), size=min(k, len(rows)), replace=False)
    subset = rows.loc[idx]
    repriced = repricer(subset.copy(), 1)
    base = subset["ret"].to_numpy(dtype=float)
    new = repriced["ret"].to_numpy(dtype=float)
    return {
        "available": True,
        "n_misdated": int(len(subset)),
        "coverage": float(getattr(repriced, "attrs", {}).get("coverage", np.nan)),
        "base_mean": float(np.nanmean(base)) if base.size else float(np.nan),
        "misdated_mean": float(np.nanmean(new)) if new.size else float(np.nan),
        "delta_mean": float(np.nanmean(new) - np.nanmean(base)) if new.size else float(np.nan),
    }


# --------------------------------------------------------------------------
# pre-registration
# --------------------------------------------------------------------------


def _run_log_path(run_dir: Path) -> Path:
    #: The guide places the log at ``results/run_log.jsonl`` under the
    #: experiment folder, next to the metrics artifacts it annotates.
    return Path(run_dir) / "results" / "run_log.jsonl"


def check_preregistration(spec: Mapping[str, Any], run_dir: Path | None,
                          now_utc: pd.Timestamp | None = None) -> dict[str, Any]:
    """Validate the spec's pre-registration against the run log and the ledger.

    Returns a receipt. Raises :class:`PreregistrationError` when the OOS stage
    must refuse:

    - no ``preregistered_at`` at all;
    - a stamp later than the first recorded run (a spec stamped after results
      were seen);
    - a PRIMARY spec whose hash no longer matches the PLANNED ledger row —
      i.e. the spec was edited after registration. Run the result-disliked
      loop (run, dislike, edit, re-run under the old stamp) and the receipt
      refuses. Grid cells legitimately differ from the primary spec and are
      exempted BY LABEL (``grid_cell: true``), not by omission.
    """
    stamp = spec.get("preregistered_at")
    if run_dir is None:
        return {"enforced": False, "valid": stamp is not None,
                "reason": "no experiment dir attached" if stamp is None else "stamp present, unverified"}
    if stamp is None:
        raise PreregistrationError(
            "spec carries no preregistered_at — the OOS stage refuses to run. "
            "Scaffold the experiment (experiments/new_experiment.py) to stamp it."
        )
    prereg = pd.Timestamp(stamp)
    if prereg.tzinfo is None:
        prereg = prereg.tz_localize("UTC")
    log = _run_log_path(run_dir)
    first_run_ts = None
    if log.exists():
        for line in log.read_text().splitlines():
            if not line.strip():
                continue
            first_run_ts = pd.Timestamp(json.loads(line)["ts"])
            break
    receipt = {"enforced": True, "valid": True, "preregistered_at": str(prereg),
               "first_run_ts": str(first_run_ts) if first_run_ts else None}
    if first_run_ts is not None and prereg > first_run_ts:
        raise PreregistrationError(
            f"preregistered_at {prereg} is after the first recorded run "
            f"{first_run_ts} — this spec was edited after results existed. "
            "Register a NEW spec for the changed hypothesis."
        )

    # Spec-hash continuity: the registered hypothesis is the one that runs.
    if not spec.get("grid_cell"):
        ledger = paths.ROOT / "experiments" / "LEDGER.csv"
        exp_id = spec.get("id")
        if ledger.exists() and exp_id:
            import csv

            with open(ledger, newline="") as fh:
                planned_hashes = [
                    row["spec_hash"] for row in csv.DictReader(fh)
                    if row.get("id") == exp_id and row.get("stage") == "planned"
                ]
            if planned_hashes and spec_hash(spec) not in planned_hashes:
                raise PreregistrationError(
                    f"{exp_id}: spec_hash {spec_hash(spec)[:12]}… does not match the "
                    f"PLANNED ledger row(s) ({planned_hashes[-1][:12]}…). The spec was "
                    "edited after registration — scaffold a NEW experiment for the "
                    "changed hypothesis (grid cells are exempt via `grid_cell: true`)."
                )
            receipt["spec_hash_checked"] = bool(planned_hashes)
    else:
        receipt["spec_hash_checked"] = False
        receipt["grid_cell"] = True
    return receipt


def append_run_log(run_dir: Path, entry: Mapping[str, Any]) -> None:
    """Append one invocation record to the experiment's ``results/run_log.jsonl``."""
    results_dir = Path(run_dir) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(_run_log_path(run_dir), "a") as fh:
        fh.write(json.dumps(dict(entry), default=str) + "\n")


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Everything one evaluation produced, for the report and the ledger."""

    spec: dict[str, Any]
    results: dict[str, Any]
    run_dir: Path | None = None
    report_path: Path | None = None

    @property
    def metrics(self) -> dict[str, Any]:
        return self.results["headline"]

    def to_json(self) -> str:
        return json.dumps(self.results, indent=1, default=str)


def evaluate(
    spec: Mapping[str, Any],
    trades: pd.DataFrame,
    *,
    gate: Gate | None = None,
    run_dir: Path | None = None,
    repricer: Callable[[pd.DataFrame, int], pd.DataFrame] | None = None,
    tail_shock: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    spy_daily: pd.DataFrame | None = None,
    alphas: Sequence[float] = ALPHA_GRID,
    fractions: Sequence[float] = SIZING_FRACTIONS,
    mc_paths: int = MC_PATHS,
    mc_block: int = MC_BLOCK,
    seed: int = 0,
    stress: bool = True,
    write_report: bool = True,
    input_files: Sequence[Path | str] = (),
    extra_sections: Sequence[Mapping[str, Any]] | Callable[["EvalResult"], Sequence[Mapping[str, Any]]] = (),
    now_utc: pd.Timestamp | None = None,
) -> EvalResult:
    """Run one candidate through the full evaluation suite.

    ``trades`` is the priced frame (engine.replay shape: one row per event ×
    fill alpha). ``gate`` is the walk-forward selector, if the candidate has
    one. ``run_dir`` attaches the run to an experiment folder — required for
    promotion-eligible results, which is where pre-registration is enforced and
    the artifacts land. ``repricer`` / ``tail_shock`` unlock the slippage,
    stale-date, and tail-injection stresses; without them those stages report
    N/A (tail injection then FAILs the checklist for short-leg specs, which is
    the point).
    """
    started = time.time()
    spec = dict(spec)
    missing = [c for c in ("id", "primary_spec") if c not in spec]
    _require(not missing, f"spec missing keys: {missing}")
    if trades is not None and len(trades):
        missing_cols = [c for c in REQUIRED_TRADE_COLUMNS if c not in trades.columns]
        _require(not missing_cols, f"trades frame missing columns: {missing_cols}")
        trades = trades.copy()
        for col in ("event_date", "entry_date", "exit_date"):
            trades[col] = pd.to_datetime(trades[col])

    sha = spec_hash(spec)
    prereg = check_preregistration(spec, run_dir, now_utc=now_utc)

    wf_cfg = spec.get("walk_forward", {}) or {}
    min_train_years = int(wf_cfg.get("min_train_years", 2))
    equity_mode = str(spec.get("equity_mode", "cashflow"))
    _require(equity_mode in ("cashflow", "sequential"), f"unknown equity_mode {equity_mode!r}")

    results: dict[str, Any] = {
        "spec_id": spec.get("id"),
        "spec_hash": sha,
        "preregistration": prereg,
        "equity_mode": equity_mode,
        "elapsed_s": 0.0,
    }

    # -- stage 1: backtest (unselected, all alphas) -------------------------
    sweep = alpha_sweep(trades, alphas)
    results["backtest"] = {
        "alpha_sweep": sweep,
        "breakeven_alpha": breakeven_alpha_from_sweep(sweep),
        "by_year": by_year_table(
            trades[np.isclose(trades["fill_alpha"], 0.5)] if "fill_alpha" in trades.columns else trades
        ),
        "n_events": int(trades["event_id"].nunique()) if len(trades) else 0,
    }

    # -- stage 2: walk-forward ----------------------------------------------
    wf = walk_forward(trades, gate, min_train_years=min_train_years)
    selected = wf["selected"]
    # The OOS sequence is chronological; stable sort keeps the input's tie
    # order, which the block bootstrap is sensitive to.
    selected = selected.sort_values(["entry_date", "exit_date"], kind="stable")
    mid_sel = selected[np.isclose(selected["fill_alpha"], 0.5)] if len(selected) and "fill_alpha" in selected.columns else selected
    base_mid = trades[np.isclose(trades["fill_alpha"], 0.5)] if len(trades) and "fill_alpha" in trades.columns else trades

    eq5 = build_equity(mid_sel, 0.05, mode=equity_mode,
                       max_deployed=spec.get("max_deployed_fraction"), record=True)
    headline = trade_stats(mid_sel["ret"].to_numpy(), mid_sel["event_date"]) if len(mid_sel) else trade_stats([])
    headline["by_year"] = by_year_table(mid_sel)
    headline["breakeven_alpha"] = breakeven_alpha_from_sweep(
        alpha_sweep(selected, alphas)) if len(selected) else None
    headline["sharpe_equity"] = sharpe_equity(eq5["equity"])
    headline["max_dd"] = eq5["max_dd"]
    headline["max_concurrency"] = eq5["max_concurrency"]
    # The leverage the 5%-sized run actually used — visible whether or not a
    # cap was set. Per-trade sizing with concurrent positions silently deploys
    # fraction x concurrency of equity; this is where that becomes a number a
    # sizing decision has to look at.
    headline["deployment"] = {
        "peak": eq5["peak_deployment"],
        "worst_cash": eq5["worst_cash"],
        "cap": spec.get("max_deployed_fraction"),
        "constrained_entries": eq5["constrained_entries"],
    }
    headline["alpha_sweep"] = alpha_sweep(selected, alphas)
    headline["capacity"] = capacity_note(selected)
    headline["dollar_weighted"] = dollar_weighted_return(mid_sel)
    # How much of the headline is not a gate result at all: in ungated years
    # every row is kept, so a headline that blends them is reporting the base
    # exposure and the gate under one number.
    ungated_rows = sum(int(d.get("n_selected", 0)) for d in wf["diagnostics"]
                       if d.get("ungated"))
    headline["ungated_share"] = (float(ungated_rows / len(mid_sel))
                                 if len(mid_sel) else float(np.nan))
    # Anti-selection guard (the S4 lesson): the unselected universe always
    # appears next to the selected one.
    headline["base_unselected"] = {
        "n": int(len(base_mid)),
        "mean": float(base_mid["ret"].mean()) if len(base_mid) else float(np.nan),
        "win_rate": float((base_mid["ret"] > 0).mean()) if len(base_mid) else float(np.nan),
    }
    headline["mc"] = {}
    results["walk_forward"] = {
        "diagnostics": wf["diagnostics"],
        "audit": wf["audit"],
        "headline_stage": "wf_oos",
    }

    # Calibration of the gate's OOS probabilities. Computed on the WHOLE
    # out-of-sample universe, not the selected subset — measuring calibration
    # on what the gate kept would condition on the very thing being measured.
    scores = wf.get("scores")
    if scores is not None and len(scores):
        merged = scores.merge(
            base_mid[["event_id", "ret"]], on="event_id", how="inner")
        results["calibration"] = calibration_block(
            merged["proba"].to_numpy(),
            (merged["ret"] > 0).to_numpy(dtype=float),
        )
    else:
        results["calibration"] = {
            "available": False,
            "reason": "gate provides no predict_proba; calibration not measured",
        }

    # -- stage 3: Monte Carlo on the WF OOS sequence ------------------------
    rets = mid_sel["ret"].to_numpy(dtype=float) if len(mid_sel) else np.array([])
    results["mc"] = monte_carlo(
        rets, fractions=fractions, block=mc_block, paths=mc_paths, seed=seed,
        mode=equity_mode, draw_order=str(spec.get("mc_draw_order", "shared")),
        path_bands_for=(0.05,) if 0.05 in fractions else (),
    )
    headline["mc"] = results["mc"]["by_fraction"]

    # -- stage 4: stress battery ---------------------------------------------
    results["stress"] = {}
    if stress:
        results["stress"]["regimes"] = stress_regimes(selected, spy_daily=spy_daily)
        results["stress"]["iv_regime"] = stress_iv_regime(selected, spy_daily=spy_daily)
        has_short_leg = bool(spec.get("has_short_leg", False))
        if tail_shock is not None and len(selected):
            results["stress"]["tail_injection"] = stress_tail_injection(
                selected, tail_shock, fractions=fractions, block=mc_block,
                paths=mc_paths, seed=seed)
        else:
            results["stress"]["tail_injection"] = {
                "available": False,
                "required": has_short_leg,
                "note": ("short-leg spec without a tail shock — checklist FAIL"
                         if has_short_leg else "the structure carries no short leg"),
            }
        results["stress"]["slippage"] = stress_slippage(selected, repricer)
        results["stress"]["stale_dates"] = stress_stale_dates(
            selected, repricer, seed=seed)

    # -- headline + deterministic equity curves at all sizings ---------------
    results["equity_curves"] = {
        f"{f:.2f}": {
            "final": build_equity(mid_sel, f, mode=equity_mode,
                                  max_deployed=spec.get("max_deployed_fraction"))["final"],
        }
        for f in fractions
    }
    # The 5%-sized curve the report plots (JSON-safe: dates as strings).
    results["equity_curve_series"] = {
        "date": [str(ts.date()) for ts in eq5["equity"].index],
        "equity": [float(v) for v in eq5["equity"].values],
    }
    # The transaction log behind the plotted curve. Written next to the report
    # so a chart can be audited row by row instead of taken on trust.
    if len(mid_sel):
        log = transaction_log(mid_sel, eq5, scores=wf.get("scores"))
        results["transaction_log"] = reconcile_transaction_log(log, eq5)
    else:
        log = None
        results["transaction_log"] = {"rows": 0, "reconciles": True,
                                      "note": "no selected trades to log"}

    results["headline"] = headline
    results["headline_stage"] = "wf_oos"

    # The accuracy checklist, computed here (not taken from the report) so
    # promote.py can refuse a challenger whose evidence has a FAIL item.
    from engine.report import accuracy_checklist

    checklist = accuracy_checklist(
        results, spec, ledger_path=paths.ROOT / "experiments" / "LEDGER.csv")
    results["checklist"] = [
        {"name": item.name, "status": item.status, "evidence": item.evidence}
        for item in checklist
    ]
    results["checklist_fails"] = sum(1 for item in checklist if item.status == "FAIL")

    # The canonical contract is asserted, not merely documented: a headline
    # missing a key would otherwise pass silently and break the leaderboard.
    missing_keys = [k for k in METRIC_KEYS if k not in headline]
    _require(not missing_keys, f"headline metrics missing canonical keys: {missing_keys}")

    results["elapsed_s"] = round(time.time() - started, 2)

    # -- artifacts -------------------------------------------------------------
    if run_dir is not None:
        run_dir = Path(run_dir)
        results_dir = run_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        if log is not None:
            log_path = results_dir / f"transactions_{sha[:12]}.csv"
            log.to_csv(log_path, index=False)
            results["transaction_log"]["path"] = str(
                log_path.relative_to(paths.ROOT) if log_path.is_relative_to(paths.ROOT)
                else log_path)
            results["transaction_log"]["sha256"] = _file_sha256(log_path)
        (results_dir / f"metrics_{sha[:12]}.json").write_text(
            json.dumps(results, indent=1, default=str))
        append_run_log(run_dir, {
            "ts": (now_utc or pd.Timestamp.now(tz="UTC")).isoformat(),
            "spec_id": spec.get("id"),
            "spec_hash": sha,
            "n_events": results["backtest"]["n_events"],
            "headline_mean_mid": headline.get("mean"),
            "sharpe_trade": headline.get("sharpe_trade"),
            "stage": "ran",
        })

    result = EvalResult(spec=spec, results=results, run_dir=run_dir)

    if write_report:
        from engine.report import Report

        # Spec-specific analyses are rendered BY the generator, not appended to
        # its output: an experiment's best content (EXP-102's defined-risk
        # falsification) used to land outside the report's formatting,
        # ordering, checklist and provenance because run.py opened REPORT.md in
        # append mode. A callable is accepted because those analyses usually
        # need the finished result to compute.
        sections = extra_sections(result) if callable(extra_sections) else extra_sections
        report = Report.from_eval(result, input_files=input_files,
                                  extra_sections=sections or ())
        out_dir = (Path(run_dir) if run_dir is not None else paths.REPORTS / str(spec.get("id")))
        result.report_path = report.write(out_dir)

    return result
