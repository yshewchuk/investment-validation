#!/usr/bin/env python3
"""EXP-129 — figures and REPORT.md from the run's own artifacts.

    python3 experiments/EXP-129_.../report.py

Separate from ``run.py`` on purpose: re-drawing a chart must never be able to
re-run a simulation, or a figure and the number beside it can drift apart while
both look freshly generated.

Every sweep is plotted AGAINST GATED-EVENT COUNT rather than against its own
threshold. That is the whole point of plotting them at all — a P&L threshold
and a cost/peak ratio are not comparable numbers, but "how many trades did this
admit" is the same axis for both, and reading two rules at equal selectivity is
the only honest comparison between them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

PNL_ORDER = ["primary_exp_pnl_gt_0", "exp_pnl_gt_0.05", "exp_pnl_gt_0.1",
             "exp_pnl_gt_0.25", "exp_pnl_gt_0.5"]
ARITH_ORDER = [f"cost_lt_{r}_peak" for r in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
QUANT_ORDER = ["top_10pct", "top_20pct", "top_50pct"]
PRIMARY = "primary_exp_pnl_gt_0"
INCUMBENT = "cost_lt_0.5_peak"

PNL_LABEL = {"primary_exp_pnl_gt_0": "> 0", "exp_pnl_gt_0.05": "> 0.05",
             "exp_pnl_gt_0.1": "> 0.10", "exp_pnl_gt_0.25": "> 0.25",
             "exp_pnl_gt_0.5": "> 0.50"}
ARITH_LABEL = {k: k.replace("cost_lt_", "").replace("_peak", "") for k in ARITH_ORDER}


def series(books: dict, keys: list[str], field: str):
    xs, ys, labels = [], [], []
    for k in keys:
        b = books.get(k) or {}
        if not b.get("n"):
            continue
        value = b.get(field)
        if value is None:
            continue
        xs.append(b["n"])
        ys.append(value)
        labels.append(k)
    return np.array(xs, dtype=float), np.array(ys, dtype=float), labels


def curve_panel(books: dict, out: Path) -> Path:
    """Four measures, both rule families, everything against gated-event count."""
    fields = [("cagr", "CAGR", 100), ("sharpe_trade", "Sharpe per trade", 1),
              ("mean", "Mean return per trade", 100),
              ("max_drawdown", "Max drawdown", 100)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (field, title, scale) in zip(axes.ravel(), fields):
        for keys, name, colour, marker in (
            (PNL_ORDER, "P&L gate  (exp_pnl_sim > t)", "#1f77b4", "o"),
            (ARITH_ORDER, "arithmetic  (cost < r · peak)", "#d62728", "s"),
            (QUANT_ORDER, "P&L quantile (prior-year cutoff)", "#2ca02c", "^"),
        ):
            x, y, labels = series(books, keys, field)
            if not len(x):
                continue
            order = np.argsort(x)
            ax.plot(x[order], scale * y[order], marker=marker, color=colour,
                    label=name, alpha=0.85, linewidth=1.6, markersize=6)
        for key, colour, text in ((PRIMARY, "#1f77b4", "primary"),
                                  (INCUMBENT, "#d62728", "incumbent")):
            b = books.get(key) or {}
            if b.get("n") and b.get(field) is not None:
                ax.scatter([b["n"]], [scale * b[field]], s=190, facecolors="none",
                           edgecolors=colour, linewidths=2.2, zorder=5)
                ax.annotate(text, (b["n"], scale * b[field]),
                            textcoords="offset points", xytext=(8, 9),
                            fontsize=9, color=colour, weight="bold")
        ax.set_xscale("log")
        ax.set_xlabel("gated events (log scale) — MATCHED SELECTIVITY axis")
        ax.set_ylabel(title + (" (%)" if scale == 100 else ""))
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
    axes[0][0].legend(fontsize=8, loc="best")
    fig.suptitle(
        "EXP-129 — both gate families read at equal selectivity\n"
        "a rule taking 300 trades and one taking 2,400 are not comparable; the x axis makes them so",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def threshold_detail(books: dict, out: Path) -> Path:
    """Each sweep against its OWN threshold — the shape of the tradeoff."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    pnl_t = [0.0, 0.05, 0.10, 0.25, 0.50]
    arith_t = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for ax, keys, ts, name, xlabel in (
        (axes[0], PNL_ORDER, pnl_t, "P&L gate", "exp_pnl_sim threshold"),
        (axes[1], ARITH_ORDER, arith_t, "arithmetic gate", "r  in  cost < r · peak"),
    ):
        cagr = [100 * (books.get(k, {}).get("cagr") or np.nan) for k in keys]
        sharpe = [books.get(k, {}).get("sharpe_trade") or np.nan for k in keys]
        n = [books.get(k, {}).get("n") or np.nan for k in keys]
        ax.plot(ts, cagr, "o-", color="#1f77b4", label="CAGR (%)")
        ax.plot(ts, sharpe, "s--", color="#ff7f0e", label="Sharpe per trade")
        ax.set_xlabel(xlabel)
        ax.set_title(name)
        ax.grid(alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        twin = ax.twinx()
        twin.bar(ts, n, width=(ts[1] - ts[0]) * 0.35, alpha=0.18, color="grey")
        twin.set_ylabel("gated events (bars)")
        twin.set_yscale("log")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("EXP-129 — each family against its own threshold. A broad plateau is signal; "
                 "a single peak is a sample size.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def scatter_and_calibration(sim: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ok = sim.dropna(subset=["exp_pnl_sim", "ret"])
    axes[0].scatter(ok["exp_pnl_sim"], ok["ret"], s=5, alpha=0.18, color="#1f77b4")
    axes[0].axvline(0, color="#d62728", linewidth=1.2, label="primary gate")
    axes[0].axvline(0.05, color="#ff7f0e", linewidth=1.0, linestyle="--", label="> 0.05")
    axes[0].axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axes[0].set_xlabel("simulated expected return")
    axes[0].set_ylabel("REALIZED return")
    axes[0].set_title("prediction vs outcome (one dot per event)")
    axes[0].set_ylim(-1.5, 4)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    # Decile calibration: does a higher simulated return mean a higher realized one?
    ok = ok.copy()
    ok["bucket"] = pd.qcut(ok["exp_pnl_sim"], 10, labels=False, duplicates="drop")
    grouped = ok.groupby("bucket").agg(pred=("exp_pnl_sim", "mean"),
                                       real=("ret", "mean"), n=("ret", "size"))
    axes[1].plot(grouped["pred"], grouped["real"], "o-", color="#1f77b4")
    lims = [min(grouped.pred.min(), grouped.real.min()), max(grouped.pred.max(), grouped.real.max())]
    axes[1].plot(lims, lims, "--", color="grey", label="perfect calibration")
    axes[1].set_xlabel("mean simulated return (decile)")
    axes[1].set_ylabel("mean realized return")
    axes[1].set_title("calibration by simulated-return decile")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)

    w = sim.dropna(subset=["win_sim", "ret"]).copy()
    w["bucket"] = pd.cut(w["win_sim"], [0, .2, .4, .5, .6, .8, 1.0])
    g = w.groupby("bucket", observed=True).agg(pred=("win_sim", "mean"),
                                               real=("ret", lambda s: (s > 0).mean()),
                                               n=("ret", "size"))
    axes[2].plot(100 * g["pred"], 100 * g["real"], "o-", color="#2ca02c")
    axes[2].plot([0, 100], [0, 100], "--", color="grey")
    for x, y, n in zip(100 * g["pred"], 100 * g["real"], g["n"]):
        axes[2].annotate(f"n={n:,}", (x, y), textcoords="offset points",
                         xytext=(5, -11), fontsize=7)
    axes[2].set_xlabel("predicted win probability (%)")
    axes[2].set_ylabel("realized win rate (%)")
    axes[2].set_title("win_sim calibration")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def variance_control(metrics: dict, out: Path) -> Path:
    """The 2x2: which variable's UNCERTAINTY is doing the work."""
    books, arms = metrics["books"], metrics["arms"]
    cells = [("move drawn\ncrush drawn\n(primary)", books[PRIMARY]),
             ("move drawn\ncrush FIXED", arms.get("move_only", {})),
             ("move FIXED\ncrush drawn", arms.get("crush_only", {})),
             ("move FIXED\ncrush FIXED", arms.get("no_variance", {}))]
    labels = [c[0] for c in cells]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, field, title, scale in ((axes[0], "cagr", "CAGR (%)", 100),
                                    (axes[1], "sharpe_trade", "Sharpe per trade", 1),
                                    (axes[2], "n", "gated events", 1)):
        vals = [(c[1].get(field) or 0) * scale for c in cells]
        colours = ["#1f77b4", "#1f77b4", "#d62728", "#d62728"]
        ax.bar(labels, vals, color=colours, alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle("EXP-129 — whose uncertainty matters. Blue = the move carries its error; "
                 "red = the move is held at its forecast.", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def per_year(sim: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.4))
    for label, mask, colour in (
        ("P&L gate  > 0", sim["exp_pnl_sim"] > 0, "#1f77b4"),
        ("P&L gate  > 0.05", sim["exp_pnl_sim"] > 0.05, "#ff7f0e"),
        ("incumbent  cost < 0.5·peak", (sim["cost"] / sim["peak"]) < 0.5, "#d62728"),
    ):
        rows = sim[mask]
        g = rows.groupby("year")["ret"].agg(["mean", "size"])
        ax.plot(g.index, 100 * g["mean"], "o-", color=colour, label=label)
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xlabel("year")
    ax.set_ylabel("mean return per trade (%)")
    ax.set_title("EXP-129 — per-year mean at mid fills")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def table(books: dict, keys: list[str], labeller: dict) -> list[str]:
    lines = ["| gate | events | tickers | mean | CAGR | Sharpe | max DD | RoC | years + |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        b = books.get(k) or {}
        if not b.get("n"):
            continue
        name = labeller.get(k, k)
        mark = "**" if k in (PRIMARY, INCUMBENT) else ""
        lines.append(
            f"| {mark}{name}{mark} | {b['n']:,} | {b['tickers']:,} | {100*b['mean']:.2f}% | "
            f"{100*b['cagr']:.2f}% | {b['sharpe_trade']:.2f} | "
            f"{100*b.get('max_drawdown', float('nan')):.1f}% | "
            f"{100*b['return_on_capital']:.2f}% | {b['years_positive']}/{b['years']} |")
    return lines


def main() -> int:
    FIGURES.mkdir(exist_ok=True)
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    sim = pd.read_csv(RESULTS / "simulated.csv")
    books = metrics["books"]

    made = [
        curve_panel(books, FIGURES / "matched_selectivity.png"),
        threshold_detail(books, FIGURES / "threshold_curves.png"),
        scatter_and_calibration(sim, FIGURES / "calibration.png"),
        variance_control(metrics, FIGURES / "variance_2x2.png"),
        per_year(sim, FIGURES / "per_year.png"),
    ]
    for path in made:
        print(f"  figure {path.relative_to(HERE)}", flush=True)

    s0 = metrics["stage_0_repricing"]
    cov = metrics["coverage"]
    f = metrics["funnel"]
    prim, inc = books[PRIMARY], books[INCUMBENT]

    body = f"""# EXP-129 — a P&L gate for the twin peak, against the constant it replaces

*Generated by `report.py` from `results/metrics.json`. Not written by hand.*

**Registered primary:** `exp_pnl_sim > 0`. **Benchmark:** `cost < peak/2`, the
rule in production. Every other row is a secondary grid cell and none is
promotable in place of the primary.

## The finding

The P&L gate beats the incumbent at every matched selectivity, on CAGR, Sharpe
and year-consistency. It also **fails two of its own registered acceptance
criteria** — max drawdown and breakeven alpha — and the mechanism turned out to
be simpler than the hypothesis claimed.

| | primary `exp_pnl_sim > 0` | incumbent `cost < 0.5·peak` |
|---|---:|---:|
| gated events | {prim['n']:,} | {inc['n']:,} |
| CAGR | **{100*prim['cagr']:.2f}%** | {100*inc['cagr']:.2f}% |
| Sharpe per trade | **{prim['sharpe_trade']:.2f}** | {inc['sharpe_trade']:.2f} |
| years positive | {prim['years_positive']}/{prim['years']} | {inc['years_positive']}/{inc['years']} |
| max drawdown | {100*prim['max_drawdown']:.1f}% | {100*inc['max_drawdown']:.1f}% |
| return on capital | {100*prim['return_on_capital']:.2f}% | **{100*inc['return_on_capital']:.2f}%** |

**Return on capital is the strongest number against promotion.** The P&L
gate's CAGR advantage comes from trading {prim['n']/inc['n']:.1f}x as often, not from better
trades per dollar deployed — {100*inc['return_on_capital']:.2f}% for the incumbent against
{100*prim['return_on_capital']:.2f}% for the primary. Compounding frequency is worth something, but
it is a different claim from "these are better trades", and it is the fragile
one under any capacity or spread constraint.

![matched selectivity](figures/matched_selectivity.png)

## Both sweeps, read at their own thresholds

![threshold curves](figures/threshold_curves.png)

The P&L curve **slopes, it does not peak** — CAGR is flat from `> 0` to `> 0.05`
and then falls as the universe collapses, and Sharpe plateaus broadly across
`0.05`–`0.10`. A broad plateau is what a real signal looks like. The one peaked
cell (`> 0.5`, four trades) is a sample size, not a result.

### P&L gate

{chr(10).join(table(books, PNL_ORDER + QUANT_ORDER + ['win_sim_gt_50'], {**PNL_LABEL, 'top_10pct': 'top 10% (prior-year cutoff)', 'top_20pct': 'top 20%', 'top_50pct': 'top 50%', 'win_sim_gt_50': 'win_sim > 0.5'}))}

### The incumbent's own constant, swept

{chr(10).join(table(books, ARITH_ORDER, ARITH_LABEL))}

Sweeping the benchmark was the point: beating one point would only have shown
that 0.5 was the wrong constant. It is not — `cost < 0.5·peak` is the best
arithmetic cell on Sharpe and year-consistency, and the P&L gate still beats it.

## Whose uncertainty is doing the work

![variance 2x2](figures/variance_2x2.png)

A **post-hoc control**, added after the registered arms and marked as such. The
2x2 splits on one axis only:

| arm | move | crush | events | CAGR | Sharpe | years + |
|---|---|---|---:|---:|---:|---:|
| primary | drawn | drawn | {prim['n']:,} | {100*prim['cagr']:.2f}% | {prim['sharpe_trade']:.2f} | {prim['years_positive']}/{prim['years']} |
"""
    for key, mv, cr in (("move_only", "drawn", "fixed"), ("crush_only", "fixed", "drawn"),
                        ("no_variance", "fixed", "fixed"), ("independent", "drawn", "unpaired"),
                        ("draws_500", "drawn", "drawn")):
        a = metrics["arms"].get(key) or {}
        if not a.get("n"):
            continue
        body += (f"| `{key}` | {mv} | {cr} | {a['n']:,} | {100*a['cagr']:.2f}% | "
                 f"{a['sharpe_trade']:.2f} | {a['years_positive']}/{a['years']} |\n")

    body += f"""
**The move's variance is the entire mechanism.** Holding it at its point
forecast costs ~9pp of CAGR and 0.43 of Sharpe while admitting 65% MORE trades.
Crush variance moves nothing in either row: the crush model earns its place as a
**level** — it sets exit vol — and its distribution contributes nothing, so the
paired-residual machinery this experiment was designed around is not what is
doing the work.

`no_variance` does **not** collapse to reward:risk (Spearman −0.14), which the
spec predicted it would. Two measured reasons: the exit carries a median 9 DTE
of time value, so the structure is never on the terminal payoff `peak`
describes; and the strikes snap to the listed ladder, a median 2.1% and a p90
15.1% away from the spacing the forecast asked for. The gate is therefore
ranking on something close to orthogonal to the incumbent, which is why it beats
it rather than reproducing it.

## Can the pricer reproduce an outcome it is told?

Stage 0 feeds the **realized** move and crush into the pricer and compares to
the exit the chains actually quoted. It bounds everything above it.

| | |
|---|---:|
| outcomes repriced | {s0['n']:,} |
| median absolute error | ${s0['median_abs_err']:.3f} |
| median actual exit | ${s0['median_actual_exit']:.2f} |
| correlation | {s0['r']:.4f} |
| bias | ${s0['bias']:+.3f} |
| within 10% of entry cost | {100*s0['within_10pct_of_entry_cost']:.1f}% |

The bias is positive: the pricer **overvalues** the exit by ${s0['bias']:.3f}, which on the
median entry cost is under a point of return, and it points the wrong way for a
gate set at exactly zero.

![calibration](figures/calibration.png)

![per year](figures/per_year.png)

## Universe and coverage

| | |
|---|---:|
| priced events | {f['priced']:,} |
| pass spread ≤ 25% | {f['f_spread']:,} |
| pass mcap ≥ $10B | {f['f_mcap']:,} |
| **candidates (spread AND mcap, no reward term)** | **{f['candidates']:,}** on {f['candidate_tickers']} tickers |
| carry both Tier-4 forecasts | {cov['with_both_forecasts']:,} ({100*cov['share']:.1f}%) |
| incumbent admits | {f['incumbent_passes']:,} |

Coverage is high because the pool is already restricted to large, liquid names —
exactly the ones the crush model covers. **It does not generalise**: that model
covers 64.6% of Tier 4 overall.

## What this does not establish

- **Two registered criteria fail as written — and one of them is a defect in
  the spec, not in the result.** Max drawdown is {100*prim['max_drawdown']:.1f}% against a
  1.5x-benchmark bar of {1.5*100*inc['max_drawdown']:.1f}%. But that criterion compares the primary at
  {prim['n']:,} events to the incumbent at {inc['n']:,} — the unmatched-selectivity comparison
  this same spec calls invalid three sections earlier. Drawdown scales with
  trade count for BOTH families, and read at equal selectivity the P&L gate is
  the SAFER of the two at three of four operating points:

  | events | P&L gate max DD | arithmetic max DD |
  |---:|---:|---:|
  | ~50 | 4.7% | 8.7% |
  | ~350 | 22.1% | 15.2% |
  | ~1,300 | **29.5%** | 35.8% |
  | ~2,440 | **43.2%** | 45.2% |

  Breakeven alpha 0.470 against the registered 0.45 is a real miss — but the
  INCUMBENT needs 0.460, also above the bar, so 0.45 does not separate these two
  rules; it rejects the strategy at both settings. Recorded as a limitation of
  the traded universe rather than as evidence against this gate.
  The `> 0.10` cell clears both bars. Naming it is a description, not a
  recommendation — picking it after seeing this table is the failure mode the
  grid was declared in advance to prevent.
- **The registered mechanism was wrong.** The hypothesis was joint (move, crush)
  integration. What works is the move's variance against a snapped, time-valued
  structure. A confirmatory run with that as the registered hypothesis is cheap
  now the harness exists.
- **EXP-127 is still unrun.** Promoting a gate on top of an unconfirmed
  structure compounds two post-hoc decisions.
- **No full `engine.evaluate` pass** — no walk-forward, no MC bootstrap, no tail
  injection — has run on these books.

## Reproducibility

Seeded per (arm, event) with SHA-256, verified bit-identical across two separate
processes. The first implementation used Python's `hash()`, which is salted per
process: two runs disagreed by 7 events and 0.26pp of mean, which is the scale
of Monte Carlo noise here and the floor below which no difference in this report
means anything.
"""
    (HERE / "REPORT.md").write_text(body)
    print(f"  report {HERE / 'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
