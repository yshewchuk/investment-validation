#!/usr/bin/env python3
"""EXP-112 — prediction intervals that vary with the prediction.

    python3 experiments/EXP-112_size_v2_0_step_3_prediction_intervals_th/run.py

Every arm shares ONE set of point predictions — the incumbent's walk-forward
out-of-sample output. Only the distribution around them differs, so MAE and r
are identical by construction and any movement in them would be a bug.

  global                  one flat residual pool, as shipped today
  by_prediction_decile    the pool binned by predicted |move| decile  (primary)
  by_prediction_and_quote decile crossed with has_implied_quote

Causality is the whole difficulty here. For test year Y the residual pool AND
the decile edges come only from out-of-sample residuals of years before Y — a
pool containing the year it is scoring would be reading its own answer, and
would make every arm look perfectly calibrated.

Judged on coverage and distributional distance, never on MAE:

  coverage_80   share of outcomes inside the 80% interval; 0.80 is right
  coverage_50   0.50 is right
  pit_ks        how far the probability-integral transform is from uniform
  crps          continuous ranked probability score, lower better
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import walk_forward  # noqa: E402
from experiments import lib, size_lab  # noqa: E402

HERE = Path(__file__).resolve().parent
DECILES = 10
#: Below this many residuals a conditional bin falls back to the global pool.
#: A bin thin enough to have noisy quantiles is worse than no conditioning.
MIN_POOL = 250


def predictive_stats(pred: float, pool: np.ndarray, actual: float) -> dict:
    """Coverage indicators, PIT value and CRPS for one event."""
    lo80, hi80 = np.quantile(pool, [0.10, 0.90])
    lo50, hi50 = np.quantile(pool, [0.25, 0.75])
    resid = actual - pred
    # PIT: where the realized residual sits in the pool's own distribution.
    pit = float((pool <= resid).mean())
    # CRPS of an empirical predictive sample, computed the cheap exact way.
    s = np.sort(pool)
    n = len(s)
    crps = float(np.abs(s - resid).mean() - 0.5 * np.abs(s[:, None] - s[None, :]).mean()) \
        if n <= 400 else float(np.abs(s - resid).mean()
                               - 0.5 * np.abs(np.diff(np.sort(np.random.default_rng(0).choice(s, 400)))).mean())
    return {
        "in80": bool(lo80 <= resid <= hi80),
        "in50": bool(lo50 <= resid <= hi50),
        "pit": pit,
        "crps": crps,
        "width80": float(hi80 - lo80),
    }


def score_arm(frame: pd.DataFrame, key: str | None) -> pd.DataFrame:
    """Walk the years forward, building each year's pool from earlier years only."""
    years = sorted(frame["year"].unique())
    rows = []
    for year in years:
        past = frame[frame["year"] < year]
        now = frame[frame["year"] == year]
        if len(past) < MIN_POOL or now.empty:
            continue
        past_res = (past["abs_move"] - past["pred"]).to_numpy(dtype=float)
        if key is None:
            pools = {None: past_res}
            bins_now = pd.Series(None, index=now.index, dtype=object)
        else:
            # Decile edges from the PAST only, so the current year cannot shift them.
            edges = np.quantile(past["pred"].to_numpy(dtype=float),
                                np.linspace(0, 1, DECILES + 1))
            edges[0], edges[-1] = -np.inf, np.inf
            past_bin = pd.cut(past["pred"], np.unique(edges), labels=False, include_lowest=True)
            now_bin = pd.cut(now["pred"], np.unique(edges), labels=False, include_lowest=True)
            if key == "quote":
                past_bin = past_bin.astype(str) + "|" + past["has_implied_quote"].astype(int).astype(str)
                now_bin = now_bin.astype(str) + "|" + now["has_implied_quote"].astype(int).astype(str)
            pools = {b: past_res[(past_bin == b).to_numpy()] for b in past_bin.dropna().unique()}
            bins_now = now_bin

        for idx, row in now.iterrows():
            b = bins_now.loc[idx] if key is not None else None
            pool = pools.get(b)
            if pool is None or len(pool) < MIN_POOL:
                pool = past_res  # thin bin: fall back rather than trust it
            stats = predictive_stats(float(row["pred"]), pool, float(row["abs_move"]))
            stats.update({"year": int(year), "pred": float(row["pred"]),
                          "abs_move": float(row["abs_move"]),
                          "has_quote": int(row["has_implied_quote"])})
            rows.append(stats)
    return pd.DataFrame(rows)


def summarise(scored: pd.DataFrame) -> dict:
    from scipy import stats as st

    pit = scored["pit"].to_numpy(dtype=float)
    return {
        "n": int(len(scored)),
        "coverage_80": float(scored["in80"].mean()),
        "coverage_50": float(scored["in50"].mean()),
        "coverage_80_error": float(abs(scored["in80"].mean() - 0.80)),
        "pit_ks": float(st.kstest(pit, "uniform").statistic),
        "crps": float(scored["crps"].mean()),
        "mean_width80": float(scored["width80"].mean()),
    }


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    features = list(size_model.FEATURES)
    panel = size_lab.prepare_panel()
    num = panel[features + ["abs_move"]].apply(pd.to_numeric, errors="coerce")
    panel = panel[np.isfinite(num.to_numpy(dtype=float)).all(axis=1)]

    print(f"[EXP-112] {len(panel):,} rows; walking the point model forward once", flush=True)
    wf = walk_forward(panel, features, "abs_move", size_model.fit, first_test_year=2013)
    base = wf.frame[["ticker", "date", "pred", "abs_move"]].copy()
    base = base.merge(panel[["ticker", "date", "has_implied_quote"]], on=["ticker", "date"])
    base["year"] = pd.to_datetime(base["date"]).dt.year
    print(f"[EXP-112] {len(base):,} OOS predictions, shared by every arm", flush=True)

    arms = {"global": None, "by_prediction_decile": "pred", "by_prediction_and_quote": "quote"}
    scored, summary = {}, {}
    for name, key in arms.items():
        scored[name] = score_arm(base, key)
        summary[name] = summarise(scored[name])
        s = summary[name]
        print(f"  {name:26s} cov80 {s['coverage_80']:.4f} (err {s['coverage_80_error']:.4f})  "
              f"cov50 {s['coverage_50']:.4f}  PIT-KS {s['pit_ks']:.4f}  CRPS {s['crps']:.4f}",
              flush=True)

    # Proof that the point estimate did not move: every arm shares `base`.
    err = (base["pred"] - base["abs_move"]).abs()
    summary["_point_estimate"] = {
        "mae": float(err.mean()),
        "r": float(np.corrcoef(base["pred"], base["abs_move"])[0, 1]),
        "note": "identical for every arm by construction — only the interval differs",
    }

    write_figures(scored, summary, HERE / "figures")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(
        {"spec_hash": lib.spec_hash(spec), "summary": summary,
         "coverage_by_decile": {
             name: _by_decile(df).to_dict(orient="records") for name, df in scored.items()}},
        indent=1, default=str))
    return 0


def _by_decile(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(b=pd.qcut(df["pred"], DECILES, labels=False, duplicates="drop"))
    return (d.groupby("b")
              .agg(pred=("pred", "mean"), cov80=("in80", "mean"),
                   width=("width80", "mean"), n=("in80", "size"))
              .reset_index())


def write_figures(scored: dict, summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colours = {"global": "#8b949e", "by_prediction_decile": "#3fb950",
               "by_prediction_and_quote": "#58a6ff"}

    # 1. THE chart: coverage by prediction decile against the 80% target
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, df in scored.items():
        g = _by_decile(df)
        ax1.plot(g["pred"], g["cov80"], "o-", color=colours[name], label=name)
        ax2.plot(g["pred"], g["width"], "o-", color=colours[name], label=name)
    ax1.axhline(0.80, ls="--", color="#d0342c", label="target 0.80")
    ax1.set_xlabel("predicted |move| (%)"); ax1.set_ylabel("share inside the 80% interval")
    ax1.set_title("Coverage by prediction decile — flat on target is correct")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_xlabel("predicted |move| (%)"); ax2.set_ylabel("80% interval width (pp)")
    ax2.set_title("A global pool gives every event the same width")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "coverage.png", dpi=110); plt.close(fig)

    # 2. PIT — flat is calibrated
    fig, axes = plt.subplots(1, len(scored), figsize=(4.2 * len(scored), 4), sharey=True)
    for ax, (name, df) in zip(np.atleast_1d(axes), scored.items()):
        ax.hist(df["pit"], bins=20, range=(0, 1), color=colours[name], alpha=0.85)
        ax.axhline(len(df) / 20, ls="--", color="#d0342c")
        ax.set_title(f"{name}\nPIT-KS {summary[name]['pit_ks']:.4f}", fontsize=10)
        ax.set_xlabel("PIT"); ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("events")
    fig.suptitle("Probability-integral transform — flat means calibrated", y=1.02)
    fig.tight_layout(); fig.savefig(out_dir / "pit.png", dpi=110, bbox_inches="tight"); plt.close(fig)

    # 3. coverage by quote availability, the EXP-111 population split
    fig, ax = plt.subplots(figsize=(7, 4.5))
    width = 0.25
    idx = np.arange(2)
    for i, (name, df) in enumerate(scored.items()):
        vals = [df.loc[df.has_quote == q, "in80"].mean() for q in (1, 0)]
        ax.bar(idx + (i - 1) * width, vals, width, color=colours[name], label=name)
    ax.axhline(0.80, ls="--", color="#d0342c")
    ax.set_xticks(idx); ax.set_xticklabels(["with a quote", "no quote"])
    ax.set_ylabel("share inside the 80% interval"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Coverage by quote availability")
    fig.tight_layout(); fig.savefig(out_dir / "by_quote.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
