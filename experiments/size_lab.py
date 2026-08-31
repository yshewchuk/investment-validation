"""Shared bench for size-model challengers, so v2.0's steps are comparable.

Every change on the road to a v2.0 size model — a data fix, a feature removal,
an architecture swap, a loss change — asks the same question: does the driver
predict ``abs_move`` better, and is the improvement real rather than one good
year? Answering it the same way each time is what makes the steps addable
instead of a pile of incomparable one-offs.

What every arm gets:

* **The same rows.** Arms are scored on the intersection where every arm's
  features are present, so a candidate carrying an extra input is never judged
  on an easier sample than the incumbent.
* **The same walk-forward.** Expanding window by year, parameters frozen before
  each test year, exactly as ``engine.models.training.common.walk_forward`` does
  it for the registered champions.
* **The same verdict rule.** Per-year win count and a Wilcoxon signed-rank p on
  the per-year MAE deltas, plus the drop-the-best-year check. Magnitude
  thresholds are deliberately not used: EXP-108 showed they reach the right
  answer for the wrong reason, and the wrong answer for a consistent small one.
* **The same figures**, so two reports can be read side by side.

The bench is descriptive. It never promotes anything: a win here earns a
gate-invariance check and a registry decision, both of which live elsewhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from engine.features import load_panel
from engine.models.training import size_model
from engine.models.training.common import walk_forward

__all__ = ["Arm", "ArmResult", "BenchResult", "run_bench", "write_figures", "SENTINEL_COLUMNS"]

FIRST_TEST_YEAR = 2013

#: Columns where ORATS encodes "no quote" as a hard zero rather than as null.
#: 21.4% of ``daily_market`` rows carry ``implied_move == 0`` against 689 nulls,
#: spread evenly across every year since 2009 — so it is a sentinel, not a
#: coverage boundary, and the events carrying it realize a HIGHER mean move
#: (6.68% vs 5.72%) than the rest. Left as zero it tells the model the market
#: expects no move on exactly the events that move most.
SENTINEL_COLUMNS = ("or_implied",)


@dataclass
class Arm:
    """One thing to try. ``fit`` defaults to the incumbent's own architecture."""

    name: str
    features: Sequence[str]
    fit: Callable = size_model.fit
    target: str = "abs_move"
    #: Applied to the target before fitting and inverted after predicting, for
    #: arms that fit a transformed scale (log, say). Identity by default.
    forward: Callable[[np.ndarray], np.ndarray] | None = None
    inverse: Callable[[np.ndarray], np.ndarray] | None = None
    note: str = ""


@dataclass
class ArmResult:
    name: str
    metrics: dict
    by_year: pd.DataFrame
    predictions: pd.DataFrame
    note: str = ""


@dataclass
class BenchResult:
    rows: int
    arms: dict = field(default_factory=dict)
    baseline: str = ""

    def verdict(self, arm: str, against: str | None = None) -> dict:
        """Consistency of ``arm`` against the baseline, per the shared rule."""
        from scipy import stats

        base = self.arms[against or self.baseline]
        cand = self.arms[arm]
        a = base.by_year.set_index("year")
        b = cand.by_year.set_index("year")
        years = [y for y in a.index if y in b.index]
        deltas = np.array([float(a.loc[y, "mae"] - b.loc[y, "mae"]) for y in years])
        if not len(deltas):
            return {"available": False}
        improved = int((deltas > 0).sum())
        p = float(stats.wilcoxon(deltas).pvalue) if len(deltas) > 5 else float("nan")
        without_best = np.delete(deltas, int(np.argmax(deltas)))
        return {
            "available": True,
            "years": [int(y) for y in years],
            "per_year_gain": {int(y): round(float(d), 5) for y, d in zip(years, deltas)},
            "years_improved": improved,
            "years_total": len(years),
            "mean_gain_pp": round(float(deltas.mean()), 5),
            "mean_gain_excluding_best_year": round(float(without_best.mean()), 5),
            "best_year": int(years[int(np.argmax(deltas))]),
            "wilcoxon_p": round(p, 5),
            "consistent": bool(
                improved >= 0.7 * len(years) and p <= 0.05 and without_best.mean() > 0
            ),
        }


def prepare_panel(*, fix_sentinels: bool = False) -> pd.DataFrame:
    """The size model's training frame, optionally with the zero-sentinel fixed.

    ``fix_sentinels`` is what EXP-110 tests: turning the encoded-zero "no quote"
    into a real null, so a model can treat it as missing instead of as a
    forecast of no movement.
    """
    panel = load_panel()
    if fix_sentinels:
        panel = panel.copy()
        for column in SENTINEL_COLUMNS:
            if column in panel.columns:
                values = pd.to_numeric(panel[column], errors="coerce")
                panel[column] = values.mask(values <= 0)
    return size_model.prepare(panel)


def common_rows(frame: pd.DataFrame, arms: Sequence[Arm], target: str) -> pd.Index:
    needed = sorted({f for arm in arms for f in arm.features} | {target})
    values = frame[needed].apply(pd.to_numeric, errors="coerce")
    return frame.index[np.isfinite(values.to_numpy(dtype=float)).all(axis=1)]


def run_bench(
    arms: Sequence[Arm],
    *,
    panel: pd.DataFrame | None = None,
    baseline: str | None = None,
    first_test_year: int = FIRST_TEST_YEAR,
    same_rows: bool = True,
) -> BenchResult:
    """Walk-forward every arm and collect per-year metrics and OOS predictions."""
    data = prepare_panel() if panel is None else panel
    target = arms[0].target
    if same_rows:
        data = data.loc[common_rows(data, arms, target)]
    print(f"[bench] {len(data):,} rows usable by every arm", flush=True)

    out = BenchResult(rows=int(len(data)), baseline=baseline or arms[0].name)
    for arm in arms:
        frame = data
        if arm.forward is not None:
            frame = frame.assign(
                **{f"_t_{arm.target}": arm.forward(frame[arm.target].to_numpy(dtype=float))}
            )
            fit_target = f"_t_{arm.target}"
        else:
            fit_target = arm.target

        result = walk_forward(
            frame, list(arm.features), fit_target, arm.fit,
            first_test_year=first_test_year,
        )
        predictions = result.frame.copy()
        if arm.inverse is not None:
            # Metrics must be reported on the ORIGINAL scale or two arms fitting
            # different transforms cannot be compared at all.
            predictions["pred"] = arm.inverse(predictions["pred"].to_numpy(dtype=float))
            predictions[arm.target] = data.loc[predictions.index, arm.target]
        metrics, by_year = _rescore(predictions, arm.target)
        out.arms[arm.name] = ArmResult(arm.name, metrics, by_year, predictions, arm.note)
        print(f"  {arm.name:34s} r={metrics['r']:.4f} mae={metrics['mae']:.4f} "
              f"rmse={metrics['rmse']:.4f} n={metrics['n']:,}", flush=True)
    return out


def _rescore(frame: pd.DataFrame, target: str) -> tuple[dict, pd.DataFrame]:
    """Metrics on the original scale, recomputed rather than trusted.

    ``walk_forward`` reports metrics against whatever it was fitted on; an arm
    that fitted ``log1p`` would otherwise publish a MAE in log points and look
    like a landslide.
    """
    y = pd.to_numeric(frame[target], errors="coerce")
    pred = pd.to_numeric(frame["pred"], errors="coerce")
    ok = y.notna() & pred.notna()
    y, pred = y[ok], pred[ok]
    err = pred - y
    metrics = {
        "n": int(len(y)),
        "r": float(np.corrcoef(pred, y)[0, 1]),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }
    rows = []
    years = pd.to_datetime(frame.loc[ok.index[ok], "date"]).dt.year
    for year, idx in y.groupby(years).groups.items():
        e = pred.loc[idx] - y.loc[idx]
        rows.append({
            "year": int(year), "n": int(len(idx)),
            "r": float(np.corrcoef(pred.loc[idx], y.loc[idx])[0, 1]) if len(idx) > 2 else np.nan,
            "mae": float(e.abs().mean()), "rmse": float(np.sqrt((e**2).mean())),
            "bias": float(e.mean()),
        })
    return metrics, pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def write_figures(bench: BenchResult, arm: str, out_dir: Path, *, against: str | None = None) -> list[Path]:
    """Four charts per challenger, the same four every time.

    A table of aggregate metrics hides the two things that decide whether a
    change is real: whether it holds year on year, and whether it holds across
    the range of the outcome rather than only where the data is dense.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    base = bench.arms[against or bench.baseline]
    cand = bench.arms[arm]
    verdict = bench.verdict(arm, against=against)
    written: list[Path] = []

    # 1. per-year MAE, both arms, with the delta beneath
    a = base.by_year.set_index("year")
    b = cand.by_year.set_index("year")
    years = [y for y in a.index if y in b.index]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), height_ratios=[2, 1], sharex=True)
    width = 0.4
    idx = np.arange(len(years))
    ax1.bar(idx - width/2, [a.loc[y, "mae"] for y in years], width, label=base.name, color="#8b949e")
    ax1.bar(idx + width/2, [b.loc[y, "mae"] for y in years], width, label=cand.name, color="#58a6ff")
    ax1.set_ylabel("MAE (pp)"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    ax1.set_title(f"Walk-forward OOS MAE by year — lower is better")
    deltas = [a.loc[y, "mae"] - b.loc[y, "mae"] for y in years]
    ax2.bar(idx, deltas, color=["#3fb950" if d > 0 else "#f85149" for d in deltas])
    ax2.axhline(0, color="#30363d", lw=1)
    ax2.set_ylabel("gain (pp)"); ax2.set_xticks(idx); ax2.set_xticklabels(years, rotation=45)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_title(f"improved {verdict['years_improved']}/{verdict['years_total']} years · "
                  f"Wilcoxon p={verdict['wilcoxon_p']} · "
                  f"mean {verdict['mean_gain_pp']:+.4f}pp "
                  f"({verdict['mean_gain_excluding_best_year']:+.4f} excl. best year)",
                  fontsize=9)
    fig.tight_layout(); path = out_dir / "by_year.png"; fig.savefig(path, dpi=110); plt.close(fig)
    written.append(path)

    # 2. predicted vs actual, binned — where in the range does it improve?
    fig, ax = plt.subplots(figsize=(8, 5))
    for res, colour in ((base, "#8b949e"), (cand, "#58a6ff")):
        d = res.predictions.dropna(subset=["pred", "abs_move"])
        bins = pd.qcut(d["pred"], 12, labels=False, duplicates="drop")
        g = d.groupby(bins).agg(pred=("pred", "mean"), actual=("abs_move", "mean"))
        ax.plot(g["pred"], g["actual"], "o-", color=colour, label=res.name)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "--", color="#d0342c", lw=1, label="perfect calibration")
    ax.set_xlabel("predicted |move| (%)"); ax.set_ylabel("realized |move| (%)")
    ax.set_title("Calibration — predicted vs realized, by prediction decile")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); path = out_dir / "calibration.png"; fig.savefig(path, dpi=110); plt.close(fig)
    written.append(path)

    # 3. error distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    for res, colour in ((base, "#8b949e"), (cand, "#58a6ff")):
        d = res.predictions.dropna(subset=["pred", "abs_move"])
        err = (d["pred"] - d["abs_move"]).clip(-20, 20)
        ax.hist(err, bins=90, histtype="step", lw=1.6, color=colour,
                label=f"{res.name} (MAE {res.metrics['mae']:.3f})")
    ax.axvline(0, color="#d0342c", lw=1, ls="--")
    ax.set_xlabel("prediction − realized (pp)"); ax.set_ylabel("events")
    ax.set_title("Error distribution (clipped to ±20pp)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); path = out_dir / "errors.png"; fig.savefig(path, dpi=110); plt.close(fig)
    written.append(path)

    # 4. residual spread against prediction — the heteroskedasticity the
    #    scorer's single global residual pool currently ignores
    fig, ax = plt.subplots(figsize=(8, 5))
    for res, colour in ((base, "#8b949e"), (cand, "#58a6ff")):
        d = res.predictions.dropna(subset=["pred", "abs_move"])
        bins = pd.qcut(d["pred"], 10, labels=False, duplicates="drop")
        g = (d.assign(err=d["pred"] - d["abs_move"], b=bins)
               .groupby("b").agg(pred=("pred", "mean"), sd=("err", "std")))
        ax.plot(g["pred"], g["sd"], "o-", color=colour, label=res.name)
    ax.set_xlabel("predicted |move| (%)"); ax.set_ylabel("residual sd (pp)")
    ax.set_title("Residual spread rises with the prediction — the scorer draws from one pool")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); path = out_dir / "residual_spread.png"; fig.savefig(path, dpi=110); plt.close(fig)
    written.append(path)
    return written


def metrics_table(bench: BenchResult, arms: Sequence[str] | None = None) -> str:
    """Markdown, original scale, deltas against the baseline."""
    base = bench.arms[bench.baseline]
    names = list(arms) if arms else list(bench.arms)
    lines = ["| arm | n | r | Δr | MAE | ΔMAE | RMSE | bias |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in names:
        m = bench.arms[name].metrics
        dr = m["r"] - base.metrics["r"]
        dm = base.metrics["mae"] - m["mae"]
        mark = " *(baseline)*" if name == bench.baseline else ""
        lines.append(
            f"| `{name}`{mark} | {m['n']:,} | {m['r']:.4f} | {dr:+.4f} | "
            f"{m['mae']:.4f} | {dm:+.4f} | {m['rmse']:.4f} | {m['bias']:+.4f} |"
        )
    return "\n".join(lines)
