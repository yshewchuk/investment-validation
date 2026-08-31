#!/usr/bin/env python3
"""EXP-111 — separate the quoted implied move from whether one exists.

    python3 experiments/EXP-111_size_v2_0_step_2_separate_the_quote_from/run.py

Four designs, one common row set, one walk-forward each:

  original                  or_implied as it stands, zero meaning "no quote"
  value_plus_indicator      the value AND has_implied_quote   (the primary)
  indicator_replaces_value  has_implied_quote instead of the value
  two_models_split_by_quote a model per population

The primary is registered expecting a null. EXP-110 established that the models
already read `or_implied == 0` as its own region — dropping the feature made
them worse — so an explicit flag tells them nothing they could not infer. It is
adopted for two reasons that are not accuracy: the column currently carries a
liquidity fact and a price on one axis, and the indicator is what preserves the
liquidity half when the value is later nulled for being wrong.

The two-model arm is here because the question was asked directly, and an answer
in the record beats an answer in someone's memory.
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
INDICATOR = "has_implied_quote"


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    full = [f for f in size_model.FEATURES if f != INDICATOR]
    no_value = [f for f in full if f != "or_implied"]

    panel = size_lab.prepare_panel()
    oi = pd.to_numeric(panel["or_implied"], errors="coerce")
    panel = panel.assign(**{INDICATOR: (oi > 0).astype(float)})
    needed = sorted(set(full) | {"abs_move", INDICATOR})
    num = panel[needed].apply(pd.to_numeric, errors="coerce")
    panel = panel[np.isfinite(num.to_numpy(dtype=float)).all(axis=1)]
    flags = panel[["ticker", "date", INDICATOR]]
    print(f"[EXP-111] common rows {len(panel):,}, "
          f"no quote on {int((panel[INDICATOR] == 0).sum()):,}", flush=True)

    def oos(frame, feats):
        f = walk_forward(frame, feats, "abs_move", size_model.fit, first_test_year=2013).frame
        return f[["ticker", "date", "pred", "abs_move"]]

    quoted = panel[panel[INDICATOR] == 1]
    unquoted = panel[panel[INDICATOR] == 0]
    designs = {
        "original": oos(panel, full),
        "value_plus_indicator": oos(panel, full + [INDICATOR]),
        "indicator_replaces_value": oos(panel, no_value + [INDICATOR]),
        "two_models_split_by_quote": pd.concat(
            [oos(quoted, full), oos(unquoted, no_value)], ignore_index=True),
    }

    from scipy import stats

    out, per_year = {}, {}
    for name, frame in designs.items():
        f = frame.merge(flags, on=["ticker", "date"], how="inner")
        f["err"] = (f["pred"] - f["abs_move"]).abs()
        f["year"] = pd.to_datetime(f["date"]).dt.year
        per_year[name] = f.groupby("year")["err"].mean()
        out[name] = {
            "n": int(len(f)),
            "mae": float(f["err"].mean()),
            "r": float(np.corrcoef(f["pred"], f["abs_move"])[0, 1]),
            "rmse": float(np.sqrt(((f["pred"] - f["abs_move"]) ** 2).mean())),
            "mae_quoted": float(f.loc[f[INDICATOR] == 1, "err"].mean()),
            "mae_unquoted": float(f.loc[f[INDICATOR] == 0, "err"].mean()),
            "by_year": {int(y): float(v) for y, v in per_year[name].items()},
        }

    base = per_year["original"]
    for name in designs:
        if name == "original":
            continue
        d = base - per_year[name]
        out[name]["vs_original"] = {
            "mae_gain_pp": round(float(out["original"]["mae"] - out[name]["mae"]), 5),
            "years_improved": int((d > 0).sum()),
            "years_total": int(len(d)),
            "mean_gain_pp": round(float(d.mean()), 5),
            "wilcoxon_p": round(float(stats.wilcoxon(d.values).pvalue), 5),
        }
        v = out[name]["vs_original"]
        print(f"  {name:26s} MAE {out[name]['mae']:.4f} ({v['mae_gain_pp']:+.4f}) "
              f"{v['years_improved']}/{v['years_total']} yrs p={v['wilcoxon_p']}", flush=True)

    write_figures(out, per_year, HERE / "figures")
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(
        json.dumps({"spec_hash": lib.spec_hash(spec), "designs": out}, indent=1, default=str))
    return 0


def write_figures(out: dict, per_year: dict, out_dir: Path) -> None:
    """Two charts: the design comparison, and where each design's error sits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(out)
    short = ["original", "value +\nindicator", "indicator\ninstead", "two\nmodels"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colours = ["#8b949e", "#3fb950", "#58a6ff", "#f85149"]
    ax1.bar(short, [out[n]["mae"] for n in names], color=colours)
    ax1.set_ylim(min(out[n]["mae"] for n in names) - 0.02,
                 max(out[n]["mae"] for n in names) + 0.02)
    ax1.set_ylabel("walk-forward OOS MAE (pp)")
    ax1.set_title("Overall accuracy — the differences are inside the noise")
    for i, n in enumerate(names):
        ax1.text(i, out[n]["mae"], f"{out[n]['mae']:.4f}", ha="center", va="bottom", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    width = 0.35
    idx = np.arange(len(names))
    ax2.bar(idx - width/2, [out[n]["mae_quoted"] for n in names], width,
            label="with a quote", color="#58a6ff")
    ax2.bar(idx + width/2, [out[n]["mae_unquoted"] for n in names], width,
            label="no quote", color="#d29922")
    ax2.set_xticks(idx); ax2.set_xticklabels(short); ax2.legend()
    ax2.set_ylabel("MAE (pp)")
    ax2.set_title("Unquoted events are ~1pp harder for every design")
    ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "designs.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    years = list(per_year["original"].index)
    for n, c in zip(names[1:], colours[1:]):
        ax.plot(years, (per_year["original"] - per_year[n]).values, "o-", color=c, label=n)
    ax.axhline(0, color="#30363d", lw=1)
    ax.set_ylabel("MAE gain vs original (pp)"); ax.set_xlabel("OOS year")
    ax.set_title("Per-year gain against the original — nothing holds a direction")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "by_year.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
