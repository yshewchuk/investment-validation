#!/usr/bin/env python3
"""EXP-113 — remove collinear inputs from the champion.

    python3 experiments/EXP-113_size_v2_0_step_4_remove_collinear_inputs_/run.py

`abs_dist_high` is not a weak feature, it is a duplicate. `dist_high` is
distance from the 52-week high and is never positive, so taking its absolute
value negates it and nothing more: corr -1.000000, VIF infinite. It reached the
live champion inside EXP-109's `all_three` bundle, and EXP-108 had already
scored it at +0.0001 r on its own — the measurement was there and the reason
was not.

A removal is judged on not harming, which is the opposite direction from an
addition, so this run reports the variance-inflation diagnostics beside the
accuracy: the point is as much that the design matrix stops being singular as
that MAE holds.
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
from experiments import lib, size_lab  # noqa: E402

HERE = Path(__file__).resolve().parent


def vif_table(frame: pd.DataFrame, features) -> list[dict]:
    """Variance inflation per feature: how much of it the others already explain."""
    X = frame[list(features)].apply(pd.to_numeric, errors="coerce")
    X = X[np.isfinite(X.to_numpy(dtype=float)).all(axis=1)]
    Z = ((X - X.mean()) / X.std()).to_numpy(dtype=float)
    out = []
    for i, name in enumerate(features):
        y, M = Z[:, i], np.delete(Z, i, axis=1)
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        r2 = 1 - ((y - M @ beta) ** 2).sum() / (y ** 2).sum()
        out.append({"feature": name, "r2_vs_others": round(float(r2), 6),
                    "vif": float("inf") if r2 >= 1 else round(1 / (1 - r2), 3)})
    return sorted(out, key=lambda d: -d["vif"])


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    full = list(size_model.FEATURES)
    panel = size_lab.prepare_panel()

    before = vif_table(panel, full)
    worst = before[0]
    print(f"[EXP-113] worst VIF before: {worst['feature']} = {worst['vif']} "
          f"(R2 vs others {worst['r2_vs_others']})", flush=True)

    arms = [
        size_lab.Arm(name="incumbent", features=full),
        size_lab.Arm(name="drop_abs_dist_high",
                     features=[f for f in full if f != "abs_dist_high"]),
        size_lab.Arm(name="drop_mean_prior_abs_move",
                     features=[f for f in full if f != "mean_prior_abs_move"]),
        size_lab.Arm(name="drop_both",
                     features=[f for f in full
                               if f not in ("abs_dist_high", "mean_prior_abs_move")]),
    ]
    bench = size_lab.run_bench(arms, panel=panel, baseline="incumbent")

    out = {"spec_hash": lib.spec_hash(spec), "rows": bench.rows,
           "vif_before": before,
           "vif_after_primary": vif_table(panel, [f for f in full if f != "abs_dist_high"]),
           "arms": {}}
    for arm in arms:
        res = bench.arms[arm.name]
        block = {"features": len(arm.features), "metrics": res.metrics,
                 "by_year": res.by_year.to_dict(orient="records")}
        if arm.name != "incumbent":
            block["verdict"] = bench.verdict(arm.name)
            v = block["verdict"]
            print(f"  {arm.name:26s} MAE {res.metrics['mae']:.4f} "
                  f"({v['mean_gain_pp']:+.4f}pp) {v['years_improved']}/{v['years_total']} yrs "
                  f"p={v['wilcoxon_p']}", flush=True)
        out["arms"][arm.name] = block

    for arm in arms[1:]:
        size_lab.write_figures(bench, arm.name, HERE / "figures" / arm.name)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"[EXP-113] worst VIF after removing abs_dist_high: "
          f"{out['vif_after_primary'][0]['feature']} = {out['vif_after_primary'][0]['vif']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
