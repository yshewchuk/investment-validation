#!/usr/bin/env python3
"""EXP-116 — GBM architecture for the size model.

    python3 experiments/EXP-116_gbm_architecture_for_the_size_model/run.py

Three arms and one correction.

The correction first, because it is what the whole experiment turns on:
EXP-110's side probe measured a GBM at MAE 3.799 against the incumbent's
3.911 and made the architecture swap look obvious. Those are 85,180 rows and
96,442 rows. The GBM was scored on the subset that survived the incumbent's
feature requirements, which is a systematically easier sample — the very
confound EXP-110 was built to catch, reproduced inside its own diagnostic.
Stage 1 scores every arm on identical rows.

  gbm         the primary: HistGBM replacing the OLS + MLP blend
  gbm_log1p   the same GBM on log1p(abs_move), skew 3.53 -> 0.15
  gbm_nan     the GBM consuming rows the incumbent must drop

`gbm_nan` is deliberately NOT comparable to the others on aggregate MAE and is
never reported as if it were. It is a claim about COVERAGE, not architecture:
`walk_forward` drops any row with a non-finite feature and a GBM does not need
it to. It is judged on how many extra rows it reaches, and on like-for-like MAE
over the rows the incumbent also scored.
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

from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import (  # noqa: E402
    WalkForwardResult,
    decile_spread,
    regression_metrics,
)
from experiments import lib, size_lab  # noqa: E402

HERE = Path(__file__).resolve().parent
FIRST_TEST_YEAR = 2013


def gbm(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=400, random_state=seed
    ).fit(X, y)


def walk_forward_keeping_incomplete(frame, features, target, fit, *,
                                    year_column="year", first_test_year=None,
                                    min_train_rows=500, seed=0):
    """``walk_forward`` without the complete-rows requirement on FEATURES.

    A local copy rather than a flag on the shipped harness: this arm may well
    not promote, and an engine that grew a NaN-tolerant path for an experiment
    that lost would be a worse engine. Everything else — expanding window,
    per-year fitting, metric computation — mirrors
    ``engine.models.training.common.walk_forward`` exactly, so a difference
    between this and the incumbent is the missing-row policy and nothing else.

    The TARGET must still be finite. A row with no answer teaches nothing.
    """
    started = time.time()
    features = tuple(features)
    data = frame[np.isfinite(frame[target].to_numpy(dtype=float))].copy()

    years = sorted(int(y) for y in data[year_column].dropna().unique())
    if first_test_year is not None:
        years = [y for y in years if y >= first_test_year]

    predictions = np.full(len(data), np.nan)
    year_values = data[year_column].to_numpy()
    tested, rows = [], []
    for year in years:
        train_mask, test_mask = year_values < year, year_values == year
        if int(train_mask.sum()) < min_train_rows or not test_mask.any():
            continue
        X_train = data.loc[train_mask, list(features)].to_numpy(dtype=float)
        y_train = data.loc[train_mask, target].to_numpy(dtype=float)
        X_test = data.loc[test_mask, list(features)].to_numpy(dtype=float)
        model = fit(X_train, y_train, seed)
        predictions[test_mask] = np.asarray(model.predict(X_test), dtype=float).ravel()
        tested.append(year)
        rows.append({"year": year, "n_train": int(train_mask.sum()),
                     **regression_metrics(data.loc[test_mask, target],
                                          predictions[test_mask])})
    data["pred"] = predictions
    scored = data[np.isfinite(data["pred"])].copy()
    metrics = regression_metrics(scored[target], scored["pred"])
    metrics["decile_spread"] = decile_spread(scored[target], scored["pred"])
    metrics["oos_years"] = len(tested)
    return WalkForwardResult(
        frame=scored, target=target, features=features,
        by_year=pd.DataFrame(rows), metrics=metrics, years=tuple(tested),
        elapsed_s=round(time.time() - started, 1),
    )


def stage_1(spec: dict) -> tuple[dict, dict]:
    """Architecture and target transform, every arm on identical rows."""
    features = list(size_model.FEATURES)
    arms = [
        size_lab.Arm(name="incumbent", features=features, note="OLS + MLP(64,32)"),
        size_lab.Arm(name="gbm", features=features, fit=gbm, note="HistGBM"),
        size_lab.Arm(name="gbm_log1p", features=features, fit=gbm,
                     forward=np.log1p, inverse=np.expm1,
                     note="HistGBM on log1p(abs_move)"),
    ]
    bench = size_lab.run_bench(arms, baseline="incumbent", same_rows=True,
                               first_test_year=FIRST_TEST_YEAR)
    criteria = spec["evaluation"]["criteria"]
    out = {"rows": bench.rows, "arms": {}}
    base = bench.arms["incumbent"].metrics
    for name, res in bench.arms.items():
        block = {"metrics": res.metrics, "note": next(
            (a.note for a in arms if a.name == name), "")}
        if name != "incumbent":
            v = bench.verdict(name)
            block["verdict"] = v
            block["rmse_delta"] = round(res.metrics["rmse"] - base["rmse"], 5)
            block["clears"] = bool(
                v["years_improved"] >= criteria["min_years_improved"]
                and v["wilcoxon_p"] <= criteria["max_wilcoxon_p"]
                and v["mean_gain_excluding_best_year"] > 0
                and block["rmse_delta"] <= 0
            )
        out["arms"][name] = block
    return out, bench


def stage_2() -> dict:
    """Coverage: how many rows can a GBM reach that the incumbent cannot?"""
    features = list(size_model.FEATURES)
    panel = size_lab.prepare_panel()
    numeric = panel[features + ["abs_move"]].apply(pd.to_numeric, errors="coerce")
    complete = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    print(f"[EXP-116] stage 2: {len(panel):,} panel rows, {int(complete.sum()):,} "
          f"complete on every feature, {int((~complete).sum()):,} the incumbent "
          f"must drop", flush=True)

    run = walk_forward_keeping_incomplete(
        panel.assign(**{c: numeric[c] for c in numeric.columns}),
        features, "abs_move", gbm, first_test_year=FIRST_TEST_YEAR,
    )
    incumbent = size_lab.run_bench(
        [size_lab.Arm(name="incumbent", features=features)],
        panel=panel, baseline="incumbent", same_rows=True,
        first_test_year=FIRST_TEST_YEAR,
    ).arms["incumbent"]

    key = ["ticker", "date"]
    shared = run.frame[key + ["pred", "abs_move"]].merge(
        incumbent.predictions[key + ["pred"]], on=key, suffixes=("_nan", "_inc"))
    like = {
        "shared_rows": int(len(shared)),
        "mae_gbm_nan": round(float((shared["pred_nan"] - shared["abs_move"]).abs().mean()), 5),
        "mae_incumbent": round(float((shared["pred_inc"] - shared["abs_move"]).abs().mean()), 5),
    }
    like["gain_pp"] = round(like["mae_incumbent"] - like["mae_gbm_nan"], 5)
    return {
        "rows_scored": int(len(run.frame)),
        "rows_incumbent": int(len(incumbent.predictions)),
        "extra_rows": int(len(run.frame) - len(incumbent.predictions)),
        "metrics_on_its_own_sample": run.metrics,
        "like_for_like": like,
        "note": (
            "The aggregate metrics above are on a DIFFERENT and larger sample "
            "than every other arm, and must never be quoted beside them. Only "
            "like_for_like compares models rather than samples."
        ),
    }


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    results = {"spec_hash": lib.spec_hash(spec), "arm": spec["primary_spec"]["arm"]}

    s1, bench = stage_1(spec)
    results["stage_1"] = s1
    results["stage_2"] = stage_2()

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=str))
    for name in ("gbm", "gbm_log1p"):
        written = size_lab.write_figures(bench, name, HERE / "figures" / name)
        print(f"[EXP-116] {name}: {len(written)} figure(s)", flush=True)

    print(f"\n{'arm':12s} {'r':>8} {'MAE':>8} {'RMSE':>8} {'dMAE':>9} {'dRMSE':>8}")
    base = s1["arms"]["incumbent"]["metrics"]
    for name, block in s1["arms"].items():
        m = block["metrics"]
        if name == "incumbent":
            print(f"{name:12s} {m['r']:>8.4f} {m['mae']:>8.4f} {m['rmse']:>8.4f}")
            continue
        v = block["verdict"]
        print(f"{name:12s} {m['r']:>8.4f} {m['mae']:>8.4f} {m['rmse']:>8.4f} "
              f"{v['mean_gain_pp']:>+9.4f} {block['rmse_delta']:>+8.4f}  "
              f"{v['years_improved']}/{v['years_total']} yrs p={v['wilcoxon_p']} "
              f"-> {'CLEARS' if block['clears'] else 'does not clear'}")
    s2 = results["stage_2"]
    print(f"\ncoverage: {s2['rows_scored']:,} rows scored vs {s2['rows_incumbent']:,} "
          f"({s2['extra_rows']:+,}); like-for-like on {s2['like_for_like']['shared_rows']:,} "
          f"shared rows: {s2['like_for_like']['gain_pp']:+.4f}pp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
