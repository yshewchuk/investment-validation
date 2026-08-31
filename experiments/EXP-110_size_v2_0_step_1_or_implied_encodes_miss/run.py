#!/usr/bin/env python3
"""EXP-110 — or_implied encodes missing as zero.

    python3 experiments/EXP-110_size_v2_0_step_1_or_implied_encodes_miss/run.py

Step 1 of the size v2.0 programme, and the one the rest sit on: a feature that
says "the market expects no move" on 12% of the training set, where those events
in fact move MORE than average.

Three arms, because nulling a sentinel is not free. The incumbent architecture
is an OLS + MLP blend and neither half consumes a null, so `walk_forward` drops
any row carrying one — a fix that improves the surviving rows while deleting
12% of the sample is not obviously a win. The arms separate those effects:

  incumbent          or_implied as it stands, zeros and all
  sentinel_to_null   zeros nulled, and the rows they sit on then dropped
  drop_or_implied    the feature removed entirely, every row kept

Reading the three together answers a question none of them answers alone:
whether the information is in the feature, in the rows, or in neither.
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


def sentinel_census() -> dict:
    """The evidence that zero is a sentinel, recomputed rather than quoted."""
    from engine.data import store

    daily = store.read_table("daily_market", columns=["date", "implied_move"])
    im = pd.to_numeric(daily["implied_move"], errors="coerce")
    by_year = (
        daily.assign(zero=(im == 0), year=pd.to_datetime(daily["date"]).dt.year)
        .groupby("year")["zero"].agg(["sum", "size", "mean"])
    )
    panel = size_lab.prepare_panel()
    oi = pd.to_numeric(panel["or_implied"], errors="coerce")
    zero = oi <= 0
    ok = (~zero) & oi.notna() & panel["abs_move"].notna()
    return {
        "daily_rows": int(len(daily)),
        "daily_zero": int((im == 0).sum()),
        "daily_zero_share": round(float((im == 0).mean()), 4),
        "daily_null": int(im.isna().sum()),
        "zero_share_by_year": {
            int(y): round(float(r["mean"]), 4)
            for y, r in by_year.iterrows() if r["size"] >= 50
        },
        "panel_rows": int(len(panel)),
        "panel_zero": int(zero.sum()),
        "mean_abs_move_when_zero": round(float(panel.loc[zero, "abs_move"].mean()), 4),
        "mean_abs_move_otherwise": round(float(panel.loc[~zero, "abs_move"].mean()), 4),
        "corr_with_zeros": round(float(oi.corr(panel["abs_move"])), 4),
        "corr_without_zeros": round(float(oi[ok].corr(panel.loc[ok, "abs_move"])), 4),
    }


def _like_for_like(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """MAE of both arms on only the rows BOTH of them scored."""
    from scipy import stats

    key = ["ticker", "date"]
    m = a[key + ["pred", "abs_move"]].merge(b[key + ["pred"]], on=key, suffixes=("_a", "_b"))
    y = m["abs_move"]
    mae_a = float((m["pred_a"] - y).abs().mean())
    mae_b = float((m["pred_b"] - y).abs().mean())
    m = m.assign(year=pd.to_datetime(m["date"]).dt.year)
    per_year = m.groupby("year").apply(
        lambda d: float((d["pred_a"] - d["abs_move"]).abs().mean()
                        - (d["pred_b"] - d["abs_move"]).abs().mean()),
        include_groups=False,
    )
    return {
        "shared_rows": int(len(m)),
        "rows_a": int(len(a)), "rows_b": int(len(b)),
        "mae_a": round(mae_a, 5), "mae_b": round(mae_b, 5),
        "gain_pp": round(mae_a - mae_b, 5),
        "years_improved": int((per_year > 0).sum()),
        "years_total": int(len(per_year)),
        "wilcoxon_p": round(float(stats.wilcoxon(per_year.values).pvalue), 5),
        "per_year_gain": {int(y): round(float(v), 5) for y, v in per_year.items()},
    }


def _gbm_probe() -> dict:
    """Can an architecture that consumes nulls do better with the sentinel fixed?

    It cannot be answered inside the current harness: ``walk_forward`` drops any
    row with a non-finite feature before the model is fitted, so a GBM sees the
    same rows an OLS+MLP does. The probe is kept because that constraint is the
    finding — exploiting explicit missingness needs a harness change first.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    def gbm(X, y, seed=0):
        return HistGradientBoostingRegressor(learning_rate=0.06, random_state=seed).fit(X, y)

    features = list(size_model.FEATURES)
    arms = {}
    for name, fix in (("gbm_zeros", False), ("gbm_nulls", True)):
        panel = size_lab.prepare_panel(fix_sentinels=fix)
        bench = size_lab.run_bench(
            [size_lab.Arm(name=name, features=features, fit=gbm)],
            panel=panel, baseline=name, same_rows=False,
        )
        arms[name] = bench.arms[name]
    like = _like_for_like(arms["gbm_zeros"].predictions, arms["gbm_nulls"].predictions)
    return {
        "gain_pp": like["gain_pp"],
        "shared_rows": like["shared_rows"],
        "rows_with_nulls": like["rows_b"],
        "mae_zeros": like["mae_a"], "mae_nulls": like["mae_b"],
        "note": "walk_forward drops NaN rows before fitting, so the GBM saw no extra rows",
    }


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    census = sentinel_census()
    print(f"[EXP-110] daily_market implied_move == 0 on "
          f"{census['daily_zero']:,}/{census['daily_rows']:,} rows "
          f"({census['daily_zero_share']:.1%}); nulls: {census['daily_null']:,}", flush=True)

    features = list(size_model.FEATURES)
    without = [f for f in features if f != "or_implied"]

    results = {}
    # Each arm runs on its OWN usable rows: the row count is the effect here,
    # so forcing a common sample would measure the wrong thing.
    for name, panel, feats in (
        ("incumbent", size_lab.prepare_panel(fix_sentinels=False), features),
        ("sentinel_to_null", size_lab.prepare_panel(fix_sentinels=True), features),
        ("drop_or_implied", size_lab.prepare_panel(fix_sentinels=False), without),
    ):
        bench = size_lab.run_bench(
            [size_lab.Arm(name=name, features=feats)],
            panel=panel, baseline=name, same_rows=False,
        )
        results[name] = bench.arms[name]

    # The confound this experiment was built to catch. sentinel_to_null scores
    # on FEWER rows, and the rows it loses are the hard ones — events where the
    # market quoted nothing realize a bigger move. Comparing aggregate MAE
    # across different samples would credit the fix for deleting difficulty.
    like = _like_for_like(results["incumbent"].predictions,
                          results["sentinel_to_null"].predictions)
    print(f"[EXP-110] like-for-like on the {like['shared_rows']:,} rows both arms scored: "
          f"{like['gain_pp']:+.4f}pp, {like['years_improved']}/{like['years_total']} years, "
          f"p={like['wilcoxon_p']}", flush=True)

    # And the follow-up it suggests: a model that CAN consume a null might use
    # "no quote" as information instead of losing the row. It cannot here —
    # walk_forward drops incomplete rows before any model sees them — so this
    # pair measures the architecture, not the missingness handling, and says so.
    gbm_probe = _gbm_probe()
    print(f"[EXP-110] GBM probe: nulling gains {gbm_probe['gain_pp']:+.4f}pp like-for-like; "
          f"rows kept {gbm_probe['rows_with_nulls']:,} (walk_forward drops NaN regardless)",
          flush=True)

    combined = size_lab.BenchResult(
        rows=results["incumbent"].metrics["n"], arms=results, baseline="incumbent"
    )
    out = {
        "spec_hash": lib.spec_hash(spec),
        "census": census,
        "arms": {
            name: {
                "metrics": res.metrics,
                "by_year": res.by_year.to_dict(orient="records"),
                "features": len(features if name != "drop_or_implied" else without),
            }
            for name, res in results.items()
        },
        "verdicts": {
            name: combined.verdict(name) for name in ("sentinel_to_null", "drop_or_implied")
        },
        "like_for_like": like,
        "gbm_probe": gbm_probe,
    }

    figures = HERE / "figures"
    for name in ("sentinel_to_null", "drop_or_implied"):
        written = size_lab.write_figures(combined, name, figures / name)
        print(f"[EXP-110] {name}: {len(written)} figure(s)", flush=True)

    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "metrics.json").write_text(json.dumps(out, indent=1, default=str))

    for name in ("sentinel_to_null", "drop_or_implied"):
        v = out["verdicts"][name]
        print(f"[EXP-110] {name:18s} improved {v['years_improved']}/{v['years_total']} yrs, "
              f"p={v['wilcoxon_p']}, mean {v['mean_gain_pp']:+.4f}pp, "
              f"excl-best {v['mean_gain_excluding_best_year']:+.4f}pp "
              f"→ {'CONSISTENT' if v['consistent'] else 'not consistent'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
