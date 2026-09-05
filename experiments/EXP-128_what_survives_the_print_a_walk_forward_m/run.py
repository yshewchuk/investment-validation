#!/usr/bin/env python3
"""EXP-128 — what survives the print: a walk-forward model for the earnings IV crush.

    python3 experiments/EXP-128_what_survives_the_print_a_walk_forward_m/run.py

Five arms, one of which is the point of the experiment and is not the primary.

  primary   HistGBM on the panel row, target crush_pct_iv30
  iv10      the same model on crush_pct_iv10 — the horizon a first-post-event
            expiry actually lives at, and the noisier one
  no_exern  or_exern30 dropped from the features. THE ABLATION. ORATS already
            publishes an ex-earnings vol at the same close, and it scores
            r 0.670 against the realized crush for free. If the primary's
            advantage disappears without it, what has been built is a
            recalibration of a vendor column — worth shipping, but as three
            lines rather than as a champion.
  level     predict post-print iv30 directly instead of the ratio
  linear    OLS on the same features — is the GBM earning its keep?

Every arm is scored on IDENTICAL rows. That is EXP-116's correction, which
found a GBM winning by 0.11pp because it had been scored on the easier subset
its own feature requirements selected.

Nothing here promotes anything. A champion needs `train_all` to register it and
`promote.decide` to clear it; this experiment decides whether there is a model
worth registering at all.
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

from engine.features import load_panel  # noqa: E402
from engine.models.training.common import (  # noqa: E402
    SEED,
    decile_spread,
    regression_metrics,
    walk_forward,
)
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crush import baselines, build_crush_frame  # noqa: E402

FIRST_TEST_YEAR = 2013

#: The panel's own columns, minus the outcome and the quarantined one. The
#: structural anchor `or_exern30` is IN — beating it should mean adding to it,
#: not rediscovering it.
def feature_columns(panel: pd.DataFrame) -> list[str]:
    from engine.features import OUTCOME_COLUMNS, QUARANTINED_FEATURES, LIVE_UNAVAILABLE

    from engine.data.features.tier4 import COLUMNS as TIER4_COLUMNS

    drop = set(OUTCOME_COLUMNS) | set(QUARANTINED_FEATURES) | set(LIVE_UNAVAILABLE) | {
        "ticker", "date", "k", "quarter", "year", "mcap_asof", "mcap_usd",
    }
    # This experiment's target lives on both sides of the print, so the frame it
    # is built on carries post-print columns. None of them may be a feature, and
    # neither may a Tier-4 forecast — `dataset` loads the panel without them,
    # but a model whose feature list is DERIVED rather than declared has to
    # exclude by construction, not by trusting its caller.
    drop |= set(TIER4_COLUMNS)
    drop |= {c for c in panel.columns if c.startswith(("post_", "crush_", "pred_"))}
    drop |= {"gap_days", "pre_date", "post_date", "event_date"}
    numeric = panel.select_dtypes(include="number").columns
    return [c for c in numeric if c not in drop]


def gbm(X, y, seed=SEED):
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=300, random_state=seed
    ).fit(X, y)


def ols(X, y, seed=SEED):  # noqa: ARG001 - the signature the harness calls
    from sklearn.linear_model import LinearRegression

    return LinearRegression().fit(X, y)


def dataset(log=print) -> pd.DataFrame:
    """The panel joined to the realized crush, one row per event."""
    panel = load_panel()
    crush = build_crush_frame(log=log)
    joined = panel.merge(
        crush.drop(columns=["year"]),
        left_on=["ticker", "date"], right_on=["ticker", "event_date"], how="inner",
    )
    log(f"[EXP-128] {len(panel):,} panel rows x {len(crush):,} crush rows "
        f"-> {len(joined):,} joined")
    joined["year"] = pd.to_datetime(joined["date"]).dt.year
    return joined[joined["year"] >= FIRST_TEST_YEAR].reset_index(drop=True)


def interval_coverage(residuals: np.ndarray, held_out: np.ndarray) -> float | None:
    """Share of ``held_out`` errors inside the 10th-90th percentile of ``residuals``."""
    pool = residuals[np.isfinite(residuals)]
    sample = held_out[np.isfinite(held_out)]
    if len(pool) < 250 or len(sample) < 250:
        return None
    lo, hi = np.quantile(pool, [0.10, 0.90])
    return float(((sample >= lo) & (sample <= hi)).mean())


def score_baselines(frame: pd.DataFrame, target: str) -> dict:
    out = {}
    for name, pred in baselines(frame, target).items():
        block = regression_metrics(frame[target], pred)
        block["decile_spread"] = decile_spread(frame[target], pred)
        out[name] = block
    return out


def run_arm(name: str, frame: pd.DataFrame, features: list[str], target: str,
            fit, log=print) -> dict:
    started = time.time()
    result = walk_forward(
        frame, features, target, fit,
        year_column="year", first_test_year=FIRST_TEST_YEAR, seed=SEED,
    )
    scored = result.frame
    metrics = dict(result.metrics)
    metrics["decile_spread"] = decile_spread(scored[target], scored["pred"])

    # The band's calibration, on the same terms Tier 4 would build it: an
    # earlier fold's held-out errors, checked against a later fold's outcomes.
    years = sorted(scored["year"].unique())
    split = years[len(years) // 2]
    early = scored[scored["year"] < split]
    late = scored[scored["year"] >= split]
    metrics["interval_coverage"] = interval_coverage(
        (early[target] - early["pred"]).to_numpy(dtype=float),
        (late[target] - late["pred"]).to_numpy(dtype=float),
    )

    base = score_baselines(scored, target)
    best_mae = min(b["mae"] for b in base.values() if b["mae"] is not None)
    best_rmse = min(b["rmse"] for b in base.values() if b["rmse"] is not None)

    by_year = []
    for year, group in scored.groupby("year"):
        row = {"year": int(year), "n": int(len(group))}
        row["mae"] = float((group["pred"] - group[target]).abs().mean())
        row["baseline_mae"] = min(
            float(np.nanmean(np.abs(pred - group[target].to_numpy(dtype=float))))
            for pred in baselines(group, target).values()
        )
        row["beats_baseline"] = bool(row["mae"] < row["baseline_mae"])
        by_year.append(row)

    won = sum(r["beats_baseline"] for r in by_year)
    log(f"[EXP-128] {name}: n={metrics['n']:,} MAE {metrics['mae']:.3f} "
        f"(best baseline {best_mae:.3f}) RMSE {metrics['rmse']:.3f} "
        f"(best {best_rmse:.3f}) {won}/{len(by_year)} years "
        f"[{time.time() - started:.0f}s]")
    return {
        "metrics": metrics,
        "baselines": base,
        "best_baseline_mae": best_mae,
        "best_baseline_rmse": best_rmse,
        "by_year": by_year,
        "years_beating_baseline": f"{won}/{len(by_year)}",
        "elapsed_s": round(time.time() - started, 1),
    }


def main() -> int:
    spec = lib.load_spec(HERE / "spec.yaml")
    frame = dataset()
    features = feature_columns(frame)
    print(f"[EXP-128] {len(features)} features, {len(frame):,} rows "
          f"{frame['year'].min()}-{frame['year'].max()}", flush=True)

    arms = {
        "primary": (features, "crush_pct_iv30", gbm),
        "iv10": (features, "crush_pct_iv10", gbm),
        "no_exern": ([f for f in features if f != "or_exern30"], "crush_pct_iv30", gbm),
        "level": (features, "post_iv30", gbm),
        "linear": (features, "crush_pct_iv30", ols),
    }
    results = {"spec_hash": lib.spec_hash(spec), "rows": int(len(frame)), "arms": {}}
    for name, (cols, target, fit) in arms.items():
        results["arms"][name] = run_arm(name, frame, cols, target, fit)

    acceptance = spec["primary_spec"]["acceptance"]
    primary = results["arms"]["primary"]
    won, total = primary["years_beating_baseline"].split("/")
    results["verdict"] = {
        "beats_mae": bool(primary["metrics"]["mae"] < primary["best_baseline_mae"]),
        "beats_rmse": bool(primary["metrics"]["rmse"] < primary["best_baseline_rmse"]),
        "consistent": bool(int(won) / max(int(total), 1) >= 0.70),
        "not_shrunken": bool((primary["metrics"]["decile_spread"] or 0) > 20.0),
        "calibrated": (
            primary["metrics"]["interval_coverage"] is not None
            and 0.78 <= primary["metrics"]["interval_coverage"] <= 0.82
        ),
        "coverage_floor": bool(primary["metrics"]["n"] >= 80_000),
        "acceptance_as_registered": acceptance,
    }
    results["verdict"]["all_clear"] = all(
        v for k, v in results["verdict"].items() if isinstance(v, bool)
    )
    # The ablation is the finding, whichever way it lands.
    results["exern_is_the_signal"] = {
        "primary_mae": primary["metrics"]["mae"],
        "no_exern_mae": results["arms"]["no_exern"]["metrics"]["mae"],
        "structural_baseline_mae": primary["baselines"]["structural"]["mae"],
    }

    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=1, default=str))
    print(json.dumps(results["verdict"], indent=1, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
