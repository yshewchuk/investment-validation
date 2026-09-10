#!/usr/bin/env python3
"""EXP-171: test recent absolute-move levels in the size model.

Run with ``python3 experiments/EXP-171_size_recent_absolute_ema/run.py``.
The primary challenger adds the existing causal ``ema2_prior_abs_move`` and
``ema4_prior_abs_move`` columns to the unchanged OLS+MLP blend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = (20260829, 20260901, 20260903, 20260905, 20260907)
FIRST_TEST_YEAR = 2013
BOOK_FIRST_YEAR = 2018
TOP_FRACTION = 0.20
ADDED = ("ema2_prior_abs_move", "ema4_prior_abs_move")
STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import regression_metrics  # noqa: E402
from experiments import common, lib  # noqa: E402


def log(message: str) -> None:
    print(f"[EXP-171 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def complete_panel() -> pd.DataFrame:
    panel = size_model.prepare(load_panel())
    features = tuple(size_model.FEATURES) + ADDED
    values = panel[list(features) + [size_model.TARGET]].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    out = panel.loc[mask].copy().reset_index(drop=True)
    log(f"matched complete case: {len(out):,}/{len(panel):,} panel rows")
    return out


def fit_oos(data: pd.DataFrame, features: tuple[str, ...], seed: int) -> pd.DataFrame:
    pieces = []
    years = sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR)
    for year in years:
        train = data[data["year"] < year]
        test = data[data["year"] == year].copy()
        if len(train) < 500 or test.empty:
            continue
        model = size_model.fit(
            train[list(features)].to_numpy(dtype=float),
            train[size_model.TARGET].to_numpy(dtype=float),
            seed=seed,
        )
        test["pred"] = model.predict(test[list(features)].to_numpy(dtype=float))
        pieces.append(test)
        log(f"seed={seed} {len(features)} features fold={year} train={len(train):,} test={len(test):,}")
    return pd.concat(pieces, ignore_index=True)


def rank_metrics(frame: pd.DataFrame) -> dict:
    ranked = frame.dropna(subset=["pred", "abs_move"]).copy()
    corr = spearmanr(ranked["pred"], ranked["abs_move"], nan_policy="omit")
    ranked["decile"] = pd.qcut(ranked["pred"], 10, labels=False, duplicates="drop")
    means = ranked.groupby("decile", observed=True)["abs_move"].mean()
    return {
        "n": int(len(ranked)),
        "spearman": float(corr.statistic),
        "top_bottom_decile": float(means.iloc[-1] - means.iloc[0]),
    }


def annual_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in frame.groupby("year", sort=True):
        rows.append({"year": int(year), **regression_metrics(group["abs_move"], group["pred"])})
    return rows


def seed_summary(data: pd.DataFrame) -> tuple[dict, dict[int, dict[str, pd.DataFrame]]]:
    baseline = tuple(size_model.FEATURES)
    challenger = baseline + ADDED
    records = []
    frames = {}
    for seed in SEEDS:
        log(f"fitting matched baseline and challenger for seed={seed}")
        base_frame = fit_oos(data, baseline, seed)
        chall_frame = fit_oos(data, challenger, seed)
        if not base_frame[["ticker", "date"]].equals(chall_frame[["ticker", "date"]]):
            raise RuntimeError("baseline and challenger OOS rows differ")
        frames[seed] = {"baseline": base_frame, "challenger": chall_frame}
        for arm, frame in frames[seed].items():
            records.append({"seed": seed, "arm": arm, **regression_metrics(frame["abs_move"], frame["pred"]), **rank_metrics(frame)})
    summary = pd.DataFrame(records)
    pivot = summary.pivot(index="seed", columns="arm", values="mae")
    return {
        "seeds": list(SEEDS),
        "arms": summary.to_dict("records"),
        "mae_delta_challenger_minus_baseline": {
            str(seed): float(pivot.loc[seed, "challenger"] - pivot.loc[seed, "baseline"])
            for seed in pivot.index
        },
        "median_mae_delta": float(np.median(pivot["challenger"] - pivot["baseline"])),
    }, frames


def primary_accuracy(frames: dict[int, dict[str, pd.DataFrame]]) -> dict:
    base = frames[SEEDS[0]]["baseline"]
    chall = frames[SEEDS[0]]["challenger"]
    base_year = pd.DataFrame(annual_metrics(base)).set_index("year")
    chall_year = pd.DataFrame(annual_metrics(chall)).set_index("year")
    rows = []
    for year in base_year.index:
        rows.append({
            "year": int(year),
            "baseline_mae": float(base_year.loc[year, "mae"]),
            "challenger_mae": float(chall_year.loc[year, "mae"]),
            "delta_mae": float(chall_year.loc[year, "mae"] - base_year.loc[year, "mae"]),
            "baseline_r": float(base_year.loc[year, "r"]),
            "challenger_r": float(chall_year.loc[year, "r"]),
        })
    deltas = np.asarray([row["delta_mae"] for row in rows], dtype=float)
    stat = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
    return {
        "baseline": {**regression_metrics(base["abs_move"], base["pred"]), **rank_metrics(base)},
        "challenger": {**regression_metrics(chall["abs_move"], chall["pred"]), **rank_metrics(chall)},
        "by_year": rows,
        "years_mae_improved": int((deltas < 0).sum()),
        "wilcoxon_two_sided": {"statistic": float(stat.statistic), "pvalue": float(stat.pvalue)},
    }


def book_scores(frame: pd.DataFrame, features: tuple[str, ...], seed: int) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in frame["year"].unique() if y >= BOOK_FIRST_YEAR):
        train = frame[frame["year"] < year]
        test = frame[frame["year"] == year].copy()
        model = size_model.fit(train[list(features)].to_numpy(dtype=float), train["abs_move"].to_numpy(dtype=float), seed=seed)
        train_edge = model.predict(train[list(features)].to_numpy(dtype=float)) - train["or_implied"].to_numpy(dtype=float)
        test["forecast_edge"] = model.predict(test[list(features)].to_numpy(dtype=float)) - test["or_implied"].to_numpy(dtype=float)
        test["selected"] = test["forecast_edge"] >= float(np.quantile(train_edge, 1.0 - TOP_FRACTION))
        pieces.append(test[["ticker", "date", "year", "forecast_edge", "selected"]])
    return pd.concat(pieces, ignore_index=True)


class PrecomputedGate:
    def __init__(self, decisions: pd.DataFrame):
        indexed = decisions.drop_duplicates(["ticker", "date"]).copy()
        indexed["key"] = indexed["ticker"].astype(str) + "|" + indexed["date"].astype(str)
        self.selected = indexed.set_index("key")["selected"].astype(bool).to_dict()

    def fit(self, train: pd.DataFrame) -> None:
        return None

    def select(self, rows: pd.DataFrame) -> pd.Series:
        keys = rows["ticker"].astype(str) + "|" + pd.to_datetime(rows["event_date"]).dt.strftime("%Y-%m-%d")
        return keys.map(self.selected).fillna(False).astype(bool)


def extra_accuracy(accuracy: dict, seeds: dict) -> list[dict]:
    base = accuracy["baseline"]
    chall = accuracy["challenger"]
    rows = [[r["year"], f"{r['baseline_mae']:.4f}", f"{r['challenger_mae']:.4f}", f"{r['delta_mae']:+.4f}"] for r in accuracy["by_year"]]
    seed_rows = [[r["seed"], r["arm"], f"{r['mae']:.4f}", f"{r['rmse']:.4f}", f"{r['r']:.4f}"] for r in seeds["arms"]]
    return [
        {"title": "Primary size-model accuracy", "body": [
            f"Matched OOS MAE: baseline {base['mae']:.4f} pp versus challenger {chall['mae']:.4f} pp; RMSE {base['rmse']:.4f} versus {chall['rmse']:.4f}.",
            f"The challenger improved MAE in {accuracy['years_mae_improved']}/{len(accuracy['by_year'])} years; paired Wilcoxon p={accuracy['wilcoxon_two_sided']['pvalue']:.4f}.",
            "The primary arm uses identical rows and the unchanged OLS+MLP architecture.",
        ]},
        {"title": "Annual matched accuracy", "columns": ["year", "baseline MAE", "challenger MAE", "delta"], "align": ["---:"] * 4, "rows": rows},
        {"title": "Fixed-seed robustness", "columns": ["seed", "arm", "MAE", "RMSE", "r"], "align": ["---:"] * 5, "rows": seed_rows},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--reuse-scores", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    base_features = tuple(size_model.FEATURES)
    challenger_features = base_features + ADDED
    if args.reuse_scores:
        accuracy = json.loads((RESULTS / "accuracy.json").read_text())
        seeds = json.loads((RESULTS / "seed_robustness.json").read_text())
        decisions = {
            "baseline": pd.read_parquet(RESULTS / "baseline_book_scores.parquet"),
            "challenger": pd.read_parquet(RESULTS / "challenger_book_scores.parquet"),
        }
        log("reusing completed accuracy and book-score artifacts")
    else:
        data = complete_panel()
        seeds, frames = seed_summary(data)
        accuracy = primary_accuracy(frames)
        write_json(RESULTS / "accuracy.json", accuracy)
        write_json(RESULTS / "seed_robustness.json", seeds)
        decisions = {
            "baseline": book_scores(data, base_features, SEEDS[0]),
            "challenger": book_scores(data, challenger_features, SEEDS[0]),
        }
        for arm, decision in decisions.items():
            decision.to_parquet(RESULTS / f"{arm}_book_scores.parquet", index=False)
            log(f"{arm} book scores: {len(decision):,} rows, selected={int(decision['selected'].sum()):,}")
    trades = common.load_engine_trades("STR-THRU")
    trades = trades[pd.to_datetime(trades["event_date"]).dt.year >= BOOK_FIRST_YEAR].copy()
    spy = common.load_spy_daily()
    inputs = [paths.PANEL, RESULTS / "accuracy.json", RESULTS / "seed_robustness.json"]
    results = {}
    for arm in ("baseline", "challenger"):
        run_dir = HERE if arm == "challenger" else HERE / "arms" / "baseline"
        this_spec = dict(spec)
        this_spec["primary_spec"] = dict(spec["primary_spec"])
        if arm == "baseline":
            this_spec["grid_cell"] = True
        gate = Gate(fit=PrecomputedGate(decisions[arm]).fit, select=PrecomputedGate(decisions[arm]).select, name=f"size_{arm}_edge")
        log(f"evaluating real-price companion book: {arm}")
        result = evaluate(
            this_spec, trades, gate=gate, run_dir=run_dir,
            repricer=common.make_repricer("STR-THRU"), spy_daily=spy,
            input_files=inputs,
            extra_sections=extra_accuracy(accuracy, seeds) if arm == "challenger" else [{"title": "Control book", "body": ["Baseline size-model forecast-edge top-20 selector."]}],
        )
        results[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, this_spec, result.results)
        log(f"{arm} book complete: n={result.results['headline']['n']:,} mean={result.results['headline']['mean']:+.4f}")
    write_json(RESULTS / "comparison.json", {
        "accuracy": accuracy,
        "seed_robustness": seeds,
        "baseline_book": results["baseline"].results["headline"],
        "challenger_book": results["challenger"].results["headline"],
    })
    log(f"report: {results['challenger'].report_path}")


if __name__ == "__main__":
    main()
