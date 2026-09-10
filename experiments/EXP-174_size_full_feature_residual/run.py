#!/usr/bin/env python3
"""EXP-174: full-feature causal residual correction."""
from __future__ import annotations

import argparse
import importlib.util
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
STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))

EXP173_PATH = ROOT / "experiments" / "EXP-173_size_surface_residual_distribution" / "run.py"
loader = importlib.util.spec_from_file_location("exp173_helpers", EXP173_PATH)
mod = importlib.util.module_from_spec(loader)
loader.loader.exec_module(mod)

BASE = tuple(mod.BASE)
SURFACE = tuple(mod.SURFACE)
HEAD_FEATURES = (*BASE, *SURFACE, "base_pred")
SEEDS = tuple(mod.SEEDS)
FIRST_TEST_YEAR = mod.FIRST_TEST_YEAR
BOOK_FIRST_YEAR = mod.BOOK_FIRST_YEAR
TOP_FRACTION = mod.TOP_FRACTION
MIN_HEAD_ROWS = mod.MIN_HEAD_ROWS


def log(message: str) -> None:
    print(f"[EXP-174 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def fit_seed(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    pieces = []
    pool_columns = [*BASE, *SURFACE, "base_pred", "residual", "surface_complete"]
    pool = pd.DataFrame(columns=pool_columns)
    years = sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR)
    for year in years:
        train = data[data["year"] < year]
        test = data[data["year"] == year].copy()
        train_ok = np.isfinite(train[[*BASE, "abs_move"]].to_numpy(dtype=float)).all(axis=1)
        train_fit = train.loc[train_ok]
        direct = mod.size_model.fit(
            train_fit[list(BASE)].to_numpy(dtype=float),
            train_fit["abs_move"].to_numpy(dtype=float), seed=seed)
        test["base_pred"] = np.asarray(direct.predict(test[list(BASE)].to_numpy(dtype=float)), dtype=float)
        test["residual"] = test["abs_move"] - test["base_pred"]
        test["corrected_pred"] = test["base_pred"]
        test["residual_head_applied"] = False
        pool_ok = pool.dropna(subset=[*BASE, *SURFACE, "base_pred", "residual"])
        ready = len(pool_ok) >= MIN_HEAD_ROWS and bool(test["surface_complete"].any())
        if ready:
            x_pool = pool_ok[list(HEAD_FEATURES)].to_numpy(dtype=float)
            y_pool = pool_ok["residual"].to_numpy(dtype=float)
            head = mod.head_model(seed).fit(x_pool, y_pool)
            idx = test["surface_complete"].to_numpy(dtype=bool)
            x_test = test.loc[idx, list(HEAD_FEATURES)].to_numpy(dtype=float)
            correction = np.asarray(head.predict(x_test), dtype=float)
            test.loc[idx, "corrected_pred"] = np.maximum(
                test.loc[idx, "base_pred"].to_numpy(dtype=float) + correction, 0.0)
            test.loc[idx, "residual_head_applied"] = True
        pieces.append(test)
        add = test[pool_columns].copy()
        pool = pd.concat([pool, add], ignore_index=True)
        log(f"seed={seed} fold={year} train={len(train_fit):,} test={len(test):,} pool={len(pool):,} head={int(test['residual_head_applied'].sum()):,}")
    return pd.concat(pieces, ignore_index=True)


def rank_metrics(frame: pd.DataFrame, column: str) -> dict:
    x = frame[["abs_move", column]].dropna()
    corr = spearmanr(x[column], x["abs_move"], nan_policy="omit")
    order = np.argsort(x[column].to_numpy(dtype=float))
    n = max(1, len(order) // 10)
    spread = float(x["abs_move"].to_numpy(dtype=float)[order[-n:]].mean() - x["abs_move"].to_numpy(dtype=float)[order[:n]].mean())
    return {"spearman": float(corr.statistic), "top_bottom_decile": spread}


def point_summary(frame: pd.DataFrame, column: str) -> dict:
    return {**mod.regression_metrics(frame["abs_move"], frame[column]), **rank_metrics(frame, column), "surface_coverage": float(frame["surface_complete"].mean()), "head_coverage": float(frame["residual_head_applied"].mean())}


def annual_delta(frame: pd.DataFrame, column: str) -> tuple[list[dict], np.ndarray]:
    rows = []
    for year in sorted(frame["year"].unique()):
        part = frame[frame["year"] == year]
        b = mod.regression_metrics(part["abs_move"], part["base_pred"])
        c = mod.regression_metrics(part["abs_move"], part[column])
        rows.append({"year": int(year), "baseline_mae": float(b["mae"]), "challenger_mae": float(c["mae"]), "delta_mae": float(c["mae"] - b["mae"])})
    return rows, np.asarray([r["delta_mae"] for r in rows], dtype=float)


def run_accuracy(data: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    frames = {}
    records = []
    for seed in SEEDS:
        frame = fit_seed(data, seed)
        frames[seed] = frame
        records.append({"seed": seed, "arm": "baseline", **point_summary(frame, "base_pred")})
        records.append({"seed": seed, "arm": "full_feature_residual", **point_summary(frame, "corrected_pred")})
        log(f"seed={seed} complete; baseline MAE={records[-2]['mae']:.4f} residual MAE={records[-1]['mae']:.4f}")
    first = frames[SEEDS[0]].copy()
    by_year, deltas = annual_delta(first, "corrected_pred")
    try:
        pvalue = float(wilcoxon(deltas, alternative="two-sided", zero_method="wilcox").pvalue)
    except ValueError:
        pvalue = None
    accuracy = {
        "baseline": point_summary(first, "base_pred"),
        "challenger": point_summary(first, "corrected_pred"),
        "years_mae_improved": int((deltas < 0).sum()),
        "wilcoxon_p": pvalue,
        "by_year": by_year,
        "rows": int(len(first)),
        "surface_rows": int(first["surface_complete"].sum()),
    }
    seed_deltas = {}
    for seed, frame in frames.items():
        seed_deltas[str(seed)] = float(point_summary(frame, "corrected_pred")["mae"] - point_summary(frame, "base_pred")["mae"])
    return accuracy, {"records": records, "mae_delta_vs_baseline": seed_deltas}, first


def book_scores(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in frame["year"].unique() if y >= BOOK_FIRST_YEAR):
        test = frame[frame["year"] == year].copy()
        train = frame[frame["year"] < year]
        edge_train = (train["corrected_pred"] - train["or_implied"]).replace([np.inf, -np.inf], np.nan).dropna()
        threshold = float(np.quantile(edge_train, 1.0 - TOP_FRACTION)) if len(edge_train) else 0.0
        test["forecast_edge"] = test["corrected_pred"] - test["or_implied"]
        test["selected"] = test["forecast_edge"] >= threshold
        pieces.append(test[["ticker", "date", "year", "forecast_edge", "selected"]])
    return pd.concat(pieces, ignore_index=True)


def extra_sections(accuracy: dict, seeds: dict) -> list[dict]:
    result = accuracy
    rows = [["baseline", f"{result['baseline']['mae']:.4f}", f"{result['baseline']['rmse']:.4f}", f"{result['baseline']['r']:.4f}", "-"] , ["full_feature_residual", f"{result['challenger']['mae']:.4f}", f"{result['challenger']['rmse']:.4f}", f"{result['challenger']['r']:.4f}", result["years_mae_improved"]]]
    seed_rows = [[seed, f"{delta:+.4f}"] for seed, delta in seeds["mae_delta_vs_baseline"].items()]
    return [
        {"title": "Full-feature residual correction", "body": ["The residual head receives every incumbent feature, the direct prediction, and the normalized surface ratios. It is fit only on prior-year OOS residuals; incomplete surface rows use the direct forecast."]},
        {"title": "Matched OOS accuracy", "columns": ["arm", "MAE", "RMSE", "r", "years improved"], "align": ["---"] * 5, "rows": rows},
        {"title": "Five-seed MAE deltas versus direct model", "columns": ["seed", "delta MAE"], "align": ["---", "---"], "rows": seed_rows},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = mod.lib.load_spec(HERE / "spec.yaml")
    if args.reuse_scores:
        accuracy = json.loads((RESULTS / "accuracy.json").read_text())
        seeds = json.loads((RESULTS / "seed_robustness.json").read_text())
        frame = pd.read_parquet(RESULTS / "oos_predictions.parquet")
        decisions = pd.read_parquet(RESULTS / "primary_book_scores.parquet")
    else:
        data = mod.panel_data()
        accuracy, seeds, frame = run_accuracy(data)
        write_json(RESULTS / "accuracy.json", accuracy)
        write_json(RESULTS / "seed_robustness.json", seeds)
        frame.to_parquet(RESULTS / "oos_predictions.parquet", index=False)
        decisions = book_scores(frame)
        decisions.to_parquet(RESULTS / "primary_book_scores.parquet", index=False)
        log(f"book scores: {len(decisions):,} rows")
    trades = mod.load_engine_trades_limited("STR-THRU")
    trades = trades[pd.to_datetime(trades["event_date"]).dt.year >= BOOK_FIRST_YEAR].copy()
    gate_state = mod.PrecomputedGate(decisions)
    gate = mod.Gate(fit=gate_state.fit, select=gate_state.select, name="size_full_feature_residual")
    result = mod.evaluate(
        spec, trades, gate=gate, run_dir=HERE,
        input_files=[mod.paths.PANEL, RESULTS / "accuracy.json", RESULTS / "seed_robustness.json", RESULTS / "oos_predictions.parquet"],
        extra_sections=extra_sections(accuracy, seeds), stress=False, mc_paths=200, mc_block=50,
    )
    if not args.no_ledger:
        mod.lib.record_evaluation(HERE, spec, result.results)
    write_json(RESULTS / "comparison.json", {"accuracy": accuracy, "seed_robustness": seeds, "primary_book": result.results["headline"]})
    log(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
