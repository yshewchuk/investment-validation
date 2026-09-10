#!/usr/bin/env python3
"""EXP-175: direct volatility-scaled stock-dislocation feature arms."""
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
SEEDS = (20260829, 20260901, 20260903, 20260905, 20260907)
FIRST_TEST_YEAR = 2013
BOOK_FIRST_YEAR = 2018
TOP_FRACTION = 0.20
STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))

EXP172_PATH = ROOT / "experiments" / "EXP-172_size_normalized_recent_move_variants" / "run.py"
loader = importlib.util.spec_from_file_location("exp172_helpers", EXP172_PATH)
mod = importlib.util.module_from_spec(loader)
loader.loader.exec_module(mod)

from engine import paths  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import regression_metrics  # noqa: E402

BASE = tuple(size_model.FEATURES)
ARM_FEATURES = {
    "abs_ret5": ("abs_ret5",),
    "abs_ret10": ("abs_ret10",),
    "abs_ret20": ("abs_ret20",),
    "abs_ret5_over_rvol30": ("abs_ret5_over_rvol30",),
    "abs_ret10_over_rvol30": ("abs_ret10_over_rvol30",),
    "abs_ret20_over_rvol30": ("abs_ret20_over_rvol30",),
}
PRIMARY = "abs_ret10_over_rvol30"


def log(message: str) -> None:
    print(f"[EXP-175 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    for horizon in (5, 10, 20):
        out[f"abs_ret{horizon}"] = pd.to_numeric(out[f"ret{horizon}"], errors="coerce").abs()
        rvol = pd.to_numeric(out["or_rvol30"], errors="coerce")
        out[f"abs_ret{horizon}_over_rvol30"] = out[f"abs_ret{horizon}"] / rvol.where(rvol > 0)
    return out


def complete_panel() -> pd.DataFrame:
    data = add_derived(size_model.prepare(load_panel()))
    needed = list(BASE) + ["abs_move"] + [f for fields in ARM_FEATURES.values() for f in fields]
    ok = np.isfinite(data[needed].to_numpy(dtype=float)).all(axis=1)
    out = data.loc[ok].copy().reset_index(drop=True)
    log(f"union-complete panel: {len(out):,}/{len(data):,} rows")
    return out


def fit_oos(data: pd.DataFrame, features: tuple[str, ...], seed: int, arm: str) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR):
        train = data[data["year"] < year]
        test = data[data["year"] == year].copy()
        model = size_model.fit(train[list(features)].to_numpy(dtype=float), train["abs_move"].to_numpy(dtype=float), seed=seed)
        test["pred"] = np.asarray(model.predict(test[list(features)].to_numpy(dtype=float)), dtype=float)
        pieces.append(test)
        log(f"seed={seed} arm={arm} fold={year} train={len(train):,} test={len(test):,}")
    return pd.concat(pieces, ignore_index=True)


def rank_metrics(frame: pd.DataFrame) -> dict:
    x = frame[["abs_move", "pred"]].dropna()
    corr = spearmanr(x["pred"], x["abs_move"], nan_policy="omit")
    order = np.argsort(x["pred"].to_numpy(dtype=float))
    n = max(1, len(order) // 10)
    y = x["abs_move"].to_numpy(dtype=float)
    return {"spearman": float(corr.statistic), "top_bottom_decile": float(y[order[-n:]].mean() - y[order[:n]].mean())}


def annual_delta(base: pd.DataFrame, cand: pd.DataFrame) -> tuple[list[dict], np.ndarray]:
    rows = []
    for year in sorted(base["year"].unique()):
        b = regression_metrics(base.loc[base["year"] == year, "abs_move"], base.loc[base["year"] == year, "pred"])
        c = regression_metrics(cand.loc[cand["year"] == year, "abs_move"], cand.loc[cand["year"] == year, "pred"])
        rows.append({"year": int(year), "baseline_mae": float(b["mae"]), "challenger_mae": float(c["mae"]), "delta_mae": float(c["mae"] - b["mae"])})
    return rows, np.asarray([r["delta_mae"] for r in rows], dtype=float)


def run_accuracy(data: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    arms = {"baseline": BASE, **{name: BASE + fields for name, fields in ARM_FEATURES.items()}}
    first_frames = {}
    records = []
    for seed in SEEDS:
        for name, features in arms.items():
            frame = fit_oos(data, features, seed, name)
            if seed == SEEDS[0]:
                first_frames[name] = frame
            metrics = {"seed": seed, "arm": name, **regression_metrics(frame["abs_move"], frame["pred"]), **rank_metrics(frame)}
            records.append(metrics)
        log(f"seed={seed} complete")
    base = first_frames["baseline"]
    accuracy_arms = {}
    for name in ARM_FEATURES:
        cand = first_frames[name]
        by_year, deltas = annual_delta(base, cand)
        try:
            pvalue = float(wilcoxon(deltas, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            pvalue = None
        accuracy_arms[name] = {
            "baseline": {**regression_metrics(base["abs_move"], base["pred"]), **rank_metrics(base)},
            "challenger": {**regression_metrics(cand["abs_move"], cand["pred"]), **rank_metrics(cand)},
            "years_mae_improved": int((deltas < 0).sum()),
            "wilcoxon_p": pvalue,
            "by_year": by_year,
        }
    seed_deltas = {}
    for seed in SEEDS:
        seed_rows = [r for r in records if r["seed"] == seed]
        base_mae = next(r["mae"] for r in seed_rows if r["arm"] == "baseline")
        seed_deltas[str(seed)] = {name: float(next(r["mae"] for r in seed_rows if r["arm"] == name) - base_mae) for name in ARM_FEATURES}
    return {"arms": accuracy_arms, "rows": int(len(base))}, {"records": records, "mae_delta_vs_baseline": seed_deltas}, first_frames[PRIMARY]


def book_scores(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in frame["year"].unique() if y >= BOOK_FIRST_YEAR):
        test = frame[frame["year"] == year].copy()
        train = frame[frame["year"] < year]
        edge = (train["pred"] - train["or_implied"]).replace([np.inf, -np.inf], np.nan).dropna()
        threshold = float(np.quantile(edge, 1.0 - TOP_FRACTION)) if len(edge) else 0.0
        test["forecast_edge"] = test["pred"] - test["or_implied"]
        test["selected"] = test["forecast_edge"] >= threshold
        pieces.append(test[["ticker", "date", "year", "forecast_edge", "selected"]])
    return pd.concat(pieces, ignore_index=True)


def extra_sections(accuracy: dict, seeds: dict) -> list[dict]:
    rows = []
    for name, result in accuracy["arms"].items():
        rows.append([name, f"{result['challenger']['mae']:.4f}", f"{result['challenger']['rmse']:.4f}", f"{result['challenger']['r']:.4f}", result["years_mae_improved"], f"{result['wilcoxon_p']:.4f}" if result["wilcoxon_p"] is not None else "NA"])
    seed_rows = []
    for seed, deltas in seeds["mae_delta_vs_baseline"].items():
        seed_rows.append([seed] + [f"{deltas[name]:+.4f}" for name in ARM_FEATURES])
    return [
        {"title": "Volatility-scaled dislocation arms", "body": ["All arms use identical union-complete rows. The primary is abs_ret10_over_rvol30; secondary raw and alternate-horizon arms are diagnostics only."]},
        {"title": "Matched OOS accuracy", "columns": ["arm", "MAE", "RMSE", "r", "years improved", "Wilcoxon p"], "align": ["---"] * 6, "rows": rows},
        {"title": "Five-seed MAE deltas versus baseline", "columns": ["seed", *ARM_FEATURES.keys()], "align": ["---"] * 7, "rows": seed_rows},
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
        primary_frame = pd.read_parquet(RESULTS / "oos_predictions.parquet")
        decisions = pd.read_parquet(RESULTS / "primary_book_scores.parquet")
    else:
        data = complete_panel()
        accuracy, seeds, primary_frame = run_accuracy(data)
        write_json(RESULTS / "accuracy.json", accuracy)
        write_json(RESULTS / "seed_robustness.json", seeds)
        primary_frame.to_parquet(RESULTS / "oos_predictions.parquet", index=False)
        decisions = book_scores(primary_frame)
        decisions.to_parquet(RESULTS / "primary_book_scores.parquet", index=False)
        log(f"book scores: {len(decisions):,} rows")
    trades = mod.load_engine_trades_limited("STR-THRU")
    trades = trades[pd.to_datetime(trades["event_date"]).dt.year >= BOOK_FIRST_YEAR].copy()
    gate_state = mod.PrecomputedGate(decisions)
    gate = mod.Gate(fit=gate_state.fit, select=gate_state.select, name="size_vol_scaled_dislocation")
    result = mod.evaluate(
        spec, trades, gate=gate, run_dir=HERE,
        input_files=[paths.PANEL, RESULTS / "accuracy.json", RESULTS / "seed_robustness.json", RESULTS / "oos_predictions.parquet"],
        extra_sections=extra_sections(accuracy, seeds), stress=False, mc_paths=200, mc_block=50,
    )
    if not args.no_ledger:
        mod.lib.record_evaluation(HERE, spec, result.results)
    write_json(RESULTS / "comparison.json", {"accuracy": accuracy, "seed_robustness": seeds, "primary_book": result.results["headline"]})
    log(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
