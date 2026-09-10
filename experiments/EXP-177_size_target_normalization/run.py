#!/usr/bin/env python3
"""EXP-177: fold-fitted target normalization arms."""
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

EXP175_PATH = ROOT / "experiments" / "EXP-175_size_vol_scaled_dislocation" / "run.py"
loader = importlib.util.spec_from_file_location("exp175_helpers", EXP175_PATH)
mod = importlib.util.module_from_spec(loader)
loader.loader.exec_module(mod)

from engine import paths  # noqa: E402

BASE = tuple(mod.BASE)
ARM_TRANSFORMS = {
    "raw_target": "raw",
    "standardized_target": "standardized",
    "robust_target": "robust",
    "log1p_target": "log1p",
}
PRIMARY = "log1p_target"


def log(message: str) -> None:
    print(f"[EXP-177 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def complete_panel() -> pd.DataFrame:
    data = mod.size_model.prepare(mod.load_panel())
    needed = list(BASE) + ["abs_move"]
    mask = np.isfinite(data[needed].to_numpy(dtype=float)).all(axis=1)
    out = data.loc[mask].copy().reset_index(drop=True)
    log(f"incumbent-complete panel: {len(out):,}/{len(data):,} rows")
    return out


def target_transform(y: np.ndarray, name: str) -> tuple[np.ndarray, callable]:
    y = np.asarray(y, dtype=float)
    if name == "raw":
        return y, lambda pred: np.maximum(pred, 0.0)
    if name == "standardized":
        center, scale = float(y.mean()), float(y.std(ddof=0))
        scale = max(scale, 1e-6)
        return (y - center) / scale, lambda pred: np.maximum(pred * scale + center, 0.0)
    if name == "robust":
        center = float(np.median(y))
        q25, q75 = np.quantile(y, [0.25, 0.75])
        scale = max(float(q75 - q25), 1e-6)
        return (y - center) / scale, lambda pred: np.maximum(pred * scale + center, 0.0)
    if name == "log1p":
        return np.log1p(np.maximum(y, 0.0)), lambda pred: np.maximum(np.expm1(pred), 0.0)
    raise ValueError(name)


def fit_oos(data: pd.DataFrame, transform: str, seed: int, arm: str) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR):
        train = data[data["year"] < year]
        test = data[data["year"] == year].copy()
        y_fit, inverse = target_transform(train["abs_move"].to_numpy(dtype=float), transform)
        model = mod.size_model.fit(train[list(BASE)].to_numpy(dtype=float), y_fit, seed=seed)
        test["pred"] = inverse(np.asarray(model.predict(test[list(BASE)].to_numpy(dtype=float)), dtype=float))
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
        bpart, cpart = base[base["year"] == year], cand[cand["year"] == year]
        b = mod.regression_metrics(bpart["abs_move"], bpart["pred"])
        c = mod.regression_metrics(cpart["abs_move"], cpart["pred"])
        rows.append({"year": int(year), "baseline_mae": float(b["mae"]), "challenger_mae": float(c["mae"]), "delta_mae": float(c["mae"] - b["mae"])})
    return rows, np.asarray([r["delta_mae"] for r in rows], dtype=float)


def run_accuracy(data: pd.DataFrame) -> tuple[dict, dict, pd.DataFrame]:
    first_frames = {}
    records = []
    for seed in SEEDS:
        for name, transform in ARM_TRANSFORMS.items():
            frame = fit_oos(data, transform, seed, name)
            if seed == SEEDS[0]:
                first_frames[name] = frame
            records.append({"seed": seed, "arm": name, **mod.regression_metrics(frame["abs_move"], frame["pred"]), **rank_metrics(frame)})
        log(f"seed={seed} complete")
    base = first_frames["raw_target"]
    out = {}
    for name in ("standardized_target", "robust_target", "log1p_target"):
        cand = first_frames[name]
        by_year, deltas = annual_delta(base, cand)
        try:
            pvalue = float(wilcoxon(deltas, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            pvalue = None
        out[name] = {"baseline": {**mod.regression_metrics(base["abs_move"], base["pred"]), **rank_metrics(base)}, "challenger": {**mod.regression_metrics(cand["abs_move"], cand["pred"]), **rank_metrics(cand)}, "years_mae_improved": int((deltas < 0).sum()), "wilcoxon_p": pvalue, "by_year": by_year}
    seed_deltas = {}
    for seed in SEEDS:
        rows = [r for r in records if r["seed"] == seed]
        bmae = next(r["mae"] for r in rows if r["arm"] == "raw_target")
        seed_deltas[str(seed)] = {name: float(next(r["mae"] for r in rows if r["arm"] == name) - bmae) for name in ("standardized_target", "robust_target", "log1p_target")}
    return {"arms": out, "baseline": {**mod.regression_metrics(base["abs_move"], base["pred"]), **rank_metrics(base)}, "rows": int(len(base))}, {"records": records, "mae_delta_vs_baseline": seed_deltas}, first_frames[PRIMARY]


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
    seed_rows = [[seed, *[f"{deltas[name]:+.4f}" for name in ("standardized_target", "robust_target", "log1p_target")]] for seed, deltas in seeds["mae_delta_vs_baseline"].items()]
    return [{"title": "Target-normalization arms", "body": ["All arms use identical incumbent-complete rows. Transforms are fitted inside each training fold and predictions are returned to abs_move percentage-point units."]}, {"title": "Matched OOS accuracy", "columns": ["arm", "MAE", "RMSE", "r", "years improved", "Wilcoxon p"], "align": ["---"] * 6, "rows": rows}, {"title": "Five-seed MAE deltas versus raw target", "columns": ["seed", "standardized", "robust", "log1p"], "align": ["---"] * 4, "rows": seed_rows}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = mod.mod.lib.load_spec(HERE / "spec.yaml")
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
    trades = mod.mod.load_engine_trades_limited("STR-THRU")
    trades = trades[pd.to_datetime(trades["event_date"]).dt.year >= BOOK_FIRST_YEAR].copy()
    gate_state = mod.mod.PrecomputedGate(decisions)
    gate = mod.mod.Gate(fit=gate_state.fit, select=gate_state.select, name="size_target_normalization")
    result = mod.mod.evaluate(spec, trades, gate=gate, run_dir=HERE, input_files=[paths.PANEL, RESULTS / "accuracy.json", RESULTS / "seed_robustness.json", RESULTS / "oos_predictions.parquet"], extra_sections=extra_sections(accuracy, seeds), stress=False, mc_paths=200, mc_block=50)
    if not args.no_ledger:
        mod.mod.lib.record_evaluation(HERE, spec, result.results)
    write_json(RESULTS / "comparison.json", {"accuracy": accuracy, "seed_robustness": seeds, "primary_book": result.results["headline"]})
    log(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
