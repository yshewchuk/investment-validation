#!/usr/bin/env python3
"""EXP-173: causal surface-conditioned residual mean and scale heads."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SEEDS = (20260829, 20260901, 20260903, 20260905, 20260907)
FIRST_TEST_YEAR = 2013
BOOK_FIRST_YEAR = 2018
TOP_FRACTION = 0.20
MIN_HEAD_ROWS = 500
SIGMA_FLOOR = 0.50
STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.data import store  # noqa: E402
from engine.evaluate import Gate, evaluate  # noqa: E402
from engine.features import load_panel  # noqa: E402
from engine.models.training import size_model  # noqa: E402
from engine.models.training.common import regression_metrics  # noqa: E402
from experiments import common, lib  # noqa: E402

BASE = tuple(size_model.FEATURES)
SURFACE = (
    "surf_pre_iv10_over_iv30",
    "surf_pre_exern10_over_exern30",
    "surf_exern30_over_iv30",
    "surf_iv30_over_rvol30",
    "surf_implied_over_history",
    "surf_pre_iv10_over_exern10",
)


def log(message: str) -> None:
    print(f"[EXP-173 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def log_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    num = pd.to_numeric(frame[numerator], errors="coerce")
    den = pd.to_numeric(frame[denominator], errors="coerce")
    ratio = num / den
    return np.log(ratio.where((num > 0) & (den > 0)))


def add_surface(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["surf_pre_iv10_over_iv30"] = log_ratio(out, "pre_iv10", "pre_iv30")
    out["surf_pre_exern10_over_exern30"] = log_ratio(out, "pre_exern_iv10", "pre_exern_iv30")
    out["surf_exern30_over_iv30"] = log_ratio(out, "or_exern30", "or_iv30")
    out["surf_iv30_over_rvol30"] = log_ratio(out, "or_iv30", "or_rvol30")
    out["surf_implied_over_history"] = log_ratio(out, "or_implied", "mean_prior_abs_move")
    out["surf_pre_iv10_over_exern10"] = log_ratio(out, "pre_iv10", "pre_exern_iv10")
    out["surface_complete"] = np.isfinite(out[list(SURFACE)].to_numpy(dtype=float)).all(axis=1)
    return out


def panel_data() -> pd.DataFrame:
    data = add_surface(size_model.prepare(load_panel()))
    needed = list(BASE) + ["abs_move"]
    values = data[needed].to_numpy(dtype=float)
    base_ok = np.isfinite(values).all(axis=1)
    out = data.loc[base_ok].copy().reset_index(drop=True)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype(int)
    log(f"incumbent-complete panel: {len(out):,}/{len(data):,}; surface-complete {int(out['surface_complete'].sum()):,}")
    return out


def head_model(seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=2.0, random_state=seed,
    )


def head_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[*SURFACE, "base_pred"]].to_numpy(dtype=float)


def baseline_interval(pool: pd.DataFrame, preds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(pool) < 20:
        return preds - 4.0, preds + 4.0
    pool_pred = pool["base_pred"].to_numpy(dtype=float)
    pool_resid = pool["residual"].to_numpy(dtype=float)
    edges = np.quantile(pool_pred, np.linspace(0, 1, 11))
    out_lo = np.empty(len(preds)); out_hi = np.empty(len(preds))
    for i, pred in enumerate(preds):
        bucket = int(np.searchsorted(edges[1:-1], pred, side="right"))
        mask = (pool_pred >= edges[bucket]) & (pool_pred <= edges[bucket + 1])
        vals = pool_resid[mask]
        if len(vals) < 20:
            vals = pool_resid
        out_lo[i] = pred + np.quantile(vals, 0.10)
        out_hi[i] = pred + np.quantile(vals, 0.90)
    return np.maximum(out_lo, 0.0), np.maximum(out_hi, 0.0)


def interval_stats(y: pd.Series, lo: pd.Series, hi: pd.Series) -> dict:
    ok = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    yy, ll, hh = y.to_numpy(dtype=float)[ok], lo.to_numpy(dtype=float)[ok], hi.to_numpy(dtype=float)[ok]
    if not len(yy):
        return {"n": 0, "coverage80": None, "width": None, "below": None, "above": None}
    return {
        "n": int(len(yy)),
        "coverage80": float(((yy >= ll) & (yy <= hh)).mean()),
        "width": float((hh - ll).mean()),
        "below": float((yy < ll).mean()),
        "above": float((yy > hh).mean()),
    }


def fit_seed(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    pieces = []
    pool = pd.DataFrame()
    years = sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR)
    for year in years:
        train = data[data["year"] < year]
        test = data[data["year"] == year].copy()
        train_ok = np.isfinite(train[list(BASE) + ["abs_move"]].to_numpy(dtype=float)).all(axis=1)
        train_fit = train.loc[train_ok]
        model = size_model.fit(train_fit[list(BASE)].to_numpy(dtype=float), train_fit["abs_move"].to_numpy(dtype=float), seed=seed)
        test["base_pred"] = np.asarray(model.predict(test[list(BASE)].to_numpy(dtype=float)), dtype=float)
        test["residual"] = test["abs_move"] - test["base_pred"]
        base_lo, base_hi = baseline_interval(pool, test["base_pred"].to_numpy(dtype=float))
        test["base_p10"], test["base_p90"] = base_lo, base_hi
        test["surface_mean_applied"] = False
        test["surface_scale_applied"] = False
        test["surface_mean_pred"] = test["base_pred"]
        test["surface_scale_pred"] = np.nan
        valid_pool = pool[pool["surface_complete"]].copy() if len(pool) else pool
        head_ready = len(valid_pool) >= MIN_HEAD_ROWS and valid_pool[list(SURFACE) + ["base_pred", "residual"]].notna().all(axis=1).sum() >= MIN_HEAD_ROWS
        if head_ready and test["surface_complete"].any():
            valid_pool = valid_pool.dropna(subset=[*SURFACE, "base_pred", "residual"])
            Xpool = head_matrix(valid_pool)
            Xtest = head_matrix(test.loc[test["surface_complete"]])
            mean_head = head_model(seed).fit(Xpool, valid_pool["residual"].to_numpy(dtype=float))
            scale_head = head_model(seed + 17).fit(Xpool, np.log1p(np.abs(valid_pool["residual"].to_numpy(dtype=float))))
            idx = test["surface_complete"].to_numpy(dtype=bool)
            correction = mean_head.predict(Xtest)
            sigma = np.maximum(np.expm1(scale_head.predict(Xtest)), SIGMA_FLOOR)
            test.loc[idx, "surface_mean_pred"] = np.maximum(test.loc[idx, "base_pred"].to_numpy(dtype=float) + correction, 0.0)
            test.loc[idx, "surface_scale_pred"] = sigma
            test.loc[idx, "surface_mean_applied"] = True
            test.loc[idx, "surface_scale_applied"] = True
            std_pool = valid_pool["residual"].to_numpy(dtype=float) / np.maximum(np.expm1(scale_head.predict(Xpool)), SIGMA_FLOOR)
            q10, q90 = np.quantile(std_pool, [0.10, 0.90])
            surf_lo = test.loc[idx, "surface_mean_pred"].to_numpy(dtype=float) + q10 * sigma
            surf_hi = test.loc[idx, "surface_mean_pred"].to_numpy(dtype=float) + q90 * sigma
            test.loc[idx, "surface_p10"] = np.maximum(surf_lo, 0.0)
            test.loc[idx, "surface_p90"] = np.maximum(surf_hi, 0.0)
        else:
            test["surface_p10"] = test["base_p10"]
            test["surface_p90"] = test["base_p90"]
        test["surface_p10"] = test["surface_p10"].fillna(test["base_p10"])
        test["surface_p90"] = test["surface_p90"].fillna(test["base_p90"])
        pieces.append(test)
        add = test[["base_pred", "residual", "surface_complete", *SURFACE]].copy()
        if "surface_scale_pred" in test:
            add["scale_sigma_oof"] = test["surface_scale_pred"]
        pool = pd.concat([pool, add], ignore_index=True)
        log(f"seed={seed} fold={year} train={len(train_fit):,} test={len(test):,} pool={len(pool):,} surface={int(test['surface_mean_applied'].sum()):,}")
    return pd.concat(pieces, ignore_index=True)


def point_summary(frame: pd.DataFrame, column: str) -> dict:
    return {**regression_metrics(frame["abs_move"], frame[column]), "surface_coverage": float(frame["surface_complete"].mean())}


def annual_delta(base: pd.DataFrame, cand: pd.DataFrame, column: str) -> tuple[list[dict], np.ndarray]:
    rows = []
    for year in sorted(base["year"].unique()):
        b = base[base["year"] == year]
        c = cand[cand["year"] == year]
        delta = float(regression_metrics(c["abs_move"], c[column])["mae"] - regression_metrics(b["abs_move"], b["base_pred"])["mae"])
        rows.append({"year": int(year), "baseline_mae": float(regression_metrics(b["abs_move"], b["base_pred"])["mae"]), "challenger_mae": float(regression_metrics(c["abs_move"], c[column])["mae"]), "delta_mae": delta})
    return rows, np.asarray([r["delta_mae"] for r in rows], dtype=float)


def run_accuracy(data: pd.DataFrame) -> tuple[dict, dict, dict]:
    frames = {}
    seed_records = []
    for seed in SEEDS:
        frame = fit_seed(data, seed)
        frames[seed] = frame
        for arm, col in (("baseline", "base_pred"), ("surface_mean", "surface_mean_pred"), ("surface_mean_plus_scale", "surface_mean_pred")):
            seed_records.append({"seed": seed, "arm": arm, **point_summary(frame, col)})
        log(f"seed={seed} complete; baseline MAE={seed_records[-3]['mae']:.4f} combined MAE={seed_records[-1]['mae']:.4f}")
    first = frames[SEEDS[0]].copy()
    arms = {}
    base_summary = point_summary(first, "base_pred")
    for arm, col in (("surface_mean", "surface_mean_pred"), ("surface_mean_plus_scale", "surface_mean_pred")):
        rows, deltas = annual_delta(first, first, col)
        try:
            pvalue = float(wilcoxon(deltas, alternative="two-sided", zero_method="wilcox").pvalue)
        except ValueError:
            pvalue = None
        arms[arm] = {"baseline": base_summary, "challenger": point_summary(first, col), "years_mae_improved": int((deltas < 0).sum()), "wilcoxon_p": pvalue, "by_year": rows}
    arms["surface_scale"] = {"baseline_interval": interval_stats(first["abs_move"], first["base_p10"], first["base_p90"]), "challenger_interval": interval_stats(first["abs_move"], first["surface_p10"], first["surface_p90"]), "fallback_rate": float((~first["surface_scale_applied"]).mean())}
    intervals = {"baseline": interval_stats(first["abs_move"], first["base_p10"], first["base_p90"]), "combined": interval_stats(first["abs_move"], first["surface_p10"], first["surface_p90"])}
    seed_deltas = {str(seed): float(point_summary(frames[seed], "surface_mean_pred")["mae"] - point_summary(frames[seed], "base_pred")["mae"]) for seed in SEEDS}
    return {"arms": arms, "intervals": intervals, "rows": int(len(first)), "surface_rows": int(first["surface_complete"].sum()), "oos": first}, {"records": seed_records, "mae_delta_vs_baseline": seed_deltas}, frames


def book_scores(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in frame["year"].unique() if y >= BOOK_FIRST_YEAR):
        test = frame[frame["year"] == year].copy()
        train = frame[frame["year"] < year]
        edge_train = train["surface_mean_pred"] - train["or_implied"]
        threshold = float(np.quantile(edge_train.replace([np.inf, -np.inf], np.nan).dropna(), 1.0 - TOP_FRACTION)) if len(edge_train.dropna()) else 0.0
        test["forecast_edge"] = test["surface_mean_pred"] - test["or_implied"]
        test["selected"] = test["forecast_edge"] >= threshold
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


def load_engine_trades_limited(strategy: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
    columns = ["strategy", "provenance", "ticker", "event_id", "event_date", "entry_date", "exit_date", "legs", "fill_alpha", "entry_cost", "exit_value", "ret"]
    pieces = []
    for year in store.table_years("trades"):
        for part in store._partition_files(store.paths.curated_partition("trades", year)):
            for batch in pq.ParquetFile(part).iter_batches(columns=columns, batch_size=4096, use_threads=False):
                frame = batch.to_pandas()
                mask = (frame["strategy"] == strategy) & (frame["provenance"].astype(str) == "engine.replay")
                if mask.any():
                    pieces.append(frame.loc[mask].copy())
    rows = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)
    for col in ("event_date", "entry_date", "exit_date"):
        rows[col] = pd.to_datetime(rows[col])
    events = store.read_table("earnings_events", columns=["event_id", "ticker", "event_date", "session"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    rows = rows.merge(events[["event_id", "session"]].drop_duplicates("event_id"), on="event_id", how="left")
    log(f"limited trade load: {len(rows):,} {strategy} rows")
    return rows


def extra_sections(accuracy: dict, seeds: dict) -> list[dict]:
    rows = []
    for name in ("surface_mean", "surface_mean_plus_scale"):
        result = accuracy["arms"][name]
        rows.append([name, f"{result['challenger']['mae']:.4f}", f"{result['challenger']['rmse']:.4f}", f"{result['challenger']['r']:.4f}", result["years_mae_improved"], f"{result['wilcoxon_p']:.4f}" if result["wilcoxon_p"] is not None else "NA"])
    interval = accuracy["arms"]["surface_scale"]
    seed_rows = [[seed, f"{delta:+.4f}"] for seed, delta in seeds["mae_delta_vs_baseline"].items()]
    return [
        {"title": "Surface-conditioned size-model comparison", "body": ["The primary combined arm changes the point forecast only where all pre-event surface ratios exist; every other row falls back to the incumbent point and interval logic."]},
        {"title": "Point-accuracy arms", "columns": ["arm", "MAE", "RMSE", "r", "years improved", "Wilcoxon p"], "align": ["---"] * 6, "rows": rows},
        {"title": "Distribution-parameter arm", "columns": ["measure", "incumbent", "surface-conditioned"], "align": ["---"] * 3, "rows": [["80% coverage", f"{interval['baseline_interval']['coverage80']:.4f}", f"{interval['challenger_interval']['coverage80']:.4f}"], ["mean width", f"{interval['baseline_interval']['width']:.4f}", f"{interval['challenger_interval']['width']:.4f}"], ["fallback rate", "0.0000", f"{interval['fallback_rate']:.4f}"]]},
        {"title": "Five-seed combined MAE deltas versus baseline", "columns": ["seed", "delta MAE"], "align": ["---", "---"], "rows": seed_rows},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    if args.reuse_scores:
        accuracy = json.loads((RESULTS / "accuracy.json").read_text())
        seeds = json.loads((RESULTS / "seed_robustness.json").read_text())
        frame = pd.read_parquet(RESULTS / "oos_predictions.parquet")
        decision_primary = pd.read_parquet(RESULTS / "primary_book_scores.parquet")
    else:
        data = panel_data()
        accuracy, seeds, frames = run_accuracy(data)
        frame = accuracy.pop("oos")
        write_json(RESULTS / "accuracy.json", accuracy)
        write_json(RESULTS / "seed_robustness.json", seeds)
        frame.to_parquet(RESULTS / "oos_predictions.parquet", index=False)
        decision_primary = book_scores(frame)
        decision_primary.to_parquet(RESULTS / "primary_book_scores.parquet", index=False)
        log(f"book scores: {len(decision_primary):,} rows")
    trades = load_engine_trades_limited("STR-THRU")
    trades = trades[pd.to_datetime(trades["event_date"]).dt.year >= BOOK_FIRST_YEAR].copy()
    spy = common.load_spy_daily()
    gate_state = PrecomputedGate(decision_primary)
    gate = Gate(fit=gate_state.fit, select=gate_state.select, name="size_surface_residual")
    result = evaluate(
        spec, trades, gate=gate, run_dir=HERE, spy_daily=spy,
        input_files=[paths.PANEL, RESULTS / "accuracy.json", RESULTS / "seed_robustness.json", RESULTS / "oos_predictions.parquet"],
        extra_sections=extra_sections(accuracy, seeds), stress=False, mc_paths=200, mc_block=50,
    )
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, result.results)
    write_json(RESULTS / "comparison.json", {"accuracy": accuracy, "seed_robustness": seeds, "primary_book": result.results["headline"]})
    log(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
