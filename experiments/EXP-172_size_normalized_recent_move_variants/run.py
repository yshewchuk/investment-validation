#!/usr/bin/env python3
"""EXP-172: compare normalized recent absolute-move feature variants.

The process is intentionally run by the caller with one BLAS thread, one CPU
core, and a memory limit so it can share the host with other workloads.
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
ARM_FEATURES = {
    "raw_ema2": ("ema2_prior_abs_move",),
    "raw_ema4": ("ema4_prior_abs_move",),
    "centered_ema4": ("ema4_abs_minus_mean_abs",),
    "ema4_relative_to_history": ("ema4_abs_over_mean_abs",),
    "ema2_relative_to_ema4": ("ema2_abs_over_ema4_abs",),
}


def log(message: str) -> None:
    print(f"[EXP-172 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def add_derived(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["ema4_abs_minus_mean_abs"] = out["ema4_prior_abs_move"] - out["mean_prior_abs_move"]
    out["ema4_abs_over_mean_abs"] = out["ema4_prior_abs_move"] / out["mean_prior_abs_move"].clip(lower=1e-6)
    out["ema2_abs_over_ema4_abs"] = out["ema2_prior_abs_move"] / out["ema4_prior_abs_move"].clip(lower=1e-6)
    return out


def complete_panel() -> pd.DataFrame:
    panel = add_derived(size_model.prepare(load_panel()))
    needed = set(BASE) | {"abs_move"}
    for fields in ARM_FEATURES.values():
        needed.update(fields)
    values = panel[sorted(needed)].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    out = panel.loc[mask].copy().reset_index(drop=True)
    log(f"matched complete case: {len(out):,}/{len(panel):,} panel rows")
    return out


def fit_oos(data: pd.DataFrame, features: tuple[str, ...], seed: int) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR):
        train = data[data["year"] < year]
        test = data[data["year"] == year].copy()
        model = size_model.fit(train[list(features)].to_numpy(dtype=float), train["abs_move"].to_numpy(dtype=float), seed=seed)
        test["pred"] = model.predict(test[list(features)].to_numpy(dtype=float))
        pieces.append(test)
        log(f"seed={seed} arm={len(features)} features fold={year} train={len(train):,} test={len(test):,}")
    return pd.concat(pieces, ignore_index=True)


def rank_metrics(frame: pd.DataFrame) -> dict:
    ranked = frame.dropna(subset=["pred", "abs_move"]).copy()
    corr = spearmanr(ranked["pred"], ranked["abs_move"], nan_policy="omit")
    ranked["decile"] = pd.qcut(ranked["pred"], 10, labels=False, duplicates="drop")
    means = ranked.groupby("decile", observed=True)["abs_move"].mean()
    return {"n": int(len(ranked)), "spearman": float(corr.statistic), "top_bottom_decile": float(means.iloc[-1] - means.iloc[0])}


def annual(frame: pd.DataFrame) -> list[dict]:
    return [{"year": int(year), **regression_metrics(group["abs_move"], group["pred"])} for year, group in frame.groupby("year", sort=True)]


def run_accuracy(data: pd.DataFrame) -> tuple[dict, dict]:
    arms = {"baseline": BASE, **{name: BASE + fields for name, fields in ARM_FEATURES.items()}}
    all_records = []
    first_frames = {}
    seed_deltas = {}
    for seed in SEEDS:
        log(f"starting seed={seed}")
        frames = {}
        for name, features in arms.items():
            frame = fit_oos(data, features, seed)
            frames[name] = frame
            metrics = {"seed": seed, "arm": name, **regression_metrics(frame["abs_move"], frame["pred"]), **rank_metrics(frame)}
            all_records.append(metrics)
            if seed == SEEDS[0]:
                first_frames[name] = frame
        base_mae = next(r["mae"] for r in all_records if r["seed"] == seed and r["arm"] == "baseline")
        seed_deltas[str(seed)] = {name: float(next(r["mae"] for r in all_records if r["seed"] == seed and r["arm"] == name) - base_mae) for name in ARM_FEATURES}
    summary = pd.DataFrame(all_records)
    accuracy = {}
    base = first_frames["baseline"]
    base_year = pd.DataFrame(annual(base)).set_index("year")
    for name in ARM_FEATURES:
        cand = first_frames[name]
        cand_year = pd.DataFrame(annual(cand)).set_index("year")
        deltas = np.asarray([cand_year.loc[y, "mae"] - base_year.loc[y, "mae"] for y in base_year.index], dtype=float)
        stat = wilcoxon(deltas, alternative="two-sided", zero_method="wilcox")
        accuracy[name] = {
            "baseline": {**regression_metrics(base["abs_move"], base["pred"]), **rank_metrics(base)},
            "challenger": {**regression_metrics(cand["abs_move"], cand["pred"]), **rank_metrics(cand)},
            "years_mae_improved": int((deltas < 0).sum()),
            "wilcoxon_p": float(stat.pvalue),
            "by_year": [{"year": int(y), "baseline_mae": float(base_year.loc[y, "mae"]), "challenger_mae": float(cand_year.loc[y, "mae"]), "delta_mae": float(cand_year.loc[y, "mae"] - base_year.loc[y, "mae"])} for y in base_year.index],
        }
    return {"arms": accuracy, "rows": int(len(base))}, {"records": all_records, "mae_delta_vs_baseline": seed_deltas}


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


def load_engine_trades_limited(strategy: str) -> pd.DataFrame:
    """Load only the columns needed by replay evaluation under the memory cap."""
    import pyarrow.parquet as pq

    columns = ["strategy", "provenance", "ticker", "event_id", "event_date",
               "entry_date", "exit_date", "legs", "fill_alpha", "entry_cost",
               "exit_value", "ret"]
    pieces = []
    for year in store.table_years("trades"):
        for part in store._partition_files(store.paths.curated_partition("trades", year)):
            parquet = pq.ParquetFile(part)
            for batch in parquet.iter_batches(columns=columns, batch_size=4096, use_threads=False):
                frame = batch.to_pandas()
                mask = ((frame["strategy"] == strategy)
                        & (frame["provenance"].astype(str) == "engine.replay"))
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
    for name, result in accuracy["arms"].items():
        rows.append([name, f"{result['challenger']['mae']:.4f}", f"{result['challenger']['rmse']:.4f}", f"{result['challenger']['r']:.4f}", result["years_mae_improved"], f"{result['wilcoxon_p']:.4f}"])
    seed_rows = []
    for seed, deltas in seeds["mae_delta_vs_baseline"].items():
        seed_rows.append([seed] + [f"{deltas[name]:+.4f}" for name in ARM_FEATURES])
    return [
        {"title": "All-arm matched OOS accuracy", "body": ["The primary is the pre-registered ema4-relative-to-history arm. Other arms are secondary diagnostics and are not promotion candidates."]},
        {"title": "Arm comparison", "columns": ["arm", "MAE", "RMSE", "r", "years improved", "Wilcoxon p"], "align": ["---"] * 6, "rows": rows},
        {"title": "Five-seed MAE deltas versus baseline", "columns": ["seed", *ARM_FEATURES.keys()], "align": ["---"] * 6, "rows": seed_rows},
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
        data = add_derived(size_model.prepare(load_panel()))
        decision_base = pd.read_parquet(RESULTS / "baseline_book_scores.parquet")
        decision_primary = pd.read_parquet(RESULTS / "primary_book_scores.parquet")
    else:
        data = complete_panel()
        accuracy, seeds = run_accuracy(data)
        write_json(RESULTS / "accuracy.json", accuracy)
        write_json(RESULTS / "seed_robustness.json", seeds)
        decision_base = book_scores(data, BASE, SEEDS[0])
        decision_primary = book_scores(data, BASE + ARM_FEATURES["ema4_relative_to_history"], SEEDS[0])
        decision_base.to_parquet(RESULTS / "baseline_book_scores.parquet", index=False)
        decision_primary.to_parquet(RESULTS / "primary_book_scores.parquet", index=False)
        log(f"book scores: baseline={len(decision_base):,}, primary={len(decision_primary):,}")
    trades = load_engine_trades_limited("STR-THRU")
    trades = trades[pd.to_datetime(trades["event_date"]).dt.year >= BOOK_FIRST_YEAR].copy()
    spy = common.load_spy_daily()
    gate_state = PrecomputedGate(decision_primary)
    gate = Gate(fit=gate_state.fit, select=gate_state.select, name="size_ema4_relative_edge")
    result = evaluate(
        spec, trades, gate=gate, run_dir=HERE,
        repricer=common.make_repricer("STR-THRU"), spy_daily=spy,
        input_files=[paths.PANEL, RESULTS / "accuracy.json", RESULTS / "seed_robustness.json"],
        extra_sections=extra_sections(accuracy, seeds),
    )
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, result.results)
    write_json(RESULTS / "comparison.json", {"accuracy": accuracy, "seed_robustness": seeds, "primary_book": result.results["headline"]})
    log(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
