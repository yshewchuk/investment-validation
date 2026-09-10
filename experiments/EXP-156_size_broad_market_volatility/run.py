#!/usr/bin/env python3
"""EXP-156: test causal broad-market volatility features in the size model."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

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
ADDED = ("spy_vol5", "spy_vol60", "spy_vol252", "spy_vol20_rel252")
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import paths
from engine.data import store
from engine.data.features import panel as panel_features
from engine.evaluate import Gate, evaluate
from engine.features import load_panel
from engine.models.training import size_model
from engine.models.training.common import regression_metrics
from experiments import common, lib


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-156 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def complete_panel() -> pd.DataFrame:
    panel = panel_features.add_regime_features(size_model.prepare(load_panel()))
    features = tuple(size_model.FEATURES) + ADDED
    values = panel[list(features) + [size_model.TARGET]].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    out = panel.loc[mask].copy().reset_index(drop=True)
    log(f"Matched complete case: {len(out):,}/{len(panel):,} panel rows")
    return out


def annual_metrics(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for year, group in frame.groupby("year", sort=True):
        metrics = regression_metrics(group["abs_move"], group["pred"])
        rows.append({"year": int(year), **metrics})
    return rows


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


def fit_oos(data: pd.DataFrame, features: tuple[str, ...], seed: int) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in data["year"].unique() if y >= FIRST_TEST_YEAR):
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


def seed_summary(data: pd.DataFrame) -> tuple[dict, dict[int, dict[str, pd.DataFrame]]]:
    baseline = tuple(size_model.FEATURES)
    challenger = baseline + ADDED
    records = []
    frames: dict[int, dict[str, pd.DataFrame]] = {}
    for seed in SEEDS:
        log(f"Fitting matched baseline and challenger for seed={seed}")
        base_frame = fit_oos(data, baseline, seed)
        chall_frame = fit_oos(data, challenger, seed)
        if not base_frame[["ticker", "date"]].equals(chall_frame[["ticker", "date"]]):
            raise RuntimeError("baseline and challenger OOS rows differ")
        frames[seed] = {"baseline": base_frame, "challenger": chall_frame}
        for arm, frame in frames[seed].items():
            metrics = regression_metrics(frame["abs_move"], frame["pred"])
            records.append({
                "seed": seed,
                "arm": arm,
                **metrics,
                **rank_metrics(frame),
            })
    summary = pd.DataFrame(records)
    pivot = summary.pivot(index="seed", columns="arm", values="mae")
    summary_json = {
        "seeds": list(SEEDS),
        "arms": summary.to_dict("records"),
        "mae_delta_challenger_minus_baseline": {
            str(seed): float(pivot.loc[seed, "challenger"] - pivot.loc[seed, "baseline"])
            for seed in pivot.index
        },
        "median_mae_delta": float(
            np.median(pivot["challenger"].to_numpy() - pivot["baseline"].to_numpy())
        ),
    }
    return summary_json, frames


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
        "baseline": {
            **regression_metrics(base["abs_move"], base["pred"]),
            **rank_metrics(base),
        },
        "challenger": {
            **regression_metrics(chall["abs_move"], chall["pred"]),
            **rank_metrics(chall),
        },
        "by_year": rows,
        "years_mae_improved": int((deltas < 0).sum()),
        "wilcoxon_two_sided": {
            "statistic": float(stat.statistic),
            "pvalue": float(stat.pvalue),
        },
    }


def book_scores(frame: pd.DataFrame, features: tuple[str, ...], seed: int) -> pd.DataFrame:
    pieces = []
    for year in sorted(int(y) for y in frame["year"].unique() if y >= BOOK_FIRST_YEAR):
        train = frame[frame["year"] < year]
        test = frame[frame["year"] == year].copy()
        if len(train) < 500 or test.empty:
            continue
        model = size_model.fit(
            train[list(features)].to_numpy(dtype=float),
            train["abs_move"].to_numpy(dtype=float),
            seed=seed,
        )
        train_edge = model.predict(train[list(features)].to_numpy(dtype=float)) - train["or_implied"].to_numpy(dtype=float)
        cutoff = float(np.quantile(train_edge, 1.0 - TOP_FRACTION))
        test["forecast_edge"] = (
            model.predict(test[list(features)].to_numpy(dtype=float))
            - test["or_implied"].to_numpy(dtype=float)
        )
        test["selected"] = test["forecast_edge"] >= cutoff
        test["cutoff"] = cutoff
        pieces.append(test[["ticker", "date", "year", "forecast_edge", "selected", "cutoff"]])
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

    def gate(self) -> Gate:
        return Gate(fit=self.fit, select=self.select, name="top20_forecast_edge")


def extra_sections(accuracy: dict, seeds: dict, baseline_book: dict) -> list[dict]:
    base = accuracy["baseline"]
    challenger = accuracy["challenger"]
    rows = []
    for row in accuracy["by_year"]:
        rows.append([
            row["year"],
            f"{row['baseline_mae']:.4f}",
            f"{row['challenger_mae']:.4f}",
            f"{row['delta_mae']:+.4f}",
            f"{row['baseline_r']:.3f}",
            f"{row['challenger_r']:.3f}",
        ])
    seed_rows = []
    for row in seeds["arms"]:
        seed_rows.append([
            row["seed"], row["arm"], f"{row['mae']:.4f}", f"{row['r']:.4f}",
            f"{row['spearman']:.4f}", f"{row['top_bottom_decile']:.4f}",
        ])
    baseline_headline = baseline_book["headline"]
    return [
        {
            "title": "Primary size-model accuracy result",
            "body": [
                f"Matched OOS MAE: baseline {base['mae']:.4f} pp versus challenger "
                f"{challenger['mae']:.4f} pp. Matched OOS correlation: "
                f"{base['r']:.4f} versus {challenger['r']:.4f}.",
                f"The challenger improved MAE in {accuracy['years_mae_improved']}/"
                f"{len(accuracy['by_year'])} annual folds; paired two-sided "
                f"Wilcoxon p={accuracy['wilcoxon_two_sided']['pvalue']:.4f}.",
                "Both models use exactly the same rows, unchanged OLS plus MLP blend, "
                "and only causal SPY-close state. The current champion is unchanged.",
            ],
        },
        {
            "title": "Annual matched accuracy",
            "columns": ["year", "base MAE", "challenger MAE", "delta", "base r", "challenger r"],
            "align": ["---:"] * 6,
            "rows": rows,
        },
        {
            "title": "Fixed-seed robustness",
            "note": "Top-bottom is the realized move difference between top and bottom forecast deciles.",
            "columns": ["seed", "arm", "MAE", "r", "Spearman", "top-bottom pp"],
            "align": ["---:"] * 6,
            "rows": seed_rows,
        },
        {
            "title": "Economic companion check",
            "body": [
                "The report headline is the challenger forecast-edge top-20 STR-THRU book. "
                "Its pre-registered control is the baseline forecast-edge top-20 book, "
                "evaluated on the same real-price replay and annual-fold selection.",
                f"Baseline control at midpoint: n={baseline_headline['n']:,}, "
                f"mean={baseline_headline['mean']:+.4%}, "
                f"Sharpe={baseline_headline['sharpe_trade']:.3f}.",
                "This book does not decide a size-model promotion. It checks whether any "
                "accuracy change survives contact with executable option-price assumptions.",
            ],
        },
    ]


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    out["primary_spec"]["challenger"] = arm
    if arm == "baseline":
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def refresh_report(result, spec: dict, run_dir: Path, inputs, sections) -> None:
    """Refresh the report after its ledger row exists.

    The evaluator renders before this runner appends the multiple-testing
    ledger row. Recomputing the checklist here makes the final persisted report
    reflect the completed experiment rather than that harmless ordering gap.
    """
    from engine.evaluate import EvalResult
    from engine.report import Report, accuracy_checklist

    checks = accuracy_checklist(
        result.results, spec, ledger_path=paths.ROOT / "experiments" / "LEDGER.csv"
    )
    result.results["checklist"] = [
        {"name": item.name, "status": item.status, "evidence": item.evidence}
        for item in checks
    ]
    result.results["checklist_fails"] = sum(
        item.status == "FAIL" for item in checks
    )
    stem = lib.spec_hash(spec)[:12]
    metrics = run_dir / "results" / f"metrics_{stem}.json"
    metrics.write_text(json.dumps(result.results, indent=1, default=str))
    refreshed = EvalResult(spec=spec, results=result.results, run_dir=run_dir)
    result.report_path = Report.from_eval(
        refreshed, input_files=inputs, extra_sections=sections
    ).write(run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    data = complete_panel()
    seed_results, frames = seed_summary(data)
    accuracy = primary_accuracy(frames)
    write_json(RESULTS / "accuracy.json", accuracy)
    write_json(RESULTS / "seed_robustness.json", seed_results)

    base_features = tuple(size_model.FEATURES)
    challenger_features = base_features + ADDED
    decisions = {
        "baseline": book_scores(data, base_features, SEEDS[0]),
        "challenger": book_scores(data, challenger_features, SEEDS[0]),
    }
    for arm, frame in decisions.items():
        frame.to_parquet(RESULTS / f"{arm}_book_scores.parquet", index=False)
        log(f"{arm} book scores: {len(frame):,} rows, selected={int(frame['selected'].sum()):,}")

    trades = common.load_engine_trades("STR-THRU")
    trade_year = pd.to_datetime(trades["event_date"]).dt.year
    trades = trades[trade_year >= BOOK_FIRST_YEAR].copy()
    spy = common.load_spy_daily()
    inputs = [
        paths.PANEL,
        paths.GSPC_DAILY,
        RESULTS / "accuracy.json",
        RESULTS / "seed_robustness.json",
        RESULTS / "baseline_book_scores.parquet",
        RESULTS / "challenger_book_scores.parquet",
    ]
    results = {}
    for arm in ("baseline", "challenger"):
        run_dir = HERE if arm == "challenger" else HERE / "arms" / arm
        this_spec = arm_spec(spec, arm)
        gate = PrecomputedGate(decisions[arm]).gate()
        log(f"Evaluating real-price companion book: {arm}")
        result = evaluate(
            this_spec,
            trades,
            gate=gate,
            run_dir=run_dir,
            repricer=common.make_repricer("STR-THRU"),
            spy_daily=spy,
            input_files=inputs,
            extra_sections=(
                (lambda result: extra_sections(accuracy, seed_results, results["baseline"].results))
                if arm == "challenger"
                else [{"title": "Control book", "body": ["Baseline size-model forecast-edge top-20 selector."]}]
            ),
        )
        results[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, this_spec, result.results)
        head = result.results["headline"]
        log(f"{arm} book: n={head['n']:,} mean={head['mean']:+.4f} sharpe={head['sharpe_trade']:.3f}")
    refresh_report(
        results["baseline"], arm_spec(spec, "baseline"), HERE / "arms" / "baseline",
        inputs, [{"title": "Control book", "body": ["Baseline size-model forecast-edge top-20 selector."]}],
    )
    refresh_report(
        results["challenger"], arm_spec(spec, "challenger"), HERE, inputs,
        extra_sections(accuracy, seed_results, results["baseline"].results),
    )
    write_json(RESULTS / "comparison.json", {
        "accuracy": accuracy,
        "seed_robustness": seed_results,
        "baseline_book": results["baseline"].results["headline"],
        "challenger_book": results["challenger"].results["headline"],
    })
    log(f"Generated report: {results['challenger'].report_path}")


if __name__ == "__main__":
    main()
