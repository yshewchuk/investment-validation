#!/usr/bin/env python3
"""EXP-152: annual top-percentile sweep over frozen move-aware gate signals."""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import gc
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SCORE_CACHE = RESULTS / "oos_scores.parquet"
DIAGNOSTICS_PATH = RESULTS / "fold_diagnostics.json"
TOP_FRACTIONS = (0.10, 0.20, 0.30, 0.40)
OPERATIONAL = "incumbent_registered_threshold"
PRIMARY_FAMILY = "model_plus_forecast_pnl_plus_analogs"
PRIMARY = f"{PRIMARY_FAMILY}_top20"
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from experiments import common, lib  # noqa: E402


def load_runner(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


previous = load_runner(
    "exp151_runner",
    ROOT / "experiments/EXP-151_str_runup_move_aware_gate_corrected_harness/run.py",
)
base = previous.base
BASE_FEATURES = base.BASE_FEATURES
FAMILY_FEATURES = {
    "base": base.ARM_FEATURES["base_top20"],
    "model_plus_raw_forecast": base.ARM_FEATURES["model_plus_raw_forecast"],
    "model_plus_forecast_pnl": base.ARM_FEATURES["model_plus_forecast_pnl"],
    "model_plus_analogs": base.ARM_FEATURES["model_plus_analogs"],
    "model_plus_raw_forecast_plus_analogs": (
        base.ARM_FEATURES["model_plus_raw_forecast_plus_analogs"]
    ),
    PRIMARY_FAMILY: base.ARM_FEATURES[PRIMARY_FAMILY],
}
FAMILIES = tuple(FAMILY_FEATURES)
ALL_ARMS = (OPERATIONAL,) + tuple(
    f"{family}_top{int(fraction * 100)}"
    for family in FAMILIES
    for fraction in TOP_FRACTIONS
)


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-152 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def arm_name(family: str, fraction: float) -> str:
    return f"{family}_top{int(fraction * 100)}"


def arm_details(arm: str) -> tuple[str, float | None]:
    if arm == OPERATIONAL:
        return "base", None
    family, fraction = arm.rsplit("_top", 1)
    return family, float(fraction) / 100.0


def fit_family(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[float, float], int, int]:
    columns = list(features)
    train_ok = np.isfinite(train[columns + ["ret"]].to_numpy(dtype=float)).all(axis=1)
    test_ok = np.isfinite(test[columns].to_numpy(dtype=float)).all(axis=1)
    fit_rows = train.loc[train_ok]
    score = np.full(len(test), np.nan)
    pwin = np.full(len(test), np.nan)
    cutoffs = {fraction: float("nan") for fraction in TOP_FRACTIONS}
    if len(fit_rows) < base.MIN_MODEL_ROWS or not test_ok.any():
        return score, pwin, cutoffs, int(len(fit_rows)), int(test_ok.sum())
    X_train = fit_rows[columns].to_numpy(dtype=float)
    y_train = fit_rows["ret"].to_numpy(dtype=float)
    model = base.gate_mod.fit(X_train, y_train)
    train_score = np.asarray(model.predict(X_train), dtype=float)
    score[test_ok] = model.predict(test.loc[test_ok, columns].to_numpy(dtype=float))
    cutoffs = {
        fraction: float(base.gate_mod.choose_threshold(train_score, fraction))
        for fraction in TOP_FRACTIONS
    }
    base_rate = float((y_train > 0).mean())
    try:
        isotonic = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(train_score, (y_train > 0).astype(float))
        pwin[test_ok] = isotonic.predict(score[test_ok])
    except ValueError:
        pwin[test_ok] = base_rate
    return score, pwin, cutoffs, int(len(fit_rows)), int(test_ok.sum())


def generate_scores(
    dataset: pd.DataFrame, registered_threshold: float, force: bool
) -> tuple[pd.DataFrame, list[dict]]:
    if SCORE_CACHE.exists() and DIAGNOSTICS_PATH.exists() and not force:
        scores = pd.read_parquet(SCORE_CACHE)
        scores["event_date"] = pd.to_datetime(scores["event_date"])
        diagnostics = json.loads(DIAGNOSTICS_PATH.read_text())
        log(f"Loaded score cache: {len(scores):,} OOS events")
        return scores, diagnostics

    base_complete = np.isfinite(
        dataset[list(BASE_FEATURES) + ["ret", "mcap_usd"]].to_numpy(dtype=float)
    ).all(axis=1)
    live_domain = dataset["mcap_usd"] >= 1e9
    pieces = []
    diagnostics = []
    for year in range(2020, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year].copy()
        test = dataset[
            (dataset["year"] == year) & base_complete & live_domain
        ].copy()
        if test.empty:
            continue
        piece = test[
            [
                "event_id",
                "ticker",
                "event_date",
                "year",
                "ret",
                "mcap_usd",
                "relative_spread",
            ]
        ].rename(columns={"ret": "realized_ret"}).reset_index(drop=True)
        fold = {
            "year": int(year),
            "n_test_live_domain": int(len(test)),
            "training_rows": {},
            "scoreable_test": {},
            "cutoffs": {},
        }
        cache = {}
        for family, features in FAMILY_FEATURES.items():
            score, pwin, cutoffs, n_train, n_test = fit_family(train, test, features)
            cache[family] = (score, pwin)
            for fraction in TOP_FRACTIONS:
                arm = arm_name(family, fraction)
                cutoff = cutoffs[fraction]
                piece[arm] = score
                piece[f"{arm}_pwin"] = pwin
                piece[f"selected_{arm}"] = (
                    np.isfinite(score) & np.isfinite(cutoff) & (score >= cutoff)
                )
                fold["training_rows"][arm] = n_train
                fold["scoreable_test"][arm] = n_test
                fold["cutoffs"][arm] = cutoff
        base_score, base_pwin = cache["base"]
        piece[OPERATIONAL] = base_score
        piece[f"{OPERATIONAL}_pwin"] = base_pwin
        piece[f"selected_{OPERATIONAL}"] = (
            np.isfinite(base_score) & (base_score >= registered_threshold)
        )
        fold["training_rows"][OPERATIONAL] = fold["training_rows"]["base_top20"]
        fold["scoreable_test"][OPERATIONAL] = fold["scoreable_test"]["base_top20"]
        fold["cutoffs"][OPERATIONAL] = registered_threshold
        diagnostics.append(fold)
        pieces.append(piece)
        counts = ", ".join(
            f"{arm}={int(piece[f"selected_{arm}"].sum())}"
            for arm in ALL_ARMS
        )
        log(f"Gate fold {year}: {len(test):,} candidates; {counts}")

    scores = pd.concat(pieces, ignore_index=True)
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    scores = scores.sort_values(["event_date", "event_id"]).reset_index(drop=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    write_json(DIAGNOSTICS_PATH, diagnostics)
    log(f"OOS scores written: {len(scores):,} events")
    return scores, diagnostics


def selected_slice(scores: pd.DataFrame, arm: str, mask=None) -> dict:
    selected = scores[f"selected_{arm}"].astype(bool)
    if mask is not None:
        selected &= mask
    rows = scores.loc[selected]
    return {
        "n": int(len(rows)),
        "mean": float(rows["realized_ret"].mean()) if len(rows) else None,
        "win_rate": float((rows["realized_ret"] > 0).mean()) if len(rows) else None,
    }


def arm_spec(spec: dict, arm: str) -> dict:
    family, fraction = arm_details(arm)
    out = deepcopy(spec)
    out["primary_spec"]["challenger"] = arm
    out["primary_spec"]["score_family"] = family
    out["primary_spec"]["top_fraction"] = fraction
    if fraction is None:
        out["primary_spec"]["gate_rule"] = (
            "The registered operational STR-RUNUP threshold is applied to the "
            "base score unchanged in every annual test fold."
        )
    if arm != PRIMARY:
        out["promotion_target"] = None
        out["grid_cell"] = True
    return out


def fmt_pct(value) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{100 * value:+.2f}%"


def completed_result(run_dir: Path, spec: dict):
    stem = lib.spec_hash(spec)[:12]
    path = run_dir / "results" / f"metrics_{stem}.json"
    if not path.exists():
        return None
    results = json.loads(path.read_text())
    if "headline" not in results:
        raise RuntimeError(f"incomplete cached evaluation: {path}")
    return SimpleNamespace(results=results, report_path=run_dir / "REPORT.md")


def release_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def sweep_sections(result, evaluations: dict, scores: pd.DataFrame) -> list[dict]:
    all_results = dict(evaluations)
    all_results[PRIMARY] = result
    liquid_mask = scores["relative_spread"].to_numpy(dtype=float) <= 0.50
    matrix = []
    for arm in ALL_ARMS:
        family, fraction = arm_details(arm)
        headline = all_results[arm].results["headline"]
        liquid = selected_slice(scores, arm, liquid_mask)
        matrix.append([
            "registered" if fraction is None else f"top {fraction:.0%}",
            family,
            f"{headline["n"]:,}",
            fmt_pct(headline["mean"]),
            fmt_pct(headline.get("dollar_weighted")),
            fmt_pct(headline["cagr"]),
            f"{headline["sharpe_trade"]:.2f}",
            f"{headline["years_positive"]}/{headline["years_evaluated"]}",
            (
                f"{headline["breakeven_alpha"]:.3f}"
                if headline.get("breakeven_alpha") is not None else "n/a"
            ),
            f"{liquid["n"]:,}",
            fmt_pct(liquid["mean"]),
        ])
    bootstrap_rows = []
    for fraction in TOP_FRACTIONS:
        challenger = arm_name(PRIMARY_FAMILY, fraction)
        incumbent = arm_name("base", fraction)
        bootstrap = base.weekly_bootstrap(
            scores, challenger=challenger, incumbent=incumbent
        )
        lo, hi = bootstrap["ci90"]
        bootstrap_rows.append([
            f"top {fraction:.0%}",
            f"{selected_slice(scores, challenger)["n"]:,}",
            f"{selected_slice(scores, incumbent)["n"]:,}",
            fmt_pct(bootstrap["observed"]),
            f"[{fmt_pct(lo)}, {fmt_pct(hi)}]",
            f"{bootstrap["p_gt_zero"]:.1%}",
        ])
    return [
        {
            "title": "Pre-registered annual selection-rate sweep",
            "body": [
                "All score fits, source signals and priced T-14 trades are fixed "
                "from EXP-151. Each top fraction is a distinct causal policy: "
                "its cutoff came from only that folds earlier training scores.",
                "The fixed registered threshold is retained as an operational "
                "benchmark. It is not repeated at every percentile because its "
                "score is identical to the base family.",
                "The combined-evidence top-20 cell is the sole headline. This "
                "entire selection-rate grid is exploratory and cannot promote "
                "a gate without a separate confirmation.",
            ],
        },
        {
            "title": "All score families and annual top-percent selections",
            "columns": [
                "selection", "score family", "selected", "mean", "capital weighted",
                "CAGR", "Sharpe", "years+", "breakeven alpha", "n spread<=50%", "mean spread<=50%",
            ],
            "align": ["---", "---"] + ["---:"] * 9,
            "rows": matrix,
        },
        {
            "title": "Combined evidence versus matching base family",
            "note": "Earnings-week bootstrap; both sides use the same annual top fraction.",
            "columns": [
                "selection", "combined n", "base n", "mean difference", "90% interval", "P(greater than zero)",
            ],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": bootstrap_rows,
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    registry = load_registry(missing_ok=False)
    incumbent = registry.champion("gate", base.STRATEGY)
    if incumbent.id != "gate_midfill_str_runup":
        raise RuntimeError(f"registered STR-RUNUP gate changed: {incumbent.id}")
    if tuple(incumbent.features) != BASE_FEATURES or incumbent.threshold is None:
        raise RuntimeError("registered STR-RUNUP gate contract changed")

    previous.RESULTS = RESULTS
    trades_all = base.load_trades()
    dataset = previous.load_frozen_signals(spec, trades_all, False)
    scores, diagnostics = generate_scores(
        dataset, float(incumbent.threshold), args.force_scores
    )
    write_json(RESULTS / "fold_diagnostics.json", diagnostics)
    signal_coverage = {
        "events": int(len(dataset)),
        "forecast_pnl_mean": int(dataset["forecast_pnl_mean"].notna().sum()),
        "analog_mean": int(dataset["analog_mean"].notna().sum()),
    }
    write_json(RESULTS / "signal_coverage.json", signal_coverage)

    event_ids = set(scores["event_id"].astype(str))
    trades = trades_all[trades_all["event_id"].astype(str).isin(event_ids)].copy()
    if trades["event_id"].nunique() != len(scores):
        raise RuntimeError("OOS scores do not reconcile to the priced trade set")
    spy = common.load_spy_daily()
    input_files = [
        previous.MOVE_AWARE_SIGNALS,
        SCORE_CACHE,
        DIAGNOSTICS_PATH,
    ]
    input_files += sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))

    evaluations = {}
    run_order = [arm for arm in ALL_ARMS if arm != PRIMARY] + [PRIMARY]
    for arm in run_order:
        this_spec = arm_spec(spec, arm)
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        family, fraction = arm_details(arm)
        prior = completed_result(run_dir, this_spec)
        if prior is not None:
            evaluations[arm] = prior
            headline = prior.results["headline"]
            log(
                f"Resuming {arm}: n={headline["n"]:,}, "
                f"mean={headline["mean"]:+.4f}, Sharpe={headline["sharpe_trade"]:.3f}"
            )
            continue
        log(f"Evaluating {arm}")
        gate = base.PrecomputedGate(scores, arm).gate()
        if arm == PRIMARY:
            extra = lambda result: sweep_sections(result, evaluations, scores)
        else:
            label = "registered threshold" if fraction is None else f"top {fraction:.0%}"
            extra = lambda result, family=family, label=label: [{
                "title": "Gate construction",
                "body": [
                    f"Score family: {family}.",
                    f"Selection policy: {label}.",
                    "The score fit used only earlier annual folds and the cutoff "
                    "was computed from that folds training predictions.",
                ],
            }]
        repricer = common.make_repricer(base.STRATEGY)
        try:
            result = evaluate(
                this_spec,
                trades,
                gate=gate,
                run_dir=run_dir,
                repricer=repricer,
                spy_daily=spy,
                input_files=input_files,
                extra_sections=extra,
            )
        finally:
            del repricer
            release_memory()
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, this_spec, result.results)
        headline = result.results["headline"]
        log(
            f"{arm}: n={headline["n"]:,}, mean={headline["mean"]:+.4f}, "
            f"CAGR={headline["cagr"]:+.4f}, Sharpe={headline["sharpe_trade"]:.3f}"
        )

    comparison = {
        arm: evaluations[arm].results["headline"] for arm in ALL_ARMS
    }
    comparison["signal_coverage"] = signal_coverage
    write_json(RESULTS / "comparison.json", comparison)
    log(f"Generated report: {evaluations[PRIMARY].report_path}")


if __name__ == "__main__":
    main()
