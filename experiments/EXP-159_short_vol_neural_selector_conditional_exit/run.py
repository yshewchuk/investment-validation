#!/usr/bin/env python3
"""EXP-159: corrected-exit neural selector plus current arithmetic gate."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CANDIDATES = (
    ROOT
    / "experiments/EXP-134_priced_right_funded_and_held_structure_s"
    / "results/candidates.parquet"
)
SCORE_CACHE = RESULTS / "oos_scores.parquet"
DIAGNOSTICS = RESULTS / "fold_diagnostics.json"
FIRST_TEST_YEAR = 2020
ARITHMETIC = "arithmetic_pnl_gate"
PRIMARY = "nn_all_categories_top20"
CONTROL = "nn_pnl_analogs_top20"
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import pnl_sim
from engine.evaluate import evaluate
from engine.features import load_panel
from experiments import common, lib


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


base = load_module(
    "exp158_runner",
    ROOT / "experiments/EXP-158_short_vol_neural_selector/run.py",
)
ARM_FEATURES = {ARITHMETIC: (), **base.ARM_FEATURES}
ALL_ARMS = tuple(ARM_FEATURES)


def log(message: str) -> None:
    print(f"[EXP-159 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
        "fill_alpha", "entry_cost", "exit_value", "exit_value_expiry",
        "exit_arb_ok", "spot_entry", "ret", "exp_pnl_sim",
        "exp_pnl_sim_select", "pred_abs_move", "pred_abs_move_sd",
        "width_over_forecast", "half_width_pct_spot", "anchor_over_spot",
        "n_legs", "n_admissible", "dte_entry", "rel_spread",
        "quote_repaired", "wide_market",
    ]
    raw = pd.read_parquet(CANDIDATES, columns=columns)
    raw = raw[raw["arm"].eq("best_all")].copy()
    for column in ("event_date", "entry_date", "exit_date"):
        raw[column] = pd.to_datetime(raw[column]).dt.normalize()
    numeric = [column for column in columns if column not in {
        "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
        "exit_arb_ok",
    }]
    for column in numeric:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    hold = (~raw["exit_arb_ok"].fillna(True).astype(bool)) | (raw["exit_value"] < 0)
    raw.loc[hold, "exit_value"] = raw.loc[hold, "exit_value_expiry"]
    raw = raw.dropna(subset=["entry_cost", "exit_value"]).copy()
    raw["pnl"] = raw["exit_value"] - raw["entry_cost"]
    raw["ret"] = raw["pnl"] / raw["entry_cost"]
    if (raw["ret"] < -1.0 - 1e-8).any():
        raise RuntimeError("conditional exits still contain losses below the debit")
    raw["entry_cost_pct"] = 100.0 * raw["entry_cost"] / raw["spot_entry"]
    raw["year"] = raw["event_date"].dt.year
    mid = raw[np.isclose(raw["fill_alpha"], 0.5)].copy()
    if mid["event_id"].duplicated().any():
        raise RuntimeError("corrected midpoint candidates are not one row per event")

    history = tuple(base.CATEGORIES["history"])
    market = tuple(base.CATEGORIES["market"])
    panel = load_panel()[["ticker", "date", "mcap_usd", *history, *market]].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.drop_duplicates(["ticker", "date"], keep="last")
    mid = mid.merge(
        panel, left_on=["ticker", "event_date"], right_on=["ticker", "date"],
        how="left", validate="one_to_one",
    ).drop(columns="date")
    for column in ("mcap_usd", *history, *market):
        mid[column] = pd.to_numeric(mid[column], errors="coerce")
    for column in ("quote_repaired", "wide_market"):
        mid[column] = mid[column].fillna(False).astype(float)
    log(
        f"Validated conditional exits: {len(mid):,} midpoint events, "
        f"{len(raw):,} fill rows, held-to-expiry={int(hold.sum()):,}"
    )
    return mid.sort_values(["entry_date", "event_id"]).reset_index(drop=True), raw


def arithmetic_selection(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    history = frame[["event_date", "exp_pnl_sim"]].copy()
    month = frame["event_date"].dt.to_period("M")
    bars = {
        key: pnl_sim.trailing_cutoff(history, key.to_timestamp())
        for key in month.unique()
    }
    cutoff = month.map(bars).astype(float).to_numpy()
    selected = (
        np.isfinite(frame["exp_pnl_sim"].to_numpy(dtype=float))
        & np.isfinite(cutoff)
        & (frame["exp_pnl_sim"].to_numpy(dtype=float) >= cutoff)
        & (frame["rel_spread"].to_numpy(dtype=float) <= 0.25)
        & (frame["mcap_usd"].to_numpy(dtype=float) >= 10e9)
    )
    return selected, cutoff


def generate_scores(dataset: pd.DataFrame, force: bool) -> tuple[pd.DataFrame, list[dict]]:
    if SCORE_CACHE.exists() and DIAGNOSTICS.exists() and not force:
        scores = pd.read_parquet(SCORE_CACHE)
        scores["event_date"] = pd.to_datetime(scores["event_date"])
        return scores, json.loads(DIAGNOSTICS.read_text())
    arithmetic, arithmetic_cutoff = arithmetic_selection(dataset)
    pieces, diagnostics = [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year]
        test = dataset[dataset["year"] == year]
        piece = test[["event_id", "ticker", "event_date", "entry_date", "exit_date", "ret"]].copy()
        piece = piece.rename(columns={"ret": "realized_ret"})
        test_positions = test.index.to_numpy()
        fold = {"year": year, "arms": {}}
        piece[ARITHMETIC] = test["exp_pnl_sim"].to_numpy(dtype=float)
        piece[f"selected_{ARITHMETIC}"] = arithmetic[test_positions]
        fold["arms"][ARITHMETIC] = {
            "features": 1, "train_rows": None, "scoreable_test": int(np.isfinite(piece[ARITHMETIC]).sum()),
            "selected": int(piece[f"selected_{ARITHMETIC}"].sum()),
            "cutoff": float(np.nanmedian(arithmetic_cutoff[test_positions])),
        }
        for arm, features in base.ARM_FEATURES.items():
            train_ok = np.isfinite(train[list(features) + ["ret"]].to_numpy(dtype=float)).all(axis=1)
            test_ok = np.isfinite(test[list(features)].to_numpy(dtype=float)).all(axis=1)
            score = np.full(len(test), np.nan)
            fit = train.loc[train_ok]
            if len(fit) >= 500 and test_ok.any():
                models = base.fit_ensemble(
                    fit[list(features)].to_numpy(dtype=float),
                    fit["ret"].to_numpy(dtype=float),
                )
                train_score = base.predict(models, fit[list(features)].to_numpy(dtype=float))
                score[test_ok] = base.predict(models, test.loc[test_ok, list(features)].to_numpy(dtype=float))
                cutoff = float(np.quantile(train_score, 0.80))
            else:
                cutoff = float("nan")
            piece[arm] = score
            piece[f"selected_{arm}"] = np.isfinite(score) & np.isfinite(cutoff) & (score >= cutoff)
            fold["arms"][arm] = {
                "features": len(features), "train_rows": int(len(fit)),
                "scoreable_test": int(test_ok.sum()), "selected": int(piece[f"selected_{arm}"].sum()),
                "cutoff": cutoff,
            }
        pieces.append(piece)
        diagnostics.append(fold)
        log(", ".join(f"{arm}={fold['arms'][arm]['selected']}" for arm in ALL_ARMS))
    scores = pd.concat(pieces, ignore_index=True).sort_values(["event_date", "event_id"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    write_json(DIAGNOSTICS, diagnostics)
    return scores, diagnostics


def rank_metrics(scores: pd.DataFrame) -> dict:
    out = {}
    for arm in ALL_ARMS:
        rows = scores[[arm, "realized_ret"]].dropna()
        decile = pd.qcut(rows[arm], 10, labels=False, duplicates="drop")
        means = rows.groupby(decile)["realized_ret"].mean()
        out[arm] = {
            "n": int(len(rows)),
            "spearman": float(spearmanr(rows[arm], rows["realized_ret"]).statistic),
            "top_bottom": float(means.iloc[-1] - means.iloc[0]),
        }
    return out


def bootstrap(scores: pd.DataFrame, left: str, right: str) -> dict:
    data = scores.copy()
    data["delta"] = (
        data[f"selected_{left}"].to_numpy(dtype=bool) * data["realized_ret"].to_numpy(dtype=float)
        - data[f"selected_{right}"].to_numpy(dtype=bool) * data["realized_ret"].to_numpy(dtype=float)
    )
    weekly = data.groupby(data["event_date"].dt.to_period("W"))["delta"].mean().to_numpy(dtype=float)
    rng = np.random.default_rng(20260908)
    draws = weekly[rng.integers(0, len(weekly), size=(10000, len(weekly)))].mean(axis=1)
    return {
        "left": left, "right": right, "weeks": int(len(weekly)),
        "observed": float(weekly.mean()), "ci90": [float(x) for x in np.quantile(draws, [0.05, 0.95])],
        "p_gt_zero": float((draws > 0).mean()),
    }


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    out["primary_spec"]["evaluated_arm"] = arm
    if arm != PRIMARY:
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def plan(specs: dict) -> None:
    ledger = lib.ledger_read()
    rows = []
    for spec in specs.values():
        digest = lib.spec_hash(spec)
        if not (ledger["spec_hash"] == digest).any():
            rows.append({
                "id": spec["id"], "spec_hash": digest, "date": "2026-09-08",
                "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False",
            })
    if rows:
        lib.ledger_append(rows)
        log(f"Registered {len(rows)} planned specifications")


def extra_sections(result, evaluations, ranks, neural_delta, arithmetic_delta, diagnostics):
    primary = result.results["headline"]
    control = evaluations[CONTROL].results["headline"]
    checks = {
        "mean": primary["mean"] > control["mean"],
        "sharpe": primary["sharpe_trade"] > control["sharpe_trade"],
        "positive_year_share": primary["years_positive"] >= control["years_positive"],
        "breakeven_alpha": primary["breakeven_alpha"] <= control["breakeven_alpha"],
        "policy_ci": neural_delta["ci90"][0] > 0,
    }
    rows = []
    for arm in ALL_ARMS:
        h = result.results["headline"] if arm == PRIMARY else evaluations[arm].results["headline"]
        r = ranks[arm]
        rows.append([
            arm, f"{h['n']:,}", f"{100*h['mean']:+.2f}%", f"{h['sharpe_trade']:.2f}",
            f"{100*h['cagr']:+.2f}%", f"{h['years_positive']}/{h['years_evaluated']}",
            f"{h['breakeven_alpha']:.3f}", f"{r['spearman']:+.3f}",
        ])
    def fmt(delta):
        lo, hi = delta["ci90"]
        return f"{100*delta['observed']:+.2f}% [{100*lo:+.2f}%, {100*hi:+.2f}%], P+={delta['p_gt_zero']:.1%}"
    return [
        {
            "title": "Corrected-exit verdict",
            "body": [
                f"**{'PRIMARY SUCCESS CRITERIA MET' if all(checks.values()) else 'PRIMARY DOES NOT CLEAR'}**.",
                "EXP-158 is superseded for this question because its raw post-print marks allowed negative value for a defined-risk debit.",
                f"Full neural minus neural control: {fmt(neural_delta)}.",
                f"Neural control minus current arithmetic gate: {fmt(arithmetic_delta)}.",
            ],
        },
        {
            "title": "Arithmetic gate and neural arms",
            "columns": ["arm", "selected", "mean", "Sharpe", "CAGR", "years+", "breakeven alpha", "rho"],
            "align": ["---"] + ["---:"] * 7,
            "rows": rows,
        },
        {
            "title": "Primary neural checks",
            "columns": ["check", "status"],
            "align": ["---", "---"],
            "rows": [[key, "PASS" if value else "FAIL"] for key, value in checks.items()],
        },
        {
            "title": "Current arithmetic benchmark",
            "body": [
                "This arm is the current DYN-SV gate, not a neural approximation: simulated expected PnL clears its trailing six-month top-20% monthly bar, relative spread is at most 25%, and market cap is at least $10B.",
                "Every neural arm is compared with the same conditional-exit candidates and full fill-alpha grid.",
            ],
        },
        {
            "title": "Fold counts",
            "columns": ["test year", "arm", "train", "scoreable", "selected", "cutoff"],
            "align": ["---:","---"] + ["---:"] * 4,
            "rows": [[
                str(fold["year"]), arm,
                "n/a" if fold["arms"][arm]["train_rows"] is None else str(fold["arms"][arm]["train_rows"]),
                str(fold["arms"][arm]["scoreable_test"]), str(fold["arms"][arm]["selected"]),
                f"{fold['arms'][arm]['cutoff']:.4f}" if np.isfinite(fold["arms"][arm]["cutoff"]) else "n/a",
            ] for fold in diagnostics for arm in ALL_ARMS],
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    specs = {arm: arm_spec(spec, arm) for arm in ALL_ARMS}
    if not args.no_ledger:
        plan(specs)
    dataset, priced = load_dataset()
    dataset = base.add_causal_analogs(dataset)
    scores, diagnostics = generate_scores(dataset, args.force_scores)
    ranks = rank_metrics(scores)
    neural_delta = bootstrap(scores, PRIMARY, CONTROL)
    arithmetic_delta = bootstrap(scores, CONTROL, ARITHMETIC)
    write_json(RESULTS / "ranking.json", ranks)
    write_json(RESULTS / "policy_bootstrap.json", {
        "primary_vs_neural_control": neural_delta,
        "neural_control_vs_arithmetic": arithmetic_delta,
    })
    ids = set(scores["event_id"].astype(str))
    priced["event_id"] = priced["event_id"].astype(str)
    trades = priced[priced["event_id"].isin(ids)].copy()
    spy = common.load_spy_daily()
    inputs = [CANDIDATES, ROOT / "data/features/panel.parquet", SCORE_CACHE, DIAGNOSTICS]
    evaluations = {}
    order = [arm for arm in ALL_ARMS if arm != PRIMARY] + [PRIMARY]
    for arm in order:
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        if arm == PRIMARY:
            extra = lambda result: extra_sections(
                result, evaluations, ranks, neural_delta, arithmetic_delta, diagnostics
            )
        elif arm == ARITHMETIC:
            extra = [{
                "title": "Current arithmetic rule",
                "body": ["Live DYN-SV expected-PnL trailing-bar rule, with its 25% spread and $10B market-cap guards, evaluated on validated conditional exits."],
            }]
        else:
            extra = [{
                "title": "Neural arm construction",
                "body": [f"Features ({len(base.ARM_FEATURES[arm])}): {', '.join(base.ARM_FEATURES[arm])}."],
            }]
        result = evaluate(
            specs[arm], trades, gate=base.PrecomputedGate(scores, arm).gate(),
            run_dir=run_dir, spy_daily=spy, tail_shock=base.tail_to_debit,
            input_files=inputs, extra_sections=extra,
        )
        evaluations[arm] = result
        if not args.no_ledger:
            lib.record_evaluation(run_dir, specs[arm], result.results)
        h = result.results["headline"]
        log(f"{arm}: n={h['n']:,}, mean={h['mean']:+.4f}, Sharpe={h['sharpe_trade']:.3f}, CAGR={h['cagr']:+.4f}")
    write_json(RESULTS / "comparison.json", {
        "ranking": ranks,
        "policy_bootstrap": {
            "primary_vs_neural_control": neural_delta,
            "neural_control_vs_arithmetic": arithmetic_delta,
        },
        "headline": {arm: evaluations[arm].results["headline"] for arm in ALL_ARMS},
    })
    log(f"Primary report: {evaluations[PRIMARY].report_path}")


if __name__ == "__main__":
    main()
