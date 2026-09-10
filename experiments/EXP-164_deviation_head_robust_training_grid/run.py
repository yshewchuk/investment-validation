#!/usr/bin/env python3
"""EXP-164: deviation-head grid crossing feature sets (lean/full) with
training techniques (quantile target, restart averaging, small batch),
MSE controls, fixed incumbent gate and expensive-first funding."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py"
MARGIN163 = ROOT / "experiments/EXP-163_two_headed_level_and_deviation_dyn_sv/margin163.py"
FIRST_TEST_YEAR = 2020
SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
MIN_FIT_ROWS = 500
INCUMBENT = "incumbent"
PRIMARY = "qt_lean"
TECHS = ("mse", "qt", "swa", "batch32")
FEATURE_SETS = ("lean", "full")
CELLS = ("mse_lean", "qt_lean", "swa_lean", "batch32_lean", "mse_full", "qt_full")
SWA_MAX_EPOCHS, SWA_BURST_LO, SWA_BURST_HI, SWA_SNAP_EVERY, SWA_SNAP_FROM = 300, 100, 160, 20, 120
POLICY = dict(start_equity=200_000.0, cap=0.66, target_share=0.25)
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate
from experiments import common, lib


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("exp164_base", SOURCE)
margin163 = load_module("exp164_margin163", MARGIN163)
FEATURE_ARMS = {"lean": base.BASE, "full": base.ARMS["nn_all_categories"]}


def log(message: str) -> None:
    print(f"[EXP-164 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


class SwaEnsemble:
    def __init__(self, scaler, nets):
        self.scaler, self.nets = scaler, nets

    def predict(self, X):
        Xs = self.scaler.transform(X)
        return np.mean([net.predict(Xs) for net in self.nets], axis=0)


def fit_mlp(X, y, seed, *, batch_size=200):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                     early_stopping=True, validation_fraction=0.15,
                     n_iter_no_change=50, batch_size=batch_size, random_state=seed),
    ).fit(X, y)
    estimator = model.named_steps["mlpregressor"]
    return model, {"n_iter": int(estimator.n_iter_), "loss": float(estimator.loss_),
                   "converged_before_cap": bool(estimator.n_iter_ < 800)}


def fit_swa(X, y, seed):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(Xs))
    n_val = max(1, int(0.15 * len(Xs)))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    net = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=False,
                       random_state=seed, max_iter=1)
    snapshots, val_hist, best, stale, epoch = [], [], np.inf, 0, 0
    for epoch in range(1, SWA_MAX_EPOCHS + 1):
        net.set_params(learning_rate_init=4e-3 if SWA_BURST_LO <= epoch < SWA_BURST_HI else 1e-3)
        net.partial_fit(Xs[tr_idx], y[tr_idx])
        if epoch % 10 == 0:
            v = float(np.mean((net.predict(Xs[val_idx]) - y[val_idx]) ** 2))
            val_hist.append(v)
            smoothed = float(np.mean(val_hist[-5:]))
            if smoothed < best - 1e-6:
                best, stale = smoothed, 0
            else:
                stale += 1
            if epoch >= SWA_SNAP_FROM and epoch % SWA_SNAP_EVERY == 0:
                snapshots.append(copy.deepcopy(net))
            if epoch >= SWA_BURST_HI and stale >= 6:
                break
    snapshots.append(copy.deepcopy(net))
    return SwaEnsemble(scaler, snapshots), {"epochs": epoch, "snapshots": len(snapshots),
                                            "best_val": best, "converged_before_cap": bool(epoch < SWA_MAX_EPOCHS)}


def fit_cell(X, y, tech, seed):
    from sklearn.preprocessing import QuantileTransformer
    if tech == "mse":
        return fit_mlp(X, y, seed)
    if tech == "batch32":
        return fit_mlp(X, y, seed, batch_size=32)
    if tech == "qt":
        qt = QuantileTransformer(output_distribution="normal",
                                 n_quantiles=min(1000, len(y))).fit(y.reshape(-1, 1))
        model, stats = fit_mlp(X, qt.transform(y.reshape(-1, 1)).ravel(), seed)
        return model, {**stats, "target_transform": "quantile_normal"}
    if tech == "swa":
        return fit_swa(X, y, seed)
    raise ValueError(tech)


def predict(models, X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict(X) for model in models], axis=0)


def within_event_spearman(frame: pd.DataFrame, score_col: str) -> float:
    rhos = []
    for _, g in frame.dropna(subset=[score_col, "pnl"]).groupby("event_id"):
        if len(g) >= 3:
            rhos.append(spearmanr(g[score_col], g["pnl"]).statistic)
    return float(np.nanmean(rhos)) if rhos else float("nan")


def cache_paths():
    return (RESULTS / "dev_scores.parquet", RESULTS / "choices.parquet", RESULTS / "fold_diagnostics.json")


def generate(dataset: pd.DataFrame, force: bool):
    dev_cache, choices_cache, diag_path = cache_paths()
    if all(p.exists() for p in (dev_cache, choices_cache, diag_path)) and not force:
        dev = pd.read_parquet(dev_cache)
        choices = pd.read_parquet(choices_cache)
        for frame in (dev, choices):
            frame["event_date"] = pd.to_datetime(frame["event_date"])
        return dev, choices, json.loads(diag_path.read_text())
    gate = base.incumbent_gate(dataset)
    event_mean = dataset.groupby("event_id")["pnl"].transform("mean")
    dataset = dataset.assign(dev_target=dataset["pnl"] - event_mean)
    dev_parts, choice_parts, diagnostics = [], [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train, test = dataset[dataset["year"] < year], dataset[dataset["year"] == year].copy()
        fold = {"year": int(year), "train": int(len(train)), "test": int(len(test)), "cells": {}}
        for cell in CELLS:
            tech, feats = cell.split("_", 1)
            features = FEATURE_ARMS[feats]
            test[f"{cell}"] = np.nan
            train_ok = np.isfinite(train[list(features) + ["dev_target"]].to_numpy(float)).all(1)
            fit = train.loc[train_ok]
            stats = {"fit": int(len(fit))}
            if len(fit) >= MIN_FIT_ROWS:
                models, seed_stats = [], []
                for seed in SEEDS:
                    model, one = fit_cell(fit[list(features)].to_numpy(float), fit["dev_target"].to_numpy(float), tech, seed)
                    models.append(model)
                    seed_stats.append({"seed": seed, **one})
                test_ok = np.isfinite(test[list(features)].to_numpy(float)).all(1)
                if test_ok.any():
                    test.loc[test_ok, cell] = predict(models, test.loc[test_ok, list(features)].to_numpy(float))
                stats.update({
                    "scoreable": int(test_ok.sum()),
                    "within_event_spearman": within_event_spearman(test, cell),
                    "seed_stats": seed_stats,
                    "mean_epochs": float(np.mean([s.get("n_iter", s.get("epochs", np.nan)) for s in seed_stats])),
                    "loss_sd": float(np.std([s.get("loss", s.get("best_val", np.nan)) for s in seed_stats])),
                })
                seed_preds = [m.predict(test.loc[test_ok, list(features)].to_numpy(float)) for m in models]
                if len(seed_preds) > 1:
                    pairs = [spearmanr(seed_preds[i], seed_preds[j]).statistic for i in range(len(seed_preds)) for j in range(i + 1, len(seed_preds))]
                    stats["seed_rank_agreement"] = float(np.nanmean(pairs))
            fold["cells"][cell] = stats
        piece = test[["candidate_id", "event_id", "ticker", "event_date", "strategy", "pnl"]].rename(columns={"pnl": "realized_pnl"}).reset_index(drop=True)
        for cell in CELLS:
            piece[cell] = test[cell].to_numpy()
        joined = test.reset_index(drop=True)
        truth = joined.sort_values(["event_id", "pnl", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "oracle_structure"})
        ev = test[["event_id", "ticker", "event_date"]].drop_duplicates("event_id").merge(gate, on="event_id", how="left").merge(truth, on="event_id", how="left")
        ev["traded_inc"] = ev["traded"].fillna(False).astype(bool)
        for cell in CELLS:
            choice = joined.dropna(subset=[cell]).sort_values(["event_id", cell, "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": cell})
            ev = ev.merge(choice, on="event_id", how="left")
        dev_parts.append(piece)
        choice_parts.append(ev)
        diagnostics.append(fold)
        rhos = ", ".join(f"{c}={fold['cells'][c].get('within_event_spearman', float('nan')):+.3f}" for c in CELLS)
        log(f"Fold {year}: {rhos}")
    dev = pd.concat(dev_parts, ignore_index=True).sort_values(["event_date", "candidate_id"])
    choices = pd.concat(choice_parts, ignore_index=True).sort_values(["event_date", "event_id"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(dev_cache, index=False)
    choices.to_parquet(choices_cache, index=False)
    write_json(diag_path, diagnostics)
    return dev, choices, diagnostics


def arm_metrics(choices: pd.DataFrame) -> dict:
    out = {}
    struct_cols = {INCUMBENT: "incumbent_structure", **{c: c for c in CELLS}}
    for arm, struct_col in struct_cols.items():
        rows = choices.dropna(subset=[struct_col])
        out[arm] = {
            "choice_coverage": int(len(rows)),
            "eligible": int(rows["traded_inc"].sum()),
            "oracle_hit": float((rows[struct_col] == rows["oracle_structure"]).mean()),
            "incumbent_structure_agreement": float((rows[struct_col] == rows["incumbent_structure"]).mean()),
        }
    return out


def cell_quality(dev: pd.DataFrame, choices: pd.DataFrame, diagnostics: list[dict]) -> dict:
    def vals(folds, key):
        return [np.nan if f.get(key) is None else f.get(key) for f in folds]

    out = {}
    for cell in CELLS:
        folds = [f["cells"][cell] for f in diagnostics]
        rhos = [np.nan if f.get("within_event_spearman") is None else f.get("within_event_spearman") for f in folds]
        out[cell] = {
            "within_event_spearman_mean": float(np.nanmean(rhos)) if np.isfinite(rhos).any() else float("nan"),
            "within_event_spearman_per_fold": rhos,
            "mean_epochs": float(np.nanmean(vals(folds, "mean_epochs"))) if np.isfinite(vals(folds, "mean_epochs")).any() else float("nan"),
            "loss_sd": float(np.nanmean(vals(folds, "loss_sd"))) if np.isfinite(vals(folds, "loss_sd")).any() else float("nan"),
            "seed_rank_agreement": float(np.nanmean(vals(folds, "seed_rank_agreement"))) if np.isfinite(vals(folds, "seed_rank_agreement")).any() else float("nan"),
        }
    return out


def selected_mid(mid: pd.DataFrame, choices: pd.DataFrame, arm: str) -> pd.DataFrame:
    struct_col = "incumbent_structure" if arm == INCUMBENT else arm
    picked = choices[["event_id", "traded_inc", struct_col]].dropna(subset=[struct_col])
    picked = picked.rename(columns={"traded_inc": "traded", struct_col: "picked_structure"})
    out = mid.merge(picked, on="event_id", how="inner")
    out = out[(out["strategy"] == out["picked_structure"]) & out["traded"].fillna(False)].copy()
    out["secured_per_contract"] = out["legs"].map(margin163.secured_per_contract)
    return out.sort_values(["entry_date", "event_id"])


def account_score(book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    funded = margin163.simulate(book, start_equity=POLICY["start_equity"], cap=POLICY["cap"],
                                target_share=POLICY["target_share"], priority_col=None)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0}
    start = POLICY["start_equity"]
    final = float(funded.attrs["final_equity"])
    span = (pd.to_datetime(funded["exit_date"]).max() - pd.to_datetime(funded["entry_date"]).min()).days
    years = span / 365.25
    unfunded = funded[~funded["funded"]]
    return funded, {
        "wanted": int(len(funded)), "funded": int(len(placed)),
        "funded_pct": float(placed.shape[0] / len(funded)),
        "peak_concurrency": int(funded["concurrency"].max()),
        "final_equity": final, "profit_usd": final - start,
        "cagr": float((final / start) ** (1 / years) - 1) if years > 0 else float("nan"),
        "max_secured_usd": float((funded["secured_before"] + funded["contracts"] * funded["secured_per_contract"]).max()),
        "defined_risk_failures": int((placed["ret"] < -1.0 - 1e-8).sum()),
        "unfunded_wanted": int(len(unfunded)),
        "unfunded_mean_realized_pnl": float(unfunded["pnl"].mean()) if len(unfunded) else 0.0,
    }


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    if arm != PRIMARY:
        out["primary_spec"]["evaluated_arm"] = arm
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def plan(spec: dict) -> None:
    digest, ledger = lib.spec_hash(spec), lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{"id": spec["id"], "spec_hash": digest, "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}])
        log("Registered primary specification in LEDGER.csv")


def report_sections(accounts: dict, metrics: dict, quality: dict, choices: pd.DataFrame) -> list[dict]:
    rows = []
    for arm in (INCUMBENT, *CELLS):
        a, e, q = accounts.get(arm, {}), metrics[arm], quality.get(arm, {})
        epochs = f"{q.get('mean_epochs', float('nan')):.0f}" if arm != INCUMBENT else "n/a"
        rho = f"{q.get('within_event_spearman_mean', float('nan')):+.3f}" if arm != INCUMBENT else "n/a"
        agree = f"{q.get('seed_rank_agreement', float('nan')):.2f}" if arm != INCUMBENT else "n/a"
        rows.append([arm, f"{a.get('wanted', 0):,}", f"{a.get('funded', 0):,}",
                     f"${a.get('final_equity', float('nan')):,.0f}", f"{a.get('cagr', float('nan')):.2%}",
                     f"{e['oracle_hit']:.1%}", rho, epochs, agree, f"{a.get('unfunded_wanted', 0):,}"])
    primary, inc = accounts.get(PRIMARY, {}), accounts.get(INCUMBENT, {})
    checks = {
        "qt_lean_beats_incumbent": primary.get("final_equity", -np.inf) > inc.get("final_equity", np.inf),
        "qt_lean_beats_mse_full": primary.get("final_equity", -np.inf) > accounts.get("mse_full", {}).get("final_equity", np.inf),
        "a_technique_beats_its_control": any(
            accounts.get(c, {}).get("final_equity", -np.inf) > accounts.get(ctrl, {}).get("final_equity", np.inf)
            for c, ctrl in (("qt_lean", "mse_lean"), ("qt_full", "mse_full"), ("swa_lean", "mse_lean"), ("batch32_lean", "mse_lean"))
        ),
        "no_defined_risk_failure": all(a.get("defined_risk_failures", 1) == 0 for a in accounts.values()),
    }
    return [
        {"title": "What varies and what does not", "body": [
            "Every cell uses the EXP-163 deviation target (event-demeaned realized PnL), the fixed incumbent gate, and the EXP-162/163 cash-secured funding with largest-secured-first ordering. Only the head varies: feature set (lean vs full) crossed with training technique (MSE control, quantile-Gaussian target, restart-snapshot averaging, batch 32).",
            "The mse_full cell is the EXP-163 dev_only configuration and doubles as the replication check. L1 loss is unavailable in this sklearn (squared_error or poisson only); the quantile target carries the robustness role.",
        ]},
        {"title": "Cell quality and cash-secured account", "note": "$200,000 start, 66% secured cap, 25% headroom target. epochs = mean iterations actually run per fit (early-stopped fits count their stopping epoch); seed agree = mean pairwise Spearman between seed predictions on test candidates.", "columns": ["arm", "wanted", "funded", "final equity", "CAGR", "oracle hit", "within-event rho", "epochs", "seed agree", "unfunded"], "align": ["---"] + ["---:"] * 9, "rows": rows},
        {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()]},
        {"title": "Selection funnel", "body": [f"Annual-OOS complete five-menu events: {len(choices):,}; incumbent-gate eligible: {int(choices['traded_inc'].sum()):,}."]},
    ]


def main() -> None:
    global RESULTS, SEEDS, FIRST_TEST_YEAR
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        RESULTS = HERE / "results_smoke"
        SEEDS = (SEEDS[0],)
        FIRST_TEST_YEAR = 2025
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger and not args.smoke:
        plan(spec)
    mid, raw = base.load_data()
    dataset = base.add_causal_analogs(mid)
    dev, choices, diagnostics = generate(dataset, args.force_scores)
    metrics, quality = arm_metrics(choices), cell_quality(dev, choices, diagnostics)
    write_json(RESULTS / "event_choice_metrics.json", metrics)
    write_json(RESULTS / "cell_quality.json", quality)
    write_json(RESULTS / "funding_policy.json", POLICY)
    spy = common.load_spy_daily()
    accounts, funded_books = {}, {}
    for arm in (INCUMBENT, *CELLS):
        book = selected_mid(mid, choices, arm)
        funded, account = account_score(book)
        accounts[arm] = account
        funded_books[arm] = base.selected_all_alphas(raw, funded)
        log(f"{arm}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} final=${account.get('final_equity', float('nan')):,.0f}")
    evaluations = {}
    for arm in (INCUMBENT, *CELLS):
        cell, run_dir = arm_spec(spec, arm), HERE if arm == PRIMARY else HERE / "arms" / arm
        extra = (lambda result: report_sections(accounts, metrics, quality, choices)) if arm == PRIMARY else [
            {"title": "Arm accounting", "body": [f"Cell: {arm} (fixed incumbent gate, expensive-first funding). Final equity ${accounts[arm].get('final_equity', float('nan')):,.0f}; funded {accounts[arm].get('funded', 0):,}/{accounts[arm].get('wanted', 0):,}."]}
        ]
        result = evaluate(cell, funded_books[arm], gate=None, run_dir=run_dir, spy_daily=spy,
                          input_files=[base.CANDIDATES, ROOT / "data/features/panel.parquet", *cache_paths()],
                          extra_sections=extra, write_report=True)
        evaluations[arm] = result.results
        log(f"{arm}: evaluated, mean={result.results['headline'].get('mean', float('nan')):+.3f}")
    write_json(RESULTS / "comparison.json", {"policy": POLICY, "event_choice_metrics": metrics,
                                             "cell_quality": quality, "account_metrics": accounts,
                                             "evaluation_headlines": {k: v["headline"] for k, v in evaluations.items()}})
    if not args.no_ledger and not args.smoke:
        lib.record_evaluation(HERE, spec, evaluations[PRIMARY])
    print(f"[EXP-164] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
