#!/usr/bin/env python3
"""EXP-162: architecture-only sweep of the causal DYN-SV structure chooser."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py"
SCORE_CACHE = RESULTS / "oos_candidate_scores.parquet"
CHOICES_CACHE = RESULTS / "oos_choices.parquet"
DIAGNOSTICS = RESULTS / "fold_diagnostics.json"
FIRST_TEST_YEAR = 2020
SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
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


base = load_module("exp162_base", SOURCE)
FEATURES = base.ARMS["nn_all_categories"]
INCUMBENT = "incumbent_simulated_pnl"
PRIMARY = "deep_128_64_32"
ARCHITECTURES = {
    "small_16": {"layers": (16,), "max_iter": 800, "patience": 50},
    "prior_64_32": {"layers": (64, 32), "max_iter": 800, "patience": 50},
    "wide_128_64": {"layers": (128, 64), "max_iter": 800, "patience": 50},
    PRIMARY: {"layers": (128, 64, 32), "max_iter": 1200, "patience": 80},
}


def log(message: str) -> None:
    print(f"[EXP-162 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def fit_architecture(X: np.ndarray, y: np.ndarray, config: dict):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models, diagnostics = [], []
    for seed in SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=config["layers"], max_iter=config["max_iter"],
                early_stopping=True, validation_fraction=0.15,
                n_iter_no_change=config["patience"], random_state=seed,
            ),
        ).fit(X, y)
        estimator = model.named_steps["mlpregressor"]
        loss_curve = [float(value) for value in estimator.loss_curve_]
        validation_curve = [float(value) for value in getattr(estimator, "validation_scores_", [])]
        best_iteration = int(np.nanargmax(validation_curve) + 1) if validation_curve else None
        diagnostics.append({
            "seed": seed, "n_iter": int(estimator.n_iter_),
            "loss": float(estimator.loss_),
            "best_validation": float(getattr(estimator, "best_validation_score_", np.nan)),
            "converged_before_cap": bool(estimator.n_iter_ < config["max_iter"]),
            "best_validation_iteration": best_iteration,
            "initial_loss": loss_curve[0] if loss_curve else None,
            "loss_curve": loss_curve,
            "validation_curve": validation_curve,
        })
        models.append(model)
    return models, diagnostics


def predict(models, X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict(X) for model in models], axis=0)


def generate_choices(dataset: pd.DataFrame, force: bool):
    if SCORE_CACHE.exists() and CHOICES_CACHE.exists() and DIAGNOSTICS.exists() and not force:
        scores, choices = pd.read_parquet(SCORE_CACHE), pd.read_parquet(CHOICES_CACHE)
        scores["event_date"] = pd.to_datetime(scores["event_date"])
        choices["event_date"] = pd.to_datetime(choices["event_date"])
        return scores, choices, json.loads(DIAGNOSTICS.read_text())
    gate = base.incumbent_gate(dataset)
    score_parts, choice_parts, diagnostics = [], [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train, test = dataset[dataset["year"] < year], dataset[dataset["year"] == year].copy()
        piece = test[["candidate_id", "event_id", "ticker", "event_date", "strategy", "pnl"]].rename(columns={"pnl": "realized_pnl"}).reset_index(drop=True)
        train_ok = np.isfinite(train[list(FEATURES) + ["pnl"]].to_numpy(float)).all(1)
        test_ok = np.isfinite(test[list(FEATURES)].to_numpy(float)).all(1)
        fit = train.loc[train_ok]
        fold = {"year": int(year), "train": int(len(fit)), "test": int(len(test)), "architectures": {}}
        for name, config in ARCHITECTURES.items():
            values = np.full(len(test), np.nan)
            seed_stats = []
            stability = {}
            if len(fit) >= 500 and test_ok.any():
                models, seed_stats = fit_architecture(fit[list(FEATURES)].to_numpy(float), fit["pnl"].to_numpy(float), config)
                seed_predictions = np.vstack([model.predict(test.loc[test_ok, list(FEATURES)].to_numpy(float)) for model in models])
                values[test_ok] = seed_predictions.mean(axis=0)
                pairwise = [spearmanr(seed_predictions[i], seed_predictions[j]).statistic for i in range(len(models)) for j in range(i)]
                stability = {"oos_seed_prediction_sd": float(seed_predictions.std(axis=0, ddof=1).mean()), "oos_seed_rank_agreement": float(np.nanmean(pairwise))}
            piece[name] = values
            fold["architectures"][name] = {"scoreable": int(test_ok.sum()), "seed_stats": seed_stats, **stability}
        joined = pd.concat([test.reset_index(drop=True), piece[list(ARCHITECTURES)].reset_index(drop=True)], axis=1)
        truth = joined.sort_values(["event_id", "pnl", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "oracle_structure"})
        incumbent = joined.sort_values(["event_id", "exp_pnl_sim", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": INCUMBENT})
        event = test[["event_id", "ticker", "event_date"]].drop_duplicates("event_id").merge(gate, on="event_id", how="left").merge(truth, on="event_id", how="left").merge(incumbent, on="event_id", how="left")
        for name in ARCHITECTURES:
            chosen = joined.dropna(subset=[name]).sort_values(["event_id", name, "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": name})
            event = event.merge(chosen, on="event_id", how="left")
        score_parts.append(piece)
        choice_parts.append(event)
        diagnostics.append(fold)
        log(f"Fold {year}: train={len(fit):,}, test={len(test):,}; architectures fitted")
    scores = pd.concat(score_parts, ignore_index=True).sort_values(["event_date", "candidate_id"])
    choices = pd.concat(choice_parts, ignore_index=True).sort_values(["event_date", "event_id"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    choices.to_parquet(CHOICES_CACHE, index=False)
    write_json(DIAGNOSTICS, diagnostics)
    return scores, choices, diagnostics


def rank_metrics(scores: pd.DataFrame) -> dict:
    out = {}
    for arm in ARCHITECTURES:
        rows = scores[[arm, "realized_pnl"]].dropna()
        decile = pd.qcut(rows[arm], 10, labels=False, duplicates="drop")
        means = rows.groupby(decile)["realized_pnl"].mean()
        out[arm] = {"n": int(len(rows)), "spearman": float(spearmanr(rows[arm], rows["realized_pnl"]).statistic), "top_bottom_pnl": float(means.iloc[-1] - means.iloc[0])}
    return out


def event_metrics(choices: pd.DataFrame) -> dict:
    out = {}
    for arm in (INCUMBENT, *ARCHITECTURES):
        rows = choices.dropna(subset=[arm])
        out[arm] = {"eligible": int(rows["traded"].sum()), "coverage": int(len(rows)), "oracle_hit": float((rows[arm] == rows["oracle_structure"]).mean()), "incumbent_agreement": float((rows[arm] == rows[INCUMBENT]).mean())}
    return out


def convergence(diagnostics: list[dict]) -> dict:
    out = {}
    for name, config in ARCHITECTURES.items():
        rows = [s for fold in diagnostics for s in fold["architectures"][name]["seed_stats"]]
        fold_stats = [fold["architectures"][name] for fold in diagnostics]
        out[name] = {"layers": list(config["layers"]), "max_iter": config["max_iter"], "fits": len(rows), "mean_n_iter": float(np.mean([r["n_iter"] for r in rows])) if rows else None, "max_n_iter": int(max([r["n_iter"] for r in rows], default=0)), "fraction_before_cap": float(np.mean([r["converged_before_cap"] for r in rows])) if rows else None, "mean_loss": float(np.mean([r["loss"] for r in rows])) if rows else None, "mean_loss_reduction": float(np.mean([r["initial_loss"] - r["loss"] for r in rows])) if rows else None, "mean_best_validation": float(np.nanmean([r["best_validation"] for r in rows])) if rows else None, "mean_oos_seed_prediction_sd": float(np.nanmean([s.get("oos_seed_prediction_sd", np.nan) for s in fold_stats])), "mean_oos_seed_rank_agreement": float(np.nanmean([s.get("oos_seed_rank_agreement", np.nan) for s in fold_stats]))}
    return out


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
        lib.ledger_append([{"id": spec["id"], "spec_hash": digest, "date": "2026-09-08", "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}])
        log("Registered primary specification")


def report_sections(accounts, events, ranks, convergence_info, choices):
    rows = []
    for arm in (INCUMBENT, *ARCHITECTURES):
        account, event, conv = accounts.get(arm, {}), events[arm], convergence_info.get(arm, {})
        rows.append([arm, "incumbent" if arm == INCUMBENT else "x".join(map(str, conv["layers"])), "n/a" if arm == INCUMBENT else f"{conv['mean_n_iter']:.0f}", "n/a" if arm == INCUMBENT else f"{conv['fraction_before_cap']:.0%}", f"{event['oracle_hit']:.1%}", f"{account.get('funded', 0):,}", f"${account.get('final_equity', float('nan')):,.0f}", f"{account.get('cagr', float('nan')):.2%}", "n/a" if arm == INCUMBENT else f"{ranks[arm]['spearman']:+.3f}"])
    primary, prior = accounts.get(PRIMARY, {}), accounts.get("prior_64_32", {})
    checks = {"oracle_hit_vs_incumbent": events[PRIMARY]["oracle_hit"] > events[INCUMBENT]["oracle_hit"], "final_equity_vs_prior": primary.get("final_equity", -np.inf) > prior.get("final_equity", np.inf), "final_equity_vs_incumbent": primary.get("final_equity", -np.inf) > accounts.get(INCUMBENT, {}).get("final_equity", np.inf), "deep_fits_stop_before_cap": convergence_info[PRIMARY]["fraction_before_cap"] == 1.0}
    return [{"title": "Architecture-only sweep", "body": ["All neural rows use the same full input set, yearly expanding OOS protocol, same five seeds, same five offered structures, incumbent eligibility, conditional exits, and cash-secured funding. Only the MLP layer sizes and patience vary."]}, {"title": "Convergence and cash-secured account", "note": "Start $200,000; 66% secured-put cap of current equity; 25% headroom target; expensive-first same-day ordering. A fit stopping before its maximum iteration cap means its validation early-stopping rule fired, not a proof of a global minimum.", "columns": ["arm", "layers", "mean iter", "early stop", "oracle hit", "funded", "final equity", "CAGR", "candidate rho"], "align": ["---"] + ["---:"] * 8, "rows": rows}, {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[k, "PASS" if v else "FAIL"] for k, v in checks.items()]}, {"title": "Selection funnel", "body": [f"Annual-OOS five-menu events: {len(choices):,}; fixed incumbent-gate eligible events: {int(choices['traded'].sum()):,}."]}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger:
        plan(spec)
    mid, raw = base.load_data()
    dataset = base.add_causal_analogs(mid)
    scores, choices, diagnostics = generate_choices(dataset, args.force_scores)
    ranks, events, conv = rank_metrics(scores), event_metrics(choices), convergence(diagnostics)
    write_json(RESULTS / "candidate_rank_metrics.json", ranks)
    write_json(RESULTS / "event_choice_metrics.json", events)
    write_json(RESULTS / "convergence.json", conv)
    spy, accounts, funded_books, evaluations = common.load_spy_daily(), {}, {}, {}
    for arm in (INCUMBENT, *ARCHITECTURES):
        book = base.selected_mid(mid, choices, arm)
        funded, accounts[arm] = base.account_score(book)
        funded_books[arm] = base.selected_all_alphas(raw, funded)
    for arm in (INCUMBENT, *ARCHITECTURES):
        cell, run_dir = arm_spec(spec, arm), HERE if arm == PRIMARY else HERE / "arms" / arm
        extra = (lambda result: report_sections(accounts, events, ranks, conv, choices)) if arm == PRIMARY else [{"title": "Architecture arm", "body": [f"Architecture: {ARCHITECTURES.get(arm, 'incumbent simulated-PnL resolver')}; fixed identical eligibility and cash-secured funding."]}]
        result = evaluate(cell, funded_books[arm], gate=None, run_dir=run_dir, spy_daily=spy, input_files=[base.CANDIDATES, ROOT / "data/features/panel.parquet", SCORE_CACHE, CHOICES_CACHE], extra_sections=extra, write_report=True)
        evaluations[arm] = result.results
        log(f"{arm}: funded={accounts[arm].get('funded',0):,}, final=${accounts[arm].get('final_equity',float('nan')):,.0f}")
    write_json(RESULTS / "comparison.json", {"architectures": ARCHITECTURES, "convergence": conv, "candidate_rank_metrics": ranks, "event_choice_metrics": events, "account_metrics": accounts, "headlines": {k: v["headline"] for k, v in evaluations.items()}})
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, evaluations[PRIMARY])
    print(f"[EXP-162] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
