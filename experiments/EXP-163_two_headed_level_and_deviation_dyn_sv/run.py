#!/usr/bin/env python3
"""EXP-163: two-headed decomposition of the DYN-SV compound decision.

Level head (one row per event) drives the trade gate and the funding queue;
deviation head (one row per candidate, event-demeaned target) drives the
structure argmax. Arms isolate each substitution against the incumbent."""
from __future__ import annotations

import argparse
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
FIRST_TEST_YEAR = 2020
SEEDS = (20260908, 20260909, 20260910, 20260911, 20260912)
MIN_FIT_ROWS = 500
INCUMBENT = "incumbent"
PRIMARY = "two_head_order"
ARM_STRUCT = {
    "incumbent": "incumbent_structure",
    "dev_only": "dev_structure",
    "level_gate": "incumbent_structure",
    "two_head": "dev_structure",
    "two_head_order": "dev_structure",
}
ARM_TRADED = {
    "incumbent": "traded_inc",
    "dev_only": "traded_inc",
    "level_gate": "traded_level",
    "two_head": "traded_level",
    "two_head_order": "traded_level",
}
ARM_ORDER = {
    "incumbent": "expensive-first",
    "dev_only": "expensive-first",
    "level_gate": "expensive-first",
    "two_head": "expensive-first",
    "two_head_order": "expected per secured dollar",
}
ARM_GATE = {
    "incumbent": "incumbent trailing rule",
    "dev_only": "incumbent trailing rule",
    "level_gate": "learned level (rate matched)",
    "two_head": "learned level (rate matched)",
    "two_head_order": "learned level (rate matched)",
}
ARM_SELECTOR = {
    "incumbent": "argmax exp_pnl_sim",
    "dev_only": "argmax deviation head",
    "level_gate": "argmax exp_pnl_sim",
    "two_head": "argmax deviation head",
    "two_head_order": "argmax deviation head",
}
POLICY = dict(start_equity=200_000.0, cap=0.66, target_share=0.25)
PRIORITY_COL = "expected_per_secured"
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate
from experiments import common, lib
import margin163


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_module("exp163_base", SOURCE)
FEATURES = base.ARMS["nn_all_categories"]
LEVEL_CONSTANT = (
    "pred_abs_move", "pred_abs_move_sd", "dte_entry", "n_admissible", "mcap_log",
    "or_implied", "or_rvol30", "mean_prior_abs_move", "ema12r_abs", "signed_streak",
    "mean_prior_or_implied", "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252",
    "spy_vol5", "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252",
)
LEVEL_AGGREGATE = (
    "exp_pnl_sim_mean", "exp_pnl_sim_max", "rel_spread_mean", "entry_cost_pct_mean",
    "analog_mean_mean", "width_over_forecast_mean",
)
LEVEL_FEATURES = LEVEL_CONSTANT + LEVEL_AGGREGATE


def log(message: str) -> None:
    print(f"[EXP-163 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def build_events(dataset: pd.DataFrame) -> pd.DataFrame:
    first = dataset.sort_values(["entry_date", "event_id", "strategy"], kind="stable").drop_duplicates("event_id")
    grouped = dataset.groupby("event_id", sort=False).agg(
        level_target=("pnl", "mean"),
        exp_pnl_sim_mean=("exp_pnl_sim", "mean"),
        exp_pnl_sim_max=("exp_pnl_sim", "max"),
        rel_spread_mean=("rel_spread", "mean"),
        entry_cost_pct_mean=("entry_cost_pct", "mean"),
        analog_mean_mean=("analog_mean", "mean"),
        width_over_forecast_mean=("width_over_forecast", "mean"),
        mcap_usd=("mcap_usd", "first"),
    )
    events = first[["event_id", "ticker", "event_date", "year", *LEVEL_CONSTANT]].merge(
        grouped, on="event_id", how="inner", validate="one_to_one"
    )
    return events


def fit_head(X: np.ndarray, y: np.ndarray):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models, stats = [], []
    for seed in SEEDS:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=800,
                         early_stopping=True, validation_fraction=0.15,
                         n_iter_no_change=50, random_state=seed),
        ).fit(X, y)
        estimator = model.named_steps["mlpregressor"]
        stats.append({"seed": seed, "n_iter": int(estimator.n_iter_), "loss": float(estimator.loss_),
                     "converged_before_cap": bool(estimator.n_iter_ < 800)})
        models.append(model)
    return models, stats


def predict(models, X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict(X) for model in models], axis=0)


def within_event_spearman(frame: pd.DataFrame, score_col: str) -> float:
    rhos = []
    for _, g in frame.dropna(subset=[score_col, "pnl"]).groupby("event_id"):
        if len(g) >= 3:
            rhos.append(spearmanr(g[score_col], g["pnl"]).statistic)
    return float(np.nanmean(rhos)) if rhos else float("nan")


def cache_paths():
    return (RESULTS / "dev_scores.parquet", RESULTS / "level_scores.parquet",
            RESULTS / "choices.parquet", RESULTS / "fold_diagnostics.json")


def generate(dataset: pd.DataFrame, events: pd.DataFrame, force: bool):
    dev_cache, level_cache, choices_cache, diag_path = cache_paths()
    if all(p.exists() for p in (dev_cache, level_cache, choices_cache, diag_path)) and not force:
        dev = pd.read_parquet(dev_cache)
        level = pd.read_parquet(level_cache)
        choices = pd.read_parquet(choices_cache)
        for frame in (dev, level, choices):
            frame["event_date"] = pd.to_datetime(frame["event_date"])
        return dev, level, choices, json.loads(diag_path.read_text())
    gate = base.incumbent_gate(dataset)
    event_mean = dataset.groupby("event_id")["pnl"].transform("mean")
    dataset = dataset.assign(dev_target=dataset["pnl"] - event_mean)
    dev_parts, level_parts, choice_parts, diagnostics = [], [], [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train, test = dataset[dataset["year"] < year], dataset[dataset["year"] == year].copy()
        ev_train, ev_test = events[events["year"] < year], events[events["year"] == year].copy()
        fold = {"year": int(year), "train_candidates": int(len(train)), "train_events": int(len(ev_train)),
                "test_candidates": int(len(test)), "test_events": int(len(ev_test))}
        test["dev_pred"] = np.nan
        train_ok = np.isfinite(train[list(FEATURES) + ["dev_target"]].to_numpy(float)).all(1)
        fit = train.loc[train_ok]
        if len(fit) >= MIN_FIT_ROWS:
            models, dev_stats = fit_head(fit[list(FEATURES)].to_numpy(float), fit["dev_target"].to_numpy(float))
            test_ok = np.isfinite(test[list(FEATURES)].to_numpy(float)).all(1)
            if test_ok.any():
                test.loc[test_ok, "dev_pred"] = predict(models, test.loc[test_ok, list(FEATURES)].to_numpy(float))
            fold["dev"] = {"fit": int(len(fit)), "scoreable": int(test_ok.sum()),
                           "within_event_spearman": within_event_spearman(test, "dev_pred"), "seed_stats": dev_stats}
        ev_test["level_pred"] = np.nan
        lvl_ok = np.isfinite(ev_train[list(LEVEL_FEATURES) + ["level_target"]].to_numpy(float)).all(1)
        lvl_fit = ev_train.loc[lvl_ok]
        threshold = np.inf
        if len(lvl_fit) >= MIN_FIT_ROWS:
            lvl_models, lvl_stats = fit_head(lvl_fit[list(LEVEL_FEATURES)].to_numpy(float), lvl_fit["level_target"].to_numpy(float))
            ev_test_ok = np.isfinite(ev_test[list(LEVEL_FEATURES)].to_numpy(float)).all(1)
            if ev_test_ok.any():
                ev_test.loc[ev_test_ok, "level_pred"] = predict(lvl_models, ev_test.loc[ev_test_ok, list(LEVEL_FEATURES)].to_numpy(float))
            inc_traded = gate.set_index("event_id").reindex(ev_train["event_id"])["traded"].fillna(False).to_numpy(bool)
            rate = float(inc_traded.mean())
            if rate > 0 and lvl_ok.any():
                train_levels = predict(lvl_models, ev_train.loc[lvl_ok, list(LEVEL_FEATURES)].to_numpy(float))
                threshold = float(np.quantile(train_levels, 1.0 - rate))
            fold["level"] = {"fit": int(len(lvl_fit)), "scoreable": int(ev_test_ok.sum()),
                             "incumbent_train_gate_rate": rate, "threshold": threshold, "seed_stats": lvl_stats}
            ok = ev_test["level_pred"].notna() & np.isfinite(ev_test["level_target"])
            if ok.any():
                fold["level"]["oos_spearman"] = float(spearmanr(ev_test.loc[ok, "level_pred"], ev_test.loc[ok, "level_target"]).statistic)
        piece = test[["candidate_id", "event_id", "ticker", "event_date", "strategy", "pnl"]].rename(columns={"pnl": "realized_pnl"}).reset_index(drop=True)
        piece["dev_pred"] = test["dev_pred"].to_numpy()
        joined = test.reset_index(drop=True)
        truth = joined.sort_values(["event_id", "pnl", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "oracle_structure"})
        incumbent = joined.sort_values(["event_id", "exp_pnl_sim", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "gate_structure"})
        dev_choice = joined.dropna(subset=["dev_pred"]).sort_values(["event_id", "dev_pred", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "dev_structure"})
        inc_pick = joined.sort_values(["event_id", "exp_pnl_sim", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id").set_index("event_id")
        dev_pick = joined.dropna(subset=["dev_pred"]).sort_values(["event_id", "dev_pred", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id").set_index("event_id")
        lvl = ev_test[["event_id", "level_pred"]].set_index("event_id")
        ev = test[["event_id", "ticker", "event_date"]].drop_duplicates("event_id").merge(gate, on="event_id", how="left")
        ev = ev.merge(truth, on="event_id", how="left").merge(dev_choice, on="event_id", how="left").merge(lvl, on="event_id", how="left")
        ev["level_threshold"] = threshold
        ev["rel_spread_inc"] = inc_pick["rel_spread"].reindex(ev["event_id"]).to_numpy()
        ev["rel_spread_dev"] = dev_pick["rel_spread"].reindex(ev["event_id"]).to_numpy()
        ev["mcap_usd"] = inc_pick["mcap_usd"].reindex(ev["event_id"]).to_numpy()
        ev["traded_inc"] = ev["traded"].fillna(False).astype(bool)
        hygiene_inc = (ev["rel_spread_inc"] <= 0.25) & (ev["mcap_usd"] >= 10e9)
        hygiene_dev = (ev["rel_spread_dev"] <= 0.25) & (ev["mcap_usd"] >= 10e9)
        level_ok = ev["level_pred"].notna() & (ev["level_pred"] >= ev["level_threshold"])
        ev["traded_level"] = (level_ok & hygiene_inc).fillna(False).astype(bool)
        ev["traded_level_dev"] = (level_ok & hygiene_dev).fillna(False).astype(bool)
        dev_parts.append(piece)
        level_parts.append(ev_test[["event_id", "ticker", "event_date", "level_pred", "level_target"]])
        choice_parts.append(ev)
        diagnostics.append(fold)
        dev_rho = fold.get("dev", {}).get("within_event_spearman", float("nan"))
        lvl_rho = fold.get("level", {}).get("oos_spearman", float("nan"))
        log(f"Fold {year}: dev within-event rho={dev_rho:+.3f}, level OOS rho={lvl_rho:+.3f}, threshold={threshold:.3f}")
    dev = pd.concat(dev_parts, ignore_index=True).sort_values(["event_date", "candidate_id"])
    level = pd.concat(level_parts, ignore_index=True).sort_values(["event_date", "event_id"])
    choices = pd.concat(choice_parts, ignore_index=True).sort_values(["event_date", "event_id"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(dev_cache, index=False)
    level.to_parquet(level_cache, index=False)
    choices.to_parquet(choices_cache, index=False)
    write_json(diag_path, diagnostics)
    return dev, level, choices, diagnostics


def arm_metrics(choices: pd.DataFrame) -> dict:
    out = {}
    for arm, struct_col in ARM_STRUCT.items():
        rows = choices.dropna(subset=[struct_col])
        traded = ARM_TRADED[arm]
        gate_col = "traded_level_dev" if arm in ("two_head", "two_head_order") else traded
        out[arm] = {
            "choice_coverage": int(len(rows)),
            "eligible": int(rows[gate_col].sum()),
            "oracle_hit": float((rows[struct_col] == rows["oracle_structure"]).mean()),
            "incumbent_structure_agreement": float((rows[struct_col] == rows["incumbent_structure"]).mean()),
        }
    return out


def head_quality(dev: pd.DataFrame, level: pd.DataFrame, diagnostics: list[dict]) -> dict:
    dev_rho = [f.get("dev", {}).get("within_event_spearman") for f in diagnostics]
    lvl_rho = [f.get("level", {}).get("oos_spearman") for f in diagnostics]
    ok = level.dropna(subset=["level_pred", "level_target"])
    return {
        "deviation_within_event_spearman": {"per_fold": dev_rho, "mean": float(np.nanmean(dev_rho))},
        "level_oos_spearman": {"per_fold": lvl_rho, "mean": float(np.nanmean(lvl_rho)),
                               "n": int(len(ok))},
    }


def selected_mid_arm(mid: pd.DataFrame, choices: pd.DataFrame, arm: str, dev: pd.DataFrame) -> pd.DataFrame:
    struct_col, gate_col = ARM_STRUCT[arm], ARM_TRADED[arm]
    if arm in ("two_head", "two_head_order"):
        gate_col = "traded_level_dev"
    picked = choices[["event_id", gate_col, struct_col]].dropna(subset=[struct_col])
    picked = picked.rename(columns={gate_col: "traded", struct_col: "picked_structure"})
    out = mid.merge(picked, on="event_id", how="inner")
    out = out[(out["strategy"] == out["picked_structure"]) & out["traded"].fillna(False)].copy()
    out["secured_per_contract"] = out["legs"].map(margin163.secured_per_contract)
    if arm == "two_head_order":
        dev_at_choice = dev.sort_values(["event_id", "dev_pred"], ascending=[True, False], kind="stable").drop_duplicates("event_id")[["event_id", "dev_pred"]]
        lvl = choices.set_index("event_id")["level_pred"]
        out = out.merge(dev_at_choice.rename(columns={"dev_pred": "dev_pred_at_choice"}), on="event_id", how="left")
        out["expected_per_secured"] = (out["event_id"].map(lvl) + out["dev_pred_at_choice"]) * 100.0 / out["secured_per_contract"]
    return out.sort_values(["entry_date", "event_id"])


def account_score(book: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, dict]:
    priority = PRIORITY_COL if arm == "two_head_order" else None
    funded = margin163.simulate(book, start_equity=POLICY["start_equity"], cap=POLICY["cap"],
                                target_share=POLICY["target_share"], priority_col=priority)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0, "unfunded_wanted": int(len(funded))}
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
    for arm in ARM_STRUCT:
        a, e = accounts.get(arm, {}), metrics[arm]
        rows.append([arm, ARM_GATE[arm], ARM_SELECTOR[arm], ARM_ORDER[arm], f"{a.get('eligible', e['eligible']):,}",
                     f"{a.get('wanted', 0):,}", f"{a.get('funded', 0):,}",
                     f"${a.get('final_equity', float('nan')):,.0f}", f"{a.get('cagr', float('nan')):.2%}",
                     f"{e['oracle_hit']:.1%}", f"{a.get('unfunded_wanted', 0):,}"])
    primary, inc = accounts.get(PRIMARY, {}), accounts.get(INCUMBENT, {})
    checks = {
        "final_equity_vs_incumbent": primary.get("final_equity", -np.inf) > inc.get("final_equity", np.inf),
        "funded_count_at_least_90pct_of_incumbent": primary.get("funded", 0) >= 0.9 * inc.get("funded", np.inf),
        "an_ablation_beats_incumbent": (accounts.get("dev_only", {}).get("final_equity", -np.inf) > inc.get("final_equity", np.inf)) or (accounts.get("level_gate", {}).get("final_equity", -np.inf) > inc.get("final_equity", np.inf)),
        "no_defined_risk_failure": all(a.get("defined_risk_failures", 1) == 0 for a in accounts.values()),
    }
    return [
        {"title": "Decomposition of the compound decision", "body": [
            "Each head is trained on the axis it is graded on: the level head on event-mean realized PnL (the trade/no-trade and capital-priority axis), the deviation head on event-demeaned realized PnL (the structure-argmax axis). Both use the expanding annual OOS protocol and a five-seed MLP(64,32) ensemble. Gate thresholds are rate-matched to the incumbent gate on each training fold.",
            "The incumbent row is the EXP-161/162 baseline: trailing expected-PnL cutoff plus spread and market-cap hygiene, argmax simulated expected PnL, largest-secured-first funding.",
        ]},
        {"title": "Head quality", "body": [
            f"Deviation head within-event Spearman vs realized PnL: {quality['deviation_within_event_spearman']['mean']:+.3f} (EXP-162 raw-target selector: +0.025).",
            f"Level head OOS Spearman vs realized event mean: {quality['level_oos_spearman']['mean']:+.3f} on {quality['level_oos_spearman']['n']:,} events (EXP-162 raw-target cross-event correlation: +0.011).",
        ]},
        {"title": "Cash-secured account by arm", "note": "$200,000 start, 66% secured cap of current equity, 25% target share of headroom, every short put cash-secured at strike x 100. unfunded = gate-passed events the account could not afford.", "columns": ["arm", "gate", "selector", "same-day order", "eligible", "wanted", "funded", "final equity", "CAGR", "oracle hit", "unfunded"], "align": ["---"] + ["---"] * 10, "rows": rows},
        {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()]},
        {"title": "Selection funnel", "body": [f"Annual-OOS complete five-menu events: {len(choices):,}; incumbent-gate eligible: {int(choices['traded_inc'].sum()):,}; level-gate eligible (incumbent selector): {int(choices['traded_level'].sum()):,}; level-gate eligible (deviation selector): {int(choices['traded_level_dev'].sum()):,}."]},
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
    events = build_events(dataset)
    log(f"Built {len(events):,} event rows for the level head")
    dev, level, choices, diagnostics = generate(dataset, events, args.force_scores)
    metrics, quality = arm_metrics(choices), head_quality(dev, level, diagnostics)
    write_json(RESULTS / "event_choice_metrics.json", metrics)
    write_json(RESULTS / "head_quality.json", quality)
    write_json(RESULTS / "funding_policy.json", {**POLICY, "priority_col": PRIORITY_COL, "priority_arm": PRIMARY})
    spy = common.load_spy_daily()
    accounts, funded_books = {}, {}
    for arm in ARM_STRUCT:
        book = selected_mid_arm(mid, choices, arm, dev)
        funded, account = account_score(book, arm)
        account["eligible"] = int(choices.dropna(subset=[ARM_STRUCT[arm]])[("traded_level_dev" if arm in ("two_head", "two_head_order") else ARM_TRADED[arm])].sum())
        accounts[arm] = account
        funded_books[arm] = base.selected_all_alphas(raw, funded)
        log(f"{arm}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} final=${account.get('final_equity', float('nan')):,.0f}")
    evaluations = {}
    for arm in ARM_STRUCT:
        cell, run_dir = arm_spec(spec, arm), HERE if arm == PRIMARY else HERE / "arms" / arm
        extra = (lambda result: report_sections(accounts, metrics, quality, choices)) if arm == PRIMARY else [
            {"title": "Arm accounting", "body": [f"Gate: {ARM_GATE[arm]}; selector: {ARM_SELECTOR[arm]}; same-day order: {ARM_ORDER[arm]}. Final equity ${accounts[arm].get('final_equity', float('nan')):,.0f}; funded {accounts[arm].get('funded', 0):,}/{accounts[arm].get('wanted', 0):,}."]}
        ]
        result = evaluate(cell, funded_books[arm], gate=None, run_dir=run_dir, spy_daily=spy,
                         input_files=[base.CANDIDATES, ROOT / "data/features/panel.parquet", *cache_paths()],
                         extra_sections=extra, write_report=True)
        evaluations[arm] = result.results
        log(f"{arm}: evaluated, mean={result.results['headline'].get('mean'):+.3f}")
    write_json(RESULTS / "comparison.json", {"policy": {**POLICY, "priority_col": PRIORITY_COL},
                                             "arms": {"gate": ARM_GATE, "selector": ARM_SELECTOR, "order": ARM_ORDER},
                                             "head_quality": quality, "event_choice_metrics": metrics,
                                             "account_metrics": accounts, "evaluation_headlines": evaluations})
    if not args.no_ledger and not args.smoke:
        lib.record_evaluation(HERE, spec, evaluations[PRIMARY])
    print(f"[EXP-163] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
