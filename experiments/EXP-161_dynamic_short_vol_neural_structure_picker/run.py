#!/usr/bin/env python3
"""EXP-161: choose the offered DYN-SV structure, then fund the real account.

The historical DYN-SV score feed does not preserve every offered row. EXP-137
does: its family books contain the same five live menu shapes, one resolved
candidate per event and family. This runner maps those reproducible historical
equivalents to their live names, never extends the menu, and keeps the current
arithmetic event gate fixed. Therefore the only policy difference is which
offered structure is placed.
"""
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
CANDIDATES = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/results/candidates.parquet"
MARGIN_PATH = ROOT / "experiments/EXP-134_priced_right_funded_and_held_structure_s/margin.py"
SCORE_CACHE = RESULTS / "oos_candidate_scores.parquet"
CHOICES_CACHE = RESULTS / "oos_choices.parquet"
DIAGNOSTICS = RESULTS / "fold_diagnostics.json"
FIRST_TEST_YEAR = 2020
ANALOG_K = 25
SEEDS = (20260908, 20260909, 20260910)
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import pnl_sim
from engine.evaluate import evaluate
from engine.features import load_panel
from experiments import common, lib

MENU_MAP = {
    "N7q2_-1_-1_1": "TWIN-P",
    "N5q2_-2_1": "TWIN-P5",
    "N4q0_-1_1": "CND-PS",
    "N3q-2_1": "BFLY-P",
    "N5q-4_1_1": "BFLY-P5",
}
MENU = tuple(MENU_MAP.values())
BASE = (
    "exp_pnl_sim", "exp_pnl_sim_select", "entry_cost_pct",
    "analog_mean", "analog_win_rate", "analog_p10", "analog_p90", "analog_n",
    "is_twin_p", "is_twin_p5", "is_cnd_ps", "is_bfly_p", "is_bfly_p5",
)
CATEGORIES = {
    "geometry": (
        "pred_abs_move", "pred_abs_move_sd", "width_over_forecast",
        "half_width_pct_spot", "anchor_over_spot", "n_legs", "n_admissible",
        "dte_entry",
    ),
    "history": (
        "mean_prior_abs_move", "ema12r_abs", "signed_streak",
        "mean_prior_or_implied", "mcap_log", "or_implied", "or_rvol30",
    ),
    "market": (
        "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
        "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252",
    ),
    "execution": ("rel_spread", "quote_repaired", "wide_market"),
}
ARMS = {
    "nn_pnl_structure_analogs": BASE,
    "nn_plus_geometry": BASE + CATEGORIES["geometry"],
    "nn_plus_history": BASE + CATEGORIES["history"],
    "nn_plus_market": BASE + CATEGORIES["market"],
    "nn_plus_execution": BASE + CATEGORIES["execution"],
    "nn_all_categories": BASE + tuple(x for group in CATEGORIES.values() for x in group),
}
INCUMBENT = "incumbent_simulated_pnl"
PRIMARY = "nn_all_categories"
CONTROL = "nn_pnl_structure_analogs"
POLICY = dict(start_equity=200_000.0, cap=0.66, target_share=0.25, expensive_first=True)


def log(message: str) -> None:
    print(f"[EXP-161 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


margin = load_module("exp161_margin", MARGIN_PATH)


def conditional_exit(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    hold = (~out["exit_arb_ok"].fillna(True).astype(bool)) | (out["exit_value"] < 0)
    out.loc[hold, "exit_value"] = out.loc[hold, "exit_value_expiry"]
    out = out.dropna(subset=["entry_cost", "exit_value"]).copy()
    out["pnl"] = out["exit_value"] - out["entry_cost"]
    out["ret"] = out["pnl"] / out["entry_cost"]
    if (out["ret"] < -1.0 - 1e-8).any():
        raise RuntimeError("conditional exits still contain a loss below the debit")
    log(f"Conditional exits repaired {int(hold.sum()):,} stored fill rows")
    return out


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
        "fill_alpha", "entry_cost", "exit_value", "exit_value_expiry", "exit_arb_ok",
        "spot_entry", "exp_pnl_sim", "exp_pnl_sim_select", "pred_abs_move",
        "pred_abs_move_sd", "width_over_forecast", "half_width_pct_spot",
        "anchor_over_spot", "n_legs", "n_admissible", "dte_entry", "rel_spread",
        "quote_repaired", "wide_market", "legs", "common_universe",
    ]
    raw = pd.read_parquet(CANDIDATES, columns=columns)
    raw = raw[raw["arm"].isin(MENU_MAP)].copy()
    raw["strategy"] = raw["arm"].map(MENU_MAP)
    for col in ("event_date", "entry_date", "exit_date"):
        raw[col] = pd.to_datetime(raw[col]).dt.normalize()
    raw = conditional_exit(raw)
    raw["event_id"] = raw["event_id"].astype(str)
    raw["candidate_id"] = raw["event_id"] + "|" + raw["strategy"]
    raw["entry_cost_pct"] = 100.0 * raw["entry_cost"] / raw["spot_entry"]
    raw["year"] = raw["event_date"].dt.year
    mid = raw[np.isclose(raw["fill_alpha"].astype(float), 0.5)].copy()
    if mid.duplicated(["event_id", "strategy"]).any():
        raise RuntimeError("midpoint candidates are not unique per event and menu structure")
    offered = mid.groupby("event_id")["strategy"].agg(lambda x: set(x))
    complete_ids = set(offered[offered.map(lambda x: x == set(MENU))].index)
    mid = mid[mid["event_id"].isin(complete_ids)].copy()
    raw = raw[raw["event_id"].isin(complete_ids)].copy()
    panel_cols = ["ticker", "date", "mcap_usd", *CATEGORIES["history"], *CATEGORIES["market"]]
    panel = load_panel()[panel_cols].copy()
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.drop_duplicates(["ticker", "date"], keep="last")
    mid = mid.merge(panel, left_on=["ticker", "event_date"], right_on=["ticker", "date"], how="left", validate="many_to_one").drop(columns="date")
    for col in ("mcap_usd", *CATEGORIES["history"], *CATEGORIES["market"]):
        mid[col] = pd.to_numeric(mid[col], errors="coerce")
    mid["mcap_log"] = np.log(mid["mcap_usd"].where(mid["mcap_usd"] > 0))
    for col in ("quote_repaired", "wide_market"):
        mid[col] = mid[col].fillna(False).astype(float)
    for strategy in MENU:
        mid[f"is_{strategy.lower().replace('-', '_')}"] = (mid["strategy"] == strategy).astype(float)
    mid = mid.sort_values(["entry_date", "event_id", "strategy"]).reset_index(drop=True)
    log(f"Loaded {len(mid):,} midpoint candidates on {len(complete_ids):,} complete five-menu events; {len(raw):,} fill rows")
    return mid, raw


def add_causal_analogs(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    cols = ("analog_mean", "analog_win_rate", "analog_p10", "analog_p90", "analog_n")
    for col in cols:
        out[col] = np.nan
    dims = ("exp_pnl_sim", "width_over_forecast", "n_legs", "anchor_over_spot", "rel_spread")
    for strategy, idx in out.groupby("strategy", sort=False).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        part = out.loc[pos].sort_values(["entry_date", "event_id"]).copy()
        values = part[list(dims)].to_numpy(float)
        pnl = part["pnl"].to_numpy(float)
        entry = part["entry_date"].to_numpy()
        exits = part["exit_date"].to_numpy()
        for i in range(len(part)):
            usable = (exits[:i] < entry[i]) & np.isfinite(pnl[:i]) & np.isfinite(values[:i]).all(1) & np.isfinite(values[i]).all()
            pool = values[:i][usable]
            outcome = pnl[:i][usable]
            if len(pool) < ANALOG_K:
                continue
            scale = pool.std(0, ddof=1)
            scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
            distance = (((pool - values[i]) / scale) ** 2).mean(1)
            take = np.argpartition(distance, ANALOG_K - 1)[:ANALOG_K]
            analog = outcome[take]
            target = int(part.index[i])
            out.at[target, "analog_mean"] = float(analog.mean())
            out.at[target, "analog_win_rate"] = float((analog > 0).mean())
            out.at[target, "analog_p10"] = float(np.quantile(analog, 0.10))
            out.at[target, "analog_p90"] = float(np.quantile(analog, 0.90))
            out.at[target, "analog_n"] = float(len(analog))
            if i and i % 2000 == 0:
                log(f"{strategy} causal analogs {i:,}/{len(part):,}")
    log(f"Same-structure analog coverage {int(out['analog_mean'].notna().sum()):,}/{len(out):,}")
    return out


def fit_ensemble(X: np.ndarray, y: np.ndarray):
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return [make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20, random_state=seed)).fit(X, y) for seed in SEEDS]


def predict(models, X: np.ndarray) -> np.ndarray:
    return np.mean([model.predict(X) for model in models], axis=0)


def incumbent_gate(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame.sort_values(["event_id", "exp_pnl_sim", "strategy"], ascending=[True, False, True], kind="stable")
    incumbent = order.drop_duplicates("event_id").copy()
    history = incumbent[["event_date", "exp_pnl_sim"]].copy()
    months = incumbent["event_date"].dt.to_period("M")
    cutoffs = {month: pnl_sim.trailing_cutoff(history, month.to_timestamp()) for month in months.unique()}
    incumbent["arithmetic_cutoff"] = months.map(cutoffs).astype(float)
    incumbent["traded"] = (
        incumbent["exp_pnl_sim"].notna()
        & incumbent["arithmetic_cutoff"].notna()
        & (incumbent["exp_pnl_sim"] >= incumbent["arithmetic_cutoff"])
        & (incumbent["rel_spread"] <= 0.25)
        & (incumbent["mcap_usd"] >= 10e9)
    )
    return incumbent[["event_id", "strategy", "traded", "arithmetic_cutoff"]].rename(columns={"strategy": "incumbent_structure"})


def generate_choices(dataset: pd.DataFrame, force: bool) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    if SCORE_CACHE.exists() and CHOICES_CACHE.exists() and DIAGNOSTICS.exists() and not force:
        scores = pd.read_parquet(SCORE_CACHE)
        choices = pd.read_parquet(CHOICES_CACHE)
        for frame in (scores, choices):
            frame["event_date"] = pd.to_datetime(frame["event_date"])
        return scores, choices, json.loads(DIAGNOSTICS.read_text())
    gate = incumbent_gate(dataset)
    pieces, choice_pieces, diagnostics = [], [], []
    for year in range(FIRST_TEST_YEAR, int(dataset["year"].max()) + 1):
        train = dataset[dataset["year"] < year]
        test = dataset[dataset["year"] == year].copy()
        score_piece = test[["candidate_id", "event_id", "ticker", "event_date", "strategy", "pnl"]].rename(columns={"pnl": "realized_pnl"}).reset_index(drop=True)
        fold = {"year": int(year), "arms": {}}
        for arm, features in ARMS.items():
            train_ok = np.isfinite(train[list(features) + ["pnl"]].to_numpy(float)).all(1)
            test_ok = np.isfinite(test[list(features)].to_numpy(float)).all(1)
            values = np.full(len(test), np.nan)
            fit = train.loc[train_ok]
            if len(fit) >= 500 and test_ok.any():
                models = fit_ensemble(fit[list(features)].to_numpy(float), fit["pnl"].to_numpy(float))
                values[test_ok] = predict(models, test.loc[test_ok, list(features)].to_numpy(float))
            score_piece[arm] = values
            fold["arms"][arm] = {"features": len(features), "train": int(len(fit)), "scoreable": int(test_ok.sum())}
        scores_here = pd.concat([test.reset_index(drop=True), score_piece[[*ARMS]].reset_index(drop=True)], axis=1)
        truth = scores_here.sort_values(["event_id", "pnl", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": "oracle_structure"})
        incumbent = scores_here.sort_values(["event_id", "exp_pnl_sim", "strategy"], ascending=[True, False, True], kind="stable").drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": INCUMBENT})
        events = test[["event_id", "ticker", "event_date"]].drop_duplicates("event_id").merge(gate, on="event_id", how="left").merge(truth, on="event_id", how="left").merge(incumbent, on="event_id", how="left")
        for arm in ARMS:
            use = scores_here.dropna(subset=[arm]).sort_values(["event_id", arm, "strategy"], ascending=[True, False, True], kind="stable")
            choice = use.drop_duplicates("event_id")[["event_id", "strategy"]].rename(columns={"strategy": arm})
            events = events.merge(choice, on="event_id", how="left")
            fold["arms"][arm]["choices"] = int(choice["event_id"].nunique())
        pieces.append(score_piece)
        choice_pieces.append(events)
        diagnostics.append(fold)
        counts = ", ".join(f"{arm}={fold['arms'][arm]['choices']}" for arm in ARMS)
        log(f"Fold {year}: candidate train={len(train):,}, test={len(test):,}; {counts}")
    scores = pd.concat(pieces, ignore_index=True).sort_values(["event_date", "candidate_id"])
    choices = pd.concat(choice_pieces, ignore_index=True).sort_values(["event_date", "event_id"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORE_CACHE, index=False)
    choices.to_parquet(CHOICES_CACHE, index=False)
    write_json(DIAGNOSTICS, diagnostics)
    return scores, choices, diagnostics


def candidate_rank_metrics(scores: pd.DataFrame) -> dict:
    out = {}
    for arm in ARMS:
        rows = scores[[arm, "realized_pnl"]].dropna()
        decile = pd.qcut(rows[arm], 10, labels=False, duplicates="drop")
        means = rows.groupby(decile)["realized_pnl"].mean()
        out[arm] = {"n": int(len(rows)), "spearman": float(spearmanr(rows[arm], rows["realized_pnl"]).statistic), "top_bottom_pnl": float(means.iloc[-1] - means.iloc[0])}
    return out


def event_metrics(choices: pd.DataFrame) -> dict:
    out = {}
    for arm in (INCUMBENT, *ARMS):
        rows = choices.dropna(subset=[arm])
        out[arm] = {
            "eligible": int(rows["traded"].sum()),
            "choice_coverage": int(len(rows)),
            "oracle_hit": float((rows[arm] == rows["oracle_structure"]).mean()),
            "incumbent_agreement": float((rows[arm] == rows[INCUMBENT]).mean()),
        }
    return out


def selected_mid(mid: pd.DataFrame, choices: pd.DataFrame, arm: str) -> pd.DataFrame:
    picked = choices[["event_id", "traded", arm]].dropna(subset=[arm]).rename(columns={arm: "picked_structure"})
    out = mid.merge(picked, on="event_id", how="inner")
    out = out[(out["strategy"] == out["picked_structure"]) & out["traded"].fillna(False)].copy()
    out["secured_per_contract"] = out["legs"].map(margin.secured_per_contract)
    return out.sort_values(["entry_date", "event_id"])


def account_score(book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    funded = margin.simulate(book, **POLICY)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0}
    start = POLICY["start_equity"]
    final = float(funded.attrs["final_equity"])
    span = (pd.to_datetime(funded["exit_date"]).max() - pd.to_datetime(funded["entry_date"]).min()).days
    years = span / 365.25
    return funded, {
        "wanted": int(len(funded)), "funded": int(len(placed)), "funded_pct": float(placed.shape[0] / len(funded)),
        "contracts_median": float(placed["contracts"].median()), "contracts_total": float(placed["contracts"].sum()),
        "peak_concurrency": int(funded["concurrency"].max()), "final_equity": final,
        "profit_usd": final - start, "cagr": float((final / start) ** (1 / years) - 1) if years > 0 else float("nan"),
        "max_secured_usd": float((funded["secured_before"] + funded["contracts"] * funded["secured_per_contract"]).max()),
        "defined_risk_failures": int((placed["ret"] < -1.0 - 1e-8).sum()),
    }


def selected_all_alphas(raw: pd.DataFrame, funded: pd.DataFrame) -> pd.DataFrame:
    keys = set(funded.loc[funded["funded"], "candidate_id"])
    return raw[raw["candidate_id"].isin(keys)].copy()


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    if arm != PRIMARY:
        out["primary_spec"]["evaluated_arm"] = arm
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def plan(spec: dict) -> None:
    digest = lib.spec_hash(spec)
    ledger = lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{"id": spec["id"], "spec_hash": digest, "date": "2026-09-08", "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}])
        log("Registered primary specification in LEDGER.csv")


def report_sections(accounts: dict, events: dict, ranks: dict, choices: pd.DataFrame) -> list[dict]:
    rows = []
    for arm in (INCUMBENT, *ARMS):
        a, e = accounts.get(arm, {}), events[arm]
        rank = ranks.get(arm, {})
        rows.append([arm, f"{e['oracle_hit']:.1%}", f"{e['incumbent_agreement']:.1%}", f"{e['eligible']:,}", f"{a.get('wanted', 0):,}", f"{a.get('funded', 0):,}", f"${a.get('final_equity', float('nan')):,.0f}", f"{a.get('cagr', float('nan')):.2%}", f"{a.get('peak_concurrency', 0):,}", "n/a" if not rank else f"{rank['spearman']:+.3f}"])
    base = accounts.get(CONTROL, {})
    primary = accounts.get(PRIMARY, {})
    checks = {
        "selected_realized_pnl": primary.get("final_equity", -np.inf) > base.get("final_equity", np.inf),
        "oracle_hit_vs_incumbent": events[PRIMARY]["oracle_hit"] > events[INCUMBENT]["oracle_hit"],
        "account_final_equity_vs_control": primary.get("final_equity", -np.inf) > base.get("final_equity", np.inf),
        "no_defined_risk_failure": primary.get("defined_risk_failures", 1) == 0,
    }
    return [
        {"title": "Exact DYN-SV menu and fixed decision boundary", "body": [
            "The offered menu is exactly TWIN-P, TWIN-P5, CND-PS, BFLY-P, and BFLY-P5, using the reproducible historical equivalents from EXP-137. The incumbent selects maximum simulated PnL. Each neural arm predicts conditional realized PnL for every offered candidate, then chooses its maximum.",
            "The current arithmetic trade/no-trade eligibility is deliberately inherited from the incumbent maximum-simulated-PnL candidate. Thus differences below are structure choice, not a changed trade threshold.",
        ]},
        {"title": "Cash-secured account and chooser comparison", "note": "Primary funding policy inherited from EXP-136: $200,000 start, 66% secured-cap of current equity, 25% target share of headroom, largest secured requirement first within an entry date. Every short put is secured at strike x 100; no spread-margin offsets.", "columns": ["arm", "oracle hit", "incumbent agree", "eligible", "wanted", "funded", "final equity", "CAGR", "peak open", "candidate rho"], "align": ["---"] + ["---:"] * 9, "rows": rows},
        {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()]},
        {"title": "Selection funnel", "body": [f"Complete five-menu events: {choices['event_id'].nunique():,}; annual-OOS events: {len(choices):,}; incumbent-gate eligible: {int(choices['traded'].sum()):,}. Funding occurs only after causal OOS selection."]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-scores", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--primary-only", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger:
        plan(spec)
    mid, raw = load_data()
    dataset = add_causal_analogs(mid)
    scores, choices, diagnostics = generate_choices(dataset, args.force_scores)
    ranks = candidate_rank_metrics(scores)
    events = event_metrics(choices)
    write_json(RESULTS / "candidate_rank_metrics.json", ranks)
    write_json(RESULTS / "event_choice_metrics.json", events)
    write_json(RESULTS / "funding_policy.json", POLICY)
    spy = common.load_spy_daily()
    accounts, funding, evaluations = {}, {}, {}
    for arm in (INCUMBENT, *ARMS):
        book = selected_mid(mid, choices, arm)
        funded, account = account_score(book)
        accounts[arm] = account
        funding[arm] = selected_all_alphas(raw, funded)
    run_arms = (PRIMARY,) if args.primary_only else (INCUMBENT, *ARMS)
    for arm in run_arms:
        account = accounts[arm]
        rows = funding[arm]
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        cell = arm_spec(spec, arm)
        extra = (lambda result: report_sections(accounts, events, ranks, choices)) if arm == PRIMARY else [{"title": "Arm accounting", "body": [f"This arm uses the fixed incumbent eligibility decision and the EXP-136 primary cash-secured policy. Account final equity: ${account.get('final_equity', float('nan')):,.0f}; funded: {account.get('funded', 0):,}/{account.get('wanted', 0):,}."]}]
        try:
            result = evaluate(cell, rows, gate=None, run_dir=run_dir, spy_daily=spy, input_files=[CANDIDATES, ROOT / "data/features/panel.parquet", SCORE_CACHE, CHOICES_CACHE], extra_sections=extra, write_report=True)
        except BaseException as exc:
            write_json(RESULTS / "failure.json", {"arm": arm, "type": type(exc).__name__, "message": str(exc)})
            raise
        evaluations[arm] = result.results["headline"]
        log(f"{arm}: eligible={int(choices['traded'].sum()):,} wanted={account.get('wanted',0):,} funded={account.get('funded',0):,} final=${account.get('final_equity',float('nan')):,.0f}")
    output = {"policy": POLICY, "candidate_rank_metrics": ranks, "event_choice_metrics": events, "account_metrics": accounts, "evaluation_headlines": evaluations, "folds": diagnostics}
    write_json(RESULTS / "comparison.json", output)
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, evaluations[PRIMARY])
    print(f"[EXP-161] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
