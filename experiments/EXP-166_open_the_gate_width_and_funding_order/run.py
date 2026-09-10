#!/usr/bin/env python3
"""EXP-166: gate width (cut100/cut50/cut0) crossed with funding order
(expensive-first vs expected-PnL-per-secured-dollar) on the EXP-165
qt_schem selector, loaded from cache. No model training."""
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

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py"
MARGIN163 = ROOT / "experiments/EXP-163_two_headed_level_and_deviation_dyn_sv/margin163.py"
E165 = ROOT / "experiments/EXP-165_payoff_schematic_features_for_the_dev/results"
INCUMBENT = "incumbent_resolver"
PRIMARY = "cut50_order"
GATES = ("cut100", "cut50", "cut0")
ARMS = ("incumbent_resolver", "cut100_expensive", "cut100_order",
        "cut50_expensive", "cut50_order", "cut0_expensive", "cut0_order")
POLICY = dict(start_equity=200_000.0, cap=0.66, target_share=0.25)
PRIORITY_COL = "expected_per_secured"
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


base = load_module("exp166_base", SOURCE)
margin163 = load_module("exp166_margin163", MARGIN163)


def log(message: str) -> None:
    print(f"[EXP-166 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def build_event_frame(mid: pd.DataFrame, choices: pd.DataFrame) -> pd.DataFrame:
    ch = choices.copy()
    ch["event_date"] = pd.to_datetime(ch["event_date"])
    picks = mid[["event_id", "event_date", "ticker", "strategy", "exp_pnl_sim", "rel_spread", "mcap_usd", "entry_date", "exit_date"]]
    inc = picks.merge(ch[["event_id", "incumbent_structure", "arithmetic_cutoff", "qt_schem"]], on="event_id", how="inner")
    inc = inc[inc["strategy"] == inc["incumbent_structure"]].drop_duplicates("event_id")
    ev = ch[["event_id", "ticker", "event_date", "oracle_structure", "incumbent_structure", "qt_schem"]].merge(
        inc[["event_id", "exp_pnl_sim", "arithmetic_cutoff"]].rename(columns={"exp_pnl_sim": "inc_exp_pnl_sim"}),
        on="event_id", how="left", validate="one_to_one")
    chosen = picks.merge(ch[["event_id", "qt_schem"]], on="event_id", how="inner")
    chosen = chosen[chosen["strategy"] == chosen["qt_schem"]].drop_duplicates("event_id")
    ev = ev.merge(chosen[["event_id", "exp_pnl_sim", "rel_spread", "mcap_usd", "entry_date", "exit_date"]].rename(
        columns={"exp_pnl_sim": "chosen_exp_pnl_sim", "rel_spread": "chosen_rel_spread", "mcap_usd": "chosen_mcap"}),
        on="event_id", how="left", validate="one_to_one")
    hyg = (ev["chosen_rel_spread"] <= 0.25) & (ev["chosen_mcap"] >= 10e9)
    for gate in GATES:
        if gate == "cut100":
            cond = ev["inc_exp_pnl_sim"] >= ev["arithmetic_cutoff"]
        elif gate == "cut50":
            cond = ev["inc_exp_pnl_sim"] >= 0.5 * ev["arithmetic_cutoff"]
        else:
            cond = pd.Series(True, index=ev.index)
        ev[f"traded_{gate}"] = (cond & hyg).fillna(False).astype(bool)
    ev["hygiene"] = hyg.fillna(False)
    return ev


def arm_books(mid: pd.DataFrame, ev: pd.DataFrame) -> dict:
    books = {}
    legs_secured = mid[["event_id", "strategy", "legs"]].drop_duplicates(["event_id", "strategy"])
    legs_secured["secured_per_contract"] = legs_secured["legs"].map(margin163.secured_per_contract)
    for arm in ARMS:
        if arm == INCUMBENT:
            struct_col, gate_col = "incumbent_structure", "traded_cut100"
        else:
            gate = arm.split("_", 1)[0]
            struct_col, gate_col = "qt_schem", f"traded_{gate}"
        picked = ev[["event_id", gate_col, struct_col]].dropna(subset=[struct_col]).rename(
            columns={gate_col: "traded", struct_col: "picked_structure"})
        out = mid.merge(picked, on="event_id", how="inner")
        out = out[(out["strategy"] == out["picked_structure"]) & out["traded"].fillna(False)].copy()
        out = out.drop_duplicates(["event_id", "strategy"])
        out = out.merge(legs_secured[["event_id", "strategy", "secured_per_contract"]],
                        on=["event_id", "strategy"], how="left", validate="one_to_one")
        if arm.endswith("_order") or arm == INCUMBENT:
            pass
        out["expected_per_secured"] = out["exp_pnl_sim"] * 100.0 / out["secured_per_contract"]
        books[arm] = out.sort_values(["entry_date", "event_id"]).reset_index(drop=True)
        log(f"book {arm}: {len(books[arm]):,} gated trades")
    return books


def account_score(book: pd.DataFrame, arm: str) -> tuple[pd.DataFrame, dict]:
    priority = PRIORITY_COL if arm.endswith("_order") else None
    funded = margin163.simulate(book, start_equity=POLICY["start_equity"], cap=POLICY["cap"],
                                target_share=POLICY["target_share"], priority_col=priority)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0}
    start = POLICY["start_equity"]
    final = float(funded.attrs["final_equity"])
    span = (pd.to_datetime(funded["exit_date"]).max() - pd.to_datetime(funded["entry_date"]).min()).days
    years = span / 365.25
    unfunded = funded[~funded["funded"]]
    util = (funded["secured_before"] / (POLICY["cap"] * funded["equity_at_entry"])).replace([np.inf, -np.inf], np.nan)
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
        "funded_mean_realized_pnl": float(placed["pnl"].mean()),
        "mean_secured_utilization": float(util.mean()),
    }


def arm_metrics(ev: pd.DataFrame, books: dict, accounts: dict) -> dict:
    out = {}
    for arm in ARMS:
        if arm == INCUMBENT:
            struct_col = "incumbent_structure"
        else:
            struct_col = "qt_schem"
        rows = ev.dropna(subset=[struct_col])
        out[arm] = {
            "oracle_hit": float((rows[struct_col] == rows["oracle_structure"]).mean()),
            "hygiene_pass": int(ev["hygiene"].sum()),
        }
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
        lib.ledger_append([{"id": spec["id"], "spec_hash": digest, "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}])
        log("Registered primary specification in LEDGER.csv")


def report_sections(accounts: dict, metrics: dict, ev: pd.DataFrame) -> list[dict]:
    rows = []
    for arm in ARMS:
        a = accounts[arm]
        rows.append([arm, f"{a.get('wanted', 0):,}", f"{a.get('funded', 0):,}", f"{a.get('unfunded_wanted', 0):,}",
                     f"{a.get('unfunded_mean_realized_pnl', 0):+.3f}", f"{a.get('funded_mean_realized_pnl', 0):+.3f}",
                     f"{a.get('mean_secured_utilization', 0):.0%}", f"{a.get('peak_concurrency', 0):,}",
                     f"${a.get('final_equity', float('nan')):,.0f}", f"{a.get('cagr', float('nan')):.2%}"])
    primary, base_line = accounts.get(PRIMARY, {}), accounts.get(INCUMBENT, {})
    checks = {
        "cut50_order_beats_incumbent_resolver": primary.get("final_equity", -np.inf) > base_line.get("final_equity", np.inf),
        "cut50_order_beats_cut100_order": primary.get("final_equity", -np.inf) > accounts.get("cut100_order", {}).get("final_equity", np.inf),
        "an_ordering_arm_beats_its_expensive_twin": (
            accounts.get("cut50_order", {}).get("final_equity", -np.inf) > accounts.get("cut50_expensive", {}).get("final_equity", np.inf)
            or accounts.get("cut0_order", {}).get("final_equity", -np.inf) > accounts.get("cut0_expensive", {}).get("final_equity", np.inf)
        ),
        "no_defined_risk_failure": all(a.get("defined_risk_failures", 1) == 0 for a in accounts.values()),
    }
    return [
        {"title": "Gate width crossed with funding order", "body": [
            "The selector is fixed (EXP-165 qt_schem, out-of-fold from cache) and hygiene is fixed (rel_spread <= 0.25, mcap >= $10B on the chosen candidate). The gate's expected-PnL condition varies: incumbent cutoff, half cutoff, or none. Funding varies: largest-secured-first versus expected PnL per secured dollar. The incumbent_resolver row is the program baseline.",
            f"Pool context: {len(ev):,} annual-OOS events; hygiene passes {int(ev['hygiene'].sum()):,}; the incumbent gate passes {int(ev['traded_cut100'].sum()):,}; half-cutoff {int(ev['traded_cut50'].sum()):,}.",
        ]},
        {"title": "Account by arm", "note": "$200,000 start, 66% secured cap, 25% headroom target. utilization = mean secured headroom consumption at entry. Unfunded = gate-passed events the account could not afford; its mean realized PnL is the opportunity cost the ordering leaves behind.", "columns": ["arm", "wanted", "funded", "unfunded", "unf mean", "funded mean", "util", "peak open", "final equity", "CAGR"], "align": ["---"] + ["---:"] * 9, "rows": rows},
        {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()]},
    ]


def main() -> None:
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        RESULTS = HERE / "results_smoke"
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger and not args.smoke:
        plan(spec)
    mid, raw = base.load_data()
    choices = pd.read_parquet(E165 / "choices.parquet")
    ev = build_event_frame(mid, choices)
    if args.smoke:
        ev = ev[ev["event_date"].dt.year >= 2025].copy()
    log(f"Event frame: {len(ev):,} events; cut100 {int(ev['traded_cut100'].sum()):,}, cut50 {int(ev['traded_cut50'].sum()):,}, cut0 {int(ev['traded_cut0'].sum()):,}")
    books = arm_books(mid, ev)
    accounts, metrics, funded_books = {}, arm_metrics(ev, books, {}), {}
    for arm in ARMS:
        funded, account = account_score(books[arm], arm)
        accounts[arm] = account
        funded_books[arm] = base.selected_all_alphas(raw, funded)
        log(f"{arm}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} final=${account.get('final_equity', float('nan')):,.0f}")
    spy = common.load_spy_daily()
    evaluations = {}
    for arm in ARMS:
        cell, run_dir = arm_spec(spec, arm), HERE if arm == PRIMARY else HERE / "arms" / arm
        extra = (lambda result: report_sections(accounts, metrics, ev)) if arm == PRIMARY else [
            {"title": "Arm accounting", "body": [f"Arm: {arm}. Final equity ${accounts[arm].get('final_equity', float('nan')):,.0f}; funded {accounts[arm].get('funded', 0):,}/{accounts[arm].get('wanted', 0):,}; unfunded mean realized {accounts[arm].get('unfunded_mean_realized_pnl', 0):+.3f}."]}
        ]
        result = evaluate(cell, funded_books[arm], gate=None, run_dir=run_dir, spy_daily=spy,
                          input_files=[base.CANDIDATES, E165 / "dev_scores.parquet", E165 / "choices.parquet"],
                          extra_sections=extra, write_report=True)
        evaluations[arm] = result.results
        log(f"{arm}: evaluated, mean={result.results['headline'].get('mean', float('nan')):+.3f}")
    write_json(RESULTS / "comparison.json", {"policy": POLICY, "event_metrics": metrics,
                                             "account_metrics": accounts,
                                             "evaluation_headlines": {k: v["headline"] for k, v in evaluations.items()}})
    if not args.no_ledger and not args.smoke:
        lib.record_evaluation(HERE, spec, evaluations[PRIMARY])
    print(f"[EXP-166] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
