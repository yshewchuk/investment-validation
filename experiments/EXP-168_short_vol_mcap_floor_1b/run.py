#!/usr/bin/env python3
"""EXP-168: sensitivity of the DYN-SV arithmetic market-cap floor.

This is intentionally a policy-only comparison. It reuses the complete-menu,
all-alpha candidate cache and the conditional-exit repair from EXP-161. The
chooser, causal expected-PnL cutoff, 25 percent spread ceiling, and funding
policy stay fixed. The one changed fact is the chosen candidate market-cap
floor.
"""
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
ARMS = ("baseline_10b", "expanded_1b", "incremental_1b_10b")
PRIMARY = "expanded_1b"
POLICY = dict(start_equity=200_000.0, cap=0.66, target_share=0.25, expensive_first=True)
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine import pnl_sim
from engine.evaluate import evaluate
from experiments import common, lib


def log(message: str) -> None:
    print(f"[EXP-168 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


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


base = load_module("exp168_base", SOURCE)

MENU_MAP = {
    "N7q2_-1_-1_1": "TWIN-P",
    "N5q2_-2_1": "TWIN-P5",
    "N4q0_-1_1": "CND-PS",
    "N3q-2_1": "BFLY-P",
    "N5q-4_1_1": "BFLY-P5",
}
MENU = set(MENU_MAP.values())
MID_COLUMNS = [
    "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
    "fill_alpha", "entry_cost", "exit_value", "exit_value_expiry", "exit_arb_ok",
    "spot_entry", "exp_pnl_sim", "rel_spread", "legs",
]
ALL_COLUMNS = [
    "arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
    "fill_alpha", "entry_cost", "exit_value", "exit_value_expiry", "exit_arb_ok",
    "spot_entry", "spot_exit", "strike", "expiry", "dte_entry", "dte_exit",
    "n_legs", "contracts", "rel_spread", "wide_market", "quote_repaired",
    "exp_pnl_sim", "pred_abs_move", "pred_abs_move_sd", "legs",
]


def plan(spec: dict) -> None:
    digest = lib.spec_hash(spec)
    ledger = lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{
            "id": spec["id"],
            "spec_hash": digest,
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "stage": "planned",
            "oos_mean_mid": "",
            "sharpe_trade": "",
            "promoted": "False",
        }])
        log("Registered primary specification in LEDGER.csv")


def conditional_exit(raw: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    hold = (~out["exit_arb_ok"].fillna(True).astype(bool)) | (out["exit_value"] < 0)
    out.loc[hold, "exit_value"] = out.loc[hold, "exit_value_expiry"]
    out = out.dropna(subset=["entry_cost", "exit_value"]).copy()
    out["pnl"] = out["exit_value"] - out["entry_cost"]
    out["ret"] = out["pnl"] / out["entry_cost"]
    if (out["ret"] < -1.0 - 1e-8).any():
        raise RuntimeError("conditional exits contain a loss below the debit")
    return out


def load_data() -> pd.DataFrame:
    raw = pd.read_parquet(base.CANDIDATES, columns=MID_COLUMNS)
    raw = raw[raw["arm"].isin(MENU_MAP)].copy()
    raw["strategy"] = raw["arm"].map(MENU_MAP)
    raw = raw[np.isclose(raw["fill_alpha"].astype(float), 0.5)].copy()
    for column in ("event_date", "entry_date", "exit_date"):
        raw[column] = pd.to_datetime(raw[column]).dt.normalize()
    raw = conditional_exit(raw)
    offered = raw.groupby("event_id")["strategy"].agg(lambda values: set(values))
    complete_ids = set(offered[offered.map(lambda values: values == MENU)].index)
    mid = raw[raw["event_id"].isin(complete_ids)].copy()
    if mid.duplicated(["event_id", "strategy"]).any():
        raise RuntimeError("midpoint candidates are not unique per event and strategy")
    panel = pd.read_parquet(ROOT / "data/features/panel.parquet", columns=["ticker", "date", "mcap_usd"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.drop_duplicates(["ticker", "date"], keep="last")
    mid = mid.merge(
        panel,
        left_on=["ticker", "event_date"],
        right_on=["ticker", "date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    mid["mcap_usd"] = pd.to_numeric(mid["mcap_usd"], errors="coerce")
    mid["candidate_id"] = mid["event_id"].astype(str) + "|" + mid["strategy"]
    mid = mid.sort_values(["entry_date", "event_id", "strategy"]).reset_index(drop=True)
    log(f"Loaded {len(mid):,} midpoint candidates on {len(complete_ids):,} complete-menu events")
    return mid


def all_alpha_rows(funded: pd.DataFrame) -> pd.DataFrame:
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return pd.DataFrame(columns=ALL_COLUMNS + ["strategy", "candidate_id", "pnl", "ret"])
    event_ids = placed["event_id"].astype(str).unique().tolist()
    wanted = set(placed["candidate_id"].astype(str))
    raw = pd.read_parquet(base.CANDIDATES, columns=ALL_COLUMNS, filters=[("event_id", "in", event_ids)])
    raw = raw[raw["arm"].isin(MENU_MAP)].copy()
    raw["strategy"] = raw["arm"].map(MENU_MAP)
    raw["candidate_id"] = raw["event_id"].astype(str) + "|" + raw["strategy"]
    raw = raw[raw["candidate_id"].isin(wanted)].copy()
    for column in ("event_date", "entry_date", "exit_date"):
        raw[column] = pd.to_datetime(raw[column]).dt.normalize()
    raw = conditional_exit(raw)
    log(f"Loaded {len(raw):,} all-alpha rows for {len(wanted):,} funded selections")
    return raw


def chosen_events(mid: pd.DataFrame) -> pd.DataFrame:
    order = mid.sort_values(
        ["event_id", "exp_pnl_sim", "strategy"],
        ascending=[True, False, True],
        kind="stable",
    )
    out = order.drop_duplicates("event_id").copy()
    history = out[["event_date", "exp_pnl_sim"]].copy()
    months = out["event_date"].dt.to_period("M")
    cutoffs = {
        month: pnl_sim.trailing_cutoff(history, month.to_timestamp())
        for month in months.unique()
    }
    out["arithmetic_cutoff"] = months.map(cutoffs).astype(float)
    out["base_eligible"] = (
        out["exp_pnl_sim"].notna()
        & out["arithmetic_cutoff"].notna()
        & (out["exp_pnl_sim"] >= out["arithmetic_cutoff"])
        & (out["rel_spread"] <= 0.25)
    )
    return out.sort_values(["entry_date", "event_id"]).reset_index(drop=True)


def arm_book(events: pd.DataFrame, arm: str) -> pd.DataFrame:
    if arm == "baseline_10b":
        mask = events["mcap_usd"] >= 10e9
    elif arm == "expanded_1b":
        mask = events["mcap_usd"] >= 1e9
    elif arm == "incremental_1b_10b":
        mask = (events["mcap_usd"] >= 1e9) & (events["mcap_usd"] < 10e9)
    else:
        raise ValueError(f"unknown arm {arm}")
    out = events[events["base_eligible"] & mask].copy()
    out["secured_per_contract"] = out["legs"].map(base.margin.secured_per_contract)
    return out.sort_values(["entry_date", "event_id"]).reset_index(drop=True)


def account_score(book: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    funded = base.margin.simulate(book, **POLICY)
    placed = funded[funded["funded"]].copy()
    if placed.empty:
        return funded, {"wanted": int(len(funded)), "funded": 0}
    start = POLICY["start_equity"]
    final = float(funded.attrs["final_equity"])
    span = (pd.to_datetime(funded["exit_date"]).max() - pd.to_datetime(funded["entry_date"]).min()).days
    years = span / 365.25
    unfunded = funded[~funded["funded"]]
    return funded, {
        "wanted": int(len(funded)),
        "funded": int(len(placed)),
        "unfunded": int(len(unfunded)),
        "funded_pct": float(len(placed) / len(funded)),
        "final_equity": final,
        "profit_usd": final - start,
        "cagr": float((final / start) ** (1 / years) - 1) if years > 0 else float("nan"),
        "peak_concurrency": int(funded["concurrency"].max()),
        "defined_risk_failures": int((placed["ret"] < -1.0 - 1e-8).sum()),
        "funded_mean_pnl": float(placed["pnl"].mean()),
        "unfunded_mean_pnl": float(unfunded["pnl"].mean()) if len(unfunded) else None,
    }


def arm_spec(spec: dict, arm: str) -> dict:
    out = deepcopy(spec)
    if arm != PRIMARY:
        out["primary_spec"]["evaluated_arm"] = arm
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def report_sections(events: pd.DataFrame, accounts: dict, headlines: dict) -> list[dict]:
    rows = []
    for arm in ARMS:
        account = accounts[arm]
        head = headlines.get(arm, {})
        rows.append([
            arm,
            f"{account.get('wanted', 0):,}",
            f"{account.get('funded', 0):,}",
            f"{account.get('unfunded', 0):,}",
            f"{head.get('mean', float('nan')):+.2%}",
            f"{head.get('win_rate', float('nan')):.1%}",
            f"${account.get('final_equity', float('nan')):,.0f}",
            f"{account.get('cagr', float('nan')):.2%}",
            f"{account.get('peak_concurrency', 0):,}",
        ])
    base_head = headlines.get("baseline_10b", {})
    expanded_head = headlines.get("expanded_1b", {})
    incremental_head = headlines.get("incremental_1b_10b", {})
    checks = {
        "expanded_midpoint_positive": expanded_head.get("mean", -np.inf) > 0,
        "expanded_not_worse_than_baseline": expanded_head.get("mean", -np.inf) >= base_head.get("mean", np.inf),
        "incremental_midpoint_positive": incremental_head.get("mean", -np.inf) > 0,
        "no_defined_risk_failure": all(accounts[a].get("defined_risk_failures", 1) == 0 for a in ARMS),
    }
    return [
        {"title": "One-variable market-cap sensitivity", "body": [
            "The event-level chooser remains maximum simulated expected PnL across the current five-structure DYN-SV menu. Every arm keeps the same causal trailing simulated-PnL cutoff, relative-spread ceiling of 25%, conditional exit repair, and cash-secured funding. Only the selected candidate market-cap floor changes.",
            f"Complete-menu annual-OOS events: {events['event_id'].nunique():,}; after expected-PnL and spread filters before market-cap filtering: {int(events['base_eligible'].sum()):,}.",
        ]},
        {"title": "Market-cap policy books", "note": "Returns are evaluated at five fill assumptions from real ORATS bid and ask chains. Account figures use $200,000 start, 66% secured-cap, 25% headroom target, largest-secured-first ordering, and fully cash-secured short puts.", "columns": ["arm", "wanted", "funded", "unfunded", "mid mean", "mid win", "final equity", "CAGR", "peak open"], "align": ["---"] + ["---:"] * 8, "rows": rows},
        {"title": "Primary checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[name, "PASS" if value else "FAIL"] for name, value in checks.items()]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger and not args.smoke:
        plan(spec)
    mid = load_data()
    events = chosen_events(mid)
    if args.smoke:
        events = events[events["event_date"].dt.year >= 2025].copy()
    log(f"Selected events {len(events):,}; pre-market-cap eligible {int(events['base_eligible'].sum()):,}")
    accounts, funded_books, headlines = {}, {}, {}
    spy = common.load_spy_daily()
    for arm in ARMS:
        book = arm_book(events, arm)
        funded, account = account_score(book)
        rows = all_alpha_rows(funded)
        accounts[arm] = account
        funded_books[arm] = rows
        log(f"{arm}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} rows={len(rows):,}")
    def primary_extra(result):
        complete = dict(headlines)
        complete[PRIMARY] = result.results["headline"]
        return report_sections(events, accounts, complete)

    report_order = ("baseline_10b", "incremental_1b_10b", PRIMARY)
    for arm in report_order:
        run_dir = HERE if arm == PRIMARY else HERE / "arms" / arm
        cell = arm_spec(spec, arm)
        extra = primary_extra if arm == PRIMARY else [
            {"title": "Arm accounting", "body": [f"{arm}: funded {accounts[arm].get('funded', 0):,}/{accounts[arm].get('wanted', 0):,}; final equity ${accounts[arm].get('final_equity', float('nan')):,.0f}."]}
        ]
        result = evaluate(
            cell,
            funded_books[arm],
            gate=None,
            run_dir=run_dir,
            spy_daily=spy,
            input_files=[base.CANDIDATES, ROOT / "data/features/panel.parquet"],
            extra_sections=extra,
            write_report=True,
        )
        headlines[arm] = result.results["headline"]
        log(f"{arm}: mid mean {headlines[arm].get('mean', float('nan')):+.3%}")
    write_json(RESULTS / "comparison.json", {
        "policy": POLICY,
        "event_counts": {
            "complete_menu": int(events["event_id"].nunique()),
            "pre_market_cap_eligible": int(events["base_eligible"].sum()),
        },
        "accounts": accounts,
        "headlines": headlines,
    })
    if not args.no_ledger and not args.smoke:
        lib.record_evaluation(HERE, spec, {"headline": headlines[PRIMARY]})
    print(f"[EXP-168] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
