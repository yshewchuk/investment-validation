#!/usr/bin/env python3
"""EXP-170: the menu7-prime ten-billion book re-run as a primary promotion
candidate, with the mandatory doubled-move tail injection in the harness."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
E169_RUN = ROOT / "experiments/EXP-169_menu7prime_confirmation/run.py"
E169_RESULTS = ROOT / "experiments/EXP-169_menu7prime_confirmation/results"
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


e169 = load_module("exp170_e169", E169_RUN)


def log(message: str) -> None:
    print(f"[EXP-170 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def plan(spec: dict) -> None:
    digest, ledger = lib.spec_hash(spec), lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{"id": spec["id"], "spec_hash": digest, "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
                            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}])
        log("Registered primary specification in LEDGER.csv")


def report_sections(account: dict, headline: dict) -> list[dict]:
    checks = {
        "beats_incumbent_champion_cagr": headline.get("cagr", 0) > 2.6842734150433207,
        "beats_incumbent_champion_sharpe": headline.get("sharpe_trade", 0) > 2.793512910610043,
        "no_defined_risk_failure": account.get("defined_risk_failures", 1) == 0,
    }
    return [
        {"title": "What this run is", "body": [
            "The EXP-169 menu7p_mcap10 book, re-run as a PRIMARY spec so the promotion machinery can judge it: same cached out-of-fold selector choices, same cut0 hygiene at the ten-billion floor, same per-secured-dollar ordering - plus the mandatory doubled-move tail injection that the grid-cell path skipped.",
            "Champion reference: the incumbent resolver baseline (EXP-166 incumbent_resolver arm): CAGR 2.68 (headline scale), trade Sharpe 2.79, 7/7 positive years, MC P(loss) at 5 percent = 0.",
        ]},
        {"title": "Candidate account", "note": "$200,000 start, 66% secured cap, 25% headroom target.", "columns": ["wanted", "funded", "final equity", "CAGR", "peak open"], "align": ["---:"] * 5, "rows": [[
            f"{account.get('wanted', 0):,}", f"{account.get('funded', 0):,}",
            f"${account.get('final_equity', float('nan')):,.0f}", f"{account.get('cagr', float('nan')):.2%}",
            f"{account.get('peak_concurrency', 0):,}",
        ]]},
        {"title": "Candidate checks", "columns": ["check", "status"], "align": ["---", "---"], "rows": [[k, "PASS" if v else "FAIL"] for k, v in checks.items()]},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger:
        plan(spec)
    mid, raw = e169.load_data_menu(e169.MENU)
    choices = pd.read_parquet(E169_RESULTS / "choices.parquet")
    book = e169.build_book(mid, choices, "menu7p_mcap10")
    funded, account = e169.account_score(book)
    funded_books = e169.base.selected_all_alphas(raw, funded)
    log(f"book: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} final=${account.get('final_equity', float('nan')):,.0f}")
    spy = common.load_spy_daily()
    result = evaluate(spec, funded_books, gate=None, run_dir=HERE, spy_daily=spy,
                      tail_shock=common.abs_move_tail_shock,
                      input_files=[e169.base.CANDIDATES, E169_RESULTS / "dev_scores.parquet", E169_RESULTS / "choices.parquet"],
                      extra_sections=lambda r: report_sections(account, r.results["headline"]),
                      write_report=True)
    headline = result.results["headline"]
    tail = result.results.get("stress", {}).get("tail_injection", {})
    log(f"evaluated: mean={headline.get('mean'):+.3f} sharpe={headline.get('sharpe_trade'):.2f} "
        f"mc_p_loss_5={result.results.get('mc', {}).get('by_fraction', {}).get('0.05', {}).get('p_loss')}")
    log(f"tail injection: available={tail.get('available')} worst base={tail.get('base_worst_trade')} shocked={tail.get('shocked_worst_trade')}")
    write_json(RESULTS / "account_metrics.json", account)
    write_json(RESULTS / "comparison.json", {"account_metrics": account,
                                             "headline": headline,
                                             "tail_injection": tail,
                                             "mc": result.results.get("mc"),
                                             "stress_regimes": result.results.get("stress", {}).get("regimes")})
    if not args.no_ledger:
        lib.record_evaluation(HERE, spec, result.results)
    print(f"[EXP-170] report: {HERE / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
