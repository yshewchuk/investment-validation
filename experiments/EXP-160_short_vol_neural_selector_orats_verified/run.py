#!/usr/bin/env python3
"""EXP-160: clean ORATS-provenance record of corrected EXP-159 comparison."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "experiments/EXP-159_short_vol_neural_selector_conditional_exit"
sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load_module("exp159_verified", SOURCE / "run.py")


def plan(specs: dict) -> None:
    ledger = old.lib.ledger_read()
    rows = []
    for spec in specs.values():
        digest = old.lib.spec_hash(spec)
        if not (ledger["spec_hash"] == digest).any():
            rows.append({
                "id": spec["id"], "spec_hash": digest, "date": "2026-09-08",
                "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False",
            })
    if rows:
        old.lib.ledger_append(rows)
        print(f"[EXP-160] Registered {len(rows)} planned specifications", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()
    spec = old.lib.load_spec(HERE / "spec.yaml")
    specs = {arm: old.arm_spec(spec, arm) for arm in old.ALL_ARMS}
    if not args.no_ledger:
        plan(specs)
    dataset, priced = old.load_dataset()
    scores = pd.read_parquet(SOURCE / "results/oos_scores.parquet")
    scores["event_date"] = pd.to_datetime(scores["event_date"])
    diagnostics = json.loads((SOURCE / "results/fold_diagnostics.json").read_text())
    ranks = old.rank_metrics(scores)
    neural_delta = old.bootstrap(scores, old.PRIMARY, old.CONTROL)
    arithmetic_delta = old.bootstrap(scores, old.CONTROL, old.ARITHMETIC)
    ids = set(scores["event_id"].astype(str))
    priced["event_id"] = priced["event_id"].astype(str)
    trades = priced[priced["event_id"].isin(ids)].copy()
    spy = old.common.load_spy_daily()
    inputs = [
        old.CANDIDATES,
        ROOT / "data/features/panel.parquet",
        SOURCE / "results/oos_scores.parquet",
        SOURCE / "results/fold_diagnostics.json",
    ]
    evaluations = {}
    order = [arm for arm in old.ALL_ARMS if arm != old.PRIMARY] + [old.PRIMARY]
    for arm in order:
        run_dir = HERE if arm == old.PRIMARY else HERE / "arms" / arm
        if arm == old.PRIMARY:
            extra = lambda result: old.extra_sections(
                result, evaluations, ranks, neural_delta, arithmetic_delta, diagnostics
            )
        elif arm == old.ARITHMETIC:
            extra = [{"title": "Current arithmetic rule", "body": [
                "Live DYN-SV expected-PnL trailing-bar rule with the 25% spread and $10B market-cap guards, evaluated on ORATS conditional exits."
            ]}]
        else:
            extra = [{"title": "Neural arm construction", "body": [
                f"Features ({len(old.base.ARM_FEATURES[arm])}): {', '.join(old.base.ARM_FEATURES[arm])}."
            ]}]
        result = evaluate(
            specs[arm], trades, gate=old.base.PrecomputedGate(scores, arm).gate(),
            run_dir=run_dir, spy_daily=spy, tail_shock=old.base.tail_to_debit,
            input_files=inputs, extra_sections=extra,
        )
        evaluations[arm] = result
        if not args.no_ledger:
            old.lib.record_evaluation(run_dir, specs[arm], result.results)
        h = result.results["headline"]
        print(f"[EXP-160] {arm}: n={h['n']:,}, mean={h['mean']:+.4f}, Sharpe={h['sharpe_trade']:.3f}", flush=True)
    out = {
        "ranking": ranks,
        "policy_bootstrap": {
            "primary_vs_neural_control": neural_delta,
            "neural_control_vs_arithmetic": arithmetic_delta,
        },
        "headline": {arm: evaluations[arm].results["headline"] for arm in old.ALL_ARMS},
    }
    (HERE / "results").mkdir(parents=True, exist_ok=True)
    (HERE / "results/comparison.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
