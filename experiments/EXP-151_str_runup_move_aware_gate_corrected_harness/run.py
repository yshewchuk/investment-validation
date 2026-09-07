#!/usr/bin/env python3
"""EXP-151: evaluate frozen move-aware scores with the corrected harness."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXP148_SIGNALS = (
    ROOT
    / "experiments/EXP-148_str_runup_forecast_pnl_analog_gate"
    / "results/dataset_with_signals.parquet"
)
MOVE_AWARE_SIGNALS = (
    ROOT
    / "experiments/EXP-150_str_runup_move_aware_forecast_pnl_gate"
    / "results/dataset_with_move_aware_signals.parquet"
)
SCORE_CACHE = RESULTS / "oos_scores.parquet"
STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))


def load_runner(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load runner {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


source = load_runner(
    "exp150_runner",
    ROOT / "experiments/EXP-150_str_runup_move_aware_forecast_pnl_gate/run.py",
)
base = source.base


def log(message: str) -> None:
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-151 {elapsed:,.0f}s] {message}", flush=True)


def load_frozen_signals(spec: dict, trades: pd.DataFrame, force: bool) -> pd.DataFrame:
    del spec, trades
    if force:
        raise ValueError("EXP-151 uses frozen EXP-150 signals; --force-signals is invalid")
    RESULTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(MOVE_AWARE_SIGNALS)
    for column in ("event_date", "entry_date", "exit_date", "expiry"):
        frame[column] = pd.to_datetime(frame[column])
    log(f"Loaded {len(frame):,} frozen move-aware signal rows")
    return frame


source_report_sections = source.report_sections


def report_sections(*args, **kwargs):
    sections = source_report_sections(*args, **kwargs)
    sections.insert(
        1,
        {
            "title": "EXP-150 harness correction",
            "body": [
                "EXP-150 omitted min_train_years=0, so the generic evaluator "
                "ignored valid precomputed decisions in 2020 and 2021.",
                "This run restores the EXP-148 setting. The base gate selects "
                "520 trades again, which is the controlled-rerun invariant.",
                "The frozen forecast values, gate scores, annual thresholds, "
                "trade prices and success criteria are otherwise unchanged.",
            ],
        },
    )
    return sections


base.HERE = HERE
base.RESULTS = RESULTS
base.BASE_CACHE = EXP148_SIGNALS
base.SIGNAL_CACHE = MOVE_AWARE_SIGNALS
base.SCORE_CACHE = SCORE_CACHE
base.STARTED = STARTED
base.log = log
base.prepare_signals = load_frozen_signals
base.report_sections = report_sections


if __name__ == "__main__":
    base.main()
