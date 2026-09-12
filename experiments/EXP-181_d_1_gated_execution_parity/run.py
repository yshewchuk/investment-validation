#!/usr/bin/env python3
"""EXP-181: decision-time-valid D0/D-1 model parity."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXP179 = ROOT / "experiments" / "EXP-179_execution_clock_d1_parity"
sys.path.insert(0, str(ROOT))

from engine.evaluate import Gate, evaluate
from engine.features import FeatureContext
from engine.models.training import gate_forecast_analog as gate_mod
from experiments import common, lib


def log(message: str) -> None:
    print(f"[EXP-181] {message}", flush=True)


def pair(clock: str) -> pd.DataFrame:
    path = EXP179 / "results" / "trades" / "STR-THRU" / f"{clock}_matched.parquet"
    frame = pd.read_parquet(path)
    if "decision_date" not in frame.columns:
        frame["decision_date"] = frame["entry_date"]
    if "provenance" not in frame.columns:
        frame["provenance"] = "engine.replay"
    for col in ("event_date", "decision_date", "entry_date", "exit_date"):
        frame[col] = pd.to_datetime(frame[col])
    log(f"STR-THRU {clock}: {frame['event_id'].nunique():,} paired events")
    return frame


def gate_dataset(clock: str, trades: pd.DataFrame, refresh: bool) -> pd.DataFrame:
    path = RESULTS / "gate_datasets" / f"str_thru_{clock}.parquet"
    if path.exists() and not refresh:
        frame = pd.read_parquet(path)
        log(f"STR-THRU {clock}: gate dataset cache {len(frame):,} rows")
        return frame
    tickers = sorted(trades["ticker"].dropna().unique())
    years = pd.to_datetime(trades["event_date"]).dt.year
    context = FeatureContext.load(tickers, years=range(int(years.min()) - 1, int(years.max()) + 1))
    started = time.time()
    frame = gate_mod.build_dataset(trades, context=context)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    log(f"STR-THRU {clock}: built decision-time gate dataset {len(frame):,} rows in {time.time() - started:.0f}s")
    return frame


def extra(clock: str, state) -> list[dict]:
    return [{
        "title": "Execution-clock gated parity",
        "body": [
            "This arm uses the deployed STR-THRU forecast-plus-analog gate architecture and stored threshold. ",
            "The D-1 feature builder reads daily state and the quoted premium at decision_date, while realized return remains the unchanged D0 execution.",
        ],
        "columns": ["measure", "value"],
        "align": ["---", "---:"],
        "rows": [
            ["clock", clock],
            ["gate", state.name],
            ["stored threshold", f"{state.threshold:.6f}"],
            ["walk-forward fits", str(sum("train_rows" in x for x in state.stats))],
            ["walk-forward selections", str(sum(x.get("passed_rows", 0) for x in state.stats))],
        ],
    }]


def run_str(clocks: tuple[str, ...], *, refresh: bool, no_ledger: bool) -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    outputs = {}
    for clock in clocks:
        trades = pair(clock)
        dataset = gate_dataset(clock, trades, refresh)
        gate, state = common.make_registered_gate("STR-THRU", dataset)
        run_dir = HERE if clock == "d1" else HERE / "arms" / "str_thru_d0"
        result = evaluate(
            spec, trades, gate=gate, run_dir=run_dir, spy_daily=spy,
            tail_shock=common.abs_move_tail_shock, stress=True, mc_paths=500,
            seed=181, input_files=[
                EXP179 / "results" / "trades" / "STR-THRU" / f"{clock}_matched.parquet",
                RESULTS / "gate_datasets" / f"str_thru_{clock}.parquet",
            ],
            extra_sections=extra(clock, state), write_report=True,
        )
        outputs[clock] = {
            "headline": result.results.get("headline", {}),
            "gate_stats": state.stats,
            "report": str(result.report_path),
        }
        (RESULTS / f"str_thru_{clock}.json").write_text(
            json.dumps(outputs[clock], indent=1, default=str) + "\n"
        )
        log(f"STR-THRU {clock}: report {result.report_path}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not no_ledger and "d1" in clocks:
        lib.record_evaluation(HERE, spec, result.results)


def merge() -> None:
    outputs = {}
    for clock in ("d0", "d1"):
        path = RESULTS / f"str_thru_{clock}.json"
        if not path.exists():
            raise RuntimeError(f"missing completed arm {path}")
        outputs[clock] = json.loads(path.read_text())
    (RESULTS / "str_thru_comparison.json").write_text(
        json.dumps(outputs, indent=1, default=str) + "\n"
    )
    log("merged STR-THRU gated D0/D-1 comparison")


class CrossClockState:
    """Fit D0 folds, then score paired D-1 feature rows with that same fold."""

    def __init__(self, d0_features: pd.DataFrame, d1_features: pd.DataFrame):
        from engine.models.registry import load_registry

        entry = load_registry(missing_ok=False).champion("gate", "STR-THRU")
        self.features = tuple(entry.features)
        self.threshold = float(entry.threshold)
        self.seed = entry.seed
        self.d0 = d0_features.set_index("event_id")
        self.d1 = d1_features.set_index("event_id")
        self.model_ = None
        self.stats: list[dict] = []

    def fit(self, train: pd.DataFrame) -> None:
        ids = train["event_id"].astype(str)
        rows = self.d0.reindex(ids)
        X = rows[list(self.features)].to_numpy(dtype=float)
        y = rows["ret"].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1) & np.isfinite(y)
        self.stats.append({"train_d0_rows": int(len(rows)), "fit_rows": int(ok.sum())})
        self.model_ = gate_mod.fit(X[ok], y[ok], seed=self.seed) if int(ok.sum()) >= common.MIN_FIT_ROWS else None

    def select(self, test: pd.DataFrame) -> pd.Series:
        if self.model_ is None:
            return pd.Series(False, index=test.index)
        rows = self.d1.reindex(test["event_id"].astype(str))
        X = rows[list(self.features)].to_numpy(dtype=float)
        ok = np.isfinite(X).all(axis=1)
        score = np.full(len(rows), np.nan)
        score[ok] = self.model_.predict(X[ok])
        mask = np.isfinite(score) & (score >= self.threshold)
        self.stats.append({"score_d1_rows": int(len(rows)), "scoreable_rows": int(ok.sum()), "passed_rows": int(mask.sum())})
        return pd.Series(mask, index=test.index)


def run_cross_clock(*, no_ledger: bool) -> None:
    cache_root = ROOT / "experiments" / "EXP-182_d_1_gated_execution_parity_registered" / "results" / "gate_datasets"
    d0_features = pd.read_parquet(cache_root / "str_thru_d0.parquet")
    d1_features = pd.read_parquet(cache_root / "str_thru_d1.parquet")
    d1_trades = pair("d1")
    state = CrossClockState(d0_features, d1_features)
    gate = Gate(fit=state.fit, select=state.select, name="D0-fold-model -> D-1 features")
    spec = lib.load_spec(HERE / "spec.yaml")
    d0_ref = json.loads((cache_root.parent / "str_thru_d0.json").read_text())["headline"]
    sections = [{
        "title": "Frozen D0 model on D-1 features",
        "body": [
            "Every fold fits on D0 feature rows and D0 realized returns from prior years only.",
            "The model is not retrained on D-1 data. It scores the paired D-1 feature rows with the stored D0 champion threshold.",
        ],
        "columns": ["measure", "D0 model on D0", "D0 model on D-1"],
        "align": ["---", "---:", "---:"],
        "rows": [
            ["midpoint mean", f"{100 * d0_ref['mean']:+.3f}%", "see headline"],
            ["trade Sharpe", f"{d0_ref['sharpe_trade']:.3f}", "see headline"],
            ["threshold", f"{state.threshold:.6f}", f"{state.threshold:.6f}"],
        ],
    }]
    result = evaluate(
        spec, d1_trades, gate=gate, run_dir=HERE, spy_daily=common.load_spy_daily(),
        tail_shock=common.abs_move_tail_shock, stress=True, mc_paths=500, seed=183,
        input_files=[
            EXP179 / "results" / "trades" / "STR-THRU" / "d1_matched.parquet",
            cache_root / "str_thru_d0.parquet", cache_root / "str_thru_d1.parquet",
        ],
        extra_sections=sections, write_report=True,
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cross_clock_gate_stats.json").write_text(json.dumps(state.stats, indent=1) + "\n")
    if not no_ledger:
        lib.record_evaluation(HERE, spec, result.results)
    log(f"cross-clock report {result.report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", choices=("d0", "d1", "both"), default="both")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--cross-clock", action="store_true")
    args = parser.parse_args()
    if args.cross_clock:
        run_cross_clock(no_ledger=args.no_ledger)
        return
    if args.merge:
        merge()
        return
    clocks = ("d0", "d1") if args.clock == "both" else (args.clock,)
    run_str(clocks, refresh=args.refresh, no_ledger=args.no_ledger)


if __name__ == "__main__":
    main()
