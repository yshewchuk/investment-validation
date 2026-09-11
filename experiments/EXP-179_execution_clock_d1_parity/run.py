#!/usr/bin/env python3
"""EXP-179 paired D0/D-1 execution-clock validation.

Each arm receives its own generated REPORT.md under
arms/<strategy>/<clock>/REPORT.md. The runner is cache-first and resumes by
strategy. It only reads the normalized option-chain store.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from engine import replay
from engine.build_trades import event_universe
from engine.data import store
from engine.evaluate import evaluate
from engine.structures import (
    D1_DECISION_OFFSETS,
    STRUCTURES,
    execution_variant_label,
    with_decision_offset,
)
from experiments import common, lib

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ARMS = HERE / "arms"
ALPHAS = tuple(replay.ALPHA_GRID)


def cache_path(strategy: str, clock: str) -> Path:
    return RESULTS / "trades" / strategy / f"${clock}.parquet"


def complete_events(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    counts = frame.groupby("event_id")["fill_alpha"].nunique()
    return set(counts[counts == len(ALPHAS)].index.astype(str))


def cached_pair(strategy: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    d0_path, d1_path = cache_path(strategy, "d0"), cache_path(strategy, "d1")
    if not d0_path.exists() or not d1_path.exists():
        return None
    d0, d1 = pd.read_parquet(d0_path), pd.read_parquet(d1_path)
    if d0.empty or d1.empty:
        print(f"  [${strategy}] incomplete replay cache ignored", flush=True)
        return None
    print(
        f"  [${strategy}] replay cache: D0 ${d0['event_id'].nunique():,}, "
        f"D-1 ${d1['event_id'].nunique():,}",
        flush=True,
    )
    return d0, d1


def existing_d0(strategy: str, label: str) -> pd.DataFrame:
    """Use the canonical D0 replay where it already exists."""
    trades = store.read_table("trades")
    rows = trades[
        (trades["strategy"] == strategy)
        & (trades["variant"] == label)
        & (trades["provenance"].astype(str) == "engine.replay")
    ].copy()
    if not rows.empty:
        print(f"  [${strategy}] using ${rows['event_id'].nunique():,} stored D0 events", flush=True)
    return rows


def replay_pair(strategy: str, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cached = cached_pair(strategy)
    if cached is not None:
        return cached
    d0_structure = STRUCTURES[strategy]()
    d1_structure = with_decision_offset(d0_structure, D1_DECISION_OFFSETS[strategy])
    d0_label = execution_variant_label(d0_structure)
    d1_label = execution_variant_label(d1_structure)
    d0 = existing_d0(strategy, d0_label)
    d0_frames: list[pd.DataFrame] = []
    d1_frames: list[pd.DataFrame] = []
    started = time.time()
    grouped = events.assign(year=pd.to_datetime(events["event_date"]).dt.year)
    for year, block in grouped.groupby("year", sort=True):
        block = block.drop(columns="year").reset_index(drop=True)
        d1_plan = replay.plan_events(d1_structure, block)
        keys = d1_plan.chain_keys
        index = replay.load_chain_index(keys, progress_every=0) if keys else replay.ChainIndex({})
        d0_year = None
        if d0.empty:
            d0_year = replay.replay(
                strategy, block, structure=d0_structure, variant=d0_label,
                index=index, progress_every=0,
            )
            d0_frames.append(d0_year.trades)
        d1 = replay.replay(
            strategy, block, structure=d1_structure, variant=d1_label,
            index=index, progress_every=0,
        )
        d1_frames.append(d1.trades)
        del index
        print(
            f"  [${strategy}] ${year}: D0 ${d0_year.n_trades if d0_year else 0:,}, D-1 ${d1.n_trades:,}; "
            f"${time.time() - started:.0f}s elapsed",
            flush=True,
        )
    if d0.empty:
        d0 = pd.concat(d0_frames, ignore_index=True) if d0_frames else pd.DataFrame()
    d1 = pd.concat(d1_frames, ignore_index=True) if d1_frames else pd.DataFrame()
    for frame, clock in ((d0, "d0"), (d1, "d1")):
        path = cache_path(strategy, clock)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return d0, d1


def matched_pair(d0: pd.DataFrame, d1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    shared = complete_events(d0) & complete_events(d1)
    left = d0[d0["event_id"].astype(str).isin(shared)].copy()
    right = d1[d1["event_id"].astype(str).isin(shared)].copy()
    left["event_id"], right["event_id"] = left["event_id"].astype(str), right["event_id"].astype(str)
    mid0 = left[np.isclose(left["fill_alpha"], 0.5)][["event_id", "ret"]]
    mid1 = right[np.isclose(right["fill_alpha"], 0.5)][
        ["event_id", "ret", "entry_cost", "quoted_cost"]
    ]
    paired = mid0.merge(mid1, on="event_id", suffixes=("_d0", "_d1"), how="inner")
    ret_diff = paired["ret_d1"] - paired["ret_d0"]
    quote_drift = paired["entry_cost"] / paired["quoted_cost"] - 1.0
    receipt = {
        "d0_priced_events": int(d0["event_id"].nunique()),
        "d1_priced_events": int(d1["event_id"].nunique()),
        "matched_events": int(len(paired)),
        "d0_coverage_of_matched": float(len(paired) / d0["event_id"].nunique()) if len(d0) else float("nan"),
        "d1_coverage_of_matched": float(len(paired) / d1["event_id"].nunique()) if len(d1) else float("nan"),
        "mid_return_diff_d1_minus_d0_mean": float(ret_diff.mean()) if len(ret_diff) else float("nan"),
        "mid_return_diff_d1_minus_d0_median": float(ret_diff.median()) if len(ret_diff) else float("nan"),
        "mid_return_diff_d1_minus_d0_p10": float(ret_diff.quantile(0.10)) if len(ret_diff) else float("nan"),
        "mid_return_diff_d1_minus_d0_p90": float(ret_diff.quantile(0.90)) if len(ret_diff) else float("nan"),
        "quoted_to_entry_cost_drift_median": float(quote_drift.median()) if len(quote_drift) else float("nan"),
        "quoted_to_entry_cost_drift_abs_median": float(quote_drift.abs().median()) if len(quote_drift) else float("nan"),
    }
    return left, right, receipt


def execution_section(strategy: str, clock: str, receipt: dict) -> list[dict]:
    return [{
        "title": "Paired execution-clock receipt",
        "note": (
            "Both arms use the same matched event IDs and unchanged D0 entry and exit dates. "
            "D-1 names contracts at the prior close and holds that name to the D0 fill. "
            "Neither arm uses a gate or chooser, so this is base-exposure evidence only."
        ),
        "columns": ["measure", "value"],
        "align": ["---", "---:"],
        "rows": [
            ["strategy", strategy],
            ["arm", clock],
            ["D0 priced events", f"${receipt['d0_priced_events']:,}"],
            ["D-1 priced events", f"${receipt['d1_priced_events']:,}"],
            ["matched events", f"${receipt['matched_events']:,}"],
            ["mean D-1 minus D0 mid return", f"${100 * receipt['mid_return_diff_d1_minus_d0_mean']:+.3f}%"],
            ["median D-1 minus D0 mid return", f"${100 * receipt['mid_return_diff_d1_minus_d0_median']:+.3f}%"],
            ["D-1 median quote-to-entry drift", f"${100 * receipt['quoted_to_entry_cost_drift_median']:+.3f}%"],
            ["D-1 median absolute quote-to-entry drift", f"${100 * receipt['quoted_to_entry_cost_drift_abs_median']:.3f}%"],
        ],
    }]


def arm_report(
    spec: dict,
    strategy: str,
    clock: str,
    trades: pd.DataFrame,
    receipt: dict,
    spy: pd.DataFrame,
) -> dict:
    arm_dir = ARMS / strategy / clock
    report_path = arm_dir / "REPORT.md"
    if report_path.exists():
        print(f"  [${strategy}/${clock}] report cache: ${report_path}", flush=True)
        metrics = sorted((arm_dir / "results").glob("metrics_*.json"))
        return json.loads(metrics[-1].read_text()) if metrics else {}
    print(
        f"  [${strategy}/${clock}] evaluating ${trades['event_id'].nunique():,} matched events",
        flush=True,
    )
    result = evaluate(
        spec,
        trades,
        gate=None,
        run_dir=arm_dir,
        spy_daily=spy,
        tail_shock=common.abs_move_tail_shock,
        stress=True,
        mc_paths=500,
        seed=179,
        input_files=[cache_path(strategy, f"${clock}_matched")],
        extra_sections=execution_section(strategy, clock, receipt),
        write_report=True,
    )
    lib.record_evaluation(HERE, spec, result.results)
    print(f"  [${strategy}/${clock}] report: ${result.report_path}", flush=True)
    return result.results


def write_comparison(rows: list[dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "comparison.json").write_text(json.dumps(rows, indent=1, default=str) + "\n")
    lines = [
        "# EXP-179 - D0 / D-1 paired execution comparison",
        "",
        "Every arm report is generated under arms/<strategy>/<clock>/REPORT.md.",
        "",
        "| strategy | matched events | D0 mid mean | D-1 mid mean | D-1 minus D0 paired mean | D0 win | D-1 win | D-1 quote drift median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| ${row['strategy']} | ${row['matched_events']:,} | "
            f"${100 * row['d0_mean']:+.3f}% | ${100 * row['d1_mean']:+.3f}% | "
            f"${100 * row['paired_diff_mean']:+.3f}% | ${100 * row['d0_win']:.1f}% | "
            f"${100 * row['d1_win']:.1f}% | ${100 * row['quote_drift_median']:+.3f}% |"
        )
    (HERE / "COMPARISON.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    events = event_universe()
    spy = common.load_spy_daily()
    summaries: list[dict] = []
    started = time.time()
    print(
        f"[EXP-179] ${len(events):,} known-session events; "
        f"${len(D1_DECISION_OFFSETS)} paired strategies",
        flush=True,
    )
    for number, strategy in enumerate(D1_DECISION_OFFSETS, start=1):
        print(f"[EXP-179] ${number}/${len(D1_DECISION_OFFSETS)} ${strategy}", flush=True)
        d0_raw, d1_raw = replay_pair(strategy, events)
        d0, d1, receipt = matched_pair(d0_raw, d1_raw)
        if d0.empty or d1.empty:
            raise RuntimeError(f"${strategy}: no matched D0/D-1 events")
        for frame, clock in ((d0, "d0_matched"), (d1, "d1_matched")):
            path = cache_path(strategy, clock)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        receipt_path = RESULTS / "receipts" / f"${strategy}.json"
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(receipt, indent=1, default=str) + "\n")
        d0_results = arm_report(spec, strategy, "d0", d0, receipt, spy)
        d1_results = arm_report(spec, strategy, "d1", d1, receipt, spy)
        d0_head = d0_results.get("headline", {})
        d1_head = d1_results.get("headline", {})
        summaries.append({
            "strategy": strategy,
            "matched_events": receipt["matched_events"],
            "paired_diff_mean": receipt["mid_return_diff_d1_minus_d0_mean"],
            "quote_drift_median": receipt["quoted_to_entry_cost_drift_median"],
            "d0_mean": d0_head.get("mean", float("nan")),
            "d1_mean": d1_head.get("mean", float("nan")),
            "d0_win": d0_head.get("win_rate", float("nan")),
            "d1_win": d1_head.get("win_rate", float("nan")),
        })
        write_comparison(summaries)
        print(f"[EXP-179] ${strategy} complete; ${time.time() - started:.0f}s elapsed", flush=True)
    print(f"[EXP-179] complete in ${time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
