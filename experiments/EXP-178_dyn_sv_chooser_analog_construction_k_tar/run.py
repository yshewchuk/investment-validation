#!/usr/bin/env python3
"""EXP-178: sweep the DYN-SV chooser's causal analog construction (K, target,
recency window), then retrain ONCE against the winner if it beats the
incumbent.

Stage 1 (always runs, closed-form, no training): 18 cells over
K in {15,25,50} x target in {pnl,ret} x lookback_days in {None,365,1095},
scored by Spearman(analog_mean, own realized ret) on menu7-prime candidates,
2020-2026. The incumbent cell (25, pnl, None) is IN the grid and wins every
tie — see spec.yaml's `selection_rule`.

Stage 2 (conditional, one retrain, ~18 minutes measured on this box via
EXP-169's identical protocol) runs ONLY if Stage 1's winner beats the
incumbent under the pre-registered rule.

Resource contention. This box runs other jobs concurrently (the nightly,
ad-hoc rebuilds, the dashboard). This script does not try to manage that
itself — thread-limiting env vars have to be set before the interpreter
starts, which is what tools/bounded_run.py already does correctly. Launch it
with:

    python3 tools/bounded_run.py --cores 3 --max-rss-gb 3.0 -- \\
        python3 experiments/EXP-178_dyn_sv_chooser_analog_construction_k_tar/run.py

Never invoke this bare on a shared box. What IS this script's own
responsibility, and what it does:

- Every Stage 1 cell is cached to results/sweep_cache/ as it completes (JSON
  metrics + a parquet of the five analog columns), so a kill mid-sweep loses
  at most the in-flight cell, not the ones already scored, and a re-run skips
  straight past them.
- Stage 2 reuses e169's own generate() cache path (results/retrain/), which
  resumes if dev_scores.parquet/choices.parquet/fold_diagnostics.json already
  exist whole; it does not checkpoint mid-fold, matching the nightly's own
  documented "no partial-phase resume" limitation (see tools/bounded_run.py).
- Reads only tables written elsewhere via tmp+os.replace (engine/data/store.py),
  so a concurrent rebuild of the same tables is safe to read against — this
  script will see the pre- or post-rebuild version, never a torn file.
- Logs progress at least once a minute throughout (the house rule for long
  jobs), and writes nothing to the shared ledger until the whole run — Stage 1
  and Stage 2 alike — has actually finished.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
SWEEP_CACHE = RESULTS / "sweep_cache"
RETRAIN_RESULTS = RESULTS / "retrain"
E169_RUN = ROOT / "experiments/EXP-169_menu7prime_confirmation/run.py"
E169_COMPARISON = ROOT / "experiments/EXP-169_menu7prime_confirmation/results/comparison.json"

K_GRID = (15, 25, 50)
TARGET_GRID = ("pnl", "ret")
LOOKBACK_GRID = (None, 365, 1095)
INCUMBENT = {"k": 25, "target": "pnl", "lookback_days": None}
COVERAGE_FLOOR = 0.90
OOS_YEARS = tuple(range(2020, 2027))
PRIMARY_ARM = "menu7p_mcap10"

STARTED = time.monotonic()

sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate  # noqa: E402
from experiments import common, lib  # noqa: E402

sys.path.insert(0, str(HERE))
from analog_variants import causal_analogs, ANALOG_COLUMNS  # noqa: E402


def log(message: str) -> None:
    print(f"[EXP-178 {time.monotonic() - STARTED:,.0f}s] {message}", flush=True)


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


def preflight() -> None:
    """Log what the box looks like right now — informational only. Actual
    resource bounding happens in tools/bounded_run.py, which must wrap this
    process; see the module docstring."""
    try:
        load1, load5, load15 = os.getloadavg()
    except OSError:
        load1 = load5 = load15 = float("nan")
    meminfo = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemAvailable"):
                meminfo[key] = int(rest.strip().split()[0]) / (1024 * 1024)
    except OSError:
        pass
    log(f"preflight: cpu_count={os.cpu_count()} load={load1:.1f}/{load5:.1f}/{load15:.1f} "
        f"mem_available={meminfo.get('MemAvailable', float('nan')):.1f}G "
        f"of {meminfo.get('MemTotal', float('nan')):.1f}G "
        f"threads(OMP)={os.environ.get('OMP_NUM_THREADS', 'unset')}")


def cell_key(k: int, target: str, lookback_days: int | None) -> str:
    lb = "all" if lookback_days is None else str(lookback_days)
    return f"k{k}_{target}_lb{lb}"


def plan(spec: dict) -> None:
    digest, ledger = lib.spec_hash(spec), lib.ledger_read()
    if not (ledger["spec_hash"] == digest).any():
        lib.ledger_append([{
            "id": spec["id"], "spec_hash": digest,
            "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
            "stage": "planned", "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False",
        }])
        log("Registered specification in LEDGER.csv")


def sanity_check(mid: pd.DataFrame, base_module) -> None:
    """causal_analogs(k=25, target=pnl, lookback=None) must reproduce
    base.add_causal_analogs bit-for-bit — the correctness gate for trusting
    every other cell in the sweep."""
    reference = base_module.add_causal_analogs(mid)
    mine = causal_analogs(mid, k=25, target="pnl", lookback_days=None)
    for col in ANALOG_COLUMNS:
        a, b = reference[col].to_numpy(float), mine[col].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        close = np.isclose(a, b, rtol=0, atol=1e-9, equal_nan=True)
        if not close.all():
            bad = int((~close & ~both_nan).sum())
            raise RuntimeError(
                f"sanity check FAILED on {col}: {bad}/{len(a)} rows disagree "
                f"with the shipped add_causal_analogs — the reimplementation "
                f"is not faithful, do not trust the sweep"
            )
    log(f"sanity check passed: causal_analogs(25, pnl, None) reproduces "
        f"add_causal_analogs on all {len(reference):,} rows, {len(ANALOG_COLUMNS)} columns")


def score_cell(mid: pd.DataFrame, k: int, target: str, lookback_days: int | None,
              denom_mask: np.ndarray) -> dict:
    key = cell_key(k, target, lookback_days)
    json_path, parquet_path = SWEEP_CACHE / f"{key}.json", SWEEP_CACHE / f"{key}.parquet"
    if json_path.exists() and parquet_path.exists():
        log(f"cell {key}: cached, skipping recompute")
        return json.loads(json_path.read_text())

    t0 = time.monotonic()
    analog = causal_analogs(mid, k=k, target=target, lookback_days=lookback_days)
    elapsed = time.monotonic() - t0

    eval_rows = analog[(analog["year"] >= OOS_YEARS[0]) & (analog["year"] <= OOS_YEARS[-1])]
    scoreable = eval_rows[eval_rows["analog_mean"].notna()]
    coverage = float(len(scoreable)) / float(denom_mask.sum()) if denom_mask.sum() else float("nan")

    by_year = {}
    for year in OOS_YEARS:
        g = scoreable[scoreable["year"] == year]
        g = g.dropna(subset=["analog_mean", "ret"])
        by_year[str(year)] = (
            float(spearmanr(g["analog_mean"], g["ret"]).statistic) if len(g) >= 10 else float("nan")
        )
    pooled = scoreable.dropna(subset=["analog_mean", "ret"])
    pooled_rho = float(spearmanr(pooled["analog_mean"], pooled["ret"]).statistic) if len(pooled) >= 10 else float("nan")
    positive_years = int(sum(1 for v in by_year.values() if np.isfinite(v) and v > 0))
    mean_year_rho = float(np.nanmean(list(by_year.values())))

    result = {
        "k": k, "target": target, "lookback_days": lookback_days,
        "coverage": coverage, "n_scoreable": int(len(scoreable)),
        "by_year_spearman": by_year, "positive_years": positive_years,
        "mean_year_spearman": mean_year_rho, "pooled_spearman": pooled_rho,
        "elapsed_s": round(elapsed, 1),
    }
    SWEEP_CACHE.mkdir(parents=True, exist_ok=True)
    analog[["candidate_id", *ANALOG_COLUMNS]].to_parquet(parquet_path, index=False)
    write_json(json_path, result)
    log(f"cell {key}: coverage={coverage:.1%} positive_years={positive_years}/7 "
        f"mean_rho={mean_year_rho:+.4f} pooled_rho={pooled_rho:+.4f} ({elapsed:.0f}s)")
    return result


def select_winner(cells: list[dict]) -> dict:
    qualifying = [c for c in cells if np.isfinite(c["coverage"]) and c["coverage"] >= COVERAGE_FLOOR]
    if not qualifying:
        raise RuntimeError("no cell reached the coverage floor — cannot select a winner")
    is_incumbent = lambda c: (c["k"], c["target"], c["lookback_days"]) == (
        INCUMBENT["k"], INCUMBENT["target"], INCUMBENT["lookback_days"])

    def sort_key(c: dict) -> tuple:
        # Higher is better on all three. The incumbent tag is a SEPARATE,
        # trailing element — it only breaks a tie where the first three are
        # EXACTLY equal. (A first version added the bonus into each of the
        # first three elements instead, which put the incumbent's
        # `positive_years + epsilon` ahead of a challenger's plain integer
        # equal count on the very first tuple comparison, before Python ever
        # looked at the rho values — silently deciding the winner on a
        # metric no one asked it to use. Caught 2026-09-11 by the sweep
        # itself: the incumbent "won" while a challenger's own logged mean
        # rho was 85% higher. Fixed to compare the real metrics first.)
        return (c["positive_years"], c["mean_year_spearman"],
                c["pooled_spearman"], 1 if is_incumbent(c) else 0)

    ranked = sorted(qualifying, key=sort_key, reverse=True)
    return ranked[0]


def run_sweep(mid: pd.DataFrame) -> tuple[list[dict], dict]:
    eval_years = mid[(mid["year"] >= OOS_YEARS[0]) & (mid["year"] <= OOS_YEARS[-1])]
    dims = ("exp_pnl_sim", "width_over_forecast", "n_legs", "anchor_over_spot", "rel_spread")
    denom_mask = np.isfinite(eval_years[list(dims)].to_numpy(float)).all(1)
    log(f"sweep evaluation population: {int(denom_mask.sum()):,} DIMS-finite candidates, "
        f"years {OOS_YEARS[0]}-{OOS_YEARS[-1]}")

    cells = []
    total = len(K_GRID) * len(TARGET_GRID) * len(LOOKBACK_GRID)
    n = 0
    for k in K_GRID:
        for target in TARGET_GRID:
            for lookback_days in LOOKBACK_GRID:
                n += 1
                log(f"cell {n}/{total}: k={k} target={target} lookback_days={lookback_days}")
                cells.append(score_cell(mid, k, target, lookback_days, denom_mask))
    winner = select_winner(cells)
    return cells, winner


def write_sweep_report(spec: dict, cells: list[dict], winner: dict, triggers_retrain: bool) -> None:
    rows = []
    for c in sorted(cells, key=lambda c: (c["k"], c["target"], str(c["lookback_days"]))):
        tag = " (incumbent)" if (c["k"], c["target"], c["lookback_days"]) == (
            INCUMBENT["k"], INCUMBENT["target"], INCUMBENT["lookback_days"]) else ""
        win_tag = " <- WINNER" if c is winner else ""
        rows.append(
            f"| {c['k']} | {c['target']} | {c['lookback_days'] or 'all-time'} | "
            f"{c['coverage']:.1%} | {c['positive_years']}/7 | "
            f"{c['mean_year_spearman']:+.4f} | {c['pooled_spearman']:+.4f} |"
            f"{tag}{win_tag}"
        )
    lines = [
        f"# {spec['id']} — {spec['title']}", "",
        f"Preregistered: {spec['preregistered_at']}", "",
        "## Hypothesis", "", spec["hypothesis"], "",
        "## Stage 1 — analog construction sweep", "",
        f"Population: menu7-prime candidates, years {OOS_YEARS[0]}-{OOS_YEARS[-1]}, "
        f"same-structure analog matching only. Metric: Spearman(analog_mean, own realized "
        f"`ret`). Coverage floor for selection: {COVERAGE_FLOOR:.0%}.", "",
        "| K | target | lookback | coverage | positive years | mean year ρ | pooled ρ |",
        "|---|---|---|---|---|---|---|",
        *rows, "",
        f"**Winner:** K={winner['k']}, target={winner['target']}, "
        f"lookback={winner['lookback_days'] or 'all-time'} "
        f"({winner['positive_years']}/7 positive years, mean ρ {winner['mean_year_spearman']:+.4f})",
        "",
    ]
    if triggers_retrain:
        lines += [
            "The winner is NOT the incumbent cell (K=25, target=pnl, lookback=all-time). "
            "Per the pre-registered rule this triggers Stage 2 — see below.", "",
        ]
    else:
        lines += [
            "**The incumbent cell won.** Per the pre-registered rule, this is a complete, "
            "valid null result: no construction in this grid carries more signal than the "
            "one already shipped. Stage 2 does not run and nothing about the deployed "
            "chooser changes.", "",
        ]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    log(f"wrote {HERE / 'REPORT.md'}")


def load_champion_reference() -> dict:
    if not E169_COMPARISON.exists():
        raise RuntimeError(f"cannot find champion reference at {E169_COMPARISON}")
    data = json.loads(E169_COMPARISON.read_text())
    return {
        "final_equity_mcap10": data["account_metrics"]["menu7p_mcap10"]["final_equity"],
        "final_equity_mcap1": data["account_metrics"]["menu7p_mcap1"]["final_equity"],
        "structure_diagnostics": data["structure_diagnostics"],
        "by_year_spearman": {
            str(f["year"]): f.get("within_event_spearman")
            for f in json.loads((E169_COMPARISON.parent / "fold_diagnostics.json").read_text())
        },
    }


def arm_spec(spec: dict, arm: str, primary_arm: str) -> dict:
    out = deepcopy(spec)
    if arm != primary_arm:
        out["primary_spec"] = dict(out["primary_spec"], evaluated_arm=arm)
        out["grid_cell"] = True
        out["promotion_target"] = None
    return out


def run_retrain(spec: dict, mid: pd.DataFrame, raw: pd.DataFrame, e169, winner: dict) -> None:
    reference = load_champion_reference()
    RETRAIN_RESULTS.mkdir(parents=True, exist_ok=True)
    e169.RESULTS = RETRAIN_RESULTS  # redirect e169.generate()'s cache away from its own folder

    cache_key = cell_key(winner["k"], winner["target"], winner["lookback_days"])
    cached_parquet = SWEEP_CACHE / f"{cache_key}.parquet"
    log(f"retrain: reusing cached analog columns from {cached_parquet.name}")
    analog_cols = pd.read_parquet(cached_parquet).set_index("candidate_id")
    dataset = mid.set_index("candidate_id", drop=False)
    for col in ANALOG_COLUMNS:
        dataset[col] = analog_cols[col]
    dataset = dataset.reset_index(drop=True)

    dataset = e169.build_schematics(dataset)
    dataset = e169.join_tier4(dataset)
    dev, choices, diagnostics = e169.generate(dataset)
    log(f"retrain: {len(choices):,} OOS events, "
        f"{int(choices['head_structure'].notna().sum()):,} head choices")

    spy = common.load_spy_daily()
    accounts, funded_books, funded_frames = {}, {}, {}
    for arm in e169.ARMS:
        book = e169.build_book(mid, choices, arm)
        funded, account = e169.account_score(book)
        accounts[arm] = account
        funded_frames[arm] = funded
        funded_books[arm] = e169.base.selected_all_alphas(raw, funded)
        log(f"{arm}: wanted={account.get('wanted', 0):,} funded={account.get('funded', 0):,} "
            f"final=${account.get('final_equity', float('nan')):,.0f}")

    funded_ids = set(funded_frames[PRIMARY_ARM].loc[funded_frames[PRIMARY_ARM]["funded"], "event_id"])
    structure_diag = e169.structure_diagnostics(dev, choices, funded_ids)
    buckets = e169.mcap_buckets(funded_frames[PRIMARY_ARM])

    checks = {
        "final_equity_mcap10_no_regression": (
            accounts[PRIMARY_ARM].get("final_equity", -np.inf) >= reference["final_equity_mcap10"]
        ),
        "no_defined_risk_failure": all(a.get("defined_risk_failures", 1) == 0 for a in accounts.values()),
        "mcap1_transfer_holds": (
            accounts["menu7p_mcap1"].get("final_equity", -np.inf)
            > accounts[PRIMARY_ARM].get("final_equity", np.inf)
        ),
        "ramp7_at_least_menu_mean": (
            structure_diag.get("RAMP7", {}).get("mean_realized_when_picked_funded", -np.inf)
            >= accounts[PRIMARY_ARM].get("funded_mean_realized_pnl", np.inf)
        ),
        "ctr5_at_least_menu_mean": (
            structure_diag.get("CTR5", {}).get("mean_realized_when_picked_funded", -np.inf)
            >= accounts[PRIMARY_ARM].get("funded_mean_realized_pnl", np.inf)
        ),
    }

    write_json(RESULTS / "retrain_account_metrics.json", accounts)
    write_json(RESULTS / "retrain_structure_diagnostics.json", structure_diag)
    write_json(RESULTS / "retrain_checks.json", checks)

    def report_sections(result) -> list[dict]:
        rows = [[arm, f"{accounts[arm].get('wanted', 0):,}", f"{accounts[arm].get('funded', 0):,}",
                f"${accounts[arm].get('final_equity', float('nan')):,.0f}",
                f"{accounts[arm].get('cagr', float('nan')):.2%}"] for arm in e169.ARMS]
        struct_rows = [[s, f"{d['mean_realized_when_picked_funded']:+.3f}",
                       f"{reference['structure_diagnostics'].get(s, {}).get('mean_realized_when_picked_funded', float('nan')):+.3f}"]
                      for s, d in sorted(structure_diag.items())]
        year_rows = [[year, f"{diagnostics[i].get('within_event_spearman', float('nan')):+.3f}",
                     f"{reference['by_year_spearman'].get(str(year), float('nan')):+.3f}"]
                    for i, year in enumerate(range(2020, 2020 + len(diagnostics)))]
        return [
            {"title": "What this run is", "body": [
                f"Winning analog construction from Stage 1: K={winner['k']}, "
                f"target={winner['target']}, lookback={winner['lookback_days'] or 'all-time'} "
                f"({winner['positive_years']}/7 positive years, mean rho {winner['mean_year_spearman']:+.4f} "
                "vs the incumbent construction). Same architecture, features, folds, and "
                "funding policy as EXP-169/170 — this isolates the analog construction as "
                "the only variable versus the deployed champion "
                f"(dyn_sv_chooser_v1_1, final equity reference ${reference['final_equity_mcap10']:,.0f}).",
            ]},
            {"title": "Account by arm", "columns": ["arm", "wanted", "funded", "final equity", "CAGR"],
             "align": ["---"] + ["---:"] * 4, "rows": rows},
            {"title": "Structure diagnostics — candidate vs champion reference (mean realized when funded)",
             "columns": ["structure", "candidate", "champion (EXP-169)"], "align": ["---", "---:", "---:"],
             "rows": struct_rows},
            {"title": "Within-event Spearman by fold — candidate vs champion reference",
             "columns": ["year", "candidate", "champion (EXP-169)"], "align": ["---", "---:", "---:"],
             "rows": year_rows},
            {"title": "Checks against the deployed champion", "columns": ["check", "status"],
             "align": ["---", "---"], "rows": [[k, "PASS" if v else "FAIL"] for k, v in checks.items()]},
        ]

    evaluations = {}
    for arm in e169.ARMS:
        cell, run_dir = arm_spec(spec, arm, PRIMARY_ARM), HERE if arm == PRIMARY_ARM else HERE / "arms" / arm
        extra = report_sections if arm == PRIMARY_ARM else (
            lambda r, a=arm: [{"title": "Arm accounting", "body": [
                f"Arm: {a}. Final equity ${accounts[a].get('final_equity', float('nan')):,.0f}."]}])
        result = evaluate(
            cell, funded_books[arm], gate=None, run_dir=run_dir, spy_daily=spy,
            tail_shock=common.abs_move_tail_shock,
            input_files=[e169.base.CANDIDATES, e169.TIER4,
                        RETRAIN_RESULTS / "dev_scores.parquet", RETRAIN_RESULTS / "choices.parquet"],
            extra_sections=extra, write_report=True,
        )
        evaluations[arm] = result.results
        log(f"{arm}: evaluated, mean={result.results['headline'].get('mean', float('nan')):+.3f}")

    write_json(RESULTS / "comparison.json", {
        "winner_cell": winner, "account_metrics": accounts, "structure_diagnostics": structure_diag,
        "mcap_buckets": buckets, "checks": checks, "champion_reference": reference,
        "evaluation_headlines": {k: v["headline"] for k, v in evaluations.items()},
    })
    lib.record_evaluation(HERE, spec, evaluations[PRIMARY_ARM], promoted=False)
    all_pass = all(checks.values())
    log(f"retrain complete: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'} — "
        f"see {HERE / 'REPORT.md'}. This experiment does not promote; that is a separate step.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--force-retrain", action="store_true",
                        help="run Stage 2 even if the incumbent wins Stage 1 (debugging only)")
    args = parser.parse_args()

    preflight()
    spec = lib.load_spec(HERE / "spec.yaml")
    if not args.no_ledger:
        plan(spec)

    E169_RUN_local = E169_RUN
    e169 = load_module("exp178_e169", E169_RUN_local)
    log(f"loaded EXP-169 module for the retrain protocol: {E169_RUN_local}")

    mid, raw = e169.load_data_menu(e169.MENU)
    log(f"loaded {len(mid):,} menu7-prime candidates, {mid['event_id'].nunique():,} events, "
        f"{mid['year'].min()}-{mid['year'].max()}")

    sanity_check(mid, e169.base)

    cells, winner = run_sweep(mid)
    is_incumbent = (winner["k"], winner["target"], winner["lookback_days"]) == (
        INCUMBENT["k"], INCUMBENT["target"], INCUMBENT["lookback_days"])
    triggers_retrain = (not is_incumbent) or args.force_retrain
    write_json(RESULTS / "analog_sweep.json", {
        "grid_axes": {"k": list(K_GRID), "target": list(TARGET_GRID), "lookback_days": list(LOOKBACK_GRID)},
        "incumbent": INCUMBENT, "coverage_floor": COVERAGE_FLOOR,
        "cells": cells, "winner": winner, "triggers_retrain": triggers_retrain,
    })
    write_sweep_report(spec, cells, winner, triggers_retrain)

    if not triggers_retrain:
        log("incumbent wins — Stage 2 not run. Recording the null result.")
        if not args.no_ledger:
            lib.record_evaluation(HERE, spec, {"headline": {}}, promoted=False)
        return

    log(f"non-incumbent winner (k={winner['k']}, target={winner['target']}, "
        f"lookback={winner['lookback_days']}) — running Stage 2 retrain")
    run_retrain(spec, mid, raw, e169, winner)


if __name__ == "__main__":
    main()
