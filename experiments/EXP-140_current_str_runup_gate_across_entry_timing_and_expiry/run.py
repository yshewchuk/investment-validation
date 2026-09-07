#!/usr/bin/env python3
"""Causal application of the current STR-RUNUP gate to EXP-135s 12 cells."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate, walk_forward
from engine.features import FeatureContext
from engine.models.registry import load_registry
from engine.models.training import gate as gate_mod
from experiments import common, lib

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SOURCE = ROOT / "experiments" / "EXP-135_runup_coarse_timing_expiry"
SOURCE_RESULTS = SOURCE / "results"
STRATEGY = "STR-RUNUP"
RULES = ("first_post_event", "second_post_event", "exit_plus_14_calendar_days")
OFFSETS = (21, 14, 7, 3)
PRIMARY = ("first_post_event", 3)
STRICT_START = 2020


def log(message):
    elapsed = time.monotonic() - STARTED
    print(f"[EXP-140 {elapsed:,.0f}s] {message}", flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clustered_interval(values, dates, seed=140, draws=2000):
    frame = pd.DataFrame({
        "value": np.asarray(values, dtype=float),
        "week": pd.to_datetime(dates).dt.to_period("W").astype(str).to_numpy(),
    }).dropna()
    if len(frame) < 2:
        return [None, None]
    grouped = frame.groupby("week").value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    while len(estimates) < draws:
        batch = min(200, draws - len(estimates))
        idx = rng.integers(0, len(grouped), size=(batch, len(grouped)))
        estimates.extend((sums[idx].sum(axis=1) / counts[idx].sum(axis=1)).tolist())
    return np.quantile(estimates, [.025, .975]).tolist()


def net_stats(frame):
    result = {"n": 0}
    mid = frame[np.isclose(frame.fill_alpha, .5)]
    result["n"] = int(mid.event_id.nunique())
    for alpha, label in ((0., "worst"), (.25, "quarter"), (.5, "mid"),
                         (.75, "three_quarters"), (1., "best")):
        values = frame[np.isclose(frame.fill_alpha, alpha)]["ret_net"]
        result[label] = float(values.mean()) if len(values) else None
    if not len(mid):
        return result
    result.update(
        median=float(mid.ret_net.median()),
        win_rate=float((mid.ret_net > 0).mean()),
        capital_weighted=float(
            (mid.exit_value_net - mid.entry_cost_net).sum()
            / mid.entry_cost_net.sum()),
        ci95=clustered_interval(mid.ret_net, mid.event_date),
        years_positive=int(sum(
            group.ret_net.mean() > 0
            for _, group in mid.groupby(mid.event_date.dt.year))),
        years_evaluated=int(mid.event_date.dt.year.nunique()),
    )
    means = [result[name] for name in
             ("worst", "quarter", "mid", "three_quarters", "best")]
    result["breakeven_alpha"] = None
    if means[0] >= 0:
        result["breakeven_alpha"] = 0.
    elif means[-1] >= 0:
        for index in range(1, 5):
            if means[index] >= 0:
                left = (index - 1) * .25
                result["breakeven_alpha"] = (
                    left + .25 * (-means[index - 1])
                    / (means[index] - means[index - 1]))
                break
    return result


def load_grid():
    path = SOURCE_RESULTS / "grid_trades.parquet"
    frame = pd.read_parquet(path)
    frame["event_date"] = pd.to_datetime(frame.event_date)
    frame["entry_date"] = pd.to_datetime(frame.entry_date)
    frame["exit_date"] = pd.to_datetime(frame.exit_date)
    mid = frame[np.isclose(frame.fill_alpha, .5)]
    counts = mid.groupby("event_id").size()
    matched_ids = set(counts[counts == 12].index)
    matched = frame[frame.event_id.isin(matched_ids)].copy()
    if len(matched_ids) != 3797:
        raise RuntimeError(
            f"EXP-135 matched cohort changed: {len(matched_ids)} rather than 3797")
    return matched, matched_ids


def feature_dataset(cell, path, context):
    if path.exists():
        frame = pd.read_parquet(path)
        log(f"Feature cache: {path.parent.parent.name}, {len(frame):,} rows")
        return frame
    log(f"Building entry-as-of features: {path.parent.parent.name}")
    frame = gate_mod.build_dataset(
        cell, panel=context.panel, daily=context.daily)
    frame.to_parquet(path, index=False)
    log(f"Feature dataset written: {len(frame):,} rows")
    return frame


def strict_selection(cell, dataset):
    gate, state = common.make_registered_gate(STRATEGY, dataset)
    wf = walk_forward(cell, gate, min_train_years=2)
    selected = wf["selected"].copy()
    strict = selected[selected.event_date.dt.year >= STRICT_START].copy()
    scoreable = set(dataset.loc[
        np.isfinite(dataset[list(gate_mod.FEATURES)].to_numpy(dtype=float)).all(axis=1),
        "event_id"])
    base = cell[
        (cell.event_date.dt.year >= STRICT_START)
        & cell.event_id.isin(scoreable)].copy()
    return strict, base, wf, state, scoreable


def cell_spec(spec, rule, offset):
    if (rule, offset) == PRIMARY:
        return spec
    result = dict(spec)
    result["primary_spec"] = dict(spec["primary_spec"])
    result["primary_spec"]["entry_sessions"] = offset
    result["primary_spec"]["expiry_rule"] = rule
    result["grid_cell"] = True
    return result


def selection_record(rule, offset, strict, base, wf, scoreable):
    mid = strict[np.isclose(strict.fill_alpha, .5)]
    return {
        "rule": rule,
        "offset": offset,
        "strict_start": STRICT_START,
        "selected_event_ids": sorted(map(str, mid.event_id.unique())),
        "scoreable_event_ids": sorted(map(str, scoreable)),
        "n_selected": int(mid.event_id.nunique()),
        "n_scoreable": int(base[
            np.isclose(base.fill_alpha, .5)].event_id.nunique()),
        "diagnostics": wf["diagnostics"],
        "audit": wf["audit"],
    }


def compare_cells(selections):
    rows = []
    mid_frames = {
        key: frame[np.isclose(frame.fill_alpha, .5)].set_index("event_id")
        for key, frame in selections.items()
    }
    comparisons = []
    for rule in RULES:
        for earlier, later in zip(OFFSETS[:-1], OFFSETS[1:]):
            comparisons.append(((rule, earlier), (rule, later), "adjacent_timing"))
    for offset in OFFSETS:
        comparisons.extend([
            (("first_post_event", offset), ("second_post_event", offset), "expiry"),
            (("first_post_event", offset), ("exit_plus_14_calendar_days", offset), "expiry"),
            (("second_post_event", offset), ("exit_plus_14_calendar_days", offset), "expiry"),
        ])
    for reference, challenger, kind in comparisons:
        left = mid_frames[reference]
        right = mid_frames[challenger]
        common_ids = left.index.intersection(right.index)
        union_ids = left.index.union(right.index)
        delta = right.loc[common_ids, "ret_net"] - left.loc[common_ids, "ret_net"]
        dates = right.loc[common_ids, "event_date"]
        rows.append({
            "kind": kind,
            "reference_rule": reference[0],
            "reference_offset": reference[1],
            "challenger_rule": challenger[0],
            "challenger_offset": challenger[1],
            "n_common_selected": int(len(common_ids)),
            "selection_jaccard": float(len(common_ids) / len(union_ids))
            if len(union_ids) else None,
            "challenger_minus_reference": float(delta.mean()) if len(delta) else None,
            "ci95": clustered_interval(delta, dates) if len(delta) else [None, None],
        })
    return rows


def make_plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(summary)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, metric, title in (
        (axes[0], "gated_mid", "Strict gated 2020-2026"),
        (axes[1], "gate_lift", "Gate lift over scoreable base")):
        for rule in RULES:
            rows = frame[frame.rule == rule].set_index("offset").reindex(OFFSETS)
            axis.plot(OFFSETS, rows[metric] * 100, marker="o", label=rule)
        axis.axhline(0, color="gray", linewidth=.8)
        axis.set_xticks(OFFSETS)
        axis.invert_xaxis()
        axis.set_xlabel("Trading sessions before pre-print exit")
        axis.set_title(title)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Net mean return per trade (%)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Current STR-RUNUP gate, refit causally within each cell")
    figure.tight_layout()
    path = HERE / "figures" / "gated_timing_expiry.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)


def pct(value):
    return "n/a" if value is None else f"{100 * value:+.2f}%"


def report_sections(summary, paired, yearly, sensitivities, registry_entry):
    rows = []
    for item in summary:
        rows.append([
            item["rule"], f"T-{item['offset']}",
            f"{item['n_scoreable']:,}", f"{item['n_selected']:,}",
            pct(item["base_mid"]), pct(item["gated_mid"]), pct(item["gate_lift"]),
            pct(item["worst"]), pct(item["best"]),
            pct(item["capital_weighted"]),
            f"{item['years_positive']}/{item['years_evaluated']}",
            f"{item['breakeven_alpha']:.3f}"
            if item["breakeven_alpha"] is not None else "n/a",
        ])
    paired_rows = []
    for item in paired:
        lo, hi = item["ci95"]
        paired_rows.append([
            item["kind"],
            f"{item['reference_rule']} T-{item['reference_offset']}",
            f"{item['challenger_rule']} T-{item['challenger_offset']}",
            f"{item['n_common_selected']:,}",
            f"{100 * item['selection_jaccard']:.1f}%"
            if item["selection_jaccard"] is not None else "n/a",
            pct(item["challenger_minus_reference"]),
            f"[{pct(lo)}, {pct(hi)}]" if lo is not None else "n/a",
        ])
    primary_year_rows = []
    for item in yearly:
        if (item["rule"], item["offset"]) != PRIMARY:
            continue
        primary_year_rows.append([
            str(item["year"]), f"{item['n']:,}", pct(item["mid"]),
            pct(item["capital_weighted"]),
            f"{item['years_positive']}/{item['years_evaluated']}",
        ])
    sensitivity_rows = []
    for item in sensitivities:
        if item["offset"] not in (7, 3):
            continue
        sensitivity_rows.append([
            item["rule"], f"T-{item['offset']}", item["slice"],
            f"{item['n']:,}", pct(item["mid"]),
            pct(item.get("capital_weighted")),
        ])
    return [
        {
            "title": "Causal gate contract",
            "body": [
                f"- Champion: {registry_entry.id}; stored threshold {registry_entry.threshold:.8f}.",
                "- The same registered 43-feature model class and threshold are used in every cell.",
                "- Each cells features are observed at its own entry close, and each annual fit sees only prior years from that cell.",
                "- Strict comparison results exclude the 2018-2019 cold-start years in which the standard harness keeps trades ungated.",
                "- Gate targets and standard evaluator headlines use gross quote returns. The tables below deduct $0.65 per contract per side.",
                "",
                "![Gated timing and expiry](figures/gated_timing_expiry.png)",
            ],
        },
        {
            "title": "All declared gated cells",
            "note": (
                "Independent cell means combine selection and payoff behavior. "
                "Scoreable base means use the same feature-complete 2020-2026 cell "
                "before threshold selection."),
            "columns": [
                "expiry rule", "entry", "scoreable", "selected", "base mid",
                "gated mid", "lift", "worst", "best", "capital weighted",
                "years+", "breakeven alpha"],
            "align": ["---", "---:"] + ["---:"] * 10,
            "rows": rows,
        },
        {
            "title": "Common-admission paired differences",
            "note": (
                "Positive means the challenger earned more on events admitted by "
                "both cells. Jaccard measures overlap between the two selected sets. "
                "Intervals use 2,000 earnings-week cluster resamples."),
            "columns": [
                "comparison", "reference", "challenger", "common selected",
                "selection Jaccard", "difference", "95% interval"],
            "align": ["---", "---", "---", "---:", "---:", "---:", "---:"],
            "rows": paired_rows,
        },
        {
            "title": "Primary strict-gated result by year",
            "columns": ["year", "selected", "net mid", "capital weighted", "years+"],
            "align": ["---", "---:", "---:", "---:", "---:"],
            "rows": primary_year_rows,
        },
        {
            "title": "Execution and right-tail sensitivity at T-7 and T-3",
            "note": (
                "Every slice is applied after causal gate selection. The full "
                "12-cell table is preserved in results/sensitivities.json."),
            "columns": [
                "expiry rule", "entry", "slice", "n", "net mid",
                "capital weighted"],
            "align": ["---", "---:", "---", "---:", "---:", "---:"],
            "rows": sensitivity_rows,
        },
        {
            "title": "Interpretation boundary",
            "body": [
                "- A better independent gated mean may come from different selection, different entry timing, or both.",
                "- The paired table isolates payoff differences but conditions on the smaller intersection of two gates.",
                "- Twelve cells were declared. Rankings are directional and carry no multiplicity adjustment.",
                "- EOD ORATS quotes test modeled fill assumptions; they do not establish achieved execution.",
            ],
        },
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ledger", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    spec = lib.load_spec(HERE / "spec.yaml")
    entry = load_registry(missing_ok=False).champion("gate", STRATEGY)
    if abs(float(entry.threshold) - float(spec["primary_spec"]["gate_threshold"])) > 1e-12:
        raise RuntimeError("Registered gate threshold changed after preregistration")
    if tuple(entry.features) != tuple(gate_mod.FEATURES):
        raise RuntimeError("Registered gate feature list changed after preregistration")

    trades, matched_ids = load_grid()
    log(f"Loaded {len(matched_ids):,} fully matched events; ORATS calls: 0")
    manifest = {
        "source_grid": str(SOURCE_RESULTS / "grid_trades.parquet"),
        "source_grid_sha256": file_sha256(SOURCE_RESULTS / "grid_trades.parquet"),
        "source_manifest_sha256": file_sha256(SOURCE_RESULTS / "input_manifest.json"),
        "registry_sha256": file_sha256(ROOT / "engine" / "models" / "registry.json"),
        "gate_artifact_sha256": entry.artifact_sha256,
        "events": len(matched_ids),
        "quota_calls": 0,
    }
    write_json(RESULTS / "input_manifest.json", manifest)

    log("Loading shared causal panel and daily market state")
    tickers = sorted(trades.ticker.unique())
    context = FeatureContext.load(
        tickers=tickers, years=range(2007, 2027), with_daily=True)
    log(
        f"Feature context loaded: {len(context.panel):,} panel rows, "
        f"{len(context.daily):,} daily rows")

    summary = []
    yearly = []
    sensitivities = []
    selections = {}
    evaluations = {}
    ledger = lib.ledger_read()
    already = set(ledger.loc[ledger.stage == "ran", "spec_hash"])

    ordered = [
        (rule, offset) for rule in RULES for offset in OFFSETS
        if (rule, offset) != PRIMARY
    ] + [PRIMARY]
    for rule, offset in ordered:
        key = f"{rule}_t{offset}"
        is_primary = (rule, offset) == PRIMARY
        run_dir = HERE if is_primary else HERE / "cells" / key
        (run_dir / "results").mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(parents=True, exist_ok=True)
        cell = trades[(trades.rule == rule) & (trades.offset == offset)].copy()
        dataset_path = run_dir / "results" / "gate_dataset.parquet"
        dataset = feature_dataset(cell, dataset_path, context)
        strict, base, wf, state, scoreable = strict_selection(cell, dataset)
        selections[(rule, offset)] = strict

        gated = net_stats(strict)
        base_stats = net_stats(base)
        item = {
            "rule": rule,
            "offset": offset,
            "n_scoreable": base_stats["n"],
            "n_selected": gated["n"],
            "base_mid": base_stats["mid"],
            "gated_mid": gated["mid"],
            "gate_lift": (
                gated["mid"] - base_stats["mid"]
                if gated["mid"] is not None and base_stats["mid"] is not None
                else None),
            **{name: gated.get(name) for name in (
                "worst", "quarter", "mid", "three_quarters", "best", "median",
                "win_rate", "capital_weighted", "ci95", "years_positive",
                "years_evaluated", "breakeven_alpha")},
        }
        summary.append(item)
        strict_mid = strict[np.isclose(strict.fill_alpha, .5)]
        for year in sorted(strict_mid.event_date.dt.year.unique()):
            year_stats = net_stats(
                strict[strict.event_date.dt.year == year])
            yearly.append({
                "rule": rule, "offset": offset, "year": int(year),
                **year_stats,
            })
        sensitivity_slices = [
            ("no_repaired_quotes", set(strict_mid.loc[
                ~strict_mid.quote_repaired, "event_id"])),
            ("entry_spread_le_10pct", set(strict_mid.loc[
                strict_mid.entry_spread_pct <= .10, "event_id"])),
            ("entry_spread_le_25pct", set(strict_mid.loc[
                strict_mid.entry_spread_pct <= .25, "event_id"])),
            ("exclude_top_1pct_winners", set(strict_mid.loc[
                strict_mid.ret_net <= strict_mid.ret_net.quantile(.99),
                "event_id"])),
        ]
        for label, event_ids in sensitivity_slices:
            slice_stats = net_stats(strict[strict.event_id.isin(event_ids)])
            sensitivities.append({
                "rule": rule, "offset": offset, "slice": label,
                **slice_stats,
            })
        write_json(
            run_dir / "results" / "selection.json",
            selection_record(rule, offset, strict, base, wf, scoreable))
        log(
            f"{key}: selected {gated['n']:,}/{base_stats['n']:,}; "
            f"net mid {pct(gated['mid'])}; lift {pct(item['gate_lift'])}")

        cell_cfg = cell_spec(spec, rule, offset)
        evaluations[(rule, offset)] = (
            cell_cfg, cell, dataset, state, run_dir, is_primary)

        if not is_primary:
            eval_gate, _ = common.make_registered_gate(STRATEGY, dataset)
            result = evaluate(
                cell_cfg, cell, gate=eval_gate, run_dir=run_dir,
                mc_paths=200, fractions=(.02, .05), write_report=True,
                input_files=[SOURCE_RESULTS / "grid_trades.parquet", dataset_path],
                extra_sections=[{
                    "title": "Strict gated net result",
                    "columns": ["scoreable", "selected", "base mid", "gated mid", "lift"],
                    "align": ["---:", "---:", "---:", "---:", "---:"],
                    "rows": [[
                        f"{base_stats['n']:,}", f"{gated['n']:,}",
                        pct(base_stats["mid"]), pct(gated["mid"]),
                        pct(item["gate_lift"])]],
                }])
            cell_hash = lib.spec_hash(cell_cfg)
            if not args.no_ledger and cell_hash not in already:
                lib.record_evaluation(run_dir, cell_cfg, result.results)
                already.add(cell_hash)

    paired = compare_cells(selections)
    write_json(RESULTS / "summary.json", summary)
    pd.DataFrame(summary).to_csv(RESULTS / "summary.csv", index=False)
    write_json(RESULTS / "by_year.json", yearly)
    pd.DataFrame(yearly).to_csv(RESULTS / "by_year.csv", index=False)
    write_json(RESULTS / "sensitivities.json", sensitivities)
    pd.DataFrame(sensitivities).to_csv(
        RESULTS / "sensitivities.csv", index=False)
    write_json(RESULTS / "paired_differences.json", paired)
    pd.DataFrame(paired).to_csv(RESULTS / "paired_differences.csv", index=False)
    make_plot(summary)

    primary_cfg, primary_cell, primary_dataset, _, run_dir, _ = evaluations[PRIMARY]
    eval_gate, _ = common.make_registered_gate(STRATEGY, primary_dataset)
    log("Generating evaluator-backed primary report")
    result = evaluate(
        primary_cfg, primary_cell, gate=eval_gate, run_dir=run_dir,
        mc_paths=500, fractions=(.02, .05), write_report=True,
        input_files=[
            SOURCE_RESULTS / "grid_trades.parquet",
            RESULTS / "input_manifest.json",
            RESULTS / "summary.json",
            RESULTS / "paired_differences.json",
            RESULTS / "by_year.json",
            RESULTS / "sensitivities.json",
            RESULTS / "gate_dataset.parquet"],
        extra_sections=report_sections(
            summary, paired, yearly, sensitivities, entry))
    primary_hash = lib.spec_hash(primary_cfg)
    if not args.no_ledger and primary_hash not in already:
        lib.record_evaluation(HERE, primary_cfg, result.results)
    log(f"Finished: {result.report_path}")


STARTED = time.monotonic()
if __name__ == "__main__":
    main()
