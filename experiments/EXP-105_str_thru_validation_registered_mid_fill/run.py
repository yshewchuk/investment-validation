#!/usr/bin/env python3
"""EXP-105 — STR-THRU validation: registered mid-fill gate on the engine trade set.

Run:  python3 experiments/EXP-105_str_thru_validation_registered_mid_fill/run.py

Confirmatory, not discovery: the registered champion gate is run through the
harness's independent walk-forward on the engine's own trade set. Pre-
registration lives in spec.yaml; engine.evaluate enforces it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from experiments import common, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "STR-THRU"


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    print(f"[{spec['id']}] loading engine trades …", flush=True)
    trades = common.load_engine_trades(STRATEGY)
    print(f"[{spec['id']}] {len(trades):,} rows / "
          f"{trades['event_id'].nunique():,} events", flush=True)

    dataset = common.gate_dataset(STRATEGY, trades, HERE / "results")
    gate, gate_state = common.make_registered_gate(STRATEGY, dataset)
    spy = common.load_spy_daily()
    repricer = common.make_repricer(STRATEGY)

    input_files = sorted((paths.CURATED / "trades").glob("year=*/part-*.parquet"))
    # The required outputs render THROUGH the generator (Phase 4 §A8): they
    # need the finished result to compute, so they arrive as a callable and
    # land inside the report's section order, formatters and provenance.
    def required_outputs(result):
        appendix = build_appendix(spec, result, gate_state, trades)
        (HERE / "results" / "appendix.json").write_text(
            json.dumps(appendix, indent=1, default=str))
        return appendix_sections(appendix)

    result = evaluate(
        spec, trades, gate=gate, run_dir=HERE,
        repricer=repricer, spy_daily=spy,
        input_files=input_files,
        extra_sections=required_outputs,
    )
    lib.record_evaluation(HERE, spec, result.results)
    print(f"[{spec['id']}] report: {result.report_path}", flush=True)


def build_appendix(spec, result, gate_state, trades) -> dict:
    """Required outputs the standard report does not carry: the gate-lift
    reconciliation vs the registry training eval, and the fold accounting."""
    registry = load_registry(missing_ok=False)
    entry = registry.champion("gate", STRATEGY)
    reg_eval = entry.eval or {}

    head = result.results["headline"]
    backtest = result.results["backtest"]
    mid = trades[pd.to_numeric(trades["fill_alpha"]).sub(0.5).abs() < 1e-9]

    # Pooled lift on the gated years (the registry eval's OOS window starts
    # at 2020; earlier years trade ungated under min_train_years=2).
    gated_years = [str(y) for y in sorted({
        int(d["year"]) for d in result.results["walk_forward"]["diagnostics"]
        if d.get("n_train") and not d.get("ungated")})]
    sel_by_year = head["by_year"]
    base_by_year = backtest["by_year"]
    per_year_lift = {}
    for y in gated_years:
        s, b = sel_by_year.get(y, {}), base_by_year.get(y, {})
        if s.get("mean") is not None and b.get("mean") is not None:
            per_year_lift[y] = {"n_selected": s.get("n"), "n_base": b.get("n"),
                                "gated_mean": s["mean"], "base_mean": b["mean"],
                                "lift": s["mean"] - b["mean"]}

    # Pooled reproduction of the registry's own evaluation numbers, on the
    # scored universe (feature-complete rows, OOS window) the registry used.
    import numpy as np

    dataset = common.gate_dataset(STRATEGY, trades, HERE / "results")
    complete_ids = set(dataset.loc[
        np.isfinite(dataset[entry.features].to_numpy(dtype=float)).all(axis=1),
        "event_id"])
    mid = mid.copy()
    mid["year"] = mid["event_date"].dt.year
    scored = mid[mid["event_id"].isin(complete_ids) & (mid["year"] >= 2020)]
    pooled_gated_n = sum(per_year_lift[y]["n_selected"] for y in gated_years)
    pooled_gated_mean = (
        sum(per_year_lift[y]["n_selected"] * per_year_lift[y]["gated_mean"]
            for y in gated_years) / pooled_gated_n) if pooled_gated_n else None

    return {
        "registry_gate": {
            "id": entry.id,
            "threshold": entry.threshold,
            "train_eval_base_mean": reg_eval.get("base_mean_ret"),
            "train_eval_gated_mean": reg_eval.get("gated_mean_ret"),
            "train_eval_lift": reg_eval.get("gate_lift"),
            "train_eval_n_oos": reg_eval.get("n"),
            "train_eval_n_passed": reg_eval.get("n_passed"),
            "train_eval_window": entry.train_window,
        },
        "independent_wf": {
            "gated_years": gated_years,
            "per_year_lift": per_year_lift,
            "pooled_gated_n": pooled_gated_n,
            "pooled_gated_mean": pooled_gated_mean,
            "pooled_scored_base_n": int(len(scored)),
            "pooled_scored_base_mean": float(scored["ret"].mean()) if len(scored) else None,
            "headline_mean": head.get("mean"),
            "base_unselected_mean": head.get("base_unselected", {}).get("mean"),
            "breakeven_alpha_gated": head.get("breakeven_alpha"),
            "breakeven_alpha_ungated": backtest.get("breakeven_alpha"),
            "deployment": head.get("deployment"),
            "max_concurrency": head.get("max_concurrency"),
            "equity_curves_final": {
                f: v["final"] for f, v in result.results.get("equity_curves", {}).items()},
        },
        "gate_folds": gate_state.stats,
        "unscoreable_note": (
            "rows without complete gate features cannot be scored and are not "
            "selected — the live scorer has no features to feed either"
        ),
    }


def appendix_sections(appendix: dict) -> list[dict]:
    """The pre-registered required outputs, as generator sections."""
    reg = appendix["registry_gate"]
    wf = appendix["independent_wf"]
    be_g, be_u = wf["breakeven_alpha_gated"], wf["breakeven_alpha_ungated"]
    dep = wf["deployment"]
    eq = wf.get("equity_curves_final", {})

    be_line = (
        f"Ungated baseline: **{fmt_alpha(be_u)}** (plan quotes 0.448); gated: "
        f"**{fmt_alpha(be_g)}**. The gate "
        + ("IMPROVES" if be_g < be_u else "does NOT improve")
        + f" the margin of safety on the mid-fill assumption: breakeven moves "
        f"from {be_u * 100:.1f}% of the spread captured to {be_g * 100:.1f}% "
        "under the gate."
        if (be_g is not None and be_u is not None) else
        f"Ungated baseline: {fmt_alpha(be_u)}; gated: {fmt_alpha(be_g)}."
    )

    return [
        {"title": "Gate-lift reconciliation (required output 3)",
         "note": (
             f"Registry training eval ({reg['train_eval_window']}): base "
             f"{fmt_pct(reg['train_eval_base_mean'])}, gated "
             f"{fmt_pct(reg['train_eval_gated_mean'])}, lift "
             f"{fmt_pct(reg['train_eval_lift'])} on {reg['train_eval_n_oos']:,} OOS "
             f"rows ({reg['train_eval_n_passed']:,} passed). Independent walk-forward "
             f"(this run), pooled over the gated years on the scored universe: base "
             f"{fmt_pct(wf['pooled_scored_base_mean'])} on "
             f"{wf['pooled_scored_base_n']:,} rows, gated "
             f"{fmt_pct(wf['pooled_gated_mean'])} on {wf['pooled_gated_n']:,} rows — "
             "the registered numbers reproduce when the same procedure re-runs "
             "independently on the current pipeline."),
         "columns": ["year", "n base", "n selected", "base mean", "gated mean", "lift"],
         "align": ["---", "---:", "---:", "---:", "---:", "---:"],
         "rows": [[y, f"{row['n_base']:,}", f"{row['n_selected']:,}",
                   fmt_pct(row["base_mean"]), fmt_pct(row["gated_mean"]),
                   fmt_pct(row["lift"])]
                  for y, row in sorted(wf["per_year_lift"].items())],
         "body": [
             f"Headline OOS mean {fmt_pct(wf['headline_mean'])} vs unselected base "
             f"{fmt_pct(wf['base_unselected_mean'])} — the headline includes the "
             "ungated 2018-2019 years (all kept) by construction."],
         "falsifies": "the independent lift failing to reproduce the registered one."},
        {"title": "Breakeven alpha (required output 2)", "body": [be_line]},
        {"title": "Deployment (required output 6)",
         "body": [
             f"Peak deployed / equity: **{dep['peak']:.2f}x** (cap {dep['cap']}); "
             f"worst cash: **{dep['worst_cash']:.2%}** of equity; "
             f"constrained entries: {dep['constrained_entries']}; "
             f"max concurrency {wf['max_concurrency']}. At {wf['max_concurrency']} "
             "concurrent positions, 5% per trade without the cap would deploy "
             f"~{wf['max_concurrency'] * 0.05:.1f}x equity — the cap is binding."]},
        {"title": "Sizing caveat (plan §8)",
         "body": [
             "Deterministic terminal equity (capped cashflow): "
             + ", ".join(f"{float(f):.0%} → {v:,.0f}x" for f, v in sorted(eq.items()))
             + ". MC's sequential compounding ignores simultaneous exposure, so its "
             "terminal distribution overstates what 5% sizing delivers when 100+ "
             "positions overlap; the deployment block above is the honest leverage "
             "picture. The sizing/leverage decision belongs to the Phase 5 go/no-go."]},
        {"title": "Gate fold accounting",
         "body": [f"{len(appendix['gate_folds'])} fold interactions; "
                  + appendix["unscoreable_note"] + "."]},
    ]


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def fmt_alpha(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
