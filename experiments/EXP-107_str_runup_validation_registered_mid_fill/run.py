#!/usr/bin/env python3
"""EXP-107 — STR-RUNUP validation: registered mid-fill gate + reconciliation.

Run:  python3 experiments/EXP-107_str_runup_validation_registered_mid_fill/run.py

Confirmatory, not discovery: the registered champion gate is run through the
harness's independent walk-forward on the engine's own trade set, and the
three disagreeing base numbers are reconciled from artifacts already on disk.
Pre-registration lives in spec.yaml; engine.evaluate enforces it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from engine import paths  # noqa: E402
from engine.evaluate import evaluate  # noqa: E402
from engine.models.registry import load_registry  # noqa: E402
from experiments import common, lib  # noqa: E402

HERE = Path(__file__).resolve().parent
STRATEGY = "STR-RUNUP"

LEGACY_S3 = (paths.EP_STRATEGIES / "s3_pre_earnings_long_vol" / "data" /
             "trades_real_t14.csv")
EXP048_RESULT = paths.EP_OPF / "results" / "exp048_midfill_rerun.json"


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

    def required_outputs(result):
        # Rendered THROUGH the generator (Phase 4 §A8): the reconciliation is
        # the point of this experiment, so it belongs inside the report's
        # section order, formatters and provenance — not appended after them.
        appendix = build_appendix(spec, result, gate_state, trades, dataset)
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


# --------------------------------------------------------------------------
# the reconciliation (pre-registered required output)
# --------------------------------------------------------------------------


def reconcile(trades: pd.DataFrame, dataset: pd.DataFrame) -> dict:
    """Explain the three disagreeing base numbers from artifacts on disk.

    1. plan +3.9% — EXP-048's mid re-price of the LEGACY S3 set (3,622
       trades, scoped universe, calendar T-14 entry).
    2. engine replay +1.02% — the unselected engine base, all years.
    3. registry gate eval -1.24% — the gate's own OOS window (2020-2026)
       restricted to rows with complete features.
    """
    from engine.data import store

    mid = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)].copy()
    mid["year"] = mid["event_date"].dt.year

    engine_all = _stats(mid)
    engine_2020 = _stats(mid[mid["year"] >= 2020])

    registry = load_registry(missing_ok=False)
    entry = registry.champion("gate", STRATEGY)
    complete_ids = set(dataset.loc[
        np.isfinite(dataset[entry.features].to_numpy(dtype=float)).all(axis=1),
        "event_id"])
    engine_gate_universe = mid[mid["event_id"].isin(complete_ids)]
    engine_gate_window = _stats(engine_gate_universe[engine_gate_universe["year"] >= 2020])

    # Legacy S3 set: worst fills are stored; the mid number is the EXP-048
    # artifact (a re-price at the recorded strikes, not re-derivable from the
    # stored cost/exit_val columns alone).
    legacy = pd.read_csv(LEGACY_S3, dtype={"ticker": str})
    legacy["date"] = pd.to_datetime(legacy["date"])
    exp048 = json.loads(EXP048_RESULT.read_text()).get("S3_long", {})

    overlap = legacy.merge(
        mid[["ticker", "event_date", "ret", "event_id"]].rename(
            columns={"event_date": "date", "ret": "engine_ret"}),
        on=["ticker", "date"], how="inner")

    # Coverage bias: priced share of the planned universe and the mcap split
    # of priced vs unpriced events.
    events = store.read_table(
        "earnings_events", years=range(2018, 2027),
        columns=["event_id", "ticker", "event_date", "session"])
    events = events[events["session"].notna()].copy()
    events["event_date"] = pd.to_datetime(events["event_date"])
    priced_ids = set(mid["event_id"])
    events["priced"] = events["event_id"].isin(priced_ids)

    sec = store.read_table("securities", years=range(2018, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    events["year"] = events["event_date"].dt.year
    merged = events.merge(sec, on=["ticker", "year"], how="left")
    priced_mcap = merged.loc[merged["priced"], "mcap_usd"].dropna()
    unpriced_mcap = merged.loc[~merged["priced"], "mcap_usd"].dropna()

    def mcap_block(s: pd.Series) -> dict:
        return {
            "n_with_mcap": int(len(s)),
            "median_usd": float(s.median()) if len(s) else None,
            "frac_above_10b": float((s > 10e9).mean()) if len(s) else None,
            "frac_1b_to_10b": float(((s > 1e9) & (s <= 10e9)).mean()) if len(s) else None,
            "frac_below_1b": float((s <= 1e9).mean()) if len(s) else None,
        }

    return {
        "engine_all_years": engine_all,
        "engine_2020_plus": engine_2020,
        "engine_gate_universe_2020_plus": engine_gate_window,
        "registry_gate_eval": {
            "base_mean_ret": (entry.eval or {}).get("base_mean_ret"),
            "n_oos": (entry.eval or {}).get("n"),
            "window": entry.train_window,
        },
        "legacy_s3": {
            "n": int(len(legacy)),
            "years": f"{int(legacy['year'].min())}-{int(legacy['year'].max())}",
            "tickers": int(legacy["ticker"].nunique()),
            "worst_fill_mean": float(legacy["ret"].mean()),
            "mid_fill_mean_exp048": exp048.get("mid_mean"),
            "mid_fill_n_exp048": exp048.get("n"),
            "mid_fill_years_pos_exp048": exp048.get("years_pos"),
            "universe_note": ("scoped (mcap >= ~1B, edge-scoped), calendar T-14 "
                              "entry; per EXP-048 caveats"),
        },
        "legacy_engine_overlap": {
            "n_legacy": int(len(legacy)),
            "n_overlap": int(len(overlap)),
            "engine_mid_mean_on_overlap": float(overlap["engine_ret"].mean()) if len(overlap) else None,
            "legacy_worst_mean_on_overlap": float(
                overlap["ret"].mean()) if len(overlap) else None,
        },
        "coverage_bias": {
            "planned_events": int(len(events)),
            "priced_events": int(events["priced"].sum()),
            "coverage": float(events["priced"].mean()),
            "priced_mcap": mcap_block(priced_mcap),
            "unpriced_mcap": mcap_block(unpriced_mcap),
        },
    }


def _stats(mid: pd.DataFrame) -> dict:
    r = mid["ret"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    return {
        "n": int(r.size),
        "mean": float(r.mean()) if r.size else None,
        "win_rate": float((r > 0).mean()) if r.size else None,
        "by_year": {
            str(int(y)): float(g.mean())
            for y, g in mid.groupby(mid["event_date"].dt.year)["ret"]
            if len(g)
        },
    }


def build_appendix(spec, result, gate_state, trades, dataset) -> dict:
    registry = load_registry(missing_ok=False)
    entry = registry.champion("gate", STRATEGY)
    reg_eval = entry.eval or {}

    head = result.results["headline"]
    backtest = result.results["backtest"]

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

    return {
        "reconciliation": reconcile(trades, dataset),
        "registry_gate": {
            "id": entry.id,
            "threshold": entry.threshold,
            "train_eval_base_mean": reg_eval.get("base_mean_ret"),
            "train_eval_gated_mean": reg_eval.get("gated_mean_ret"),
            "train_eval_lift": reg_eval.get("gate_lift"),
            "train_eval_n_oos": reg_eval.get("n"),
            "train_eval_window": entry.train_window,
        },
        "independent_wf": {
            "gated_years": gated_years,
            "per_year_lift": per_year_lift,
            "headline_mean": head.get("mean"),
            "base_unselected_mean": head.get("base_unselected", {}).get("mean"),
            "breakeven_alpha_gated": head.get("breakeven_alpha"),
            "breakeven_alpha_ungated": backtest.get("breakeven_alpha"),
            "deployment": head.get("deployment"),
            "max_concurrency": head.get("max_concurrency"),
        },
        "gate_folds": gate_state.stats,
    }


def appendix_sections(appendix: dict) -> list[dict]:
    """The pre-registered required outputs, as generator sections."""
    rec = appendix["reconciliation"]
    reg = appendix["registry_gate"]
    wf = appendix["independent_wf"]
    ea, e2, eg = rec["engine_all_years"], rec["engine_2020_plus"], rec["engine_gate_universe_2020_plus"]
    ls, ov, cb = rec["legacy_s3"], rec["legacy_engine_overlap"], rec["coverage_bias"]
    rg = rec["registry_gate_eval"]
    pm, um = cb["priced_mcap"], cb["unpriced_mcap"]
    be_g, be_u = wf["breakeven_alpha_gated"], wf["breakeven_alpha_ungated"]
    dep = wf["deployment"]

    reconciliation = {
        "title": "The three-number reconciliation (pre-registered required output)",
        "columns": ["source", "universe", "window", "base mean/trade"],
        "align": ["---", "---", "---", "---:"],
        "rows": [
            ["plan citing EXP-048 (legacy S3, mid re-price)",
             f"scoped, calendar T-14, n={ls['n']:,}", ls["years"],
             f"**{fmt_pct(ls['mid_fill_mean_exp048'])}**"],
            ["engine replay (this run, unselected)",
             f"full engine universe, n={ea['n']:,}", "2018-2026",
             f"**{fmt_pct(ea['mean'])}**"],
            ["registry gate eval", f"feature-complete subset, n={rg['n_oos']:,}",
             "2020-2026", f"**{fmt_pct(rg['base_mean_ret'])}**"],
        ],
        "body": [
            "Decomposition, each line computed from the artifacts on disk:",
            "",
            f"1. **Window.** Engine base restricted to 2020+ (the gate's OOS window): "
            f"{fmt_pct(e2['mean'])} on {e2['n']:,} trades vs {fmt_pct(ea['mean'])} on "
            "all years. 2018-2019 carry positive drift the 2020+ window does not see.",
            f"2. **Universe (feature coverage).** Engine base further restricted to the "
            f"gate's feature-complete rows: {fmt_pct(eg['mean'])} on {eg['n']:,} trades "
            f"— vs the registry's recorded {fmt_pct(rg['base_mean_ret'])}. "
            + ("The registry number reproduces on the current pipeline."
               if eg["mean"] is not None and rg["base_mean_ret"] is not None
               and abs(eg["mean"] - rg["base_mean_ret"]) < 0.005
               else "DIVERGES from the registry number — investigate."),
            f"3. **Legacy set.** The +3.9% comes from EXP-048's mid re-price of the "
            f"legacy S3 set ({ls['n']:,} trades, {ls['tickers']:,} tickers, "
            f"{ls['universe_note']}). The stored legacy returns are worst-fill "
            f"({fmt_pct(ls['worst_fill_mean'])}); the mid number is not re-derivable "
            "from the stored cost/exit_val columns — it exists only in the EXP-048 "
            "artifact quoted above.",
            f"4. **Overlap check.** {ov['n_overlap']:,} of the {ov['n_legacy']:,} legacy "
            f"events match an engine event on (ticker, date). Engine mid mean on the "
            f"overlap: {fmt_pct(ov['engine_mid_mean_on_overlap'])} — the engine spec "
            "(trading-day T-14, engine strike/expiry selection) prices the same prints "
            "differently than the legacy calendar-T-14 spec.",
        ],
        "promote_to_verdict": True,
        "verdict_row": ("Does the plan's +3.9% reproduce?",
                        f"Engine replay gives {fmt_pct(ea['mean'])} on {ea['n']:,} "
                        f"trades; the legacy figure is a different universe and spec",
                        "§8.5.1"),
    }

    coverage = {
        "title": "Coverage bias (required output)",
        "note": (f"Priced {cb['priced_events']:,} of {cb['planned_events']:,} planned "
                 f"events ({cb['coverage']:.1%}). Mcap of priced vs unpriced events:"),
        "columns": ["", "n w/ mcap", "median mcap", ">10B", "1-10B", "<1B"],
        "align": ["---", "---:", "---:", "---:", "---:", "---:"],
        "rows": [
            ["priced", f"{pm['n_with_mcap']:,}", f"${pm['median_usd']/1e9:.1f}B",
             f"{pm['frac_above_10b']:.1%}", f"{pm['frac_1b_to_10b']:.1%}",
             f"{pm['frac_below_1b']:.1%}"],
            ["unpriced", f"{um['n_with_mcap']:,}", f"${um['median_usd']/1e9:.1f}B",
             f"{um['frac_above_10b']:.1%}", f"{um['frac_1b_to_10b']:.1%}",
             f"{um['frac_below_1b']:.1%}"],
        ],
        "body": ["T-14 chains exist mainly where they were pulled for liquid names; the "
                 "strategy is validated only on that slice until the Sep pull enlarges it."],
        "falsifies": "the priced slice being representative of the planned universe.",
    }

    gate_lift = {
        "title": "Gate-lift reconciliation",
        "note": (f"Registry training eval ({reg['train_eval_window']}): base "
                 f"{fmt_pct(reg['train_eval_base_mean'])}, gated "
                 f"{fmt_pct(reg['train_eval_gated_mean'])}, lift "
                 f"{fmt_pct(reg['train_eval_lift'])} on {reg['train_eval_n_oos']:,} OOS "
                 "rows. Independent walk-forward (this run), per gated year:"),
        "columns": ["year", "n base", "n selected", "base mean", "gated mean", "lift"],
        "align": ["---", "---:", "---:", "---:", "---:", "---:"],
        "rows": [[y, f"{row['n_base']:,}", f"{row['n_selected']:,}",
                  fmt_pct(row["base_mean"]), fmt_pct(row["gated_mean"]),
                  fmt_pct(row["lift"])]
                 for y, row in sorted(wf["per_year_lift"].items())],
        "body": [f"Headline OOS mean {fmt_pct(wf['headline_mean'])} vs unselected base "
                 f"{fmt_pct(wf['base_unselected_mean'])}."],
    }

    return [
        reconciliation,
        coverage,
        gate_lift,
        {"title": "Breakeven alpha",
         "body": [f"Ungated baseline: **{fmt_alpha(be_u)}** (plan quotes 0.478 — a 2.2 pp "
                  f"margin under mid); gated: **{fmt_alpha(be_g)}**. If the gate does not "
                  "widen this, STR-RUNUP is not a capital candidate regardless of its "
                  "mean."]},
        {"title": "Deployment",
         "body": [f"Peak deployed / equity: **{dep['peak']:.2f}x** (cap {dep['cap']}); "
                  f"worst cash: **{dep['worst_cash']:.2%}** of equity; "
                  f"constrained entries: {dep['constrained_entries']}; "
                  f"max concurrency {wf['max_concurrency']}."]},
        {"title": "Gate fold accounting",
         "body": [f"{len(appendix['gate_folds'])} fold interactions; rows without "
                  "complete features are unscoreable and not selected."]},
    ]


def fmt_pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:+.2f}%"


def fmt_alpha(x) -> str:
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    main()
