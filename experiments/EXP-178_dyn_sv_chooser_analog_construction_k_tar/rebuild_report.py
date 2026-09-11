#!/usr/bin/env python3
"""Rebuild EXP-178's reports with per-arm detail, a chooser accuracy table,
and the diagnostics needed to reason about why better ranking didn't produce
a better book.

Reuses the cached dev_scores/choices for both the champion (EXP-169's own
results/) and the candidate (this experiment's results/retrain/) — no
retraining, no dataset rebuild beyond reloading `mid`/`raw`, which is a ~10s
pandas read. Everything here is pure aggregation over what Stage 2 already
computed and saved.

Fixes a real asymmetry in the first report: EXP-169's own PRIMARY arm is
menu7p_mcap1, so its saved structure_diagnostics.json is the mcap1 book, while
this experiment's PRIMARY_ARM is menu7p_mcap10 (matching what EXP-170 actually
promoted). The first report's "champion reference" table was comparing the
candidate's mcap10 numbers against the champion's mcap1 numbers without saying
so. This rebuilds BOTH arms for BOTH models from the same cached choices, so
every comparison in the new report is arm-matched.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RETRAIN_RESULTS = RESULTS / "retrain"
E169_DIR = ROOT / "experiments/EXP-169_menu7prime_confirmation"
E169_RUN = E169_DIR / "run.py"

STARTED = time.monotonic()
sys.path.insert(0, str(ROOT))
from engine.evaluate import evaluate  # noqa: E402
from experiments import common, lib  # noqa: E402


def log(msg: str) -> None:
    print(f"[EXP-178-report {time.monotonic() - STARTED:,.0f}s] {msg}", flush=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, default=str))


def oracle_hit_rate(choices: pd.DataFrame, year: int | None = None) -> float:
    g = choices if year is None else choices[pd.to_datetime(choices["event_date"]).dt.year == year]
    g = g.dropna(subset=["oracle_structure", "head_structure"])
    return float((g["oracle_structure"] == g["head_structure"]).mean()) if len(g) else float("nan")


def build_accuracy_table(champ_choices, cand_choices, champ_diag, cand_diag) -> list[dict]:
    years = sorted(set(pd.to_datetime(champ_choices["event_date"]).dt.year) |
                   set(pd.to_datetime(cand_choices["event_date"]).dt.year))
    champ_by_year = {int(f["year"]): f for f in champ_diag}
    cand_by_year = {int(f["year"]): f for f in cand_diag}
    rows = []
    for year in years:
        cf, kf = champ_by_year.get(year, {}), cand_by_year.get(year, {})
        rows.append({
            "year": year,
            "champion_within_event_rho": cf.get("within_event_spearman"),
            "candidate_within_event_rho": kf.get("within_event_spearman"),
            "champion_oracle_hit": oracle_hit_rate(champ_choices, year),
            "candidate_oracle_hit": oracle_hit_rate(cand_choices, year),
        })
    rows.append({
        "year": "all",
        "champion_within_event_rho": float(np.nanmean([r["champion_within_event_rho"] for r in rows])),
        "candidate_within_event_rho": float(np.nanmean([r["candidate_within_event_rho"] for r in rows])),
        "champion_oracle_hit": oracle_hit_rate(champ_choices),
        "candidate_oracle_hit": oracle_hit_rate(cand_choices),
    })
    return rows


def per_structure_wanted_vs_funded(book: pd.DataFrame) -> dict:
    """`book` is the full picked-and-hygiene-passed frame (funded + unfunded)
    from build_book/account_score — every row a structure the chooser picked
    that cleared rel_spread/mcap hygiene, whether or not capital funded it."""
    out = {}
    for structure, g in book.groupby("strategy"):
        placed = g[g["funded"]]
        out[structure] = {
            "wanted": int(len(g)),
            "funded": int(len(placed)),
            "wanted_mean_realized_pnl": float(g["pnl"].mean()) if len(g) else float("nan"),
            "funded_mean_realized_pnl": float(placed["pnl"].mean()) if len(placed) else float("nan"),
            "wanted_mean_expected_per_secured": float(g["expected_per_secured"].mean()) if len(g) else float("nan"),
            "funded_mean_expected_per_secured": float(placed["expected_per_secured"].mean()) if len(placed) else float("nan"),
        }
    return out


def score_vs_funding_priority_agreement(choices: pd.DataFrame, mid: pd.DataFrame, e169) -> dict:
    """Within each event, does the chooser's own head-pick also carry the
    highest expected-PnL-per-secured-dollar among the event's OFFERED
    candidates? A chooser that ranks realized PnL well but disagrees with the
    funding queue's own ordering metric gets outranked for capital by a
    candidate the chooser liked less — funding is ordered on
    exp_pnl_sim/secured, never on the chooser's own score."""
    m = mid[["event_id", "strategy", "exp_pnl_sim", "legs"]].copy()
    m["secured"] = m["legs"].map(e169.margin163.secured_per_contract)
    m["expected_per_secured"] = m["exp_pnl_sim"] * 100.0 / m["secured"]
    picks = choices.dropna(subset=["head_structure"])[["event_id", "head_structure"]]
    top_by_priority = (
        m.sort_values(["event_id", "expected_per_secured"], ascending=[True, False])
        .drop_duplicates("event_id")[["event_id", "strategy"]]
        .rename(columns={"strategy": "priority_top_structure"})
    )
    joined = picks.merge(top_by_priority, on="event_id", how="left")
    agree = float((joined["head_structure"] == joined["priority_top_structure"]).mean())
    return {"n_events": int(len(joined)), "chooser_agrees_with_funding_priority_order": agree}


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    e169 = load_module("exp178_final_e169", E169_RUN)
    mid, raw = e169.load_data_menu(e169.MENU)
    log(f"loaded {len(mid):,} candidates")

    champ_dev = pd.read_parquet(E169_DIR / "results/dev_scores.parquet")
    champ_choices = pd.read_parquet(E169_DIR / "results/choices.parquet")
    champ_diag = json.loads((E169_DIR / "results/fold_diagnostics.json").read_text())
    cand_dev = pd.read_parquet(RETRAIN_RESULTS / "dev_scores.parquet")
    cand_choices = pd.read_parquet(RETRAIN_RESULTS / "choices.parquet")
    cand_diag = json.loads((RETRAIN_RESULTS / "fold_diagnostics.json").read_text())
    log(f"champion picks: {champ_choices['head_structure'].notna().sum():,} of {len(champ_choices):,}; "
        f"candidate picks: {cand_choices['head_structure'].notna().sum():,} of {len(cand_choices):,}")

    accuracy_table = build_accuracy_table(champ_choices, cand_choices, champ_diag, cand_diag)
    log("accuracy table built")

    per_arm = {}
    for arm in e169.ARMS:
        per_arm[arm] = {}
        for model_name, choices in (("champion", champ_choices), ("candidate", cand_choices)):
            book = e169.build_book(mid, choices, arm)
            funded, account = e169.account_score(book)
            structure_diag = e169.structure_diagnostics(
                champ_dev if model_name == "champion" else cand_dev, choices,
                set(funded.loc[funded["funded"], "event_id"]),
            )
            buckets = e169.mcap_buckets(funded)
            wf = per_structure_wanted_vs_funded(funded)
            per_arm[arm][model_name] = {
                "account": account, "structure_diagnostics": structure_diag,
                "mcap_buckets": buckets, "wanted_vs_funded": wf,
            }
            log(f"{arm}/{model_name}: wanted={account.get('wanted', 0):,} "
                f"funded={account.get('funded', 0):,} final=${account.get('final_equity', float('nan')):,.0f}")

    priority = {
        "champion": score_vs_funding_priority_agreement(champ_choices, mid, e169),
        "candidate": score_vs_funding_priority_agreement(cand_choices, mid, e169),
    }
    log(f"funding-priority agreement: champion={priority['champion']['chooser_agrees_with_funding_priority_order']:.1%} "
        f"candidate={priority['candidate']['chooser_agrees_with_funding_priority_order']:.1%}")

    write_json(RESULTS / "full_comparison.json", {
        "accuracy_table": accuracy_table, "per_arm": per_arm,
        "funding_priority_agreement": priority,
    })

    # -- rebuild the reports, one per arm, through the normal evaluate() path
    spy = common.load_spy_daily()
    PRIMARY_ARM = "menu7p_mcap10"

    def fmt_pct(x): return f"{x:+.1%}" if np.isfinite(x) else "n/a"
    def fmt_pnl(x): return f"{x:+.3f}" if np.isfinite(x) else "n/a"

    def accuracy_section() -> dict:
        rows = [[str(r["year"]), fmt_pct(r["champion_oracle_hit"]), fmt_pct(r["candidate_oracle_hit"]),
                fmt_pnl(r["champion_within_event_rho"]), fmt_pnl(r["candidate_within_event_rho"])]
               for r in accuracy_table]
        return {
            "title": "Chooser accuracy — arm-independent (the pick is made once; funding differs by arm)",
            "note": "Oracle hit = share of events where the head pick equals the offered-menu argmax-PnL structure. "
                   "Within-event ρ = Spearman(chooser score, realized PnL) pooled within each event, averaged.",
            "columns": ["year", "champion oracle hit", "candidate oracle hit",
                       "champion within-event ρ", "candidate within-event ρ"],
            "align": ["---"] + ["---:"] * 4, "rows": rows,
        }

    def why_section(arm: str) -> dict:
        champ, cand = per_arm[arm]["champion"], per_arm[arm]["candidate"]
        ca, ka = champ["account"], cand["account"]
        rows = [[
            "final equity", f"${ca.get('final_equity', float('nan')):,.0f}", f"${ka.get('final_equity', float('nan')):,.0f}",
        ], [
            "funded / wanted", f"{ca.get('funded', 0):,}/{ca.get('wanted', 0):,}", f"{ka.get('funded', 0):,}/{ka.get('wanted', 0):,}",
        ], [
            "funded mean realized", fmt_pnl(ca.get('funded_mean_realized_pnl', float('nan'))), fmt_pnl(ka.get('funded_mean_realized_pnl', float('nan'))),
        ], [
            "unfunded mean realized", fmt_pnl(ca.get('unfunded_mean_realized_pnl', float('nan'))), fmt_pnl(ka.get('unfunded_mean_realized_pnl', float('nan'))),
        ], [
            "chooser agrees with funding-priority order",
            fmt_pct(priority["champion"]["chooser_agrees_with_funding_priority_order"]),
            fmt_pct(priority["candidate"]["chooser_agrees_with_funding_priority_order"]),
        ]]
        struct_rows = []
        contrib_rows = []
        total_champ_contrib = total_cand_contrib = 0.0
        for s in sorted(set(champ["wanted_vs_funded"]) | set(cand["wanted_vs_funded"])):
            cw, kw = champ["wanted_vs_funded"].get(s, {}), cand["wanted_vs_funded"].get(s, {})
            struct_rows.append([
                s, f"{cw.get('wanted', 0):,}/{cw.get('funded', 0):,}", f"{kw.get('wanted', 0):,}/{kw.get('funded', 0):,}",
                fmt_pnl(cw.get("wanted_mean_realized_pnl", float("nan"))), fmt_pnl(kw.get("wanted_mean_realized_pnl", float("nan"))),
                fmt_pnl(cw.get("funded_mean_realized_pnl", float("nan"))), fmt_pnl(kw.get("funded_mean_realized_pnl", float("nan"))),
            ])
            champ_contrib = cw.get("funded", 0) * cw.get("funded_mean_realized_pnl", 0.0)
            cand_contrib = kw.get("funded", 0) * kw.get("funded_mean_realized_pnl", 0.0)
            total_champ_contrib += champ_contrib
            total_cand_contrib += cand_contrib
            contrib_rows.append([
                s, f"{champ_contrib:+.1f}", f"{cand_contrib:+.1f}", f"{cand_contrib - champ_contrib:+.1f}",
            ])
        contrib_rows.sort(key=lambda r: float(r[3].replace("+", "")))
        contrib_rows.append(["**total**", f"{total_champ_contrib:+.1f}", f"{total_cand_contrib:+.1f}",
                             f"{total_cand_contrib - total_champ_contrib:+.1f}"])
        return {
            "title": f"Why ranking improved but the {arm} book didn't — the funding layer",
            "body": [
                "Funding is ordered by exp_pnl_sim per secured dollar of the CHOSEN structure, never by the "
                "chooser's own score. A chooser can rank realized PnL better within an event while still "
                "picking structures that carry a worse funding-priority number, and capital goes first to "
                "the highest-priority picks regardless of which model chose them. The 'agreement' row above "
                "is how often the chooser's own pick is also the event's top funding-priority candidate; a "
                "lower number for the candidate means more of its picks queue behind other events' picks for "
                "the same capital.",
            ],
            "columns": ["metric", "champion", "candidate"], "align": ["---", "---:", "---:"], "rows": rows,
            "table2_title": f"Per structure — wanted/funded and mean realized PnL, {arm}",
            "table2_columns": ["structure", "champion wanted/funded", "candidate wanted/funded",
                               "champion wanted mean", "candidate wanted mean",
                               "champion funded mean", "candidate funded mean"],
            "table2_align": ["---"] + ["---:"] * 6, "table2_rows": struct_rows,
            "table3_title": f"Aggregate funded-PnL contribution by structure, {arm} "
                            "(funded count × mean realized — a rough, capital-unweighted proxy for each "
                            "structure's share of the final-equity gap; sorted worst delta first)",
            "table3_columns": ["structure", "champion contribution", "candidate contribution", "delta"],
            "table3_align": ["---", "---:", "---:", "---:"], "table3_rows": contrib_rows,
        }

    def account_section(arm: str) -> dict:
        champ, cand = per_arm[arm]["champion"]["account"], per_arm[arm]["candidate"]["account"]
        return {
            "title": f"Account — {arm}",
            "columns": ["metric", "champion", "candidate"], "align": ["---", "---:", "---:"],
            "rows": [
                ["wanted", f"{champ.get('wanted', 0):,}", f"{cand.get('wanted', 0):,}"],
                ["funded", f"{champ.get('funded', 0):,}", f"{cand.get('funded', 0):,}"],
                ["final equity", f"${champ.get('final_equity', float('nan')):,.0f}", f"${cand.get('final_equity', float('nan')):,.0f}"],
                ["CAGR", f"{champ.get('cagr', float('nan')):.2%}", f"{cand.get('cagr', float('nan')):.2%}"],
                ["peak concurrency", f"{champ.get('peak_concurrency', 0):,}", f"{cand.get('peak_concurrency', 0):,}"],
                ["defined-risk failures", f"{champ.get('defined_risk_failures', 0):,}", f"{cand.get('defined_risk_failures', 0):,}"],
            ],
        }

    def structure_section(arm: str) -> dict:
        champ_sd, cand_sd = per_arm[arm]["champion"]["structure_diagnostics"], per_arm[arm]["candidate"]["structure_diagnostics"]
        rows = []
        for s in sorted(set(champ_sd) | set(cand_sd)):
            cd, kd = champ_sd.get(s, {}), cand_sd.get(s, {})
            rows.append([
                s, f"{cd.get('funded_picks', 0):,}", f"{kd.get('funded_picks', 0):,}",
                fmt_pnl(cd.get("mean_realized_when_picked_funded", float("nan"))),
                fmt_pnl(kd.get("mean_realized_when_picked_funded", float("nan"))),
                fmt_pct(cd.get("pick_precision_vs_oracle", float("nan"))),
                fmt_pct(kd.get("pick_precision_vs_oracle", float("nan"))),
            ])
        return {
            "title": f"Structure diagnostics — {arm} (this arm's own funded set)",
            "columns": ["structure", "champion funded", "candidate funded",
                       "champion mean realized", "candidate mean realized",
                       "champion precision", "candidate precision"],
            "align": ["---"] + ["---:"] * 6, "rows": rows,
        }

    def render_extra(arm: str):
        def _extra(_result):
            sections = [account_section(arm), structure_section(arm), accuracy_section(), why_section(arm)]
            # flatten the why_section's second table into its own section entry,
            # since the report renderer takes one table per section
            expanded = []
            for s in sections:
                extra_tables = []
                for n in (2, 3):
                    key = f"table{n}_rows"
                    if key in s:
                        extra_tables.append({
                            "title": s.pop(f"table{n}_title"), "columns": s.pop(f"table{n}_columns"),
                            "align": s.pop(f"table{n}_align"), "rows": s.pop(key),
                        })
                expanded.append(s)
                expanded.extend(extra_tables)
            return expanded
        return _extra

    for arm in e169.ARMS:
        book = e169.build_book(mid, cand_choices, arm)
        funded, _ = e169.account_score(book)
        funded_book = e169.base.selected_all_alphas(raw, funded)
        run_dir = HERE if arm == PRIMARY_ARM else HERE / "arms" / arm
        cell = dict(spec)
        if arm != PRIMARY_ARM:
            cell["primary_spec"] = dict(cell["primary_spec"], evaluated_arm=arm)
            cell["grid_cell"] = True
            cell["promotion_target"] = None
        result = evaluate(
            cell, funded_book, gate=None, run_dir=run_dir, spy_daily=spy,
            tail_shock=common.abs_move_tail_shock,
            input_files=[e169.base.CANDIDATES, e169.TIER4,
                        RETRAIN_RESULTS / "dev_scores.parquet", RETRAIN_RESULTS / "choices.parquet"],
            extra_sections=render_extra(arm), write_report=True,
        )
        log(f"{arm}: report rebuilt, mean={result.results['headline'].get('mean', float('nan')):+.3f}")

    log("done — no ledger row written (numbers unchanged from the recorded run)")


if __name__ == "__main__":
    main()
