#!/usr/bin/env python3
"""EXP-121 — CND-P put condor: defined-risk short-vol on the print.

Run:  python3 experiments/EXP-121_cnd_p_put_condor_defined_risk_short_vol/run.py

Descriptive measurement, no gate, no promotion target. The primary width is not
written into spec.yaml — the spec registers the RULE and stage 0 applies it, so
the spec is never edited after registration and its hash still matches the
PLANNED ledger row. Every other candidate width runs as a labelled grid cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from engine.evaluate import evaluate  # noqa: E402
from engine.structures import put_condor  # noqa: E402
from experiments import common, lib  # noqa: E402

import condor  # noqa: E402

HERE = Path(__file__).resolve().parent
STAGE0 = HERE / "results" / "stage0_geometry.json"


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    if not STAGE0.exists():
        raise SystemExit(
            "stage0_geometry.json is missing — run stage0_geometry.py first. "
            "The primary width is its output, and the spec registered the rule "
            "rather than the number.")
    stage0 = json.loads(STAGE0.read_text())
    rule = stage0["width_rule"]
    width = float(rule["chosen_width"])
    widths = [float(w) for w in spec["grid"]["width"]]
    print(f"[{spec['id']}] stage 0 chose width {width} "
          f"(by fallback: {rule['chosen_by_fallback']})", flush=True)

    priced = condor.build_all_widths(widths)
    spy = common.load_spy_daily()

    for w in [width] + [x for x in widths if x != width]:
        trades = priced[w]
        is_primary = w == width
        cell = dict(spec)
        cell["primary_spec"] = dict(spec["primary_spec"])
        cell["primary_spec"]["width"] = w
        if not is_primary:
            # Grid cells legitimately differ from the registered primary spec;
            # the label exempts them from the spec-hash check and marks them
            # as secondary everywhere they are reported.
            cell["grid_cell"] = True
        else:
            # The primary runs on the spec EXACTLY as registered — the resolved
            # width travels in the report, not in the hashed spec.
            cell = dict(spec)

        print(f"[{spec['id']}] width {w}: {len(trades):,} rows / "
              f"{trades['event_id'].nunique():,} events", flush=True)
        if trades.empty:
            print(f"[{spec['id']}] width {w}: nothing priced, skipped", flush=True)
            continue

        mid = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)].copy()
        geometry = {
            str(k): {
                "resolvability": v["resolvability"],
                "spacing_pct_median": v["spacing_pct_median"],
                "fail_causes": v["fail_causes"],
            }
            for k, v in stage0["widths"].items()
        }
        mechanics = condor.risk_mechanics(mid, geometry=geometry if is_primary else None)
        (HERE / "results" / f"risk_mechanics_w{int(round(w * 1000)):04d}.json").write_text(
            json.dumps(mechanics, indent=1, default=str))

        skips_path = HERE / "results" / f"skips_w{int(round(w * 1000)):04d}.json"
        skips = json.loads(skips_path.read_text()) if skips_path.exists() else {}

        def sections(result, m=mechanics, ww=w, primary=is_primary, s0=stage0,
                     sk=skips):
            return appendix_sections(result, m, ww, primary, s0, sk)

        # The ±1-day slippage and stale-date stresses reload every shifted
        # chain in the set, which costs more than the rest of the run put
        # together. They are a required output for the HEADLINE, so the primary
        # gets them and the grid cells report them as unavailable — which the
        # report says, rather than quietly leaving the stage blank.
        # write_report=is_primary — every width shares run_dir=HERE, so a
        # second call at the default (True) OVERWRITES REPORT.md: the four
        # widths ran primary-first, and the width=0.1 grid cell's report was
        # what actually ended up on disk when the run finished, silently
        # discarding the one this whole experiment is registered to measure.
        # metrics_<hash>.json and transactions_<hash>.csv are unconditional
        # (keyed by spec hash, not by filename) and survive for every width
        # regardless; only the rendered REPORT.md needs this guard.
        result = evaluate(
            cell, trades, gate=None, run_dir=HERE,
            repricer=(common.make_repricer(condor.STRATEGY,
                                           structure=put_condor(width=w))
                      if is_primary else None),
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[condor.trades_path(w)],
            extra_sections=sections,
            write_report=is_primary,
        )
        lib.record_evaluation(HERE, cell, result.results)
        print(f"[{spec['id']}] width {w}: report "
              f"{result.report_path or '(grid cell — not written; see metrics json)'}",
              flush=True)


# --------------------------------------------------------------------------
# appendix — the pre-registered required outputs
# --------------------------------------------------------------------------


def appendix_sections(result, m: dict, width: float, primary: bool,
                      stage0: dict, skips: dict) -> list[dict]:
    d = m["defined_risk"]
    a = m["assignment_exposure"]
    g = m["realized_geometry"]
    o = m["oracle_ceiling"]
    head = result.results["headline"]
    tail = result.results["stress"].get("tail_injection", {})
    label = "PRIMARY" if primary else "SECONDARY (grid cell)"
    sections: list[dict] = []

    # -- defined risk -------------------------------------------------------
    verdict = ("**Yes**" if d["n"] == 0 else
               f"**{d['share']:.1%} of trades lost more than the debit**")
    sections.append({
        "title": f"Defined-risk falsification ({label}, width {width})",
        "note": (
            f"A long condor's terminal payoff is bounded below by zero — equal "
            f"spacing makes the settlement below the bottom strike exactly "
            f"`(K4-K3) - (K2-K1) = 0` — so a return below -100% of the debit is a "
            f"statement about the EXIT quotes, not about the structure. "
            f"Observed: **{d['n']:,} of {a['n']:,} ({d['share']:.2%})** at mid "
            f"fills, worst **{d['worst_ret'] * 100:.1f}%** "
            f"({d['worst_trade'].get('ticker')} "
            f"{d['worst_trade'].get('event_date')}, realized move "
            f"{d['worst_trade'].get('realized_move', 0) * 100:+.1f}%)."),
        "columns": ["classification", "count"],
        "align": ["---", "---:"],
        "rows": [[k, f"{v:,}"] for k, v in sorted(d["classification"].items(),
                                                  key=lambda kv: -kv[1])] or [["none", "0"]],
        "body": ["", "Per mcap bucket:", "",
                 "| bucket | n | exceeded | worst ret | p01 ret |",
                 "|---|---:|---:|---:|---:|"]
                + [f"| {b} | {r['n']:,} | {r['n_exceeded']:,} | "
                   f"{r['worst_ret'] * 100:.1f}% | {r['p01_ret'] * 100:.1f}% |"
                   for b, r in d["by_mcap_bucket"].items()]
                + ["", "Per-year exceedance counts: "
                   + (", ".join(f"{y}: {n}" for y, n in sorted(d["by_year"].items()))
                      or "none")],
        "promote_to_verdict": primary,
        "verdict_row": ("Is CND-P defined-risk at real fills?", verdict, ""),
        "falsifies": "any exceedance that classifies as a real loss rather than an "
                     "exit-quote artifact.",
    })

    # -- assignment ---------------------------------------------------------
    sections.append({
        "title": f"Early-assignment exposure ({label})",
        "note": (
            f"`short_hi` is in the money at entry by construction "
            f"({a['short_hi_itm_at_entry']:.1%} — it is the strike above spot). At "
            f"the post-print close it is still in the money "
            f"**{a['short_hi_itm_at_post_print_close']:.1%}** of the time, by more "
            f"than one spacing **{a['short_hi_itm_by_more_than_one_spacing']:.1%}** "
            f"of the time (median depth {a['median_short_hi_depth_pct']:.1f}%). "
            f"`short_lo` is in the money "
            f"{a['short_lo_itm_at_post_print_close']:.1%}; both shorts "
            f"{a['both_shorts_itm']:.1%}."),
        "body": [
            f"Entry DTE: median {a['dte_entry']['median']:.0f}, range "
            f"{a['dte_entry']['min']:.0f}-{a['dte_entry']['max']:.0f}.",
            "",
            "Deep in-the-money American puts are the ones worth exercising early. "
            "The position is closed at the first post-print close, so the exposure "
            "window is one session — but it is a real one, and the "
            f"{a['short_hi_itm_by_more_than_one_spacing']:.1%} figure is the share "
            "of events where the assigned leg would not have been covered by the "
            "long wing's intrinsic alone.",
        ],
        "promote_to_verdict": primary,
        "verdict_row": ("How exposed is the in-the-money short to assignment?",
                        f"**{a['short_hi_itm_at_post_print_close']:.1%}** ITM at the "
                        f"post-print close, "
                        f"{a['short_hi_itm_by_more_than_one_spacing']:.1%} by more "
                        f"than one spacing", ""),
    })

    # -- resolvability ------------------------------------------------------
    body = ["| width | resolvability | median spacing (% of spot) | top failure cause |",
            "|---|---:|---:|---|"]
    for w_key, block in stage0["widths"].items():
        causes = block["fail_causes"]
        top = max(causes.items(), key=lambda kv: kv[1])[0] if causes else "—"
        body.append(
            f"| {w_key} | {block['resolvability']:.1%} | "
            f"{block['spacing_pct_median']:.2f}% | {top} |")
    rule = stage0["width_rule"]
    body += [
        "",
        f"Median oquants implied move over the same events: "
        f"**{rule['median_implied_move_pct_oquants']:.2f}%**. The registered rule "
        f"takes the smallest width resolving on >= {rule['resolve_floor']:.0%} of "
        f"events whose median spacing is at least that — chosen: "
        f"**{rule['chosen_width']}**"
        + (" (by the fallback branch)" if rule["chosen_by_fallback"] else "") + ".",
        "",
        "Demanding EXACT symmetry is what costs resolvability. It is not "
        "negotiable: an unevenly spaced condor is not defined-risk, so an event "
        "whose ladder cannot carry one is refused rather than approximated.",
    ]
    if skips:
        total = sum(int(v) for v in skips.values())
        body += [
            "",
            f"Replay skip ledger at width {width} ({total:,} planned events "
            "produced no row):",
            "",
            "| reason | events |", "|---|---:|",
        ] + [f"| `{k}` | {int(v):,} |"
             for k, v in sorted(skips.items(), key=lambda kv: -int(kv[1]))
             if int(v)]
        zero_cost = int(skips.get("zero_cost", 0))
        if zero_cost:
            body += [
                "",
                f"**{zero_cost:,} events were dropped as `zero_cost`** — the condor "
                "priced at a credit at some fill alpha, and `ret` is quoted on the "
                "debit, so there is no denominator. Every number in this report is "
                "therefore conditioned on the structure costing something at the "
                "BEST fill: the cheapest condors, which are the ones the market "
                "priced closest to free, are systematically absent. That is the "
                "same conditioning EXP-102 found on CAL-P and it is not fixed here.",
            ]
    sections.append({
        "title": "Resolvability — what exact symmetry costs",
        "body": body,
    })

    # -- the oracle ceiling -------------------------------------------------
    hind = o["hindsight_quintile_by_realized_abs_move"]
    trad = o["tradeable_quintile_by_implied_move"]
    rows = []
    for block, name in ((hind, "realized |move| (HINDSIGHT)"),
                        (trad, "implied move (tradeable)")):
        for q in block.get("quintiles", []):
            rows.append([name, str(q["quintile"]), f"{q['n']:,}",
                         f"{q['mean_ret'] * 100:+.2f}%", f"{q['win_rate']:.1%}"])
    ceiling = hind.get("quietest_quintile_mean_ret")
    sections.append({
        "title": f"The oracle ceiling ({label})",
        "note": (
            "Sorting by the REALIZED absolute move is hindsight and is not a "
            "strategy. It is the bound: the quietest quintile is the best any "
            "selector could do if it knew the answer, so a gate (EXP-122) has to "
            f"fit under it. Quietest quintile by realized move: "
            f"**{ceiling * 100:+.2f}%/trade** "
            f"(win {hind.get('quietest_quintile_win_rate', 0):.1%}). The same cut "
            f"on the implied move — a quote known before the print — returns "
            f"**{trad.get('quietest_quintile_mean_ret', float('nan')) * 100:+.2f}%**."),
        "columns": ["cut", "quintile", "n", "mean ret", "win"],
        "align": ["---", "---:", "---:", "---:", "---:"],
        "rows": rows,
        "promote_to_verdict": primary,
        "verdict_row": ("What is the ceiling for a CND-P gate?",
                        f"**{ceiling * 100:+.2f}%/trade** in the quietest quintile "
                        f"by realized move (hindsight)", ""),
    })

    # -- geometry + breakeven ----------------------------------------------
    sections.append({
        "title": "Realized geometry, breakeven alpha, deployment",
        "body": [
            f"A market wider than half its mid was quoted somewhere in "
            f"**{g['wide_market_share']:.1%}** of trades (repaired crossed quote: "
            f"{g['quote_repaired_share']:.1%}). The in-the-money wing is the leg "
            "that drives this — deep ITM puts are the thinnest quotes on the "
            "ladder — and the share rises with width, which is a cost of the wide "
            "geometry that the mid-fill assumption does not price.",
            "",
            f"Realized spacing: median **{g['spacing_pct_median']:.2f}%** of spot "
            f"(p25 {g['spacing_pct_p25']:.2f}%, p75 {g['spacing_pct_p75']:.2f}%). "
            f"Debit as a share of the maximum payoff: median "
            f"**{g['cost_over_spacing_median']:.3f}** — the terminal payoff peaks "
            "at exactly one spacing, so this is the fraction of the best case paid "
            "up front, and one minus it is the most the trade can make.",
            "",
            f"Breakeven alpha **{head.get('breakeven_alpha')}**. CND-P crosses FOUR "
            "bid-ask spreads to open and four to close, against two for STR-THRU "
            "(0.448), CAL-P (0.475) and STR-RUNUP (0.478), so a worse breakeven is "
            "expected on leg count alone; above 0.5 the structure does not make "
            "money at mid at all.",
            "",
            f"Peak deployment {head['deployment']['peak']:.2f}x (cap "
            f"{head['deployment']['cap']}), worst cash "
            f"{head['deployment']['worst_cash']:.2%}, max concurrency "
            f"{head.get('max_concurrency')}.",
        ]})

    # -- tail injection -----------------------------------------------------
    if tail.get("available") is False:
        tail_body = [f"NOT RUN: {tail.get('note')}"]
    else:
        mc5 = tail.get("mc", {}).get("0.05", {})
        tail_body = [
            f"The shock set is the worst 1% by ABSOLUTE realized move, not by "
            f"signed move: a condor is hurt by a big move in either direction, and "
            f"ranking on the signed move would shock only the down tail and leave "
            f"half the ruin cases untouched. "
            f"{tail.get('n_shocked')} trades re-priced; worst trade "
            f"{tail.get('base_worst_trade', float('nan')) * 100:.1f}% → "
            f"**{tail.get('shocked_worst_trade', float('nan')) * 100:.1f}%** of the "
            f"debit. MC P(loss) at 5%: **{mc5.get('p_loss')}**, terminal p05 "
            f"{mc5.get('terminal_p05')}.",
            "",
            "A doubled move past either wing takes the structure to its floor and no "
            "further — that is the point of the geometry, and the shock is a check "
            "that the floor holds at quoted exits rather than only at expiry.",
        ]
    sections.append({"title": "Tail injection (mandatory — has_short_leg)",
                     "body": tail_body})
    return sections


if __name__ == "__main__":
    main()
