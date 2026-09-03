#!/usr/bin/env python3
"""EXP-123 — TWIN-P twin-peak: does a cheap structure on stable mega-caps pay.

Run:  python3 experiments/EXP-123_twin_p_twin_peak_does_a_cheap_structure/run.py

Primary: steps=1, >=$10B, all four registered entry filters. Secondaries, all
labelled: the >$100B slice, and steps=2. Only the primary writes REPORT.md —
every cell shares run_dir, and the default would have each one overwrite the
last (the defect EXP-121 shipped with).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.evaluate import evaluate  # noqa: E402
from engine.structures import twin_peak  # noqa: E402
from experiments import common, lib  # noqa: E402

import twinp  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    built = {steps: twinp.build(steps) for steps in spec["grid"]["steps"]}

    cells = [
        ("primary", 1, "passes", True),
        ("mega_cap_slice", 1, "passes_mega", False),
        ("steps_2", 2, "passes", False),
    ]
    for label, steps, column, is_primary in cells:
        trades = built[steps].copy()
        if trades.empty:
            print(f"[EXP-123] {label}: nothing priced", flush=True)
            continue
        trades["passes_mega"] = trades["passes"] & trades["mega"]

        mid = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)]
        n_pass = int(mid[column].fillna(False).sum())
        print(f"[EXP-123] {label} (steps={steps}): {mid['event_id'].nunique():,} "
              f"resolvable events, {n_pass:,} pass the entry rule", flush=True)

        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["universe"] = label
            cell["grid_cell"] = True

        result = evaluate(
            cell, trades, gate=twinp.make_filter_gate(column), run_dir=HERE,
            repricer=(common.make_repricer(twinp.STRATEGY,
                                           structure=twin_peak(steps=steps))
                      if is_primary else None),
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[twinp.trades_path(steps)],
            extra_sections=(lambda r, t=trades, c=column, p=is_primary:
                            sections(r, t, c, p)),
            write_report=is_primary,
        )
        lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-123] {label}: report "
              f"{result.report_path or '(secondary — see metrics json)'}", flush=True)


def sections(result, trades, column, is_primary) -> list[dict]:
    """The pre-registered required outputs the harness does not already emit."""
    mid = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)].copy()
    sel = mid[mid[column].fillna(False)]
    label = "PRIMARY" if is_primary else "SECONDARY"
    out: list[dict] = []

    rows, cur = [], mid
    rows.append(["resolvable (all seven strikes)", f"{len(cur):,}", ""])
    for name, col in [("+ reward > risk (c < w)", "f_reward"),
                      ("+ rel spread <= 25%", "f_spread"),
                      ("+ mcap >= $10B", "f_mcap")]:
        before = len(cur)
        cur = cur[cur[col].fillna(False)]
        rows.append([name, f"{len(cur):,}", f"-{before - len(cur):,}"])
    out.append({
        "title": f"Entry-rule funnel ({label})",
        "note": ("Each term is arithmetic on the entry close. Nothing here is "
                 "fitted, so there is no training set to leak and the rule is "
                 "the same on every event it has ever seen."),
        "columns": ["filter", "events", "cost"],
        "align": ["---", "---:", "---:"],
        "rows": rows,
    })

    if len(sel):
        docs = sel["legs"].apply(json.loads)
        spread_rt = 2 * docs.apply(
            lambda d: sum(float(l.get("qty", 1)) * 0.5 * (float(l["ask"]) - float(l["bid"]))
                          for l in d["entry"]))
        rr = sel["max_profit"] / sel["cost"]
        out.append({
            "title": f"Reward, risk and the cost of eight legs ({label})",
            "body": [
                f"Median reward:risk **{rr.median():.2f}:1** — max profit "
                f"${sel['max_profit'].median():.2f} against a debit of "
                f"${sel['cost'].median():.2f}, on spacing w = ${sel['w'].median():.2f}.",
                "",
                f"Round-trip spread across the eight legs: median "
                f"**${spread_rt.median():.2f}**, "
                f"{100 * (spread_rt / sel['cost']).median():.0f}% of the debit but "
                f"{100 * (spread_rt / sel['max_profit']).median():.0f}% of max profit — "
                "the second number is the one that decides whether it trades. It "
                f"clears its own round-trip spread on "
                f"**{100 * (spread_rt < sel['max_profit']).mean():.1f}%** of events.",
                "",
                "CND-P for comparison: reward:risk 0.31:1, breakeven alpha 0.501. "
                "That is why this experiment registered its falsification on "
                "execution rather than on the mean.",
            ]})

        move = (docs.apply(lambda d: abs(float(d["spot_exit"]) / float(d["spot_entry"]) - 1.0))
                * sel["spot_entry"] / sel["w"])
        bands = [(0, 0.9, "inside the ATM dip"), (0.9, 2.1, "on the plateau (max payoff)"),
                 (2.1, 3.0, "outer ramp"), (3.0, np.inf, "past the wing — pays nothing")]
        out.append({
            "title": f"Where the realized move landed ({label})",
            "note": ("The payoff is flat at 2w on |move| in [w, 2w], dips to w at zero "
                     "and reaches zero at 4w. That shape is the whole thesis, so where "
                     "the prints actually fell is the diagnostic."),
            "columns": ["band", "events", "share"],
            "align": ["---", "---:", "---:"],
            "rows": [[lbl, f"{int(((move >= lo) & (move < hi)).sum()):,}",
                      f"{100 * float(((move >= lo) & (move < hi)).mean()):.1f}%"]
                     for lo, hi, lbl in bands],
        })

        exceeded = sel[sel["ret"] < -1.0]
        out.append({
            "title": f"Defined-risk falsification ({label})",
            "note": (f"The payoff floor is exactly zero, so a return below -100% of the "
                     f"debit is a statement about the EXIT quotes, not the structure. "
                     f"Observed: **{len(exceeded):,} of {len(sel):,} "
                     f"({100 * len(exceeded) / max(len(sel), 1):.2f}%)** at mid fills, "
                     f"worst **{100 * sel['ret'].min():.1f}%**."),
            "falsifies": "an exceedance that is a real loss rather than an exit-quote "
                         "artifact — the mirror symmetry is what makes the floor zero.",
        })
    return out


if __name__ == "__main__":
    main()
