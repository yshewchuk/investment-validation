#!/usr/bin/env python3
"""EXP-124 — TWIN-P shifted: buying room on the side that decays fastest.

Run:  python3 experiments/EXP-124_.../run.py

Primary is anchor_offset=+1. Offsets 0 and +2 run as labelled grid cells, 0
being EXP-123's baseline re-run on identical code so the comparison is not made
across two runs. Only the primary writes REPORT.md.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shifted  # noqa: E402

HERE = Path(__file__).resolve().parent
BASELINE_MEAN = 0.0261      # EXP-123 primary, mid fills
BASELINE_BREAKEVEN = 0.489


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    offsets = list(spec["grid"]["anchor_offset"])
    built = {o: shifted.build(o) for o in offsets}
    summary = {}

    for offset in [1] + [o for o in offsets if o != 1]:
        trades = built[offset]
        if trades.empty:
            print(f"[EXP-124] offset {offset:+d}: nothing priced", flush=True)
            continue
        is_primary = offset == 1
        decided = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)]
        chosen = set(decided.loc[decided["passes"].fillna(False).astype(bool), "event_id"])
        evaluated = trades[trades["event_id"].isin(chosen)]
        mid = evaluated[np.isclose(evaluated["fill_alpha"].astype(float), 0.5)]
        summary[offset] = {"n": len(mid), "mean": float(mid["ret"].mean()),
                           "win": float((mid["ret"] > 0).mean()),
                           "segments": shifted.segments(mid)}
        print(f"[EXP-124] offset {offset:+d}: {len(mid):,} pass, "
              f"mean {100*mid['ret'].mean():+.2f}%", flush=True)
        if mid.empty:
            continue

        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["universe"] = f"anchor_offset={offset:+d}"
            cell["grid_cell"] = True

        result = evaluate(
            cell, evaluated, gate=None, run_dir=HERE,
            repricer=(common.make_repricer(
                shifted.STRATEGY, structure=twin_peak(steps=1, anchor_offset=offset))
                if is_primary else None),
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[shifted.trades_path(offset)],
            extra_sections=(lambda r, s=summary, o=offset, p=is_primary:
                            sections(r, s, o, p)),
            write_report=is_primary,
        )
        lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-124] offset {offset:+d}: report "
              f"{result.report_path or '(grid cell)'}", flush=True)

    (HERE / "results" / "offset_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))


def sections(result, summary, offset, is_primary) -> list[dict]:
    label = "PRIMARY" if is_primary else "SECONDARY"
    out = []
    rows = [[f"{o:+d}" + (" (EXP-123 baseline)" if o == 0 else ""), f"{v['n']:,}",
             f"{100*v['mean']:+.2f}%", f"{100*v['win']:.1f}%"]
            for o, v in sorted(summary.items())]
    out.append({
        "title": f"Shift against the unshifted baseline ({label})",
        "note": (f"EXP-123's primary was **+2.61%/trade** on 1,147 events with a "
                 f"breakeven alpha of 0.489. Offset 0 is that same configuration "
                 f"re-run here on identical code, so any difference between it and "
                 f"the published figure is a reproducibility question, not a result."),
        "columns": ["anchor offset", "events", "mean/trade", "win"],
        "align": ["---", "---:", "---:", "---:"],
        "rows": rows,
    })
    seg = summary.get(offset, {}).get("segments", [])
    if seg:
        out.append({
            "title": f"Where the prints landed, and what the structure was worth ({label})",
            "note": ("`exit / debit` is the quantity the shift is meant to move: "
                     "EXP-123 measured 0.14 deep-up against 0.36 deep-down, and "
                     "0.89 against 1.48 on the ramps, at equivalent distances."),
            "columns": ["band", "n", "share", "mean ret", "win", "exit / debit"],
            "align": ["---", "---:", "---:", "---:", "---:", "---:"],
            "rows": [[s["band"], f"{s['n']:,}", f"{100*s['share']:.1f}%",
                      f"{100*s['mean_ret']:+.2f}%", f"{100*s['win']:.1f}%",
                      f"{s['exit_over_debit']:.2f}"] for s in seg],
        })
    return out


if __name__ == "__main__":
    main()
