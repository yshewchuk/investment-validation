#!/usr/bin/env python3
"""EXP-125 — TWIN-P sized to the forecast.

Run:  python3 experiments/EXP-125_.../run.py

Primary is the PREDICTED-width arm. The IMPLIED-width arm is the registered
control and runs as a labelled grid cell — it is the benchmark the primary has
to beat, not a cell whose result may be promoted if it happens to win.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from engine.evaluate import evaluate  # noqa: E402
from engine.structures import twin_peak  # noqa: E402
from experiments import common, lib  # noqa: E402

import sized  # noqa: E402

HERE = Path(__file__).resolve().parent
FIXED_WIDTH_MEAN = 0.0547     # EXP-124 offset +1
FIXED_WIDTH_DEAD = 0.375      # its share landing beyond a wing


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    spy = common.load_spy_daily()
    built = {arm: sized.build(arm) for arm in spec["grid"]["sizing"]}
    summary = {}

    for arm in ["predicted", "implied"]:
        trades = built.get(arm)
        if trades is None or trades.empty:
            print(f"[EXP-125] {arm}: nothing priced", flush=True)
            continue
        is_primary = arm == "predicted"
        dec = trades[np.isclose(trades["fill_alpha"].astype(float), 0.5)]
        chosen = set(dec.loc[dec["passes"].fillna(False).astype(bool), "event_id"])
        ev = trades[trades["event_id"].isin(chosen)]
        mid = ev[np.isclose(ev["fill_alpha"].astype(float), 0.5)].copy()
        summary[arm] = landing(mid)
        print(f"[EXP-125] {arm}: {len(mid):,} pass, mean {100*mid['ret'].mean():+.2f}%",
              flush=True)
        if mid.empty:
            continue

        cell = dict(spec)
        if not is_primary:
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"]["universe"] = "implied-sized control"
            cell["grid_cell"] = True

        result = evaluate(
            cell, ev, gate=None, run_dir=HERE,
            repricer=None,   # per-event widths: a shifted-date reprice would
                             # need the forecast re-derived at the shifted date
            tail_shock=common.abs_move_tail_shock, spy_daily=spy,
            input_files=[sized.trades_path(arm)],
            extra_sections=(lambda r, s=summary, a=arm, p=is_primary: sections(r, s, a, p)),
            write_report=is_primary,
        )
        lib.record_evaluation(HERE, cell, result.results)
        print(f"[EXP-125] {arm}: report {result.report_path or '(control)'}", flush=True)

    (HERE / "results" / "arm_summary.json").write_text(
        json.dumps(summary, indent=1, default=str))


def landing(mid: pd.DataFrame) -> dict:
    if mid.empty:
        return {}
    d = mid["legs"].apply(json.loads)
    A = d.apply(lambda x: {l["name"]: float(l["strike"]) for l in x["entry"]}["atm"])
    pos = ((d.apply(lambda x: float(x["spot_exit"])) - A) / mid["w"]).abs()
    return {
        "n": int(len(mid)),
        "mean": float(mid["ret"].mean()),
        "win": float((mid["ret"] > 0).mean()),
        "median": float(mid["ret"].median()),
        "w_pct_spot": float((100 * mid["w"] / mid["spot_entry"]).median()),
        "c_over_w": float((mid["cost"] / mid["w"]).median()),
        "dead": float((pos >= 4).mean()),
        "inside_2w": float((pos < 2).mean()),
        "years_positive": int(sum(1 for _, g in mid.groupby("year") if g["ret"].mean() > 0)),
        "years": int(mid["year"].nunique()),
    }


def sections(result, summary, arm, is_primary) -> list[dict]:
    label = "PRIMARY (predicted)" if is_primary else "CONTROL (implied)"
    rows = []
    for k, v in summary.items():
        if not v:
            continue
        rows.append([k, f"{v['n']:,}", f"{100*v['mean']:+.2f}%", f"{100*v['win']:.1f}%",
                     f"{v['w_pct_spot']:.2f}%", f"{100*v['dead']:.1f}%",
                     f"{100*v['inside_2w']:.1f}%", f"{v['years_positive']}/{v['years']}"])
    rows.append(["fixed width (EXP-124 +1)", "925", "+5.47%", "55.0%", "1.08%", "37.5%",
                 "32.2%", "6/9"])
    return [{
        "title": f"Sized against fixed width, and the forecast against the market ({label})",
        "note": ("The control is the point. Better geometry gets PRICED — a structure "
                 "more likely to pay costs more — so per-event sizing only wins to the "
                 "extent the sizing signal beats the market's own forecast. size_v1_4 "
                 "does beat it on identical rows (MAE 3.796pp against oquants' 4.637, "
                 "winning in all 14 years), and this table is whether that carries "
                 "through into a traded structure."),
        "columns": ["arm", "n", "mean/trade", "win", "w %spot", "dead", "inside ±2w", "years+"],
        "align": ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:"],
        "rows": rows,
        "body": ["", "`dead` is the share landing beyond a wing, where the structure pays "
                 "nothing. Fixed width sat at 37.5%; if per-event sizing does not move "
                 "that, `c < w` has re-selected the treatment away and a null result "
                 "cannot be read as the forecast failing."],
    }]


if __name__ == "__main__":
    main()
