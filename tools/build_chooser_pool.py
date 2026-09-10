#!/usr/bin/env python3
"""Build the DYN-SV chooser's analog pool.

The champion `dyn_sv_chooser_v1_1` was fitted with five analog columns produced
by EXP-161's `add_causal_analogs`: for each candidate, the 25 nearest PRIOR
candidates in the space of the structure's own geometry and simulated payoff,
averaged as dollar P&L. Serving those faithfully needs the population they were
drawn from, which is what this writes.

Source is EXP-137's enumerated candidate panel, because that is literally the
frame EXP-161 walked -- not the engine.replay trades, which carry neither
`exp_pnl_sim` nor the width/anchor/spread geometry, and which hold no rows at
all for RAMP7 or CTR5.

Verified 2026-09-10: the served neighbourhood reproduces the training values on
all 64,910 rows that had them -- bit-exact for win_rate/p10/p90/n and to 9e-16
for the mean, with identical coverage.

    python3 tools/build_chooser_pool.py [--source PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine import paths  # noqa: E402
from engine.score import CHOOSER_ANALOG_POOL, Scorer  # noqa: E402

E161 = ROOT / "experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py"
DEFAULT_SOURCE = ROOT / "experiments/EXP-137_one_book_per_family_which_enumerated_str/results/candidates.parquet"

#: EXP-169's menu7-prime: the enumerated arm -> the registered structure name.
MENU = {
    "N7q2_-1_-1_1": "TWIN-P", "N5q2_-2_1": "TWIN-P5", "N4q0_-1_1": "CND-PS",
    "N3q-2_1": "BFLY-P", "N5q-4_1_1": "BFLY-P5",
    "N7q-2_-1_1_1": "RAMP7", "N5q-2_-1_2": "CTR5",
}
#: Mid fill, because that is the book the chooser was trained and promoted on.
ALPHA = 0.5


def _load_e161():
    spec = importlib.util.spec_from_file_location("e161_pool", E161)
    module = importlib.util.module_from_spec(spec)
    sys.modules["e161_pool"] = module
    spec.loader.exec_module(module)
    return module


def build(source: Path) -> pd.DataFrame:
    e161 = _load_e161()
    columns = ["arm", "event_id", "ticker", "event_date", "entry_date", "exit_date",
               "fill_alpha", "entry_cost", "exit_value", "exit_value_expiry",
               "exit_arb_ok", "exp_pnl_sim", "width_over_forecast",
               "anchor_over_spot", "n_legs", "rel_spread"]
    raw = pd.read_parquet(source, columns=columns)
    raw = raw[raw["arm"].isin(MENU) & np.isclose(raw["fill_alpha"].astype(float), ALPHA)].copy()
    raw["strategy"] = raw["arm"].map(MENU)
    for column in ("event_date", "entry_date", "exit_date"):
        raw[column] = pd.to_datetime(raw[column]).dt.normalize()
    # The same conditional-exit repair the training frame went through; without
    # it the realized P&L is not the one the neighbourhood averaged.
    raw = e161.conditional_exit(raw)
    raw["pnl"] = raw["exit_value"] - raw["entry_cost"]
    keep = ["strategy", "event_id", "ticker", "entry_date", "exit_date", "pnl",
            *Scorer._CHOOSER_ANALOG_DIMS]
    out = raw[keep].dropna(subset=["exit_date", "pnl"])
    return out.sort_values(["strategy", "entry_date"]).reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        print(f"source not found: {source}", file=sys.stderr)
        return 1
    frame = build(source)
    out = Path(args.out) if args.out else paths.assert_writable(
        paths.FEATURES / CHOOSER_ANALOG_POOL)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    span = f"{frame['entry_date'].min().date()} → {frame['entry_date'].max().date()}"
    print(f"{len(frame):,} rows, {frame['strategy'].nunique()} structures, {span}")
    print(frame.groupby("strategy").size().to_string())
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
