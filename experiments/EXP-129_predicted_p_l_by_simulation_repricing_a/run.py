#!/usr/bin/env python3
"""EXP-129 — Predicted P&L by simulation: repricing a twin peak under paired move and crush draws.

Run:  python3 experiments/EXP-129_predicted_p_l_by_simulation_repricing_a/run.py

Pre-registration lives in spec.yaml; engine.evaluate enforces it. The primary
spec's OOS result is the headline; grid cells are secondary.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from engine.evaluate import evaluate  # noqa: E402
from experiments import lib  # noqa: E402

HERE = Path(__file__).resolve().parent


def build_trades():
    """Return the priced trade frame (engine.replay output shape).

    TODO(experimenter): generate the candidate trades via engine.replay over
    the event universe, or load a pre-built priced dataset. One row per
    (event x fill_alpha) with at least the columns engine.evaluate requires:
    event_id, ticker, event_date, entry_date, exit_date, fill_alpha,
    entry_cost, exit_value, ret.
    """
    raise NotImplementedError("wire the candidate trade generator here")


def main() -> None:
    spec = lib.load_spec(HERE / "spec.yaml")
    trades = build_trades()

    # Grid: the primary spec runs first and is the headline; each grid cell
    # then runs as a secondary spec (its own ledger row, labeled in the
    # report appendix).
    result = evaluate(spec, trades, run_dir=HERE)
    lib.record_evaluation(HERE, spec, result.results)

    grid = spec.get("grid") or {}
    for key, values in grid.items():
        for value in values:
            cell = dict(spec)
            cell["primary_spec"] = dict(spec["primary_spec"])
            cell["primary_spec"][key] = value
            # Grid cells legitimately differ from the registered primary spec;
            # the label exempts them from the spec-hash continuity check —
            # they are secondary results, never the headline.
            cell["grid_cell"] = True
            cell_result = evaluate(cell, trades, run_dir=HERE)
            lib.record_evaluation(HERE, cell, cell_result.results)

    print(f"report: {result.report_path}")


if __name__ == "__main__":
    main()
