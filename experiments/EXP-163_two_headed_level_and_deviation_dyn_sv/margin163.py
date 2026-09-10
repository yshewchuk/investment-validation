#!/usr/bin/env python3
"""EXP-163 funding walk: EXP-134 cash-secured simulate with an optional
same-day priority column. The EXP-134 module is imported untouched for its
constants and secured-per-contract rule; only the walk is re-implemented here
so that ``priority_col`` can order same-day competitors by expected PnL per
secured dollar instead of by secured requirement."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_MARGIN_PATH = ROOT / "experiments/EXP-134_priced_right_funded_and_held_structure_s/margin.py"
_spec = importlib.util.spec_from_file_location("exp163_margin_base", _MARGIN_PATH)
base = importlib.util.module_from_spec(_spec)
sys.modules["exp163_margin_base"] = base
assert _spec.loader is not None
_spec.loader.exec_module(base)

CONTRACT_MULTIPLIER = base.CONTRACT_MULTIPLIER


def simulate(trades: pd.DataFrame, *, start_equity: float, cap: float,
             target_share: float | None, priority_col: str | None = None) -> pd.DataFrame:
    """Walk the book exactly as EXP-134 margin.simulate does, with one change:
    when ``priority_col`` is given, same-entry-date trades are visited in
    descending priority instead of descending secured requirement."""
    if priority_col is None:
        order, ascending = ["entry_date", "secured_per_contract"], [True, False]
    else:
        order, ascending = ["entry_date", priority_col], [True, False]
    t = trades.sort_values(order, ascending=ascending, kind="stable").reset_index(drop=True)
    entry = pd.to_datetime(t["entry_date"]).to_numpy()
    exit_ = pd.to_datetime(t["exit_date"]).to_numpy()
    cost = t["entry_cost"].to_numpy(dtype=float) * CONTRACT_MULTIPLIER
    value = t["exit_value"].to_numpy(dtype=float) * CONTRACT_MULTIPLIER
    secure = t["secured_per_contract"].to_numpy(dtype=float)

    cash = start_equity
    open_pos: list[tuple] = []
    out = np.zeros((len(t), 6))

    for i in range(len(t)):
        still = []
        for pos in open_pos:
            if pos[0] <= entry[i]:
                cash += pos[1] * pos[3]
            else:
                still.append(pos)
        open_pos = still
        deployed = sum(p[1] * p[2] for p in open_pos)
        secured_now = sum(p[4] for p in open_pos)
        equity = cash + deployed
        headroom = cap * equity - secured_now
        if target_share is None:
            want = (0.05 * equity / cost[i]) if cost[i] > 0 else 0.0
        else:
            want = (target_share * headroom / secure[i]) if secure[i] > 0 else np.inf
            want = max(want, 1.0)
        allowed = (headroom / secure[i]) if secure[i] > 0 else want
        contracts = max(0.0, min(want, allowed))
        funded = contracts >= 1.0 if secure[i] > 0 else contracts > 0
        if not funded:
            contracts = 0.0
        else:
            contracts = np.floor(contracts)
        if contracts > 0:
            cash -= contracts * cost[i]
            open_pos.append((exit_[i], contracts, cost[i], value[i], contracts * secure[i]))
        out[i] = [contracts, equity, secured_now, headroom, float(funded), len(open_pos)]

    for pos in open_pos:
        cash += pos[1] * pos[3]

    t["contracts"] = out[:, 0]
    t["equity_at_entry"] = out[:, 1]
    t["secured_before"] = out[:, 2]
    t["headroom_at_entry"] = out[:, 3]
    t["funded"] = out[:, 4].astype(bool)
    t["concurrency"] = out[:, 5].astype(int)
    t["pnl_usd"] = t["contracts"] * (value - cost)
    t.attrs["final_equity"] = cash
    return t


def secured_per_contract(legs) -> float:
    return base.secured_per_contract(legs)
