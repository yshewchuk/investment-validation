"""A parameterized causal same-structure analog, generalizing the one hardcoded
K=25 / dollar-PnL / all-time construction in
``experiments/EXP-161_dynamic_short_vol_neural_structure_picker/run.py::add_causal_analogs``
(which ``engine/score.py::_chooser_analogs`` reproduces bit-for-bit for serving).

Three axes, none of which has ever been measured against an alternative:

``k``             neighbor count. Production/training use 25.
``target``        what is averaged over the neighbors — "pnl" (dollar P&L per
                  contract, the trained default) or "ret" (``pnl / entry_cost``,
                  scale-free across cheap and expensive premiums).
``lookback_days`` the eligible pool's age ceiling. ``None`` reproduces
                  production (all history back to 2018-01-04 is eligible, aged
                  or not); an integer restricts the pool to neighbors whose
                  EXIT fell within that many calendar days before this row's
                  ENTRY — a recency window, in case the analog's usefulness is
                  regime-dependent rather than stationary over 8+ years.

Eligibility is POSITIONAL, not date-filtered, to match the original bit for
bit: the frame is sorted by (entry_date, event_id) and the pool for row i is a
prefix of that order, not every historical row with an earlier date. That
mirrors what a live board can actually see (same-day siblings have not
exited yet) and reproducing it exactly is what lets ``k=25, target="pnl",
lookback_days=None`` be checked against the shipped construction as a
correctness gate before trusting the sweep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DIMS = ("exp_pnl_sim", "width_over_forecast", "n_legs", "anchor_over_spot", "rel_spread")
ANALOG_COLUMNS = ("analog_mean", "analog_win_rate", "analog_p10", "analog_p90", "analog_n")


def causal_analogs(
    frame: pd.DataFrame, *, k: int, target: str, lookback_days: int | None,
) -> pd.DataFrame:
    if target not in ("pnl", "ret"):
        raise ValueError(f"target must be 'pnl' or 'ret', got {target!r}")
    out = frame.copy()
    for col in ANALOG_COLUMNS:
        out[col] = np.nan
    for strategy, idx in out.groupby("strategy", sort=False).groups.items():
        pos = np.asarray(list(idx), dtype=int)
        part = out.loc[pos].sort_values(["entry_date", "event_id"]).copy()
        values = part[list(DIMS)].to_numpy(float)
        outcome = part[target].to_numpy(float)
        entry = part["entry_date"].to_numpy()
        exits = part["exit_date"].to_numpy()
        n = len(part)
        for i in range(n):
            causal = exits[:i] < entry[i]
            if lookback_days is not None:
                floor = entry[i] - np.timedelta64(int(lookback_days), "D")
                causal = causal & (exits[:i] >= floor)
            usable = (
                causal
                & np.isfinite(outcome[:i])
                & np.isfinite(values[:i]).all(1)
                & np.isfinite(values[i]).all()
            )
            pool = values[:i][usable]
            if len(pool) < k:
                continue
            pool_outcome = outcome[:i][usable]
            scale = pool.std(0, ddof=1)
            scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
            distance = (((pool - values[i]) / scale) ** 2).mean(1)
            take = np.argpartition(distance, k - 1)[:k]
            analog = pool_outcome[take]
            target_idx = int(part.index[i])
            out.at[target_idx, "analog_mean"] = float(analog.mean())
            out.at[target_idx, "analog_win_rate"] = float((analog > 0).mean())
            out.at[target_idx, "analog_p10"] = float(np.quantile(analog, 0.10))
            out.at[target_idx, "analog_p90"] = float(np.quantile(analog, 0.90))
            out.at[target_idx, "analog_n"] = float(k)
    return out
