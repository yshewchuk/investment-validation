"""EXP-189 account (design section 2): EXP-188 Amendment 2 policy with start_equity 200,000.

Reuses ``experiments.size_pnl_account``: its ``margin`` (EXP-134 ``margin.simulate``, unchanged) and its
``secured_per_contract``. Only the starting equity differs: every short put is secured at strike x 100,
cap 66% of marked equity, target share 25% of headroom, expensive-first on entry-date ties.
"""
import numpy as np
import pandas as pd

from experiments import size_pnl_account as base

START_EQUITY = 200_000.0
POLICY = {**base.POLICY, "start_equity": START_EQUITY}

secured_per_contract = base.secured_per_contract


def simulate_account(rows, mark):
    """Metrics of one account over the TAKEN ``rows``; ``mark`` "mid" walks the exit_value_mid marks instead of the primary ones."""
    if mark not in ("mid", "primary"):
        raise ValueError("mark must be 'mid' or 'primary'")
    start = POLICY["start_equity"]
    if rows.empty:
        return {"wanted": 0, "funded": 0, "final_equity": start, "total_return": 0.0, "max_drawdown": 0.0, "peak_open": 0}
    exit_col = "exit_value_mid" if mark == "mid" else "exit_value"
    for col in ("entry_cost", "secured_per_contract", exit_col):  # a missing or non-finite input is refused, never simulated
        if col not in rows.columns or not np.isfinite(rows[col].to_numpy(dtype=float)).all():
            raise ValueError(f"account inputs must be finite: {col}")
    for col in ("entry_date", "exit_date"):  # a missing or unparseable date, or NaT for any row, is refused, never simulated
        try:
            parsed = pd.to_datetime(rows[col])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("account inputs must carry valid entry_date and exit_date") from exc
        if parsed.isna().any():
            raise ValueError("account inputs must carry valid entry_date and exit_date")
    frame = rows.drop(columns=["exit_value"]).rename(columns={"exit_value_mid": "exit_value"}) if mark == "mid" else rows
    t = base.margin.simulate(frame[["entry_date", "exit_date", "entry_cost", "exit_value", "secured_per_contract"]], **POLICY)
    final = float(t.attrs["final_equity"])
    done, max_dd = t[t["funded"]], 0.0
    if not done.empty:  # the realised curve at EXIT DATES: same-date P&L is summed before any peak is taken
        per_date = done.assign(exit_date=pd.to_datetime(done["exit_date"])).groupby("exit_date")["pnl_usd"].sum().sort_index()
        curve = start + np.concatenate([[0.0], np.cumsum(per_date.to_numpy(dtype=float))])
        max_dd = float(np.min(curve / np.maximum.accumulate(curve) - 1.0))
    return {"wanted": len(t), "funded": int(t["funded"].sum()), "final_equity": final, "total_return": final / start - 1.0,
            "max_drawdown": max_dd, "peak_open": int(t["concurrency"].max()) if len(t) else 0}


def quarter_returns(rows, quarters):
    """Account return per calendar quarter: funded pnl_usd by EXIT quarter over the equity at the quarter's start; 0.0 for a quarter with none."""
    pnl = {}
    if not rows.empty:
        simulate_account(rows, "mid")  # validates the inputs exactly as the account does
        t = base.margin.simulate(rows.drop(columns=["exit_value"]).rename(columns={"exit_value_mid": "exit_value"})[
            ["entry_date", "exit_date", "entry_cost", "exit_value", "secured_per_contract"]], **POLICY)
        done = t[t["funded"]]
        pnl = done.assign(q=pd.PeriodIndex(pd.to_datetime(done["exit_date"]), freq="Q")).groupby("q")["pnl_usd"].sum().to_dict()
    equity, out = START_EQUITY, []
    for q in quarters:
        out.append(float(pnl.get(q, 0.0)) / equity)
        equity += float(pnl.get(q, 0.0))
    return out
