"""EXP-189 account (design section 2): EXP-188 Amendment 2 policy with start_equity 200,000, reusing size_pnl_account."""
import numpy as np
import pandas as pd

from experiments import size_pnl_account as base
from experiments.fair_value_fills import VARIANTS

START_EQUITY = 200_000.0
POLICY = {**base.POLICY, "start_equity": START_EQUITY}

secured_per_contract = base.secured_per_contract


def deciding_mark(rows):
    """"primary" on an F sweep's rows (F decides), else "mid"."""
    return "primary" if "entry_cost_mid" in rows.columns else "mid"


def deciding_columns(rows):
    """(exit, pnl) columns the objective and T8/T9 read: F's on F rows, else the mid pair."""
    return ("exit_value", "pnl_dollars") if "entry_cost_mid" in rows.columns else ("exit_value_mid", "pnl_dollars_mid")


def _columns(rows, mark):
    """(entry, exit) columns of view ``mark`` on ``rows``; an unknown mark, or one these rows do not carry, is a ValueError."""
    if mark == "primary":
        return "entry_cost", "exit_value"
    if mark == "mid":
        return ("entry_cost_mid", "exit_value_mid") if "entry_cost_mid" in rows.columns else ("entry_cost", "exit_value_mid")
    if "entry_cost_mid" in rows.columns and (mark == "harsh" or mark in VARIANTS):
        return f"entry_cost_{mark}", f"exit_value_{mark}"
    raise ValueError(f"mark must be 'mid' or 'primary' (or an F view on F rows), got {mark!r}")


def _frame(rows, mark):
    entry, exit_ = _columns(rows, mark)
    return rows.assign(entry_cost=rows[entry], exit_value=rows[exit_])[["entry_date", "exit_date", "entry_cost", "exit_value", "secured_per_contract"]]


def simulate_account(rows, mark):
    """Metrics of one account over the TAKEN ``rows`` at view ``mark``; each view funds at its own debit."""
    entry, exit_ = _columns(rows, mark)  # an unknown mark is refused even for empty rows
    start = POLICY["start_equity"]
    if rows.empty:
        return {"wanted": 0, "funded": 0, "final_equity": start, "total_return": 0.0, "max_drawdown": 0.0, "peak_open": 0}
    for col in (entry, "secured_per_contract", exit_):  # a missing or non-finite input is refused, never simulated
        if col not in rows.columns or not np.isfinite(rows[col].to_numpy(dtype=float)).all():
            raise ValueError(f"account inputs must be finite: {col}")
    for col in ("entry_date", "exit_date"):  # a missing or unparseable date, or NaT for any row, is refused, never simulated
        try:
            parsed = pd.to_datetime(rows[col])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("account inputs must carry valid entry_date and exit_date") from exc
        if parsed.isna().any():
            raise ValueError("account inputs must carry valid entry_date and exit_date")
    t = base.margin.simulate(_frame(rows, mark), **POLICY)
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
        simulate_account(rows, deciding_mark(rows))  # validates the inputs exactly as the account does
        t = base.margin.simulate(_frame(rows, deciding_mark(rows)), **POLICY)
        done = t[t["funded"]]
        pnl = done.assign(q=pd.PeriodIndex(pd.to_datetime(done["exit_date"]), freq="Q")).groupby("q")["pnl_usd"].sum().to_dict()
    equity, out = START_EQUITY, []
    for q in quarters:
        out.append(float(pnl.get(q, 0.0)) / equity)
        equity += float(pnl.get(q, 0.0))
    return out
