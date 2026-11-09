"""Pure conditional settlement for experimental candidate trades."""
from __future__ import annotations

import pandas as pd


def conditional_exit(raw: pd.DataFrame) -> pd.DataFrame:
    """Repair invalid conditional exits and calculate realized P&L.

    The input is copied. Rows with missing ``entry_cost`` or post-repair
    ``exit_value`` are dropped; a remaining loss below the debit is refused.
    """
    out = raw.copy()
    hold = (~out["exit_arb_ok"].fillna(True).astype(bool)) | (out["exit_value"] < 0)
    out.loc[hold, "exit_value"] = out.loc[hold, "exit_value_expiry"]
    out = out.dropna(subset=["entry_cost", "exit_value"]).copy()
    out["pnl"] = out["exit_value"] - out["entry_cost"]
    out["ret"] = out["pnl"] / out["entry_cost"]
    if (out["ret"] < -1.0 - 1e-8).any():
        raise RuntimeError("conditional exits still contain a loss below the debit")
    return out
