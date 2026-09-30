"""Pure market-regime calculations over caller-supplied source observations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .panel_math import _anchor_index

__all__ = ["add_regime_features"]


def add_regime_features(
    events: pd.DataFrame,
    market: pd.DataFrame,
    *,
    as_of_column: str = "date",
) -> pd.DataFrame:
    """Port legacy regime math without its filesystem reader or cache.

    ``market`` provides chronological normalized, timezone-naive dates and
    float-convertible closes. Event/decision dates are normalized by callers.
    Select strictly before each event and on-or-before its explicit decision,
    retaining the selected source date as ``regime_asof`` even if a window is
    too short to produce a feature. Missing windows stay NaN; no eligible row
    means NaT. Inputs are not mutated; malformed inputs are not repaired.
    """
    closes = market["close"].to_numpy(dtype=float)
    dates = market["date"].to_numpy()
    simple_ret = closes[1:] / closes[:-1] - 1.0

    out = events.copy()
    n = len(out)
    cols = {c: np.full(n, np.nan) for c in
            ("spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol5",
             "spy_vol20", "spy_vol60", "spy_vol252", "spy_vol20_rel252")}
    anchor = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    idx = _anchor_index(
        dates,
        out["date"].to_numpy(),
        out[as_of_column].to_numpy() if as_of_column != "date" else None,
    )
    for i, j in enumerate(idx):
        if j < 0 or j >= len(closes):
            continue
        anchor[i] = dates[j]
        spot = closes[j]
        if j >= 21:
            cols["spy_ret21"][i] = (spot / closes[j - 21] - 1.0) * 100
        if j >= 63:
            cols["spy_ret63"][i] = (spot / closes[j - 63] - 1.0) * 100
        if j >= 252:
            cols["spy_ret252"][i] = (spot / closes[j - 252] - 1.0) * 100
            cols["spy_dd252"][i] = (spot / closes[j - 251 : j + 1].max() - 1.0) * 100
        vols = {}
        for window in (5, 20, 60, 252):
            if j >= window:
                vols[window] = simple_ret[j - window : j].std(ddof=1) * np.sqrt(252) * 100
                cols[f"spy_vol{window}"][i] = vols[window]
        if 20 in vols and 252 in vols and vols[252] > 0:
            cols["spy_vol20_rel252"][i] = vols[20] / vols[252] - 1.0
    for name, values in cols.items():
        out[name] = values
    out["regime_asof"] = anchor
    return out
