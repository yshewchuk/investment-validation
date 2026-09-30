"""Pure shared event-streak and adjusted-price features for native callers."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from engine.v2.features.panel_math import _anchor_index

_MARKET_COLUMNS = ("dist_high", "dist_ema", "ret5", "ret10", "ret20")


def _signed_streak(out: pd.DataFrame) -> np.ndarray:
    """Preserve the legacy recursion over prior events, including zero moves."""
    signs = np.sign(out["move"].to_numpy(dtype=float))
    tickers = out["ticker"].to_numpy()
    length_before = np.zeros(len(out), dtype=int)
    sign_before = np.zeros(len(out), dtype=int)
    run, prev_sign = 0, 0
    for i in range(len(out)):
        if i == 0 or tickers[i] != tickers[i - 1]:
            length_before[i], sign_before[i] = 0, 0
            run, prev_sign = 1, signs[i]
            continue
        length_before[i], sign_before[i] = run, prev_sign
        if signs[i] == prev_sign and signs[i] != 0:
            run += 1
        else:
            run, prev_sign = 1, signs[i]
    return length_before * sign_before


def _price_features(group: pd.DataFrame, px: pd.DataFrame,
                    as_of_column: str) -> dict[str, np.ndarray]:
    px = px.sort_values("date")
    closes = px["close_adj"].to_numpy(dtype=float)
    pdates = px["date"].to_numpy()
    ema252 = pd.Series(closes).ewm(span=252, adjust=False).mean().to_numpy()
    high252 = pd.Series(closes).rolling(252, min_periods=120).max().to_numpy()
    result = {name: np.full(len(group), np.nan) for name in _MARKET_COLUMNS}
    result["runup_asof"] = np.full(len(group), np.datetime64("NaT", "ns"),
                                   dtype="datetime64[ns]")
    row_idx = _anchor_index(pdates, group["date"].to_numpy(),
                            group[as_of_column].to_numpy())
    for j, idx in enumerate(row_idx):
        idx = int(idx)
        if idx < 0:
            continue
        result["runup_asof"][j] = pdates[idx]
        if (idx >= 252 and np.isfinite(high252[idx])
                and np.isfinite(ema252[idx]) and ema252[idx] > 0):
            result["dist_high"][j] = (closes[idx] / high252[idx] - 1.0) * 100
            result["dist_ema"][j] = (closes[idx] / ema252[idx] - 1.0) * 100
        if idx >= 20 and closes[idx - 20] > 0:
            result["ret20"][j] = (closes[idx] / closes[idx - 20] - 1.0) * 100
            result["ret10"][j] = (closes[idx] / closes[idx - 10] - 1.0) * 100
            result["ret5"][j] = (closes[idx] / closes[idx - 5] - 1.0) * 100
    return result


def add_runup_features(frame: pd.DataFrame,
                       prices_by_ticker: Mapping[str, pd.DataFrame],
                       as_of_column: str) -> pd.DataFrame:
    """Return legacy-equivalent arithmetic from explicitly supplied inputs.

    The caller owns event-history visibility and adjusted-price source selection.
    Price anchors precede the event and do not exceed the named decision date.
    Explicitly pass ``"date"`` for the historical event-date convention.
    """
    out = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    # Require the decision column even when no ticker has usable price history.
    out[as_of_column]
    out["signed_streak"] = _signed_streak(out)
    out["ema12r_abs"] = out["ema12_prior_abs_move"].where(
        out["n_prior"] >= 12, out["mean_prior_abs_move"])
    for column in _MARKET_COLUMNS:
        out[column] = np.nan
    out["runup_asof"] = np.datetime64("NaT", "ns")
    for ticker, group in out.groupby("ticker", sort=True):
        px = prices_by_ticker.get(ticker)
        if px is None or len(px) < 300 or "close_adj" not in px.columns:
            continue
        for column, values in _price_features(group, px, as_of_column).items():
            out.loc[group.index, column] = values
    return out
