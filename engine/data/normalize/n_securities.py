"""Normalize per-ticker symbology and size into Tier-2 ``securities``.

One row per (ticker, year): the listing range, the year-end market cap in true
USD, and — importantly — which unit era that figure came from.

``mcap_quantized`` marks the pre-2017-06-28 era, where ORATS delivers market cap
as an *integer number of billions*. A $1.4B name and a $1.6B name both read as
``1`` and ``2`` respectively, so the 1–10B slice in that era is quantized to
about seven distinguishable buckets. Any small-cap slice study that reaches back
before 2017 has to account for that, which it can only do if the flag is in the
data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.data.normalize.common import MCAP_QUANTIZED_BEFORE, mcap_era

__all__ = ["normalize_from_daily"]


def normalize_from_daily(daily: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Derive ``securities`` from an already-normalized ``daily_market`` frame.

    Built from Tier 2 rather than re-parsing Tier 1, so the market caps in
    ``securities`` and ``daily_market`` cannot disagree: there is one conversion,
    applied once, in :mod:`engine.data.normalize.n_daily`.
    """
    if daily is None or daily.empty:
        return pd.DataFrame(), {"rows": 0}

    frame = daily[["ticker", "date", "year", "mcap_usd"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])

    span = frame.groupby("ticker")["date"].agg(first_date="min", last_date="max")

    # Year-end size = the last observation with a market cap in that year, so a
    # ticker that stopped reporting mid-year keeps its last real figure rather
    # than a null.
    sized = frame.dropna(subset=["mcap_usd"]).sort_values(["ticker", "year", "date"])
    last = sized.groupby(["ticker", "year"]).last().reset_index()

    counts = (
        frame.groupby(["ticker", "year"]).size().rename("n_obs").reset_index()
    )

    out = counts.merge(
        last[["ticker", "year", "date", "mcap_usd"]], on=["ticker", "year"], how="left"
    )
    out = out.merge(span, on="ticker", how="left")

    with np.errstate(divide="ignore", invalid="ignore"):
        out["mcap_log"] = np.where(out["mcap_usd"] > 0, np.log(out["mcap_usd"]), np.nan)

    ref_date = out["date"].fillna(pd.Timestamp("2000-01-01"))
    out["mcap_unit_era"] = mcap_era(ref_date)
    out["mcap_quantized"] = ref_date < MCAP_QUANTIZED_BEFORE
    # Reconstruct the delivered figure so the conversion stays auditable from
    # Tier 2 alone, without re-reading a 400 MB raw tree.
    era_mult = {"billions": 1e9, "millions": 1e6, "thousands": 1e3}
    out["mcap_raw"] = out["mcap_usd"] / out["mcap_unit_era"].map(era_mult)
    out["src"] = "orats.cores"

    out = out.drop(columns=["date"]).sort_values(["ticker", "year"]).reset_index(drop=True)

    report = {
        "rows": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
        "with_mcap": int(out["mcap_usd"].notna().sum()),
        "quantized_rows": int(out["mcap_quantized"].sum()),
        "era_counts": {
            str(k): int(v) for k, v in out["mcap_unit_era"].value_counts().items()
        },
    }
    return out, report
