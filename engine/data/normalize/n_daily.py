"""Normalize ORATS daily summaries + cores into Tier-2 ``daily_market``.

Source: ``earnings_predictions/data/raw/orats/{summaries,cores}/{TICKER}.json.gz``,
2,936 tickers back to 2007 — about 9.4M rows, so the build streams per ticker
and flushes per year rather than materializing the table.

Unit handling, all applied here and nowhere else:

* IV-like fields arrive as decimals (``0.407453``) and are stored as vol points
  (``40.7453``), matching the panel convention every existing model was fit on.
* ``mktCap`` goes through the three-era conversion in
  :func:`~engine.data.normalize.common.mcap_to_usd`.
* FLT_MAX sentinels are masked before anything else touches the numbers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from engine import paths
from engine.data.normalize.common import (
    clip_implausible,
    mask_sentinels,
    mcap_to_usd,
    read_gz_json,
)

__all__ = ["SUMMARY_FIELDS", "normalize_ticker", "iter_normalized", "list_tickers"]

#: ``raw ORATS key -> (tier-2 column, multiplier)``. The multiplier turns
#: decimals into vol points / percent, which is the panel's unit.
SUMMARY_FIELDS: dict[str, tuple[str, float]] = {
    "stockPrice": ("spot", 1.0),
    "iv10d": ("iv10", 100.0),
    "iv30d": ("iv30", 100.0),
    "exErnIv10d": ("exern_iv10", 100.0),
    "exErnIv30d": ("exern_iv30", 100.0),
    "impliedMove": ("implied_move", 100.0),
    "rVol30": ("rvol30", 100.0),
    "skewing": ("skew", 1.0),
    "contango": ("contango", 1.0),
    "fwd90_30": ("fwd90_30", 100.0),
    "fexErn90_30": ("fexern90_30", 100.0),
    "ieeEarnEffect": ("iee", 1.0),
}


def list_tickers(root: Path | None = None) -> list[str]:
    root = root or paths.RAW_ORATS_SUMMARIES
    if not root.exists():
        return []
    return sorted(p.name[: -len(".json.gz")] for p in root.glob("*.json.gz"))


def normalize_ticker(
    ticker: str,
    *,
    summaries_dir: Path | None = None,
    cores_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return one ticker's ``daily_market`` rows plus a small quality report."""
    summaries_dir = summaries_dir or paths.RAW_ORATS_SUMMARIES
    cores_dir = cores_dir or paths.RAW_ORATS_CORES

    spath = summaries_dir / f"{ticker}.json.gz"
    if not spath.exists():
        return pd.DataFrame(), {"ticker": ticker, "reason": "no summaries file"}
    rows = read_gz_json(spath) or []
    if not rows:
        return pd.DataFrame(), {"ticker": ticker, "reason": "empty summaries"}

    src = pd.DataFrame(rows)
    if "tradeDate" not in src.columns:
        return pd.DataFrame(), {"ticker": ticker, "reason": "no tradeDate column"}

    out = pd.DataFrame(
        {
            "ticker": ticker,
            "date": pd.to_datetime(src["tradeDate"], errors="coerce"),
        }
    )
    for raw_key, (col, mult) in SUMMARY_FIELDS.items():
        if raw_key in src.columns:
            out[col] = mask_sentinels(src[raw_key]) * mult
        else:
            out[col] = np.nan

    out = out.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    out["src_iv"] = "orats.summaries"

    # -- cores: market cap, three-era conversion --------------------------
    #
    # The two ORATS series do not share a date grid, and for some tickers they
    # barely overlap at all: ABX, RTX and META carry cores history back to 2007
    # while their summaries start in 2026 (reused or renamed symbols). Keying
    # the table on summaries alone would throw away nearly twenty years of
    # market cap for those names, so the row index is the UNION of both series
    # and each block keeps its own source column.
    cpath = cores_dir / f"{ticker}.json.gz"
    mcap_rows = 0
    cores_only_rows = 0
    core = None
    if cpath.exists():
        crows = read_gz_json(cpath) or []
        if crows:
            candidate = pd.DataFrame(crows)
            if {"tradeDate", "mktCap"} <= set(candidate.columns):
                candidate = candidate.assign(
                    date=pd.to_datetime(candidate["tradeDate"], errors="coerce")
                )
                candidate = candidate.dropna(subset=["date"]).drop_duplicates("date", keep="last")
                candidate["mcap_usd"] = mcap_to_usd(candidate["mktCap"], candidate["date"])
                core = candidate.dropna(subset=["mcap_usd"]).sort_values("date")

    if core is not None and len(core):
        extra_dates = core.loc[~core["date"].isin(set(out["date"])), ["date"]]
        cores_only_rows = int(len(extra_dates))
        if cores_only_rows:
            out = pd.concat([out, extra_dates.assign(ticker=ticker)], ignore_index=True)
        out = out.sort_values("date").reset_index(drop=True)
        # As-of (backward) rather than exact: every row carries the most recent
        # market cap known on that trade date. Strictly backward looking, so it
        # cannot leak.
        out = pd.merge_asof(out, core[["date", "mcap_usd"]], on="date", direction="backward")
        mcap_rows = int(out["mcap_usd"].notna().sum())
    if "mcap_usd" not in out.columns:
        out["mcap_usd"] = np.nan
    out = out.sort_values("date").reset_index(drop=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        out["mcap_log"] = np.where(out["mcap_usd"] > 0, np.log(out["mcap_usd"]), np.nan)

    # oquants implied moves before 2022 are reconstructions; the ORATS
    # impliedMove used here is a quoted figure for its whole history, so the
    # flag is False rather than absent — the column means "this number is a
    # reconstruction", and for this source it never is.
    out["implied_reconstructed"] = False
    out["year"] = out["date"].dt.year
    out["src_spot"] = out["src_iv"]
    out["src_mcap"] = np.where(out["mcap_usd"].notna(), "orats.cores", None)

    out, clipped = clip_implausible(out)
    report = {
        "ticker": ticker,
        "rows": int(len(out)),
        "mcap_rows": mcap_rows,
        "cores_only_rows": cores_only_rows,
        "clipped": clipped,
        "first": str(out["date"].min().date()) if len(out) else None,
        "last": str(out["date"].max().date()) if len(out) else None,
    }
    return out, report


def iter_normalized(
    tickers: Iterable[str] | None = None, **kwargs
) -> Iterator[tuple[pd.DataFrame, dict]]:
    """Stream ``(frame, report)`` per ticker, in sorted order (deterministic)."""
    for ticker in sorted(tickers) if tickers is not None else list_tickers():
        yield normalize_ticker(ticker, **kwargs)
