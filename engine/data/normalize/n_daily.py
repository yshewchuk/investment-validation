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

from functools import lru_cache
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

__all__ = [
    "SUMMARY_FIELDS",
    "normalize_ticker",
    "iter_normalized",
    "list_tickers",
    "fetch_daily_index",
]

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


def list_tickers(root: Path | None = None, *, include_fetch: bool = True) -> list[str]:
    """Every ticker with daily data, from the legacy trees and the fetch store."""
    root = root or paths.RAW_ORATS_SUMMARIES
    tickers = set()
    if root.exists():
        tickers |= {p.name[: -len(".json.gz")] for p in root.glob("*.json.gz")}
    if include_fetch:
        tickers |= set(fetch_daily_index())
    return sorted(tickers)


@lru_cache(maxsize=1)
def fetch_daily_index() -> dict[str, dict[str, list[dict]]]:
    """Index the fetch store's MARKET-WIDE summaries/cores rows by ticker.

    On a fresh machine this is the *only* source of daily data: the legacy
    per-ticker trees do not exist, and the restore path re-pulls through the
    fetch wrapper. Without this bridge a restored machine rebuilds empty tables,
    which is the failure the recovery drill could not see (it checks imports and
    the no-data test subset, not a full rebuild).

    **Per-ticker responses are deliberately NOT indexed here.** ORATS serves a
    ticker's entire history — ~4,900 rows — from one ``?ticker=X`` call, and
    once the nightly began backfilling those, indexing every row of every
    response meant holding ~1.8M dicts in memory: the rebuild reached 5 GB RSS
    in 28 seconds and was killed. Those responses do not need an index at all,
    because they are addressable by cache key: :func:`_ticker_history_rows`
    loads exactly the one being asked for. The index is left holding only the
    market-wide (``tradeDate``) responses, which are a few thousand rows each.

    Cached because a rebuild asks for it once per ticker.
    """
    from engine.data.normalize.fetch_store import iter_orats_rows

    index: dict[str, dict[str, list[dict]]] = {}
    for endpoint, kind in (("hist/summaries", "summaries"), ("hist/cores", "cores")):
        for source in iter_orats_rows(endpoint):
            # A response fetched for ONE ticker is fetched back on demand.
            if (source.params or {}).get("ticker"):
                continue
            for row in source.rows:
                ticker = row.get("ticker")
                if not ticker:
                    continue
                index.setdefault(str(ticker), {"summaries": [], "cores": []})[kind].append(row)
    return index


def _ticker_history_rows(ticker: str, kind: str) -> list[dict]:
    """The cached ``?ticker=X`` response for one ticker, or nothing.

    A direct key lookup rather than a scan: the Tier-1 key is a hash of
    (source, endpoint, params), so the entry for this exact request is found
    without touching any other. This is what keeps a rebuild's memory flat as
    per-ticker history accumulates.
    """
    import gzip
    import json as _json

    from engine.data.fetch import Fetcher, cache_key

    endpoint = {"summaries": "hist/summaries", "cores": "hist/cores"}[kind]
    key = cache_key("orats", endpoint, {"ticker": str(ticker)})
    path = Fetcher().body_path("orats", key)
    if not path.exists():
        return []
    try:
        with gzip.open(path, "rb") as fh:
            payload = _json.loads(fh.read())
    except (OSError, EOFError, ValueError):
        return []
    rows = payload.get("data") if isinstance(payload, dict) else payload
    return list(rows or [])


def _rows_for(ticker: str, kind: str, directory: Path) -> list[dict]:
    """Legacy per-ticker file plus anything the fetch store holds, unioned.

    Both generations are kept: the legacy tree is the historical bulk and the
    fetch store is everything pulled since, and a daily refresh must extend the
    series rather than replace it. Duplicate trade dates are collapsed by the
    caller (last wins), so an overlapping re-pull is a no-op rather than a
    conflict.
    """
    rows: list[dict] = []
    path = directory / f"{ticker}.json.gz"
    if path.exists():
        rows.extend(read_gz_json(path) or [])
    rows.extend(fetch_daily_index().get(ticker, {}).get(kind, []))
    # The per-ticker history pull, loaded by key rather than through the index.
    rows.extend(_ticker_history_rows(ticker, kind))
    return rows


def normalize_ticker(
    ticker: str,
    *,
    summaries_dir: Path | None = None,
    cores_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return one ticker's ``daily_market`` rows plus a small quality report."""
    summaries_dir = summaries_dir or paths.RAW_ORATS_SUMMARIES
    cores_dir = cores_dir or paths.RAW_ORATS_CORES

    rows = _rows_for(ticker, "summaries", summaries_dir)
    if not rows:
        return pd.DataFrame(), {"ticker": ticker, "reason": "no summaries rows"}

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
    mcap_rows = 0
    cores_only_rows = 0
    core = None
    crows = _rows_for(ticker, "cores", cores_dir)
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
        out = pd.merge_asof(
            out,
            core[["date", "mcap_usd"]].rename(columns={"date": "mcap_asof"}),
            left_on="date",
            right_on="mcap_asof",
            direction="backward",
        )
        # How old the carried figure is. An unbounded backward join will happily
        # carry a delisted name's last cap forward for years, and nothing
        # downstream could tell that from a fresh observation. Recording the
        # age keeps the staleness visible to the analog buckets that consume it
        # rather than silently baked into a size feature.
        out["mcap_age_days"] = (out["date"] - out["mcap_asof"]).dt.days
        out.loc[out["mcap_usd"].isna(), "mcap_age_days"] = np.nan
        mcap_rows = int(out["mcap_usd"].notna().sum())
    for column in ("mcap_usd", "mcap_asof", "mcap_age_days"):
        if column not in out.columns:
            out[column] = pd.NaT if column == "mcap_asof" else np.nan
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
