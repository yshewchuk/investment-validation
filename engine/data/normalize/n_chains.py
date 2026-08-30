"""Normalize cached ORATS ``/hist/strikes`` files into Tier-2 ``option_chains``.

These are the only sanctioned P&L prices in the program: real EOD bid/ask,
validated against Polygon real trades (median ratio 1.006) and live yfinance
quotes (±1.7–3.4%). ~19k files hold ~6.5M strike rows; each raw row carries a
call *and* a put at one strike, so Tier 2 holds roughly twice that.

File layout (four pull generations, all still on disk)::

    {date}_b{N}.json.gz        S2 entry-date and post-print exit-date chains
    {date}_t14_b{N}.json.gz    T-14 chains for the run-up work
    {date}_c2_b{N}.json.gz     the calendar-structure pull
    {date}.done*               resume markers, no payload

Document shape is ``{"entry_date", "tickers", "rows": [...]}``.

Entry and exit pulls share the plain ``_b{N}`` name, so the same (ticker, date,
expiry, strike) can appear in two files. Rows are deduplicated on the primary
key, keeping the first occurrence in sorted-filename order — which makes the
result a function of the file set, not of directory iteration order.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from engine import paths
from engine.data.normalize.common import mask_sentinels, read_gz_json

__all__ = [
    "CHAIN_FILE_RE",
    "parse_filename",
    "rows_to_frame",
    "normalize_file",
    "normalize_fetch_rows",
    "iter_chain_files",
    "iter_fetch_sources",
    "iter_normalized",
    "PRIMARY_KEY",
]

#: Tier-2 primary key for a chain row. Duplicates across source files are
#: resolved on this, keeping the first in sorted-source order.
PRIMARY_KEY = ("ticker", "obs_date", "expiry", "strike", "right")

#: ``2018-01-02_t14_b3.json.gz`` → date 2018-01-02, kind t14, batch 3.
CHAIN_FILE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<kind>t14|c2))?_b(?P<batch>\d+)\.json\.gz$"
)

#: Why the chain was pulled. ``eod`` covers the entry and exit pulls, which
#: share a filename pattern and cannot be told apart after the fact.
KIND_LABELS = {None: "eod", "t14": "t14", "c2": "c2"}

#: Chains that arrived through the Tier-1 fetch wrapper rather than one of the
#: pre-engine pulls. Kept distinct so coverage work can tell new data from old.
FETCH_CHAIN_KIND = "fetch"


def parse_filename(name: str) -> dict | None:
    match = CHAIN_FILE_RE.match(name)
    if not match:
        return None
    return {
        "date": match.group("date"),
        "chain_kind": KIND_LABELS[match.group("kind")],
        "batch": int(match.group("batch")),
    }


def iter_chain_files(root: Path | None = None) -> list[Path]:
    """Every payload file in the strikes cache, deterministically ordered."""
    root = root or paths.RAW_ORATS_STRIKES
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*.json.gz") if parse_filename(p.name))


def rows_to_frame(
    rows: list[dict], *, source_id: str, chain_kind: str
) -> tuple[pd.DataFrame, dict]:
    """Turn raw ORATS strike rows into Tier-2 rows (one per strike *and side*).

    The single row parser. Both pull generations — the legacy wrapped files and
    the fetch store's verbatim responses — come through here, so a fix to a
    price convention cannot apply to one and miss the other.
    """
    if not rows:
        return pd.DataFrame(), {"file": source_id, "rows_in": 0, "rows_out": 0}

    src = pd.DataFrame(rows)
    required = {"ticker", "tradeDate", "expirDate", "strike"}
    missing = required - set(src.columns)
    if missing:
        return pd.DataFrame(), {"file": source_id, "reason": f"missing {sorted(missing)}"}

    obs_date = pd.to_datetime(src["tradeDate"], errors="coerce")
    expiry = pd.to_datetime(src["expirDate"], errors="coerce")
    spot = mask_sentinels(src["spotPrice"]) if "spotPrice" in src else np.full(len(src), np.nan)
    if "stockPrice" in src.columns:
        stock = mask_sentinels(src["stockPrice"])
        spot = np.where(np.isfinite(spot), spot, stock)
    strike = mask_sentinels(src["strike"])
    # ORATS reports the CALL delta; the put delta at the same strike is delta-1.
    call_delta = mask_sentinels(src["delta"]) if "delta" in src else np.full(len(src), np.nan)

    def optional(key: str) -> np.ndarray:
        """A field the older cache never requested.

        NaN where absent, and deliberately not zero: "we never asked for size"
        and "there was no size" are different facts, and a zero would assert
        the second one for 19,061 files that only support the first.
        """
        if key not in src.columns:
            return np.full(len(src), np.nan)
        return mask_sentinels(src[key])

    frames = []
    for right, bid_key, ask_key, iv_key, vol_key, oi_key, bs_key, as_key in (
        ("C", "callBidPrice", "callAskPrice", "callMidIv",
         "callVolume", "callOpenInterest", "callBidSize", "callAskSize"),
        ("P", "putBidPrice", "putAskPrice", "putMidIv",
         "putVolume", "putOpenInterest", "putBidSize", "putAskSize"),
    ):
        if bid_key not in src.columns or ask_key not in src.columns:
            continue
        bid = mask_sentinels(src[bid_key])
        ask = mask_sentinels(src[ask_key])
        iv = mask_sentinels(src[iv_key]) if iv_key in src.columns else np.full(len(src), np.nan)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": src["ticker"].astype(str),
                    "obs_date": obs_date,
                    "expiry": expiry,
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "iv": iv,
                    "delta": call_delta if right == "C" else call_delta - 1.0,
                    "spot": spot,
                    # Liquidity — present from the 2026-09 pull onward.
                    "volume": optional(vol_key),
                    "open_interest": optional(oi_key),
                    "bid_size": optional(bs_key),
                    "ask_size": optional(as_key),
                }
            )
        )
    if not frames:
        return pd.DataFrame(), {"file": source_id, "reason": "no price columns"}

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["ticker", "obs_date", "expiry", "strike"])
    # DTE is recomputed rather than trusted: the cached `dte` field is what the
    # API said at pull time, and the schema contract is dte == expiry - obs_date.
    out["dte"] = (out["expiry"] - out["obs_date"]).dt.days
    out["year"] = out["obs_date"].dt.year
    out["src"] = "orats.hist.strikes"
    out["src_file"] = source_id
    out["chain_kind"] = chain_kind

    report = {
        "file": source_id,
        "chain_kind": chain_kind,
        "rows_in": int(len(src)),
        "rows_out": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
    }
    return out, report


def normalize_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse one legacy chain file (the wrapped `{entry_date, tickers, rows}` form)."""
    meta = parse_filename(path.name)
    if meta is None:
        return pd.DataFrame(), {"file": path.name, "reason": "unrecognized filename"}
    doc = read_gz_json(path)
    rows = (doc or {}).get("rows") or []
    frame, report = rows_to_frame(rows, source_id=path.name, chain_kind=meta["chain_kind"])
    report["date"] = meta["date"]
    return frame, report


def normalize_fetch_rows(source) -> tuple[pd.DataFrame, dict]:
    """Parse one fetch-store payload (the verbatim `{"data": [...]}` form)."""
    return rows_to_frame(
        source.rows, source_id=source.source_id, chain_kind=FETCH_CHAIN_KIND
    )


def iter_fetch_sources(stats: dict | None = None):
    """Every `hist/strikes` response the Tier-1 fetch wrapper has cached.

    ``stats``, when given, receives the payload accounting
    (:func:`~engine.data.normalize.fetch_store.iter_orats_rows`): how many
    bodies were scanned, how many were empty, and how many cost quota but
    parsed to nothing (unrecognized/unreadable) — the numbers that turn a
    silent zero-row rebuild into a visible one.
    """
    from engine.data.normalize.fetch_store import iter_orats_rows

    return list(iter_orats_rows("hist/strikes", stats=stats))


def iter_normalized(
    files: Iterable[Path] | None = None,
) -> Iterator[tuple[pd.DataFrame, dict]]:
    for path in list(files) if files is not None else iter_chain_files():
        yield normalize_file(path)
