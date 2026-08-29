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
    "normalize_file",
    "iter_chain_files",
    "iter_normalized",
]

#: ``2018-01-02_t14_b3.json.gz`` → date 2018-01-02, kind t14, batch 3.
CHAIN_FILE_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})(?:_(?P<kind>t14|c2))?_b(?P<batch>\d+)\.json\.gz$"
)

#: Why the chain was pulled. ``eod`` covers the entry and exit pulls, which
#: share a filename pattern and cannot be told apart after the fact.
KIND_LABELS = {None: "eod", "t14": "t14", "c2": "c2"}


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


def normalize_file(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse one chain file into Tier-2 rows (one per strike *and side*)."""
    meta = parse_filename(path.name)
    if meta is None:
        return pd.DataFrame(), {"file": path.name, "reason": "unrecognized filename"}

    doc = read_gz_json(path)
    rows = (doc or {}).get("rows") or []
    if not rows:
        return pd.DataFrame(), {"file": path.name, "rows_in": 0, "rows_out": 0}

    src = pd.DataFrame(rows)
    required = {"ticker", "tradeDate", "expirDate", "strike"}
    missing = required - set(src.columns)
    if missing:
        return pd.DataFrame(), {"file": path.name, "reason": f"missing {sorted(missing)}"}

    obs_date = pd.to_datetime(src["tradeDate"], errors="coerce")
    expiry = pd.to_datetime(src["expirDate"], errors="coerce")
    spot = mask_sentinels(src["spotPrice"]) if "spotPrice" in src else np.full(len(src), np.nan)
    if "stockPrice" in src.columns:
        stock = mask_sentinels(src["stockPrice"])
        spot = np.where(np.isfinite(spot), spot, stock)
    strike = mask_sentinels(src["strike"])
    # ORATS reports the CALL delta; the put delta at the same strike is delta-1.
    call_delta = mask_sentinels(src["delta"]) if "delta" in src else np.full(len(src), np.nan)

    frames = []
    for right, bid_key, ask_key, iv_key in (
        ("C", "callBidPrice", "callAskPrice", "callMidIv"),
        ("P", "putBidPrice", "putAskPrice", "putMidIv"),
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
                }
            )
        )
    if not frames:
        return pd.DataFrame(), {"file": path.name, "reason": "no price columns"}

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["ticker", "obs_date", "expiry", "strike"])
    # DTE is recomputed rather than trusted: the cached `dte` field is what the
    # API said at pull time, and the schema contract is dte == expiry - obs_date.
    out["dte"] = (out["expiry"] - out["obs_date"]).dt.days
    out["year"] = out["obs_date"].dt.year
    out["src"] = "orats.hist.strikes"
    out["src_file"] = path.name
    out["chain_kind"] = meta["chain_kind"]

    report = {
        "file": path.name,
        "date": meta["date"],
        "chain_kind": meta["chain_kind"],
        "rows_in": int(len(src)),
        "rows_out": int(len(out)),
        "tickers": int(out["ticker"].nunique()),
    }
    return out, report


def iter_normalized(
    files: Iterable[Path] | None = None,
) -> Iterator[tuple[pd.DataFrame, dict]]:
    for path in list(files) if files is not None else iter_chain_files():
        yield normalize_file(path)
