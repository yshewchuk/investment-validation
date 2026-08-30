"""Normalize cached Polygon daily-aggs payloads into Tier-2 ``option_daily``.

These rows are the program's real-traded evidence. ``option_chains`` holds EOD
*quotes* (ORATS bid/ask); this table holds what actually *traded* — close,
VWAP, volume and the number of fills — for the same contracts on the same
days, wherever Polygon's entitlement window (2024-08-19 onward on this plan)
overlaps the chains. The gap between an ORATS mid and a Polygon close/VWAP is
the empirical fill-quality number the whole mid-fill assumption rests on, and
before this table existed there was no store it could live in.

The payloads arrive through the Tier-1 fetch wrapper, one ``/v2/aggs`` call
per contract covering its whole life. The contract identity is therefore in
the request (the endpoint string), not the response — the body only carries
the bars — so the contract is parsed back out of the endpoint and every bar
inherits it. A response the endpoint cannot be read from is flagged, not
guessed at.

Entitlement note (probed 2026-08-30): this plan's Polygon options cover daily
aggregates, reference and live snapshots only. ``/v3/trades``, ``/v3/quotes``
and intraday bars all return NOT_AUTHORIZED, so this is the finest-grained
real-trade data the store can carry — no bid/ask, no timestamps inside the day.
"""
from __future__ import annotations

import re
from typing import Iterator

import numpy as np
import pandas as pd

from engine.data.fetch import CachedEntry, iter_cached

__all__ = [
    "AGGS_ENDPOINT_PREFIX",
    "AGGS_ENDPOINT_SUFFIX",
    "CONTRACT_RE",
    "parse_contract",
    "contract_from_endpoint",
    "bars_to_frame",
    "normalize_entry",
    "iter_aggs_sources",
]

#: The fetch-store endpoint shape this normalizer reads. One call per contract:
#: ``v2/aggs/ticker/{contract}/range/1/day`` with from/to as params.
AGGS_ENDPOINT_PREFIX = "v2/aggs/ticker/"
AGGS_ENDPOINT_SUFFIX = "/range/1/day"

#: ``O:TSLA240906C00210000`` → TSLA, 2024-09-06, C, 210.0. Symbols may carry
#: dots and the odd punctuation a listing change produces (BRK.B, BF.B, …).
CONTRACT_RE = re.compile(
    r"^O:(?P<symbol>[A-Z0-9.&\-]+)(?P<yymmdd>\d{6})(?P<right>[CP])(?P<strike>\d{8})$"
)


def parse_contract(contract_ticker: str) -> dict | None:
    """Split an OCC id into (ticker, expiry, right, strike); ``None`` if malformed."""
    match = CONTRACT_RE.match(contract_ticker or "")
    if not match:
        return None
    yymmdd = match.group("yymmdd")
    try:
        expiry = pd.Timestamp(
            2000 + int(yymmdd[0:2]), int(yymmdd[2:4]), int(yymmdd[4:6])
        )
    except ValueError:
        return None
    return {
        "ticker": match.group("symbol"),
        "expiry": expiry,
        "right": match.group("right"),
        "strike": int(match.group("strike")) / 1000.0,
    }


def contract_from_endpoint(endpoint: str) -> str | None:
    """Recover the contract ticker from an aggs endpoint, if it is one.

    Tolerates both range encodings: the bare ``.../range/1/day`` used in
    planning and the ``.../range/1/day/{from}/{to}`` form the API actually
    requires (the date range travels as path segments, not query params).
    """
    if not endpoint.startswith(AGGS_ENDPOINT_PREFIX):
        return None
    rest = endpoint[len(AGGS_ENDPOINT_PREFIX):]
    marker = "/range/1/day"
    idx = rest.find(marker)
    if idx < 0:
        return None
    contract = rest[:idx]
    return contract or None


def bars_to_frame(
    contract_ticker: str, results: list[dict], *, source_id: str
) -> tuple[pd.DataFrame, dict]:
    """Turn one contract's daily aggs into Tier-2 rows.

    The single row parser. Bar timestamps are taken as UTC dates: Polygon
    stamps daily option bars within the trade date in UTC, and the legacy
    ``pull_chains.py`` already validated that convention against known entry
    and exit dates on this same data.
    """
    parsed = parse_contract(contract_ticker)
    if parsed is None:
        return pd.DataFrame(), {
            "file": source_id,
            "reason": f"unparseable contract ticker {contract_ticker!r}",
        }
    if not results:
        return pd.DataFrame(), {
            "file": source_id,
            "contract": contract_ticker,
            "rows_in": 0,
            "rows_out": 0,
            "reason": "empty results",
        }

    src = pd.DataFrame([r for r in results if isinstance(r, dict)])
    if src.empty or "t" not in src.columns:
        return pd.DataFrame(), {"file": source_id, "reason": "no bar rows with timestamps"}

    obs_date = pd.to_datetime(src["t"], unit="ms", utc=True, errors="coerce")
    obs_date = obs_date.dt.tz_localize(None).dt.normalize()

    def numeric(key: str) -> np.ndarray:
        if key not in src.columns:
            return np.full(len(src), np.nan)
        return pd.to_numeric(src[key], errors="coerce").to_numpy(dtype=float)

    out = pd.DataFrame(
        {
            "contract_ticker": contract_ticker,
            "ticker": parsed["ticker"],
            "obs_date": obs_date,
            "expiry": parsed["expiry"],
            "strike": parsed["strike"],
            "right": parsed["right"],
            "open": numeric("o"),
            "high": numeric("h"),
            "low": numeric("l"),
            "close": numeric("c"),
            # Polygon spells volume `v` and vwap `vw`; the mapping lives here
            # and nowhere else.
            "vwap": numeric("vw"),
            "volume": numeric("v"),
        }
    )
    n_trades = (
        pd.to_numeric(src["n"], errors="coerce")
        if "n" in src.columns
        else pd.Series(np.nan, index=src.index)
    )
    out["n_trades"] = n_trades.astype("Int64")

    bad = (
        out["obs_date"].isna()
        | out["close"].isna()
        | (out["close"] <= 0)
        | (out["high"] < out["low"])
    )
    excluded = int(bad.sum())
    out = out.loc[~bad].copy()
    out["year"] = out["obs_date"].dt.year
    out["src"] = "polygon.v2.aggs"
    out["src_file"] = source_id

    report = {
        "file": source_id,
        "contract": contract_ticker,
        "rows_in": int(len(src)),
        "rows_out": int(len(out)),
        "excluded": excluded,
    }
    return out, report


def normalize_entry(entry: CachedEntry) -> tuple[pd.DataFrame, dict]:
    """Parse one cached aggs payload. Tolerant: an odd body yields zero rows
    plus a reason, never an exception — one bad payload must not abort a
    rebuild of the whole store."""
    contract = contract_from_endpoint(entry.endpoint)
    if contract is None:
        return pd.DataFrame(), {
            "file": entry.source_id,
            "reason": f"endpoint is not an aggs call: {entry.endpoint!r}",
        }
    try:
        payload = entry.json()
    except (OSError, EOFError, ValueError) as exc:
        return pd.DataFrame(), {
            "file": entry.source_id,
            "reason": f"unreadable body: {type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict) or not (
        "results" in payload or "resultsCount" in payload or "status" in payload
    ):
        return pd.DataFrame(), {
            "file": entry.source_id,
            "reason": f"unrecognized aggs envelope: {type(payload).__name__}",
        }
    # A contract that never traded gets `resultsCount: 0` with NO `results` key
    # — a legitimate zero-row answer, not a malformed payload.
    results = payload.get("results") or []
    return bars_to_frame(contract, results, source_id=entry.source_id)


def iter_aggs_sources(
    *, root=None, stats: dict | None = None
) -> Iterator[CachedEntry]:
    """Every cached Polygon daily-aggs payload, deterministically ordered.

    ``stats`` receives the accounting that keeps a zero-row rebuild visible:
    how many entries were scanned and how many matched the aggs shape. Other
    Polygon payloads (reference, snapshots) are counted but skipped — they
    belong to no Tier-2 table.
    """
    if stats is not None:
        stats.setdefault("scanned", 0)
        stats.setdefault("aggs", 0)
        stats.setdefault("other", 0)
    for entry in iter_cached("polygon", root=root):
        if stats is not None:
            stats["scanned"] += 1
        if contract_from_endpoint(entry.endpoint) is None:
            if stats is not None:
                stats["other"] += 1
            continue
        if stats is not None:
            stats["aggs"] += 1
        yield entry
