#!/usr/bin/env python3
"""Refresh the legacy ``^GSPC`` daily CSV the regime features read.

``engine.paths.GSPC_DAILY`` was fetched once from yfinance on 2026-08-27 and
nothing refreshes it, so ``add_regime_features`` keeps anchoring every scored
event on the same frozen series. This pull re-fetches the same
``period="max"`` history through the Tier-1 fetcher and rewrites the file in
the layout both readers already parse: three multi-header lines, then
positional ``date,adj,close,high,low,open,volume`` rows.

Guards, because this file feeds a champion-model feature: a fetch or parse
failure leaves the stored bytes untouched, and a fresh body may never shrink
the series' start or move its end backwards — the whole point is catching up,
so a smaller or older answer is refused by reason instead of written. An
identical rewrite is skipped entirely: ``panel._gspc_series`` caches on file
mtime, so touching the file with the same bytes would invalidate it for
nothing.

``live=True`` plus a caller-supplied ``refetch_token`` follow
``price_refresh.fetch_one``: the cache key for ``{"ticker": "^GSPC",
"period": "max"}`` never varies on its own, so without them the first body
ever cached would answer forever.
"""
from __future__ import annotations

import io
import os
import sys
import time
from pathlib import Path

import pandas as pd

__all__ = ["refresh_gspc_daily", "HEADER_LINES"]

#: Rows 0-2 of the legacy layout; readers (``panel._gspc_series``,
#: ``calendar.trading_calendar``) skip exactly these three lines.
HEADER_LINES = (
    "Price,Adj Close,Close,High,Low,Open,Volume\n",
    "Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC\n",
    "Date,,,,,,\n",
)


def _log(message: str) -> None:
    print(f"[gspc] {message}", file=sys.stderr, flush=True)


def _parse_body(body) -> pd.DataFrame | None:
    """The fetched body as ``date``/``adj``/``close``/``high``/``low``/
    ``open``/``volume``, or ``None`` when it is empty or unparseable.

    Same reading contract as ``panel._yf_history_from_tier1``: first column
    is the date, prices come from the named columns. The body carries no
    Adj Close, so the legacy ``adj`` slot is written as ``close`` — the
    split-adjusted series Close already is. Rows with an unparseable date or
    a NaN close are dropped, and duplicate dates keep the last occurrence.
    """
    try:
        frame = pd.read_csv(io.BytesIO(body if isinstance(body, bytes) else str(body).encode()))
    except (ValueError, OSError, EOFError, TypeError):
        return None
    if frame.empty or "Close" not in frame.columns:
        return None
    dates = pd.to_datetime(frame[frame.columns[0]], errors="coerce", utc=True)
    dates = dates.dt.tz_localize(None).dt.normalize()
    close = pd.to_numeric(frame["Close"], errors="coerce")
    out = pd.DataFrame({"date": dates, "adj": close, "close": close})
    for name, column in (("high", "High"), ("low", "Low"), ("open", "Open"),
                         ("volume", "Volume")):
        out[name] = (pd.to_numeric(frame[column], errors="coerce")
                     if column in frame.columns else float("nan"))
    out = out.dropna(subset=["date", "close"])
    if out.empty:
        return None
    out = out.sort_values("date").drop_duplicates(subset="date", keep="last")
    return out.reset_index(drop=True)


def _render(frame: pd.DataFrame) -> bytes:
    """The whole legacy file — three header lines, then the rows."""
    out = frame.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["volume"] = out["volume"].fillna(0).astype("int64")
    buf = io.StringIO()
    buf.writelines(HEADER_LINES)
    out[["date", "adj", "close", "high", "low", "open", "volume"]].to_csv(
        buf, header=False, index=False)
    return buf.getvalue().encode("utf-8")


def _finish(result: dict, started: float) -> dict:
    outcome = "ok" if result["ok"] else f"refused: {result['reason']}"
    _log(f"{outcome} in {time.time() - started:.1f}s")
    return result


def refresh_gspc_daily(fetcher, path, *, refetch_token: str | None = None) -> dict:
    """Fetch ^GSPC max history and refresh ``path`` in the legacy layout.

    Never raises for an ordinary per-source failure (the ``price_refresh.
    fetch_one`` lesson; ``CredentialRotated`` propagates because it is a
    whole-run condition). Returns ``{"ok": True, "last_date":
    "YYYY-MM-DD", "rows": N}`` or ``{"ok": False, "reason": ...}`` with the
    reason in ``fetch_failed`` / ``unparseable`` / ``shrunk`` /
    ``older_than_existing``.
    """
    from engine.data.sources.base import FetchError

    started = time.time()
    path = Path(path)
    _log(f"refresh {path} starting")

    try:
        rec = fetcher.fetch(
            "yfinance", "history", {"ticker": "^GSPC", "period": "max"},
            live=True, note="gspc-refresh", refetch_token=refetch_token,
        )
    except (FetchError, OSError, ValueError):
        return _finish({"ok": False, "reason": "fetch_failed"}, started)
    if rec is None or rec.status != 200:
        return _finish({"ok": False, "reason": "fetch_failed"}, started)

    frame = _parse_body(rec.body)
    if frame is None:
        return _finish({"ok": False, "reason": "unparseable"}, started)

    if path.exists():
        try:
            existing = pd.read_csv(path, skiprows=3, header=None)
        except (ValueError, OSError):
            existing = pd.DataFrame()
        if len(existing):
            if len(frame) < len(existing):
                return _finish({"ok": False, "reason": "shrunk"}, started)
            existing_first = pd.to_datetime(existing.iloc[0, 0], errors="coerce")
            existing_last = pd.to_datetime(existing.iloc[-1, 0], errors="coerce")
            if pd.notna(existing_first) and frame["date"].iloc[0] > existing_first:
                return _finish({"ok": False, "reason": "shrunk"}, started)
            if pd.notna(existing_last) and frame["date"].iloc[-1] < existing_last:
                return _finish({"ok": False, "reason": "older_than_existing"}, started)

    data = _render(frame)
    if not (path.exists() and path.read_bytes() == data):
        tmp = path.with_name(path.name + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return _finish(
        {"ok": True, "last_date": str(frame["date"].iloc[-1].date()),
         "rows": int(len(frame))},
        started,
    )
