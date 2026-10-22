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
the series' start, move its end backwards, or drop a stored date — the whole
point is catching up, so a smaller, older or hole-ridden answer is refused by
reason instead of written. An identical rewrite is skipped entirely:
``panel._gspc_series`` caches on file mtime, so touching the file with the
same bytes would invalidate it for nothing. Every write goes through
``engine.paths.assert_writable`` (``GSPC_DAILY`` is its one
``WRITABLE_EXCEPTIONS`` carve-out, #1103); a ``PermissionError`` there is
refused as reason ``unwritable``, never raised — as is any ``OSError`` from
creating the ``.lock`` sibling, staging the temp file or replacing.
Concurrent refreshes serialize through an exclusive ``fcntl.flock`` on that
``.lock`` sibling, held from before the existing-file read to after the
replace.

``live=True`` plus a caller-supplied ``refetch_token`` follow
``price_refresh.fetch_one``: the cache key for ``{"ticker": "^GSPC",
"period": "max"}`` never varies on its own, so without them the first body
ever cached would answer forever.
"""
from __future__ import annotations

import fcntl
import io
import os
import stat
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths

__all__ = ["refresh_gspc_daily", "HEADER_LINES", "MAX_EXISTING_BYTES"]

#: Rows 0-2 of the legacy layout; readers (``panel._gspc_series``,
#: ``calendar.trading_calendar``) skip exactly these three lines.
HEADER_LINES = (
    "Price,Adj Close,Close,High,Low,Open,Volume\n",
    "Ticker,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC,^GSPC\n",
    "Date,,,,,,\n",
)

#: Ceiling on the existing CSV we read to guard the rewrite. The stored series
#: is a twenty-year daily file (well under 1 MiB); anything larger is not the
#: thing this pull is meant to refresh, so it is refused rather than slurped
#: into memory or overwritten.
MAX_EXISTING_BYTES = 64 * 1024 * 1024


def _log(message: str) -> None:
    print(f"[gspc] {message}", file=sys.stderr, flush=True)


def _parse_body(body) -> pd.DataFrame | None:
    """The fetched body as ``date``/``adj``/``close``/``high``/``low``/
    ``open``/``volume``, or ``None`` when it is empty or unparseable.

    Same reading contract as ``panel._yf_history_from_tier1``: first column
    is the date, prices come from the named columns. When the body carries an
    ``Adj Close`` column the legacy ``adj`` slot gets that true adjusted
    series; otherwise it is written as ``close`` — the split-adjusted series
    Close already is. Rows with an unparseable date, a NaN close or a
    non-finite close are dropped; non-finite other prices and volume become
    NaN before rendering (so ``astype("int64")`` on volume can never raise;
    NaN/inf volume lands as ``0``), and duplicate dates keep the last
    occurrence.
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
    adj = (pd.to_numeric(frame["Adj Close"], errors="coerce")
           if "Adj Close" in frame.columns else close)
    out = pd.DataFrame({"date": dates, "adj": adj, "close": close})
    for name, column in (("high", "High"), ("low", "Low"), ("open", "Open"),
                         ("volume", "Volume")):
        out[name] = (pd.to_numeric(frame[column], errors="coerce")
                     if column in frame.columns else float("nan"))
    for name in ("adj", "high", "low", "open", "volume"):
        out[name] = out[name].where(np.isfinite(out[name]))
    out = out.dropna(subset=["date"])
    out = out[np.isfinite(out["close"])]
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
    ``older_than_existing`` / ``unreadable`` / ``unwritable``.
    """
    from engine.data.sources.base import CredentialRotated, FetchError

    started = time.time()
    try:
        path = Path(path)
    except (TypeError, OSError, ValueError):
        return _finish({"ok": False, "reason": "unreadable"}, started)
    _log(f"refresh {path} starting")

    try:
        rec = fetcher.fetch(
            "yfinance", "history", {"ticker": "^GSPC", "period": "max"},
            live=True, note="gspc-refresh", refetch_token=refetch_token,
        )
    except CredentialRotated:
        raise
    except (FetchError, OSError, ValueError):
        return _finish({"ok": False, "reason": "fetch_failed"}, started)
    if rec is None or rec.status != 200:
        return _finish({"ok": False, "reason": "fetch_failed"}, started)

    frame = _parse_body(rec.body)
    if frame is None:
        return _finish({"ok": False, "reason": "unparseable"}, started)

    # The read-compare-replace section below is serialized against a
    # concurrent refresh of the same file through an exclusive flock on a
    # ``.lock`` sibling (both the lock and the ``.tmp`` staging file are
    # ``gspc_daily.csv.*`` names, the staging siblings ``paths`` already
    # allows); it is held from before the existing-file read to after the
    # replace, so two refreshes can never interleave a guard with a rewrite.
    lock_path = path.with_name(path.name + ".lock")
    try:
        paths.assert_writable(lock_path)
        if not path.parent.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return _finish({"ok": False, "reason": "unwritable"}, started)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        existing_bytes = None
        try:
            fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            fd = None  # first write: nothing to guard against
        except (OSError, ValueError):
            return _finish({"ok": False, "reason": "unreadable"}, started)
        if fd is not None:
            try:
                st = os.fstat(fd)
                if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_EXISTING_BYTES:
                    return _finish({"ok": False, "reason": "unreadable"}, started)
                # Read at most MAX_EXISTING_BYTES + 1 so a file that grew between
                # fstat and the read (or lied about its size) is refused, not
                # slurped whole into memory.
                chunks = []
                remaining = MAX_EXISTING_BYTES + 1
                while remaining > 0:
                    block = os.read(fd, min(remaining, 1 << 20))
                    if not block:
                        break
                    chunks.append(block)
                    remaining -= len(block)
                if remaining <= 0:
                    return _finish({"ok": False, "reason": "unreadable"}, started)
                existing_bytes = b"".join(chunks)
                existing = pd.read_csv(io.BytesIO(existing_bytes), skiprows=3, header=None)
            except (OSError, ValueError):
                return _finish({"ok": False, "reason": "unreadable"}, started)
            finally:
                os.close(fd)
            if len(existing):
                if len(frame) < len(existing):
                    return _finish({"ok": False, "reason": "shrunk"}, started)
                existing_first = pd.to_datetime(existing.iloc[0, 0], errors="coerce")
                existing_last = pd.to_datetime(existing.iloc[-1, 0], errors="coerce")
                if pd.notna(existing_first) and frame["date"].iloc[0] > existing_first:
                    return _finish({"ok": False, "reason": "shrunk"}, started)
                if pd.notna(existing_last) and frame["date"].iloc[-1] < existing_last:
                    return _finish({"ok": False, "reason": "older_than_existing"}, started)
                # Every stored date must survive into the new series, or the rewrite
                # would punch a hole the panel reads as a gap in the market history.
                stored = set(map(str, existing.iloc[:, 0]))
                if not stored <= set(frame["date"].dt.strftime("%Y-%m-%d")):
                    return _finish({"ok": False, "reason": "shrunk"}, started)

        data = _render(frame)
        if existing_bytes != data:
            tmp = path.with_name(
                f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            for guard in (path, tmp):
                try:
                    paths.assert_writable(guard)
                except PermissionError:
                    return _finish({"ok": False, "reason": "unwritable"}, started)
            try:
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(data)
                os.replace(tmp, path)
            except OSError:
                return _finish({"ok": False, "reason": "unwritable"}, started)
            finally:
                # A failed replace leaves the stored bytes untouched; only the
                # temp sibling needs removing, and a cleanup failure there is
                # never allowed to raise out of a refusal.
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
    finally:
        os.close(lock_fd)
    return _finish(
        {"ok": True, "last_date": str(frame["date"].iloc[-1].date()),
         "rows": int(len(frame))},
        started,
    )
