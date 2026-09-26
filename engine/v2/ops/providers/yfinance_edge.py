"""Native yfinance edges (spec s4c Rewrite 2).

``yfinance`` is a library call, not an HTTP GET, so its injectable seam is a
``history_fn``/``earnings_fn`` callable returning a pandas ``DataFrame``, never
``http_get``. The factories here wrap those callables into the byte-returning
fetchers the two native stores consume (the stores keep the DST/normalize and
session-parsing work that was already there); the default callables port
``engine/data/sources/yf.py``'s bodies verbatim, including the US/Eastern
conversion the BMO/AMC derivation depends on.

Classification (spec R1) happens HERE: a real frame is ``complete``, an
empty/absent frame is ``legitimate_empty``, and a network/library exception
(``OSError``/``TimeoutError``/``ValueError`` -- pandas parsing inside the
default callables raises the last) is ``transient``, never swallowed into an
empty history (R3). A programming error (``TypeError``, ``AttributeError``, an
``OpsError``) propagates. The returned tuple is the ORATS-shaped
``(raw_bytes, response_kind, response_meta, rows)``; the store parses the CSV
and re-validates it before caching (R2).

Nothing here imports ``yfinance`` at module import time -- the default
callables import it lazily, so importing this module (and constructing the
callback) touches no network and no heavy dependency.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

__all__ = ["EARNINGS_LIMIT", "yfinance_earnings_fetcher", "yfinance_history_fetcher"]

#: Announcement rows requested per ticker: the next print plus history.
EARNINGS_LIMIT = 12

#: ORATS-style pre-market cutoff: anything before noon is BMO.
BMO_CUTOFF = 1200


def yfinance_history_fetcher(*, history_fn: Callable[[str], pd.DataFrame] | None = None) \
        -> Callable[[str], tuple]:
    """Return the ``fetcher(ticker) -> (csv_bytes, kind, meta, rows)`` seam."""

    def fetcher(ticker: str):
        try:
            frame = (history_fn or _default_history)(str(ticker))
        except (OSError, TimeoutError, ValueError) as exc:
            # Network/library errors only (see the module docstring): a
            # programming error must never read as a transient outage.
            return b"", "transient", {"error": type(exc).__name__}, []
        if frame is None or frame.empty:
            return b"", "legitimate_empty", {}, []
        return frame.to_csv().encode(), "complete", {"rows": int(len(frame))}, []

    return fetcher


def _default_history(ticker: str) -> pd.DataFrame:
    """The ported body of ``YFinanceAdapter.request``'s history branch."""
    import yfinance  # imported lazily; heavy and not needed to import the module

    return yfinance.Ticker(ticker).history(
        period="max", auto_adjust=False, actions=False)


def yfinance_earnings_fetcher(*, earnings_fn: Callable[[str], pd.DataFrame] | None = None) \
        -> Callable[[str], tuple]:
    """Return the ``fetcher(ticker) -> (csv_bytes, kind, meta, rows)`` seam."""

    def fetcher(ticker: str):
        try:
            frame = (earnings_fn or _default_earnings)(str(ticker))
        except (OSError, TimeoutError, ValueError) as exc:
            # Network/library errors only (see the module docstring).
            return b"", "transient", {"error": type(exc).__name__}, []
        if frame is None or frame.empty:
            return b"", "legitimate_empty", {}, []
        return frame.to_csv(index=False).encode(), "complete", {"rows": int(len(frame))}, []

    return fetcher


def _default_earnings(ticker: str) -> pd.DataFrame:
    """The ported body of ``YFinanceAdapter._earnings``.

    The session is derived here because it depends on the index being
    converted to US/Eastern first: a naive hour read would put every BMO print
    on the wrong side of the cutoff for half the year. No data (delisted or
    never covered) is an empty frame; a raised library error is the fetcher's
    ``transient`` classification, never swallowed here (spec R3).
    """
    import yfinance

    frame = yfinance.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT)

    rows = []
    if frame is not None and len(frame):
        for stamp in frame.index:
            ts = pd.Timestamp(stamp)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/New_York")
            annc_tod = f"{ts.hour:02d}{ts.minute:02d}"
            rows.append({
                "ticker": ticker,
                "event_date": str(ts.normalize().date()),
                "annc_tod": annc_tod,
                "session": _session_from_annc_tod(annc_tod) or "",
            })
    return pd.DataFrame(rows, columns=["ticker", "event_date", "annc_tod", "session"])


def _session_from_annc_tod(annc_tod) -> str | None:
    """Ported from ``engine.calendar.session_from_annc_tod`` (pure, no I/O)."""
    if annc_tod is None:
        return None
    if isinstance(annc_tod, float) and annc_tod != annc_tod:
        return None
    text = str(annc_tod).strip()
    if not text or text.lower() in ("none", "nan"):
        return None
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return None
    try:
        hhmm = int(digits[-4:]) if len(digits) >= 3 else int(digits) * 100
    except ValueError:
        return None
    if not 0 <= hhmm <= 2359:
        return None
    return "BMO" if hhmm < BMO_CUTOFF else "AMC"
