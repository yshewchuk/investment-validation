"""Native yfinance edges (spec s4c Rewrite 2).

``yfinance`` is a library call, not an HTTP GET, so its injectable seam is a
``history_fn``/``earnings_fn`` callable returning a pandas ``DataFrame``, never
``http_get``. The factories here wrap those callables into the byte-returning
fetchers the two native stores consume (the stores keep the DST/normalize and
session-parsing work that was already there); the default callables port
``engine/data/sources/yf.py``'s bodies verbatim, including the US/Eastern
conversion the BMO/AMC derivation depends on.

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
        -> Callable[[str], bytes | None]:
    """Return the ``fetcher(ticker) -> csv_bytes`` seam the moves store calls."""

    def fetcher(ticker: str):
        frame = (history_fn or _default_history)(str(ticker))
        if frame is None or frame.empty:
            return None
        return frame.to_csv().encode()

    return fetcher


def _default_history(ticker: str) -> pd.DataFrame:
    """The ported body of ``YFinanceAdapter.request``'s history branch."""
    import yfinance  # imported lazily; heavy and not needed to import the module

    return yfinance.Ticker(ticker).history(
        period="max", auto_adjust=False, actions=False)


def yfinance_earnings_fetcher(*, earnings_fn: Callable[[str], pd.DataFrame] | None = None) \
        -> Callable[[str], bytes | None]:
    """Return the ``fetcher(ticker) -> csv_bytes`` seam the calendar store calls."""

    def fetcher(ticker: str):
        frame = (earnings_fn or _default_earnings)(str(ticker))
        if frame is None or frame.empty:
            return None
        return frame.to_csv(index=False).encode()

    return fetcher


def _default_earnings(ticker: str) -> pd.DataFrame:
    """The ported body of ``YFinanceAdapter._earnings``.

    The session is derived here because it depends on the index being
    converted to US/Eastern first: a naive hour read would put every BMO print
    on the wrong side of the cutoff for half the year. No data (delisted or
    never covered) is an empty frame, exactly like the legacy 404.
    """
    import yfinance

    try:
        frame = yfinance.Ticker(ticker).get_earnings_dates(limit=EARNINGS_LIMIT)
    except Exception:  # yfinance raises a zoo of types for "no data"
        frame = None

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
