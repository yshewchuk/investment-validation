"""yfinance adapter.

yfinance is a library, not an HTTP endpoint, so this adapter shapes its output
into the same :class:`~engine.data.sources.base.Response` the fetch wrapper
expects: a CSV body that lands in the Tier-1 store verbatim like any other
fetched payload. Unmetered, but still cached — the point of Tier 1 is that a
repeated request never touches the network, whatever the source charges.

Two endpoints:

* ``history`` — the spot cross-check in the validation battery (ORATS close vs
  yfinance close, 1.3% tolerance) and the daily price history behind the run-up
  features.
* ``earnings`` — announcement dates for one ticker, past and future, indexed by
  a tz-aware timestamp whose TIME carries the BMO/AMC session. It is the
  session source for the forward calendar: measured against ORATS ``anncTod``
  on 716 overlapping historical events it agreed on 99.72% (714/716), which is
  what earns it priority over Nasdaq, whose session cannot be checked
  retrospectively at all. One call per ticker at ~0.7–0.9s, so callers bound
  the set — Nasdaq discovers WHO reports and this confirms WHEN in the day.
"""
from __future__ import annotations

import io
import time
from typing import Any

from engine.data.sources.base import Response

__all__ = ["YFinanceAdapter"]


#: Announcement rows requested per ticker. Enough to cover the next print plus
#: a couple of years of history — the history is what makes the session
#: checkable against ORATS once an event is past.
EARNINGS_LIMIT = 12


class YFinanceAdapter:
    """``endpoint`` is the yfinance call: ``"history"`` or ``"earnings"``."""

    name = "yfinance"

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> Response:
        import yfinance  # imported lazily; heavy and not needed to import the module

        ep = endpoint.strip("/")
        if ep == "earnings":
            return self._earnings(yfinance, params)
        if ep != "history":
            raise ValueError(f"unsupported yfinance endpoint {endpoint!r}")
        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("yfinance history requires a `ticker` param")

        started = time.monotonic()
        frame = yfinance.Ticker(str(ticker)).history(
            start=params.get("start"),
            end=params.get("end"),
            period=params.get("period") or ("max" if not params.get("start") else None),
            auto_adjust=False,
            actions=False,
        )
        buf = io.StringIO()
        frame.to_csv(buf)
        body = buf.getvalue().encode()
        return Response(
            status=200 if not frame.empty else 404,
            body=body,
            headers={"content-type": "text/csv", "x-rows": str(len(frame))},
            url=f"yfinance://history/{ticker}",
            elapsed_s=time.monotonic() - started,
        )

    def _earnings(self, yfinance, params: dict[str, Any]) -> Response:
        """Announcement dates for one ticker, as CSV with an explicit session.

        The session is derived HERE rather than by a downstream reader, because
        it depends on the index being converted to US/Eastern first: the raw
        timestamps are tz-aware, and a naive read of the hour would put every
        BMO print on the wrong side of the cutoff for half the year.

        A ticker with no earnings data (delisted, or never covered) is a 404
        with an empty body — a real answer, cached like any other, so the next
        run does not ask again.
        """
        import pandas as pd

        from engine.calendar import session_from_annc_tod

        ticker = params.get("ticker")
        if not ticker:
            raise ValueError("yfinance earnings requires a `ticker` param")
        limit = int(params.get("limit") or EARNINGS_LIMIT)

        started = time.monotonic()
        try:
            frame = yfinance.Ticker(str(ticker)).get_earnings_dates(limit=limit)
        except Exception:  # yfinance raises a zoo of types for "no data"
            frame = None

        rows = []
        if frame is not None and len(frame):
            for stamp in frame.index:
                ts = pd.Timestamp(stamp)
                if ts.tzinfo is not None:
                    ts = ts.tz_convert("America/New_York")
                annc_tod = f"{ts.hour:02d}{ts.minute:02d}"
                rows.append(
                    {
                        "ticker": str(ticker),
                        "event_date": str(ts.normalize().date()),
                        "annc_tod": annc_tod,
                        "session": session_from_annc_tod(annc_tod) or "",
                    }
                )

        buf = io.StringIO()
        pd.DataFrame(
            rows, columns=["ticker", "event_date", "annc_tod", "session"]
        ).to_csv(buf, index=False)
        body = buf.getvalue().encode()
        return Response(
            status=200 if rows else 404,
            body=body,
            headers={"content-type": "text/csv", "x-rows": str(len(rows))},
            url=f"yfinance://earnings/{ticker}",
            elapsed_s=time.monotonic() - started,
        )

    def quota_from(self, response: Response) -> int | None:
        return None

    def is_auth_failure(self, response: Response) -> bool:
        return False
