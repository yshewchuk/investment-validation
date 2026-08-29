"""yfinance adapter.

yfinance is a library, not an HTTP endpoint, so this adapter shapes its output
into the same :class:`~engine.data.sources.base.Response` the fetch wrapper
expects: a CSV body that lands in the Tier-1 store verbatim like any other
fetched payload. Unmetered, but still cached — the point of Tier 1 is that a
repeated request never touches the network, whatever the source charges.

Used for the spot cross-check in the validation battery (ORATS close vs
yfinance close, 1.3% tolerance) and for the daily price history behind the
run-up features.
"""
from __future__ import annotations

import io
import time
from typing import Any

from engine.data.sources.base import Response

__all__ = ["YFinanceAdapter"]


class YFinanceAdapter:
    """``endpoint`` is the yfinance call: currently only ``"history"``."""

    name = "yfinance"

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> Response:
        import yfinance  # imported lazily; heavy and not needed to import the module

        ep = endpoint.strip("/")
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

    def quota_from(self, response: Response) -> int | None:
        return None

    def is_auth_failure(self, response: Response) -> bool:
        return False
