"""Native Nasdaq forward-calendar fetcher (spec s4c Rewrite 2).

One fetch unit denominates one ``date`` on Nasdaq's keyless
``api.nasdaq.com/api/calendar/earnings`` endpoint; the whole market answers in
one call. This is the v2 network edge for that endpoint, shaped exactly like
``orats_daily_market.py``: a fetcher closure whose sole test seam is
``http_get(url, timeout) -> (status, headers, body)``. Credentials are not a
concept here -- the endpoint is unmetered and keyless -- so nothing is read
from the environment.

The URL builder and the ``data.rows`` parser are ported from
``engine/data/sources/nasdaq.py`` and the legacy pull's own ``_nasdaq_rows``;
the browser user-agent lives in the default client, never on the URL.
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any, Callable, Mapping

from engine.v2.ops.errors import fail
from engine.v2.ops.incremental_data import classify_response

__all__ = ["BASE_URL", "REQUEST_TIMEOUT_SECONDS", "SESSION_BY_TIME", "nasdaq_calendar_fetcher"]

TABLE_NAME = "earnings_events"
ENDPOINT = "calendar/earnings"
BASE_URL = "https://api.nasdaq.com/api"
REQUEST_TIMEOUT_SECONDS = 30.0

#: Nasdaq's ``time`` field -> the program's BMO/AMC session. ``time-not-supplied``
#: is deliberately absent: an unmapped value must become NULL.
SESSION_BY_TIME = {
    "time-pre-market": "BMO",
    "time-after-hours": "AMC",
}

#: Nasdaq refuses the default urllib/requests user-agent with a 403; the
#: endpoint is public and keyless, and this is not evasion of a rate limit.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


def nasdaq_calendar_fetcher(*, http_get: Callable[..., tuple] | None = None) \
        -> Callable[[dict], tuple]:
    """Return a ``fetcher(unit)`` callable for the forward-calendar job.

    The returned tuple is ``(raw_bytes, response_kind, response_meta, rows)``
    -- the same shape ``orats_daily_market_fetcher`` uses, so the shared raw
    receipt cache can mark a unit as fetched.
    """

    def fetcher(unit):
        if str(unit.get("table_name", "")) != TABLE_NAME:
            raise fail("INVALID_REQUEST", "nasdaq fetcher only serves earnings_events")
        request = http_get or _requests_get
        day = str(unit["partition_key"])
        status, _headers, body = request(_nasdaq_url({"date": day}),
                                         timeout=REQUEST_TIMEOUT_SECONDS)
        rows = _nasdaq_rows(body)
        outcome = _classify(unit, status, rows)
        return body, outcome.kind, {"status": int(status), "date": day}, rows

    return fetcher


def _requests_get(url: str, *, timeout: float) -> tuple[int, dict, bytes]:
    """The default network client: one plain keyless GET with a browser UA."""
    import requests

    response = requests.get(url, timeout=timeout, headers={
        "User-Agent": USER_AGENT, "Accept": "application/json"})
    return response.status_code, dict(response.headers), response.content


def _nasdaq_url(params: Mapping[str, Any]) -> str:
    query = {key: value for key, value in sorted(params.items()) if value is not None}
    url = f"{BASE_URL}/{ENDPOINT}"
    return f"{url}?{urllib.parse.urlencode(query)}" if query else url


def _nasdaq_rows(body: Any) -> list[dict]:
    """The ``data.rows`` list of one response; ``[]`` on anything malformed."""
    try:
        payload = json.loads(body.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    rows = data.get("rows") if isinstance(data, dict) else None
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _classify(unit: dict, status: int, rows: list[dict]):
    expected = tuple(str(key) for key in unit.get("expected_keys", ()))
    request_id = str(unit.get("request_id", ""))
    if 200 <= status < 300:
        returned = expected if rows else ()
        empty = () if rows else expected
        return classify_response(status, expected, returned_keys=returned,
                                 empty_keys=empty, final=True, request_id=request_id)
    return classify_response(status, expected, final=True, request_id=request_id)
