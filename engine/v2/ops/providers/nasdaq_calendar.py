"""Native Nasdaq forward-calendar fetcher (spec s4c Rewrite 2).

One fetch unit denominates one ``date`` on Nasdaq's keyless
``api.nasdaq.com/api/calendar/earnings`` endpoint; the whole market answers in
one call. This is the v2 network edge for that endpoint, shaped exactly like
``orats_daily_market.py``: a fetcher closure whose sole test seam is
``http_get(url, timeout) -> (status, headers, body)``. Credentials are not a
concept here -- the endpoint is unmetered and keyless -- so nothing is read
from the environment.

Classification (spec R1) happens HERE, before the store caches anything: a
2xx with the documented ``data.rows`` list is ``complete`` (rows) or
``legitimate_empty`` (no rows), a 2xx whose body does not parse to that shape
is ``refused``, 404 is ``not_final`` (the date is not published yet), 429 and
5xx are ``transient``, 401/403 (and only those) are ``credential_invalid``,
any other non-2xx is ``refused`` (bad source data), and a raised network error
is ``transient`` -- never a silently empty row list. A programming error out of
``http_get`` propagates, it is never classified as an outage.

The URL builder and the ``data.rows`` parser are ported from
``engine/data/sources/nasdaq.py`` and the legacy pull's own ``_nasdaq_rows``;
the browser user-agent lives in the default client, never on the URL.
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any, Callable, Mapping

from engine.v2.ops.errors import fail

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
    receipt cache can mark a unit as fetched. ``rows`` is only non-empty for a
    ``complete`` response.
    """

    def fetcher(unit):
        if str(unit.get("table_name", "")) != TABLE_NAME:
            raise fail("INVALID_REQUEST", "nasdaq fetcher only serves earnings_events")
        request = http_get or _requests_get
        day = str(unit["partition_key"])
        try:
            status, _headers, body = request(_nasdaq_url({"date": day}),
                                             timeout=REQUEST_TIMEOUT_SECONDS)
        except (OSError, TimeoutError) as exc:
            # Network errors only: requests' RequestException is an OSError
            # subclass, so the default client is covered; anything else (a
            # TypeError, an OpsError) is a programming error and propagates.
            return b"", "transient", {"error": type(exc).__name__, "date": day}, []
        document = _json_document(body)
        kind = _response_kind(status, document)
        rows = _nasdaq_rows(document) if kind in ("complete", "legitimate_empty") else []
        return body, kind, {"status": int(status), "date": day}, rows

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


def _json_document(body: Any) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _rows_field(document: Any) -> list[dict] | None:
    """The ``data.rows`` list of one response, or ``None`` for any other shape."""
    data = document.get("data") if isinstance(document, dict) else None
    rows = data.get("rows") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return None
    return [row for row in rows if isinstance(row, dict)]


def _nasdaq_rows(document: Any) -> list[dict]:
    """The parsed ``data.rows`` list; ``[]`` on anything malformed."""
    return _rows_field(document) or []


def _response_kind(status: int, document: Any) -> str:
    """R1 classification of one Nasdaq response, before anything is cached.

    401/403 alone is ``credential_invalid``; an unparseable body or any other
    non-2xx is ``refused`` (bad source data, non-retryable).
    """
    if status in (401, 403):
        return "credential_invalid"
    if status == 404:
        return "not_final"
    if status == 429 or status >= 500:
        return "transient"
    if not 200 <= status < 300:
        return "refused"
    rows = _rows_field(document)
    if rows is None:
        return "refused"  # a 2xx without the documented shape is not usable data
    return "complete" if rows else "legitimate_empty"
