"""Nasdaq adapter — the forward earnings calendar.

The gap this fills: ORATS ``/hist/earnings`` is a HISTORY endpoint. Its cached
payloads stop at the last print that already happened (AAPL's last row was
2026-07-30 in a file fetched 2026-08-28), so no amount of re-pulling it
produces an upcoming calendar. Without another source the monitoring board has
nothing to score.

``api.nasdaq.com/api/calendar/earnings?date=YYYY-MM-DD`` answers with every
company reporting on that date — the whole market in ONE call, no key. A
three-week horizon is ~21 calls, against ~2,900 for any per-ticker source.

Two properties measured before this was wired, both of which the calendar
merge depends on:

* **``time`` is only populated going forward.** On historical dates Nasdaq
  returns ``time-not-supplied`` for essentially everything (1,392 of 1,395 rows
  over ten past dates), so its session cannot be validated retrospectively
  against ORATS ``anncTod``. It is therefore the lowest-priority session
  source; see :func:`engine.calendar.build_calendar`.
* **Forward session coverage is partial and firms up as the print nears**:
  51.6% overall, 75.0% across the 1–10B and >10B slices this program trades,
  9% at 2–3 weeks out against 54% inside a week. So the board fills in as an
  event approaches — which is when the entry decision is actually taken — and
  yfinance covers part of the remainder.

Unmetered and keyless, but it is a browser endpoint: the default Python
user-agent gets refused, so one is set here.
"""
from __future__ import annotations

import time
import urllib.parse
from typing import Any

from engine.data.sources.base import Response, redact

__all__ = ["NasdaqAdapter", "BASE_URL", "SESSION_BY_TIME"]

BASE_URL = "https://api.nasdaq.com/api"

#: Nasdaq's ``time`` field → the program's BMO/AMC session. ``time-not-supplied``
#: is deliberately absent: an unmapped value must become NULL, because "Nasdaq
#: has not said yet" and "the print is after the close" are different facts and
#: only one of them is safe to trade on.
SESSION_BY_TIME = {
    "time-pre-market": "BMO",
    "time-after-hours": "AMC",
}

#: Nasdaq refuses the default urllib/requests user-agent with a 403. This is
#: not evasion of a rate limit or an auth control — the endpoint is public and
#: keyless, and it backs a page any browser can load.
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)


class NasdaqAdapter:
    """``endpoint`` is the api path, e.g. ``"calendar/earnings"``."""

    name = "nasdaq"

    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")

    def build_url(self, endpoint: str, params: dict[str, Any]) -> str:
        path = endpoint.strip("/")
        query = {k: v for k, v in sorted(params.items()) if v is not None}
        url = f"{self.base_url}/{path}"
        return f"{url}?{urllib.parse.urlencode(query)}" if query else url

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> Response:
        import requests  # lazily, so this module imports without network deps

        url = self.build_url(endpoint, params)
        started = time.monotonic()
        resp = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/json"},
        )
        return Response(
            status=resp.status_code,
            body=resp.content,
            headers={k: v for k, v in resp.headers.items()},
            url=redact(url),
            elapsed_s=time.monotonic() - started,
        )

    def quota_from(self, response: Response) -> int | None:
        return None  # unmetered

    def is_auth_failure(self, response: Response) -> bool:
        # There is no credential to rotate. A 403 here means the user-agent was
        # refused, which is a code fix rather than a key fix — so it must NOT
        # raise CredentialRotated and stop the nightly run.
        return False
