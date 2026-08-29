"""ORATS adapter — api.orats.io.

Two things distinguish ORATS from the other sources:

* **Auth is a query parameter**, not a header (``?token=...``). Plain
  ``requests`` works; there is no curl quirk here. It also means the URL itself
  is a secret, so every URL leaving this module goes through
  :func:`~engine.data.sources.base.redact`.
* **Quota, not rate, is the constraint**: 20,000 calls/month against 1,000
  req/min. The response headers carry the running count, except when a response
  is served from a CDN cache and the headers are absent entirely — which is why
  the quota ledger counts logged rows rather than trusting the header.

Known-bad paths that must never be retried (they 403 because they are not real
endpoint names): ``/hist/stocks``, ``/stocks``, ``/histdata``, ``/ivdays``,
``/histearnings``, ``/straddle``, ``/straddleday``.
"""
from __future__ import annotations

import os
import time
import urllib.parse
from typing import Any

from engine.data.sources.base import CredentialRotated, Response, redact

__all__ = ["OratsAdapter", "OratsResponse", "BASE_URL", "KNOWN_BAD_PATHS"]

BASE_URL = "https://api.orats.io/datav2"

KNOWN_BAD_PATHS = frozenset(
    {"hist/stocks", "stocks", "histdata", "ivdays", "histearnings", "straddle", "straddleday"}
)

QUOTA_HEADERS = ("X-Monthly-Quota-Remaining", "x-monthly-quota-remaining")
USED_HEADERS = ("X-Monthly-Quota-Used", "x-monthly-quota-used")
ALLOWED_HEADERS = ("X-Monthly-Quota-Allowed", "x-monthly-quota-allowed")
RATE_HEADERS = ("X-RateLimit-Remaining", "x-ratelimit-remaining")


class OratsResponse(Response):
    def quota_remaining(self) -> int | None:
        return _header_int(self.headers, QUOTA_HEADERS)


def _header_int(headers: dict[str, str], names: tuple[str, ...]) -> int | None:
    for name in names:
        if name in headers:
            try:
                return int(str(headers[name]).strip())
            except (TypeError, ValueError):
                return None
    return None


class OratsAdapter:
    """Historical + live ORATS endpoints."""

    name = "orats"

    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL):
        self._key = api_key or os.environ.get("ORATS_API_KEY") or ""
        self.base_url = base_url.rstrip("/")

    def build_url(self, endpoint: str, params: dict[str, Any]) -> str:
        """Full request URL *including* the token — never logged or persisted."""
        path = endpoint.strip("/")
        if path in KNOWN_BAD_PATHS:
            raise ValueError(
                f"{path!r} is a known-bad ORATS path (403, not a real endpoint) — "
                "retrying it only spends time"
            )
        query = {k: v for k, v in sorted(params.items()) if v is not None}
        query["token"] = self._key
        return f"{self.base_url}/{path}?" + urllib.parse.urlencode(query, doseq=True)

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> OratsResponse:
        import requests  # imported lazily so the module imports without network deps

        if not self._key:
            raise CredentialRotated("ORATS_API_KEY is unset — cannot call ORATS")
        url = self.build_url(endpoint, params)
        started = time.monotonic()
        resp = requests.get(url, timeout=timeout)
        return OratsResponse(
            status=resp.status_code,
            body=resp.content,
            headers={k: v for k, v in resp.headers.items()},
            url=redact(url),
            elapsed_s=time.monotonic() - started,
        )

    def quota_from(self, response: Response) -> int | None:
        return _header_int(response.headers, QUOTA_HEADERS)

    def quota_used(self, response: Response) -> int | None:
        return _header_int(response.headers, USED_HEADERS)

    def rate_remaining(self, response: Response) -> int | None:
        return _header_int(response.headers, RATE_HEADERS)

    def is_auth_failure(self, response: Response) -> bool:
        if response.status == 401:
            return True
        # ORATS answers a bad key with 403 and a body that says so; a 403 on a
        # real path with a good key does not occur, but a 403 on a mistyped path
        # does — so the body text is what separates the two.
        if response.status == 403:
            text = response.body[:400].lower()
            return b"key" in text or b"token" in text or b"unauthor" in text
        return False
