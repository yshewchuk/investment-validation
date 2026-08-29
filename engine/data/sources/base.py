"""The source-adapter contract.

:mod:`engine.data.fetch` is source-agnostic: it handles cache keys, the raw
store, the fetch log, throttling and quota, and knows nothing about how any
particular API authenticates. Everything source-specific — auth style, URL
shape, quota headers, the curl-vs-requests quirk — lives in one adapter.

Adapters must never let a credential escape. ``Response.url`` is persisted into
the Tier-1 metadata next to the body, and ORATS authenticates with a *query
parameter*, so an unredacted URL would write the API key to disk in plain text.
:func:`redact` is applied by every adapter before a URL is returned.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["Response", "SourceAdapter", "CredentialRotated", "FetchError", "redact"]


class FetchError(RuntimeError):
    """A fetch failed in a way retrying will not fix."""


class CredentialRotated(FetchError):
    """A 401 / login redirect: the credential changed. Stop, tell the user.

    Never caught into a retry loop. A rotated key produces an unbounded stream
    of identical failures, and the only fix is a human updating ``.env``.
    """


#: Query-parameter names whose values are secrets.
SECRET_PARAMS = ("token", "apikey", "api_key", "apiKey", "key", "access_token")

_SECRET_RE = re.compile(
    r"(?i)\b(" + "|".join(re.escape(p) for p in SECRET_PARAMS) + r")=([^&\s]+)"
)


def redact(text: str) -> str:
    """Replace secret query-parameter values with ``<redacted>``."""
    return _SECRET_RE.sub(r"\1=<redacted>", text)


@dataclass
class Response:
    """What an adapter hands back to the fetch wrapper."""

    status: int
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)
    url: str = ""  # ALWAYS redacted
    elapsed_s: float = 0.0

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def quota_remaining(self) -> int | None:
        """Calls left this month, when the source reports it."""
        return None


class SourceAdapter(Protocol):
    """What :func:`engine.data.fetch.fetch` requires of a source."""

    name: str

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> Response:
        """Perform one network call. Must not retry, sleep, or raise on non-2xx."""
        ...

    def quota_from(self, response: Response) -> int | None:
        """Remaining monthly quota parsed from ``response``, or None."""
        ...

    def is_auth_failure(self, response: Response) -> bool:
        """True when ``response`` indicates a rotated or rejected credential."""
        ...
