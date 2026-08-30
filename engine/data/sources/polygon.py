"""Polygon adapter — shells out to ``curl``.

This is not a style preference. A fresh Python process using ``urllib`` gets
401 "Unknown API Key" from Polygon with the *same* key that ``curl`` accepts in
the same shell — observed repeatedly, never explained (AGENTS.md). Rather than
rediscover it, the adapter shells out.

The key is passed through the subprocess environment and referenced by
``$POLYGON_API_KEY`` inside the header argument, so it never appears in the
process argument list (where any ``ps`` would show it).

Rate: price/aggs are ~10 req/min on this plan, hence the 6.5s gate and 65s
backoff in :mod:`engine.data.throttle`, and one Polygon process at a time.
"""
from __future__ import annotations

import os
import subprocess
import time
import urllib.parse
from typing import Any

from engine.data.sources.base import CredentialRotated, Response, redact

__all__ = ["PolygonAdapter", "BASE_URL", "option_ticker"]

BASE_URL = "https://api.polygon.io"

#: Marker curl writes after the body so status and headers can be split out.
_SENTINEL = "\n__HTTP_STATUS__:"


def _config_escape(text: str) -> str:
    """Encode a value for a curl ``-K`` config file.

    curl 8.18 rejects a literal newline inside a quoted config value with the
    misleading error ``option --config: is unknown`` — which is how a sentinel
    carrying a real newline must be written as the two characters ``\\n`` that
    curl then expands back into a newline when it applies the value. Backslash
    and double-quote get the same treatment for the same reason.
    """
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def option_ticker(symbol: str, expiry: str, right: str, strike: float) -> str:
    """OCC contract id: ``O:{SYM}{YYMMDD}{C|P}{strike*1000:08d}``."""
    yymmdd = "".join(expiry.split("-"))[2:]
    r = right.upper()[0]
    if r not in ("C", "P"):
        raise ValueError(f"right must be C or P, got {right!r}")
    return f"O:{symbol.upper()}{yymmdd}{r}{round(strike * 1000):08d}"


class PolygonAdapter:
    name = "polygon"

    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL):
        self._key = api_key or os.environ.get("POLYGON_API_KEY") or ""
        self.base_url = base_url.rstrip("/")

    def build_url(self, endpoint: str, params: dict[str, Any]) -> str:
        path = endpoint.strip("/")
        query = {k: v for k, v in sorted(params.items()) if v is not None}
        url = f"{self.base_url}/{path}"
        return f"{url}?{urllib.parse.urlencode(query, doseq=True)}" if query else url

    def build_config(self, url: str, timeout: float) -> str:
        """The curl ``-K`` config text for one request.

        The bearer token goes in via stdin config rather than an argv element,
        so it never appears in the process table.
        """
        return (
            f'header = "Authorization: Bearer {self._key}"\n'
            f'url = "{url}"\n'
            f"silent\nshow-error\n"
            f"max-time = {int(timeout)}\n"
            f'write-out = "{_config_escape(_SENTINEL)}%{{http_code}}"\n'
        )

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> Response:
        if not self._key:
            raise CredentialRotated("POLYGON_API_KEY is unset — cannot call Polygon")
        url = self.build_url(endpoint, params)
        config = self.build_config(url, timeout)
        started = time.monotonic()
        proc = subprocess.run(
            ["curl", "--config", "-"],
            input=config.encode(),
            capture_output=True,
            check=False,
        )
        elapsed = time.monotonic() - started

        raw = proc.stdout
        status = 0
        marker = _SENTINEL.encode()
        if marker in raw:
            raw, _, tail = raw.rpartition(marker)
            try:
                status = int(tail.strip() or 0)
            except ValueError:
                status = 0
        elif proc.returncode != 0:
            raise ConnectionError(
                f"curl exited {proc.returncode}: {redact(proc.stderr.decode(errors='replace'))[:300]}"
            )
        return Response(
            status=status,
            body=raw,
            headers={},
            url=redact(url),
            elapsed_s=elapsed,
        )

    def quota_from(self, response: Response) -> int | None:
        return None  # Polygon bills by plan tier, not by a per-call quota

    def is_auth_failure(self, response: Response) -> bool:
        if response.status == 401:
            return True
        return b"Unknown API Key" in response.body[:400]
