"""oquants adapter — the Playwright cookie→token dance.

Direct calls to ``api2.oquants.com`` fail. The working flow (AGENTS.md) is:

1. launch headless chromium and install the session cookie from ``.env``;
2. navigate to ``oquants.com/models`` and let the app initialise;
3. ``fetch('/api/auth/token')`` from inside the page to mint a bearer token;
4. use ``Authorization: Bearer <token>`` against ``api2.oquants.com``.

The token is minted **once per process** and reused: step 1–3 costs a browser
launch, and doing it per call turns a bulk pull into a browser-launch loop.

Standing rule from ``bt/straddle/VERDICT_2026-08-27.md``: oquants
straddle/return marks are model-fitted ("Smooth Straddle Px"), **not traded
prices**, and are banned from any P&L path. This adapter exists for event dates,
realized moves, and IV series — never for pricing a trade.
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from typing import Any

from engine.data.sources.base import CredentialRotated, Response, redact

__all__ = ["OquantsAdapter", "API_BASE", "APP_BASE", "PNL_BANNED_ENDPOINTS"]

API_BASE = "https://api2.oquants.com/api/v1"
APP_BASE = "https://oquants.com"

#: Endpoints whose payloads are model-fitted marks. Loud failure beats a
#: plausible-looking backtest built on prices that never traded.
PNL_BANNED_ENDPOINTS = frozenset(
    {"dashboard/volatility/backtest", "research/backtest", "research/run-model-live"}
)


class OquantsAdapter:
    name = "oquants"

    def __init__(
        self,
        cookie_name: str | None = None,
        cookie_value: str | None = None,
        api_base: str = API_BASE,
    ):
        self.cookie_name = cookie_name or os.environ.get("OQUANTS_COOKIE_NAME") or ""
        self._cookie_value = cookie_value or os.environ.get("OQUANTS_COOKIE_VALUE") or ""
        self.api_base = api_base.rstrip("/")
        self._token: str | None = None

    # -- auth -------------------------------------------------------------

    def token(self, refresh: bool = False) -> str:
        """Mint (once) and return the bearer token."""
        if self._token and not refresh:
            return self._token
        if not (self.cookie_name and self._cookie_value):
            raise CredentialRotated(
                "OQUANTS_COOKIE_NAME / OQUANTS_COOKIE_VALUE unset — cannot authenticate"
            )
        self._token = self._mint_token()
        return self._token

    def _mint_token(self) -> str:  # pragma: no cover - needs a real browser
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                context = browser.new_context()
                context.add_cookies(
                    [
                        {
                            "name": self.cookie_name,
                            "value": self._cookie_value,
                            "domain": "oquants.com",
                            "path": "/",
                            "secure": True,
                            "httpOnly": True,
                            "sameSite": "Lax",
                        }
                    ]
                )
                page = context.new_page()
                page.goto(f"{APP_BASE}/models", wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                token = page.evaluate(
                    "async () => (await (await fetch('/api/auth/token')).json()).token"
                )
            finally:
                browser.close()
        if not token:
            raise CredentialRotated(
                "oquants returned no token — the session cookie has expired"
            )
        return str(token)

    # -- requests ---------------------------------------------------------

    def build_url(self, endpoint: str, params: dict[str, Any]) -> str:
        path = endpoint.strip("/")
        query = {k: v for k, v in sorted(params.items()) if v is not None}
        url = f"{self.api_base}/{path}"
        return f"{url}?{urllib.parse.urlencode(query, doseq=True)}" if query else url

    def request(self, endpoint: str, params: dict[str, Any], timeout: float) -> Response:
        import requests

        if endpoint.strip("/") in PNL_BANNED_ENDPOINTS:
            raise ValueError(
                f"{endpoint!r} returns model-fitted marks, which are banned from "
                "P&L paths (bt/straddle/VERDICT_2026-08-27.md). Use ORATS chains."
            )
        url = self.build_url(endpoint, params)
        started = time.monotonic()
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {self.token()}"},
            timeout=timeout,
        )
        return Response(
            status=resp.status_code,
            body=resp.content,
            headers=dict(resp.headers),
            url=redact(url),
            elapsed_s=time.monotonic() - started,
        )

    def quota_from(self, response: Response) -> int | None:
        return None

    def is_auth_failure(self, response: Response) -> bool:
        if response.status in (401, 403):
            return True
        # oquants answers errors 200-with-a-body: {"success": false, "data": null}
        head = response.body[:200].lstrip()
        if head.startswith(b"{"):
            try:
                doc = json.loads(response.body)
            except (ValueError, UnicodeDecodeError):
                return False
            if isinstance(doc, dict) and doc.get("success") is False:
                message = str(doc.get("message", "")).lower()
                return "auth" in message or "token" in message or "session" in message
        return False
