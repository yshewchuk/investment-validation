"""Authenticated, read-only operations transport backed by versioned artifacts."""
from __future__ import annotations

import hmac
import http.cookies
import http.server
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlsplit

from engine.v2.foundation import ArtifactError, safe_relative_path

HTTPStatus = http.HTTPStatus

__all__ = ["OperationsHandler", "create_server", "shell_document"]

_VIEWS = ("board", "explorer", "book", "models", "derivation", "flags")


def _safe_file(root: Path, relative: str) -> Path:
    parts = safe_relative_path(relative)
    if root.is_symlink():
        raise ValueError("indirect artifact root")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("indirect artifact path")
    if not current.is_file():
        raise FileNotFoundError(relative)
    return current


def _read_health(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("health artifact is indirect")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema_version") != "operations_health.v1.0":
        raise ValueError("unsupported health artifact")
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def shell_document(*, frozen_at: str | None = None) -> bytes:
    """Small outer shell; legacy bytes are loaded inside its immutable frame."""
    frozen = frozen_at or "unknown"
    routes = ("trades/board", "trades/explorer", "trades/book", "models/modelx",
              "models/derivation", "models/health")
    views = "".join(f'<a href="/#/{route}">{view}</a> ' for view, route in zip(_VIEWS, routes))
    html = f'''<!doctype html><meta charset="utf-8"><title>Operations shell</title>
<style>body{{margin:0;font:14px sans-serif}}#ops{{padding:8px;background:#20252b;color:#eee}}#ops.unknown{{background:#634}}nav a{{margin-right:12px}}main{{min-height:90vh}}</style>
<div id="ops">health: <span id="state">unknown</span> <small id="stamp">offline frozen at {frozen}</small></div>
<nav>{views}</nav><main><iframe id="legacy" title="legacy dashboard" src="/release/current/index.html" style="width:100%;height:90vh;border:0"></iframe></main>
<script>
const state=document.querySelector('#state'), stamp=document.querySelector('#stamp'), banner=document.querySelector('#ops');
async function health(){{try{{const r=await fetch('/health.json',{{credentials:'same-origin'}});if(!r.ok)throw Error();const h=await r.json();const b=h.code_budgets||{{}};state.textContent=h.withheld_release?'withheld':(b.consecutive_nights?'degraded':'current');stamp.textContent='updated '+h.generated_at+'; failures '+(b.consecutive_nights||0);banner.className='';}}catch(e){{state.textContent='unknown / stale';stamp.textContent='offline frozen at {frozen}';banner.className='unknown';}}}}
health(); setInterval(health,30000);
const frame=document.querySelector('#legacy'); function route(){{frame.src='/release/current/index.html'+(location.hash||'#/trades/board');}} route(); window.addEventListener('hashchange',route);
</script>'''
    return html.encode()


class OperationsHandler(http.server.BaseHTTPRequestHandler):
    """Handler factory state is assigned by ``create_server``; no ops imports."""

    server_version = "v2-operations/1"

    def do_GET(self):  # noqa: N802
        config = self.server.config
        path = unquote(urlsplit(self.path).path)
        if path == "/health.json":
            return self._artifact(config, _read_health, config.health_path, auth=True)
        if path == "/" or path.lstrip("/") in _VIEWS:
            return self._send(HTTPStatus.OK, shell_document(frozen_at=config.frozen_at), "text/html")
        if path.startswith("/legacy/"):
            return self._send(HTTPStatus.OK, shell_document(frozen_at=config.frozen_at), "text/html")
        if path == "/release/current":
            return self._current_route(config)
        if path.startswith("/release/"):
            return self._release_route(config, path)
        return self._send(HTTPStatus.NOT_FOUND, b"missing\n", "text/plain")

    def _current_route(self, config):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        try:
            release = config.release_root / "CURRENT"
            if release.is_symlink() or not release.is_file():
                raise ValueError("unsafe current pointer")
            name = release.read_text().strip()
            if len(safe_relative_path(name)) != 1:
                raise ValueError("unsafe current pointer")
            return self._send(HTTPStatus.FOUND, b"", "text/plain",
                              {"Location": "/release/" + name + "/index.html"})
        except (ArtifactError, OSError, ValueError):
            return self._send(HTTPStatus.NOT_FOUND, b"unknown release\n", "text/plain")

    def _release_route(self, config, path):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        rel = path.removeprefix("/release/").split("/", 1)
        if len(rel) != 2 or not rel[0] or not rel[1]:
            return self._send(HTTPStatus.NOT_FOUND, b"missing\n", "text/plain")
        try:
            if rel[0] == "current":
                pointer = config.release_root / "CURRENT"
                if pointer.is_symlink() or not pointer.is_file():
                    raise ValueError("unsafe current pointer")
                rel[0] = pointer.read_text().strip()
            if len(safe_relative_path(rel[0])) != 1:
                raise ValueError("unsafe release id")
            file = _safe_file(config.release_root / "releases" / rel[0], rel[1])
            content_type = mimetypes.guess_type(file.name)[0] or "application/octet-stream"
            return self._send(HTTPStatus.OK, file.read_bytes(), content_type)
        except (ArtifactError, OSError, ValueError):
            return self._send(HTTPStatus.NOT_FOUND, b"missing\n", "text/plain")

    def _artifact(self, config, reader, path, *, auth):
        if auth and not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        try:
            return self._send(HTTPStatus.OK, reader(path), "application/json")
        except (OSError, ValueError, json.JSONDecodeError):
            return self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"unknown\n", "text/plain")

    def _authorized(self):
        expected = self.server.config.token
        if not expected:
            return False
        supplied = self.headers.get("Authorization", "")
        cookie = http.cookies.SimpleCookie()
        cookie.load(self.headers.get("Cookie", ""))
        cookie_value = cookie.get("operations_token")
        return hmac.compare_digest(supplied, "Bearer " + expected) or (
            cookie_value is not None and hmac.compare_digest(cookie_value.value, expected))

    def log_message(self, format, *args):
        return

    def _send(self, status, body, content_type, headers=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)


def create_server(address, *, token: str, health_path: Path | str, release_root: Path | str,
                  frozen_at: str = "unknown"):
    config = type("Config", (), {"token": token, "health_path": Path(health_path),
                                  "release_root": Path(release_root), "frozen_at": frozen_at})
    server = http.server.ThreadingHTTPServer(address, OperationsHandler)
    server.config = config
    return server
