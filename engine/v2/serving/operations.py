"""Authenticated, read-only operations transport backed by versioned artifacts."""
from __future__ import annotations

import hmac
import http.cookies
import http.server
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from engine.v2.foundation import ArtifactError, safe_relative_path
from engine.v2.models.deployment import (
    DeploymentError,
    ReleaseNotStaged,
    current_pointer,
    resolve_release,
)

HTTPStatus = http.HTTPStatus

__all__ = ["OperationsHandler", "create_server", "model_release_page_document", "shell_document"]

_VIEWS = ("board", "explorer", "book", "models", "derivation", "flags")

MODEL_RELEASE_VIEW_V1 = "model_release_view.v1.0"
MODEL_RELEASE_NOT_CONFIGURED = "MODEL_RELEASE_NOT_CONFIGURED"
MODEL_RELEASE_NOT_DEPLOYED = "MODEL_RELEASE_NOT_DEPLOYED"
MODEL_RELEASE_POINTER_UNRESOLVED = "MODEL_RELEASE_POINTER_UNRESOLVED"
MODEL_RELEASE_NOT_BOUND = "MODEL_RELEASE_NOT_BOUND"


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


def _resolve_current_id(config) -> str:
    """Read ``CURRENT`` once; the one place both ``/release/current`` routes trust it."""
    pointer = config.release_root / "CURRENT"
    if pointer.is_symlink() or not pointer.is_file():
        raise ValueError("unsafe current pointer")
    name = pointer.read_text().strip()
    if len(safe_relative_path(name)) != 1:
        raise ValueError("unsafe current pointer")
    return name


def _read_health(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("health artifact is indirect")
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or document.get("schema_version") != "operations_health.v1.0":
        raise ValueError("unsupported health artifact")
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def shell_document(*, frozen_at: str | None = None) -> bytes:
    """Small outer shell; legacy bytes are loaded inside its immutable frame.

    The frame's release id is resolved exactly once, from the authenticated
    ``/release/current.json`` route, when the shell first loads. Every later
    hashchange re-navigates the frame using that SAME pinned id; the shell
    never re-resolves ``current`` after the first load, so a ``CURRENT``
    pointer switch mid-session cannot move an already-open frame to a
    different release. ``/release/current/...`` keeps working for direct,
    non-shell requests (back-compat), but the shell itself never builds a
    frame URL from it.
    """
    frozen = frozen_at or "unknown"
    routes = ("trades/board", "trades/explorer", "trades/book", "models/modelx",
              "models/derivation", "models/health")
    views = "".join(f'<a href="/#/{route}">{view}</a> ' for view, route in zip(_VIEWS, routes))
    html = f'''<!doctype html><meta charset="utf-8"><title>Operations shell</title>
<style>body{{margin:0;font:14px sans-serif}}#ops{{padding:8px;background:#20252b;color:#eee}}#ops.unknown{{background:#634}}nav a{{margin-right:12px}}main{{min-height:90vh}}</style>
<div id="ops">health: <span id="state">unknown</span> <small id="stamp">offline frozen at {frozen}</small> <small id="release">release: resolving...</small></div>
<nav>{views}</nav><main><iframe id="legacy" title="legacy dashboard" src="about:blank" style="width:100%;height:90vh;border:0"></iframe></main>
<script>
const state=document.querySelector('#state'), stamp=document.querySelector('#stamp'), banner=document.querySelector('#ops'), releaseEl=document.querySelector('#release'), frame=document.querySelector('#legacy');
let pinned=null;
async function health(){{try{{const r=await fetch('/health.json',{{credentials:'same-origin'}});if(!r.ok)throw Error();const h=await r.json();const b=h.code_budgets||{{}};state.textContent=h.withheld_release?'withheld':(b.consecutive_nights?'degraded':'current');stamp.textContent='updated '+h.generated_at+'; failures '+(b.consecutive_nights||0);banner.className='';}}catch(e){{state.textContent='unknown / stale';stamp.textContent='offline frozen at {frozen}';banner.className='unknown';}}}}
health(); setInterval(health,30000);
async function resolveRelease(){{try{{const r=await fetch('/release/current.json',{{credentials:'same-origin'}});if(!r.ok)throw Error();const j=await r.json();return j.release_id;}}catch(e){{return null;}}}}
function route(){{if(!pinned)return;frame.src='/release/'+pinned+'/index.html'+(location.hash||'#/trades/board');}}
async function init(){{pinned=await resolveRelease();releaseEl.textContent=pinned?('release: '+pinned):'release: unavailable';route();}}
window.addEventListener('hashchange',route); init();
</script>'''
    return html.encode()


def _model_release_document(root: Path) -> tuple[HTTPStatus, dict]:
    """The body ``/models/release.json`` serves, and the status to send it with.

    Reads ``current_pointer`` exactly once; every field below is derived from
    THAT pointer's ``release_id``, so a promotion racing this call can never
    produce a body whose ``dependencies.previous_release_id`` names a
    different release than the top-level ``release_id`` itself -- the same
    "resolve once, reuse everywhere" rule ``_resolve_current_id`` documents
    for the board release pointer. No release ever deployed and an
    unresolvable pointer are distinct, explicit refusals -- never an empty
    success.

    Interim (bug C7): this document still describes the currently DEPLOYED
    model release, not the model release the served board was actually
    scored with -- no board release anywhere records which model release
    scored it (see the C7 investigation hand-back), so there is nothing to
    resolve that binding from yet. Real board binding lands with the Phase 6
    native-scoring publication slice. Until then every branch that names a
    release also carries ``board_binding: null`` and
    ``board_binding_reason: MODEL_RELEASE_NOT_BOUND`` so a reader never
    mistakes ``release_id``/``deployed_release_id`` (the DEPLOYED pointer)
    for what the board shows.
    """
    try:
        pointer = current_pointer(root)
    except (OSError, ValueError):
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "schema_version": MODEL_RELEASE_VIEW_V1, "status": "unavailable",
            "reason_code": "MODEL_RELEASE_POINTER_UNREADABLE", "release_id": None,
        }
    if pointer is None:
        return HTTPStatus.OK, {
            "schema_version": MODEL_RELEASE_VIEW_V1, "status": "refused",
            "reason_code": MODEL_RELEASE_NOT_DEPLOYED, "release_id": None,
        }
    try:
        release = resolve_release(root, pointer.release_id)
    except (ReleaseNotStaged, DeploymentError, OSError, ValueError):
        return HTTPStatus.SERVICE_UNAVAILABLE, {
            "schema_version": MODEL_RELEASE_VIEW_V1, "status": "unavailable",
            "reason_code": MODEL_RELEASE_POINTER_UNRESOLVED, "release_id": pointer.release_id,
            "board_binding": None, "board_binding_reason": MODEL_RELEASE_NOT_BOUND,
        }
    members = [
        {
            "binding_id": binding.binding_id,
            "model_id": binding.model_id,
            "role": binding.role,
            "strategy_id": binding.strategy_id,
            "decision_clock_id": binding.decision_clock_id,
            "adapter": binding.adapter,
            "artifacts": [
                {"name": member.name, "content_hash": member.content_hash}
                for member in binding.members
            ],
        }
        for binding in sorted(release.bindings, key=lambda binding: binding.binding_id)
    ]
    return HTTPStatus.OK, {
        "schema_version": MODEL_RELEASE_VIEW_V1, "status": "deployed",
        "release_id": release.release_id, "deployment_id": release.deployment_id,
        "deployed_release_id": release.release_id,
        "board_binding": None, "board_binding_reason": MODEL_RELEASE_NOT_BOUND,
        "members": members,
        "dependencies": {
            "previous_release_id": pointer.previous_release_id,
            "promotion_action": pointer.action,
            "promoted_at": pointer.at,
        },
    }


def model_release_page_document() -> bytes:
    """Small standalone page for the deployed model release (P6-4).

    Fetches ``/models/release.json`` client-side -- the SAME resolve-once
    document the JSON route serves -- and renders it as-is via
    ``textContent`` only (no ``innerHTML``, nothing here builds HTML out of
    server data). A refusal (``status`` other than ``"deployed"``) renders as
    an explicit message, never as an empty table.
    """
    html = '''<!doctype html><meta charset="utf-8"><title>Model release</title>
<style>body{margin:0;font:14px sans-serif;padding:16px}#summary.refused{color:#a33}
pre{background:#f4f4f4;padding:8px;overflow:auto}</style>
<h1>Deployed model release</h1>
<div id="summary">loading...</div>
<pre id="detail"></pre>
<script>
async function load(){
  const summary=document.querySelector('#summary'), detail=document.querySelector('#detail');
  try{
    const r=await fetch('/models/release.json',{credentials:'same-origin'});
    const j=await r.json();
    detail.textContent=JSON.stringify(j,null,2);
    if(j.status==='deployed'){
      summary.textContent='currently DEPLOYED release '+j.release_id+' (deployment '+j.deployment_id+') -- not necessarily what the board was scored with: '+(j.board_binding||('no board binding ('+j.board_binding_reason+')'));
      summary.className='';
    } else {
      summary.textContent='no release deployed: '+(j.reason_code||'UNKNOWN');
      summary.className='refused';
    }
  }catch(e){
    summary.textContent='model release unavailable';
    summary.className='refused';
  }
}
load();
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
        if path == "/models/release.json":
            return self._model_release_json_route(config)
        if path == "/models/release":
            return self._send(HTTPStatus.OK, model_release_page_document(), "text/html")
        if path.startswith("/actions/whatif/"):
            return self._whatif_result_route(config, path)
        if path == "/release/current.json":
            return self._current_json_route(config)
        if path == "/release/current":
            return self._current_route(config)
        if path.startswith("/release/"):
            return self._release_route(config, path)
        return self._send(HTTPStatus.NOT_FOUND, b"missing\n", "text/plain")

    def do_POST(self):  # noqa: N802
        config = self.server.config
        path = unquote(urlsplit(self.path).path)
        if path == "/actions/refresh":
            return self._refresh_route(config)
        if path == "/actions/whatif":
            return self._whatif_submit_route(config)
        return self._send(HTTPStatus.NOT_FOUND, b"missing\n", "text/plain")

    def _refresh_route(self, config):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        if config.submit_refresh is None:
            return self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"refresh not configured\n", "text/plain")
        payload, error = self._read_json_body()
        if error is not None:
            return self._send(HTTPStatus.BAD_REQUEST, error, "text/plain")
        try:
            status, body = config.submit_refresh(payload)
        except Exception:
            return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, b"refresh action failed\n", "text/plain")
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        return self._send(HTTPStatus(status), data, "application/json")

    def _whatif_submit_route(self, config):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        if config.submit_whatif is None:
            return self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"whatif not configured\n", "text/plain")
        payload, error = self._read_json_body(max_bytes=65536)
        if error is not None:
            return self._send(HTTPStatus.BAD_REQUEST, error, "text/plain")
        try:
            status, body = config.submit_whatif(payload)
        except Exception:
            return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, b"whatif action failed\n", "text/plain")
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        return self._send(HTTPStatus(status), data, "application/json")

    def _whatif_result_route(self, config, path):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        if config.fetch_whatif is None:
            return self._send(HTTPStatus.SERVICE_UNAVAILABLE, b"whatif not configured\n", "text/plain")
        job_id = path.removeprefix("/actions/whatif/")
        query = parse_qs(urlsplit(self.path).query)
        release_id = (query.get("release_id") or [None])[0]
        if not job_id or not release_id:
            return self._send(HTTPStatus.BAD_REQUEST, b"job id and release_id are required\n", "text/plain")
        try:
            current = _resolve_current_id(config)
        except (ArtifactError, OSError, ValueError):
            current = None
        if current is None or release_id != current:
            return self._send(HTTPStatus.CONFLICT, b"release_id is not the current release\n", "text/plain")
        try:
            status, body = config.fetch_whatif(job_id)
        except Exception:
            return self._send(HTTPStatus.INTERNAL_SERVER_ERROR, b"whatif result fetch failed\n", "text/plain")
        data = json.dumps(body, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        return self._send(HTTPStatus(status), data, "application/json")

    def _read_json_body(self, *, max_bytes=4096):
        """Returns (payload_dict, None) or (None, error_body_bytes). Bounded."""
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, b"invalid content-length\n"
        if length <= 0 or length > max_bytes:
            return None, b"request body required and bounded to 4096 bytes\n"
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None, b"invalid json body\n"
        if not isinstance(payload, dict):
            return None, b"json body must be an object\n"
        return payload, None

    def _current_route(self, config):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        try:
            name = _resolve_current_id(config)
            return self._send(HTTPStatus.FOUND, b"", "text/plain",
                              {"Location": "/release/" + name + "/index.html"})
        except (ArtifactError, OSError, ValueError):
            return self._send(HTTPStatus.NOT_FOUND, b"unknown release\n", "text/plain")

    def _current_json_route(self, config):
        """Resolve-once source for the shell: the pinned id as plain JSON.

        Never a redirect, so a fetch()ing shell gets the id itself instead of
        a Location header to follow — the id is then reused for every later
        frame navigation instead of re-resolving ``current``.
        """
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        try:
            name = _resolve_current_id(config)
            body = json.dumps({"release_id": name}, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            return self._send(HTTPStatus.OK, body, "application/json")
        except (ArtifactError, OSError, ValueError):
            return self._send(HTTPStatus.NOT_FOUND, b"unknown release\n", "text/plain")

    def _model_release_json_route(self, config):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        if config.model_release_root is None:
            document = {"schema_version": MODEL_RELEASE_VIEW_V1, "status": "refused",
                       "reason_code": MODEL_RELEASE_NOT_CONFIGURED, "release_id": None}
            status = HTTPStatus.OK
        else:
            status, document = _model_release_document(Path(config.model_release_root))
        body = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        return self._send(status, body, "application/json")

    def _release_route(self, config, path):
        if not self._authorized():
            return self._send(HTTPStatus.UNAUTHORIZED, b"unauthorized\n", "text/plain")
        rel = path.removeprefix("/release/").split("/", 1)
        if len(rel) != 2 or not rel[0] or not rel[1]:
            return self._send(HTTPStatus.NOT_FOUND, b"missing\n", "text/plain")
        try:
            if rel[0] == "current":
                rel[0] = _resolve_current_id(config)
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
                  frozen_at: str = "unknown", submit_refresh=None, submit_whatif=None,
                  fetch_whatif=None, model_release_root: Path | str | None = None):
    config = type("Config", (), {"token": token, "health_path": Path(health_path),
                                  "release_root": Path(release_root), "frozen_at": frozen_at,
                                  "submit_refresh": submit_refresh, "submit_whatif": submit_whatif,
                                  "fetch_whatif": fetch_whatif,
                                  "model_release_root": (Path(model_release_root)
                                                          if model_release_root is not None
                                                          else None)})
    server = http.server.ThreadingHTTPServer(address, OperationsHandler)
    server.config = config
    return server
