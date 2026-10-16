from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request

import pytest
import uvicorn

from engine.v2.serving import projections
from engine.v2.serving import api as api_module
from engine.v2.serving.api import create_app, main
from tests.v2.serving.test_v2_serving_publication_binding import _releases  # existing fixture builder shared with the publication-binding tests

TOKEN = "t-serving-root"


class _Resp:
    def __init__(self, status, body, headers):
        self.status_code, self.text, self.headers = status, body.decode(), headers

    def json(self):
        return json.loads(self.text)


def _start(app):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def _stop(server, thread):
    server.should_exit = True
    thread.join()


def _get(base, path, token, headers=None):
    request = urllib.request.Request(base + path, headers=headers or {})
    request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), exc.headers


class _Client:
    """Real uvicorn server over an app; stopped by the autouse fixture."""

    def __init__(self, app):
        self.server, self.thread, self.base = _start(app)
        _SERVERS.append((self.server, self.thread))

    def get(self, path, headers=None):
        extra = {k: v for k, v in (headers or {}).items() if k != "Authorization"}
        return _Resp(*_get(self.base, path, token=TOKEN, headers=extra))


_SERVERS = []


@pytest.fixture(autouse=True)
def _stop_servers():
    yield
    while _SERVERS:
        _stop(*_SERVERS.pop())


def _setup(tmp_path):
    repo, snap, serving_conn, store, release_a, release_b = _releases(tmp_path)
    root = tmp_path / "serving"
    pub = root / "publisher"
    return serving_conn, release_a, release_b, root, pub


def _point(serving_conn, pub, release, scope="__absent__"):
    binding = projections.projection_binding(serving_conn, release.release_id)
    if scope != "__absent__":
        binding["scope"] = scope
    d = pub / "releases" / "ops1"
    d.mkdir(parents=True, exist_ok=True)
    (d / "projection_binding.json").write_text(json.dumps(binding))
    (pub / "CURRENT").write_text("ops1")


def _client(root, pub, serving_db=None):
    return _Client(create_app(
        serving_db=serving_db or str(root / "serving.sqlite"),
        store_root=str(root / "objects"),
        serving_root=str(root),
        publication_root=str(pub),
        token=TOKEN,
    )), {"Authorization": f"Bearer {TOKEN}"}


def _capture_app(monkeypatch, argv):
    apps, calls = [], []
    real_create_app = api_module.create_app

    def wrapper(*args, **kwargs):
        calls.append(kwargs)
        return real_create_app(*args, **kwargs)

    monkeypatch.setattr(api_module, "create_app", wrapper)
    monkeypatch.setattr(api_module.uvicorn, "run", lambda app, **kw: apps.append(app))
    assert main(argv) == 0
    return apps[0], calls[0]


def test_main_default_paths(monkeypatch):
    monkeypatch.setenv("V2_DASHBOARD_TOKEN", TOKEN)
    app, kwargs = _capture_app(monkeypatch, ["--port", "1", "--serving-root", "/r"])
    assert kwargs["serving_db"] == os.path.join("/r", "serving.sqlite")
    assert kwargs["store_root"] == os.path.join("/r", "objects")
    assert kwargs["publication_root"] == os.path.join("/r", "publisher")
    assert app is not None


def test_main_explicit_paths(monkeypatch):
    monkeypatch.setenv("V2_DASHBOARD_TOKEN", TOKEN)
    app, kwargs = _capture_app(monkeypatch, ["--port", "1", "--serving-root", "/r",
                                             "--serving-db", "/x.db", "--store-root", "/o",
                                             "--publication-root", "/p"])
    assert kwargs["serving_db"] == "/x.db"
    assert kwargs["store_root"] == "/o"
    assert kwargs["publication_root"] == "/p"
    assert app is not None


def test_scope_passthrough(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a, scope="shadow")
    client, h = _client(root, pub)
    assert client.get("/api/v1/releases/current", headers=h).json()["scope"] == "shadow"

    _point(serving_conn, pub, release_a)
    body = client.get("/api/v1/releases/current", headers=h).json()
    assert "scope" in body and body["scope"] is None

    _point(serving_conn, pub, release_a, scope=5)
    body = client.get("/api/v1/releases/current", headers=h).json()
    assert "scope" in body and body["scope"] is None


def test_etag(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    client, h = _client(root, pub)
    e1 = client.get("/api/v1/releases/current", headers=h).headers["ETag"]

    _point(serving_conn, pub, release_a, scope="shadow")
    client, h = _client(root, pub)
    e2 = client.get("/api/v1/releases/current", headers=h).headers["ETag"]
    assert e2 != e1
    assert client.get("/api/v1/releases/current", headers=h).headers["ETag"] == e2
    r = client.get("/api/v1/releases/current", headers={**h, "If-None-Match": e2})
    assert r.status_code == 304


def test_release_has_no_scope(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a, scope="shadow")
    client, h = _client(root, pub)
    assert "scope" not in client.get(f"/api/v1/releases/{release_a.release_id}", headers=h).json()


def test_main_serving_root_only(tmp_path, monkeypatch):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a, scope="shadow")
    monkeypatch.setenv("V2_DASHBOARD_TOKEN", TOKEN)

    app, _ = _capture_app(monkeypatch, ["--port", "1", "--serving-root", str(root)])
    client = _Client(app)
    h = {"Authorization": f"Bearer {TOKEN}"}
    r = client.get("/api/v1/releases/current", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["release_id"] == release_a.release_id
    assert body["scope"] == "shadow"

    other = tmp_path / "empty"
    other.mkdir()
    app, _ = _capture_app(monkeypatch, ["--port", "1", "--serving-root", str(root),
                                        "--publication-root", str(other)])
    r = _Client(app).get("/api/v1/releases/current", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "NO_CURRENT_RELEASE"


def test_open_time_failure(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    client, h = _client(root, pub, serving_db=str(tmp_path))
    r = client.get("/api/v1/releases/current", headers=h)
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "SERVING_INDEX_UNREADABLE"
    assert body["retryable"] is True
    assert str(tmp_path) not in r.text

    r = client.get(f"/api/v1/releases/{release_a.release_id}", headers=h)
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "SERVING_INDEX_UNREADABLE"
    assert body["retryable"] is True


def test_operations(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "OPERATIONS_UNAVAILABLE"
    assert r.json()["retryable"] is True

    status = {"schema_version": "operations_status.v1.0", "release_id": "r1"}
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "operations_status.json").write_text(json.dumps(status))
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 200
    assert r.json() == status


def test_operations_wrong_schema_version(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "operations_status.json").write_text(
        json.dumps({"schema_version": "operations_status.v0", "release_id": "r1"}))
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "OPERATIONS_UNAVAILABLE"
    assert r.json()["retryable"] is True


def test_operations_symlinked_status_file(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    real = pub / "real_status.json"
    real.write_text(json.dumps({"schema_version": "operations_status.v1.0", "release_id": "r1"}))
    (pub / "operations_status.json").symlink_to(real)
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "OPERATIONS_UNAVAILABLE"


def test_operations_oversized_status_file(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "operations_status.json").write_bytes(b"x" * 1048577)
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "OPERATIONS_UNAVAILABLE"
