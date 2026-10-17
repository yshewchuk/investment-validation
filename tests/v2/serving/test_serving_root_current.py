from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from engine.v2.serving import projections
from engine.v2.serving import api as api_module
from engine.v2.serving.api import create_app, main
from tests.v2.serving.release_support import build_releases

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
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, f"http://127.0.0.1:{port}"


def _stop(server, thread):
    server.should_exit = True
    thread.join()


def _get(base, path, token, headers=None):
    request = urllib.request.Request(base + path, headers=headers or {})
    request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
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
    repo, snap, serving_conn, store, release_a, release_b = build_releases(tmp_path)
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


def _bypass_prechecks(monkeypatch):
    """Make the ``_read_small_text`` guards the only thing refusing a file.

    The publication prechecks reject a symlink or a non-regular file before the
    bounded read ever runs. The server runs in in-process threads, so patching
    ``os.path`` here is seen by the request handler.
    """
    monkeypatch.setattr(os.path, "islink", lambda path: False)
    monkeypatch.setattr(os.path, "isfile", lambda path: True)


def _nonregular(path, kind, text):
    """Replace ``path`` with a symlink (to a file holding ``text``), a FIFO, or a dir."""
    path.unlink(missing_ok=True)
    if kind == "symlink":
        real = path.with_name(path.name + ".real")
        real.write_text(text)
        path.symlink_to(real)
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()


def _expect_no_current(client, h):
    r = client.get("/api/v1/releases/current", headers=h)
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "NO_CURRENT_RELEASE"
    assert body["retryable"] is True


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


def test_plain_resolver_scope_is_null(tmp_path):
    _, release_a, _, root, _ = _setup(tmp_path)
    app = create_app(
        resolver=lambda: release_a.release_id,
        serving_db=str(root / "serving.sqlite"),
        store_root=str(root / "objects"),
        serving_root=str(root),
        token=TOKEN,
    )
    h = {"Authorization": f"Bearer {TOKEN}"}
    r = _Client(app).get("/api/v1/releases/current", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["release_id"] == release_a.release_id
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


_STATUS_BASE = {"schema_version": "operations_status.v1.0", "release_id": "r1"}
_TEXT_CAP = 1 << 20


def _padded_status(size):
    """A valid operations-status document whose serialized UTF-8 form is
    exactly ``size`` bytes."""
    document = dict(_STATUS_BASE)
    document["pad"] = ""
    document["pad"] = "x" * (size - len(json.dumps(document)))
    text = json.dumps(document)
    assert len(text.encode("utf-8")) == size
    return document, text


def test_operations_oversized_status_file(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    document, text = _padded_status(_TEXT_CAP + 1)
    (pub / "operations_status.json").write_text(text)
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "OPERATIONS_UNAVAILABLE"
    assert r.json()["retryable"] is True


def test_operations_exactly_cap_status_file(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    document, text = _padded_status(_TEXT_CAP)
    (pub / "operations_status.json").write_text(text)
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 200
    assert r.json() == document


def test_current_symlinked_pointer_is_absent(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    real = pub / "real_current"
    real.write_text("ops1")
    (pub / "CURRENT").unlink()
    (pub / "CURRENT").symlink_to(real)
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_current_oversized_binding_is_absent(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    binding_path = pub / "releases" / "ops1" / "projection_binding.json"
    document = {"projection_release_id": release_a.release_id, "pad": ""}
    document["pad"] = "x" * (_TEXT_CAP + 1 - len(json.dumps(document)))
    text = json.dumps(document)
    assert len(text.encode("utf-8")) == _TEXT_CAP + 1
    binding_path.write_text(text)
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_oversize_current_pointer(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    (pub / "CURRENT").write_bytes(b"ops1" + b" " * (_TEXT_CAP + 1 - 4))
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_current_pointer_at_cap_is_accepted(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    (pub / "CURRENT").write_bytes(b"ops1" + b" " * (_TEXT_CAP - 4))
    client, h = _client(root, pub)
    r = client.get("/api/v1/releases/current", headers=h)
    assert r.status_code == 200
    assert r.json()["release_id"] == release_a.release_id


def test_non_utf8_current_pointer(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    (pub / "CURRENT").write_bytes(b"\xff\xfe")
    client, h = _client(root, pub)
    _expect_no_current(client, h)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_current_nonregular_pointer_is_absent(tmp_path, monkeypatch, kind):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    _bypass_prechecks(monkeypatch)
    _nonregular(pub / "CURRENT", kind, "ops1")
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_current_directory_pointer_is_absent(tmp_path, monkeypatch):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    _bypass_prechecks(monkeypatch)
    _nonregular(pub / "CURRENT", "directory", "ops1")
    client, h = _client(root, pub)
    _expect_no_current(client, h)


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_binding_nonregular_is_absent(tmp_path, monkeypatch, kind):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    _bypass_prechecks(monkeypatch)
    binding = projections.projection_binding(serving_conn, release_a.release_id)
    _nonregular(pub / "releases" / "ops1" / "projection_binding.json", kind, json.dumps(binding))
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_binding_non_utf8_is_absent(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    (pub / "releases" / "ops1" / "projection_binding.json").write_bytes(b"\xff\xfe")
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_fifo_current_refused_by_fstat_alone(tmp_path, monkeypatch):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    _bypass_prechecks(monkeypatch)
    current = pub / "CURRENT"
    current.unlink()
    os.mkfifo(current)
    fd = os.open(current, os.O_RDWR | os.O_NONBLOCK)
    rfd = None
    try:
        os.write(fd, b"ops1")
        rfd = os.open(current, os.O_RDONLY | os.O_NONBLOCK)
        os.close(fd)
        fd = None
        client, h = _client(root, pub)
        _expect_no_current(client, h)
    finally:
        if fd is not None:
            os.close(fd)
        if rfd is not None:
            os.close(rfd)


def _deep_nesting_text():
    return "[" * 100000 + "]" * 100000


def test_deeply_nested_binding_reads_absent(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    binding_path = pub / "releases" / "ops1" / "projection_binding.json"
    binding_path.write_text(_deep_nesting_text())
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_deeply_nested_status_reads_absent(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "operations_status.json").write_text(_deep_nesting_text())
    client, h = _client(pub.parent, pub)
    _expect_operations_unavailable(client, h)


def test_non_utf8_current_refused_by_strict_decode(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    binding = projections.projection_binding(serving_conn, release_a.release_id)
    lossy = pub / "releases" / "ops1\ufffd"
    lossy.mkdir(parents=True, exist_ok=True)
    (lossy / "projection_binding.json").write_text(json.dumps(binding))
    (pub / "CURRENT").write_bytes(b"ops1\xff")
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_current_regular_mode_755_resolves(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    os.chmod(pub / "CURRENT", 0o755)
    client, h = _client(root, pub)
    r = client.get("/api/v1/releases/current", headers=h)
    assert r.status_code == 200
    assert r.json()["release_id"] == release_a.release_id


def test_current_read_failure(tmp_path, monkeypatch):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    real_read = os.read

    def failing_read(fd, count):
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            target = ""
        if target.endswith("/CURRENT"):
            raise OSError(5, "EIO")
        return real_read(fd, count)

    monkeypatch.setattr(os, "read", failing_read)
    client, h = _client(root, pub)
    _expect_no_current(client, h)


def test_current_pointer_no_fd_leak(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a)
    client, h = _client(root, pub)
    client.get("/api/v1/releases/current", headers=h)
    before = len(os.listdir("/proc/self/fd"))
    for _ in range(20):
        client.get("/api/v1/releases/current", headers=h)
    after = len(os.listdir("/proc/self/fd"))
    assert after - before <= 3


@pytest.mark.parametrize("flag", ["serving-root", "serving-db", "store-root",
                                   "publication-root", "ops-root",
                                   "native-parity-report-path"])
def test_main_empty_path_flag(monkeypatch, capsys, flag):
    monkeypatch.setenv("V2_DASHBOARD_TOKEN", TOKEN)
    calls = []
    monkeypatch.setattr(api_module.uvicorn, "run", lambda app, **kw: calls.append(app))
    argv = ["--port", "1", "--serving-root", ""] if flag == "serving-root" else [
        "--port", "1", "--serving-root", "/r", f"--{flag}", ""]
    assert main(argv) == 2
    assert calls == []
    assert f"--{flag} must not be empty" in capsys.readouterr().out


def _expect_operations_unavailable(client, h):
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "OPERATIONS_UNAVAILABLE"
    assert body["retryable"] is True


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_operations_nonregular_status_file(tmp_path, monkeypatch, kind):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    _bypass_prechecks(monkeypatch)
    _nonregular(pub / "operations_status.json", kind, json.dumps(_STATUS_BASE))
    client, h = _client(pub.parent, pub)
    _expect_operations_unavailable(client, h)


def test_operations_non_utf8_status_file(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "operations_status.json").write_bytes(b"\xff\xfe")
    client, h = _client(pub.parent, pub)
    _expect_operations_unavailable(client, h)
