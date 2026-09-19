"""HTTP-level controls for the independent operations health/release surface."""
from __future__ import annotations

import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from engine.v2.serving.operations import create_server


def _get(url, token=None):
    request = Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request)


def _post(url, body, token=None, cookie=None):
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    request = Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if cookie:
        request.add_header("Cookie", cookie)
    return urlopen(request)


def _serve_refresh(tmp_path, **kwargs):
    health = tmp_path / "health.json"
    health.write_text('{"schema_version":"operations_health.v1.0"}')
    (tmp_path / "releases").mkdir(exist_ok=True)
    server = create_server(("127.0.0.1", 0), token="secret", health_path=health,
                           release_root=tmp_path, **kwargs)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, "http://127.0.0.1:" + str(server.server_port)


def test_health_auth_current_release_and_immutable_shell(tmp_path):
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                 "generated_at": "t1", "withheld_release": None}))
    releases = tmp_path / "releases" / "r1"
    releases.mkdir(parents=True)
    board = releases / "index.html"
    board.write_bytes(b"old-board")
    (tmp_path / "CURRENT").write_text("r1\n")
    server = create_server(("127.0.0.1", 0), token="secret", health_path=health,
                           release_root=tmp_path, frozen_at="t0")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/health.json")
        assert error.value.code == 401
        assert json.loads(_get(base + "/health.json", "secret").read())['generated_at'] == "t1"
        assert _get(base + "/release/current", "secret").read() == b"old-board"
        assert _get(base + "/release/r1/index.html", "secret").read() == b"old-board"
        assert _get(base + "/board").read().startswith(b"<!doctype html>")
        health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                      "generated_at": "t2", "withheld_release": "r2"}))
        assert json.loads(_get(base + "/health.json", "secret").read())['generated_at'] == "t2"
        assert board.read_bytes() == b"old-board"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_release_traversal_and_symlink_are_denied(tmp_path):
    health = tmp_path / "health.json"
    health.write_text('{"schema_version":"operations_health.v1.0"}')
    (tmp_path / "releases").mkdir()
    server = create_server(("127.0.0.1", 0), token="secret", health_path=health,
                           release_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/r1/../../etc/passwd", "secret")
        assert error.value.code in (400, 404)
        outside = tmp_path / "outside"
        outside.write_bytes(b"secret")
        (tmp_path / "releases" / "r1").symlink_to(tmp_path)
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/r1/outside", "secret")
        assert error.value.code == 404
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_refresh_requires_auth(tmp_path):
    server, thread, base = _serve_refresh(
        tmp_path, submit_refresh=lambda payload: (202, {"jobs": []}))
    try:
        with pytest.raises(HTTPError) as error:
            _post(base + "/actions/refresh", {"plan_ref": "p1"})
        assert error.value.code == 401
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_refresh_rejects_bad_token(tmp_path):
    server, thread, base = _serve_refresh(
        tmp_path, submit_refresh=lambda payload: (202, {"jobs": []}))
    try:
        with pytest.raises(HTTPError) as error:
            _post(base + "/actions/refresh", {"plan_ref": "p1"}, token="wrong")
        assert error.value.code == 401
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_refresh_accepts_cookie_token(tmp_path):
    server, thread, base = _serve_refresh(
        tmp_path, submit_refresh=lambda payload: (202, {"jobs": [], "received": payload}))
    try:
        response = _post(base + "/actions/refresh", {"plan_ref": "p1"},
                         cookie="operations_token=secret")
        assert response.status == 202
        assert json.loads(response.read()) == {"jobs": [], "received": {"plan_ref": "p1"}}
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_refresh_not_configured(tmp_path):
    server, thread, base = _serve_refresh(tmp_path)
    try:
        with pytest.raises(HTTPError) as error:
            _post(base + "/actions/refresh", {"plan_ref": "p1"}, token="secret")
        assert error.value.code == 503
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_refresh_rejects_oversized_or_malformed_body(tmp_path):
    server, thread, base = _serve_refresh(
        tmp_path, submit_refresh=lambda payload: (202, {"jobs": []}))
    try:
        for body in (b"not json", b"[1, 2, 3]", b"x" * 5000):
            with pytest.raises(HTTPError) as error:
                _post(base + "/actions/refresh", body, token="secret")
            assert error.value.code == 400
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_refresh_forwards_stub_status(tmp_path):
    failure = {"error": "plan_ref must reference a nightly plan"}
    server, thread, base = _serve_refresh(
        tmp_path, submit_refresh=lambda payload: (400, failure))
    try:
        with pytest.raises(HTTPError) as error:
            _post(base + "/actions/refresh", {"plan_ref": "p1"}, token="secret")
        assert error.value.code == 400
        assert json.loads(error.value.read()) == failure
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_whatif_requires_auth(tmp_path):
    server, thread, base = _serve_refresh(
        tmp_path, submit_whatif=lambda payload: (202, {"job_id": "job_x"}))
    try:
        with pytest.raises(HTTPError) as error:
            _post(base + "/actions/whatif", {"request": {}, "native_inputs": {}})
        assert error.value.code == 401
    finally:
        server.shutdown(); thread.join(timeout=2); server.server_close()


def test_whatif_submit_forwards_to_stub(tmp_path):
    server, thread, base = _serve_refresh(
        tmp_path, submit_whatif=lambda payload: (202, {"job_id": "job_x", "seen": payload}))
    try:
        response = _post(base + "/actions/whatif", {"request": {"a": 1}, "native_inputs": {"b": 2}},
                         token="secret")
        assert response.status == 202
        body = json.loads(response.read())
        assert body["job_id"] == "job_x"
        assert body["seen"] == {"request": {"a": 1}, "native_inputs": {"b": 2}}
    finally:
        server.shutdown(); thread.join(timeout=2); server.server_close()


def test_whatif_result_requires_matching_release_id(tmp_path):
    (tmp_path / "CURRENT").write_text("r1")
    (tmp_path / "releases" / "r1").mkdir(parents=True, exist_ok=True)
    server, thread, base = _serve_refresh(
        tmp_path, fetch_whatif=lambda job_id: (200, {"job_id": job_id}))
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/actions/whatif/job_x?release_id=wrong", token="secret")
        assert error.value.code == 409
        response = _get(base + "/actions/whatif/job_x?release_id=r1", token="secret")
        assert response.status == 200
        assert json.loads(response.read()) == {"job_id": "job_x"}
    finally:
        server.shutdown(); thread.join(timeout=2); server.server_close()


def test_whatif_result_not_configured(tmp_path):
    server, thread, base = _serve_refresh(tmp_path)
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/actions/whatif/job_x?release_id=r1", token="secret")
        assert error.value.code == 503
    finally:
        server.shutdown(); thread.join(timeout=2); server.server_close()
