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
