from __future__ import annotations

import json
import os
from types import SimpleNamespace

from fastapi.testclient import TestClient

from engine.v2.serving import projections
from engine.v2.serving.api import create_app, _serving_paths
from tests.v2.serving.test_v2_serving_publication_binding import _releases

TOKEN = "t-serving-root"


def _setup(tmp_path, scope="__absent__"):
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
    return TestClient(create_app(
        serving_db=serving_db or str(root / "serving.sqlite"),
        store_root=str(root / "objects"),
        serving_root=str(root),
        publication_root=str(pub),
        token=TOKEN,
    )), {"Authorization": f"Bearer {TOKEN}"}


def test_scope_passthrough(tmp_path):
    serving_conn, release_a, _, root, pub = _setup(tmp_path)
    _point(serving_conn, pub, release_a, scope="shadow")
    client, h = _client(root, pub)
    assert client.get("/api/v1/releases/current", headers=h).json()["scope"] == "shadow"

    _point(serving_conn, pub, release_a)
    app = create_app(serving_db=str(root / "serving.sqlite"), store_root=str(root / "objects"),
                     serving_root=str(root), publication_root=str(pub), token=TOKEN)
    client = TestClient(app)
    body = client.get("/api/v1/releases/current", headers=h).json()
    assert "scope" in body and body["scope"] is None

    _point(serving_conn, pub, release_a, scope=5)
    client, h = _client(root, pub)
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


def test_serving_paths():
    ns = SimpleNamespace(serving_root="/r", serving_db=None, store_root=None, publication_root=None)
    assert _serving_paths(ns) == (os.path.join("/r", "serving.sqlite"),
                                  os.path.join("/r", "objects"),
                                  os.path.join("/r", "publisher"))
    ns = SimpleNamespace(serving_root="/r", serving_db="/x.db", store_root="/o", publication_root="/p")
    assert _serving_paths(ns) == (os.path.join("/x.db"), os.path.join("/o"), os.path.join("/p"))


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


def test_operations(tmp_path):
    _, _, _, _, pub = _setup(tmp_path)
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 503
    assert r.json()["code"] == "OPERATIONS_UNAVAILABLE"

    status = {"schema_version": "operations_status.v1.0", "release_id": "r1"}
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "operations_status.json").write_text(json.dumps(status))
    client, h = _client(pub.parent, pub)
    r = client.get("/api/v1/operations", headers=h)
    assert r.status_code == 200
    assert r.json() == status