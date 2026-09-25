"""S9H: ``/analogs.json`` reachable through the real launcher (native history analogs).

Drives ``preview.run`` on a loopback ephemeral port over a serving index
seeded through the real ``projections.build_candidate`` path (so the v3
migration and ``_score_summary_fields`` wiring are exercised for real), then
reads the native history-analogs document back over HTTP. One boundary covers
argv parsing, ``_server.build_server``'s ``create_server`` hand-off, the
operations route and ``analog_projection`` together. Negative controls: the
route is auth-gated exactly like ``/health.json`` (no Authorization
header/cookie is 401, a wrong token is 401); an unknown event is a 404
``EVENT_NOT_FOUND``; a missing query parameter is a 400; and an unconfigured
index keeps the explicit 503 refusal.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from engine.v2.dashboard import preview
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore
from engine.v2.serving import projections
from tests.test_v2_serving_projections import (
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
)

TOKEN = "analogs-launcher-secret"


def _dashboard_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    release_dir = bundle / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_text("<!doctype html><title>legacy</title>")
    (bundle / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0"}))
    return bundle, health


def _committed_release(tmp_path):
    """One real serving root holding a committed candidate with analog row ids."""
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2", [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_root = tmp_path / "serving"
    serving_root.mkdir()
    serving_store = ArtifactStore(serving_root / "objects")
    serving_conn = projections.connect(str(serving_root / "serving.sqlite"))
    row = _row(n_analogs=2, exp_pnl_analog=0.08, win_analog=0.6)
    display = _compact(row, selected_row_ids=["row-a", "row-b"],
                       contributing_row_ids=["row-a"])
    release = projections.build_candidate(
        _preview_input(), _score_doc(rows=[row]), _bundle(display),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")
    serving_conn.close()
    return serving_root, release


def _run_launcher(tmp_path, monkeypatch, *, serving_index=True):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, TOKEN)
    bundle, health = _dashboard_bundle(tmp_path)
    serving_root, release = _committed_release(tmp_path)
    argv = ["--host", "127.0.0.1", "--port", "0",
            "--release-root", str(bundle), "--health-path", str(health)]
    if serving_index:
        argv += ["--serving-index-path", str(serving_root / "serving.sqlite")]
    server, thread, release_id = preview.run(argv)
    return server, thread, release_id, release


def _analogs_url(server, *, release_id=None, event_id=None) -> str:
    params = {"release_id": release_id, "event_id": event_id}
    clean = {key: value for key, value in params.items() if value is not None}
    return f"http://127.0.0.1:{server.server_port}/analogs.json?" + urlencode(clean)


def _get_analogs(server, *, release_id=None, event_id=None, token=TOKEN):
    request = Request(_analogs_url(server, release_id=release_id, event_id=event_id))
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request, timeout=5)


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_analogs_json_reachable_via_preview_run(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(tmp_path, monkeypatch)
    try:
        assert release_id == "r1"  # the rest of the launcher still starts and pins normally
        response = _get_analogs(server, release_id=release.release_id, event_id="e1")
        assert response.status == 200
        body = json.loads(response.read())
        assert body["schema_version"] == "history_analogs_view.v1"
        assert body["status"] == "available"
        assert body["release_id"] == release.release_id
        assert body["event_id"] == "e1"
        assert body["n_analogs"] == 2
        assert body["selected_row_ids"] == ["row-a", "row-b"]
        assert body["contributing_row_ids"] == ["row-a"]
        assert all(isinstance(row_id, str)
                   for row_id in body["selected_row_ids"] + body["contributing_row_ids"])
        assert TOKEN not in json.dumps(body)

        for bad in (None, "wrong-token"):
            with pytest.raises(HTTPError) as error:
                _get_analogs(server, release_id=release.release_id, event_id="e1", token=bad)
            assert error.value.code == 401
    finally:
        _stop(server, thread)


def test_analogs_json_unknown_event_is_a_named_refusal(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(tmp_path, monkeypatch)
    try:
        with pytest.raises(HTTPError) as error:
            _get_analogs(server, release_id=release.release_id, event_id="nope")
        assert error.value.code == 404
        body = json.loads(error.value.read())
        assert body["schema_version"] == "history_analogs_view.v1"
        assert body["status"] == "refused"
        assert body["reason_code"] == "EVENT_NOT_FOUND"
    finally:
        _stop(server, thread)


def test_analogs_json_missing_event_id_is_a_bad_request(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(tmp_path, monkeypatch)
    try:
        with pytest.raises(HTTPError) as error:
            _get_analogs(server, release_id=release.release_id)
        assert error.value.code == 400
    finally:
        _stop(server, thread)


def test_analogs_json_without_a_configured_index_is_a_503(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(tmp_path, monkeypatch, serving_index=False)
    try:
        with pytest.raises(HTTPError) as error:
            _get_analogs(server, release_id=release.release_id, event_id="e1")
        assert error.value.code == 503
    finally:
        _stop(server, thread)


def test_shell_nav_links_to_the_analogs_view(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(tmp_path, monkeypatch)
    try:
        shell = urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5).read().decode()
        assert '<a href="/analogs">analogs</a>' in shell

        page = urlopen(f"http://127.0.0.1:{server.server_port}/analogs", timeout=5)
        assert page.status == 200
        assert b"History analogs" in page.read()
    finally:
        _stop(server, thread)
