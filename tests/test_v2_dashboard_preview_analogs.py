"""S9H: ``/analogs.json`` reachable through the real launcher (native history analogs).

Drives ``preview.run`` on a loopback ephemeral port over a serving index
seeded through the real ``projections.build_candidate`` path (so the v3
migration and ``_score_summary_fields`` wiring are exercised for real), then
reads the native history-analogs document back over HTTP. One boundary covers
argv parsing, ``_server.build_server``'s ``create_server`` hand-off, the
operations route and ``analog_projection`` together. Negative controls: the
route is auth-gated exactly like ``/health.json`` (no Authorization
header/cookie is 401, a wrong token is 401); an unknown event is a 404
``EVENT_NOT_FOUND``; a missing query parameter is a 400; an unconfigured
index keeps the explicit 503 refusal; and a GET never migrates: a missing
index file is a typed 503 ``SERVING_INDEX_MISSING`` and an unmigrated index
is a typed 503 ``SERVING_INDEX_OUTDATED`` that stays at its old schema.
"""
from __future__ import annotations

import json
import sqlite3
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
    # Fixture-injected ids are deliberate here: this test covers the HTTP route
    # over a synthetic index. The real scorer -> renderer -> bridge -> index
    # path that supplies the ids is covered end to end by
    # tests/test_v2_serving_analog_ids.py.
    display = _compact(row, selected_row_ids=["row-a", "row-b"],
                       contributing_row_ids=["row-a"])
    release = projections.build_candidate(
        _preview_input(), _score_doc(rows=[row]), _bundle(display),
        repository=repo, snapshot_ref=snap, store=serving_store, conn=serving_conn,
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")
    serving_conn.close()
    return serving_root, release


def _outdated_index(tmp_path: Path) -> Path:
    """A serving index stopped at migration 1: no analog columns yet."""
    path = tmp_path / "outdated.sqlite"
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        conn.execute(projections._SCHEMA_VERSIONS_DDL)
        for statement in projections._V1:
            conn.execute(statement)
        version, name, statements = projections._MIGRATIONS[0]
        conn.execute(
            "INSERT INTO schema_versions (owner, version, name, checksum, applied_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (projections._OWNER, version, name,
             projections._checksum(version, name, statements), "2024-01-01T00:00:00Z"))
    finally:
        conn.close()
    return path


def _run_launcher(tmp_path, monkeypatch, *, serving_index=True):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, TOKEN)
    bundle, health = _dashboard_bundle(tmp_path)
    serving_root, release = _committed_release(tmp_path)
    argv = ["--host", "127.0.0.1", "--port", "0",
            "--release-root", str(bundle), "--health-path", str(health)]
    if serving_index is True:
        argv += ["--serving-index-path", str(serving_root / "serving.sqlite")]
    elif serving_index is not False:
        argv += ["--serving-index-path", str(serving_index)]
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
        assert len(body["analogs"]) == 1
        entry = body["analogs"][0]
        assert entry["strategy"] == "STR-THRU"
        assert entry["n_analogs"] == 2
        assert entry["selected_row_ids"] == ["row-a", "row-b"]
        assert entry["contributing_row_ids"] == ["row-a"]
        assert isinstance(entry["score_id"], str) and entry["score_id"]
        assert all(isinstance(row_id, str)
                   for row_id in entry["selected_row_ids"] + entry["contributing_row_ids"])
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


def test_analogs_json_missing_index_file_is_a_typed_503(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(
        tmp_path, monkeypatch, serving_index=tmp_path / "missing.sqlite")
    try:
        with pytest.raises(HTTPError) as error:
            _get_analogs(server, release_id=release.release_id, event_id="e1")
        assert error.value.code == 503
        body = json.loads(error.value.read())
        assert body["status"] == "unavailable"
        assert body["reason_code"] == "SERVING_INDEX_MISSING"
    finally:
        _stop(server, thread)


def test_analogs_json_outdated_index_is_a_typed_503_and_is_never_migrated(tmp_path, monkeypatch):
    """A GET must not run the write path: the index stays at migration 1."""
    path = _outdated_index(tmp_path)
    server, thread, release_id, release = _run_launcher(
        tmp_path, monkeypatch, serving_index=path)
    try:
        with pytest.raises(HTTPError) as error:
            _get_analogs(server, release_id=release.release_id, event_id="e1")
        assert error.value.code == 503
        body = json.loads(error.value.read())
        assert body["status"] == "unavailable"
        assert body["reason_code"] == "SERVING_INDEX_OUTDATED"
    finally:
        _stop(server, thread)

    conn = sqlite3.connect(path)
    try:
        versions = {int(row[0]) for row in conn.execute(
            "SELECT version FROM schema_versions WHERE owner = ?", (projections._OWNER,))}
        columns = {row[1] for row in conn.execute(
            "PRAGMA table_info(serving_score_summary)")}
    finally:
        conn.close()
    assert versions == {1}
    assert not {"selected_row_ids", "contributing_row_ids", "n_analogs"} & columns


def test_analogs_json_strategy_filter_reaches_the_route(tmp_path, monkeypatch):
    server, thread, release_id, release = _run_launcher(tmp_path, monkeypatch)
    try:
        url = _analogs_url(server, release_id=release.release_id, event_id="e1")
        request = Request(url + "&strategy=STR-THRU")
        request.add_header("Authorization", "Bearer " + TOKEN)
        body = json.loads(urlopen(request, timeout=5).read())
        assert [entry["strategy"] for entry in body["analogs"]] == ["STR-THRU"]

        request = Request(url + "&strategy=NOPE")
        request.add_header("Authorization", "Bearer " + TOKEN)
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=5)
        assert error.value.code == 404
        assert json.loads(error.value.read())["reason_code"] == "EVENT_NOT_FOUND"
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
