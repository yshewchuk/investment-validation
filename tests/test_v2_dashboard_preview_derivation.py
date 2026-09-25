"""Phase 6 slice 9: ``/derivation.json`` reachable through the real launcher (P6-4).

Drives ``preview.run`` on a loopback ephemeral port and reads the native
derivation document back over HTTP, so one boundary covers argv parsing,
``_server.build_server``'s ``create_server`` hand-off, the operations route
and ``derivation_projection`` together. Negative control: the route is
auth-gated exactly like ``/health.json`` -- no Authorization header/cookie is
401, and a wrong token is 401, never the registry document.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from engine.v2.dashboard import preview

TOKEN = "derivation-launcher-secret"


def _dashboard_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    release_dir = bundle / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_text("<!doctype html><title>legacy</title>")
    (bundle / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0"}))
    return bundle, health


def _run_launcher(tmp_path, monkeypatch):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, TOKEN)
    bundle, health = _dashboard_bundle(tmp_path)
    return preview.run(["--host", "127.0.0.1", "--port", "0",
                        "--release-root", str(bundle), "--health-path", str(health)])


def _get_derivation(server, *, token=TOKEN):
    request = Request(f"http://127.0.0.1:{server.server_port}/derivation.json")
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request, timeout=5)


def _get_shell(server):
    return urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=5)


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_derivation_json_reachable_via_preview_run(tmp_path, monkeypatch):
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch)
    try:
        assert release_id == "r1"  # the rest of the launcher still starts and pins normally
        response = _get_derivation(server)
        assert response.status == 200
        body = json.loads(response.read())
        assert body["schema_version"] == "strategy_derivation.v1"
        assert body["strategies"]  # non-empty, straight from the native registry
        assert TOKEN not in json.dumps(body)

        for bad in (None, "wrong-token"):
            with pytest.raises(HTTPError) as error:
                _get_derivation(server, token=bad)
            assert error.value.code == 401
    finally:
        _stop(server, thread)


def test_shell_document_nav_links_to_the_derivation_view(tmp_path, monkeypatch):
    """The shell nav must route derivation to the native page, not the frame.

    Slice 9 removed ``"derivation"`` from ``_VIEWS`` so the native
    ``/derivation`` page could be reached, which also dropped the shell's
    derivation link. This pins the restored entry: the shell served by a real
    ``preview.run`` carries an ``href="/derivation"`` nav link to the native
    view (never the retired ``#/models/derivation`` legacy frame route), the
    calibration health view stays linked, and the derivation href itself
    resolves to the native page rather than back to the shell.
    """
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch)
    try:
        assert release_id == "r1"
        shell = _get_shell(server).read().decode()
        assert '<a href="/derivation">derivation</a>' in shell
        assert 'href="/#/models/derivation"' not in shell
        assert 'href="/#/models/health"' in shell  # calibration health view stays reachable

        page = urlopen(f"http://127.0.0.1:{server.server_port}/derivation", timeout=5)
        assert page.status == 200
        assert b"Strategy derivation" in page.read()
    finally:
        _stop(server, thread)
