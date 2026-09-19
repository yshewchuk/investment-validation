"""P3-0 compatibility preview: launcher, frame pinning, and safety controls.

No real Phase 2 release exists yet, so every test here builds its own
synthetic release bundles under ``tmp_path`` (per
``guides/rearchitecture_phase3_parity_launch.md`` §8, P3-0). Each bundle's
``data/board.json`` names its own release id, which is what the assertions
compare against — the exact "the pinned URL still returns R1" checks that §9's
L02 requires.
"""
from __future__ import annotations

import http.client
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from playwright.sync_api import expect, sync_playwright

from engine.v2.dashboard import preview
from engine.v2.serving.operations import create_server

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# synthetic bundle / server helpers
# --------------------------------------------------------------------------


def _write_bundle(root: Path, release_id: str) -> None:
    """A minimal legacy-shaped bundle whose page fetches its OWN data/board.json
    and shows the release id it reads back — the marker L02 tracks."""
    release_dir = root / "releases" / release_id
    (release_dir / "data").mkdir(parents=True)
    (release_dir / "index.html").write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>legacy</title>"
        "<div id=\"marker\"></div><script>"
        "fetch('data/board.json').then(r=>r.json()).then(j=>{"
        "document.getElementById('marker').textContent=j.release;});"
        "</script>"
    )
    (release_dir / "data" / "board.json").write_text(json.dumps({"release": release_id}))


def _write_current(root: Path, release_id: str) -> None:
    (root / "CURRENT").write_text(release_id + "\n")


def _write_health(root: Path) -> Path:
    health = root / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "2026-09-14T00:00:00Z",
                                  "withheld_release": None}))
    return health


def _serve(root: Path, health_path: Path, *, token="secret", frozen_at="t0"):
    server = create_server(("127.0.0.1", 0), token=token, health_path=health_path,
                           release_root=root, frozen_at=frozen_at)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def _get(url, token=None):
    request = Request(url)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request, timeout=5)


def _get_json(url, token=None):
    return json.loads(_get(url, token).read())


def _get_raw(server, path, token=None):
    """Bypass redirect-following entirely, so a Location header can be inspected."""
    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    headers = {"Authorization": "Bearer " + token} if token else {}
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response, body


# --------------------------------------------------------------------------
# L01 -- auth protects data/release files/health; the shell always loads
# --------------------------------------------------------------------------


def test_l01_valid_token_loads_shell_release_files_and_health(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="secret")
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        assert _get(base + "/").read().startswith(b"<!doctype html>")
        assert _get_json(base + "/release/current.json", "secret") == {"release_id": "r1"}
        assert b"marker" in _get(base + "/release/r1/index.html", "secret").read()
        assert _get_json(base + "/release/r1/data/board.json", "secret") == {"release": "r1"}
        assert json.loads(_get(base + "/health.json", "secret").read())["schema_version"] == \
            "operations_health.v1.0"
    finally:
        _stop(server, thread)


def test_l01_missing_or_invalid_token_fails_data_release_and_health(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="secret")
    base = f"http://127.0.0.1:{server.server_port}"
    protected = ("/release/current.json", "/release/r1/index.html",
                "/release/r1/data/board.json", "/health.json")
    try:
        for route in protected:
            for token in (None, "wrong-token"):
                with pytest.raises(HTTPError) as error:
                    _get(base + route, token)
                assert error.value.code == 401
    finally:
        _stop(server, thread)


def test_l01_token_never_appears_in_redirect_html_or_logs(tmp_path, capfd):
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="top-secret-token")
    try:
        response, body = _get_raw(server, "/release/current", "top-secret-token")
        assert response.status == 302
        location = response.getheader("Location")
        assert "top-secret-token" not in location
        assert b"top-secret-token" not in body
        shell = _get(f"http://127.0.0.1:{server.server_port}/").read()
        assert b"top-secret-token" not in shell
        json_body = _get(f"http://127.0.0.1:{server.server_port}/release/current.json",
                         "top-secret-token").read()
        assert b"top-secret-token" not in json_body
    finally:
        _stop(server, thread)
    captured = capfd.readouterr()
    assert "top-secret-token" not in captured.out
    assert "top-secret-token" not in captured.err


# --------------------------------------------------------------------------
# L02 over HTTP -- a pinned resolution survives a CURRENT switch
# --------------------------------------------------------------------------


def test_l02_http_pinned_urls_survive_a_current_switch(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_bundle(tmp_path, "r2")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="secret")
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        resolved = _get_json(base + "/release/current.json", "secret")["release_id"]
        assert resolved == "r1"
        pinned_page = f"{base}/release/{resolved}/index.html"
        pinned_data = f"{base}/release/{resolved}/data/board.json"
        assert _get_json(pinned_data, "secret") == {"release": "r1"}
        assert b"marker" in _get(pinned_page, "secret").read()

        _write_current(tmp_path, "r2")

        # The pinned URLs (page and data) still return R1.
        assert _get_json(pinned_data, "secret") == {"release": "r1"}
        assert b"marker" in _get(pinned_page, "secret").read()

        # A fresh resolution now returns R2.
        fresh = _get_json(base + "/release/current.json", "secret")["release_id"]
        assert fresh == "r2"
        assert _get_json(f"{base}/release/{fresh}/data/board.json", "secret") == {"release": "r2"}
    finally:
        _stop(server, thread)


# --------------------------------------------------------------------------
# L02 in the browser -- the frame pins across hashchange, a reload re-resolves
# --------------------------------------------------------------------------


@pytest.mark.xdist_group("serial")
@pytest.mark.browser  # drives a real Playwright browser or needs node/npm (ui/ build)
def test_l02_browser_frame_pins_r1_then_reload_shows_r2(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_bundle(tmp_path, "r2")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="browser-secret")
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            context.add_cookies([{"name": "operations_token", "value": "browser-secret",
                                  "url": base}])
            page = context.new_page()
            page.goto(base + "/#/trades/board")
            frame = page.frame_locator("iframe#legacy")
            expect(frame.locator("#marker")).to_have_text("r1")
            src = page.locator("iframe#legacy").get_attribute("src")
            assert "/release/r1/" in src

            _write_current(tmp_path, "r2")

            # Navigating the hash route must not move the already-open frame.
            page.goto(base + "/#/trades/explorer")
            expect(frame.locator("#marker")).to_have_text("r1")
            src = page.locator("iframe#legacy").get_attribute("src")
            assert "/release/r1/" in src

            # A fresh shell load re-resolves current and picks up R2.
            page.reload()
            frame = page.frame_locator("iframe#legacy")
            expect(frame.locator("#marker")).to_have_text("r2")
            src = page.locator("iframe#legacy").get_attribute("src")
            assert "/release/r2/" in src
            browser.close()
    finally:
        _stop(server, thread)


# --------------------------------------------------------------------------
# Safety: traversal, symlinked release file, release id containing "/"
# --------------------------------------------------------------------------


def test_traversal_is_refused(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="secret")
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/../etc/passwd", "secret")
        assert error.value.code in (400, 404)
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/r1/../../etc/passwd", "secret")
        assert error.value.code in (400, 404)
    finally:
        _stop(server, thread)


def test_symlinked_release_data_file_is_refused(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"release":"leaked"}')
    (tmp_path / "releases" / "r1" / "data" / "leak.json").symlink_to(outside)
    server, thread = _serve(tmp_path, health, token="secret")
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/r1/data/leak.json", "secret")
        assert error.value.code == 404
    finally:
        _stop(server, thread)


def test_release_id_containing_slash_is_refused(tmp_path):
    _write_bundle(tmp_path, "r1")
    # A corrupted/malicious CURRENT pointer, not a real release id.
    (tmp_path / "CURRENT").write_text("r1/evil\n")
    health = _write_health(tmp_path)
    server, thread = _serve(tmp_path, health, token="secret")
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/current.json", "secret")
        assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            _get(base + "/release/current", "secret")
        assert error.value.code == 404
    finally:
        _stop(server, thread)


# --------------------------------------------------------------------------
# Launcher: env-var token, loopback guard, ephemeral port
# --------------------------------------------------------------------------


def test_launcher_refuses_without_env_token(tmp_path, monkeypatch):
    monkeypatch.delenv(preview.TOKEN_ENV_VAR, raising=False)
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    with pytest.raises(SystemExit, match=preview.TOKEN_ENV_VAR):
        preview.run(["--host", "127.0.0.1", "--port", "0",
                     "--release-root", str(tmp_path), "--health-path", str(health)])


def test_launcher_refuses_non_loopback_host_without_flag(tmp_path, monkeypatch):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, "secret")
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    with pytest.raises(SystemExit, match="allow-non-loopback"):
        preview.run(["--host", "8.8.8.8", "--port", "0",
                     "--release-root", str(tmp_path), "--health-path", str(health)])


def test_launcher_non_loopback_with_flag_is_allowed_in_process(tmp_path, monkeypatch):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, "secret")
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    server, thread, release_id = preview.run([
        "--host", "127.0.0.1", "--port", "0", "--allow-non-loopback",
        "--release-root", str(tmp_path), "--health-path", str(health)])
    try:
        assert release_id == "r1"
    finally:
        _stop(server, thread)


def _read_line(stream, timeout):
    box: queue.Queue = queue.Queue(maxsize=1)
    threading.Thread(target=lambda: box.put(stream.readline()), daemon=True).start()
    try:
        return box.get(timeout=timeout)
    except queue.Empty:
        return None


def test_launcher_subprocess_starts_on_ephemeral_port_and_serves_shell(tmp_path):
    _write_bundle(tmp_path, "r1")
    _write_current(tmp_path, "r1")
    health = _write_health(tmp_path)
    env = dict(os.environ, V2_DASHBOARD_TOKEN="subprocess-secret")
    process = subprocess.Popen(
        [sys.executable, "-m", "engine.v2.dashboard.preview",
         "--host", "127.0.0.1", "--port", "0",
         "--release-root", str(tmp_path), "--health-path", str(health)],
        cwd=REPO_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        preview_line = _read_line(process.stdout, timeout=10)
        assert preview_line and preview_line.startswith("preview: http://"), preview_line
        release_line = _read_line(process.stdout, timeout=10)
        assert release_line and release_line.strip() == "current release: r1"
        assert "subprocess-secret" not in preview_line
        base = preview_line.removeprefix("preview: ").strip()
        body = _get(base + "/").read()
        assert body.startswith(b"<!doctype html>")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
