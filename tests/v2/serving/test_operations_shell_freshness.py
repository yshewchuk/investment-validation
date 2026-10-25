"""Operations shell freshness: expected sessions and a fixed release pin."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import pytest
from playwright.sync_api import sync_playwright

from engine.v2.foundation import latest_completed_session
from engine.v2.serving.operations import _shell_script, create_server

_NODE = shutil.which("node")
_needs_node = pytest.mark.skipif(_NODE is None, reason="node is required")


def _write_bundle(root: Path, release_id: str) -> None:
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


def _serve(root: Path, health_path: Path, *, frozen_at="2026-10-05T21:00:00Z"):
    server = create_server(("127.0.0.1", 0), token="browser-secret",
                           health_path=health_path, release_root=root,
                           frozen_at=frozen_at)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


@pytest.mark.parametrize(("now", "expected"), [
    (datetime(2026, 10, 5, 15, 59, tzinfo=timezone.utc), "2026-10-02"),
    (datetime(2026, 10, 5, 20, 0, tzinfo=timezone.utc), "2026-10-05"),
    (datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc), "2026-10-02"),
    (datetime(2026, 7, 3, 20, 0, tzinfo=timezone.utc), "2026-07-02"),
    (datetime(2027, 1, 1, 20, 0, tzinfo=timezone.utc), "2026-12-31"),
])
def test_latest_completed_session_uses_the_1600_et_close_and_holidays(now, expected):
    assert latest_completed_session(now) == expected


def test_current_json_exposes_the_expected_completed_session(tmp_path):
    _write_bundle(tmp_path, "r1")
    (tmp_path / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "2026-10-05T17:00:00Z"}))
    server, thread = _serve(tmp_path, health)
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/release/current.json")
        request.add_header("Authorization", "Bearer browser-secret")
        response = urlopen(request, timeout=5)
        body = json.loads(response.read())
        assert body == {"release_id": "r1"}
        assert response.headers["X-Expected-Session"] == "2026-10-05"
        assert response.headers["Cache-Control"] == "no-store"
    finally:
        _stop(server, thread)


@pytest.mark.browser
@pytest.mark.xdist_group("serial")
def test_shell_marks_recent_old_session_stale_and_keeps_its_pin(tmp_path):
    """A recent observation for an old input session is stale, not current.

    The browser keeps its r1 frame while the published pointer moves to r2,
    and only an explicit reload opts into r2.
    """
    _write_bundle(tmp_path, "r1")
    _write_bundle(tmp_path, "r2")
    (tmp_path / "CURRENT").write_text("r1\n")

    now = datetime.now(timezone.utc)
    expected = latest_completed_session(now)
    old_session = latest_completed_session(now - timedelta(days=7))
    assert old_session != expected
    generated_at = (now - timedelta(hours=1)).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({
        "schema_version": "operations_health.v1.1",
        "generated_at": generated_at,
        "current_release": {"release_id": "r1"},
        "requested_session": expected,
        "resolved_session": old_session,
        "withheld_release": None,
        "code_budgets": {"ok": True, "latest": {"occurrence": expected},
                         "consecutive_nights": 0, "unknown_occurrences": []},
    }))

    server, thread = _serve(tmp_path, health, frozen_at=now.isoformat())
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            context.add_cookies([{"name": "operations_token", "value": "browser-secret",
                                  "url": base}])
            page = context.new_page()
            page.add_init_script(
                'var _setInterval = window.setInterval;'
                'window.setInterval = function (fn, ms) {'
                ' return _setInterval(fn, Math.min(ms, 100)); };'
            )
            page.goto(base + "/")

            page.wait_for_function(
                '() => { const st = document.querySelector("#state").textContent; '
                'return st.indexOf("stale") === 0 '
                '&& st.indexOf("resolved session is older than expected session") >= 0 '
                '&& document.querySelector("#published").textContent === "published release: r1"; }',
                timeout=3000,
            )
            assert "current" not in page.locator("#state").inner_text()
            assert "1h ago" in page.locator("#stamp").inner_text()
            assert page.locator("#requested").inner_text() == f"requested session: {expected}"
            assert page.locator("#resolved").inner_text() == f"resolved session: {old_session}"
            assert page.locator("#expected").inner_text().startswith("expected session: ")
            assert "no failures recorded" in page.locator("#outcome").inner_text()
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")

            (tmp_path / "CURRENT").write_text("r2\n")

            page.wait_for_function(
                '() => document.querySelector("#published").textContent '
                '=== "published release: r2"',
                timeout=3000,
            )
            assert page.locator("#release").inner_text() == "pinned release: r1"
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")
            assert "/release/r2/" not in (page.locator("#legacy").get_attribute("src") or "")
            assert "displayed board is stale" in page.locator("#drift").inner_text()
            assert page.locator("#optin").is_visible()

            page.locator("#optin").click()
            page.wait_for_function(
                '() => { const src = document.querySelector("#legacy").getAttribute("src") || ""; '
                'return src.indexOf("/release/r2/") >= 0; }',
                timeout=3000,
            )
            browser.close()
    finally:
        _stop(server, thread)


def _shell_status(tmp_path, health, *, pinned="r1", published="r1",
                  expected="2026-10-05"):
    """Run the shell's real status functions under node with a tiny DOM stub."""
    wrapper = _shell_script("2026-10-05T21:00:00Z")
    assert wrapper.startswith("<script>") and wrapper.endswith("</script>")
    source = wrapper[len("<script>"):-len("</script>")]
    source += "\n" + "\n".join([
        "Date.now = function () { return Date.parse('2026-10-05T21:00:00Z'); };",
        f"health = {json.dumps(health)};",
        f"pinned = {json.dumps(pinned)};",
        f"published = {json.dumps(published)};",
        f"expected = {json.dumps(expected)};",
        "const __outcome = outcome();",
        "evaluate();",
        "JSON.stringify({outcome: __outcome, state: state.textContent, "
        "banner: banner.className});",
    ])
    harness = tmp_path / "shell-status.js"
    harness.write_text(
        "const vm = require('vm');\n"
        f"const source = {json.dumps(source)};\n"
        "const elements = {};\n"
        "function element() { return {textContent: '', className: '', hidden: true, "
        "addEventListener() {}}; }\n"
        "const context = {\n"
        "  console,\n"
        "  document: {querySelector(selector) { return elements[selector] || "
        "(elements[selector] = element()); }},\n"
        "  window: {addEventListener() {}},\n"
        "  location: {hash: ''},\n"
        "  fetch: function () { return new Promise(function () {}); },\n"
        "  setInterval: function () { return 0; },\n"
        "};\n"
        "vm.createContext(context);\n"
        "process.stdout.write(vm.runInContext(source, context, "
        "{filename: 'shell.js'}));\n",
        encoding="utf-8",
    )
    result = subprocess.run([_NODE, str(harness)], check=True,
                            capture_output=True, text=True, timeout=20)
    return json.loads(result.stdout)


def _health_document():
    return {
        "schema_version": "operations_health.v1.1",
        "generated_at": "2026-10-05T20:00:00Z",
        "withheld_release": None,
        "current_release": {"release_id": "r1"},
        "requested_session": "2026-10-05",
        "resolved_session": "2026-10-05",
        "code_budgets": {"ok": True, "latest": {"occurrence": "2026-10-05"},
                         "consecutive_nights": 0, "unknown_occurrences": []},
    }


@_needs_node
def test_shell_treats_false_code_budgets_ok_as_a_failed_outcome(tmp_path):
    health = _health_document()
    health["code_budgets"] = {"ok": False, "latest": {"occurrence": "2026-10-05"},
                              "consecutive_nights": 0, "unknown_occurrences": []}
    result = _shell_status(tmp_path, health)
    assert result["outcome"].startswith("latest outcome: fail:")
    assert "latest engineering outcome failed" in result["outcome"]
    assert result["state"].startswith("degraded:")
    assert "latest engineering outcome failed" in result["state"]
    assert "current" not in result["state"]


@_needs_node
def test_shell_keeps_unknown_precedence_over_failed_engineering_outcome(tmp_path):
    health = _health_document()
    health["code_budgets"] = {"ok": False, "latest": {"occurrence": "2026-10-05"},
                              "consecutive_nights": 0,
                              "unknown_occurrences": ["2026-10-04"]}
    result = _shell_status(tmp_path, health)
    assert result["outcome"].startswith("latest outcome: unknown:")
    assert "latest engineering outcome failed" in result["outcome"]
    assert "scheduled observations missing: 2026-10-04" in result["outcome"]
    assert result["state"].startswith("unknown:")
    assert "latest engineering outcome failed" in result["state"]
    assert "scheduled observations missing: 2026-10-04" in result["state"]


@_needs_node
def test_shell_keeps_malformed_evidence_unknown_with_unknown_occurrences(tmp_path):
    health = _health_document()
    health["code_budgets"] = {"ok": False, "latest": None,
                              "consecutive_nights": 0,
                              "unknown_occurrences": ["2026-10-04"]}
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == (
        "latest outcome: unknown: engineering schedule evidence missing or malformed")
    assert "latest engineering outcome failed" not in result["outcome"]
    assert result["state"].startswith("unknown:")
    assert "engineering schedule evidence missing or malformed" in result["state"]
