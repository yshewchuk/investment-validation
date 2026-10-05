"""Real Chromium smoke over the immutable legacy app inside the v2 shell."""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from engine.v2.serving.operations import create_server

# Drives a real Playwright browser (see tests/conftest.py's grouping rule).
pytestmark = [pytest.mark.xdist_group("serial"), pytest.mark.browser]  # drives a real Playwright browser or needs node/npm (ui/ build)


@pytest.mark.parametrize("browser_name", ["chromium"])
def test_shell_renders_all_legacy_views_and_deep_link(tmp_path, browser_name):
    static = Path("engine/dashboard/static")
    release = tmp_path / "releases" / "r1"
    shutil.copytree(static / "assets", release / "assets")
    (release / "data").mkdir(parents=True)
    shutil.copy(static / "index.html", release / "index.html")
    scripts = {"meta.js": "window.META={};", "board.js": "window.BOARD={rows:[]};",
               "health.js": "window.HEALTH={};", "flags.js": "window.FLAGS={flags:[]};",
               "strategies.js": "window.STRATEGIES=[];", "book.js": "window.BOOK={};"}
    for name, text in scripts.items():
        (release / "data" / name).write_text(text)
    (tmp_path / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "2026-09-12T00:00:00Z",
                                  "withheld_release": "r0",
                                  "code_budgets": {"consecutive_nights": 2}}))
    server = create_server(("127.0.0.1", 0), token="browser-secret", health_path=health,
                           release_root=tmp_path, frozen_at="2026-09-11T00:00:00Z")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with sync_playwright() as playwright:
            browser = getattr(playwright, browser_name).launch(headless=True)
            context = browser.new_context()
            context.add_cookies([{"name": "operations_token", "value": "browser-secret",
                                  "url": base}])
            page = context.new_page()
            page.goto(base + "/#/trades/board")
            frame = page.frame_locator("iframe#legacy")
            frame.locator("#view-board").wait_for(state="visible")
            assert "withheld" in page.locator("#state").inner_text()
            routes = (("trades/board", "board"), ("trades/explorer", "explorer"),
                      ("trades/book", "book"), ("models/modelx", "modelx"),
                      ("models/derivation", "derivation"), ("models/health", "health"))
            for route, view in routes:
                page.goto(base + "/#/" + route)
                frame.locator("#view-" + view).wait_for(state="visible")
            page.goto(base + "/#/trades/explorer/AAPL")
            frame.locator("#view-explorer").wait_for(state="visible")
            assert page.locator("#stamp").inner_text().startswith("updated")
            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_shell_marks_old_pinned_board_and_missing_scheduled_observation(tmp_path):
    static = Path("engine/dashboard/static")
    now_iso = "2026-10-04T00:00:00Z"
    generated_at = "2026-09-10T00:00:00Z"
    scheduled = "2026-09-20"

    def materialize(release, name):
        shutil.copytree(static / "assets", release / "assets")
        (release / "data").mkdir(parents=True)
        shutil.copy(static / "index.html", release / "index.html")
        scripts = {"meta.js": "window.META={};", "board.js": "window.BOARD={rows:[]};",
                   "health.js": "window.HEALTH={};", "flags.js": "window.FLAGS={flags:[]};",
                   "strategies.js": "window.STRATEGIES=[];", "book.js": "window.BOOK={};"}
        for fname, text in scripts.items():
            (release / "data" / fname).write_text(text)
        return name

    materialize(tmp_path / "releases" / "r1", "r1")
    materialize(tmp_path / "releases" / "r2", "r2")
    pointer = tmp_path / "CURRENT"
    pointer.write_text("r1\n")

    health = tmp_path / "health.json"
    health.write_text(json.dumps({
        "schema_version": "operations_health.v1.0",
        "generated_at": generated_at,
        "withheld_release": None,
        "current_release": {"release_id": "r1"},
        "code_budgets": {"consecutive_nights": 0, "unknown_occurrences": [scheduled]},
    }))

    server = create_server(("127.0.0.1", 0), token="browser-secret", health_path=health,
                           release_root=tmp_path, frozen_at=now_iso)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context()
            context.add_cookies([{"name": "operations_token", "value": "browser-secret",
                                  "url": base}])
            page = context.new_page()
            page.add_init_script(
                'Date.now = function () { return Date.parse("' + now_iso + '"); };'
                'var _setInterval = window.setInterval;'
                'window.setInterval = function (fn, ms) { return _setInterval(fn, Math.min(ms, 200)); };'
            )
            page.goto(base + "/")

            page.wait_for_function(
                '(scheduled) => { const src = document.querySelector("#legacy").getAttribute("src") || ""; '
                'const st = document.querySelector("#state").textContent; return '
                'src.indexOf("/release/r1/") >= 0 '
                '&& document.querySelector("#published").textContent === "published release: r1" '
                '&& st.indexOf("scheduled observations missing: " + scheduled) >= 0 '
                '&& st.indexOf("observation older than 24 hours") >= 0 '
                '&& st.indexOf("mismatch") < 0; }',
                arg=scheduled, timeout=2000,
            )

            state = page.locator("#state").inner_text()
            stamp = page.locator("#stamp").inner_text()
            frame_src = page.locator("#legacy").get_attribute("src") or ""
            assert "current" not in state
            assert scheduled in state
            assert "observation older than 24 hours" in state
            assert "24d 0h ago" in stamp
            assert page.locator("#release").inner_text() == "pinned release: r1"
            assert "/release/r1/" in frame_src
            assert not page.locator("#optin").is_visible()

            pointer.write_text("r2\n")

            page.wait_for_function(
                '() => document.querySelector("#published").textContent === "published release: r2"',
                timeout=2000,
            )
            assert page.locator("#published").inner_text() == "published release: r2"
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")
            assert "/release/r2/" not in (page.locator("#legacy").get_attribute("src") or "")
            assert "displayed board is stale" in page.locator("#drift").inner_text()
            assert page.locator("#optin").is_visible()

            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
