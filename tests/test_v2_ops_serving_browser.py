"""Real Chromium smoke over the immutable legacy app inside the v2 shell."""
from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

from engine.v2.ops import cli
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.serving.operations import create_server
from tests.ops_support import FakeClock, seed_delivered_health_release

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


def test_shell_requires_session_and_published_identity_for_current(tmp_path):
    """Session evidence and a live published id are hard preconditions of "current".

    Regression for issue #161: missing requested/resolved session evidence must
    read unknown (never current), malformed full session identifiers do the
    same without moving the frame pin, and an aborted current-release poll
    must fall
    back to the existing "published current unavailable" unknown reason while
    the frame keeps its r1 pin. A held health poll that completes only after a
    newer fetch failure is likewise superseded: it cannot replace the latest
    unavailable observation, and only a later scheduled success restores
    current.
    """
    static = Path("engine/dashboard/static")
    now_iso = "2026-10-04T00:00:00Z"

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
    health_document = {"schema_version": "operations_health.v1.0",
                       "generated_at": "2026-10-03T12:00:00Z",
                       "withheld_release": None,
                       "current_release": {"release_id": "r1"},
                       "code_budgets": {"consecutive_nights": 0, "unknown_occurrences": []}}
    health.write_text(json.dumps(health_document))

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

            # Pinned and published both say r1, budgets are clean, yet the
            # session fields are absent: unknown, and never current.
            page.wait_for_function(
                '() => { const st = document.querySelector("#state").textContent; '
                'return st.indexOf("session identity evidence missing or malformed") >= 0 '
                '&& document.querySelector("#published").textContent === "published release: r1"; }',
                timeout=2000,
            )
            state = page.locator("#state").inner_text()
            assert state.startswith("unknown")
            assert "current" not in state
            assert "session identity evidence missing or malformed" in state

            health.write_text(json.dumps({**health_document,
                                          "requested_session": "2026-10-04",
                                          "resolved_session": "2026-10-02"}))
            page.wait_for_function(
                '() => document.querySelector("#state").textContent === "current"',
                timeout=2000,
            )

            # Malformed full identifiers (not date-only) refuse current: the
            # SAME missing-or-malformed unknown reason, and the frame keeps
            # its r1 pin without throwing.
            health.write_text(json.dumps({**health_document,
                                          "requested_session": "unexpected-2026-10-04",
                                          "resolved_session": "unexpected-2026-10-02"}))
            page.wait_for_function(
                '() => document.querySelector("#state").textContent'
                '.indexOf("session identity evidence missing or malformed") >= 0',
                timeout=2000,
            )
            state = page.locator("#state").inner_text()
            assert state.startswith("unknown")
            assert "current" not in state
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")

            # Valid date-only session evidence restores current.
            health.write_text(json.dumps({**health_document,
                                          "requested_session": "2026-10-04",
                                          "resolved_session": "2026-10-02"}))
            page.wait_for_function(
                '() => document.querySelector("#state").textContent === "current"',
                timeout=2000,
            )

            # Superseded health responses must not replace a newer fetch
            # failure: hold poll 1, answer poll 2 with 503, keep holding every
            # later poll, then complete the older held request only once the
            # shell already reports unknown for the failed fetch. The page
            # keeps its own record so the test can prove the older response
            # completed, not merely that time passed.
            page.evaluate(
                '() => {'
                ' window.__healthResponses = [];'
                ' window.__healthFetches = 0;'
                ' var _fetch = window.fetch;'
                ' window.fetch = function (input, init) {'
                ' var url = typeof input === "string" ? input : ((input && input.url) || "");'
                ' var pending = _fetch.call(window, input, init);'
                ' if (url.indexOf("/health.json") >= 0) {'
                ' window.__healthFetches += 1;'
                ' pending.then(function (r) {'
                ' r.clone().text().then(function (t) {'
                ' window.__healthResponses.push(r.status + ":" + t);'
                ' });'
                ' }, function () { window.__healthResponses.push("rejected"); });'
                ' }'
                ' return pending;'
                ' };'
                '}'
            )
            held_health_routes = []
            health_attempts = [0]

            def intercept_health(route):
                health_attempts[0] += 1
                if health_attempts[0] == 2:
                    route.fulfill(status=503, body="unavailable",
                                  content_type="text/plain")
                else:
                    held_health_routes.append(route)

            valid_health_body = json.dumps({**health_document,
                                            "requested_session": "2026-10-04",
                                            "resolved_session": "2026-10-02"})
            page.route("**/health.json", intercept_health)
            page.wait_for_function(
                '() => { const st = document.querySelector("#state").textContent; '
                'return st.indexOf("health fetch failed: unavailable") >= 0 '
                '&& st.indexOf("unknown") === 0; }',
                timeout=2000,
            )

            # The newer 503 is the shell's latest observation. Fulfilling the
            # older held poll with the SAME valid document must be discarded:
            # the response completes (a missing sequence guard would flip the
            # shell back to current within the next few poll cycles), while
            # unknown + the fetch-failure note and the r1 frame pin persist.
            held_health_routes.pop(0).fulfill(status=200, body=valid_health_body,
                                              content_type="application/json")
            page.wait_for_function(
                '() => window.__healthResponses.some(function (entry) {'
                ' return entry.indexOf("200:") === 0; })',
                timeout=2000,
            )
            latest_fetches = page.evaluate('() => window.__healthFetches')
            page.wait_for_function(
                '() => window.__healthFetches >= ' + str(latest_fetches + 3),
                timeout=2000,
            )
            state = page.locator("#state").inner_text()
            assert state.startswith("unknown")
            assert "health fetch failed: unavailable" in state
            assert "current" not in state
            assert page.locator("#published").inner_text() == "published release: r1"
            assert page.locator("#release").inner_text() == "pinned release: r1"
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")

            # Drain the held polls, drop the intercept, and let a normal
            # scheduled poll restore current before the publication checks.
            for held_route in list(held_health_routes):
                held_health_routes.remove(held_route)
                held_route.fulfill(status=200, body=valid_health_body,
                                   content_type="application/json")
            page.unroute("**/health.json", intercept_health)
            page.wait_for_function(
                '() => document.querySelector("#state").textContent === "current"',
                timeout=2000,
            )

            # Aborting later current polls must demote via the EXISTING
            # unavailable-publication reason, not disturb the r1 frame pin.
            page.route("**/release/current.json", lambda route: route.abort())
            page.wait_for_function(
                '() => { const st = document.querySelector("#state").textContent; '
                'return st.indexOf("published current unavailable") >= 0 && st.indexOf("unknown") === 0; }',
                timeout=2000,
            )
            state = page.locator("#state").inner_text()
            assert state.startswith("unknown")
            assert "published current unavailable" in state
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")

            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_shell_treats_cli_produced_missing_or_mismatched_sessions_as_unknown(tmp_path):
    """Real CLI health starts current, then missing/mismatched sessions go unknown."""
    static = Path("engine/dashboard/static")
    root = tmp_path
    now_iso = "2026-10-04T00:00:00Z"
    requested_session = "2026-10-04"
    resolved_session = "2026-10-02"

    assert cli.main(["--root", str(root), "init"]) == 0
    release = root / "releases" / "r1"
    shutil.copytree(static / "assets", release / "assets")
    (release / "data").mkdir(parents=True)
    shutil.copy(static / "index.html", release / "index.html")
    scripts = {"meta.js": "window.META={};", "board.js": "window.BOARD={rows:[]};",
               "health.js": "window.HEALTH={};", "flags.js": "window.FLAGS={flags:[]};",
               "strategies.js": "window.STRATEGIES=[];", "book.js": "window.BOOK={};"}
    for name, text in scripts.items():
        (release / "data" / name).write_text(text)
    (root / "CURRENT").write_text("r1\n")

    clock = FakeClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        seed_delivered_health_release(conn, release_id="r1",
                                      requested_session=requested_session,
                                      resolved_session=resolved_session)
    finally:
        conn.close()

    health_path = root / "health.json"
    assert cli.main(["--root", str(root), "health", "--out", str(health_path)]) == 0
    health_document = json.loads(health_path.read_text())
    assert health_document["schema_version"] == "operations_health.v1.1"
    assert health_document["requested_session"] == requested_session
    assert health_document["resolved_session"] == resolved_session
    assert health_document["current_release"]["release_id"] == "r1"
    assert health_document["current_release"]["occurrence"] == resolved_session

    server = create_server(("127.0.0.1", 0), token="browser-secret",
                           health_path=health_path, release_root=root, frozen_at=now_iso)
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
                'var _setInterval = window.setInterval;'
                'window.setInterval = function (fn, ms) {'
                ' return _setInterval(fn, Math.min(ms, 200)); };'
            )
            page.goto(base + "/")
            page.wait_for_function(
                '() => document.querySelector("#state").textContent === "current"',
                timeout=2000,
            )
            assert page.locator("#published").inner_text() == "published release: r1"
            assert page.locator("#release").inner_text() == "pinned release: r1"

            missing_sessions = {key: value for key, value in health_document.items()
                                if key not in ("requested_session", "resolved_session")}
            missing_sessions["schema_version"] = "operations_health.v1.0"
            health_path.write_text(json.dumps(missing_sessions))
            page.wait_for_function(
                '() => { const st = document.querySelector("#state").textContent; '
                'return st.indexOf("unknown") === 0 && '
                'st.indexOf("session identity evidence missing or malformed") >= 0; }',
                timeout=2000,
            )
            state = page.locator("#state").inner_text()
            assert state.startswith("unknown")
            assert "current" not in state
            assert "session identity evidence missing or malformed" in state
            assert page.locator("#published").inner_text() == "published release: r1"
            assert page.locator("#release").inner_text() == "pinned release: r1"

            mismatched_sessions = dict(health_document,
                                       requested_session=resolved_session,
                                       resolved_session=requested_session)
            health_path.write_text(json.dumps(mismatched_sessions))
            page.wait_for_function(
                '() => { const st = document.querySelector("#state").textContent; '
                'return st.indexOf("unknown") === 0 && '
                'st.indexOf("resolved session is later than requested session") >= 0; }',
                timeout=2000,
            )
            state = page.locator("#state").inner_text()
            assert state.startswith("unknown")
            assert "current" not in state
            assert "resolved session is later than requested session" in state

            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


def test_shell_discards_superseded_current_release_responses(tmp_path):
    """Out-of-order /release/current.json responses never move the published id back.

    Regression for issue #161's latest-request-wins coordination: the stub
    delays the init response (r1) and one later poll response (r3) while
    immediate polls (r2, r4) overtake them. The pin lands on r1 when the
    delayed init finally completes, yet the published label keeps the newer
    ids both times.
    """
    static = Path("engine/dashboard/static")
    now_iso = "2026-10-04T00:00:00Z"

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
                                  "generated_at": "2026-10-03T12:00:00Z",
                                  "withheld_release": None,
                                  "current_release": {"release_id": "r1"},
                                  "code_budgets": {"consecutive_nights": 0,
                                                   "unknown_occurrences": []},
                                  "requested_session": "2026-10-04",
                                  "resolved_session": "2026-10-02"}))

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
            # Capture the shell's pollCurrent interval callback, then script
            # /release/current.json: call 1 (init) and call 3 are held for a
            # manual flush; calls 2 and 4 answer immediately with r2 and r4.
            page.add_init_script(
                'Date.now = function () { return Date.parse("' + now_iso + '"); };'
                'window.__consumed = [];'
                'window.__pendingRelease = [];'
                'window.__currentCalls = 0;'
                'var _setInterval = window.setInterval;'
                'window.setInterval = function (fn, ms) {'
                ' if (ms === 30000 && typeof fn === "function" && fn.name === "pollCurrent")'
                ' window.__currentPoll = fn;'
                ' return _setInterval(fn, ms);'
                '};'
                'var _fetch = window.fetch;'
                'window.fetch = function (input, init) {'
                ' var url = typeof input === "string" ? input : ((input && input.url) || "");'
                ' if (url.indexOf("/release/current.json") < 0) return _fetch.call(window, input, init);'
                ' window.__currentCalls += 1;'
                ' var id = "r" + window.__currentCalls;'
                ' var response = {ok: true, json: function () {'
                ' window.__consumed.push(id);'
                ' return Promise.resolve({release_id: id});'
                ' }};'
                ' if (window.__currentCalls % 2 === 1) {'
                ' return new Promise(function (resolve) {'
                ' window.__pendingRelease.push(function () { resolve(response); });'
                ' });'
                ' }'
                ' return Promise.resolve(response);'
                '};'
                'window.__flushRelease = function () {'
                ' var release = window.__pendingRelease.shift();'
                ' if (release) release();'
                '};'
            )
            page.goto(base + "/")
            page.wait_for_function('() => typeof window.__currentPoll === "function"',
                                   timeout=2000)

            # Poll 1 answers r2 while the init request for r1 is still pending.
            page.evaluate('void window.__currentPoll()')
            page.wait_for_function(
                '() => document.querySelector("#published").textContent === "published release: r2"',
                timeout=2000,
            )

            # The delayed init r1 finally completes: it may pin the frame, but
            # it must not overwrite the newer published r2.
            page.evaluate('window.__flushRelease()')
            page.wait_for_function(
                '() => { const src = document.querySelector("#legacy").getAttribute("src") || ""; '
                'return src.indexOf("/release/r1/") >= 0 '
                '&& document.querySelector("#published").textContent === "published release: r2"; }',
                timeout=2000,
            )
            assert page.locator("#release").inner_text() == "pinned release: r1"

            # Poll 2 (delayed r3) stays pending; poll 3 (immediate r4) wins.
            page.evaluate('void window.__currentPoll()')
            page.evaluate('void window.__currentPoll()')
            page.wait_for_function(
                '() => document.querySelector("#published").textContent === "published release: r4"',
                timeout=2000,
            )

            # The superseded r3 arrives last and is discarded, pin and label intact.
            page.evaluate('window.__flushRelease()')
            page.wait_for_function('() => window.__consumed.indexOf("r3") >= 0', timeout=2000)
            assert page.locator("#published").inner_text() == "published release: r4"
            assert page.locator("#release").inner_text() == "pinned release: r1"
            assert "/release/r1/" in (page.locator("#legacy").get_attribute("src") or "")

            browser.close()
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()
