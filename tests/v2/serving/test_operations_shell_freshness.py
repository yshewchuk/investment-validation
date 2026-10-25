"""Operations shell freshness: expected sessions and a fixed release pin."""
from __future__ import annotations

import json
import shutil
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pytest
from playwright.sync_api import sync_playwright

from engine.v2.foundation import latest_completed_session, market_schedule
from engine.v2.ops.health import budget_streak, record_check
from engine.v2.serving.operations import create_server, shell_document
from tests.ops_support import catalog

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


def _script_for(frozen_at: str) -> str:
    """The shell's inline script, reached only through the public document."""
    document = shell_document(frozen_at=frozen_at).decode()
    return document[document.index("<script>"):]


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
    (datetime(2026, 10, 5, 19, 59, tzinfo=timezone.utc), "2026-10-02"),
    (datetime(2026, 10, 5, 20, 0, tzinfo=timezone.utc), "2026-10-05"),
    (datetime(2026, 10, 3, 20, 0, tzinfo=timezone.utc), "2026-10-02"),
    (datetime(2026, 7, 3, 20, 0, tzinfo=timezone.utc), "2026-07-02"),
    (datetime(2027, 1, 1, 20, 0, tzinfo=timezone.utc), "2026-12-31"),
    # Black Friday 2026 (day after Thanksgiving): the modelled 13:00 EST
    # early close -- 18:00 UTC -- completes the session; 17:59 UTC is still
    # mid-session, so the last completed session is Wednesday 2026-11-25.
    (datetime(2026, 11, 27, 17, 59, tzinfo=timezone.utc), "2026-11-25"),
    (datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc), "2026-11-27"),
    # Christmas Eve 2026 (Thursday session): same 13:00 EST boundary.
    (datetime(2026, 12, 24, 17, 59, tzinfo=timezone.utc), "2026-12-23"),
    (datetime(2026, 12, 24, 18, 0, tzinfo=timezone.utc), "2026-12-24"),
    # 2027-12-24 is a Friday but the Christmas observance falls on it, so it
    # is a holiday, not an early-close session: the answer walks back to the
    # most recent earlier scheduled session, Thursday 2027-12-23.
    (datetime(2027, 12, 24, 18, 0, tzinfo=timezone.utc), "2027-12-23"),
    # July 3 2025 (Thursday session; July 4 is the Friday holiday): the third
    # modelled early close -- 13:00 EDT is 17:00 UTC, so 16:59 UTC is mid-
    # session and the last completed session is Wednesday 2025-07-02.
    (datetime(2025, 7, 3, 16, 59, tzinfo=timezone.utc), "2025-07-02"),
    (datetime(2025, 7, 3, 17, 0, tzinfo=timezone.utc), "2025-07-03"),
    # The early-close rule is day-specific: Thursday 2026-07-02 has no
    # early-close rule, so 13:59 EDT is mid-session and the previous session
    # still answers.
    (datetime(2026, 7, 2, 17, 59, tzinfo=timezone.utc), "2026-07-01"),
    (datetime(2026, 7, 2, 18, 0, tzinfo=timezone.utc), "2026-07-01"),
])
def test_latest_completed_session_uses_the_scheduled_close_and_holidays(now, expected):
    assert latest_completed_session(now) == expected


def test_latest_completed_session_accepts_a_valid_frozen_snapshot_clock():
    assert latest_completed_session(frozen_at="2026-10-05T21:00:00Z") == "2026-10-05"
    assert latest_completed_session(frozen_at="2026-10-05T19:59:00Z") == "2026-10-02"
    assert latest_completed_session(frozen_at="2026-11-27T18:00:00Z") == "2026-11-27"
    assert latest_completed_session(frozen_at="2026-11-27T17:59:00Z") == "2026-11-25"


@pytest.mark.parametrize("frozen_at", ["", "not-a-timestamp", "2026-13-01T00:00:00Z",
                                       0, False, [], {}])
def test_latest_completed_session_refuses_malformed_frozen_clock(frozen_at):
    with pytest.raises(ValueError):
        latest_completed_session(frozen_at=frozen_at)
    with pytest.raises(ValueError):
        latest_completed_session(datetime(2026, 10, 5, 21, tzinfo=timezone.utc),
                                 frozen_at=frozen_at)


def test_latest_completed_session_treats_none_and_unknown_as_live(monkeypatch):
    """Live means the wall clock, so freeze the module's clock to a known
    Eastern instant (Monday 2026-10-05, 17:30 EDT, past the 16:00 close) and
    assert the concrete session date instead of a real-clock boundary roll."""
    fixed = datetime(2026, 10, 5, 17, 30, tzinfo=ZoneInfo("America/New_York"))

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed if tz is None else fixed.astimezone(tz)

    monkeypatch.setattr(market_schedule, "datetime", FrozenDatetime)
    assert latest_completed_session() == "2026-10-05"
    assert latest_completed_session(frozen_at=None) == "2026-10-05"
    assert latest_completed_session(frozen_at="unknown") == "2026-10-05"


@pytest.mark.parametrize(("frozen_at", "expected_session"), [
    ("2026-10-05T19:59:00Z", "2026-10-02"),  # minutes before the 16:00 ET close
    ("2026-10-05T20:00:00Z", "2026-10-05"),  # exactly at the close
    ("2026-10-05T21:00:00Z", "2026-10-05"),  # after the close
    ("2026-11-27T17:59:00Z", "2026-11-25"),  # Black Friday, before 13:00 EST
    ("2026-11-27T18:00:00Z", "2026-11-27"),  # Black Friday, at the early close
])
def test_current_json_routes_snapshot_clocks_to_the_expected_session(tmp_path, frozen_at,
                                                                     expected_session):
    _write_bundle(tmp_path, "r1")
    (tmp_path / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "2026-10-05T17:00:00Z"}))
    server, thread = _serve(tmp_path, health, frozen_at=frozen_at)
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/release/current.json")
        request.add_header("Authorization", "Bearer browser-secret")
        response = urlopen(request, timeout=5)
        body = json.loads(response.read())
        assert body == {"release_id": "r1"}
        assert response.headers["X-Expected-Session"] == expected_session
        assert response.headers["Cache-Control"] == "no-store"
    finally:
        _stop(server, thread)


def test_current_json_refuses_a_malformed_frozen_clock_without_live_time(tmp_path):
    _write_bundle(tmp_path, "r1")
    (tmp_path / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "2026-10-05T17:00:00Z"}))
    server, thread = _serve(tmp_path, health, frozen_at="not-a-timestamp")
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/release/current.json")
        request.add_header("Authorization", "Bearer browser-secret")
        response = urlopen(request, timeout=5)
        assert json.loads(response.read()) == {"release_id": "r1"}
        assert "X-Expected-Session" not in response.headers
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
            assert page.locator("#expected").inner_text() == f"expected session: {expected}"
            assert page.locator("#ops").get_attribute("class") == "stale"
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
    wrapper = _script_for("2026-10-05T21:00:00Z")
    assert wrapper.startswith("<script>") and wrapper.endswith("</script>")
    source = wrapper[len("<script>"):-len("</script>")]
    source += "\n" + "\n".join([
        "Date.now = function () { return Date.parse('2026-10-05T21:00:00Z'); };",
        f"health = {json.dumps(health)};",
        f"pinned = {json.dumps(pinned)};",
        f"published = {json.dumps(published)};",
        f"expected = {json.dumps(expected)};",
        "const __outcome = outcome();",
        "render();",
        "JSON.stringify({outcome: __outcome, state: state.textContent, "
        "banner: banner.className, expected: expectedEl.textContent});",
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
def test_shell_treats_a_failed_latest_night_as_a_failed_outcome(tmp_path):
    health = _health_document()
    health["code_budgets"] = {"ok": False, "latest": {"occurrence": "2026-10-05"},
                              "consecutive_nights": 1, "unknown_occurrences": []}
    result = _shell_status(tmp_path, health)
    assert result["outcome"].startswith("latest outcome: fail:")
    assert "latest engineering outcome failed" in result["outcome"]
    assert "1 consecutive failed engineering night(s)" in result["outcome"]
    assert "unknown (not observed)" not in result["outcome"]
    assert result["state"].startswith("degraded:")
    assert "latest engineering outcome failed" in result["state"]
    assert "current" not in result["state"]
    assert result["banner"] == "degraded"


@_needs_node
def test_shell_treats_an_unknown_latest_night_as_unknown_not_failed(tmp_path):
    """The producers' unobserved/unknown latest night shape stays unknown.

    ``budget_streak`` reports ``ok`` false with a zero consecutive-failure
    streak exactly when the latest recorded night is unknown (or nothing was
    ever recorded); the outcome line and the banner must then say unknown,
    never "latest engineering outcome failed".
    """
    health = _health_document()
    health["code_budgets"] = {"ok": False, "latest": {"occurrence": "2026-10-05"},
                              "consecutive_nights": 0,
                              "unknown_occurrences": ["2026-10-05"]}
    result = _shell_status(tmp_path, health)
    assert result["outcome"].startswith("latest outcome: unknown:")
    assert "latest engineering outcome unknown (not observed)" in result["outcome"]
    assert "scheduled observations missing: 2026-10-05" in result["outcome"]
    assert "latest engineering outcome failed" not in result["outcome"]
    assert result["state"].startswith("unknown:")
    assert "latest engineering outcome failed" not in result["state"]
    assert "scheduled observations missing: 2026-10-05" in result["state"]
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_treats_a_streakless_false_ok_with_no_missing_night_as_unknown(tmp_path):
    health = _health_document()
    health["code_budgets"] = {"ok": False, "latest": {"occurrence": "2026-10-05"},
                              "consecutive_nights": 0, "unknown_occurrences": []}
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == ("latest outcome: unknown: latest engineering "
                                 "outcome unknown (not observed)")
    assert "failed" not in result["outcome"]
    assert result["state"].startswith("unknown:")
    assert "latest engineering outcome failed" not in result["state"]
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_shows_the_real_producer_streak_shapes(tmp_path):
    """Producer-compatible: every document below is exactly what
    ``budget_streak`` emits for the same ``health_observations`` rows --
    never hand-shaped evidence. Fresh catalog, a pass, an unknown latest
    night over a pass, then the same night re-recorded failed.
    """
    conn, _, _ = catalog(tmp_path)
    health = _health_document()

    fresh = budget_streak(conn)
    health["code_budgets"] = fresh
    result = _shell_status(tmp_path, health)
    assert fresh["ok"] is False and fresh["latest"] is None
    assert result["outcome"] == (
        "latest outcome: unknown: engineering schedule evidence missing or malformed")
    assert result["banner"] == "unknown"

    record_check(conn, "2026-10-02", "engineering", True, {"attempt": 1})
    passing = budget_streak(conn)
    health["code_budgets"] = passing
    result = _shell_status(tmp_path, health)
    assert passing["ok"] is True and passing["consecutive_nights"] == 0
    assert result["outcome"] == "latest outcome: pass: no failures recorded"
    assert result["state"] == "current"
    assert result["banner"] == ""

    record_check(conn, "2026-10-05", "engineering", None, {"attempt": 1})
    unobserved = budget_streak(conn)
    health["code_budgets"] = unobserved
    result = _shell_status(tmp_path, health)
    assert unobserved["ok"] is False and unobserved["consecutive_nights"] == 0
    assert unobserved["unknown_occurrences"] == ["2026-10-05"]
    assert result["outcome"].startswith("latest outcome: unknown:")
    assert "latest engineering outcome failed" not in result["outcome"]
    assert "latest engineering outcome unknown (not observed)" in result["outcome"]
    assert "scheduled observations missing: 2026-10-05" in result["outcome"]
    assert result["state"].startswith("unknown:")
    assert result["banner"] == "unknown"

    record_check(conn, "2026-10-05", "engineering", False, {"attempt": 2})
    failed = budget_streak(conn)
    health["code_budgets"] = failed
    result = _shell_status(tmp_path, health)
    assert failed["ok"] is False and failed["consecutive_nights"] == 1
    assert result["outcome"].startswith("latest outcome: fail:")
    assert "latest engineering outcome failed" in result["outcome"]
    assert "1 consecutive failed engineering night(s)" in result["outcome"]
    assert "unknown (not observed)" not in result["outcome"]
    assert result["state"].startswith("degraded:")
    assert result["banner"] == "degraded"
    conn.close()


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
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_admits_null_and_object_withheld_evidence(tmp_path):
    result = _shell_status(tmp_path, _health_document())
    assert result["outcome"].startswith("latest outcome: pass:")
    assert result["state"] == "current"
    assert result["banner"] == ""

    health = _health_document()
    health["withheld_release"] = {"release_id": "r0", "occurrence": "2026-10-04"}
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == "latest outcome: fail: release withheld: (details omitted)"
    assert result["state"].startswith("degraded:")
    assert "release withheld: (details omitted)" in result["state"]
    assert result["banner"] == "degraded"


@_needs_node
def test_shell_renders_exact_expected_session_for_current_evidence(tmp_path):
    result = _shell_status(tmp_path, _health_document())
    assert result["expected"] == "expected session: 2026-10-05"
    assert result["state"] == "current"
    assert result["banner"] == ""


@_needs_node
def test_shell_treats_a_missing_expected_session_header_as_unknown(tmp_path):
    result = _shell_status(tmp_path, _health_document(), expected=None)
    assert result["expected"] == "expected session: unavailable"
    assert result["state"].startswith("unknown:")
    assert "expected session evidence missing or malformed" in result["state"]
    assert result["banner"] == "unknown"


@pytest.mark.parametrize("value", ["", "r0", False, 0, [], ["r0"]])
@_needs_node
def test_shell_treats_malformed_withheld_evidence_as_unknown(tmp_path, value):
    health = _health_document()
    health["withheld_release"] = value
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == (
        "latest outcome: unknown: withheld release evidence missing or malformed")
    assert result["state"].startswith("unknown:")
    assert "withheld release evidence missing or malformed" in result["state"]
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_treats_missing_withheld_evidence_as_unknown(tmp_path):
    health = _health_document()
    del health["withheld_release"]
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == "latest outcome: unknown: withheld release evidence missing"
    assert result["state"].startswith("unknown:")
    assert "withheld release evidence missing" in result["state"]
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_treats_resolved_session_later_than_expected_as_unknown(tmp_path):
    """The rs>ex refusal branch: a health snapshot naming a session the
    calendar has not yet completed is unknown, never current or stale."""
    health = _health_document()
    health["resolved_session"] = "2026-10-06"
    result = _shell_status(tmp_path, health)
    assert result["state"].startswith("unknown:")
    assert "resolved session is later than expected session" in result["state"]
    assert "older than expected session" not in result["state"]
    assert result["banner"] == "unknown"


@pytest.mark.parametrize("code_budgets", [
    {},
    "not-an-object",
    [],
    {"ok": "true", "latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": 0, "unknown_occurrences": []},
    {"ok": 1, "latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": 0, "unknown_occurrences": []},
    {"latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": 0, "unknown_occurrences": []},
    {"ok": True, "latest": "2026-10-05",
     "consecutive_nights": 0, "unknown_occurrences": []},
    {"ok": True, "latest": [],
     "consecutive_nights": 0, "unknown_occurrences": []},
    {"ok": True, "consecutive_nights": 0, "unknown_occurrences": []},
    {"ok": True, "latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": "0", "unknown_occurrences": []},
    {"ok": True, "latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": True, "unknown_occurrences": []},
    {"ok": True, "latest": {"occurrence": "2026-10-05"}, "consecutive_nights": 0},
    {"ok": True, "latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": 0, "unknown_occurrences": {}},
    {"ok": True, "latest": {"occurrence": "2026-10-05"},
     "consecutive_nights": 0, "unknown_occurrences": "2026-10-04"},
])
@_needs_node
def test_shell_admits_only_the_typed_schedule_evidence(tmp_path, code_budgets):
    """Every mistyped schedule-evidence shape is refused as unknown by both
    the outcome line and the status evaluator -- never trusted as pass,
    degraded or stale."""
    health = _health_document()
    health["code_budgets"] = code_budgets
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == (
        "latest outcome: unknown: engineering schedule evidence missing or malformed")
    assert result["state"].startswith("unknown:")
    assert "engineering schedule evidence missing or malformed" in result["state"]
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_admits_an_absent_schedule_evidence_key_as_unknown(tmp_path):
    """v1.0 health documents carry no code_budgets evidence at all: unknown,
    never trusted."""
    health = _health_document()
    del health["code_budgets"]
    result = _shell_status(tmp_path, health)
    assert result["outcome"] == (
        "latest outcome: unknown: engineering schedule evidence missing or malformed")
    assert result["state"].startswith("unknown:")
    assert result["banner"] == "unknown"


@_needs_node
def test_shell_reports_stale_banner_for_old_but_valid_evidence(tmp_path):
    health = _health_document()
    health["generated_at"] = "2026-10-03T20:00:00Z"
    result = _shell_status(tmp_path, health)
    assert result["state"].startswith("stale:")
    assert "observation older than 24 hours" in result["state"]
    assert result["banner"] == "stale"
