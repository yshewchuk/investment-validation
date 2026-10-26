"""Regression coverage for operations-shell native failure display."""
import json
import threading

import pytest

from engine.v2.serving.operations import create_server, shell_document


def _failure():
    problem = {"schema_version": "problem.v1.0", "code": "MODEL_NOT_READY",
               "category": "dependency", "retryable": True,
               "message": "native score batch release binding is unavailable",
               "stage": None, "trace_id": None, "dependency_refs": [],
               "retry_after_seconds": None, "diagnostic_ref": None,
               "details": {"reason": "NO_CURRENT_RELEASE"}}
    return {"occurrence": "2026-10-24", "scope_hash": "scope-hash",
            "stage_id": "native_score_batch", "status": "FAILED",
            "error_code": "MODEL_NOT_READY",
            "problem_json": json.dumps(problem, sort_keys=True, separators=(",", ":")),
            "created_at": "2026-10-24T00:00:00Z",
            "updated_at": "2026-10-24T00:00:00Z"}


def test_shell_requires_typed_v12_failures_and_marks_each_degraded():
    script = shell_document(frozen_at="2026-10-26T12:00:00Z").decode()
    assert "const nsf=health.native_stage_failures;" in script
    assert "health.schema_version==='operations_health.v1.2'" in script
    assert "native stage failure evidence missing or malformed: expected an array" in script
    assert "native stage failure problem details malformed" in script
    assert "native stage failure: occurrence '" in script
    assert "state.textContent=" in script


@pytest.mark.xdist_group("serial")
@pytest.mark.browser
def test_shell_renders_native_failure_as_degraded(tmp_path):
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        playwright = sync_api.sync_playwright().start()
    except sync_api.Error as exc:
        pytest.skip(f"Playwright unavailable: {exc}")
    try:
        try:
            browser = playwright.chromium.launch(headless=True)
        except sync_api.Error as exc:
            pytest.skip(f"Chromium unavailable: {exc}")
        now_iso = "2026-10-26T12:00:00Z"
        (tmp_path / "CURRENT").write_text("r1\n")
        server = create_server(("127.0.0.1", 0), token="browser-secret",
                               health_path=tmp_path / "health.json",
                               release_root=tmp_path, frozen_at=now_iso)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = "http://127.0.0.1:" + str(server.server_port)
        try:
            failure = _failure()
            payload = {"schema_version": "operations_health.v1.2",
                       "generated_at": now_iso, "current_release": {"release_id": "r1"},
                       "requested_session": "2026-10-23", "resolved_session": "2026-10-23",
                       "withheld_release": None,
                       "code_budgets": {"ok": True, "consecutive_nights": 0,
                                        "unknown_occurrences": [], "latest": {},
                                        "latest_occurrence": "2026-10-23"},
                       "native_stage_failures": [failure]}
            context = browser.new_context()
            context.add_cookies([{"name": "operations_token", "value": "browser-secret",
                                  "url": base}])
            page = context.new_page()
            page.add_init_script('Date.now = function () { return Date.parse("' + now_iso + '"); };')
            page.route("**/release/current.json", lambda route: route.fulfill(
                status=200, content_type="application/json",
                headers={"X-Expected-Session": "2026-10-23"}, body='{"release_id":"r1"}'))
            page.route("**/health.json", lambda route: route.fulfill(
                status=200, content_type="application/json", body=json.dumps(payload)))
            page.goto(base + "/")
            reason = "native stage failure: occurrence 2026-10-24, stage native_score_batch, error MODEL_NOT_READY, status FAILED"
            page.wait_for_function(
                "expected => document.querySelector('#state').textContent.includes(expected)",
                arg=reason, timeout=3000)
            assert "native stage failure: occurrence 2026-10-24" in page.locator("#state").inner_text()
            assert "degraded" in page.locator("#ops").get_attribute("class")
            payload.pop("native_stage_failures")
            page.reload()
            missing_reason = "native stage failure evidence missing or malformed: expected an array"
            page.wait_for_function(
                "expected => document.querySelector('#state').textContent.includes(expected)",
                arg=missing_reason, timeout=3000)
            assert page.locator("#ops").get_attribute("class") == "unknown"

            payload["native_stage_failures"] = [dict(failure, problem_json="{")]
            page.reload()
            malformed_reason = "native stage failure problem details malformed"
            page.wait_for_function(
                "expected => document.querySelector('#state').textContent.includes(expected)",
                arg=malformed_reason, timeout=3000)
            assert page.locator("#ops").get_attribute("class") == "unknown"
            context.close()
        finally:
            server.shutdown()
            thread.join(timeout=2)
            server.server_close()
        browser.close()
    finally:
        playwright.stop()
