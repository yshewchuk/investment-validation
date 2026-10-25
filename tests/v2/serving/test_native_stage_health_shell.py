"""Regression tests for native stage failure health rendering in the operations shell."""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from engine.v2.serving import operations
from engine.v2.serving.operations import (
    _HEALTH_ARTIFACT_MAX_BYTES,
    _read_calibration_health,
    _read_health,
    create_server,
    shell_document,
)


def _native_stage_failure() -> dict:
    return {
        "occurrence": "2026-10-24",
        "stage_id": "native_score_batch",
        "error_code": "MODEL_NOT_READY",
        "status": "FAILED",
    }


def _health_artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "operations-health.json"
    artifact.write_text(json.dumps({
        "schema_version": "operations_health.v1.2",
        "generated_at": "2026-10-25T12:00:00Z",
        "native_stage_failures": [_native_stage_failure()],
    }))
    return artifact


@pytest.mark.parametrize("reader", [_read_health, _read_calibration_health])
def test_health_readers_accept_and_preserve_native_stage_failures(tmp_path, reader):
    """Both readers must accept v1.2 and keep the additive failure entry."""
    artifact = _health_artifact(tmp_path)
    raw = reader(artifact)

    document = json.loads(raw)
    assert document["schema_version"] == "operations_health.v1.2"
    assert document["generated_at"] == "2026-10-25T12:00:00Z"
    assert document["native_stage_failures"] == [_native_stage_failure()]

    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert raw == canonical


@pytest.mark.parametrize("reader", [_read_health, _read_calibration_health])
def test_health_readers_reject_oversized_artifact_before_decoding(tmp_path, monkeypatch, reader):
    """Both readers refuse an oversized artifact before JSON decoding."""
    artifact = tmp_path / "operations-health.json"
    artifact.write_bytes(b"{" + b"x" * _HEALTH_ARTIFACT_MAX_BYTES)

    def never_decode(*args, **kwargs):
        pytest.fail("oversized health artifact reached JSON decoding")

    monkeypatch.setattr(operations.json, "loads", never_decode)
    with pytest.raises(ValueError, match="unsupported health artifact"):
        reader(artifact)


def test_shell_document_renders_native_stage_failures_into_health_reason():
    """The embedded JS must accept v1.2 and name each failure in its reason path."""
    document = shell_document().decode()
    compact = re.sub(r"\s+", " ", document)

    assert "operations_health.v1.2" in compact
    assert "const nsf=health.native_stage_failures;" in compact

    # The failure reason path reads occurrence, stage_id, error_code and status.
    assert "native stage failure: occurrence " in compact
    assert "occurrence '+occ+', stage '+sid+', error '+ec+', status '+st" in compact

    # v1.2 requires the array; a missing/non-array value is loud, not omitted.
    assert "health.schema_version==='operations_health.v1.2'" in compact
    assert "native stage failure evidence missing or malformed: expected an array" in compact

    # Dynamic output goes through textContent (never innerHTML).
    assert "state.textContent=" in compact
    assert "stamp.textContent=" in compact


@pytest.mark.xdist_group("serial")
@pytest.mark.browser  # drives a real Playwright browser or needs node/npm (ui/ build)
def test_shell_renders_native_stage_failures_in_browser(tmp_path):
    """A real shell shows a v1.2 FAILED/MODEL_NOT_READY row in its health reason."""
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
        try:
            now_iso = "2026-10-25T19:00:00Z"
            (tmp_path / "CURRENT").write_text("r1\n")
            server = create_server(("127.0.0.1", 0), token="browser-secret",
                                   health_path=tmp_path / "health.json",
                                   release_root=tmp_path, frozen_at=now_iso)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = "http://127.0.0.1:" + str(server.server_port)
            try:
                failure = _native_stage_failure()
                health_payload = {
                    "schema_version": "operations_health.v1.2",
                    "generated_at": "2026-10-25T12:00:00Z",
                    "current_release": {"release_id": "r1"},
                    "requested_session": "2026-10-25",
                    "resolved_session": "2026-10-24",
                    "code_budgets": {"consecutive_nights": 0, "unknown_occurrences": []},
                    "native_stage_failures": [failure],
                }
                reason = ("native stage failure: occurrence " + failure["occurrence"] +
                          ", stage " + failure["stage_id"] + ", error " +
                          failure["error_code"] + ", status " + failure["status"])
                context = browser.new_context()
                context.add_cookies([{"name": "operations_token", "value": "browser-secret",
                                      "url": base}])
                page = context.new_page()
                page.add_init_script(
                    'Date.now = function () { return Date.parse("' + now_iso + '"); };'
                )
                page.route("**/health.json", lambda route: route.fulfill(
                    status=200, content_type="application/json",
                    body=json.dumps(health_payload)))
                page.goto(base + "/")
                page.wait_for_function(
                    '(expected) => document.querySelector("#state").textContent.indexOf(expected) >= 0',
                    arg=reason, timeout=2000,
                )
                state = page.locator("#state").inner_text()
                assert "native stage failure: occurrence " in state
                assert failure["occurrence"] in state
                assert failure["stage_id"] in state
                assert failure["error_code"] in state
                assert failure["status"] in state
                context.close()
            finally:
                server.shutdown()
                thread.join(timeout=2)
                server.server_close()
        finally:
            browser.close()
    finally:
        playwright.stop()


@pytest.mark.parametrize("reader", [_read_health, _read_calibration_health])
def test_health_readers_accept_artifact_exactly_at_the_cap(tmp_path, reader):
    """The boundary is cap-INCLUSIVE for readers too: exactly
    ``_HEALTH_ARTIFACT_MAX_BYTES`` bytes decode and are served; only strictly
    larger artifacts are refused."""
    artifact = tmp_path / "operations-health.json"
    document = {"schema_version": "operations_health.v1.2",
                "generated_at": "2026-10-25T12:00:00Z", "pad": ""}
    document["pad"] = "x" * (_HEALTH_ARTIFACT_MAX_BYTES
                             - len(json.dumps(document).encode("utf-8")))
    payload = json.dumps(document).encode("utf-8")
    assert len(payload) == _HEALTH_ARTIFACT_MAX_BYTES
    artifact.write_bytes(payload)
    served = json.loads(reader(artifact))
    assert served["schema_version"] == "operations_health.v1.2"
    assert served["pad"] == document["pad"]
