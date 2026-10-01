"""Phase 6: ``/native_parity(.json)`` reachable through the real launcher (PR2).

Drives ``preview.run`` on a loopback ephemeral port and reads the
native-vs-legacy parity summary back over HTTP, so one boundary covers argv
parsing, ``_server.build_server``'s ``create_server`` hand-off, the operations
route and ``native_parity_projection`` together. The JSON route is auth-gated
exactly like ``/health.json`` -- no Authorization header is 401, never the
summary -- and ``/native_parity`` itself is a static page shell like
``/derivation`` and ``/analogs``.
"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from engine.v2.dashboard import preview
from engine.v2.ops.native_parity_report import (
    PARITY_DIMENSIONS,
    apply_native_refusals,
    compare_native_vs_legacy,
    write_parity_report,
)
from engine.v2.parity.dimensions import (
    ANALOG_FIELDS,
    FINANCIAL_FIELDS,
    FORECAST_FIELDS,
    GATE_FIELDS,
    SIMULATION_FIELDS,
)

TOKEN = "native-parity-launcher-secret"

_DEFAULT_VALUES = {
    name: 0.0
    for name in (
        FORECAST_FIELDS + SIMULATION_FIELDS + FINANCIAL_FIELDS
        + GATE_FIELDS + ANALOG_FIELDS
    )
}


def _row(**overrides):
    row = dict(_DEFAULT_VALUES)
    row["gate_pass"] = True
    row["n_analogs"] = 3
    row.update(overrides)
    return row


def _dashboard_bundle(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    release_dir = bundle / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_text("<!doctype html><title>legacy</title>")
    (bundle / "CURRENT").write_text("r1\n")
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0"}))
    return bundle, health


def _fixture_report_path(tmp_path: Path, captured=None) -> Path:
    """A real report built by the production comparator, one engineered mismatch.

    ``captured`` (optional) is appended verbatim as the exporter's optional
    ``captured_comparison`` block; the rest of the document still comes from
    the real comparator.
    """
    legacy_rows = {"AAPL-2026-01-01": _row()}
    native_rows = {"AAPL-2026-01-01": _row(forecast_p10=1.0)}
    report = compare_native_vs_legacy(legacy_rows, native_rows, PARITY_DIMENSIONS)
    path = write_parity_report(
        apply_native_refusals(report, {}), tmp_path / "native_parity_report.json")
    if captured is not None:
        document = json.loads(path.read_text())
        document["captured_comparison"] = captured
        path.write_text(json.dumps(document))
    return path


def _run_launcher(tmp_path, monkeypatch, *extra_args):
    monkeypatch.setenv(preview.TOKEN_ENV_VAR, TOKEN)
    bundle, health = _dashboard_bundle(tmp_path)
    return preview.run(["--host", "127.0.0.1", "--port", "0",
                        "--release-root", str(bundle), "--health-path", str(health),
                        *extra_args])


def _get_native_parity_json(server, *, token=TOKEN):
    request = Request(f"http://127.0.0.1:{server.server_port}/native_parity.json")
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    return urlopen(request, timeout=5)


def _stop(server, thread) -> None:
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def test_native_parity_json_refused_when_flag_omitted(tmp_path, monkeypatch):
    server, thread, release_id = _run_launcher(tmp_path, monkeypatch)
    try:
        assert release_id == "r1"  # the rest of the launcher still starts and pins normally
        with pytest.raises(HTTPError) as error:
            _get_native_parity_json(server)
        assert error.value.code == 503
        assert b"not configured" in error.value.read()
    finally:
        _stop(server, thread)


def test_native_parity_json_no_report_when_report_path_missing(tmp_path, monkeypatch):
    missing = tmp_path / "no-report-yet.json"
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(missing))
    try:
        response = _get_native_parity_json(server)
        assert response.status == 200
        body = json.loads(response.read())
        assert body["status"] == "no_report"
    finally:
        _stop(server, thread)


def test_native_parity_json_available_with_real_report(tmp_path, monkeypatch):
    report_path = _fixture_report_path(tmp_path)
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(report_path))
    try:
        response = _get_native_parity_json(server)
        assert response.status == 200
        body = json.loads(response.read())
        assert body["status"] == "available"
        assert body["mismatched_row_count"] >= 1
        assert "captured_comparison" not in body
    finally:
        _stop(server, thread)


def test_native_parity_page_served_without_auth(tmp_path, monkeypatch):
    server, thread, _ = _run_launcher(tmp_path, monkeypatch)
    try:
        response = urlopen(f"http://127.0.0.1:{server.server_port}/native_parity", timeout=5)
        assert response.status == 200
        assert response.headers.get_content_type() == "text/html"
    finally:
        _stop(server, thread)


def test_native_parity_json_requires_auth(tmp_path, monkeypatch):
    server, thread, _ = _run_launcher(tmp_path, monkeypatch)
    try:
        with pytest.raises(HTTPError) as error:
            _get_native_parity_json(server, token=None)
        assert error.value.code == 401
    finally:
        _stop(server, thread)


_CAPTURED_GROUPS = ("forecasts", "simulation", "financial_diagnostics", "verdicts", "analogs")


def _captured_block():
    """Synthetic exporter-shaped captured block; no private capture fixtures."""
    return {
        "schema_version": "captured_native_comparison.v1.0",
        "scope": "selected_saved_replay",
        "full_population_verified": False,
        "cutover_qualified": False,
        "current_board": False,
        "identity": {
            "ticker": "AAPL", "strategy": "STR-THRU", "event_date": "2026-01-01",
            "session": "AMC", "as_of": "2026-01-02", "entry_date": "2026-01-05",
            "exit_date": "2026-02-02",
        },
        "clocks": {
            "corpus_as_of": "2026-01-02",
            "requested_decision_at": "2026-01-02T15:30:00-05:00",
            "decision_as_of": "2026-01-02",
            "quote_as_of": "2026-01-02T00:00:00Z",
            "event_date": "2026-01-01",
            "session": "AMC",
        },
        "provenance": {
            "corpus_hash": "sha256-corpus", "fixture_id": "AAPL-2026-01-01",
            "payload_hash": "sha256-payload", "legacy_request_hash": "sha256-legacy-req",
            "native_request_hash": "sha256-native-req", "trace_hash": "sha256-trace",
            "same_input_receipt": "sha256-receipt", "frozen_release_id": "rel-1",
            "native_snapshot_ref": "snap-1",
        },
        "legacy": {group: {"field_a": 1.0, "field_b": None} for group in _CAPTURED_GROUPS},
        "native": {group: {"field_a": 1.25, "field_b": 0.0} for group in _CAPTURED_GROUPS},
        "checks": [{"name": "unused", "passed": True}],
        "numeric_comparisons": [{"field": "field_a", "delta": 0.25}],
        "runtime_stage_count": 3,
    }


def test_native_parity_json_projects_captured_comparison(tmp_path, monkeypatch):
    report_path = _fixture_report_path(tmp_path, captured=_captured_block())
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(report_path))
    try:
        body = json.loads(_get_native_parity_json(server).read())
        assert body["status"] == "available"
        captured = body["captured_comparison"]
        assert captured["scope"] == "selected_saved_replay"
        assert captured["current_board"] is False
        assert captured["full_population_verified"] is False
        assert captured["cutover_qualified"] is False
        assert len(captured["clocks"]) == 6
        assert len(captured["provenance"]) == 9
        assert captured["identity"]["strategy"] == "STR-THRU"
        assert len(captured["rows"]) == 10
        assert captured["rows"][0] == {"group": "forecasts", "field": "field_a",
                                       "legacy": 1.0, "native": 1.25,
                                       "legacy_display": 1.0, "native_display": 1.25}
        assert all(set(row) == {"group", "field", "legacy", "native",
                                "legacy_display", "native_display"}
                   for row in captured["rows"])
        assert captured["rows"][1]["legacy"] is None
        assert captured["rows"][1]["native"] == 0.0
        assert captured["rows"][1]["legacy_display"] is None
        assert captured["rows"][1]["native_display"] == 0.0
    finally:
        _stop(server, thread)


def test_native_parity_json_invalid_captured_block_is_unavailable(tmp_path, monkeypatch):
    block = _captured_block()
    block["current_board"] = 0
    report_path = _fixture_report_path(tmp_path, captured=block)
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(report_path))
    try:
        with pytest.raises(HTTPError) as error:
            _get_native_parity_json(server)
        assert error.value.code == 503
        body = json.loads(error.value.read())
        assert body["status"] == "unavailable"
        assert body["reason_code"] == "NATIVE_PARITY_REPORT_MALFORMED"
    finally:
        _stop(server, thread)


def test_native_parity_json_preserves_html_like_captured_values(tmp_path, monkeypatch):
    hostile = '<img src=x onerror="alert(1)">'
    block = _captured_block()
    block["identity"]["ticker"] = hostile
    block["provenance"]["fixture_id"] = hostile
    block["legacy"]["forecasts"][hostile] = None
    block["native"]["forecasts"][hostile] = 0.0
    report_path = _fixture_report_path(tmp_path, captured=block)
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(report_path))
    try:
        captured = json.loads(_get_native_parity_json(server).read())["captured_comparison"]
        assert captured["identity"]["ticker"] == hostile
        assert captured["provenance"]["fixture_id"] == hostile
        rows = {(row["group"], row["field"]): row for row in captured["rows"]}
        assert rows[("forecasts", hostile)]["legacy"] is None
        assert rows[("forecasts", hostile)]["native"] == 0.0
    finally:
        _stop(server, thread)


def test_native_parity_json_unsafe_int_display_survives_http_json(tmp_path, monkeypatch):
    block = _captured_block()
    block["native"]["simulation"]["field_a"] = 10 ** 400
    block["legacy"]["verdicts"]["field_a"] = -(2 ** 53)
    report_path = _fixture_report_path(tmp_path, captured=block)
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(report_path))
    try:
        raw = _get_native_parity_json(server).read()
        rows = {(row["group"], row["field"]): row
                for row in json.loads(raw)["captured_comparison"]["rows"]}
        assert rows[("simulation", "field_a")]["native"] == 10 ** 400
        assert rows[("simulation", "field_a")]["native_display"] == str(10 ** 400)
        assert rows[("verdicts", "field_a")]["legacy"] == -(2 ** 53)
        assert rows[("verdicts", "field_a")]["legacy_display"] == str(-(2 ** 53))
        assert f'"native_display":"{10 ** 400}"'.encode() in raw
    finally:
        _stop(server, thread)


def test_native_parity_json_projects_boolean_gate_pass_verdict_over_http(tmp_path, monkeypatch):
    block = _captured_block()
    block["legacy"]["verdicts"]["gate_pass"] = True
    block["native"]["verdicts"]["gate_pass"] = False
    report_path = _fixture_report_path(tmp_path, captured=block)
    server, thread, _ = _run_launcher(
        tmp_path, monkeypatch, "--native-parity-report-path", str(report_path))
    try:
        raw = _get_native_parity_json(server).read()
        captured = json.loads(raw)["captured_comparison"]
        rows = {(row["group"], row["field"]): row for row in captured["rows"]}
        verdict = rows[("verdicts", "gate_pass")]
        assert verdict["legacy"] is True and verdict["native"] is False
        assert verdict["legacy_display"] is True and verdict["native_display"] is False
        assert captured["identity"] == block["identity"]
        assert captured["clocks"] == block["clocks"]
        assert captured["provenance"] == block["provenance"]
        assert (b'{"field":"gate_pass","group":"verdicts","legacy":true,'
                b'"legacy_display":true,"native":false,"native_display":false}') in raw
    finally:
        _stop(server, thread)
