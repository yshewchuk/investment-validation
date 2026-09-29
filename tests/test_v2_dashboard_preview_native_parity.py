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


def _fixture_report_path(tmp_path: Path) -> Path:
    """A real report built by the production comparator, one engineered mismatch."""
    legacy_rows = {"AAPL-2026-01-01": _row()}
    native_rows = {"AAPL-2026-01-01": _row(forecast_p10=1.0)}
    report = compare_native_vs_legacy(legacy_rows, native_rows, PARITY_DIMENSIONS)
    return write_parity_report(
        apply_native_refusals(report, {}), tmp_path / "native_parity_report.json")


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
