"""P6 route probe: route table, GET-only reachability and fail-closed rows.

The probe is driven against a REAL ``create_server`` instance on an ephemeral
loopback port (the same fixture idiom as ``tests/test_v2_ops_serving.py``),
with every callback the route table declares wired to a deterministic 2xx
stub. Only GET routes are requested; every declared POST route must appear in
the receipt as ``skipped_post`` and must never reach its server callback.
``/release/current`` is a documented 302 redirect, so the probe follows it to
the release page -- urllib's default -- and records the final 200. The bearer
token comes only from ``V2_PROBE_TOKEN`` and never appears in a receipt.
"""
from __future__ import annotations

import json
import threading

import pytest

from engine.v2.serving.operations import create_server, route_table
from tools import v2_route_probe as probe_mod


@pytest.fixture(autouse=True)
def _probe_token(monkeypatch):
    monkeypatch.setenv(probe_mod.TOKEN_ENV, "secret")


def _serve(tmp_path):
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "t1", "withheld_release": None}))
    calibration = tmp_path / "calibration-health.json"
    calibration.write_text(json.dumps({"generated_at": "t1", "n_scored": 0}))
    release_dir = tmp_path / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_bytes(b"<!doctype html>board")
    (tmp_path / "CURRENT").write_text("r1\n")
    calls: list[str] = []

    def _record(kind, response):
        def callback(payload):
            calls.append(kind)
            return 202, response
        return callback

    server = create_server(("127.0.0.1", 0), token="secret", health_path=health,
                           release_root=tmp_path, frozen_at="t0",
                           calibration_health_path=calibration,
                           submit_refresh=_record("refresh", {"jobs": []}),
                           submit_whatif=_record("whatif", {"job_id": "job_probe"}),
                           fetch_whatif=lambda job_id: (200, {"job_id": job_id}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}", calls


def _stop(server, thread):
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def _probe(tmp_path, base, *, session="s1"):
    return probe_mod.probe(base_url=base, session=session,
                           evidence_dir=tmp_path / "evidence")


def _key(row):
    return row["method"], row.get("declared_path", row["path"])


def _ok(row):
    status = row.get("status")
    return isinstance(status, int) and 200 <= status < 300


def _probed(row):
    return "skipped" not in row and "skipped_post" not in row


def test_probe_reaches_every_declared_get_route_and_writes_the_receipt(tmp_path):
    server, thread, base, calls = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base)
    finally:
        _stop(server, thread)

    assert receipt["schema_version"] == "route_probe_receipt.v1.0"
    assert receipt["session"] == "s1"
    assert receipt["generated_at"]
    assert receipt["all_2xx"] is True
    rows = {_key(row): row for row in receipt["routes"]}
    for route in route_table():
        row = rows[(route["method"], route["path"])]
        if route["method"] == "POST":
            assert "skipped_post" in row, route
            assert "status" not in row, route
        elif route["path"] == "/actions/whatif/":
            assert row["skipped"] == "no job id in this session"
        else:
            assert "skipped" not in row, route
            assert _ok(row), (route, row)
            assert row["bytes"] >= 0
            assert row["latency_ms"] >= 0.0
            assert row["requested_at"]
    # Documented non-200 success: /release/current's 302 is followed to the
    # identical 200 a browser's fetch would see.
    assert rows[("GET", "/release/current")]["status"] == 200
    assert calls == []
    written = tmp_path / "evidence" / "s1-route_probe.json"
    assert json.loads(written.read_text()) == receipt


def test_post_routes_are_never_requested(tmp_path):
    server, thread, base, calls = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base, session="s8")
    finally:
        _stop(server, thread)

    assert calls == []
    posts = [row for row in receipt["routes"] if row["method"] == "POST"]
    assert {row["path"] for row in posts} == {"/actions/refresh", "/actions/whatif"}
    assert all("skipped_post" in row and "status" not in row for row in posts)


def test_receipt_never_contains_the_probe_token(tmp_path, monkeypatch):
    token = "probe-token-must-not-be-recorded"
    monkeypatch.setenv(probe_mod.TOKEN_ENV, token)
    server, thread, base, _ = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base, session="s9")
    finally:
        _stop(server, thread)

    written = (tmp_path / "evidence" / "s9-route_probe.json").read_text()
    assert token not in written
    assert token not in json.dumps(receipt)


def test_probe_fails_closed_when_the_base_path_does_not_exist(tmp_path):
    server, thread, base, _ = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base + "/missing", session="s2")
    finally:
        _stop(server, thread)

    assert receipt["all_2xx"] is False
    assert all(not _ok(row) for row in receipt["routes"] if _probed(row))
    assert any(row.get("status") == 404 for row in receipt["routes"])


def test_one_missing_route_is_a_failure_with_per_route_detail(tmp_path):
    server, thread, base, _ = _serve(tmp_path)
    try:
        (tmp_path / "releases" / "r1" / "index.html").unlink()
        receipt = _probe(tmp_path, base, session="s4")
    finally:
        _stop(server, thread)

    assert receipt["all_2xx"] is False
    failed = {_key(row) for row in receipt["routes"] if _probed(row) and not _ok(row)}
    assert failed == {("GET", "/release/current"), ("GET", "/release/")}


def test_main_exits_one_and_writes_a_failed_receipt(tmp_path):
    server, thread, base, _ = _serve(tmp_path)
    try:
        code = probe_mod.main([
            "--base-url", base + "/missing", "--session", "s3",
            "--evidence-dir", str(tmp_path / "evidence")])
    finally:
        _stop(server, thread)

    assert code == 1
    receipt = json.loads((tmp_path / "evidence" / "s3-route_probe.json").read_text())
    assert receipt["all_2xx"] is False


def test_whatif_result_route_is_skipped_without_a_job_id(tmp_path):
    server, thread, base, _ = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base, session="s5")
    finally:
        _stop(server, thread)

    skipped = [row for row in receipt["routes"] if "skipped" in row]
    assert skipped == [{"method": "GET", "path": "/actions/whatif/",
                        "skipped": "no job id in this session"}]
    assert receipt["all_2xx"] is True


def test_probe_refuses_without_the_env_token(tmp_path, monkeypatch):
    monkeypatch.delenv(probe_mod.TOKEN_ENV, raising=False)
    with pytest.raises(probe_mod.RouteProbeError, match=probe_mod.TOKEN_ENV):
        probe_mod.probe(base_url="http://127.0.0.1:1", session="s6",
                        evidence_dir=tmp_path / "evidence")


def test_main_refuses_and_writes_no_receipt_without_the_env_token(tmp_path, monkeypatch):
    monkeypatch.delenv(probe_mod.TOKEN_ENV, raising=False)
    evidence = tmp_path / "evidence"
    code = probe_mod.main(["--base-url", "http://127.0.0.1:1", "--session", "s7",
                           "--evidence-dir", str(evidence)])

    assert code == 2
    assert not (evidence / "s7-route_probe.json").exists()


def test_route_table_lists_get_and_post_paths_from_the_server_table():
    rows = route_table()

    assert rows
    assert all(row["path"].startswith("/") for row in rows)
    assert {row["method"] for row in rows} == {"GET", "POST"}
    assert {("POST", row["path"]) for row in rows if row["method"] == "POST"} == {
        ("POST", "/actions/refresh"), ("POST", "/actions/whatif")}
    assert all(row["method"] == "GET" for row in rows if row["parameterized"])
