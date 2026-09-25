"""P6 route probe: route table, positive reachability and fail-closed rows.

The probe is driven against a REAL ``create_server`` instance on an ephemeral
loopback port (the same fixture idiom as ``tests/test_v2_ops_serving.py``),
with every callback the route table declares wired to a deterministic 2xx
stub. ``/release/current`` is a documented 302 redirect, so the probe follows
it to the release page -- urllib's default -- and records the final 200.
"""
from __future__ import annotations

import json
import threading

import pytest

from engine.v2.serving.operations import create_server, route_table
from tools import v2_route_probe as probe_mod


def _serve(tmp_path, *, whatif_job=True):
    health = tmp_path / "health.json"
    health.write_text(json.dumps({"schema_version": "operations_health.v1.0",
                                  "generated_at": "t1", "withheld_release": None}))
    calibration = tmp_path / "calibration-health.json"
    calibration.write_text(json.dumps({"generated_at": "t1", "n_scored": 0}))
    release_dir = tmp_path / "releases" / "r1"
    release_dir.mkdir(parents=True)
    (release_dir / "index.html").write_bytes(b"<!doctype html>board")
    (tmp_path / "CURRENT").write_text("r1\n")
    server = create_server(("127.0.0.1", 0), token="secret", health_path=health,
                           release_root=tmp_path, frozen_at="t0",
                           calibration_health_path=calibration,
                           submit_refresh=lambda payload: (202, {"jobs": []}),
                           submit_whatif=lambda payload: (202, {"job_id": "job_probe"}
                                                          if whatif_job else {"accepted": True}),
                           fetch_whatif=lambda job_id: (200, {"job_id": job_id}))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def _stop(server, thread):
    server.shutdown()
    thread.join(timeout=2)
    server.server_close()


def _probe(tmp_path, base, *, session="s1"):
    return probe_mod.probe(base_url=base, token="secret", session=session,
                           evidence_dir=tmp_path / "evidence",
                           refresh_body={"plan_ref": "p1"},
                           whatif_body={"request": {}, "native_inputs": {}})


def _key(row):
    return row["method"], row.get("declared_path", row["path"])


def _ok(row):
    status = row.get("status")
    return isinstance(status, int) and 200 <= status < 300


def test_probe_reaches_every_declared_route_and_writes_the_receipt(tmp_path):
    server, thread, base = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base)
    finally:
        _stop(server, thread)

    assert receipt["schema_version"] == "route_probe_receipt.v1.0"
    assert receipt["session"] == "s1"
    assert receipt["all_2xx"] is True
    rows = {_key(row): row for row in receipt["routes"]}
    for route in route_table():
        row = rows[(route["method"], route["path"])]
        assert "skipped" not in row, route
        assert _ok(row), (route, row)
        assert row["bytes"] >= 0
        assert row["latency_ms"] >= 0.0
        assert row["requested_at"]
    # Documented non-200 successes: both POSTs answer 202; /release/current's
    # 302 is followed to the identical 200 a browser's fetch would see.
    assert rows[("POST", "/actions/refresh")]["status"] == 202
    assert rows[("POST", "/actions/whatif")]["status"] == 202
    assert rows[("GET", "/release/current")]["status"] == 200
    written = tmp_path / "evidence" / "s1-route_probe.json"
    assert json.loads(written.read_text()) == receipt


def test_probe_fails_closed_when_the_base_path_does_not_exist(tmp_path):
    server, thread, base = _serve(tmp_path)
    try:
        receipt = _probe(tmp_path, base + "/missing", session="s2")
    finally:
        _stop(server, thread)

    assert receipt["all_2xx"] is False
    assert all(not _ok(row) for row in receipt["routes"] if "skipped" not in row)
    assert any(row.get("status") == 404 for row in receipt["routes"])


def test_one_missing_route_is_a_failure_with_per_route_detail(tmp_path):
    server, thread, base = _serve(tmp_path)
    try:
        (tmp_path / "releases" / "r1" / "index.html").unlink()
        receipt = _probe(tmp_path, base, session="s4")
    finally:
        _stop(server, thread)

    assert receipt["all_2xx"] is False
    failed = {_key(row) for row in receipt["routes"]
              if "skipped" not in row and not _ok(row)}
    assert failed == {("GET", "/release/current"), ("GET", "/release/")}


def test_main_exits_one_and_writes_a_failed_receipt(tmp_path):
    server, thread, base = _serve(tmp_path)
    try:
        code = probe_mod.main([
            "--base-url", base + "/missing", "--token", "secret", "--session", "s3",
            "--refresh-body", '{"plan_ref": "p1"}',
            "--whatif-body", '{"request": {}, "native_inputs": {}}',
            "--evidence-dir", str(tmp_path / "evidence")])
    finally:
        _stop(server, thread)

    assert code == 1
    receipt = json.loads((tmp_path / "evidence" / "s3-route_probe.json").read_text())
    assert receipt["all_2xx"] is False


def test_whatif_result_is_skipped_without_a_job_id_in_the_session(tmp_path):
    server, thread, base = _serve(tmp_path, whatif_job=False)
    try:
        receipt = _probe(tmp_path, base, session="s5")
    finally:
        _stop(server, thread)

    skipped = [row for row in receipt["routes"] if "skipped" in row]
    assert len(skipped) == 1
    assert skipped[0] == {"method": "GET", "path": "/actions/whatif/",
                          "skipped": "no job id in this session"}
    assert receipt["all_2xx"] is True


def test_probe_refuses_a_declared_post_route_without_its_body(tmp_path):
    with pytest.raises(probe_mod.RouteProbeError, match="whatif"):
        probe_mod.probe(base_url="http://127.0.0.1:1", token="secret", session="s6",
                        evidence_dir=tmp_path / "evidence",
                        refresh_body={"plan_ref": "p1"})


def test_main_refuses_and_writes_no_receipt_without_a_post_body(tmp_path):
    evidence = tmp_path / "evidence"
    code = probe_mod.main(["--base-url", "http://127.0.0.1:1", "--token", "secret",
                           "--session", "s7", "--refresh-body", '{"plan_ref": "p1"}',
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
