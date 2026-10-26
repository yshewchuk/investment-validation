"""Regression coverage for v1.2 native-stage health and read-only CLI access."""
import json
import sqlite3

import pytest

from engine.v2.foundation import SystemClock
from engine.v2.ops import cli
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.health import health, write_health
from engine.v2.ops.schema import NATIVE_STAGE_FAILURE_TEXT_BYTES
from tests.ops_support import seed_delivered_health_release


def _problem():
    return {"schema_version": "problem.v1.0", "code": "MODEL_NOT_READY",
            "category": "dependency", "retryable": True,
            "message": "native score batch release binding is unavailable",
            "stage": None, "trace_id": None, "dependency_refs": [],
            "retry_after_seconds": None, "diagnostic_ref": None,
            "details": {"reason": "NO_CURRENT_RELEASE"}}


def _insert(conn, occurrence, *, resolved_at=None, scope_hash="scope",
            stage_id="native_score_batch", problem_json=None):
    stamp = "2026-01-01T00:00:00Z"
    if problem_json is None:
        problem_json = json.dumps(_problem(), sort_keys=True, separators=(",", ":"))
    conn.execute(
        "INSERT INTO native_stage_failures "
        "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at,resolved_at) "
        "VALUES (?,?,?,'FAILED','MODEL_NOT_READY',?,?,?,?)",
        (occurrence, scope_hash, stage_id, problem_json,
         stamp, stamp, resolved_at))


def test_health_projects_only_newest_100_unresolved_failures(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    seed_delivered_health_release(conn, release_id="health-release",
                                  requested_session="2026-01-01",
                                  resolved_session="2026-01-01")
    try:
        for index in range(101):
            _insert(conn, f"{index:03d}")
        _insert(conn, "999", resolved_at="2026-01-02T00:00:00Z")
        rows = health(conn, clock=clock)["native_stage_failures"]
        assert [row["occurrence"] for row in rows] == [f"{i:03d}" for i in range(100, 0, -1)]
        assert rows[0]["scope_hash"] == "scope"
        assert rows[0]["stage_id"] == "native_score_batch"
        assert rows[0]["occurrence"] == "100"
        assert rows[0]["status"] == "FAILED"
        assert rows[0]["error_code"] == "MODEL_NOT_READY"
        assert json.loads(rows[0]["problem_json"]) == _problem()
    finally:
        conn.close()


def test_health_projection_orders_ties_by_scope_and_stage(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    seed_delivered_health_release(conn, release_id="health-release",
                                  requested_session="2026-01-01",
                                  resolved_session="2026-01-01")
    try:
        _insert(conn, "same", scope_hash="z-scope", stage_id="native_score_batch")
        _insert(conn, "same", scope_hash="a-scope", stage_id="stage-b")
        _insert(conn, "same", scope_hash="a-scope", stage_id="stage-a")
        rows = health(conn, clock=clock)["native_stage_failures"]
        assert [(row["scope_hash"], row["stage_id"]) for row in rows] == [
            ("a-scope", "stage-a"),
            ("a-scope", "stage-b"),
            ("z-scope", "native_score_batch"),
        ]
    finally:
        conn.close()


def test_health_projection_replaces_oversized_durable_failure_text(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    seed_delivered_health_release(conn, release_id="health-release",
                                  requested_session="2026-01-01",
                                  resolved_session="2026-01-01")
    try:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        _insert(conn, "oversized",
                problem_json="x" * (NATIVE_STAGE_FAILURE_TEXT_BYTES["problem_json"] + 1))
        conn.execute("PRAGMA ignore_check_constraints = OFF")
        rows = health(conn, clock=clock)["native_stage_failures"]
        row = rows[0]
        assert row["occurrence"] == "oversized"
        assert row["scope_hash"] == "scope"
        assert row["stage_id"] == "native_score_batch"
        assert row["status"] == "FAILED"
        assert row["created_at"] == "2026-01-01T00:00:00Z"
        assert row["updated_at"] == "2026-01-01T00:00:00Z"
        assert row["error_code"] == "RESOURCE_LIMIT_EXCEEDED"
        assert json.loads(row["problem_json"])["code"] == "RESOURCE_LIMIT_EXCEEDED"
    finally:
        conn.close()


def test_health_writer_trims_jobs_and_preserves_failure(tmp_path):
    failure = {"occurrence": "2026-01-01", "status": "FAILED"}
    document = {"schema_version": "operations_health.v1.2",
                "jobs": [{"queue_reason_json": "x" * 1_100_000}],
                "native_stage_failures": [failure]}
    target = tmp_path / "health.json"
    write_health(target, document)
    result = json.loads(target.read_bytes())
    assert target.stat().st_size <= 1_048_576
    text = result["jobs"][0]["queue_reason_json"]
    assert len(text.encode("utf-8")) <= 4096 and text.endswith("…[truncated]")
    assert result["health_truncated"]["trimmed_text_fields"] == 1
    assert result["native_stage_failures"] == [failure]
    assert len(document["jobs"][0]["queue_reason_json"]) == 1_100_000


def test_health_writer_drops_oldest_jobs_after_trimming_text(tmp_path):
    jobs = [{"job_id": f"{index:03d}", "queue_reason_json": "x" * 5_000}
            for index in range(300)]
    document = {"schema_version": "operations_health.v1.2",
                "jobs": jobs,
                "native_stage_failures": []}
    target = tmp_path / "health.json"
    write_health(target, document)
    result = json.loads(target.read_bytes())
    assert target.stat().st_size <= 1_048_576
    truncated = result["health_truncated"]
    assert truncated["trimmed_text_fields"] == 300
    dropped = truncated["dropped_jobs"]
    assert dropped > 0
    remaining = result["jobs"]
    assert len(remaining) == 300 - dropped
    assert remaining[0]["job_id"] == f"{dropped:03d}"
    assert remaining[-1]["job_id"] == "299"


def test_health_writer_refuses_when_mandatory_content_exceeds_cap(tmp_path):
    target = tmp_path / "health.json"
    target.write_bytes(b"prior")
    document = {"schema_version": "operations_health.v1.2", "jobs": [],
                "native_stage_failures": [{"problem_json": "x" * 1_100_000}]}
    with pytest.raises(OpsError) as error:
        write_health(target, document)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert target.read_bytes() == b"prior"


def test_health_cli_pre_migration_catalog_returns_integrity_failed_json(tmp_path, capsys):
    sqlite3.connect(tmp_path / "catalog.sqlite").close()
    output = tmp_path / "health.json"
    output.write_bytes(b"stale artifact")

    assert cli.main(["health", "--root", str(tmp_path), "--out", str(output)]) == 2

    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "INTEGRITY_FAILED"
    assert document["message"] == "ops schema is missing migrations"
    assert "traceback" not in document
    assert not output.exists()


def test_health_cli_missing_catalog_returns_invalid_request_json(tmp_path, capsys):
    assert cli.main(["health", "--root", str(tmp_path)]) == 2

    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "INVALID_REQUEST"
    assert document["message"] == "health catalog is missing, locked, or unavailable"
    assert "traceback" not in document


def test_health_cli_malformed_catalog_returns_integrity_failed_json(tmp_path, capsys):
    (tmp_path / "catalog.sqlite").write_bytes(b"not a sqlite database")
    output = tmp_path / "health.json"
    output.write_bytes(b"stale artifact")

    assert cli.main(["health", "--root", str(tmp_path), "--out", str(output)]) == 2

    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "INTEGRITY_FAILED"
    assert document["message"] == "health catalog could not be verified or read"
    assert "traceback" not in document
    assert not output.exists()


def test_health_cli_payload_cap_refusal_removes_stale_output(tmp_path, monkeypatch, capsys):
    open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock()).close()
    output = tmp_path / "health.json"
    output.write_bytes(b"stale artifact")
    document = {"schema_version": "operations_health.v1.2", "jobs": [],
                "native_stage_failures": [{"problem_json": "x" * 1_100_000}]}
    monkeypatch.setattr(cli, "health", lambda *a, **k: document)

    assert cli.main(["health", "--root", str(tmp_path), "--out", str(output)]) == 2

    result = json.loads(capsys.readouterr().out)
    assert result["code"] == "RESOURCE_LIMIT_EXCEEDED"
    assert not output.exists()


def test_health_cli_refuses_migration_checksum_mismatch_readonly(tmp_path, capsys):
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    try:
        owner, version = conn.execute(
            "SELECT owner, version FROM schema_versions "
            "ORDER BY owner, version LIMIT 1").fetchone()
        conn.execute(
            "UPDATE schema_versions SET checksum = ? WHERE owner = ? AND version = ?",
            ("deliberate-mismatch", owner, version))
        conn.commit()
    finally:
        conn.close()

    assert cli.main(["health", "--root", str(tmp_path)]) == 2

    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "INTEGRITY_FAILED"
    assert document["details"]["reason"] == "checksum_mismatch"

    conn = sqlite3.connect(tmp_path / "catalog.sqlite")
    try:
        stored = conn.execute(
            "SELECT checksum FROM schema_versions WHERE owner = ? AND version = ?",
            (owner, version)).fetchone()[0]
        assert stored == "deliberate-mismatch"
    finally:
        conn.close()


def test_health_cli_opens_catalog_readonly_and_skips_backfill(tmp_path, monkeypatch):
    open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock()).close()
    observed = []

    def dispatch(args, root, conn, clock):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE forbidden(x)")
        observed.append(True)
        return {"ok": True}

    monkeypatch.setattr(cli, "dispatch", dispatch)
    monkeypatch.setattr(cli, "open_catalog", lambda *a, **k: pytest.fail("write-opened catalog"))
    monkeypatch.setattr(cli, "_ensure_outcome_sessions_backfilled",
                        lambda *a, **k: pytest.fail("ran writable backfill"))
    assert cli.main(["health", "--root", str(tmp_path)]) == 0
    assert observed == [True]
