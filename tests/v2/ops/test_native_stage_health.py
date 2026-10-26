"""Regression coverage for v1.2 native-stage health and read-only CLI access."""
import json
import sqlite3

import pytest

from engine.v2.foundation import SystemClock
from engine.v2.ops import cli
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.health import health, write_health


def _problem():
    return {"schema_version": "problem.v1.0", "code": "MODEL_NOT_READY",
            "category": "dependency", "retryable": True,
            "message": "native score batch release binding is unavailable",
            "stage": None, "trace_id": None, "dependency_refs": [],
            "retry_after_seconds": None, "diagnostic_ref": None,
            "details": {"reason": "NO_CURRENT_RELEASE"}}


def _insert(conn, occurrence, *, resolved_at=None):
    stamp = "2026-01-01T00:00:00Z"
    conn.execute(
        "INSERT INTO native_stage_failures "
        "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at,resolved_at) "
        "VALUES (?,?,?,'FAILED','MODEL_NOT_READY',?,?,?,?)",
        (occurrence, "scope", "native_score_batch",
         json.dumps(_problem(), sort_keys=True, separators=(",", ":")),
         stamp, stamp, resolved_at))


def test_health_projects_only_newest_100_unresolved_failures(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    try:
        for index in range(101):
            _insert(conn, f"{index:03d}")
        _insert(conn, "999", resolved_at="2026-01-02T00:00:00Z")
        rows = health(conn, clock=clock)["native_stage_failures"]
        assert [row["occurrence"] for row in rows] == [f"{i:03d}" for i in range(100, 0, -1)]
        assert rows[0]["status"] == "FAILED"
        assert rows[0]["error_code"] == "MODEL_NOT_READY"
        assert json.loads(rows[0]["problem_json"]) == _problem()
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


def test_health_writer_refuses_when_mandatory_content_exceeds_cap(tmp_path):
    target = tmp_path / "health.json"
    target.write_bytes(b"prior")
    document = {"schema_version": "operations_health.v1.2", "jobs": [],
                "native_stage_failures": [{"problem_json": "x" * 1_100_000}]}
    with pytest.raises(OpsError) as error:
        write_health(target, document)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert target.read_bytes() == b"prior"


def test_health_cli_pre_migration_catalog_returns_invalid_request_json(tmp_path, capsys):
    sqlite3.connect(tmp_path / "catalog.sqlite").close()

    assert cli.main(["health", "--root", str(tmp_path)]) == 2

    document = json.loads(capsys.readouterr().out)
    assert document["code"] == "INVALID_REQUEST"
    assert document["message"] == "health catalog is missing, locked, or unavailable"
    assert "traceback" not in document


def test_health_cli_opens_catalog_readonly_and_skips_backfill(tmp_path, monkeypatch):
    sqlite3.connect(tmp_path / "catalog.sqlite").close()
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
