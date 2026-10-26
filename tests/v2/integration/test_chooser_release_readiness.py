"""Real release-reader paths for STR-THRU; only synthetic temporary state."""
# packages: engine.v2.scoring, engine.v2.models, engine.v2.ops
import json
import sqlite3

import pytest

from engine.v2.models import deployment
from engine.v2.ops import supervisor as supervisor_module
from engine.v2.ops.supervisor import Service
from tests.ops_support import POLICY, REGISTRY, catalog


@pytest.mark.parametrize(
    ("missing_root", "expected_reason"),
    [
        (True, "MISSING_RELEASE_ROOT"),
        (False, "NO_CURRENT_RELEASE"),
    ],
)
def test_native_release_root_or_none_reports_missing_or_absent_release(
        tmp_path, monkeypatch, capsys, missing_root, expected_reason):
    if missing_root:
        monkeypatch.delenv(deployment.MODEL_RELEASE_ROOT_ENV, raising=False)
    else:
        monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    try:
        result = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        assert result is None
        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "native_score_batch_release_unavailable"
        assert event["reason"] == expected_reason
        failure = conn.execute(
            "SELECT stage_id, status, error_code, problem_json "
            "FROM native_stage_failures WHERE occurrence = ? AND scope_hash = ?",
            ("2026-01-15", "test-scope-hash")).fetchone()
        assert failure is not None
        assert failure["stage_id"] == "native_score_batch"
        assert failure["status"] == "FAILED"
        assert failure["error_code"] == "MODEL_NOT_READY"
        saved_problem = json.loads(failure["problem_json"])
        assert saved_problem["details"]["reason"] == expected_reason
    finally:
        conn.close()


def test_native_release_persistence_failure_is_caught_and_logged(
        tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(deployment.MODEL_RELEASE_ROOT_ENV, raising=False)
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)

    def _raise_catalog_error(conn):
        raise sqlite3.OperationalError("catalog transaction unavailable")

    monkeypatch.setattr(supervisor_module, "transaction", _raise_catalog_error)
    try:
        result = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        assert result is None
        events = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                  if line.strip()]
        assert len(events) == 2
        persistence_events = [
            event for event in events
            if event["event"] == "native_stage_failure_persistence_failed"]
        unavailable_events = [
            event for event in events
            if event["event"] == "native_score_batch_release_unavailable"]
        assert len(persistence_events) == 1
        assert len(unavailable_events) == 1
        persistence = persistence_events[0]
        assert persistence == {
            "event": "native_stage_failure_persistence_failed",
            "occurrence": "2026-01-15",
            "stage_id": "native_score_batch",
            "problem_code": "MODEL_NOT_READY",
            "exception_type": "OperationalError",
        }
        assert "catalog transaction unavailable" not in json.dumps(persistence)
        unavailable = unavailable_events[0]
        assert unavailable["reason"] == "MISSING_RELEASE_ROOT"
        assert unavailable["problem"]["code"] == "MODEL_NOT_READY"
    finally:
        conn.close()
