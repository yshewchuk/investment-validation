"""Real release-reader paths for STR-THRU; only synthetic temporary state."""
# packages: engine.v2.scoring, engine.v2.models, engine.v2.ops
import json
import sqlite3
from types import SimpleNamespace

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


def test_native_release_root_or_none_maps_no_current_release_from_binding(
        tmp_path, monkeypatch, capsys):
    """A NoCurrentRelease raised by resolve_production_release_binding maps to
    the NO_CURRENT_RELEASE reason in the refusal details."""
    from engine.v2.scoring import release_bindings

    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    pointer = SimpleNamespace(sequence=1, release_id="release-1",
                              previous_release_id=None, action="promote",
                              at="2026-01-15T00:00:00Z")
    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    monkeypatch.setattr(deployment, "current_pointer", lambda root: pointer)
    monkeypatch.setattr(
        release_bindings, "resolve_production_release_binding",
        lambda: (_ for _ in ()).throw(
            release_bindings.NoCurrentRelease("no release has ever been promoted")))
    try:
        result = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        assert result is None
        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "native_score_batch_release_unavailable"
        assert event["reason"] == "NO_CURRENT_RELEASE"
        failure = conn.execute(
            "SELECT error_code, problem_json FROM native_stage_failures "
            "WHERE occurrence = ? AND scope_hash = ?",
            ("2026-01-15", "test-scope-hash")).fetchone()
        assert failure is not None
        assert failure["error_code"] == "MODEL_NOT_READY"
        saved_problem = json.loads(failure["problem_json"])
        assert saved_problem["details"]["reason"] == "NO_CURRENT_RELEASE"
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


def test_native_release_success_is_cached_and_second_call_skips_verification(
        tmp_path, monkeypatch, capsys):
    """A successful verification returns the root and memoizes it; the next
    call with the same root and release_id returns the cached root without
    re-running resolve_production_release_binding."""
    from engine.v2.scoring import release_bindings

    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    pointer = SimpleNamespace(sequence=1, release_id="release-1",
                              previous_release_id=None, action="promote",
                              at="2026-01-15T00:00:00Z")
    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    monkeypatch.setattr(deployment, "current_pointer", lambda root: pointer)
    calls = []

    def _resolve():
        calls.append(1)
        return None

    monkeypatch.setattr(
        release_bindings, "resolve_production_release_binding", _resolve)
    try:
        first = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        second = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        assert first == str(tmp_path)
        assert second == str(tmp_path)
        assert len(calls) == 1
        assert service._native_release_memo == {
            "root": str(tmp_path), "release_id": "release-1", "ok": True}
        assert capsys.readouterr().out == ""
    finally:
        conn.close()


def test_native_release_cached_refusal_reports_without_reverifying(
        tmp_path, monkeypatch, capsys):
    """A refused verification memoizes MODEL_NOT_READY; a second call re-reports
    and leaves the failure row persisted without re-running the expensive
    resolve_production_release_binding."""
    from engine.v2.scoring import release_bindings

    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    pointer = SimpleNamespace(sequence=1, release_id="release-1",
                              previous_release_id=None, action="promote",
                              at="2026-01-15T00:00:00Z")
    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    monkeypatch.setattr(deployment, "current_pointer", lambda root: pointer)
    calls = []

    def _resolve():
        calls.append(1)
        raise release_bindings.ModelNotReady("chooser_analog_pool", "missing")

    monkeypatch.setattr(
        release_bindings, "resolve_production_release_binding", _resolve)
    reports = []
    original_report = service._report_native_release_problem

    def _spy_report(detail_code, message, *, occurrence, scope_hash):
        reports.append((detail_code, message))
        return original_report(detail_code, message, occurrence=occurrence,
                               scope_hash=scope_hash)

    monkeypatch.setattr(service, "_report_native_release_problem", _spy_report)
    try:
        first = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        first_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                        if line.strip()]
        second = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        second_events = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                         if line.strip()]
        assert first is None
        assert second is None
        assert len(calls) == 1
        assert len(reports) == 2
        assert [reason for reason, _ in reports] == ["MODEL_NOT_READY", "MODEL_NOT_READY"]
        assert service._native_release_memo == {
            "root": str(tmp_path), "release_id": "release-1", "ok": False,
            "error_code": "MODEL_NOT_READY"}
        assert len(first_events) == 1
        assert first_events[0]["event"] == "native_score_batch_release_unavailable"
        assert first_events[0]["reason"] == "MODEL_NOT_READY"
        assert second_events == []
        failure = conn.execute(
            "SELECT status, error_code, problem_json, resolved_at "
            "FROM native_stage_failures WHERE occurrence = ? AND scope_hash = ?",
            ("2026-01-15", "test-scope-hash")).fetchone()
        assert failure is not None
        assert failure["status"] == "FAILED"
        assert failure["error_code"] == "MODEL_NOT_READY"
        assert failure["resolved_at"] is None
        saved_problem = json.loads(failure["problem_json"])
        assert saved_problem["details"]["reason"] == "MODEL_NOT_READY"
    finally:
        conn.close()


def test_resolved_native_stage_failure_reopens_for_same_occurrence_and_scope(
        tmp_path, monkeypatch, capsys):
    """The production persistence contract reopens a retained, resolved failure
    row when the same occurrence and scope fails again; a different occurrence
    or scope stays resolved."""
    from engine.v2.scoring import release_bindings

    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    pointer = SimpleNamespace(sequence=1, release_id="release-1",
                              previous_release_id=None, action="promote",
                              at="2026-01-15T00:00:00Z")
    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    monkeypatch.setattr(deployment, "current_pointer", lambda root: pointer)
    monkeypatch.setattr(
        release_bindings, "resolve_production_release_binding",
        lambda: (_ for _ in ()).throw(
            release_bindings.ModelNotReady("chooser_analog_pool", "missing")))
    problem = json.dumps({"code": "MODEL_NOT_READY"})
    conn.executemany(
        "INSERT INTO native_stage_failures "
        "(occurrence,scope_hash,stage_id,status,error_code,problem_json,"
        "created_at,updated_at,resolved_at) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            ("2026-01-15", "test-scope-hash", "native_score_batch", "FAILED",
             "MODEL_NOT_READY", problem, "2026-01-15T00:00:00Z",
             "2026-01-15T00:00:00Z", "2026-01-15T01:00:00Z"),
            ("2026-01-16", "test-scope-hash", "native_score_batch", "FAILED",
             "MODEL_NOT_READY", problem, "2026-01-16T00:00:00Z",
             "2026-01-16T00:00:00Z", "2026-01-16T01:00:00Z"),
            ("2026-01-15", "other-scope", "native_score_batch", "FAILED",
             "MODEL_NOT_READY", problem, "2026-01-15T00:00:00Z",
             "2026-01-15T00:00:00Z", "2026-01-15T01:00:00Z"),
        ])
    conn.commit()
    try:
        result = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        assert result is None
        event = json.loads(capsys.readouterr().out)
        assert event["event"] == "native_score_batch_release_unavailable"
        rows = {
            (row["occurrence"], row["scope_hash"]): row
            for row in conn.execute(
                "SELECT occurrence, scope_hash, resolved_at, updated_at, problem_json "
                "FROM native_stage_failures")
        }
        reopened = rows[("2026-01-15", "test-scope-hash")]
        assert reopened["resolved_at"] is None
        assert reopened["updated_at"] != "2026-01-15T00:00:00Z"
        saved_problem = json.loads(reopened["problem_json"])
        assert saved_problem["details"]["reason"] == "MODEL_NOT_READY"
        assert rows[("2026-01-16", "test-scope-hash")][
            "resolved_at"] == "2026-01-16T01:00:00Z"
        assert rows[("2026-01-15", "other-scope")][
            "resolved_at"] == "2026-01-15T01:00:00Z"
    finally:
        conn.close()
