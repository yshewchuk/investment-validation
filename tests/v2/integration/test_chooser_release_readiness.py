"""Real release-reader paths for STR-THRU; only synthetic temporary state."""
# packages: engine.v2.scoring, engine.v2.models, engine.v2.ops
import json
import sqlite3

import pytest

from engine.v2.models import deployment
from engine.v2.models.frozen_state import serialize_frozen_state
from engine.v2.ops import health as health_module
from engine.v2.ops import supervisor as supervisor_module
from engine.v2.ops.catalog import transaction
from engine.v2.ops.health import health
from engine.v2.ops.native_score_batch import run_native_score_batch_worker
from engine.v2.ops.supervisor import Service
from engine.v2.scoring import release_bindings
from engine.v2.scoring.release_bindings import ModelNotReady
from tests.ops_support import POLICY, REGISTRY, catalog
from tests.test_v2_ops_native_score_batch import _event_doc, _stage_release, _worker_parameters
from tests.v2.scoring.test_release_chooser_states import _artifact, _stored
from tests.v2.scoring.test_v2_scoring_release_bindings import _row, _write_catalog


@pytest.mark.parametrize("member", ["chooser_analog_pool", "admissible_table:dyn_sv"])
@pytest.mark.parametrize("status", [None, "VALID", "AT_LIMIT", "BELOW_LIMIT", "OVERSIZE",
                                    "MISSING", "PENDING", "STAGED"])
def test_str_thru_release_readiness_and_worker_follow_declared_chooser_state(
        tmp_path, monkeypatch, capsys, member, status):
    _stage_release(tmp_path)
    if status in ("VALID", "AT_LIMIT", "BELOW_LIMIT", "OVERSIZE"):
        artifact = _artifact(member)
        _write_catalog(tmp_path, rows=[_row(member, [_stored(tmp_path, artifact)])])
        if status != "VALID":
            headroom = {"AT_LIMIT": 0, "BELOW_LIMIT": 1, "OVERSIZE": -1}[status]
            monkeypatch.setattr(release_bindings, "_CHOOSER_MAX_ARTIFACT_BYTES",
                                len(serialize_frozen_state(artifact)) + headroom)
    elif status is not None:
        # Empty STAGED objects are invalid too. There is no DYN-SV request.
        _write_catalog(tmp_path, rows=[_row(member, [], status=status)])
    monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    (work / "events.json").write_text(json.dumps([_event_doc(strategy="STR-THRU")]))
    parameters = _worker_parameters(tmp_path, expected_ids=["evt-1"])
    try:
        root = service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash="test-scope-hash")
        if status in (None, "VALID", "AT_LIMIT", "BELOW_LIMIT"):
            assert root == str(tmp_path)
            assert service._native_release_memo["ok"] is True
            assert capsys.readouterr().out == ""
            run_native_score_batch_worker(parameters, work)
            records = json.loads((work / "records.json").read_text())["records"]
            assert set(records) == {"TEST|STR-THRU|2026-01-15|am"}
            assert records["TEST|STR-THRU|2026-01-15|am"]["canonical_request"][
                "strategy_version"] == "STR-THRU"
            assert records["TEST|STR-THRU|2026-01-15|am"]["chooser_candidates"] == []
            assert records["TEST|STR-THRU|2026-01-15|am"]["chooser_selection"] is None
            assert json.loads((work / "refusals.json").read_text())["refusals"] == {}
        else:
            assert root is None
            assert service._native_release_memo["ok"] is False
            problem = json.loads(capsys.readouterr().out)
            assert problem["event"] == "native_score_batch_release_unavailable"
            assert problem["reason"] == "MODEL_NOT_READY"
            assert problem["problem"]["code"] == "MODEL_NOT_READY"
            assert member not in problem["problem"]["message"]
            assert str(tmp_path) not in problem["problem"]["message"]
            failures = conn.execute(
                "SELECT occurrence, scope_hash, stage_id, status, error_code, problem_json "
                "FROM native_stage_failures WHERE occurrence = ? AND scope_hash = ?",
                ("2026-01-15", "test-scope-hash")).fetchall()
            assert len(failures) == 1
            failure = failures[0]
            assert failure["stage_id"] == "native_score_batch"
            assert failure["status"] == "FAILED"
            assert failure["error_code"] == "MODEL_NOT_READY"
            assert json.loads(failure["problem_json"])["code"] == "MODEL_NOT_READY"
            saved_problem = json.loads(failure["problem_json"])
            assert saved_problem["category"] == "dependency"
            assert saved_problem["retryable"] is False
            assert saved_problem["message"] == (
                "native score batch release binding is unavailable")
            assert saved_problem["details"]["reason"] == "MODEL_NOT_READY"
            assert service._native_release_root_or_none(
                occurrence="2026-01-16", scope_hash="test-scope-hash") is None
            all_failures = conn.execute(
                "SELECT occurrence, updated_at FROM native_stage_failures "
                "WHERE scope_hash = ? ORDER BY occurrence",
                ("test-scope-hash",)).fetchall()
            assert len(all_failures) == 2
            assert all_failures[1]["occurrence"] == "2026-01-16"
            second_updated_at = all_failures[1]["updated_at"]
            changes_before = conn.total_changes
            assert service._native_release_root_or_none(
                occurrence="2026-01-16", scope_hash="test-scope-hash") is None
            assert conn.total_changes == changes_before
            assert conn.execute(
                "SELECT updated_at FROM native_stage_failures "
                "WHERE occurrence = ? AND scope_hash = ?",
                ("2026-01-16", "test-scope-hash")).fetchone()["updated_at"] == (
                    second_updated_at)
            monkeypatch.setattr(
                health_module, "_current_delivered_release_with_scope",
                lambda conn: (None, None))
            monkeypatch.setattr(
                health_module, "_withheld_release",
                lambda conn, current: None)
            monkeypatch.setattr(
                health_module, "_release_session_evidence",
                lambda conn, current, expected_scope: (None, None))
            document = health(conn, clock=clock)
            assert any(
                row["occurrence"] == "2026-01-15"
                and row["status"] == "FAILED"
                and row["error_code"] == "MODEL_NOT_READY"
                for row in document["native_stage_failures"])
            health_failure = next(
                row for row in document["native_stage_failures"]
                if row["occurrence"] == "2026-01-15")
            assert json.loads(health_failure["problem_json"]) == saved_problem
            with pytest.raises(ModelNotReady) as error:
                run_native_score_batch_worker(parameters, work)
            assert error.value.member_id == member
            assert not (work / "records.json").exists()
            assert not (work / "refusals.json").exists()
    finally:
        conn.close()


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


def test_native_release_recovery_resolves_failures_for_health(tmp_path, monkeypatch):
    """A failed release check is history once the release recovers: both the
    fresh-verification and memo-ok success paths mark the scope's rows
    resolved, and ops health no longer reports them."""
    _stage_release(tmp_path)
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    scope_hash = "test-scope-hash"
    try:
        monkeypatch.delenv(deployment.MODEL_RELEASE_ROOT_ENV, raising=False)
        assert service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash=scope_hash) is None
        failed = conn.execute(
            "SELECT resolved_at FROM native_stage_failures "
            "WHERE occurrence = ? AND scope_hash = ? AND stage_id = ?",
            ("2026-01-15", scope_hash, "native_score_batch")).fetchone()
        assert failed["resolved_at"] is None

        monkeypatch.setenv(deployment.MODEL_RELEASE_ROOT_ENV, str(tmp_path))
        assert service._native_release_root_or_none(
            occurrence="2026-01-15", scope_hash=scope_hash) == str(tmp_path)
        recovered = conn.execute(
            "SELECT resolved_at FROM native_stage_failures "
            "WHERE occurrence = ? AND scope_hash = ? AND stage_id = ?",
            ("2026-01-15", scope_hash, "native_score_batch")).fetchone()
        assert recovered["resolved_at"] is not None

        # The memo-ok fast path must resolve a later failure for the same scope.
        problem = json.dumps({"code": "MODEL_NOT_READY"})
        with transaction(conn):
            conn.execute(
                "INSERT INTO native_stage_failures "
                "(occurrence,scope_hash,stage_id,status,error_code,problem_json,"
                "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                ("2026-01-16", scope_hash, "native_score_batch", "FAILED",
                 "MODEL_NOT_READY", problem, "2026-01-16T00:00:00Z",
                 "2026-01-16T00:00:00Z"))
        assert service._native_release_root_or_none(
            occurrence="2026-01-16", scope_hash=scope_hash) == str(tmp_path)
        memo_resolved = conn.execute(
            "SELECT resolved_at FROM native_stage_failures "
            "WHERE occurrence = ? AND scope_hash = ? AND stage_id = ?",
            ("2026-01-16", scope_hash, "native_score_batch")).fetchone()
        assert memo_resolved["resolved_at"] is not None

        monkeypatch.setattr(
            health_module, "_current_delivered_release_with_scope",
            lambda conn: (None, None))
        monkeypatch.setattr(
            health_module, "_withheld_release",
            lambda conn, current: None)
        monkeypatch.setattr(
            health_module, "_release_session_evidence",
            lambda conn, current, expected_scope: (None, None))
        document = health(conn, clock=clock)
        assert document["native_stage_failures"] == []
        assert conn.execute(
            "SELECT COUNT(*) FROM native_stage_failures").fetchone()[0] == 2
    finally:
        conn.close()


def test_later_native_release_failure_reopens_resolved_row_for_health(tmp_path, monkeypatch):
    """A new failure after recovery is current health again: the retained
    resolved row is reopened instead of staying hidden as history."""
    conn, clock, _ = catalog(tmp_path)
    service = Service(conn, tmp_path / "ops", REGISTRY, POLICY,
                      clock=clock, code_source=tmp_path, store_root=tmp_path)
    occurrence, scope_hash = "2026-01-15", "test-scope-hash"
    problem = json.dumps({"code": "MODEL_NOT_READY"})
    with transaction(conn):
        conn.execute(
            "INSERT INTO native_stage_failures "
            "(occurrence,scope_hash,stage_id,status,error_code,problem_json,"
            "created_at,updated_at,resolved_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (occurrence, scope_hash, "native_score_batch", "FAILED",
             "MODEL_NOT_READY", problem, "2026-01-15T00:00:00Z",
             "2026-01-16T00:00:00Z", "2026-01-16T00:00:00Z"))
    monkeypatch.delenv(deployment.MODEL_RELEASE_ROOT_ENV, raising=False)
    try:
        assert service._native_release_root_or_none(
            occurrence=occurrence, scope_hash=scope_hash) is None
        row = conn.execute(
            "SELECT status, error_code, problem_json, resolved_at, updated_at "
            "FROM native_stage_failures "
            "WHERE occurrence = ? AND scope_hash = ? AND stage_id = ?",
            (occurrence, scope_hash, "native_score_batch")).fetchone()
        assert row["status"] == "FAILED"
        assert row["error_code"] == "MODEL_NOT_READY"
        assert row["resolved_at"] is None
        assert row["updated_at"] != "2026-01-16T00:00:00Z"
        assert json.loads(row["problem_json"])["details"]["reason"] == "MISSING_RELEASE_ROOT"

        monkeypatch.setattr(
            health_module, "_current_delivered_release_with_scope",
            lambda conn: (None, None))
        monkeypatch.setattr(
            health_module, "_withheld_release",
            lambda conn, current: None)
        monkeypatch.setattr(
            health_module, "_release_session_evidence",
            lambda conn, current, expected_scope: (None, None))
        document = health(conn, clock=clock)
        assert [failure["occurrence"]
                for failure in document["native_stage_failures"]] == [occurrence]
        assert json.loads(
            document["native_stage_failures"][0]["problem_json"]
        )["details"]["reason"] == "MISSING_RELEASE_ROOT"
    finally:
        conn.close()
