"""D3: the withheld-release banner must clear once delivery catches up (§5)."""
import json
import sqlite3
from datetime import date, timedelta

import pytest

from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.health import NATIVE_STAGE_FAILURE_DIAGNOSTIC, health, write_health
from tests.ops_support import catalog, seed_delivered_health_release


def _withheld_release(conn, release_id, occurrence):
    with transaction(conn):
        conn.execute(
            "INSERT INTO releases(release_id,occurrence,manifest_json,manifest_hash,"
            "expected_current,eligible,published_at,delivered_at) VALUES (?,?,?,?,?,?,?,?)",
            (release_id, occurrence, "{}", "hash-" + release_id, None, 0, None, None))


def test_missing_receipt_chain_is_a_typed_validation_failure(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _withheld_release(conn, "r5", "2026-09-05")
    with pytest.raises(OpsError) as err:
        health(conn, clock=clock)
    assert err.value.code == "VALIDATION_FAILED"


def test_withheld_on_night_5_delivered_on_night_6_is_cleared(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _withheld_release(conn, "r5", "2026-09-05")
    seed_delivered_health_release(conn, release_id="r6", requested_session="2026-09-06",
                                  resolved_session="2026-09-06")
    document = health(conn, clock=clock)
    assert document["withheld_release"] is None
    assert document["current_release"]["release_id"] == "r6"


def test_delivered_on_night_5_withheld_on_night_6_is_shown(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    _withheld_release(conn, "r6", "2026-09-06")
    document = health(conn, clock=clock)
    assert document["withheld_release"]["release_id"] == "r6"


def test_withheld_and_delivered_on_the_same_occurrence_is_cleared(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    _withheld_release(conn, "r5-withheld", "2026-09-05")
    seed_delivered_health_release(conn, release_id="r5-delivered",
                                  requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    document = health(conn, clock=clock)
    assert document["withheld_release"] is None


def test_health_emits_v1_2_schema_and_copies_receipt_sessions(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-04")
    document = health(conn, clock=clock)
    assert document["schema_version"] == "operations_health.v1.2"
    assert document["requested_session"] == "2026-09-05"
    assert document["resolved_session"] == "2026-09-04"
    assert document["current_release"]["occurrence"] == "2026-09-04"


def test_resolved_session_after_requested_is_a_typed_validation_failure(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-06")
    with pytest.raises(OpsError) as err:
        health(conn, clock=clock)
    assert err.value.code == "VALIDATION_FAILED"


def test_write_health_is_byte_identical_across_repeated_calls(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    first, second = tmp_path / "health-a.json", tmp_path / "health-b.json"
    write_health(first, health(conn, clock=clock))
    write_health(second, health(conn, clock=clock))
    assert first.read_bytes() == second.read_bytes()


def test_native_stage_failures_health_bounds_and_ordering(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    scope_hash = "test-scope-hash"
    stage_id = "native_score_batch"
    problem = json.dumps({"code": "MODEL_NOT_READY"})
    occurrences = [(date(2026, 1, 1) + timedelta(days=i)).isoformat() for i in range(105)]
    with transaction(conn):
        for occurrence in occurrences:
            conn.execute(
                "INSERT INTO native_stage_failures "
                "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (occurrence, scope_hash, stage_id, "FAILED", "MODEL_NOT_READY", problem,
                 "2026-09-15T00:00:00Z", "2026-09-15T00:00:00Z"))
    document = health(conn, clock=clock)
    failures = document["native_stage_failures"]
    returned = [row["occurrence"] for row in failures]
    assert len(failures) == 100
    assert returned == list(reversed(occurrences[5:]))
    assert set(occurrences[:5]).isdisjoint(returned)
    assert returned[0] == occurrences[-1]
    conn.close()


def test_native_stage_failures_oversized_text_is_bounded_or_refused(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    oversized = "x" * 8193
    params = ("2026-09-15", "test-scope-hash", "native_score_batch", "FAILED",
              "MODEL_NOT_READY", oversized, "2026-09-15T00:00:00Z",
              "2026-09-15T00:00:00Z")
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(conn):
            conn.execute(
                "INSERT INTO native_stage_failures "
                "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)", params)
    conn.execute("PRAGMA ignore_check_constraints = ON")
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO native_stage_failures "
                "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)", params)
    finally:
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    document = health(conn, clock=clock)
    failures = document["native_stage_failures"]
    assert len(failures) == 1
    row = failures[0]
    assert row["status"] == "FAILED"
    assert row["error_code"] == "RESOURCE_LIMIT_EXCEEDED"
    assert row["problem_json"] == NATIVE_STAGE_FAILURE_DIAGNOSTIC
    conn.close()


def test_native_stage_failures_oversized_non_problem_text_is_bounded_or_refused(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    oversized_occurrence = "x" * 65
    problem = json.dumps({"code": "MODEL_NOT_READY"})
    params = (oversized_occurrence, "test-scope-hash", "native_score_batch", "FAILED",
              "MODEL_NOT_READY", problem, "2026-09-15T00:00:00Z",
              "2026-09-15T00:00:00Z")
    with pytest.raises(sqlite3.IntegrityError):
        with transaction(conn):
            conn.execute(
                "INSERT INTO native_stage_failures "
                "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)", params)
    conn.execute("PRAGMA ignore_check_constraints = ON")
    try:
        with transaction(conn):
            conn.execute(
                "INSERT INTO native_stage_failures "
                "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)", params)
    finally:
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    document = health(conn, clock=clock)
    failures = document["native_stage_failures"]
    assert len(failures) == 1
    row = failures[0]
    assert row["occurrence"] == "RESOURCE_LIMIT_EXCEEDED"
    assert row["status"] == "FAILED"
    assert row["error_code"] == "RESOURCE_LIMIT_EXCEEDED"
    assert row["problem_json"] == problem
    conn.close()


def test_serving_read_and_ops_write_share_one_health_cap():
    """The writer and the reader enforce the SAME byte cap from the low-level
    foundation artifact layer -- a published artifact is always servable, by
    construction."""
    from engine.v2.foundation import OPERATIONS_HEALTH_MAX_BYTES
    from engine.v2.ops import health as ops_health
    from engine.v2.serving import operations as serving_operations

    assert OPERATIONS_HEALTH_MAX_BYTES == 1024 * 1024
    assert ops_health.OPERATIONS_HEALTH_MAX_BYTES == OPERATIONS_HEALTH_MAX_BYTES
    assert serving_operations._HEALTH_ARTIFACT_MAX_BYTES == OPERATIONS_HEALTH_MAX_BYTES


def _seed_jobs_with_large_queue_reasons(conn, count, reason_bytes):
    reason = json.dumps({"reason": "waiting_for_feed", "detail": "x" * reason_bytes})
    with transaction(conn):
        for index in range(count):
            stamp = f"2026-09-15T{index // 60 % 24:02d}:{index % 60:02d}:00Z"
            conn.execute(
                "INSERT INTO jobs (job_id,namespace,idempotency_key,request_digest,principal,"
                "kind,spec_json,resource_class,checkpoint_contract_ref,retry_json,state,priority,"
                "max_attempts,created_at,updated_at,queue_reason_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"job-{index:05d}", "test", f"ik-{index}", "digest", "test", "legacy_score",
                 "{}", "standard", "contract-ref", "{}", "queued", 0, 1, stamp, stamp, reason))


def test_write_health_trims_optional_content_and_preserves_native_failures(tmp_path):
    from engine.v2.foundation import OPERATIONS_HEALTH_MAX_BYTES
    from engine.v2.serving.operations import _read_health

    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    problem = json.dumps({"code": "MODEL_NOT_READY"})
    with transaction(conn):
        conn.execute(
            "INSERT INTO native_stage_failures "
            "(occurrence,scope_hash,stage_id,status,error_code,problem_json,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            ("2026-09-15", "test-scope-hash", "native_score_batch", "FAILED",
             "MODEL_NOT_READY", problem, "2026-09-15T00:00:00Z", "2026-09-15T00:00:00Z"))
    _seed_jobs_with_large_queue_reasons(conn, 400, 40_000)

    document = health(conn, clock=clock)
    assert len(json.dumps(document).encode("utf-8")) > OPERATIONS_HEALTH_MAX_BYTES
    artifact = tmp_path / "health.json"
    write_health(artifact, document)

    raw = artifact.read_bytes()
    assert len(raw) <= OPERATIONS_HEALTH_MAX_BYTES
    served = json.loads(raw)
    assert served["schema_version"] == "operations_health.v1.2"
    # every native failure row survives the trim, verbatim
    assert served["native_stage_failures"] == document["native_stage_failures"]
    # truncation is indicated, and it only ever touched optional content
    truncation = served["health_truncated"]
    assert truncation["dropped_jobs"] > 0
    assert truncation["trimmed_text_fields"] > 0
    assert len(served["jobs"]) == len(document["jobs"]) - truncation["dropped_jobs"]
    # the newest job row is retained, with its oversized queue reason trimmed
    assert served["jobs"][-1]["job_id"] == document["jobs"][-1]["job_id"]
    assert served["jobs"][-1]["queue_reason_json"].endswith("…[truncated]")
    # the artifact serving refuses above is exactly the one ops wrote: the
    # serving read accepts it and canonicalizes the bytes it was handed
    canonical = json.dumps(served, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert _read_health(artifact) == canonical
    conn.close()


def test_write_health_refuses_impossible_mandatory_rows_without_replacing(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-05")
    artifact = tmp_path / "health.json"
    write_health(artifact, health(conn, clock=clock))
    original = artifact.read_bytes()

    document = health(conn, clock=clock)
    document["native_stage_failures"] = [{
        "occurrence": "2026-09-15", "scope_hash": "test-scope-hash",
        "stage_id": "native_score_batch", "status": "FAILED",
        "error_code": "RESOURCE_LIMIT_EXCEEDED",
        "problem_json": "x" * (1024 * 1024 + 4096),
        "created_at": "2026-09-15T00:00:00Z", "updated_at": "2026-09-15T00:00:00Z"}]
    with pytest.raises(OpsError) as err:
        write_health(artifact, document)
    assert err.value.code == "RESOURCE_LIMIT_EXCEEDED"
    # the refusal happens BEFORE the old artifact is replaced: same bytes,
    # and no orphaned temporary file was left behind
    assert artifact.read_bytes() == original
    assert not list(tmp_path.glob("health.json.*"))
    conn.close()
