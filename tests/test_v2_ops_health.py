"""D3: the withheld-release banner must clear once delivery catches up (§5)."""
import pytest

from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.health import health, write_health
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


def test_health_emits_v1_1_schema_and_copies_receipt_sessions(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    seed_delivered_health_release(conn, release_id="r5", requested_session="2026-09-05",
                                  resolved_session="2026-09-04")
    document = health(conn, clock=clock)
    assert document["schema_version"] == "operations_health.v1.1"
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
