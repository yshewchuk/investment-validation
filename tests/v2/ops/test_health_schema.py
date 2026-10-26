"""Behavioral test for the health() producer schema version."""
from engine.v2.ops.health import health, record_check
from tests.ops_support import catalog, seed_delivered_health_release


def test_health_reports_operations_health_v1_2(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    try:
        seed_delivered_health_release(conn, release_id="schema-test",
                                      requested_session="2026-10-05",
                                      resolved_session="2026-10-02")
        report = health(conn, clock=clock)
    finally:
        conn.close()
    assert report["schema_version"] == "operations_health.v1.2"
    assert report["code_budgets"]["latest_occurrence"] is None


def test_code_budgets_reports_the_newest_observed_night(tmp_path):
    """``latest_occurrence`` names the newest recorded observation, by
    occurrence, regardless of its outcome -- a failed newest night is still
    the newest observed one."""
    conn, clock, _ = catalog(tmp_path)
    try:
        seed_delivered_health_release(conn, release_id="schema-test",
                                      requested_session="2026-10-05",
                                      resolved_session="2026-10-02")
        record_check(conn, "2026-10-02", "engineering", True, {"attempt": 1})
        record_check(conn, "2026-10-05", "engineering", False, {"attempt": 1})
        report = health(conn, clock=clock)
    finally:
        conn.close()
    assert report["code_budgets"]["latest_occurrence"] == "2026-10-05"
