"""Behavioral test for the health() producer schema version."""
from engine.v2.ops.health import health
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
