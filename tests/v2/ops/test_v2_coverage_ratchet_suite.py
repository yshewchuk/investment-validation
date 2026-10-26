"""Coverage-suite membership for relocated ops tests."""
from checks.v2_coverage_ratchet import PHASE1_RENAMED, phase1_suite

def test_relocated_native_health_test_remains_in_phase1_suite():
    assert PHASE1_RENAMED.get("tests/test_v2_ops_health.py") == "tests/v2/ops/test_v2_ops_health.py"
    assert "tests/v2/ops/test_v2_ops_health.py" in phase1_suite()


def test_relocated_outcome_backfill_test_remains_in_phase1_suite():
    assert "tests/v2/ops/test_outcome_session_backfill_wiring.py" in phase1_suite()
