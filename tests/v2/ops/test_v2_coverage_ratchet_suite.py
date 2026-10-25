"""Coverage-suite membership for relocated ops tests."""
from checks.v2_coverage_ratchet import phase1_suite

def test_relocated_native_health_test_remains_in_phase1_suite():
    assert "tests/v2/ops/test_v2_ops_health.py" in phase1_suite()
