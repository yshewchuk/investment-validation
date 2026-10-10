"""The committed phase-2 coverage baseline names the suite that exists now."""
from __future__ import annotations

import json
from pathlib import Path

from checks import v2_coverage_ratchet as ratchet

ROOT = Path(__file__).resolve().parents[3]


def test_stored_suite_version_matches_the_current_suite():
    baseline = json.loads((ROOT / "checks/v2_coverage_ratchet_phase2_baseline.json").read_text())
    suite = ratchet.phase2_suite(ROOT)
    assert sorted(baseline["test_files"]) == suite
    assert baseline["suite_version"] == ratchet.phase2_suite_version(suite)