from __future__ import annotations

from engine.v2.ops.native_parity_report import (
    PARITY_DIMENSIONS,
    _row_mismatches,
    compare_native_vs_legacy,
)
from engine.v2.parity.tolerance import SCORE_RECORD_V1

_PLACEHOLDER = {"n_analogs": 0, "ci_low": None, "ci_high": None}


def _analog_findings(legacy, native):
    found = _row_mismatches("k", legacy, native, ("analogs",), SCORE_RECORD_V1)
    return [f for m in found for f in m["finding_fields"]]


def test_placeholder_zero_with_absent_native_agrees():
    assert _analog_findings(dict(_PLACEHOLDER), {}) == []


def test_real_zero_analog_result_still_mismatches_absent_native():
    # a real run: n_analogs 0 but the CI was set, so it is not a placeholder
    legacy = {"n_analogs": 0, "ci_low": -1.5, "ci_high": 2.0}
    assert "n_analogs" in _analog_findings(legacy, {})


def test_placeholder_zero_against_native_value_still_mismatches():
    assert "n_analogs" in _analog_findings(dict(_PLACEHOLDER), {"n_analogs": 7})


def test_placeholder_zero_against_native_zero_agrees_only_with_native_ci_absent():
    # native wrote 0 itself: a native value exists, so legacy 0 compares as 0
    assert _analog_findings(dict(_PLACEHOLDER), {"n_analogs": 0}) == []


def test_no_pool_rows_are_listed_and_stay_mismatches():
    legacy = {
        "a|S|2026-01-02": {"n_analogs": 1013, "ci_low": 0.1, "ci_high": 0.9},
        "b|S|2026-01-02": dict(_PLACEHOLDER),
        "c|S|2026-01-02": {"n_analogs": 50, "ci_low": 0.1, "ci_high": 0.9},
    }
    native = {
        "a|S|2026-01-02": {},
        "b|S|2026-01-02": {},
        "c|S|2026-01-02": {"n_analogs": 50, "ci_low": 0.1, "ci_high": 0.9},
    }
    report = compare_native_vs_legacy(legacy, native, ("analogs",))
    assert report["analog_no_pool"] == ["a|S|2026-01-02"]
    assert [m["row_key"] for m in report["mismatches"]] == ["a|S|2026-01-02"]
    assert "n_analogs" in report["mismatches"][0]["finding_fields"]
    assert "analogs" in PARITY_DIMENSIONS
