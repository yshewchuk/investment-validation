from __future__ import annotations

from engine.v2.ops.native_parity_report import (
    NATIVE_PARITY_TOLERANCE,
    compare_native_vs_legacy,
)
from engine.v2.parity.receipt import AGREE
from engine.v2.parity.record_comparator import compare_records

_PLACEHOLDER = {"n_analogs": 0, "ci_low": None, "ci_high": None, "flags": []}


def _analog_findings(legacy, native):
    report = compare_native_vs_legacy({"k": legacy}, {"k": native}, ("analogs",))
    return [field for mismatch in report["mismatches"] for field in mismatch["finding_fields"]]


def test_placeholder_zero_with_absent_native_agrees():
    assert _analog_findings(dict(_PLACEHOLDER), {}) == []


def test_empty_pool_run_flagged_thin_still_mismatches_absent_native():
    # the matcher ran and found nothing: n 0, no CI, but THIN_ANALOGS is set
    legacy = {"n_analogs": 0, "ci_low": None, "ci_high": None, "flags": ["THIN_ANALOGS"]}
    assert "n_analogs" in _analog_findings(legacy, {})


def test_row_without_a_flags_list_is_not_a_placeholder():
    legacy = {"n_analogs": 0, "ci_low": None, "ci_high": None}
    assert "n_analogs" in _analog_findings(legacy, {})


def test_zero_with_a_ci_bound_or_partial_native_still_mismatches():
    assert "n_analogs" in _analog_findings(
        {"n_analogs": 0, "ci_low": -1.5, "ci_high": None, "flags": []}, {})
    assert "n_analogs" in _analog_findings(dict(_PLACEHOLDER), {"ci_low": 0.1})


def test_placeholder_zero_against_native_value_still_mismatches():
    assert "n_analogs" in _analog_findings(dict(_PLACEHOLDER), {"n_analogs": 7})


def test_placeholder_zero_against_native_zero_agrees():
    # native wrote 0 itself: a native value exists, so legacy 0 compares as 0
    assert _analog_findings(dict(_PLACEHOLDER), {"n_analogs": 0}) == []


def test_equal_pool_ci_within_ten_percent_of_width_agrees():
    legacy = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": 20, "ci_low": 0.13, "ci_high": 0.47}
    assert _analog_findings(legacy, native) == []
    assert NATIVE_PARITY_TOLERANCE.policy_id == "score_record.parity.v2"


def test_equal_pool_ci_beyond_ten_percent_of_width_flags_low():
    legacy = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": 20, "ci_low": 0.15, "ci_high": 0.5}
    assert _analog_findings(legacy, native) == ["ci_low"]


def test_ci_tolerance_scales_from_legacy_not_native_width():
    legacy = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": 20, "ci_low": 0.136, "ci_high": 0.2}
    findings = _analog_findings(legacy, native)
    assert "ci_low" not in findings
    assert "ci_high" in findings


def test_equal_pool_ci_at_exactly_ten_percent_boundary_agrees():
    legacy = {"n_analogs": 20, "ci_low": 0.0, "ci_high": 1.0}
    native = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 1.0}
    assert _analog_findings(legacy, native) == []


def test_ci_finding_names_tolerance_reason_and_null_mask_is_exact():
    legacy = {"n_analogs": 20, "ci_low": 0.0, "ci_high": 1.0}
    native = {"n_analogs": 20, "ci_low": 0.11, "ci_high": 1.0}
    receipt = compare_records(
        legacy, native, tolerance_policy=NATIVE_PARITY_TOLERANCE)
    assert len(receipt.findings) == 1
    assert receipt.findings[0].tolerance_applied == (
        "within 10% of legacy analog CI width when n_analogs match")
    assert receipt.findings[0].left_value == 0.0
    assert receipt.findings[0].right_value == 0.11

    native_null = {"n_analogs": 20, "ci_low": None, "ci_high": 1.0}
    null_receipt = compare_records(
        legacy, native_null, tolerance_policy=NATIVE_PARITY_TOLERANCE)
    assert len(null_receipt.findings) == 1
    assert null_receipt.findings[0].tolerance_applied == "exact"


def test_accepted_ci_tolerance_agrees_and_stage_hashes_agree():
    legacy = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": 20, "ci_low": 0.13, "ci_high": 0.47}
    receipt = compare_records(
        legacy, native, tolerance_policy=NATIVE_PARITY_TOLERANCE)
    assert receipt.verdict == AGREE
    assert receipt.findings == ()
    assert receipt.stage_hashes
    assert all(row.agrees for row in receipt.stage_hashes)
    assert any(row.stage_id == "analogs" for row in receipt.stage_hashes)


def test_integer_ci_endpoints_use_declared_tolerance():
    legacy = {"n_analogs": 20, "ci_low": 0, "ci_high": 10}
    native = {"n_analogs": 20, "ci_low": 1, "ci_high": 9}
    receipt = compare_records(
        legacy, native, tolerance_policy=NATIVE_PARITY_TOLERANCE)
    assert receipt.verdict == AGREE
    assert receipt.findings == ()


def test_typed_pool_count_mismatch_disables_ci_tolerance():
    legacy = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": 20.0, "ci_low": 0.13, "ci_high": 0.47}
    receipt = compare_records(
        legacy, native, tolerance_policy=NATIVE_PARITY_TOLERANCE)
    assert {f.field_path for f in receipt.findings} == {
        "ci_low", "ci_high", "n_analogs"}


def test_null_n_analogs_does_not_enable_ci_tolerance():
    legacy = {"n_analogs": None, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": None, "ci_low": 0.13, "ci_high": 0.47}
    receipt = compare_records(
        legacy, native, tolerance_policy=NATIVE_PARITY_TOLERANCE)
    assert {f.field_path for f in receipt.findings} == {"ci_low", "ci_high"}


def test_unequal_pool_disables_ci_tolerance():
    legacy = {"n_analogs": 20, "ci_low": 0.1, "ci_high": 0.5}
    native = {"n_analogs": 21, "ci_low": 0.13, "ci_high": 0.47}
    assert set(_analog_findings(legacy, native)) == {"ci_low", "ci_high", "n_analogs"}


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
