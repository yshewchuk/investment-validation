"""Focused tests for the strict Phase 3B acceptance registry and gate."""
from __future__ import annotations

import copy
import json

import pytest

from checks import rearchitecture_phase3b as phase3b


@pytest.fixture(scope="module")
def fixture_evidence():
    return phase3b.deterministic_fixture_evidence()


def _codes(result, subject_id=None):
    return {
        finding["code"] for finding in result["findings"]
        if subject_id is None or finding.get("subject_id") == subject_id
    }


def test_registry_is_fixed_complete_and_uniquely_named():
    rows = phase3b.registry()
    assert [row["subject_id"] for row in rows] == [
        "P3B01", "P3B02", "P3B03", "P3B04",
        "P3B05", "P3B06", "P3B07", "P3B08",
    ]
    assert [row["name"] for row in rows] == [
        "inventory_contracts",
        "coverage_receipt_failures",
        "cache_first_acquisition",
        "daily_market_merge_rebuild_equivalence",
        "correction_tombstone_conflict",
        "dependency_explanation_finality",
        "atomic_fault_recovery",
        "resource_counters",
    ]
    assert len({row["name"] for row in rows}) == len(rows)


def test_deterministic_fixture_exercises_all_subjects_and_machine_fields(fixture_evidence):
    result = phase3b.evaluate(fixture_evidence)
    assert result["ok"] is True
    assert result["status"] == "FIXTURE_PASS"
    assert result["counts"] == {
        "registered_subjects": 8,
        "passed_subjects": 8,
        "failed_subjects": 0,
        "artifact_findings": 0,
        "negative_controls_passed": 24,
        "negative_controls_total": 24,
    }
    assert result["changed_partitions"] == [
        "daily_market/2026", "earnings_events/2026"
    ]
    assert result["no_op_writes"] == 0
    assert all(result["negative_controls"].values())
    assert set(result["subjects"]) == set(phase3b.SUBJECT_BY_ID)


def test_fixture_evidence_is_byte_deterministic():
    first = phase3b.deterministic_fixture_evidence()
    second = phase3b.deterministic_fixture_evidence()
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":"))


def test_missing_evidence_fails_every_registered_subject():
    result = phase3b.evaluate(None)
    assert result["ok"] is False
    assert result["status"] == "FAIL"
    assert result["counts"]["failed_subjects"] == 8
    assert "MISSING_EVIDENCE" in _codes(result)
    assert all(subject["status"] == "FAIL" for subject in result["subjects"].values())


def test_missing_or_corrupt_artifact_fails_closed(fixture_evidence):
    missing = copy.deepcopy(fixture_evidence)
    del missing["artifacts"]["artifact-cache"]
    result = phase3b.evaluate(missing)
    assert result["ok"] is False
    assert "MISSING_ARTIFACT" in _codes(result, "P3B03")
    assert result["counts"]["artifact_findings"] == 1

    corrupt = copy.deepcopy(fixture_evidence)
    corrupt["artifacts"]["artifact-cache"]["inline"]["provider_calls"] = 999
    result = phase3b.evaluate(corrupt)
    assert "ARTIFACT_HASH_MISMATCH" in _codes(result, "P3B03")


@pytest.mark.parametrize(
    ("coverage_state", "completed_keys"),
    [("partial", 2), ("complete", 1), (None, 2)],
)
def test_partial_or_unclassified_coverage_cannot_pass(
        fixture_evidence, coverage_state, completed_keys):
    evidence = copy.deepcopy(fixture_evidence)
    record = evidence["subjects"]["P3B02"]
    if coverage_state is None:
        del record["coverage_state"]
    else:
        record["coverage_state"] = coverage_state
    record["counts"]["completed_keys"] = completed_keys
    result = phase3b.evaluate(evidence)
    assert result["ok"] is False
    assert "PARTIAL_COVERAGE" in _codes(result, "P3B02")
    assert result["subjects"]["P3B02"]["status"] == "FAIL"


def test_noop_writes_and_failed_negative_control_are_visible(fixture_evidence):
    evidence = copy.deepcopy(fixture_evidence)
    evidence["subjects"]["P3B04"]["no_op_writes"] = 1
    evidence["subjects"]["P3B05"]["negative_controls"]["conflict_refused"] = False
    result = phase3b.evaluate(evidence)
    assert result["ok"] is False
    assert result["no_op_writes"] == 1
    assert "NOOP_REWROTE_DATA" in _codes(result, "P3B04")
    assert "NEGATIVE_CONTROL_FAILED" in _codes(result, "P3B05")
    assert result["negative_controls"]["P3B05.conflict_refused"] is False
    assert result["counts"]["negative_controls_passed"] == 23


def test_missing_subject_and_changed_partition_evidence_fail_closed(fixture_evidence):
    evidence = copy.deepcopy(fixture_evidence)
    del evidence["subjects"]["P3B07"]
    evidence["subjects"]["P3B04"]["changed_partitions"] = []
    result = phase3b.evaluate(evidence)
    assert "MISSING_SUBJECT" in _codes(result, "P3B07")
    assert "MISSING_CHANGED_PARTITIONS" in _codes(result, "P3B04")
    assert result["counts"]["failed_subjects"] == 2


def test_fixture_artifacts_cannot_be_relabelled_as_real_acceptance(fixture_evidence):
    fixture_result = phase3b.evaluate(fixture_evidence)
    real_evidence = copy.deepcopy(fixture_evidence)
    real_evidence["evidence_scope"] = "frozen_real_data"
    real_evidence["run_id"] = "frozen-real-data-test"
    real_result = phase3b.evaluate(real_evidence)
    assert fixture_result["status"] == "FIXTURE_PASS"
    assert real_result["status"] == "FAIL"
    assert "FIXTURE_ARTIFACT_FOR_REAL_DATA" in _codes(real_result)
    assert real_result["counts"]["failed_subjects"] == 8


def test_cli_emits_machine_readable_json(capsys):
    assert phase3b.main(["--deterministic-fixture", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["schema_version"] == phase3b.RESULT_SCHEMA
    assert output["status"] == "FIXTURE_PASS"
    assert output["counts"]["registered_subjects"] == 8
