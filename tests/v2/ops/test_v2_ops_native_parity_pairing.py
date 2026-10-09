"""PR-4 pairing core: legacy row keying, the empty-side report, and the
additive native-refusal report fields.  Pure functions only -- no job wiring,
no filesystem, no clock.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.foundation.score_population import population_key
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native.parity_inputs import legacy_parity_rows
from engine.v2.ops.native_parity_report import (
    _empty_native_report,
    _native_rows_and_refusals,
    _population_key_from_board_request_key,
    apply_native_refusals,
)
from engine.v2.ops.native_score_batch import run_native_score_batch_worker
from engine.v2.parity.tolerance import SCORE_RECORD_V1
from tests.test_v2_ops_native_score_batch import (
    _event_doc,
    _stage_release,
    _worker_parameters,
)

_DIMENSIONS = ("forecasts", "simulation", "financial_diagnostics", "verdicts",
               "analogs")


def test_legacy_parity_rows_no_rows_key_returns_empty():
    assert legacy_parity_rows({}) == {}


def test_legacy_parity_rows_keys_by_population_key():
    first = {"ticker": "ABC", "strategy": "gap-crush", "event_date": "2026-01-02",
             "iv_rank": 42.0}
    second = {"ticker": "XYZ", "strategy": "mr-print", "event_date": "2026-01-05",
             "iv_rank": 7.5}
    rows = legacy_parity_rows({"rows": [first, second]})
    assert set(rows) == {"|".join(["ABC", "gap-crush", "2026-01-02"]),
                         "|".join(["XYZ", "mr-print", "2026-01-05"])}
    assert rows["ABC|gap-crush|2026-01-02"] is first
    assert rows["XYZ|mr-print|2026-01-05"] is second


def test_legacy_parity_rows_rejects_non_list_rows():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": "not-a-list"})
    assert exc.value.code == "VALIDATION_FAILED"


def test_legacy_parity_rows_rejects_non_mapping_row():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [{"ticker": "A", "strategy": "S",
                                      "event_date": "2026-01-01"}, "oops"]})
    assert exc.value.code == "VALIDATION_FAILED"


def test_legacy_parity_rows_rejects_missing_field():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [{"ticker": "A", "event_date": "2026-01-01"}]})
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["field"] == "strategy"


def test_legacy_parity_rows_rejects_empty_field():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [{"ticker": "", "strategy": "S",
                                      "event_date": "2026-01-01"}]})
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["field"] == "ticker"


def test_legacy_parity_rows_rejects_non_string_field():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [{"ticker": 123, "strategy": "S",
                                      "event_date": "2026-01-01"}]})
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["field"] == "ticker"


def test_legacy_parity_rows_rejects_delimiter_in_ticker():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [{"ticker": "A|B", "strategy": "S",
                                      "event_date": "2026-01-01"}]})
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["field"] == "ticker"


def test_legacy_parity_rows_rejects_delimiter_in_strategy():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [{"ticker": "A", "strategy": "S|T",
                                      "event_date": "2026-01-01"}]})
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["field"] == "strategy"


def test_legacy_parity_rows_rejects_duplicate_key():
    with pytest.raises(OpsError) as exc:
        legacy_parity_rows({"rows": [
            {"ticker": "A", "strategy": "S", "event_date": "2026-01-01", "price": 1.0},
            {"ticker": "A", "strategy": "S", "event_date": "2026-01-01", "price": 2.0},
        ]})
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details["indices"] == [0, 1]


def test_empty_native_report_all_legacy_only_legacy():
    report = _empty_native_report({"a": {}, "b": {}}, {}, _DIMENSIONS,
                                  SCORE_RECORD_V1)
    assert report["only_legacy"] == ["a", "b"]
    assert report["only_native"] == []
    assert report["compared"] == []
    assert report["mismatches"] == []
    assert report["schema_version"] == "native_parity_report.v1.2"
    assert report["tolerance_policy_id"] == SCORE_RECORD_V1.policy_id


def test_empty_native_report_disjoint_both_sides():
    report = _empty_native_report({"a": {}}, {"z": {}}, _DIMENSIONS,
                                  SCORE_RECORD_V1)
    assert report["only_legacy"] == ["a"]
    assert report["only_native"] == ["z"]


def test_apply_native_refusals_moves_matched_key():
    report = {"only_legacy": ["a", "b"], "compared": [], "only_native": [],
              "mismatches": []}
    updated = apply_native_refusals(report, {"a": "SOME_CODE"})
    assert updated["only_legacy"] == ["b"]
    assert updated["native_refused"] == [{"row_key": "a", "refusal_code": "SOME_CODE"}]
    assert updated["native_refused_unmatched"] == []
    assert report["only_legacy"] == ["a", "b"]


def test_apply_native_refusals_unmatched_key_goes_to_unmatched_list():
    report = {"only_legacy": ["a"], "compared": [], "only_native": [],
              "mismatches": []}
    updated = apply_native_refusals(report, {"z": "OTHER_CODE"})
    assert updated["only_legacy"] == ["a"]
    assert updated["native_refused"] == []
    assert updated["native_refused_unmatched"] == [
        {"row_key": "z", "refusal_code": "OTHER_CODE"}]


def test_apply_native_refusals_unkeyable_always_unmatched():
    report = {"only_legacy": [], "compared": [], "only_native": [],
              "mismatches": []}
    unkeyable = ({"key": {"ticker": "T", "strategy": "S",
                          "event_date": "2026-01-01", "session": "AM"},
                  "code": "INVALID_KEY_FIELD", "detail": "bad session"},)
    updated = apply_native_refusals(report, {}, unkeyable)
    assert updated["native_refused_unmatched"] == [
        {"row_key": {"ticker": "T", "strategy": "S",
                     "event_date": "2026-01-01", "session": "AM"},
         "refusal_code": "INVALID_KEY_FIELD"}]


def test_apply_native_refusals_deterministic_order():
    report = {"only_legacy": [], "compared": [], "only_native": [],
              "mismatches": []}
    updated = apply_native_refusals(report, {"z": "C1", "a": "C2"})
    assert updated["native_refused_unmatched"] == [
        {"row_key": "a", "refusal_code": "C2"},
        {"row_key": "z", "refusal_code": "C1"}]


def test_apply_native_refusals_default_unkeyable_empty():
    report = {"only_legacy": [], "compared": [], "only_native": [],
              "mismatches": []}
    updated = apply_native_refusals(report, {})
    assert updated["native_refused_unmatched"] == []


def test_population_key_from_board_request_key_splits_and_joins():
    assert _population_key_from_board_request_key("ABC|STR-THRU|2026-01-15|am") == (
        "ABC|STR-THRU|2026-01-15")


def test_population_key_from_board_request_key_agrees_with_population_key():
    legacy_row = {"ticker": "ABC", "strategy": "STR-THRU",
                  "event_date": "2026-01-15"}
    assert population_key(legacy_row) == _population_key_from_board_request_key(
        "ABC|STR-THRU|2026-01-15|am")


@pytest.mark.parametrize("bad_key", ["ABC|STR-THRU|2026-01-15", "A|B|C|D|E"])
def test_population_key_from_board_request_key_rejects_wrong_part_count(bad_key):
    with pytest.raises(OpsError) as exc:
        _population_key_from_board_request_key(bad_key)
    assert exc.value.code == "VALIDATION_FAILED"


def test_native_rows_and_refusals_happy_path():
    records_document = {"records": {"ABC|STR-THRU|2026-01-15|am": {"forecast": 1}}}
    refusals_document = {
        "refusals": {"XYZ|STR-THRU|2026-01-20|am": {
            "code": "RELEASE_MISSING_ROLE", "detail": "..."}},
        "unkeyable_refusals": [],
    }
    native_rows, native_refusals, unkeyable_refusals = _native_rows_and_refusals(
        records_document, refusals_document)
    assert native_rows == {"ABC|STR-THRU|2026-01-15": {"forecast": 1}}
    assert native_refusals == {"XYZ|STR-THRU|2026-01-20": "RELEASE_MISSING_ROLE"}
    assert unkeyable_refusals == ()


def test_native_rows_and_refusals_requires_unkeyable_refusals_key():
    with pytest.raises(KeyError):
        _native_rows_and_refusals({"records": {}}, {"refusals": {}})


def test_native_rows_and_refusals_passes_unkeyable_refusals_through():
    entry = {"key": {"ticker": "T|X", "strategy": "STR-THRU",
                     "event_date": "2026-01-15", "session": "am"},
             "code": "INVALID_KEY_FIELD", "detail": "..."}
    refusals_document = {"refusals": {}, "unkeyable_refusals": [entry]}
    native_rows, native_refusals, unkeyable_refusals = _native_rows_and_refusals(
        {"records": {}}, refusals_document)
    assert native_rows == {}
    assert native_refusals == {}
    assert unkeyable_refusals == (entry,)


def test_native_rows_and_refusals_rejects_collision_after_projection():
    records_document = {"records": {"ABC|STR-THRU|2026-01-15|am": {"forecast": 1}}}
    refusals_document = {
        "refusals": {"ABC|STR-THRU|2026-01-15|pm": {
            "code": "RELEASE_MISSING_ROLE", "detail": "..."}},
        "unkeyable_refusals": [],
    }
    with pytest.raises(OpsError) as exc:
        _native_rows_and_refusals(records_document, refusals_document)
    assert exc.value.code == "VALIDATION_FAILED"


def test_native_score_batch_and_native_parity_keys_agree_end_to_end(tmp_path):
    _stage_release(tmp_path)
    root = tmp_path / "staging"
    root.mkdir()
    event_doc = _event_doc()
    event_doc["key"]["ticker"] = "ABC"
    event_doc["calendar_row"]["ticker"] = "ABC"
    (root / "events.json").write_text(json.dumps([event_doc]))
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    records_document = json.loads((root / "records.json").read_text())
    refusals_document = json.loads((root / "refusals.json").read_text())
    native_rows, _, _ = _native_rows_and_refusals(records_document, refusals_document)
    legacy_rows = legacy_parity_rows({"rows": [{
        "ticker": "ABC", "strategy": "STR-THRU",
        "event_date": "2026-01-15", "other_field": 1}]})
    assert len(native_rows) == 1
    assert set(native_rows) == set(legacy_rows)
