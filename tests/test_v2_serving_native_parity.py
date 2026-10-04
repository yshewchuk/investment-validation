"""Tests for the read-only native-parity summary projection (P?-?).

Fixture reports are built by calling the REAL production functions in
``engine.v2.ops.native_parity_report`` against small synthetic row dicts --
never a hand-written JSON report literal -- so the projection is exercised
against the same shapes the nightly job writes.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from http import HTTPStatus

import pytest

from engine.v2.ops.native_parity_report import (
    PARITY_DIMENSIONS,
    apply_native_refusals,
    compare_native_vs_legacy,
    write_parity_report,
)
from engine.v2.parity.dimensions import (
    ANALOG_FIELDS,
    FINANCIAL_FIELDS,
    FORECAST_FIELDS,
    GATE_FIELDS,
    SIMULATION_FIELDS,
)
from engine.v2.serving.native_parity_projection import (
    NATIVE_PARITY_REPORT_MALFORMED,
    NATIVE_PARITY_SUMMARY_V1,
    native_parity_summary,
)

_DEFAULT_VALUES = {
    name: 0.0
    for name in (
        FORECAST_FIELDS + SIMULATION_FIELDS + FINANCIAL_FIELDS
        + GATE_FIELDS + ANALOG_FIELDS
    )
}


def _row(**overrides):
    row = dict(_DEFAULT_VALUES)
    row["gate_pass"] = True
    row["n_analogs"] = 3
    row.update(overrides)
    return row


def _fixture_rows():
    legacy_rows = {
        "AAPL-2026-01-01": _row(),
        "MSFT-2026-01-02": _row(forecast_p10=1.0),
        "NVDA-2026-01-03": _row(gate_score=0.5, exp_pnl_sim=1.0),
        "AMZN-2026-01-06": _row(ci_low=0.5),
        "TSLA-2026-01-04": _row(),
    }
    native_rows = {
        "AAPL-2026-01-01": _row(),
        "MSFT-2026-01-02": _row(forecast_p10=1.25),
        "NVDA-2026-01-03": _row(gate_score=0.75, exp_pnl_sim=1.5),
        "AMZN-2026-01-06": _row(ci_low=0.75),
        "GOOG-2026-01-05": _row(),
    }
    return legacy_rows, native_rows


def _build_report(legacy_rows, native_rows, native_refusals=None, unkeyable_refusals=()):
    report = compare_native_vs_legacy(legacy_rows, native_rows, PARITY_DIMENSIONS)
    return apply_native_refusals(report, native_refusals or {}, unkeyable_refusals)


_UNKEYABLE_REFUSAL = {
    "key": {"ticker": "ZZZZ", "strategy": "S", "event_date": "2026-01-01",
            "session": "AMC"},
    "code": "INVALID_KEY_FIELD",
    "detail": "x",
}


def test_missing_report_returns_no_report(tmp_path):
    assert native_parity_summary(tmp_path / "absent.json") == (
        HTTPStatus.OK,
        {"schema_version": NATIVE_PARITY_SUMMARY_V1, "status": "no_report"},
    )


def test_malformed_not_json_returns_unavailable(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("not json{")
    status, body = native_parity_summary(path)
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_json_array_returns_unavailable(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("[]")
    status, body = native_parity_summary(path)
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_missing_required_field_returns_unavailable(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "schema_version": "native_parity_report.v1.1",
        "compared": [],
        "only_legacy": [],
        "only_native": [],
    }))
    status, body = native_parity_summary(path)
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_available_full_report_counts_and_worst_rows(tmp_path):
    report = _build_report(
        *_fixture_rows(),
        native_refusals={"TSLA-2026-01-04": "NO_INPUT"},
        unkeyable_refusals=(_UNKEYABLE_REFUSAL,),
    )
    path = write_parity_report(report, tmp_path / "report.json")
    status, body = native_parity_summary(path)
    assert status == HTTPStatus.OK
    assert body["schema_version"] == NATIVE_PARITY_SUMMARY_V1
    assert body["status"] == "available"
    assert body["partial"] is False
    assert body["source_schema_version"] == report["schema_version"]
    assert body["compared_count"] == len(report["compared"])
    assert body["only_legacy_count"] == len(report["only_legacy"])
    assert body["only_native_count"] == len(report["only_native"])
    expected_mismatched_rows = {entry["row_key"] for entry in report["mismatches"]}
    assert body["mismatched_row_count"] == len(expected_mismatched_rows)
    assert body["matched_row_count"] == len(report["compared"]) - len(expected_mismatched_rows)
    assert body["matched_row_count"] + body["mismatched_row_count"] == body["compared_count"]
    expected_field_counts = dict(sorted(Counter(
        field
        for entry in report["mismatches"]
        for field in entry["finding_fields"]
    ).items()))
    assert body["field_mismatch_counts"] == expected_field_counts
    assert set(body["field_mismatch_counts"]) == {
        "ci_low", "exp_pnl_sim", "forecast_p10", "gate_score",
    }
    assert all(count > 0 for count in body["field_mismatch_counts"].values())
    worst_rows = body["worst_rows"]
    assert worst_rows
    for row in worst_rows:
        assert set(row) == {"row_key", "mismatched_field_count", "dimensions"}
    counts = [row["mismatched_field_count"] for row in worst_rows]
    assert counts == sorted(counts, reverse=True)
    assert body["native_refused_count"] == 1
    assert body["native_refused_unmatched_count"] == 1
    assert body["native_refused_reasons"] == {"NO_INPUT": 1, "INVALID_KEY_FIELD": 1}


def test_available_partial_when_refusal_fields_absent(tmp_path):
    legacy_rows, native_rows = _fixture_rows()
    report = compare_native_vs_legacy(legacy_rows, native_rows, PARITY_DIMENSIONS)
    path = write_parity_report(report, tmp_path / "report.json")
    status, body = native_parity_summary(path)
    assert status == HTTPStatus.OK
    assert body["status"] == "available"
    assert body["partial"] is True
    assert body["native_refused_count"] == 0
    assert body["native_refused_unmatched_count"] == 0
    assert body["native_refused_reasons"] == {}
    assert body["matched_row_count"] + body["mismatched_row_count"] == body["compared_count"]


def test_worst_limit_caps_returned_rows(tmp_path):
    report = _build_report(
        *_fixture_rows(),
        native_refusals={"TSLA-2026-01-04": "NO_INPUT"},
        unkeyable_refusals=(_UNKEYABLE_REFUSAL,),
    )
    path = write_parity_report(report, tmp_path / "report.json")
    status, body = native_parity_summary(path, worst_limit=1)
    assert status == HTTPStatus.OK
    totals = Counter()
    for entry in report["mismatches"]:
        totals[entry["row_key"]] += len(entry["finding_fields"])
    assert len({entry["row_key"] for entry in report["mismatches"]}) >= 3
    top_key, top_count = min(totals.items(), key=lambda item: (-item[1], item[0]))
    assert len(body["worst_rows"]) == 1
    assert body["worst_rows"][0]["row_key"] == top_key
    assert body["worst_rows"][0]["mismatched_field_count"] == top_count


def test_symlinked_report_path_is_treated_as_missing(tmp_path):
    report = _build_report(*_fixture_rows())
    report_path = write_parity_report(report, tmp_path / "report.json")
    link = tmp_path / "report-link.json"
    try:
        link.symlink_to(report_path)
    except OSError:
        pytest.skip("symlinks unsupported")
    status, body = native_parity_summary(link)
    assert status == HTTPStatus.OK
    assert body == {"schema_version": NATIVE_PARITY_SUMMARY_V1, "status": "no_report"}


def _full_report(tmp_path):
    report = _build_report(
        *_fixture_rows(),
        native_refusals={"TSLA-2026-01-04": "NO_INPUT"},
        unkeyable_refusals=(_UNKEYABLE_REFUSAL,),
    )
    path = write_parity_report(report, tmp_path / "report.json")
    return path, json.loads(path.read_text())


def test_malformed_schema_version_null_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = dict(report)
    report["schema_version"] = None
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_finding_fields_not_a_list_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    assert report["mismatches"], "fixture must have at least one mismatch"
    report["mismatches"][0]["finding_fields"] = "gate_score"
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_native_refused_present_as_null_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["native_refused"] = None
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_native_refused_unmatched_wrong_type_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["native_refused_unmatched"] = "not-a-list"
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_open_permission_error_returns_unavailable(tmp_path, monkeypatch):
    path, _ = _full_report(tmp_path)

    def _raise_permission_error(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("os.open", _raise_permission_error)

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo unsupported")
def test_fifo_report_path_does_not_hang_and_is_not_available(tmp_path):
    fifo_path = tmp_path / "report.fifo"
    import os
    os.mkfifo(fifo_path)

    status, body = native_parity_summary(fifo_path)

    assert status in (HTTPStatus.OK, HTTPStatus.SERVICE_UNAVAILABLE)
    assert body["status"] in ("no_report", "unavailable")


def test_malformed_mismatch_null_row_key_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["mismatches"][0]["row_key"] = None
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_mismatch_null_dimension_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["mismatches"][0]["dimension"] = None
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_mismatch_row_key_not_in_compared_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["mismatches"][0]["row_key"] = "NOT-A-COMPARED-KEY"
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_duplicate_compared_key_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["compared"] = report["compared"] + [report["compared"][0]]
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_malformed_refusal_code_null_returns_unavailable(tmp_path):
    path, report = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["native_refused"][0]["refusal_code"] = None
    path.write_text(json.dumps(report))

    status, body = native_parity_summary(path)

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


_CAPTURED_GROUPS = ("forecasts", "simulation", "financial_diagnostics", "verdicts", "analogs")
_CAPTURED_PROVENANCE_FIELDS = ("corpus_hash", "fixture_id", "payload_hash",
                               "legacy_request_hash", "native_request_hash", "trace_hash",
                               "same_input_receipt", "frozen_release_id",
                               "native_snapshot_ref")


def _captured_block():
    """A synthetic block shaped exactly like the exporter's v1 capture."""
    return {
        "schema_version": "captured_native_comparison.v1.0",
        "scope": "selected_saved_replay",
        "full_population_verified": False,
        "cutover_qualified": False,
        "current_board": False,
        "identity": {
            "ticker": "AAPL", "strategy": "STR-THRU", "event_date": "2026-01-01",
            "session": "AMC", "as_of": "2026-01-02", "entry_date": "2026-01-05",
            "exit_date": "2026-02-02",
        },
        "clocks": {
            "corpus_as_of": "2026-01-02",
            "requested_decision_at": "2026-01-02T15:30:00-05:00",
            "decision_as_of": "2026-01-02",
            "quote_as_of": "2026-01-02T00:00:00Z",
            "event_date": "2026-01-01",
            "session": "AMC",
        },
        "provenance": {
            "corpus_hash": "sha256-corpus", "fixture_id": "AAPL-2026-01-01",
            "payload_hash": "sha256-payload", "legacy_request_hash": "sha256-legacy-req",
            "native_request_hash": "sha256-native-req", "trace_hash": "sha256-trace",
            "same_input_receipt": "sha256-receipt", "frozen_release_id": "rel-1",
            "native_snapshot_ref": "snap-1",
        },
        "legacy": {group: {"field_a": 1.0, "field_b": None} for group in _CAPTURED_GROUPS},
        "native": {group: {"field_a": 1.25, "field_b": 0.0} for group in _CAPTURED_GROUPS},
        "checks": [{"name": "unused", "passed": True}],
        "numeric_comparisons": [{"field": "field_a", "delta": 0.25}],
        "runtime_stage_count": 3,
    }


def _captured_report(tmp_path, block):
    path, _ = _full_report(tmp_path)
    report = json.loads(path.read_text())
    report["captured_comparison"] = block
    path.write_text(json.dumps(report))
    return path


def _assert_captured_malformed(tmp_path, block):
    status, body = native_parity_summary(_captured_report(tmp_path, block))
    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert body["status"] == "unavailable"
    assert body["reason_code"] == NATIVE_PARITY_REPORT_MALFORMED


def test_captured_block_absent_preserves_existing_summary(tmp_path):
    path, _ = _full_report(tmp_path)
    status, body = native_parity_summary(path)
    assert status == HTTPStatus.OK
    assert body["status"] == "available"
    assert "captured_comparison" not in body


def test_valid_captured_block_projects_bounded_paired_rows(tmp_path):
    block = _captured_block()
    status, body = native_parity_summary(_captured_report(tmp_path, block))
    assert status == HTTPStatus.OK
    assert body["status"] == "available"
    captured = body["captured_comparison"]
    assert set(captured) == {"schema_version", "scope", "full_population_verified",
                             "cutover_qualified", "current_board", "identity", "clocks",
                             "provenance", "rows"}
    assert captured["schema_version"] == "captured_native_comparison.v1.0"
    assert captured["scope"] == "selected_saved_replay"
    assert captured["full_population_verified"] is False
    assert captured["cutover_qualified"] is False
    assert captured["current_board"] is False
    assert captured["identity"] == block["identity"]
    assert captured["clocks"] == block["clocks"]
    assert captured["provenance"] == block["provenance"]
    assert len(captured["clocks"]) == 6
    assert len(captured["provenance"]) == 9
    assert captured["provenance"]["legacy_request_hash"] != \
        captured["provenance"]["native_request_hash"]
    rows = captured["rows"]
    assert len(rows) == 10
    assert [row["group"] for row in rows] == sorted(
        [row["group"] for row in rows], key=_CAPTURED_GROUPS.index)
    for group in _CAPTURED_GROUPS:
        fields = [row["field"] for row in rows if row["group"] == group]
        assert fields == sorted(fields)
    assert rows[0] == {"group": "forecasts", "field": "field_a",
                       "legacy": 1.0, "native": 1.25,
                       "legacy_display": 1.0, "native_display": 1.25}
    assert rows[1] == {"group": "forecasts", "field": "field_b",
                       "legacy": None, "native": 0.0,
                       "legacy_display": None, "native_display": 0.0}
    assert all(set(row) == {"group", "field", "legacy", "native",
                            "legacy_display", "native_display"} for row in rows)


def test_captured_null_stays_distinct_from_zero_and_values_are_not_rounded(tmp_path):
    block = _captured_block()
    block["legacy"]["verdicts"]["field_a"] = 0.30000000000000004
    block["native"]["verdicts"]["field_a"] = 1000000000000000001
    status, body = native_parity_summary(_captured_report(tmp_path, block))
    assert status == HTTPStatus.OK
    rows = {(row["group"], row["field"]): row
            for row in body["captured_comparison"]["rows"]}
    assert rows[("verdicts", "field_a")]["legacy"] == 0.30000000000000004
    assert rows[("verdicts", "field_a")]["native"] == 1000000000000000001
    assert rows[("verdicts", "field_a")]["legacy_display"] == 0.30000000000000004
    assert rows[("verdicts", "field_a")]["native_display"] == "1000000000000000001"
    assert rows[("verdicts", "field_b")]["legacy"] is None
    assert rows[("verdicts", "field_b")]["native"] == 0.0
    assert rows[("verdicts", "field_b")]["legacy_display"] is None
    assert rows[("verdicts", "field_b")]["native_display"] == 0.0
    assert rows[("analogs", "field_b")]["legacy"] is None
    assert rows[("analogs", "field_b")]["legacy_display"] is None


def test_captured_block_present_null_is_malformed(tmp_path):
    _assert_captured_malformed(tmp_path, None)


def test_captured_block_wrong_schema_version_is_malformed(tmp_path):
    block = _captured_block()
    block["schema_version"] = "captured_native_comparison.v1.1"
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_wrong_scope_is_malformed(tmp_path):
    block = _captured_block()
    block["scope"] = "current_board"
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_wrong_nested_type_is_malformed(tmp_path):
    block = _captured_block()
    block["identity"] = "AAPL"
    _assert_captured_malformed(tmp_path, block)
    block = _captured_block()
    block["legacy"]["forecasts"] = [1.0]
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_wrong_strategy_is_malformed(tmp_path):
    block = _captured_block()
    block["identity"]["strategy"] = "STR-OTHER"
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_missing_identity_clock_or_provenance_field_is_malformed(tmp_path):
    for section, fields in (("identity", ("ticker",)),
                            ("clocks", ("quote_as_of",)),
                            ("provenance", _CAPTURED_PROVENANCE_FIELDS)):
        block = _captured_block()
        for name in fields:
            block = _captured_block()
            del block[section][name]
            _assert_captured_malformed(tmp_path, block)


def test_captured_block_extra_provenance_key_is_malformed(tmp_path):
    block = _captured_block()
    block["provenance"]["extra"] = "sha256-extra"
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_clock_contradiction_is_malformed(tmp_path):
    for clock, value in (("decision_as_of", "2026-01-03"),
                         ("event_date", "2026-01-09"),
                         ("session", "REG")):
        block = _captured_block()
        block["clocks"][clock] = value
        _assert_captured_malformed(tmp_path, block)


def test_captured_block_empty_string_identity_is_malformed(tmp_path):
    block = _captured_block()
    block["identity"]["ticker"] = ""
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_flag_int_zero_or_true_is_malformed(tmp_path):
    for flag, value in (("current_board", 0), ("cutover_qualified", 0),
                        ("full_population_verified", 0), ("current_board", True),
                        ("current_board", None)):
        block = _captured_block()
        block[flag] = value
        _assert_captured_malformed(tmp_path, block)


def test_captured_block_bool_or_nonfinite_value_is_malformed(tmp_path):
    for value in (True, False, float("nan"), float("inf"), float("-inf")):
        block = _captured_block()
        block["native"]["simulation"]["field_a"] = value
        _assert_captured_malformed(tmp_path, block)
    block = _captured_block()
    block["legacy"]["simulation"]["field_a"] = "1.0"
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_group_or_field_misalignment_is_malformed(tmp_path):
    block = _captured_block()
    del block["native"]["forecasts"]["field_b"]
    _assert_captured_malformed(tmp_path, block)
    block = _captured_block()
    del block["legacy"]["analogs"]
    _assert_captured_malformed(tmp_path, block)
    block = _captured_block()
    block["native"]["verdicts"] = {}
    _assert_captured_malformed(tmp_path, block)


def test_captured_block_overlong_string_or_name_is_malformed(tmp_path):
    block = _captured_block()
    block["identity"]["ticker"] = "x" * 513
    _assert_captured_malformed(tmp_path, block)
    block = _captured_block()
    long_name = "f" * 129
    block["legacy"]["forecasts"][long_name] = 1.0
    block["native"]["forecasts"][long_name] = 1.0
    _assert_captured_malformed(tmp_path, block)


def _aligned_group_rows(block, per_group, extra_field=None):
    """Aligned legacy/native numeric fields, ``per_group`` per each of the five groups."""
    fields = {f"f{i:03d}": 1.0 for i in range(per_group)}
    for side in ("legacy", "native"):
        block[side] = {group: dict(fields) for group in _CAPTURED_GROUPS}
    if extra_field is not None:
        block["legacy"]["forecasts"][extra_field] = 1.0
        block["native"]["forecasts"][extra_field] = 1.0
    return block


def test_captured_block_overlarge_row_count_is_malformed(tmp_path):
    _assert_captured_malformed(tmp_path, _aligned_group_rows(_captured_block(), 21))
    status, body = native_parity_summary(
        _captured_report(tmp_path, _aligned_group_rows(_captured_block(), 20)))
    assert status == HTTPStatus.OK
    rows = body["captured_comparison"]["rows"]
    assert len(rows) == 100
    assert all(sum(1 for row in rows if row["group"] == group) == 20
               for group in _CAPTURED_GROUPS)
    _assert_captured_malformed(
        tmp_path, _aligned_group_rows(_captured_block(), 20, "f100"))


def test_captured_block_arbitrary_precision_int_is_preserved(tmp_path):
    block = _captured_block()
    block["native"]["simulation"]["field_a"] = 10 ** 400
    status, body = native_parity_summary(_captured_report(tmp_path, block))
    assert status == HTTPStatus.OK
    assert body["status"] == "available"
    rows = {(row["group"], row["field"]): row
            for row in body["captured_comparison"]["rows"]}
    assert rows[("simulation", "field_a")]["native"] == 10 ** 400
    assert rows[("simulation", "field_a")]["native_display"] == str(10 ** 400)


_MAX_SAFE_INTEGER = 2 ** 53 - 1


def test_captured_display_is_exact_decimal_only_beyond_safe_integers(tmp_path):
    block = _captured_block()
    block["legacy"]["forecasts"]["field_a"] = _MAX_SAFE_INTEGER
    block["native"]["forecasts"]["field_a"] = -_MAX_SAFE_INTEGER
    block["legacy"]["simulation"]["field_a"] = _MAX_SAFE_INTEGER + 1
    block["native"]["simulation"]["field_a"] = -(_MAX_SAFE_INTEGER + 1)
    block["legacy"]["financial_diagnostics"]["field_a"] = 1000000000000000001
    block["native"]["financial_diagnostics"]["field_a"] = 10 ** 400
    block["legacy"]["verdicts"]["field_a"] = 0
    block["native"]["verdicts"]["field_a"] = 0.5
    status, body = native_parity_summary(_captured_report(tmp_path, block))
    assert status == HTTPStatus.OK
    rows = {(row["group"], row["field"]): row
            for row in body["captured_comparison"]["rows"]}
    edge = rows[("forecasts", "field_a")]
    assert edge["legacy"] == _MAX_SAFE_INTEGER
    assert edge["legacy_display"] == _MAX_SAFE_INTEGER
    assert type(edge["legacy_display"]) is int
    assert edge["native"] == -_MAX_SAFE_INTEGER
    assert edge["native_display"] == -_MAX_SAFE_INTEGER
    assert type(edge["native_display"]) is int
    beyond = rows[("simulation", "field_a")]
    assert beyond["legacy"] == _MAX_SAFE_INTEGER + 1
    assert beyond["legacy_display"] == str(_MAX_SAFE_INTEGER + 1)
    assert beyond["native"] == -(_MAX_SAFE_INTEGER + 1)
    assert beyond["native_display"] == str(-(_MAX_SAFE_INTEGER + 1))
    huge = rows[("financial_diagnostics", "field_a")]
    assert huge["legacy"] == 1000000000000000001
    assert huge["legacy_display"] == "1000000000000000001"
    assert huge["native"] == 10 ** 400
    assert huge["native_display"] == str(10 ** 400)
    safe = rows[("verdicts", "field_a")]
    assert safe["legacy"] == 0
    assert safe["legacy_display"] == 0
    assert type(safe["legacy_display"]) is int
    assert safe["native"] == 0.5
    assert safe["native_display"] == 0.5
    nulls = rows[("analogs", "field_b")]
    assert nulls["legacy"] is None
    assert nulls["legacy_display"] is None


def test_captured_block_html_like_strings_survive_unchanged(tmp_path):
    hostile = '<img src=x onerror="alert(1)">'
    block = _captured_block()
    block["identity"]["ticker"] = hostile
    block["provenance"]["corpus_hash"] = hostile
    block["legacy"]["verdicts"][hostile] = None
    block["native"]["verdicts"][hostile] = 0.0
    status, body = native_parity_summary(_captured_report(tmp_path, block))
    assert status == HTTPStatus.OK
    captured = body["captured_comparison"]
    assert captured["identity"]["ticker"] == hostile
    assert captured["provenance"]["corpus_hash"] == hostile
    assert set(captured["identity"]) == set(block["identity"])
    assert set(captured["provenance"]) == set(_CAPTURED_PROVENANCE_FIELDS)
    rows = {(row["group"], row["field"]): row for row in captured["rows"]}
    assert rows[("verdicts", hostile)]["legacy"] is None
    assert rows[("verdicts", hostile)]["native"] == 0.0


def _gate_pass_block(legacy_value, native_value):
    block = _captured_block()
    block["legacy"]["verdicts"]["gate_pass"] = legacy_value
    block["native"]["verdicts"]["gate_pass"] = native_value
    return block


def _verdicts_gate_pass_row(body):
    return {(row["group"], row["field"]): row
            for row in body["captured_comparison"]["rows"]}[("verdicts", "gate_pass")]


def test_captured_gate_pass_boolean_pair_passes_through_unchanged(tmp_path):
    status, body = native_parity_summary(
        _captured_report(tmp_path, _gate_pass_block(True, False)))
    assert status == HTTPStatus.OK
    verdict = _verdicts_gate_pass_row(body)
    assert verdict["legacy"] is True and verdict["native"] is False
    assert verdict["legacy_display"] is True and verdict["native_display"] is False
    assert type(verdict["legacy_display"]) is bool
    assert type(verdict["native_display"]) is bool


def test_captured_gate_pass_null_is_accepted_distinctly(tmp_path):
    status, body = native_parity_summary(
        _captured_report(tmp_path, _gate_pass_block(None, False)))
    assert status == HTTPStatus.OK
    verdict = _verdicts_gate_pass_row(body)
    assert verdict["legacy"] is None and verdict["legacy_display"] is None
    assert verdict["native"] is False and verdict["native_display"] is False


def test_captured_gate_pass_rejects_non_boolean_values(tmp_path):
    for value in (0, 1, 0.0, 1.0, "true", "false", "", [True], {"a": 1}):
        _assert_captured_malformed(tmp_path, _gate_pass_block(value, value))


def test_captured_bool_rejected_outside_the_boolean_verdict_field(tmp_path):
    for group, field in (("forecasts", "gate_pass"), ("analogs", "gate_pass"),
                         ("verdicts", "gate_score"), ("verdicts", "field_a"),
                         ("simulation", "field_a")):
        for value in (True, False):
            block = _captured_block()
            block["legacy"][group][field] = value
            block["native"][group][field] = value
            _assert_captured_malformed(tmp_path, block)


def test_captured_lone_surrogate_in_forwarded_string_is_malformed(tmp_path):
    """The captured projection forwards bounded strings verbatim; a lone
    surrogate there is the strict encoder's refusal, not a raw byte write."""
    block = _captured_block()
    block["identity"]["ticker"] = "\ud800"
    _assert_captured_malformed(tmp_path, block)


def test_captured_lone_surrogate_in_forwarded_field_key_is_malformed(tmp_path):
    block = _captured_block()
    name = "field_\ud800"
    for side in ("legacy", "native"):
        block[side]["forecasts"][name] = 1.0
    _assert_captured_malformed(tmp_path, block)
