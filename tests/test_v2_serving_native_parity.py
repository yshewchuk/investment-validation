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
