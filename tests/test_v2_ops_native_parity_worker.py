"""Worker semantics for the ``native_parity`` job kind.

``run_native_parity_worker`` reads the three staged inputs (``score.json``
from the paired legacy score job, ``records.json``/``refusals.json`` from
the paired ``native_score_batch`` job), classifies every row through the
pairing core, and writes ``native_parity_report.json``.  Fixture shapes
mirror ``tests/test_v2_ops_native_parity_pairing.py`` and
``tests/test_v2_ops_native_score_batch.py``.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.contracts import ScoreRecord
from engine.v2.foundation import format_timestamp, to_document
from engine.v2.ops import worker
from engine.v2.ops.errors import OpsError
from engine.v2.ops.native.parity_inputs import legacy_parity_rows
from engine.v2.ops.native_parity_report import (
    PARITY_DIMENSIONS,
    SCHEMA_VERSION,
    _empty_native_report,
    _native_comparison_row,
    _stamp_report_identity,
    apply_native_refusals,
    compare_native_vs_legacy,
    run_native_parity_worker,
)
from engine.v2.parity.tolerance import SCORE_RECORD_V1
from tests.ops_support import FakeClock

_OUTPUTS = [{"name": "report", "path": "native_parity_report.json",
             "schema": SCHEMA_VERSION}]


def _legacy_row(ticker="ABC", strategy="STR-THRU", event_date="2026-01-15", **extra):
    return {"ticker": ticker, "strategy": strategy, "event_date": event_date, **extra}


def _canonical_key(ticker="ABC", strategy="STR-THRU", event_date="2026-01-15",
                   session="regular"):
    return "|".join((ticker, strategy, event_date, session))


def _row_key(row):
    return "|".join((row["ticker"], row["strategy"], row["event_date"]))


def _write_inputs(root, *, rows=(), records=None, refusals=None, unkeyable=(),
                  records_schema_version="native_score_batch_records.v2.0",
                  refusals_schema_version="native_score_batch_refusals.v2.0"):
    (root / "score.json").write_text(json.dumps({"rows": list(rows)}))
    (root / "records.json").write_text(json.dumps(
        {"schema_version": records_schema_version, "records": records or {}}))
    (root / "refusals.json").write_text(json.dumps(
        {"schema_version": refusals_schema_version,
         "refusals": refusals or {}, "unkeyable_refusals": list(unkeyable)}))


def _gate_record(gate_score=0.5, gate_pass=True):
    return {"gate_terms": {"gate_score": gate_score,
                           "gate_threshold": None, "gate_pass": gate_pass}}


def _happy_rows_and_records():
    shared = {"gate_score": 0.5, "gate_pass": True}
    first = _legacy_row(ticker="ABC", **shared)
    second = _legacy_row(ticker="XYZ", strategy="MR-PRINT",
                         event_date="2026-01-16", **shared)
    records = {
        _canonical_key("ABC", "STR-THRU", "2026-01-15"): _gate_record(),
        _canonical_key("XYZ", "MR-PRINT", "2026-01-16", session="pm"): _gate_record(),
    }
    return [first, second], records


def _native_record_document(entry_cost_pct=None):
    """One real ``to_document(ScoreRecord)`` -- the nested shape
    ``native_score_batch`` writes into ``records.json``, rather than a
    hand-written flat dict."""
    return to_document(ScoreRecord(
        score_id="native-score-1",
        canonical_request={},
        resolved_request={},
        event_ref={},
        clock_id="clock-1",
        snapshot_ref="snapshot-1",
        dependency_hash="dep-1",
        model_artifact_ids=(),
        selected_contracts=(),
        legs=(),
        entry_exit_plan={},
        quote_provenance={},
        forecasts={
            "driver_prediction": None, "forecast_abs_move": None,
            "runup_move_prediction": None, "exp_pnl_sim": None,
            "chooser_score": None,
        },
        uncertainty={
            "model_p10": None, "model_p90": None, "forecast_p10": None,
            "forecast_p90": None, "forecast_sd": None,
        },
        residual_state_ref=None,
        analog_state_ref=None,
        payoff_state_ref=None,
        feature_values={},
        null_masks={},
        feature_lineage_refs=(),
        gate_terms={"gate_score": None, "gate_threshold": None, "gate_pass": None},
        chooser_candidates=(),
        chooser_selection=None,
        financial_diagnostics={
            "entry_cost_pct": entry_cost_pct, "model_vs_market": None,
            "fair_premium_pct": None, "premium_vs_fair": None,
            "cost_over_width": None,
        },
        requested_payoff_views=(),
        validation_status="scored",
        reason_codes=(),
        warnings=(),
        evidence_refs=(),
    ))


def _read_report(root):
    return json.loads((root / "native_parity_report.json").read_text())


def test_run_native_parity_worker_happy_path(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    result = run_native_parity_worker({"expected_ids": ("a", "b")}, tmp_path)

    assert result["outputs"] == _OUTPUTS
    assert result["completed_ids"] == ["a", "b"]
    assert result["no_work"] is False
    report = _read_report(tmp_path)
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["compared"] == sorted(_row_key(row) for row in rows)
    assert report["only_legacy"] == []
    assert report["only_native"] == []
    assert report["mismatches"] == []
    assert report["native_refused"] == []
    assert report["native_refused_unmatched"] == []


def test_run_native_parity_worker_reports_a_planted_mismatch(tmp_path):
    legacy_row = _legacy_row(ticker="AAA", gate_score=0.5, gate_pass=True)
    native_record = _gate_record(gate_score=0.5, gate_pass=False)
    key = _canonical_key("AAA", "STR-THRU", "2026-01-15")
    _write_inputs(tmp_path, rows=[legacy_row], records={key: native_record})

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["compared"] == [_row_key(legacy_row)]
    assert len(report["mismatches"]) == 1
    mismatch = report["mismatches"][0]
    assert mismatch["row_key"] == _row_key(legacy_row)
    assert mismatch["dimension"] == "verdicts"
    assert "gate_pass" in mismatch["finding_fields"]
    assert mismatch["values"] == {"gate_pass": {"legacy": True, "native": False}}


def test_run_native_parity_worker_flattens_nested_agreeing_record(tmp_path):
    """Fix A: a real nested ``to_document(ScoreRecord)`` must not read as
    all-``None`` -- a genuinely agreeing ``financial_diagnostics`` field is
    compared, not falsely reported."""
    key = _canonical_key("AAA", "STR-THRU", "2026-01-15")
    _write_inputs(tmp_path, rows=[_legacy_row(ticker="AAA", entry_cost_pct=1.23)],
                  records={key: _native_record_document(entry_cost_pct=1.23)})

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["compared"] == ["AAA|STR-THRU|2026-01-15"]
    assert report["mismatches"] == []


def test_run_native_parity_worker_detects_nested_record_difference(tmp_path):
    """Fix A: a changed value inside the nested record must be found, not
    masked by the flat lookup returning ``None``."""
    key = _canonical_key("AAA", "STR-THRU", "2026-01-15")
    _write_inputs(tmp_path, rows=[_legacy_row(ticker="AAA", entry_cost_pct=1.24)],
                  records={key: _native_record_document(entry_cost_pct=1.23)})

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["compared"] == ["AAA|STR-THRU|2026-01-15"]
    assert len(report["mismatches"]) == 1
    mismatch = report["mismatches"][0]
    assert mismatch["dimension"] == "financial_diagnostics"
    assert "entry_cost_pct" in mismatch["finding_fields"]


def test_run_native_parity_worker_rejects_unsupported_records_schema_version(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records,
                  records_schema_version="native_score_batch_records.v1.0")

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details == {
        "schema_version": "native_score_batch_records.v1.0"}


def test_run_native_parity_worker_rejects_unsupported_refusals_schema_version(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records,
                  refusals_schema_version="native_score_batch_refusals.v1.0")

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details == {
        "schema_version": "native_score_batch_refusals.v1.0"}


def test_run_native_parity_worker_rejects_non_mapping_records_document(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)
    (tmp_path / "records.json").write_text(json.dumps(["not", "a", "mapping"]))

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details == {"schema_version": None}


def test_run_native_parity_worker_rejects_non_mapping_refusals_document(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)
    (tmp_path / "refusals.json").write_text(json.dumps(["not", "a", "mapping"]))

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"
    assert exc.value.problem.details == {"schema_version": None}


def test_run_native_parity_worker_empty_legacy_rows_raises(tmp_path):
    _write_inputs(tmp_path, rows=[], records={_canonical_key(): {}})

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"


def test_run_native_parity_worker_all_refused_keyable(tmp_path):
    row = _legacy_row(ticker="AAA")
    key = _canonical_key("AAA", "STR-THRU", "2026-01-15", session="am")
    _write_inputs(tmp_path, rows=[row], records={},
                  refusals={key: {"code": "RELEASE_MISSING_ROLE", "detail": "..."}})

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["compared"] == []
    assert report["only_legacy"] == []
    assert report["native_refused"] == [
        {"row_key": _row_key(row), "refusal_code": "RELEASE_MISSING_ROLE"}]
    assert report["native_refused_unmatched"] == []


def test_run_native_parity_worker_all_refused_unkeyable_only(tmp_path):
    row = _legacy_row(ticker="AAA")
    entry = {"key": {"ticker": "T|X", "strategy": "STR-THRU",
                     "event_date": "2026-01-15", "session": "am"},
             "code": "INVALID_KEY_FIELD", "detail": "bad ticker"}
    _write_inputs(tmp_path, rows=[row], records={}, unkeyable=[entry])

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["compared"] == []
    assert report["only_legacy"] == [_row_key(row)]
    assert report["native_refused"] == []
    assert report["native_refused_unmatched"] == [
        {"row_key": entry["key"], "refusal_code": "INVALID_KEY_FIELD"}]


def test_run_native_parity_worker_disjoint_native_rows_with_keyed_refusal(tmp_path):
    """A matching keyed refusal lets the disjoint native row be reported
    instead of raising."""
    row = _legacy_row(ticker="AAA")
    native_key = _canonical_key("ZZZ", "STR-THRU", "2026-01-15", session="am")
    refusal_key = _canonical_key("AAA", "STR-THRU", "2026-01-15", session="am")
    _write_inputs(tmp_path, rows=[row], records={native_key: {"gate_pass": None}},
                  refusals={refusal_key: {"code": "RELEASE_MISSING_ROLE",
                                          "detail": "..."}})

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["compared"] == []
    assert report["only_legacy"] == []
    assert report["only_native"] == ["ZZZ|STR-THRU|2026-01-15"]
    assert report["native_refused"] == [
        {"row_key": _row_key(row), "refusal_code": "RELEASE_MISSING_ROLE"}]


def test_run_native_parity_worker_intraday_refusal_preserves_timestamp(tmp_path):
    """An intraday refusal keeps its full timestamp in the unmatched entry and
    the report bytes are identical on a deterministic rerun."""
    row = _legacy_row(ticker="AAA")
    refusal_key = _canonical_key(
        "AAA", "STR-THRU", "2026-01-15T09:30:00", session="am")
    _write_inputs(tmp_path, rows=[row], records={},
                  refusals={refusal_key: {"code": "RELEASE_MISSING_ROLE",
                                          "detail": "not applicable"}})

    result = run_native_parity_worker(
        {"expected_ids": ("2026-01-15|scope",)}, tmp_path, clock=FakeClock())

    assert result["outputs"] == _OUTPUTS
    report = _read_report(tmp_path)
    assert report["only_legacy"] == [_row_key(row)]
    assert report["native_refused"] == []
    assert report["native_refused_unmatched"] == [
        {"row_key": "AAA|STR-THRU|2026-01-15T09:30:00",
         "refusal_code": "RELEASE_MISSING_ROLE"}]

    report_bytes = (tmp_path / "native_parity_report.json").read_bytes()
    run_native_parity_worker(
        {"expected_ids": ("2026-01-15|scope",)}, tmp_path, clock=FakeClock())
    assert (tmp_path / "native_parity_report.json").read_bytes() == report_bytes


def test_run_native_parity_worker_unrelated_refusal_does_not_justify_empty_report(tmp_path):
    """A keyed refusal for an unrelated row does not excuse absent native rows for
    the legacy row; the worker still fails validation."""
    row = _legacy_row(ticker="AAA")
    unrelated_key = _canonical_key(
        "ZZZ", "STR-THRU", "2026-01-15T09:30:00", session="am")
    _write_inputs(tmp_path, rows=[row], records={},
                  refusals={unrelated_key: {"code": "RELEASE_MISSING_ROLE",
                                            "detail": "..."}})

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"


def test_run_native_parity_worker_bytes_match_the_comparator_path(tmp_path):
    """The worker's serialized report equals the one the unchanged shared-key
    comparator and stamping path produce for the same inputs."""
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    run_native_parity_worker(
        {"expected_ids": ("a",)}, tmp_path, clock=FakeClock())

    legacy_rows = legacy_parity_rows({"rows": rows})
    native_rows = {
        "|".join(key.split("|")[:3]): _native_comparison_row(record)
        for key, record in records.items()
    }
    expected = compare_native_vs_legacy(
        legacy_rows, native_rows, PARITY_DIMENSIONS,
        tolerance_policy=SCORE_RECORD_V1)
    expected = apply_native_refusals(expected, {}, ())
    expected = _stamp_report_identity(expected, as_of=None, clock=FakeClock())

    assert json.dumps(
        expected, sort_keys=True, separators=(",", ":")).encode() == (
            tmp_path / "native_parity_report.json").read_bytes()


def test_run_native_parity_worker_genuinely_missing_native_rows_raises(tmp_path):
    """Empty native rows with no refusals still raise VALIDATION_FAILED."""
    _write_inputs(tmp_path, rows=[_legacy_row()], records={}, refusals={},
                  unkeyable=[])

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"


def test_run_native_parity_worker_malformed_canonical_key_raises(tmp_path):
    _write_inputs(tmp_path, rows=[_legacy_row()], records={"AAA|CALL_SPREAD": {}})

    with pytest.raises(OpsError) as exc:
        run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)
    assert exc.value.code == "VALIDATION_FAILED"


def test_run_native_parity_worker_empty_expected_ids_is_no_work(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    result = run_native_parity_worker({"expected_ids": ()}, tmp_path)

    assert result["no_work"] is True
    assert result["completed_ids"] == []
    assert (tmp_path / "native_parity_report.json").is_file()
    report = _read_report(tmp_path)
    assert report["as_of"] is None
    assert isinstance(report["generated_at"], str) and report["generated_at"]


def test_run_native_parity_worker_stamps_the_run_identity(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)
    clock = FakeClock()

    run_native_parity_worker(
        {"expected_ids": ("2026-01-15|abc123",)}, tmp_path, clock=clock)

    report = _read_report(tmp_path)
    assert report["as_of"] == "2026-01-15"
    assert report["generated_at"] == format_timestamp(clock.now())
    assert report["mismatches"] == []


def test_run_native_parity_worker_opaque_id_leaves_as_of_none(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    run_native_parity_worker({"expected_ids": ("a",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["as_of"] is None


def test_run_native_parity_worker_non_date_prefix_leaves_as_of_none(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    run_native_parity_worker({"expected_ids": ("opaque|scope",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["as_of"] is None


def test_run_native_parity_worker_empty_prefix_leaves_as_of_none(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    run_native_parity_worker({"expected_ids": ("|scope",)}, tmp_path)

    report = _read_report(tmp_path)
    assert report["as_of"] is None


def test_comparators_stay_free_of_run_identity():
    report = compare_native_vs_legacy(
        {"a|b|c": {"gate_pass": True}}, {"a|b|c": {"gate_pass": True}},
        ("verdicts",))
    assert "as_of" not in report and "generated_at" not in report
    empty = _empty_native_report({"a|b|c": {}}, {}, ("verdicts",), SCORE_RECORD_V1)
    assert "as_of" not in empty and "generated_at" not in empty


def test_dispatch_routes_native_parity_to_the_worker(tmp_path):
    rows, records = _happy_rows_and_records()
    _write_inputs(tmp_path, rows=rows, records=records)

    result = worker.dispatch("native_parity", {"expected_ids": ["a"]}, tmp_path)

    assert result["outputs"] == _OUTPUTS
    assert result["completed_ids"] == ["a"]
    assert result["no_work"] is False
    assert _read_report(tmp_path)["compared"] == sorted(_row_key(row) for row in rows)
