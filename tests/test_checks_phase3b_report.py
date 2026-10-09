"""The generated Phase 3B acceptance report: Markdown, mirrored, honest scope.

R3B-1/R3B-5: `reports/phase3b_acceptance/report.json` was hand-authored,
claimed `"production_acceptance": true`, and (being JSON) never reached the
private mirror. `checks/phase3b_report.py` replaces it with a report rendered
from code over a real run receipt and the acceptance gate's own evaluation of
it. These tests assert the replacement actually holds: the file is Markdown,
the mirror's own `collect()` picks it up, and it never claims full production
acceptance.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import phase3b_report  # noqa: E402
from checks import rearchitecture_phase3b as gate  # noqa: E402
from engine.report import Report  # noqa: E402
from tools.private_mirror import collect  # noqa: E402

RUN_ROOT = phase3b_report.DEFAULT_RUN_ROOT


@pytest.fixture(scope="module")
def real_run():
    if not phase3b_report.DEFAULT_RECEIPT.is_file():
        pytest.skip("frozen phase3b acceptance run is not present on this host")
    receipt = json.loads(phase3b_report.DEFAULT_RECEIPT.read_text())
    evidence = json.loads(phase3b_report.DEFAULT_EVIDENCE.read_text())
    gate_result = gate.evaluate(evidence, artifact_root=RUN_ROOT)
    return receipt, evidence, gate_result


def test_gate_passes_over_the_retained_run(real_run):
    _, _, gate_result = real_run
    assert gate_result["ok"] is True
    assert gate_result["status"] == "PASS"
    assert gate_result["counts"]["passed_subjects"] == 8
    assert gate_result["counts"]["negative_controls_passed"] == 24


def test_report_is_rendered_from_code_over_the_real_receipt(tmp_path, real_run):
    receipt, evidence, gate_result = real_run
    context = phase3b_report.build_context(
        receipt, evidence, gate_result,
        input_files=[phase3b_report.DEFAULT_RECEIPT, phase3b_report.DEFAULT_EVIDENCE])
    path = Report(context).write(tmp_path, filename="report.md")

    assert path.suffix == ".md"
    text = path.read_text()

    # Never claims full production acceptance.
    assert "production_acceptance" not in text

    # States the measured scope explicitly rather than hiding it: the 64-row
    # cap and the real (much larger) table sizes both appear.
    assert "64" in text
    assert "9.1M" in text  # real daily_market size, stated in the accepted-limitation prose
    assert "bounded" in text.lower()
    assert "supervisor decision, 2026-09-18" in text

    # Numbers are pulled from the real receipt, not invented: the source row
    # counts for two tables appear verbatim.
    assert f"{receipt['source_files']['daily_market']['rows_in_source']:,}" in text
    assert f"{receipt['source_files']['option_chains']['rows_in_source']:,}" in text

    # Retained snapshot refs from the evidence are carried through.
    for ref in evidence["retained_snapshot_refs"]:
        assert ref in text


def _render_synthetic_report(tmp_path) -> str:
    """Render the report over a minimal synthetic run, not the frozen one.

    The known-red section is a property of the renderer, not of the retained
    run, so these tests exercise `phase3b_report.sections(...)` through the
    real `Report.write` path over the smallest receipt/evidence/gate triple
    that a passing gate can sit on top of.
    """
    receipt = {
        "source_files": {
            "synthetic_table": {
                "rows_in_source": 1,
                "path": "synthetic.parquet",
                "content_hash": "0" * 64,
            },
        },
        "table_results": {
            "synthetic_table": {
                "rows_before": 0,
                "no_op_rewrites": 0,
                "no_op_committed": True,
                "retry_idempotent": True,
                "changed_partitions": [],
                "rebuild_equal": True,
                "persisted_rows_equal": True,
                "downstream_results_equal": True,
                "clean_rebuild_rows": 0,
            },
        },
        "runtime_ms": 1,
        "peak_rss_bytes": 1,
    }
    evidence = {"retained_snapshot_refs": []}
    gate_result = {
        "ok": True,
        "status": "PASS",
        "counts": {
            "registered_subjects": 8,
            "passed_subjects": 8,
            "failed_subjects": 0,
            "negative_controls_total": 0,
            "negative_controls_passed": 0,
        },
        "negative_controls": {},
        "subjects": {},
        "findings": [],
    }
    context = {
        "kind": "audit",
        "spec": {
            "id": "REARCH-PHASE-3B-ACCEPTANCE",
            "title": "Rearchitecture Phase 3B — incremental EOD data acceptance",
            "type": "descriptive",
        },
        "results": {},
        "headline": {},
        "backtest": {},
        "checklist": [],
        "provenance": {},
        "survivorship_note": "",
        "calibration": None,
        "funnel": [],
        "extra_sections": phase3b_report.sections(receipt, evidence, gate_result),
    }
    path = Report(context).write(tmp_path, filename="report.md")
    return path.read_text()


def test_report_lists_zero_known_red_tests_alongside_the_gate_pass(tmp_path, monkeypatch):
    # The acceptance gate passing must never be allowed to read as "the suite
    # is green" -- the report keeps the known-red section and its "not the
    # suite" caveat even when the list is empty. The former synthetic R3B-7
    # test was rechecked 2026-10-09 and passes, so it is no longer named.
    monkeypatch.setattr(phase3b_report, "KNOWN_RED_TESTS", ())
    text = _render_synthetic_report(tmp_path)
    assert "Known red tests in the wider v2 suite at close" in text
    assert "Not established" in text
    assert "0 known red test(s)" in text
    assert ("the acceptance gate PASS covers only its own 8 subjects, not the "
            "wider suite") in text
    assert ("test_action_finality_writes_a_coverage_output_from_monkeypatched_frames"
            not in text)


def test_report_lists_a_recorded_known_red_test_by_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        phase3b_report, "KNOWN_RED_TESTS",
        (("tests/v2/engine/test_synthetic.py::test_known_red", "known red at close"),))
    text = _render_synthetic_report(tmp_path)
    assert "tests/v2/engine/test_synthetic.py::test_known_red" in text
    assert "known red at close" in text
    assert "Not established" in text


def test_report_reaches_its_default_location_and_the_private_mirror(real_run):
    exit_code = phase3b_report.main([])
    assert exit_code == 0

    report_path = phase3b_report.OUT_DIR / "report.md"
    assert report_path.is_file()
    assert report_path.suffix == ".md"

    found, _ = collect()
    assert report_path in found, (
        "the generated report is not matched by tools/private_mirror.py's "
        "INCLUDE — it will not reach the private mirror"
    )

    text = report_path.read_text()
    assert "production_acceptance" not in text
