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


def test_report_discloses_the_known_red_test_alongside_the_gate_pass(tmp_path, real_run):
    # The acceptance gate passing must never be allowed to read as "the suite
    # is green" -- the report must name the one test known red at close
    # (R3B-7) in the same document as the gate PASS.
    receipt, evidence, gate_result = real_run
    assert gate_result["ok"] is True
    context = phase3b_report.build_context(
        receipt, evidence, gate_result,
        input_files=[phase3b_report.DEFAULT_RECEIPT, phase3b_report.DEFAULT_EVIDENCE])
    path = Report(context).write(tmp_path, filename="report.md")
    text = path.read_text()
    assert "test_action_finality_writes_a_coverage_output_from_monkeypatched_frames" in text
    assert "R3B-7" in text


def test_report_reaches_its_default_location_and_the_private_mirror():
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
