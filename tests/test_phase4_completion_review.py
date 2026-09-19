import json
from copy import deepcopy
from pathlib import Path

import pytest

from checks import phase4_real
from checks.phase4_completion_review import review
from checks.phase4_real import build_evidence
from checks.rearchitecture_phase4_gate import check as foundation_gate

pytestmark = pytest.mark.needs_corpus  # reads fixtures/tier0 or another untracked fixture tree

ROOT = Path(__file__).resolve().parents[1]


def _evidence(tmp_path):
    return build_evidence(ROOT / "fixtures/tier0", tmp_path)


def test_full_saved_release_compared_is_false_when_comparison_incomplete(tmp_path):
    """Planted-defect proof (R4-12): this repo's own `fixtures/tier0` corpus
    has zero verified traces right now (every record disposes
    `input_trace: missing`), so the real saved-release comparison is 0/20
    compared. `full_saved_release_compared` must go False here even though
    the champion-artifact registry is fully intact -- proving the control
    now tracks the real record-by-record comparison, not merely champion
    artifact hashes (the bug this task fixes)."""
    evidence = _evidence(tmp_path)
    assert evidence["population"]["compared"] == 0
    assert evidence["population"]["incomparable"] > 0
    assert evidence["completion_controls"]["full_saved_release_compared"] is False
    assert evidence["completion_controls"]["champion_artifacts_verified"] is True


def test_complete_report_written_is_false_for_a_json_dump_report(tmp_path):
    """Planted-defect proof (R4-12): the old control accepted a bare JSON
    dump pasted under a Markdown header as "the report" merely because a
    file existed. Feed that exact shape to the new content check and it
    must be rejected."""
    evidence = _evidence(tmp_path)
    fake_old_style_report = (
        "# Phase 4 scoring acceptance\n\n"
        + json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    assert phase4_real._report_is_complete(fake_old_style_report, evidence) is False


def test_complete_report_written_is_false_when_a_section_is_missing(tmp_path):
    """Planted-defect proof (R4-12): truncate the real, rendered report so a
    required section (Provenance onward) is absent, and the control must
    reject it even though the file exists and starts out well-formed."""
    evidence = _evidence(tmp_path)
    report_path = tmp_path / "phase4_report.md"
    truncated = report_path.read_text().split("## 8. Provenance")[0]
    assert phase4_real._report_is_complete(truncated, evidence) is False


def test_native_full_release_gate_is_phase4_completion(tmp_path):
    evidence = _evidence(tmp_path)
    assert foundation_gate(evidence)["ok"] is True

    result = review(ROOT, evidence)

    assert result["status"] == "COMPLETE"
    assert result["complete"] is True
    assert result["blockers"] == []


def test_phase5_boolean_is_accepted_when_the_interface_exists(tmp_path):
    evidence = deepcopy(_evidence(tmp_path))
    evidence["phase5_inference_integrated"] = True

    result = review(ROOT, evidence)
    assert not any(row["blocker_id"] == "P4-B01" for row in result["blockers"])
