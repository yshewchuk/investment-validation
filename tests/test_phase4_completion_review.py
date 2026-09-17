from copy import deepcopy
from pathlib import Path

from checks.phase4_completion_review import review
from checks.phase4_real import build_evidence
from checks.rearchitecture_phase4_gate import check as foundation_gate

ROOT = Path(__file__).resolve().parents[1]


def _evidence(tmp_path):
    return build_evidence(ROOT / "fixtures/tier0", tmp_path)


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
