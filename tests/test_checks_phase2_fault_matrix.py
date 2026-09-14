"""D11 CLI: ``checks/rearchitecture_phase2_fault_matrix.py``.

Covers all fifteen §7.3 points and proves the validator's
``FAULT_MATRIX_INCOMPLETE`` check both passes a complete matrix and fails
when one point is missing — the negative control the brief calls for.
"""
from __future__ import annotations

import json

from checks import rearchitecture_phase2_evidence as p2evidence
from checks import rearchitecture_phase2_fault_matrix as fm


def test_build_covers_every_declared_fault_point(tmp_path):
    rows = fm.build(tmp_path)
    assert [row["point"] for row in rows] == list(fm.FAULT_POINTS)
    assert set(fm.FAULT_POINTS) == set(p2evidence.FAULT_POINTS)
    for row in rows:
        assert row["outcome"] in p2evidence.FAULT_OUTCOMES
        assert row["verified_objects"] is True


def test_cli_publishes_and_matrix_passes_the_validator(tmp_path):
    artifact_root = tmp_path / "artifacts"
    code = fm.main(["--artifact-root", str(artifact_root)])
    assert code == 0
    data = (artifact_root / "fault_matrix.json").read_bytes()
    findings: list = []
    field_ok: dict = {}
    p2evidence._check_fault_matrix(data, findings, field_ok)
    assert findings == []
    assert field_ok == {}


def test_removing_one_point_fails_the_validator(tmp_path):
    rows = fm.build(tmp_path)
    incomplete = [row for row in rows if row["point"] != "before_commit"]
    data = json.dumps(incomplete).encode()
    findings: list = []
    field_ok: dict = {}
    p2evidence._check_fault_matrix(data, findings, field_ok)
    assert [f["code"] for f in findings] == ["FAULT_MATRIX_INCOMPLETE"]
    assert findings[0]["missing"] == ["before_commit"]
    assert field_ok["fault_matrix_ref"] is False
