"""The phase-0 report renders what the gate measured — including when it failed.

The first report hard-coded its controls table and the sentence "all proved
independent", so a failing gate would still have rendered a confident report.
These tests hand the generator a failing gate and assert the report says so.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import rearchitecture_phase0_report as rep  # noqa: E402
from checks.replay_identity import SEEDED_CONTROLS  # noqa: E402


def _controls(problem: str | None = None) -> dict:
    return {cause: {"commit": spec["commit"], "target": f"00{i}_X",
                    "observed": {"record": ["forecast: forecast_abs_move"]},
                    "problems": [problem] if problem else []}
            for i, (cause, spec) in enumerate(SEEDED_CONTROLS.items())}


def gate_doc() -> dict:
    checks = {key: {"ok": True, "seconds": 0.1, "detail": f"{key} fine"}
              for _, key, _ in rep.GATE_ROWS}
    checks["tier0_corpus"].update(
        pairs=18, corpus_version="v1", corpus_hash="sha256:" + "a" * 64,
        cases={"manifest": "agree", "seeded_controls": "agree"}, uncovered_axes=[],
        seeded_controls={"controls": _controls(), "field_set_mismatches": []})
    checks["tier1_real_replay"].update(pairs_replayed=18, findings=[])
    checks["tier1_seeded_controls"].update(controls=_controls())
    return {"ok": True, "failed": [], "code_hash": "sha256:" + "c" * 64, "checks": checks}


@pytest.fixture(autouse=True)
def no_git(monkeypatch):
    monkeypatch.setattr(rep, "_touched", lambda: ([], ["checks/tier0_corpus.py"]))


def section(gate: dict, title: str) -> dict:
    return next(s for s in rep.sections(gate) if s["title"] == title)


def test_a_green_gate_renders_every_claim_met():
    gate = section(gate_doc(), "The phase-0 exit gate")
    assert all(row[1] == "**met**" for row in gate["rows"])
    assert gate["verdict_row"][1].startswith("**Yes**")


def test_a_failing_replay_receipt_renders_as_failed_with_its_reason():
    doc = gate_doc()
    doc["checks"]["tier1_real_replay"].update(ok=False, detail="code changed since the replay ran")
    gate = section(doc, "The phase-0 exit gate")
    row = next(r for r in gate["rows"] if "re-scored by the real engine" in r[0])
    assert row[1] == "**failed**" and "code changed" in row[2]
    assert gate["verdict_row"][1].startswith("**Not yet**")
    assert f"{len(rep.GATE_ROWS) - 1}/{len(rep.GATE_ROWS)}" in gate["verdict_row"][1]
    open_rows = section(doc, "Open, recorded not fixed")["rows"]
    assert any("tier1_real_replay" in r[0] for r in open_rows)


def test_a_failed_control_is_shown_with_its_problem_and_the_verdict_says_no():
    doc = gate_doc()
    doc["checks"]["tier1_seeded_controls"].update(
        ok=False, controls=_controls("record: no finding — the seeded defect went undetected"))
    controls = section(doc, "What the seeded controls showed")
    assert any("undetected" in row[2] for row in controls["rows"])
    assert controls["verdict_row"][1].startswith("**Not yet**")


def test_an_uncovered_axis_is_listed_as_open():
    doc = gate_doc()
    doc["checks"]["tier0_corpus"]["uncovered_axes"] = ["dyn_sv:tie"]
    rows = section(doc, "Open, recorded not fixed")["rows"]
    assert any("dyn_sv:tie" in r[0] for r in rows)


def test_a_legacy_edit_renders_as_broken(monkeypatch):
    monkeypatch.setattr(rep, "_touched", lambda: (["engine/score.py"], ["engine/score.py"]))
    rows = section(gate_doc(), "What phase 0 deliberately did not do")["rows"]
    assert any("**BROKEN**" in r[1] for r in rows)


def test_the_report_never_claims_causal_independence():
    text = json.dumps(rep.sections(gate_doc()))
    assert "proved independent" not in text
