"""Run-record layout, EXP number resolution, trimming, and hygiene pins."""
from __future__ import annotations

import json
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest

from checks.repo_hygiene import Report, check_path_policy
from engine.v2.ops.errors import OpsError
from experiments.record_export import export_run_record, resolve_exp_number
from tools.private_mirror import INCLUDE

RESULT_NAMES = (
    "spec.json",
    "request.json",
    "event_ids.json",
    "outcome.json",
    "export_manifest.json",
)


def _export(root, *, exp_number=1, experiment_id="x", slug=None,
            spec=None, request=None, event_ids=None, outcome=None,
            report_text="# report body\n", recording_mode="ledger"):
    return export_run_record(
        root,
        exp_number=exp_number,
        experiment_id=experiment_id,
        slug=slug if slug is not None else experiment_id,
        spec=spec if spec is not None else {"hypothesis": "h"},
        request=request if request is not None else {"action": "run"},
        event_ids=event_ids if event_ids is not None else ["e1"],
        outcome=outcome if outcome is not None else {"failure_code": None},
        report_text=report_text,
        recording_mode=recording_mode,
    )


def _write_manifest(folder: Path, experiment_id: str, exp_number: int):
    results = folder / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "export_manifest.json").write_text(json.dumps({
        "experiment_id": experiment_id,
        "exp_number": exp_number,
        "recording_mode": "ledger",
        "trimmed": [],
        "files": {},
    }))


def test_layout_writes_results_and_report(tmp_path):
    result = _export(tmp_path, exp_number=1, experiment_id="x", slug="x")
    folder = tmp_path / "EXP-001_x"
    results = folder / "results"
    for name in RESULT_NAMES:
        assert (results / name).is_file(), name
    assert (folder / "REPORT.md").is_file()
    man = json.loads((results / "export_manifest.json").read_text())
    assert man["experiment_id"] == "x"
    assert man["exp_number"] == 1
    assert result["status"] == "written"
    assert result["dir"] == str(folder)


def test_resolve_auto_is_highest_plus_one_ignoring_non_exp(tmp_path):
    (tmp_path / "EXP-001_a").mkdir()
    (tmp_path / "EXP-002_b").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "scratch").mkdir()
    assert resolve_exp_number(tmp_path, "x", "auto") == 3


def test_resolve_auto_on_empty_root_is_one(tmp_path):
    assert resolve_exp_number(tmp_path, "x", "auto") == 1


def test_resolve_linked_id_reuses_number_with_request_none(tmp_path):
    _write_manifest(tmp_path / "EXP-007_linked", "linked", 7)
    assert resolve_exp_number(tmp_path, "linked", None) == 7


def test_resolve_accepts_digit_string(tmp_path):
    assert resolve_exp_number(tmp_path, "x", "12") == 12


def test_resolve_no_number_and_no_link_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_number_held_by_other_experiment_conflicts(tmp_path):
    _write_manifest(tmp_path / "EXP-003_other", "other", 3)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 3)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_resolve_linked_id_requested_other_number_conflicts(tmp_path):
    _write_manifest(tmp_path / "EXP-007_x", "x", 7)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 8)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_unmanifested_folder_conflicts_and_blocks_export(tmp_path):
    (tmp_path / "EXP-005_x").mkdir()
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 5)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=5, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (tmp_path / "EXP-005_x" / "results" / "spec.json").exists()


def test_oversize_event_ids_are_trimmed(tmp_path):
    ids = [f"evt-{i}" for i in range(200_000)]
    _export(tmp_path, event_ids=ids)
    results = tmp_path / "EXP-001_x" / "results"
    raw = (results / "event_ids.json").read_bytes()
    assert len(raw) <= 1_000_000
    doc = json.loads(raw)
    assert "original_bytes" in doc
    assert doc["original_bytes"] > 1_000_000
    assert "sha256" in doc
    man = json.loads((results / "export_manifest.json").read_text())
    assert "event_ids.json" in man["trimmed"]


def test_identical_second_export_is_unchanged(tmp_path):
    first = _export(tmp_path)
    second = _export(tmp_path)
    assert first["status"] == "written"
    assert second["status"] == "unchanged"


def test_different_outcome_keeps_first_outcome_bytes(tmp_path):
    _export(tmp_path, outcome={"failure_code": None, "v": 1})
    outcome_path = tmp_path / "EXP-001_x" / "results" / "outcome.json"
    before = outcome_path.read_bytes()
    result = _export(tmp_path, outcome={"failure_code": "BOOM", "v": 2})
    assert outcome_path.read_bytes() == before
    assert result["status"] == "unchanged"


def test_hygiene_pin_results_paths_and_report(tmp_path):
    _export(tmp_path, experiment_id="x", slug="x")
    results = tmp_path / "EXP-001_x" / "results"
    assert results.is_dir()
    for name in RESULT_NAMES:
        assert (results / name).is_file(), name
        rel = f"experiments/EXP-001_x/results/{name}"
        report = Report()
        check_path_policy(rel, report)
        assert len(report.violations) >= 1, rel
        matched = any(
            rel.startswith(directory + "/")
            and fnmatch(rel[len(directory) + 1:], pattern)
            for directory, pattern in INCLUDE
        )
        assert matched, rel
    repo_root = Path(__file__).resolve().parents[3]
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "experiments/EXP-001_x/REPORT.md"],
            cwd=repo_root,
            capture_output=True,
        )
    except (FileNotFoundError, OSError):
        pytest.skip("git is not available")
    assert proc.returncode == 0
