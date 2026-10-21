"""Run-record layout, EXP number resolution, trimming, and hygiene pins."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from fnmatch import fnmatch
from pathlib import Path

import pytest

from checks.repo_hygiene import Report, check_path_policy
from engine.v2.ops.errors import OpsError
from experiments import lib
from experiments.record_export import (
    MAX_JSON_BYTES,
    export_native_run,
    export_run_record,
    resolve_exp_number,
)
from tools.private_mirror import INCLUDE

RESULT_NAMES = (
    "spec.json",
    "request.json",
    "event_ids.json",
    "outcome.json",
    "export_manifest.json",
)


def _dump(doc) -> bytes:
    return (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _export(root, *, exp_number=101, experiment_id="x", slug=None,
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
    spec = {"hypothesis": "h"}
    request = {"action": "run"}
    event_ids = ["e1"]
    outcome = {"failure_code": None}
    report_text = "# report body\n"
    result = _export(tmp_path, exp_number=101, experiment_id="x", slug="x",
                     spec=spec, request=request, event_ids=event_ids,
                     outcome=outcome, report_text=report_text)
    folder = tmp_path / "EXP-101_x"
    results = folder / "results"
    for name in RESULT_NAMES:
        assert (results / name).is_file(), name
    assert json.loads((results / "spec.json").read_bytes()) == spec
    assert json.loads((results / "request.json").read_bytes()) == request
    assert json.loads((results / "event_ids.json").read_bytes()) == event_ids
    assert json.loads((results / "outcome.json").read_bytes()) == outcome
    report = (folder / "REPORT.md").read_text()
    assert "EXP-101 x" in report
    assert report_text in report
    man = json.loads((results / "export_manifest.json").read_bytes())
    assert man["experiment_id"] == "x"
    assert man["exp_number"] == 101
    assert man["trimmed"] == []
    for name, digest in man["files"].items():
        assert digest == hashlib.sha256((results / name).read_bytes()).hexdigest(), name
    assert result["status"] == "written"
    assert result["dir"] == str(folder)


def test_resolve_auto_is_highest_plus_one_ignoring_non_exp(tmp_path):
    (tmp_path / "EXP-101_a").mkdir()
    (tmp_path / "EXP-102_b").mkdir()
    (tmp_path / "notes").mkdir()
    (tmp_path / "scratch").mkdir()
    assert resolve_exp_number(tmp_path, "x", "auto") == 103


def test_resolve_auto_on_empty_root_is_first_number(tmp_path):
    assert resolve_exp_number(tmp_path, "x", "auto") == 101


def test_sequential_auto_claims_distinct_empty_folders(tmp_path):
    first = resolve_exp_number(tmp_path, "alpha", "auto")
    second = resolve_exp_number(tmp_path, "beta", "auto")
    assert (first, second) == (101, 102)
    assert (tmp_path / "EXP-101_alpha").is_dir()
    assert (tmp_path / "EXP-102_beta").is_dir()
    result = _export(tmp_path, exp_number=first, experiment_id="alpha", slug="alpha")
    assert result["status"] == "written"
    assert result["dir"] == str(tmp_path / "EXP-101_alpha")


def test_resolve_linked_id_reuses_number_with_request_none(tmp_path):
    _write_manifest(tmp_path / "EXP-107_linked", "linked", 107)
    assert resolve_exp_number(tmp_path, "linked", None) == 107


def test_resolve_linked_number_below_first_is_invalid(tmp_path):
    _write_manifest(tmp_path / "EXP-007_linked", "linked", 7)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "linked", None)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_accepts_digit_string(tmp_path):
    assert resolve_exp_number(tmp_path, "x", "112") == 112


@pytest.mark.parametrize("bad", [1, "1", 0, -1, True, False])
def test_resolve_below_first_number_is_invalid(tmp_path, bad):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", bad)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_export_bool_exp_number_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=True)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_export_small_int_exp_number_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=7)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_no_number_and_no_link_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_number_held_by_other_experiment_conflicts(tmp_path):
    _write_manifest(tmp_path / "EXP-103_other", "other", 103)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 103)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_resolve_linked_id_requested_other_number_conflicts(tmp_path):
    _write_manifest(tmp_path / "EXP-107_x", "x", 107)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 108)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_unmanifested_folder_conflicts_and_blocks_export(tmp_path):
    results = tmp_path / "EXP-105_x" / "results"
    results.mkdir(parents=True)
    (results / "stale.json").write_text("{}")
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 105)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=105, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (results / "spec.json").exists()


def test_resolve_long_digit_string_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", "1" * 40)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_preexisting_oversize_spec_conflicts_without_manifest(tmp_path):
    results = tmp_path / "EXP-101_x" / "results"
    results.mkdir(parents=True)
    (results / "spec.json").write_bytes(b"x" * (MAX_JSON_BYTES + 1))
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (results / "export_manifest.json").exists()


def test_symlinked_existing_spec_conflicts(tmp_path):
    results = tmp_path / "EXP-101_x" / "results"
    results.mkdir(parents=True)
    real = tmp_path / "real.json"
    real.write_bytes(b'{"hypothesis": "h"}')
    (results / "spec.json").symlink_to(real)
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (results / "export_manifest.json").exists()


def test_auto_retry_when_other_slug_claims_same_number(tmp_path, monkeypatch):
    real = lib.next_experiment_number
    calls = {"n": 0}

    def fake_next(root=None):
        calls["n"] += 1
        return 101 if calls["n"] <= 2 else real(root)

    monkeypatch.setattr(
        "experiments.record_export.next_experiment_number",
        fake_next,
    )
    first = resolve_exp_number(tmp_path, "alpha", "auto")
    second = resolve_exp_number(tmp_path, "beta", "auto")
    assert first == 101
    assert second == 102
    exp101 = [p.name for p in tmp_path.iterdir()
              if p.name.startswith("EXP-101_")]
    exp102 = [p.name for p in tmp_path.iterdir()
              if p.name.startswith("EXP-102_")]
    assert exp101 == ["EXP-101_alpha"]
    assert exp102 == ["EXP-102_beta"]


class _Registration:
    def __init__(self, document):
        self.document = document


def test_export_native_run_independent_oracle(tmp_path):
    registration = _Registration({
        "execution_plan": {"experiment_id": "exp-x"},
        "snapshot": {"snapshot_id": "snap-1"},
        "event_ids": ["a", "b"],
        "source_closure": {"f.py": "h"},
        "environment": {"python": "3"},
    })
    saved = {
        "outcome": {"failure_code": None},
        "receipt": {"recording_mode": "smoke"},
    }
    result = export_native_run(
        tmp_path,
        registration,
        {"hypothesis": "h"},
        {"action": "run"},
        saved,
        exp_number=101,
        replayed=True,
        started=time.monotonic(),
    )
    folder = Path(result["dir"])
    outcome = json.loads((folder / "results" / "outcome.json").read_bytes())
    assert outcome["snapshot_id"] == "snap-1"
    assert outcome["recording_mode"] == "smoke"
    assert outcome["replayed"] is True
    expected_hash = hashlib.sha256(
        json.dumps({"f.py": "h"}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert outcome["source_closure_hash"] == expected_hash
    assert outcome["environment"] == {"python": "3"}
    event_ids = json.loads((folder / "results" / "event_ids.json").read_bytes())
    assert event_ids == ["a", "b"]


def test_export_native_run_refusal_reports_failure_code(tmp_path):
    registration = _Registration({
        "execution_plan": {"experiment_id": "exp-x"},
        "snapshot": {"snapshot_id": "snap-1"},
        "event_ids": [],
        "source_closure": {},
        "environment": {},
    })
    saved = {
        "outcome": {"failure_code": "HOLDOUT_ACCESS_DENIED"},
        "receipt": {"recording_mode": "smoke"},
    }
    result = export_native_run(
        tmp_path,
        registration,
        {"hypothesis": "h"},
        {"action": "run"},
        saved,
        exp_number=101,
        replayed=False,
        started=time.monotonic(),
    )
    folder = Path(result["dir"])
    assert "HOLDOUT_ACCESS_DENIED" in (folder / "REPORT.md").read_text()


def test_oversize_manifest_is_ignored_as_unreadable(tmp_path):
    results = tmp_path / "EXP-101_x" / "results"
    results.mkdir(parents=True)
    (results / "export_manifest.json").write_bytes(b"x" * (MAX_JSON_BYTES + 1))
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 101)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_oversize_event_ids_are_trimmed(tmp_path):
    ids = [f"evt-{i}" for i in range(200_000)]
    untrimmed = _dump(ids)
    assert len(untrimmed) > MAX_JSON_BYTES
    _export(tmp_path, event_ids=ids)
    results = tmp_path / "EXP-101_x" / "results"
    raw = (results / "event_ids.json").read_bytes()
    assert len(raw) <= MAX_JSON_BYTES
    doc = json.loads(raw)
    assert doc["trimmed"] is True
    assert doc["original_bytes"] == len(untrimmed)
    assert doc["sha256"] == hashlib.sha256(untrimmed).hexdigest()
    kept = doc["kept"]
    assert 0 < len(kept) < len(ids)
    assert kept == ids[:len(kept)]
    overflow = {
        "trimmed": True,
        "original_bytes": len(untrimmed),
        "sha256": doc["sha256"],
        "kept": ids[:len(kept) + 1],
    }
    assert len(_dump(overflow)) > MAX_JSON_BYTES
    man = json.loads((results / "export_manifest.json").read_text())
    assert "event_ids.json" in man["trimmed"]
    assert man["files"]["event_ids.json"] == hashlib.sha256(raw).hexdigest()


def test_identical_second_export_is_unchanged(tmp_path):
    first = _export(tmp_path)
    second = _export(tmp_path)
    assert first["status"] == "written"
    assert second["status"] == "unchanged"


def test_different_outcome_keeps_first_outcome_bytes(tmp_path):
    _export(tmp_path, outcome={"failure_code": None, "v": 1})
    results = tmp_path / "EXP-101_x" / "results"
    outcome_path = results / "outcome.json"
    before = outcome_path.read_bytes()
    result = _export(tmp_path, outcome={"failure_code": "BOOM", "v": 2})
    assert outcome_path.read_bytes() == before
    assert result["status"] == "unchanged"
    man = json.loads((results / "export_manifest.json").read_text())
    assert man["files"]["outcome.json"] == hashlib.sha256(before).hexdigest()


def test_hygiene_pin_results_paths_and_report(tmp_path):
    _export(tmp_path, experiment_id="x", slug="x")
    results = tmp_path / "EXP-101_x" / "results"
    assert results.is_dir()
    for name in RESULT_NAMES:
        assert (results / name).is_file(), name
        rel = f"experiments/EXP-101_x/results/{name}"
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
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "experiments/EXP-101_x/REPORT.md"],
            cwd=repo_root,
            capture_output=True,
            env=env,
        )
    except (FileNotFoundError, OSError):
        pytest.skip("git is not available")
    assert proc.returncode == 0
