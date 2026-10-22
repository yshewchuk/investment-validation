"""Run-record layout, EXP number resolution, trimming, and hygiene pins."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from checks.repo_hygiene import Report, check_path_policy
from engine.v2.ops.errors import OpsError, fail
from experiments import lib, record_export
from experiments.record_export import (
    MAX_JSON_BYTES,
    export_failure_cause,
    export_native_run,
    export_run_record,
    public_result,
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


def test_export_failure_cause_maps_ops_os_and_value_errors():
    assert export_failure_cause(fail("RESOURCE_UNAVAILABLE", "x")) == "RESOURCE_UNAVAILABLE"
    assert export_failure_cause(fail("EXPERIMENT_IDENTITY_CONFLICT", "y")) == "EXPERIMENT_IDENTITY_CONFLICT"
    assert export_failure_cause(OSError("disk full")) == "RESOURCE_UNAVAILABLE"
    assert export_failure_cause(ValueError("bad json")) == "INVALID_EXPERIMENT_SPEC"


def test_export_failure_cause_reraises_other_exceptions():
    error = TypeError("unexpected")
    with pytest.raises(TypeError) as caught:
        export_failure_cause(error)
    assert caught.value is error


def test_public_result_keeps_only_status_and_trimmed():
    result = {"dir": "/tmp/EXP-101_x", "status": "written", "trimmed": ["event_ids.json"]}
    assert public_result(result) == {"status": "written", "trimmed": ["event_ids.json"]}
    assert "dir" not in public_result(result)


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
    report = (folder / "REPORT.md").read_text()
    assert "EXP-101 x" in report
    assert "Status: ran" in report
    assert report_text in report
    man = json.loads((results / "export_manifest.json").read_bytes())
    assert man["experiment_id"] == "x"
    assert man["exp_number"] == 101
    assert man["recording_mode"] == "ledger"
    assert man["trimmed"] == []
    assert set(man["files"]) == {"spec.json", "request.json", "event_ids.json", "outcome.json"}
    for name in man["files"]:
        assert man["files"][name] == hashlib.sha256((results / name).read_bytes()).hexdigest(), name
    assert json.loads((results / "spec.json").read_bytes()) == spec
    assert json.loads((results / "request.json").read_bytes()) == request
    assert json.loads((results / "event_ids.json").read_bytes()) == event_ids
    assert json.loads((results / "outcome.json").read_bytes()) == outcome
    assert result["status"] == "written"
    assert result["dir"] == str(folder)


def test_direct_export_for_linked_id_with_other_number_conflicts(tmp_path):
    _export(tmp_path, exp_number=101, experiment_id="x")
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=102, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not any(child.name.startswith("EXP-102_") for child in tmp_path.iterdir())


def test_resolve_linked_number_from_manifest(tmp_path):
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


def test_resolve_linked_id_with_duplicate_number_folder_conflicts(tmp_path):
    _export(tmp_path, exp_number=101, experiment_id="x")
    other = tmp_path / "EXP-101_other"
    other.mkdir()
    _write_manifest(other, "y", 101)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


@pytest.mark.parametrize("experiment_id", [None, ""])
def test_resolve_bad_experiment_id_is_invalid(tmp_path, experiment_id):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, experiment_id, 101)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


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


def test_resolve_non_ascii_digit_string_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", "²")
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_number_above_max_is_invalid(tmp_path):
    for bad in (1_000_000_000, "1000000000"):
        with pytest.raises(OpsError) as exc:
            resolve_exp_number(tmp_path, "exp-x", bad)
        assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_preexisting_oversize_spec_conflicts(tmp_path):
    _write_manifest(tmp_path / "EXP-101_x", "x", 101)
    results = tmp_path / "EXP-101_x" / "results"
    planted = (results / "export_manifest.json").read_bytes()
    (results / "spec.json").write_bytes(b"x" * (MAX_JSON_BYTES + 1))
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "spec.json" in str(exc.value)
    assert "unreadable, not a regular file, or oversized" in str(exc.value)
    assert (results / "export_manifest.json").read_bytes() == planted
    assert not (results / "event_ids.json").exists()


def test_symlinked_existing_spec_conflicts(tmp_path):
    _write_manifest(tmp_path / "EXP-101_x", "x", 101)
    results = tmp_path / "EXP-101_x" / "results"
    planted = (results / "export_manifest.json").read_bytes()
    real = tmp_path / "real.json"
    real.write_bytes(b'{"hypothesis": "h"}')
    (results / "spec.json").symlink_to(real)
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "spec.json" in str(exc.value)
    assert (results / "export_manifest.json").read_bytes() == planted
    assert not (results / "event_ids.json").exists()


def test_resolve_auto_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", "auto")
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_exp_number_string_101_returns_101(tmp_path):
    assert resolve_exp_number(tmp_path, "x", "101") == 101


def test_preexisting_unmanifested_other_folder_conflicts(tmp_path):
    (tmp_path / "EXP-101_other").mkdir()
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 101)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_symlinked_run_folder_conflicts_and_writes_nothing(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    (tmp_path / "EXP-101_x").symlink_to(target)
    with pytest.raises(OpsError) as exc:
        _export(tmp_path)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (target / "results").exists()
    assert not (target / "REPORT.md").exists()


def test_manifestless_folder_with_results_refuses_as_crashed_or_partial_export(tmp_path):
    results = tmp_path / "EXP-101_x" / "results"
    results.mkdir(parents=True)
    (results / "spec.json").write_text('{"hypothesis": "h"}')
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "crashed or partial export" in str(exc.value)


def test_preflight_atomic_symlink_outcome_creates_no_partial_writes(tmp_path):
    _write_manifest(tmp_path / "EXP-101_x", "x", 101)
    results = tmp_path / "EXP-101_x" / "results"
    planted = (results / "export_manifest.json").read_bytes()
    real = tmp_path / "elsewhere" / "outcome.json"
    real.parent.mkdir()
    real.write_text('{"failure_code": null}')
    (results / "outcome.json").symlink_to(real)
    with pytest.raises(OpsError) as exc:
        _export(tmp_path)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "outcome.json" in str(exc.value)
    assert not (results / "spec.json").exists()
    assert (results / "export_manifest.json").read_bytes() == planted
    assert not (results / "event_ids.json").exists()


def test_oversize_event_ids_trimmed_return_list(tmp_path):
    ids = [f"evt-{i}" for i in range(200_000)]
    result = _export(tmp_path, event_ids=ids)
    assert result["trimmed"] == ["event_ids.json"]
    doc = json.loads((tmp_path / "EXP-101_x" / "results" / "event_ids.json").read_bytes())
    kept = doc["kept"]
    assert isinstance(kept, list)
    assert len(kept) > 0
    assert kept == ids[:len(kept)]


def test_refusal_case_writes_outcome_with_failure_code_and_replayed(tmp_path):
    registration = _Registration({
        "execution_plan": {"experiment_id": "exp-x"},
        "snapshot": {"snapshot_id": "snap-1"},
        "event_ids": [],
        "source_closure": {},
        "environment": {},
        "magnitude_request": {"magnitude_model": "history_mean"},
        "magnitude_execution": {"magnitude_request": "exec", "shards": 2},
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
    outcome = json.loads((folder / "results" / "outcome.json").read_bytes())
    assert outcome["failure_code"] == "HOLDOUT_ACCESS_DENIED"
    assert outcome["replayed"] is False
    assert outcome["magnitude_request"] == {"magnitude_model": "history_mean"}
    assert outcome["magnitude_execution"] == {"magnitude_request": "exec", "shards": 2}


def test_concurrent_exports_leave_manifest_consistent_with_disk(tmp_path):
    def export(i):
        return _export(tmp_path, exp_number=101, experiment_id="x",
                       outcome={"failure_code": None, "writer": i})
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(export, range(2)))
    assert all(result["dir"] for result in results)
    assert sorted(result["status"] for result in results) == ["unchanged", "written"]
    written_writer = next(i for i, result in enumerate(results) if result["status"] == "written")
    results_dir = tmp_path / "EXP-101_x" / "results"
    man = json.loads((results_dir / "export_manifest.json").read_bytes())
    for name, digest in man["files"].items():
        assert digest == hashlib.sha256((results_dir / name).read_bytes()).hexdigest(), name
    outcome = json.loads((results_dir / "outcome.json").read_text())
    assert outcome["writer"] == written_writer
    assert (tmp_path / ".record_export.lock").is_file()


class _Registration:
    def __init__(self, document):
        self.document = document


def test_resolve_linked_id_reuses_number_with_request_none(tmp_path):
    export_run_record(
        tmp_path,
        exp_number=101,
        experiment_id="x",
        slug="x",
        spec={"hypothesis": "h"},
        request={"action": "run"},
        event_ids=["e1"],
        outcome={"failure_code": None},
        report_text="# report body\n",
        recording_mode="ledger",
    )
    assert resolve_exp_number(tmp_path, "x", None) == 101


def test_resolve_unlinked_id_with_none_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_symlinked_run_folder_with_matching_id_conflicts(tmp_path):
    real = tmp_path / "real-x"
    _write_manifest(real, "x", 101)
    (tmp_path / "EXP-101_x").symlink_to(real)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_export_native_run_independent_oracle(tmp_path, monkeypatch):
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
    started_value = time.monotonic() - 5.0
    monkeypatch.setattr(record_export.time, "monotonic", lambda: started_value + 7.0)
    monkeypatch.setattr(record_export, "resource", SimpleNamespace(
        getrusage=lambda who: SimpleNamespace(ru_maxrss=123456), RUSAGE_SELF=0))
    result = export_native_run(
        tmp_path,
        registration,
        {"hypothesis": "h"},
        {"action": "run"},
        saved,
        exp_number=101,
        replayed=True,
        started=started_value,
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
    assert outcome["elapsed_seconds"] == 7.0
    assert outcome["peak_rss_kb"] == 123456
    proc = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
    )
    if proc.returncode == 0:
        assert outcome["commit_id"] == proc.stdout.strip()
    else:
        assert outcome["commit_id"] is None
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


def test_export_native_run_unreadable_report_is_resource_unavailable(tmp_path):
    registration = _Registration({
        "execution_plan": {"experiment_id": "exp-x"},
        "snapshot": {"snapshot_id": "snap-1"},
        "event_ids": [],
        "source_closure": {},
        "environment": {},
    })
    saved = {
        "outcome": {"failure_code": None},
        "receipt": {"recording_mode": "smoke"},
    }
    with pytest.raises(OpsError) as exc:
        export_native_run(
            tmp_path,
            registration,
            {"hypothesis": "h"},
            {"action": "run"},
            saved,
            exp_number=101,
            replayed=False,
            started=time.monotonic(),
            report_path=tmp_path / "missing-report.md",
        )
    assert exc.value.code == "RESOURCE_UNAVAILABLE"
    assert not any(child.name.startswith("EXP-") for child in tmp_path.iterdir())


def test_large_report_is_truncated_and_replay_unchanged(tmp_path):
    registration = _Registration({
        "execution_plan": {"experiment_id": "exp-x"},
        "snapshot": {"snapshot_id": "snap-1"},
        "event_ids": [],
        "source_closure": {},
        "environment": {},
    })
    saved = {
        "outcome": {"failure_code": None},
        "receipt": {"recording_mode": "smoke"},
    }
    report = tmp_path / "report.md"
    report.write_bytes(b"a" * 2_000_000)
    first = export_native_run(
        tmp_path,
        registration,
        {"hypothesis": "h"},
        {"action": "run"},
        saved,
        exp_number=101,
        replayed=False,
        started=time.monotonic(),
        report_path=report,
    )
    folder = Path(first["dir"])
    report_md = folder / "REPORT.md"
    assert first["status"] == "written"
    assert report_md.stat().st_size <= MAX_JSON_BYTES
    report_md_text = report_md.read_text()
    assert report_md_text.startswith("# EXP-101")
    report_bytes = report.read_bytes()
    first_chars = report_bytes[:1000].decode("utf-8", errors="ignore")
    assert first_chars in report_md_text
    embedded = report_md_text.split("\n\n", 2)[1]
    assert embedded.endswith("[report truncated to fit the 1 MB export cap]")
    second = export_native_run(
        tmp_path,
        registration,
        {"hypothesis": "h"},
        {"action": "run"},
        saved,
        exp_number=101,
        replayed=False,
        started=time.monotonic(),
        report_path=report,
    )
    assert second["status"] == "unchanged"


def test_direct_api_oversized_report_truncated_then_unchanged(tmp_path):
    first = _export(tmp_path, report_text="a" * 2_000_000)
    report_md = tmp_path / "EXP-101_x" / "REPORT.md"
    assert report_md.stat().st_size <= MAX_JSON_BYTES
    text = report_md.read_text()
    assert "[report truncated" in text
    assert first["status"] == "written"
    second = _export(tmp_path, report_text="a" * 2_000_000)
    assert second["status"] == "unchanged"


def test_bogus_git_dir_env_still_gets_real_commit_id(tmp_path, monkeypatch):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    proc = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        env=env,
    )
    expected = proc.stdout.strip() if proc.returncode == 0 else None
    monkeypatch.setenv("GIT_DIR", "/nonexistent/bogus-git-dir")
    registration = _Registration({
        "execution_plan": {"experiment_id": "exp-x"},
        "snapshot": {"snapshot_id": "snap-1"},
        "event_ids": [],
        "source_closure": {},
        "environment": {},
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
        replayed=False,
        started=time.monotonic(),
    )
    folder = Path(result["dir"])
    outcome = json.loads((folder / "results" / "outcome.json").read_bytes())
    assert outcome["commit_id"] == expected


def test_partial_write_leaves_manifestless_folder_and_retry_refuses(tmp_path, monkeypatch):
    real_write_new = record_export._write_new
    calls = {"n": 0}

    def flaky_write_new(path, data):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("simulated partial write")
        return real_write_new(path, data)

    monkeypatch.setattr(record_export, "_write_new", flaky_write_new)
    with pytest.raises(OSError):
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert not (tmp_path / "EXP-101_x" / "results" / "export_manifest.json").exists()
    monkeypatch.setattr(record_export, "_write_new", real_write_new)
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, exp_number=101, experiment_id="x")
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_manifest_number_mismatch_is_identity_conflict(tmp_path):
    folder = tmp_path / "EXP-101_x"
    _write_manifest(folder, "x", 102)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", None)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_slug_none_is_invalid_spec(tmp_path):
    with pytest.raises(OpsError) as exc:
        export_run_record(
            tmp_path,
            exp_number=101,
            experiment_id="x",
            slug=None,
            spec={"hypothesis": "h"},
            request={"action": "run"},
            event_ids=["e1"],
            outcome={"failure_code": None},
            report_text="# report body\n",
            recording_mode="ledger",
        )
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_set_value_in_spec_is_invalid_spec(tmp_path):
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, spec={"bad": set()})
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


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


def test_replay_returns_persisted_trimmed_list(tmp_path, capsys):
    ids = [f"evt-{i}" for i in range(200_000)]
    first = _export(tmp_path, event_ids=ids)
    assert first["trimmed"] == ["event_ids.json"]
    second = _export(tmp_path, event_ids=["a"])
    assert second["trimmed"] == ["event_ids.json"]
    assert second["status"] == "unchanged"
    assert not (tmp_path / "EXP-101_x" / "results" / "runs").exists()
    assert capsys.readouterr().err == ""

    other = tmp_path / "other"
    small_first = _export(other, event_ids=["a"])
    assert small_first["trimmed"] == []
    oversized = _export(other, event_ids=ids)
    assert oversized["trimmed"] == []
    assert not (other / "EXP-101_x" / "results" / "runs").exists()


def test_identical_second_export_is_unchanged(tmp_path, capsys):
    first = _export(tmp_path)
    second = _export(tmp_path)
    assert first["status"] == "written"
    assert second["status"] == "unchanged"
    assert capsys.readouterr().err == ""


def _run_key(spec, request, recording_mode="ledger"):
    return hashlib.sha256(
        _dump(spec) + b"\0" + _dump(request) + b"\0" + recording_mode.encode("utf-8")
    ).hexdigest()[:12]


def test_different_spec_writes_under_results_runs(tmp_path, capsys):
    _export(tmp_path, spec={"hypothesis": "h"})
    top = tmp_path / "EXP-101_x" / "results"
    before = {name: (top / name).read_bytes() for name in RESULT_NAMES}
    before_report = (tmp_path / "EXP-101_x" / "REPORT.md").read_bytes()
    spec = {"hypothesis": "other"}
    request = {"action": "run"}
    result = _export(tmp_path, spec=spec, request=request)
    assert result["status"] == "written"
    key = _run_key(spec, request)
    run_dir = top / "runs" / key
    for name in RESULT_NAMES:
        assert (run_dir / name).is_file(), name
        assert (top / name).read_bytes() == before[name], name
    assert (run_dir / "REPORT.md").is_file()
    assert (tmp_path / "EXP-101_x" / "REPORT.md").read_bytes() == before_report
    assert json.loads((run_dir / "spec.json").read_bytes()) == spec
    again = _export(tmp_path, spec=spec, request=request)
    assert again["status"] == "unchanged"
    assert capsys.readouterr().err == ""


def test_recorded_run_after_smoke_lands_under_runs(tmp_path, capsys):
    spec = {"hypothesis": "h"}
    request = {"action": "run"}
    first = _export(tmp_path, spec=spec, request=request, recording_mode="smoke")
    assert first["status"] == "written"
    top = tmp_path / "EXP-101_x" / "results"
    assert (top / "spec.json").is_file()
    second = _export(tmp_path, spec=spec, request=request, recording_mode="recorded")
    assert second["status"] == "written"
    key = _run_key(spec, request, "recorded")
    run_dir = top / "runs" / key
    for name in RESULT_NAMES:
        assert (run_dir / name).is_file(), name
    assert (run_dir / "REPORT.md").is_file()
    replay = _export(tmp_path, spec=spec, request=request, recording_mode="recorded")
    assert replay["status"] == "unchanged"
    assert capsys.readouterr().err == ""


def test_runs_directory_symlink_is_identity_conflict(tmp_path):
    _export(tmp_path, spec={"hypothesis": "h"})
    top = tmp_path / "EXP-101_x" / "results"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (top / "runs").symlink_to(elsewhere)
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, spec={"hypothesis": "other"})
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not list(elsewhere.iterdir())


def test_replay_tamper_with_spec_bytes_is_identity_conflict(tmp_path):
    _export(tmp_path)
    results = tmp_path / "EXP-101_x" / "results"
    (results / "spec.json").write_bytes(b'{"tampered": true}\n')
    with pytest.raises(OpsError) as exc:
        _export(tmp_path)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_nan_spec_is_invalid_spec(tmp_path):
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, spec={"x": float("nan")})
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_export_direct_number_above_max_is_invalid(tmp_path):
    with pytest.raises(OpsError) as exc:
        export_run_record(
            tmp_path,
            exp_number=1_000_000_000,
            experiment_id="x",
            slug="x",
            spec={"hypothesis": "h"},
            request={"action": "run"},
            event_ids=["e1"],
            outcome={"failure_code": None},
            report_text="# report body\n",
            recording_mode="ledger",
        )
    assert exc.value.code == "INVALID_EXPERIMENT_SPEC"


def test_resolve_through_symlinked_folder_with_fifo_manifest_does_not_hang(tmp_path):
    real = tmp_path / "real"
    results = real / "results"
    results.mkdir(parents=True)
    os.mkfifo(results / "export_manifest.json")
    (tmp_path / "EXP-101_x").symlink_to(real)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "exp-x", 101)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert "symlink" in str(exc.value)


def test_read_capped_on_fifo_returns_none(tmp_path):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    assert record_export._read_capped(fifo) is None


def test_replay_with_manifest_missing_document_name_is_identity_conflict(tmp_path):
    _export(tmp_path)
    manifest_path = tmp_path / "EXP-101_x" / "results" / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["files"]["outcome.json"]
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(OpsError) as exc:
        _export(tmp_path)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_manifest_without_files_object_conflicts_before_runs(tmp_path):
    _export(tmp_path)
    results = tmp_path / "EXP-101_x" / "results"
    (results / "export_manifest.json").write_text(json.dumps(
        {"experiment_id": "x", "exp_number": 101}
    ))
    with pytest.raises(OpsError) as exc:
        _export(tmp_path, spec={"hypothesis": "other"})
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not (results / "runs").exists()


def test_oversize_mapping_spec_is_trimmed_without_kept(tmp_path):
    result = _export(tmp_path, spec={"k": "x" * 1_200_000})
    assert result["trimmed"] == ["spec.json"]
    raw = (tmp_path / "EXP-101_x" / "results" / "spec.json").read_bytes()
    assert len(raw) <= 1_000_000
    doc = json.loads(raw)
    assert doc["trimmed"] is True
    assert "original_bytes" in doc
    assert "sha256" in doc
    assert "kept" not in doc


def test_symlinked_folder_resolution_conflicts(tmp_path):
    target = tmp_path / "target"
    (target / "results").mkdir(parents=True)
    (target / "results" / "export_manifest.json").write_text(json.dumps({
        "experiment_id": "x",
        "exp_number": 101,
        "recording_mode": "ledger",
        "trimmed": [],
        "files": {},
    }))
    (tmp_path / "EXP-101_x").symlink_to(target)
    with pytest.raises(OpsError) as exc:
        resolve_exp_number(tmp_path, "x", 101)
    assert exc.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_different_outcome_keeps_first_outcome_bytes(tmp_path, capsys):
    _export(tmp_path, outcome={"failure_code": None, "v": 1})
    results = tmp_path / "EXP-101_x" / "results"
    outcome_path = results / "outcome.json"
    before = outcome_path.read_bytes()
    result = _export(tmp_path, outcome={"failure_code": "BOOM", "v": 2})
    assert outcome_path.read_bytes() == before
    assert result["status"] == "unchanged"
    man = json.loads((results / "export_manifest.json").read_text())
    assert man["files"]["outcome.json"] == hashlib.sha256(before).hexdigest()
    assert not (results / "runs").exists()
    assert capsys.readouterr().err == ""


def test_hygiene_pin_results_paths_and_report(tmp_path):
    _export(tmp_path, experiment_id="x", slug="x")
    _export(tmp_path, spec={"hypothesis": "other"})
    results = tmp_path / "EXP-101_x" / "results"
    assert results.is_dir()
    for name in RESULT_NAMES:
        assert (results / name).is_file(), name
        rel = f"experiments/EXP-101_x/results/{name}"
        report = Report()
        check_path_policy(rel, report)
        assert len(report.violations) == 1, rel
        assert all(rel in str(violation) for violation in report.violations), rel
        matched = any(
            Path(hit) == results / name
            for pattern in [pattern for directory, pattern in INCLUDE
                            if directory == "experiments"]
            for hit in tmp_path.glob(pattern)
        )
        assert matched, rel
    runs = list(tmp_path.glob("EXP-101_x/results/runs/*/*.json"))
    assert runs
    for run_json in runs:
        matched = any(
            Path(hit) == run_json
            for pattern in [pattern for directory, pattern in INCLUDE
                            if directory == "experiments"]
            for hit in tmp_path.glob(pattern)
        )
        assert matched, run_json
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
