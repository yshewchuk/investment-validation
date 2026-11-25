from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
from pathlib import Path

import pytest
import yaml

from engine.v2.models.frozen_state_io import read_bounded_state
from engine.v2.ops.fingerprints import source_closure
from experiments.size_search_binding import _source_files

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "tests.yml"
_GIT_OVERRIDES = {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"}


def closure_a(root):
    """Sealed v14 panel closure."""
    return source_closure(root, ["experiments/size_panel.py"])


def closure_b(root):
    """Sealed v14 search closure, plus the model registry bytes (as size_search_binding builds it)."""
    keys = source_closure(root, ["experiments/size_search.py", "experiments/magnitude_search_worker.py"])
    registry = read_bounded_state(root / "engine/models/registry.json", 16 * 1024 * 1024)
    return {**keys, "engine/models/registry.json": "sha256:" + hashlib.sha256(registry).hexdigest()}


def _is_exp190_change(paths):
    """True if any changed path is in EXP-190's scope (size_quantile modules or EXP-190 directories)."""
    return any("size_quantile" in path or "EXP-190" in path for path in paths)


def _diff_keys(left, right):
    return sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))


def _clean_env():
    return {key: value for key, value in os.environ.items() if key not in _GIT_OVERRIDES}


def _git(args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, env=_clean_env(), capture_output=True, text=True, check=False,
                          timeout=120)


def test_replica_matches_the_binding_source_files():
    assert closure_b(ROOT) == _source_files()


def test_closure_keys_never_name_quantile_or_exp_190_files():
    for closure in (closure_a(ROOT), closure_b(ROOT)):
        for key in closure:
            assert "size_quantile" not in key and "EXP-190" not in key, key


def test_path_predicate_scopes_exp_190_changes():
    assert _is_exp190_change(["experiments/size_quantile_x.py"])
    assert _is_exp190_change(["experiments/EXP-190_x/spec.yaml"])
    assert not _is_exp190_change(["engine/fills.py"])
    assert _is_exp190_change(["engine/fills.py", "experiments/size_quantile_x.py"])


def test_closures_are_unchanged_since_the_merge_base(tmp_path):
    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    base_run = _git(["merge-base", "HEAD", "origin/main"])
    base = base_run.stdout.strip()
    if base_run.returncode != 0 or not base:
        workflow_name = yaml.safe_load(read_bounded_state(WORKFLOW, 1 << 20).decode("utf-8"))["name"]
        in_pr_tests = (os.environ.get("GITHUB_EVENT_NAME") == "pull_request"
                       and os.environ.get("GITHUB_WORKFLOW") == workflow_name)
        if in_pr_tests:
            pytest.fail("merge-base of HEAD and origin/main cannot be resolved in the PR tests workflow")
        pytest.skip("merge-base of HEAD and origin/main cannot be resolved here")

    diff = _git(["diff", "--name-only", "--no-renames", base])
    assert diff.returncode == 0, diff.stderr
    untracked = _git(["ls-files", "--others", "--exclude-standard"])
    assert untracked.returncode == 0, untracked.stderr
    changed = set(filter(None, diff.stdout.splitlines()))
    changed |= set(filter(None, untracked.stdout.splitlines()))
    if not _is_exp190_change(changed):
        pytest.skip("not an EXP-190 change")

    tar_path = tmp_path / "base.tar"
    archive = _git(["archive", "--format=tar", "-o", str(tar_path), base])
    assert archive.returncode == 0, archive.stderr
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    with tarfile.open(tar_path) as tar:
        tar.extractall(base_dir, filter="data")

    diff_a = _diff_keys(closure_a(base_dir), closure_a(ROOT))
    assert not diff_a, f"closure A differs from the merge base in: {diff_a}"
    diff_b = _diff_keys(closure_b(base_dir), closure_b(ROOT))
    assert not diff_b, f"closure B differs from the merge base in: {diff_b}"
