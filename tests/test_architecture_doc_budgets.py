"""Negative controls for the ARCHITECTURE.md line budget.

Proved by planting what the check exists to catch, same style as
tests/test_code_budgets.py: asserting only that the real tree is green would
pass identically if the check did nothing.
"""
from __future__ import annotations
# land: always-run

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import architecture_doc_budgets as adb  # noqa: E402


def test_the_real_tree_is_within_budget():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "architecture_doc_budgets.py"), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def test_default_mode_reads_staged_not_worktree(tmp_path, monkeypatch):
    """No --all: a staged-only ARCHITECTURE.md change is what gets checked,
    not whatever is sitting in the worktree."""
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY"):
        monkeypatch.delenv(var, raising=False)
    env = {k: v for k, v in os.environ.items()
           if k not in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")}

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, env=env, check=True)
    doc = repo / "ARCHITECTURE.md"
    doc.write_text("x\n" * 5)
    subprocess.run(["git", "add", "ARCHITECTURE.md"], cwd=repo, env=env, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, env=env, check=True)

    # Stage an over-budget version, but leave the worktree file small.
    doc.write_text("x\n" * (adb.EXEMPT["ARCHITECTURE.md"] + 1))
    subprocess.run(["git", "add", "ARCHITECTURE.md"], cwd=repo, env=env, check=True)
    doc.write_text("x\n" * 3)

    default_report = adb.check_files(adb._sources(repo, use_worktree=False))
    assert not default_report.ok

    all_report = adb.check_files(adb._sources(repo, use_worktree=True))
    assert all_report.ok


def test_a_new_doc_over_budget_fails():
    big = ("x\n" * (adb.BUDGET + 1)).encode()
    report = adb.check_files({"engine/v2/newpkg/ARCHITECTURE.md": big})
    assert not report.ok
    assert report.violations[0].path == "engine/v2/newpkg/ARCHITECTURE.md"


def test_a_doc_at_exactly_budget_passes():
    ok = ("x\n" * adb.BUDGET).encode()
    assert adb.check_files({"engine/v2/newpkg/ARCHITECTURE.md": ok}).ok


def test_root_architecture_doc_is_in_scope():
    cap = adb.EXEMPT["ARCHITECTURE.md"]
    over = ("x\n" * (cap + 1)).encode()
    at_cap = ("x\n" * cap).encode()
    assert not adb.check_files({"ARCHITECTURE.md": over}).ok
    assert adb.check_files({"ARCHITECTURE.md": at_cap}).ok


def test_exempt_doc_is_capped_at_its_pinned_size_not_unlimited():
    path = "engine/v2/ops/ARCHITECTURE.md"
    cap = adb.EXEMPT[path]
    over = ("x\n" * (cap + 1)).encode()
    at_cap = ("x\n" * cap).encode()
    assert not adb.check_files({path: over}).ok
    assert adb.check_files({path: at_cap}).ok


def test_non_architecture_markdown_is_not_checked():
    huge = ("x\n" * 5000).encode()
    report = adb.check_files({"engine/v2/ops/README.md": huge})
    assert report.ok
    assert report.docs == 0


def test_worktree_read_failure_raises_not_silently_empty():
    with pytest.raises(OSError):
        adb._read_worktree_strict(ROOT / "no-such-dir-xyz", "ARCHITECTURE.md")


def test_staged_read_failure_raises_not_silently_empty():
    with pytest.raises(subprocess.CalledProcessError):
        adb._read_staged_strict(ROOT, "no/such/tracked/path/ARCHITECTURE.md")


def test_ambient_git_dir_does_not_silently_empty_the_source_list(tmp_path, monkeypatch):
    """An inherited GIT_DIR/GIT_WORK_TREE pointing at an unrelated repo must
    not make tracked_paths (and so _sources) silently see zero files for the
    real root -- that would let the whole budget check pass with docs=0."""
    other = tmp_path / "other_repo"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=other, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=other, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=other, check=True)
    (other / "f.txt").write_text("hi\n")
    subprocess.run(["git", "add", "f.txt"], cwd=other, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=other, check=True)

    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    sources = adb._sources(ROOT, use_worktree=True)
    assert "ARCHITECTURE.md" in sources