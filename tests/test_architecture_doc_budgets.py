"""Growth-ceiling checks for ARCHITECTURE.md docs.

Proves what the check exists to catch, same style as tests/test_code_budgets.py:
asserting only that the real tree passes would pass identically if the check
did nothing.
"""
from __future__ import annotations
# land: always-run

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import architecture_doc_budgets as adb  # noqa: E402

_LEAK = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY")


def _clean_env():
    return {k: v for k, v in os.environ.items() if k not in _LEAK}


def _git(args, cwd, env=None, check=True):
    return subprocess.run(["git", *args], cwd=cwd, env=env or _clean_env(),
                           capture_output=True, text=True, check=check)


def _init_repo(path):
    path.mkdir()
    _git(["init", "-q"], path)
    _git(["config", "user.email", "t@t"], path)
    _git(["config", "user.name", "t"], path)


def _lines(n):
    return ("x\n" * n).encode()


# --------------------------------------------------------------------------
# check_files: pure, no git
# --------------------------------------------------------------------------


def test_growth_within_limit_passes():
    base = {"a/ARCHITECTURE.md": _lines(500)}
    new = {"a/ARCHITECTURE.md": _lines(500 + adb.MAX_PR_GROWTH)}
    assert adb.check_files(base, new).ok


def test_growth_over_limit_fails():
    base = {"a/ARCHITECTURE.md": _lines(500)}
    new = {"a/ARCHITECTURE.md": _lines(500 + adb.MAX_PR_GROWTH + 1)}
    report = adb.check_files(base, new)
    assert not report.ok
    assert report.violations[0].reason == "growth"


def test_growth_message_says_split_or_move_to_pr_body():
    base = {"a/ARCHITECTURE.md": _lines(10)}
    new = {"a/ARCHITECTURE.md": _lines(10 + adb.MAX_PR_GROWTH + 5)}
    v = adb.check_files(base, new).violations[0]
    assert "split it or move detail to the PR body" in str(v)


def test_doc_at_ceiling_cannot_grow_at_all():
    base = {"a/ARCHITECTURE.md": _lines(adb.CEILING)}
    new = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 1)}
    report = adb.check_files(base, new)
    assert not report.ok
    assert report.violations[0].reason == "ceiling"


def test_ceiling_message_says_needs_a_compression_pr():
    base = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 200)}
    new = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 201)}
    v = adb.check_files(base, new).violations[0]
    assert "doc at ceiling, needs a compression PR" in str(v)


def test_doc_over_ceiling_can_shrink():
    base = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 500)}
    new = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 400)}
    assert adb.check_files(base, new).ok


def test_doc_over_ceiling_unchanged_passes():
    base = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 50)}
    new = {"a/ARCHITECTURE.md": _lines(adb.CEILING + 50)}
    assert adb.check_files(base, new).ok


def test_new_doc_within_limit_passes():
    new = {"a/ARCHITECTURE.md": _lines(adb.MAX_PR_GROWTH)}
    assert adb.check_files({}, new).ok


def test_new_doc_over_limit_fails():
    new = {"a/ARCHITECTURE.md": _lines(adb.MAX_PR_GROWTH + 1)}
    report = adb.check_files({}, new)
    assert not report.ok
    assert report.violations[0].base_lines == 0


def test_removed_doc_is_not_scored():
    base = {"a/ARCHITECTURE.md": _lines(2000)}
    new = {}
    report = adb.check_files(base, new)
    assert report.ok
    assert report.docs == 0


def test_non_architecture_markdown_is_ignored():
    new = {"a/README.md": _lines(5000)}
    report = adb.check_files({}, new)
    assert report.ok
    assert report.docs == 0


def test_growth_can_cross_the_ceiling_within_one_pr_allowance():
    """Deliberate: the growth check and the ceiling check are independent.
    A doc just under the ceiling may cross it in one PR if the crossing is
    itself within MAX_PR_GROWTH -- the freeze only starts applying to the
    NEXT PR, once the doc's base size is at/over CEILING."""
    base = {"a/ARCHITECTURE.md": _lines(adb.CEILING - 10)}
    new = {"a/ARCHITECTURE.md": _lines(adb.CEILING - 10 + adb.MAX_PR_GROWTH)}
    assert adb.check_files(base, new).ok


# --------------------------------------------------------------------------
# _read_base / _resolve_base_ref: real git, no network
# --------------------------------------------------------------------------


def test_read_base_missing_path_returns_none(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "f.txt").write_text("hi\n")
    _git(["add", "f.txt"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    assert adb._read_base(repo, "no/such/ARCHITECTURE.md", "HEAD") is None


def test_read_base_bad_ref_raises(tmp_path):
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "ARCHITECTURE.md").write_text("x\n")
    _git(["add", "ARCHITECTURE.md"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    with pytest.raises(subprocess.CalledProcessError):
        adb._read_base(repo, "ARCHITECTURE.md", "not-a-real-ref")


def test_resolve_base_ref_uses_existing_local_ref_without_fetching(tmp_path, monkeypatch):
    for v in _LEAK:
        monkeypatch.delenv(v, raising=False)
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "f.txt").write_text("hi\n")
    _git(["add", "f.txt"], repo)
    _git(["commit", "-q", "-m", "init"], repo)
    # An unreachable URL: if _resolve_base_ref tried to fetch, this would fail.
    _git(["remote", "add", "origin", "https://example.invalid/nope.git"], repo)
    _git(["update-ref", "refs/remotes/origin/main", "HEAD"], repo)
    assert adb._resolve_base_ref(repo, "main") == "origin/main"


def test_resolve_base_ref_fetches_from_a_local_remote_when_missing(tmp_path, monkeypatch):
    for v in _LEAK:
        monkeypatch.delenv(v, raising=False)
    remote = tmp_path / "remote"
    _init_repo(remote)
    (remote / "ARCHITECTURE.md").write_text("x\n" * 7)
    _git(["add", "ARCHITECTURE.md"], remote)
    _git(["commit", "-q", "-m", "init"], remote)
    _git(["branch", "-M", "main"], remote)

    local = tmp_path / "local"
    _git(["clone", "-q", "--no-local", str(remote), str(local)], tmp_path)
    # A shallow PR checkout has no origin/main tracking ref yet.
    _git(["update-ref", "-d", "refs/remotes/origin/main"], local, check=False)

    ref = adb._resolve_base_ref(local, "main")
    assert ref == "origin/main"
    content = adb._read_base(local, "ARCHITECTURE.md", ref)
    assert content == b"x\n" * 7


# --------------------------------------------------------------------------
# staged vs worktree, and the two hard-won git-env regressions from #181
# --------------------------------------------------------------------------


def test_default_mode_reads_staged_not_worktree(tmp_path, monkeypatch):
    for v in _LEAK:
        monkeypatch.delenv(v, raising=False)
    env = _clean_env()
    repo = tmp_path / "repo"
    _init_repo(repo)
    doc = repo / "ARCHITECTURE.md"
    doc.write_text("x\n" * 100)
    _git(["add", "ARCHITECTURE.md"], repo, env)
    _git(["commit", "-q", "-m", "init"], repo, env)
    _git(["branch", "-M", "main"], repo, env)
    _git(["update-ref", "refs/remotes/origin/main", "HEAD"], repo, env)

    # Stage an over-growth version, but leave the worktree file small.
    doc.write_text("x\n" * (100 + adb.MAX_PR_GROWTH + 1))
    _git(["add", "ARCHITECTURE.md"], repo, env)
    doc.write_text("x\n" * (100 + 5))

    staged_base, staged_new = adb._sources(repo, use_worktree=False, base_ref=None)
    assert not adb.check_files(staged_base, staged_new).ok

    all_base, all_new = adb._sources(repo, use_worktree=True, base_ref=None)
    assert adb.check_files(all_base, all_new).ok


def test_ambient_git_dir_does_not_silently_empty_the_source_list(tmp_path, monkeypatch):
    for v in _LEAK:
        monkeypatch.delenv(v, raising=False)
    env = _clean_env()
    other = tmp_path / "other_repo"
    _init_repo(other)
    (other / "f.txt").write_text("hi\n")
    _git(["add", "f.txt"], other, env)
    _git(["commit", "-q", "-m", "init"], other, env)

    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))

    # base_ref="HEAD" (not origin/main): this test's concern is tracked_paths
    # under an ambient GIT_DIR/GIT_WORK_TREE hijack, not base-ref resolution,
    # and HEAD always resolves regardless of checkout depth -- origin/main
    # may not exist locally in a shallow CI checkout.
    base, new = adb._sources(ROOT, use_worktree=True, base_ref="HEAD")
    assert "ARCHITECTURE.md" in new


def test_git_index_file_is_honored_not_stripped(tmp_path, monkeypatch):
    for v in _LEAK:
        monkeypatch.delenv(v, raising=False)
    env = _clean_env()
    repo = tmp_path / "repo"
    _init_repo(repo)
    doc = repo / "ARCHITECTURE.md"
    doc.write_text("x\n" * 5)
    _git(["add", "ARCHITECTURE.md"], repo, env)
    _git(["commit", "-q", "-m", "init"], repo, env)

    temp_index = tmp_path / "temp-index"
    shutil.copyfile(repo / ".git" / "index", temp_index)
    temp_env = {**env, "GIT_INDEX_FILE": str(temp_index)}
    doc.write_text("x\n" * 90)
    _git(["add", "ARCHITECTURE.md"], repo, temp_env)

    monkeypatch.setenv("GIT_INDEX_FILE", str(temp_index))
    content = adb._read_staged_strict(repo, "ARCHITECTURE.md")
    assert len(content.splitlines()) == 90


# --------------------------------------------------------------------------
# the real tree
# --------------------------------------------------------------------------


def test_the_real_tree_passes_against_main():
    # No --base-ref: this exercises the real auto-detect (and, in a shallow
    # CI checkout with no local origin/main, the fetch-fallback) path in
    # _resolve_base_ref, rather than assuming origin/main already exists.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "architecture_doc_budgets.py"), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
