"""Tests for ``tools/land.py``, against synthetic git repos in ``tmp_path``.

Every sandbox is self-contained: a bare ``origin.git`` and a ``work/`` clone
inside the test's own tmp dir, with the real ``checks/repo_hygiene.py``,
``tools/oc_check.py`` and ``tools/bounded_run.py`` copied in so land's
subprocess calls behave exactly like production. Nothing here touches the real
checkout beyond reading those three files, and no test pushes anywhere real.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAND_PY = REPO_ROOT / "tools" / "land.py"

REAL_ENV_BODY = "TESTVAR=some_long_enough_generic_value_123\n"
ZERO_ENV_BODY = "OQUANTS_COOKIE_NAME=whatever\n"
GITIGNORE = "__pycache__/\n*.pyc\n.pytest_cache/\n"

COPIED_FILES = (
    "checks/repo_hygiene.py",
    "tools/oc_check.py",
    "tools/bounded_run.py",
)


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed in {cwd}:\n{proc.stdout}{proc.stderr}")
    return proc


def _status(work: Path) -> str:
    return _git(work, "status", "--porcelain").stdout


def _head(work: Path) -> str:
    return _git(work, "rev-parse", "HEAD").stdout.strip()


def _branch(work: Path) -> str:
    return _git(work, "symbolic-ref", "-q", "--short", "HEAD").stdout.strip()


def _origin_main(origin: Path) -> str:
    return _git(origin, "rev-parse", "main").stdout.strip()


def _commit(work: Path, message: str) -> None:
    _git(work, "add", "-A")
    _git(work, "commit", "-m", message)


def _write_env(tmp_path: Path, name: str, body: str) -> Path:
    path = tmp_path / name
    path.write_text(body)
    return path


def _run_land(work: Path, env_path: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["LAND_TEST_ENV_OVERRIDE"] = str(env_path)
    return subprocess.run(
        [sys.executable, str(LAND_PY), *args],
        cwd=str(work), env=env, capture_output=True, text=True)


@pytest.fixture
def sandbox(tmp_path):
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(
        ["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(origin), "symbolic-ref", "HEAD", "refs/heads/main"],
        check=True, capture_output=True)
    subprocess.run(
        ["git", "clone", str(origin), str(work)], check=True, capture_output=True)
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Land Test")
    for rel in COPIED_FILES:
        dest = work / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO_ROOT / rel, dest)
    (work / ".gitignore").write_text(GITIGNORE)
    (work / "README.md").write_text("hello\n")
    _commit(work, "initial commit")
    _git(work, "push", "origin", "main")
    return work, origin


def _feature_commit(work: Path, name: str = "feature") -> None:
    _git(work, "checkout", "-b", name)
    (work / "feature.txt").write_text("feature\n")
    _commit(work, f"{name} commit")
    _git(work, "checkout", "main")


def test_clean_tree_required_before_any_git_state(sandbox, tmp_path):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    _feature_commit(work)

    before_head = _head(work)
    before_origin = _origin_main(origin)
    (work / "dirty.txt").write_text("uncommitted\n")

    proc = _run_land(work, env_path, "feature", "-m", "merge feature")

    assert proc.returncode == 1
    assert "clean" in proc.stderr
    assert _head(work) == before_head
    assert _origin_main(origin) == before_origin

    (work / "dirty.txt").unlink()
    assert _status(work) == ""


def test_conflict_is_aborted_and_head_restored(sandbox, tmp_path):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)

    (work / "shared.txt").write_text("base\n")
    _commit(work, "add shared")
    _git(work, "push", "origin", "main")

    _git(work, "checkout", "-b", "feature")
    (work / "shared.txt").write_text("feature side\n")
    _commit(work, "feature edit")
    _git(work, "checkout", "main")
    (work / "shared.txt").write_text("main side\n")
    _commit(work, "main edit")
    _git(work, "push", "origin", "main")

    before_head = _head(work)
    before_branch = _branch(work)
    before_origin = _origin_main(origin)

    proc = _run_land(work, env_path, "feature", "-m", "merge feature")

    assert proc.returncode == 3
    assert "shared.txt" in proc.stderr
    assert "unmerged paths" not in _git(work, "status").stdout.lower()
    assert _status(work) == ""
    assert _head(work) == before_head
    assert _branch(work) == before_branch
    assert _origin_main(origin) == before_origin


def test_zero_loaded_secret_patterns_block_the_push(sandbox, tmp_path):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "zero.env", ZERO_ENV_BODY)
    _feature_commit(work)

    before_origin = _origin_main(origin)
    proc = _run_land(work, env_path, "feature", "-m", "merge feature")

    assert proc.returncode == 5
    assert str(env_path) in proc.stderr
    assert "0 secret pattern" in proc.stderr
    assert _origin_main(origin) == before_origin
    assert _status(work) == ""


def test_missing_env_fails_before_git_state_moves(sandbox, tmp_path):
    work, _origin = sandbox
    env_path = tmp_path / "does_not_exist.env"
    _feature_commit(work)

    before_head = _head(work)
    before_branch = _branch(work)

    proc = _run_land(work, env_path, "feature", "-m", "merge feature")

    assert proc.returncode == 4
    assert str(env_path) in proc.stderr
    assert _head(work) == before_head
    assert _branch(work) == before_branch
    assert _status(work) == ""


def test_dry_run_pushes_nothing_and_leaves_no_trace(sandbox, tmp_path):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    _feature_commit(work)

    before_head = _head(work)
    before_branch = _branch(work)
    before_origin = _origin_main(origin)

    proc = _run_land(work, env_path, "feature", "-m", "merge feature", "--dry-run")

    assert proc.returncode == 0, proc.stderr
    assert "skipped (--dry-run)" in proc.stdout
    assert len(proc.stdout.strip().splitlines()) <= 5
    assert _origin_main(origin) == before_origin
    assert _head(work) == before_head
    assert _branch(work) == before_branch
    assert _status(work) == ""


def test_failing_tests_restore_head_and_push_nothing(sandbox, tmp_path):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)

    _git(work, "checkout", "-b", "feature")
    (work / "tests").mkdir()
    (work / "tests" / "test_something.py").write_text(
        "def test_always_fails():\n    assert False\n")
    _commit(work, "add always failing test")
    _git(work, "checkout", "main")

    before_head = _head(work)
    before_branch = _branch(work)
    before_origin = _origin_main(origin)

    proc = _run_land(work, env_path, "feature", "-m", "merge feature")

    assert proc.returncode == 8, (proc.returncode, proc.stdout, proc.stderr)
    assert "tests failed" in proc.stderr
    assert _origin_main(origin) == before_origin
    assert _head(work) == before_head
    assert _branch(work) == before_branch
    assert _status(work) == ""