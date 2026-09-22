"""Tests for ``tools/land.py``, against synthetic git repos in ``tmp_path``.

Every sandbox is self-contained: a bare ``origin.git`` and a ``work/`` clone
inside the test's own tmp dir, with the real ``checks/repo_hygiene.py``,
``tools/oc_check.py`` and ``tools/bounded_run.py`` copied in so land's
subprocess calls behave exactly like production. Nothing here touches the real
checkout beyond reading those three files, and no test pushes anywhere real.
"""
from __future__ import annotations

import importlib.util
import io
import shutil
import subprocess
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAND_PY = REPO_ROOT / "tools" / "land.py"

_spec = importlib.util.spec_from_file_location("land", str(LAND_PY))
land = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(land)

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


class _LandResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run_land(work: Path, env_path: Path, *args: str) -> _LandResult:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        returncode = land.main(list(args), env_path=env_path, root=work)
    return _LandResult(returncode, stdout.getvalue(), stderr.getvalue())


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
    # 5 original summary lines + the deletion guard's one-line "0 removed" OK.
    assert len(proc.stdout.strip().splitlines()) <= 6
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


# --- Deletion guard -------------------------------------------------------
#
# RECENT_ADDITION_WINDOW is monkeypatched down to 1 commit in the tier-1/
# tier-2 tests below so "recently added" means "added in origin/main's very
# last commit", regardless of how few commits the sandbox repo has (the
# unpatched default, 50, would clamp to the sandbox's root commit and make
# everything ever added look "recent").

def _write_and_commit(work: Path, rel: str, body: str, message: str) -> None:
    (work / rel).write_text(body)
    _commit(work, message)


def test_clean_additive_merge_passes_deletion_guard(sandbox, tmp_path):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    _feature_commit(work)

    proc = _run_land(work, env_path, "feature", "-m", "merge feature", "--dry-run")

    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "0 line(s) removed" in proc.stdout


def test_tier1_reverting_a_recently_landed_line_refuses_and_head_restored(
        sandbox, tmp_path, monkeypatch):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    # An old, unrelated commit, then the commit that lands the important
    # line as origin/main's tip -- inside the 1-commit window.
    _write_and_commit(work, "old.txt", "old padding content here\n", "old padding")
    _write_and_commit(
        work, "config.py", "RECENT_IMPORTANT_VALUE = 12345\n",
        "land the important recent value")
    _git(work, "push", "origin", "main")

    # A branch off that tip whose only change deletes the just-landed line
    # -- the stale-checkout-revert shape this guard exists for.
    _git(work, "checkout", "-b", "feature")
    (work / "config.py").write_text("")
    _commit(work, "accidentally drop the recent value")
    _git(work, "checkout", "main")

    before_head = _head(work)
    before_branch = _branch(work)
    before_origin = _origin_main(origin)

    proc = _run_land(work, env_path, "feature", "-m", "merge feature")

    assert proc.returncode == 10, (proc.returncode, proc.stdout, proc.stderr)
    assert "tier 1" in proc.stderr
    assert "RECENT_IMPORTANT_VALUE" in proc.stderr
    assert "reverts recently-landed work" in proc.stderr or "no longer exist" in proc.stderr
    assert "--reviewed-deletions-from" in proc.stderr
    assert _origin_main(origin) == before_origin
    assert _head(work) == before_head
    assert _branch(work) == before_branch
    assert _status(work) == ""


def test_tier1_is_not_rescuable_by_allow_deletions(sandbox, tmp_path, monkeypatch):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    _write_and_commit(work, "old.txt", "old padding content here\n", "old padding")
    _write_and_commit(
        work, "config.py", "RECENT_IMPORTANT_VALUE = 12345\n",
        "land the important recent value")
    _git(work, "push", "origin", "main")

    _git(work, "checkout", "-b", "feature")
    (work / "config.py").write_text("")
    _commit(work, "accidentally drop the recent value")
    _git(work, "checkout", "main")

    before_head = _head(work)
    before_origin = _origin_main(origin)

    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature", "--allow-deletions", "1")

    assert proc.returncode == 10, (proc.returncode, proc.stdout, proc.stderr)
    assert "tier 1" in proc.stderr
    assert "--reviewed-deletions-from" in proc.stderr
    assert _origin_main(origin) == before_origin
    assert _head(work) == before_head
    assert _status(work) == ""


def test_tier2_unrelated_deletion_refuses_and_is_rescuable_by_exact_count(
        sandbox, tmp_path, monkeypatch):
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    # The line to be removed lands well before the window; a later,
    # unrelated commit becomes the tip, so "recently added" (the last 1
    # commit) does not contain it -- tier 1 must stay quiet here.
    _write_and_commit(
        work, "config.py", "OLD_UNRELATED_VALUE = 999\n", "land an old value")
    _write_and_commit(work, "padding.txt", "unrelated padding change\n", "padding")
    _git(work, "push", "origin", "main")

    _git(work, "checkout", "-b", "feature")
    (work / "config.py").write_text("")
    _commit(work, "remove the old value")
    _git(work, "checkout", "main")

    before_head = _head(work)
    before_branch = _branch(work)
    before_origin = _origin_main(origin)

    # No --allow-deletions: refuse (tier 2, exit 11).
    proc = _run_land(work, env_path, "feature", "-m", "merge feature")
    assert proc.returncode == 11, (proc.returncode, proc.stdout, proc.stderr)
    assert "tier 1" not in proc.stderr
    assert "OLD_UNRELATED_VALUE" in proc.stderr
    assert "--allow-deletions" in proc.stderr
    assert _origin_main(origin) == before_origin
    assert _head(work) == before_head
    assert _branch(work) == before_branch
    assert _status(work) == ""

    # Wrong count: still refuses, reporting both numbers.
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature", "--allow-deletions", "2")
    assert proc.returncode == 11, (proc.returncode, proc.stdout, proc.stderr)
    assert "--allow-deletions 2" in proc.stderr
    assert "removed-line count 1" in proc.stderr
    assert _head(work) == before_head
    assert _status(work) == ""

    # Exact count: passes.
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature", "--allow-deletions", "1",
        "--dry-run")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "matches --allow-deletions 1" in proc.stdout
    assert _head(work) == before_head
    assert _status(work) == ""




def test_tier1_genuinely_absent_lines_require_reviewed_deletions_from(
        sandbox, tmp_path, monkeypatch):
    """Test that genuinely deleted lines fire tier 1 but can be overridden with
    --reviewed-deletions-from flag listing the exact commits that added them.
    """
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    # Add some lines we'll later genuinely delete
    _write_and_commit(work, "old.txt", "padding\n", "padding")
    _write_and_commit(
        work, "config.py",
        "SETTING_1 = True\nSETTING_2 = False\n",
        "land config")
    _git(work, "push", "origin", "main")

    # Get the SHA of the commit that added the config
    config_commit_sha = _git(work, "rev-parse", "HEAD").stdout.strip()[:7]

    # On feature branch, genuinely delete both lines
    _git(work, "checkout", "-b", "feature")
    (work / "config.py").write_text("")
    _commit(work, "remove config settings")
    _git(work, "checkout", "main")

    before_head = _head(work)
    before_origin = _origin_main(origin)

    # First, verify it fires tier 1
    proc = _run_land(work, env_path, "feature", "-m", "merge feature")
    assert proc.returncode == 10, (proc.returncode, proc.stdout, proc.stderr)
    assert "tier 1" in proc.stderr
    assert "ABSENT" in proc.stderr or "reverts" in proc.stderr
    assert _head(work) == before_head
    assert _status(work) == ""

    # Wrong commit SHA: still refuses
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature",
        "--reviewed-deletions-from", "wrongsha")
    assert proc.returncode == 10, (proc.returncode, proc.stdout, proc.stderr)
    assert "do not match" in proc.stderr
    assert _head(work) == before_head
    assert _status(work) == ""

    # Correct commit SHA: passes
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature",
        "--reviewed-deletions-from", config_commit_sha, "--dry-run")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert config_commit_sha in proc.stdout
    assert _head(work) == before_head
    assert _status(work) == ""


def test_tier1_still_fires_on_absolute_genuine_deletion(sandbox, tmp_path, monkeypatch):
    """Test that the real incident (cac6c00 → d40a505) still fires tier 1.

    This tests the original incident: a branch deletes a call and its imports
    that were recently added to main. This should always fire and not be
    weakened by the relocation-aware fix.
    """
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    # Add a function call and its imports
    _write_and_commit(work, "old.txt", "padding\n", "padding")
    _write_and_commit(
        work, "main_code.py",
        "from helpers import get_value\n\ndef process():\n    val = get_value()\n    return val\n",
        "land function call")
    _git(work, "push", "origin", "main")

    # On feature branch, delete both the call AND the import
    # The key is that they're both deleted entirely (not moved)
    _git(work, "checkout", "-b", "feature")
    (work / "main_code.py").write_text("def process():\n    return None\n")
    _commit(work, "remove call and import")
    _git(work, "checkout", "main")

    before_head = _head(work)
    before_origin = _origin_main(origin)

    # Must fire tier 1 - this is the real incident that must never slip through
    proc = _run_land(work, env_path, "feature", "-m", "merge feature")
    assert proc.returncode == 10, (proc.returncode, proc.stdout, proc.stderr)
    assert "tier 1" in proc.stderr
    # The ABSENT lines should be reported
    assert "ABSENT" in proc.stderr or "reverts" in proc.stderr
    assert _origin_main(origin) == before_origin
    assert _head(work) == before_head
    assert _status(work) == ""


def test_tier1_passes_cleanly_when_line_moves_to_different_file(
        sandbox, tmp_path, monkeypatch):
    """RELOCATION TEST: Line added to main, moved to different file on feature branch.
    Tier 1 must pass cleanly (no override flag needed) because content still exists.
    """
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    # Add a distinctive line to file A, push
    _write_and_commit(work, "old.txt", "padding\n", "padding")
    _write_and_commit(
        work, "file_a.py", "special_value = get_important_data()\n",
        "land important line in A")
    _git(work, "push", "origin", "main")

    # Move that exact line to file B (remove from A, add to B)
    _git(work, "checkout", "-b", "feature")
    (work / "file_a.py").write_text("")
    (work / "file_b.py").write_text("special_value = get_important_data()\n")
    _commit(work, "move important line to B")
    _git(work, "checkout", "main")

    # Tier 1 must PASS (relocation exemption): content moved, not deleted from tree.
    # Tier 2 still sees the line as removed from origin/main, so needs --allow-deletions.
    # This confirms tier 1's exemption is working (would have fired without it).
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature",
        "--allow-deletions", "1", "--dry-run")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    # Confirm line was exempted at tier 1 (either "MOVED" or clean pass)
    assert "tier 1" not in proc.stderr or "MOVED" in proc.stderr
    assert _status(work) == ""


def test_tier1_with_real_history_cac6c00_to_d40a505():
    """Regression test: real laundering incident (cac6c00 → d40a505).
    Direct unit test of _content_exists_in_tree predicate.
    Verifies: (1) decisive laundered line is ABSENT at d40a505 (tier 1 must fire),
    (2) one of the 12 moved lines EXISTS at d40a505 (relocation exemption works).
    """
    import pytest
    import subprocess
    root = REPO_ROOT

    # Skip only in genuine shallow clone (commits unreachable in worktree's own repo)
    proc = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-t", "d40a505"],
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("d40a505 not in worktree (shallow clone)")

    # The decisive laundered line: removed in merge, must NOT exist at d40a505
    absent_line = "inputs = with_stored_forecasts(inputs, resolve_stored_forecasts(inputs))"
    assert not land._content_exists_in_tree(root, "d40a505", absent_line), \
        "Laundered line must be ABSENT at d40a505 for tier 1 to fire"

    # One of the 12 moved lines: must EXIST at d40a505 (relocation exemption)
    moved_line = "raw = vector[name]"
    assert land._content_exists_in_tree(root, "d40a505", moved_line), \
        "Moved line must EXIST at d40a505 for relocation exemption to work"


def test_tier1_passes_with_correct_reviewed_deletions_from_wrong_without(
        sandbox, tmp_path, monkeypatch):
    """Test: 841150e case (2 ABSENT lines). Tier 1 fires, correct SHA allows,
    wrong/empty SHA still refuses.
    """
    work, origin = sandbox
    env_path = _write_env(tmp_path, "real.env", REAL_ENV_BODY)
    monkeypatch.setattr(land, "RECENT_ADDITION_WINDOW", 1)

    # Commit that will add 2 lines we'll later delete
    _write_and_commit(work, "old.txt", "padding\n", "padding")
    _write_and_commit(
        work, "code.py",
        "incomplete = True\nif incomplete:\n    return None\n",
        "add condition check")
    _git(work, "push", "origin", "main")
    commit_sha = _git(work, "rev-parse", "HEAD").stdout.strip()[:7]

    # Delete all those lines
    _git(work, "checkout", "-b", "feature")
    (work / "code.py").write_text("")
    _commit(work, "remove condition")
    _git(work, "checkout", "main")

    # Tier 1 fires
    proc = _run_land(work, env_path, "feature", "-m", "merge feature")
    assert proc.returncode == 10
    assert "tier 1" in proc.stderr
    assert _status(work) == ""

    # Empty override fails
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature",
        "--reviewed-deletions-from", "")
    assert proc.returncode == 10
    assert "do not match" in proc.stderr
    assert _status(work) == ""

    # Correct SHA allows (with --dry-run to avoid push)
    proc = _run_land(
        work, env_path, "feature", "-m", "merge feature",
        "--reviewed-deletions-from", commit_sha, "--dry-run")
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert commit_sha in proc.stdout
    assert _status(work) == ""