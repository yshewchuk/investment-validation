"""Wiring tests for the Markdown lint: our config enables each rule, and the pre-commit hook runs it."""
# packages: ops

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
HOOK = ROOT / "checks" / "hooks" / "pre-commit"
CONFIG = ROOT / ".markdownlint-cli2.jsonc"
needs_node = pytest.mark.skipif(
    shutil.which("npx") is None,
    reason="node/npx not installed; markdownlint-cli2 cannot run here (CI has Node)",
)
# (rule, text, reported line, marker): each fixture misses the blank line on ONE side only,
# so a config that requires only one side cannot pass both cases of a rule.
CASES = [
    ("MD022", "intro\n# Heading\n\ntext\n", 2, "Above"),
    ("MD022", "# Heading\ntext\n", 1, "Below"),
    ("MD031", "para\n```\ncode\n```\n\nafter\n", 2, ""),
    ("MD031", "para\n\n```\ncode\n```\nafter\n", 5, ""),
    ("MD058", "para\n| a | b |\n|---|---|\n| 1 | 2 |\n\nafter\n", 2, ""),
    ("MD058", "| a | b |\n|---|---|\n| 1 | 2 |\n- item\n", 3, ""),
]


def _pin() -> str:
    return re.search(r"markdownlint-cli2@[\d.]+", HOOK.read_text(encoding="utf-8")).group(0)


# The hook's npx pin and the workflow's action must run the same CLI. Action v20 (this
# commit) bundles markdownlint-cli2 0.18.1; bump the three together or this test fails.
EXPECTED_CLI = "markdownlint-cli2@0.18.1"
ACTION_SHA = "992badcdf24e3b8eb7e87ff9287fe931bcb00c6e"


def test_hook_and_workflow_action_pin_the_same_cli():
    assert _pin() == EXPECTED_CLI
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    assert f"DavidAnson/markdownlint-cli2-action@{ACTION_SHA}" in workflow


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _hook_repo(tmp_path, monkeypatch):
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "checks").mkdir()
    # The other hook steps are not under test: stub each one as a passing script.
    hook_text = HOOK.read_text(encoding="utf-8")
    for name in set(re.findall(r"checks/([a-z_0-9]+\.py)", hook_text)):
        (tmp_path / "checks" / name).write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    shutil.copy(CONFIG, tmp_path / ".markdownlint-cli2.jsonc")
    (tmp_path / "pre-commit").write_text(hook_text, encoding="utf-8")
    _git(tmp_path, "init", "-q")
    return tmp_path / "pre-commit"


def _run_hook(hook, cwd, env=None):
    return subprocess.run(["/bin/sh", str(hook)], cwd=cwd, capture_output=True, text=True, env=env)


@needs_node
@pytest.mark.parametrize("rule, text, line, marker", CASES)
def test_config_reports_each_side_of_each_rule(tmp_path, rule, text, line, marker):
    (tmp_path / "bad.md").write_text(text, encoding="utf-8")
    proc = subprocess.run(
        ["npx", "--yes", _pin(), "--config", str(CONFIG), "bad.md"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    rows = [r for r in out.splitlines() if re.match(rf"bad\.md:{line}(:\d+)? {rule}", r)]
    assert rows, f"{rule} not reported on line {line}:\n{out}"
    assert any(marker in r for r in rows), f"{rule} {marker} not reported:\n{out}"


@needs_node
def test_hook_blocks_a_staged_violation_and_passes_a_clean_file(tmp_path, monkeypatch):
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "bad.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "bad.md")
    blocked = _run_hook(hook, tmp_path)
    assert blocked.returncode != 0
    assert "MD022" in blocked.stdout + blocked.stderr

    (tmp_path / "bad.md").write_text("intro\n\n# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "bad.md")
    assert _run_hook(hook, tmp_path).returncode == 0

    # A non-Markdown file whose content would violate MD022 if it were linted must be
    # ignored: dropping the '*.md' pathspec from the hook makes this fail.
    (tmp_path / "notes.py").write_text("# heading\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.py")
    proc = _run_hook(hook, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@needs_node
def test_hook_lints_only_staged_files(tmp_path, monkeypatch):
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "ok.md").write_text("# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "ok.md")
    # Tracked-but-unstaged and untracked bad files must not be linted: a hook built on
    # `git ls-files` instead of the staged diff would fail here.
    (tmp_path / "tracked.md").write_text("# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.md")
    _git(tmp_path, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "base")
    (tmp_path / "tracked.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    (tmp_path / "untracked.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    (tmp_path / "ok.md").write_text("# Heading\n\ntext\n\nmore\n", encoding="utf-8")
    _git(tmp_path, "add", "ok.md")
    proc = _run_hook(hook, tmp_path)
    assert proc.returncode == 0, proc.stdout + proc.stderr


@needs_node
def test_hook_lints_the_working_tree_copy_of_a_staged_file(tmp_path, monkeypatch):
    # Documented exception (hook header): markdownlint cannot read a blob, so a clean
    # staged copy with a bad working copy is reported.
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "w.md").write_text("# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "w.md")
    (tmp_path / "w.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    proc = _run_hook(hook, tmp_path)
    assert proc.returncode != 0
    assert "MD022" in proc.stdout + proc.stderr


def test_hook_fails_closed_without_npx(tmp_path, monkeypatch):
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "ok.md").write_text("# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "ok.md")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("git", "python3", "xargs"):
        found = shutil.which(tool)
        assert found
        os.symlink(found, bin_dir / tool)
    env = {**os.environ, "PATH": str(bin_dir)}
    proc = _run_hook(hook, tmp_path, env=env)
    assert proc.returncode != 0
    assert "npx not found" in proc.stderr