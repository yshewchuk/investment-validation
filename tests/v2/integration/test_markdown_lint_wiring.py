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
    # Only executable lines count: a stale pin in a comment must not satisfy the test.
    code = [ln for ln in HOOK.read_text(encoding="utf-8").splitlines() if not ln.lstrip().startswith("#")]
    pins = [m.group(0) for ln in code for m in re.finditer(r"markdownlint-cli2@[\d.]+", ln)]
    assert len(pins) == 1, f"expected exactly one pinned invocation in the hook, got {pins}"
    return pins[0]


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
def test_config_reports_each_side_of_each_rule(tmp_path):
    expected = []
    names = []
    for i, (rule, text, line, marker) in enumerate(CASES):
        name = f"case{i}_{rule}.md"
        names.append(name)
        (tmp_path / name).write_text(text, encoding="utf-8")
        expected.append((name, line, rule, marker))
    proc = subprocess.run(
        ["npx", "--yes", _pin(), "--config", str(CONFIG), *names],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    for name, line, rule, marker in expected:
        case = f"{name}:{line} {rule}"
        rows = [r for r in out.splitlines() if re.match(rf"{re.escape(name)}:{line}(:\d+)? {rule}", r)]
        assert rows, f"{case} not reported in file {name}:\n{out}"
        if marker:
            assert any(marker in r for r in rows), f"{case} {marker} not reported:\n{out}"


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


def _commit(repo, message="base"):
    _git(repo, "-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", message)


@needs_node
def test_hook_lints_a_modified_file(tmp_path, monkeypatch):
    # Guards against a `--diff-filter=A` hook: only an existing file is modified here.
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "m.md").write_text("# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "m.md")
    _commit(tmp_path)
    (tmp_path / "m.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "m.md")
    proc = _run_hook(hook, tmp_path)
    assert proc.returncode != 0
    assert "MD022" in proc.stdout + proc.stderr


@needs_node
def test_hook_lints_a_renamed_and_edited_file(tmp_path, monkeypatch):
    hook = _hook_repo(tmp_path, monkeypatch)
    body = "".join(f"line {i} of a paragraph that stays the same\n\n" for i in range(12))
    (tmp_path / "old.md").write_text("# Heading\n\n" + body, encoding="utf-8")
    _git(tmp_path, "add", "old.md")
    _commit(tmp_path)
    _git(tmp_path, "mv", "old.md", "new.md")
    (tmp_path / "new.md").write_text("intro\n# Heading\n" + body, encoding="utf-8")
    _git(tmp_path, "add", "new.md")
    proc = _run_hook(hook, tmp_path)
    assert proc.returncode != 0
    assert "MD022" in proc.stdout + proc.stderr


@needs_node
def test_hook_lints_every_staged_file_not_just_the_first(tmp_path, monkeypatch):
    # Staged paths are listed sorted: the clean files come first, the bad one is LAST.
    hook = _hook_repo(tmp_path, monkeypatch)
    for name in ("a.md", "b.md"):
        (tmp_path / name).write_text("# Heading\n\ntext\n", encoding="utf-8")
    (tmp_path / "z.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "a.md", "b.md", "z.md")
    proc = _run_hook(hook, tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "z.md" in out and "MD022" in out
    assert "a.md:" not in out and "b.md:" not in out


@needs_node
def test_hook_lints_a_first_bad_file_followed_by_clean_ones(tmp_path, monkeypatch):
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "a.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    for name in ("b.md", "z.md"):
        (tmp_path / name).write_text("# Heading\n\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "a.md", "b.md", "z.md")
    proc = _run_hook(hook, tmp_path)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "a.md:" in out and "MD022" in out
    assert "b.md:" not in out and "z.md:" not in out


def test_hook_with_only_non_markdown_staged_does_not_run_the_linter(tmp_path, monkeypatch):
    # Needs no Node: with no .md staged, `xargs -r` must skip npx entirely. A fake npx that
    # fails loudly proves it is never invoked.
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / "notes.py").write_text("# heading\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "notes.py")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ("git", "python3", "xargs"):
        os.symlink(shutil.which(tool), bin_dir / tool)
    npx = bin_dir / "npx"
    npx.write_text("#!/bin/sh\necho NPX-WAS-RUN >&2\nexit 9\n", encoding="utf-8")
    npx.chmod(0o755)
    proc = _run_hook(hook, tmp_path, env={**os.environ, "PATH": str(bin_dir)})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NPX-WAS-RUN" not in proc.stdout + proc.stderr


@needs_node
def test_cli_with_repo_config_reports_a_file_a_deny_all_gitignore_excludes(tmp_path):
    # The repo's .gitignore is default-deny; config "gitignore": true would skip every doc.
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    (tmp_path / "bad.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    proc = subprocess.run(
        ["npx", "--yes", _pin(), "--config", str(CONFIG), "bad.md"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "MD022" in proc.stdout + proc.stderr


@needs_node
def test_hook_reports_a_tracked_file_that_a_deny_all_gitignore_matches(tmp_path, monkeypatch):
    hook = _hook_repo(tmp_path, monkeypatch)
    (tmp_path / ".gitignore").write_text("*\n", encoding="utf-8")
    (tmp_path / "bad.md").write_text("intro\n# Heading\ntext\n", encoding="utf-8")
    _git(tmp_path, "add", "-f", "bad.md")
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