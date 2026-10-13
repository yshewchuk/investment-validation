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
# One MD022 (heading), one MD031 (fence) and one MD058 (table) violation.
BAD = (
    "intro\n# Heading\ntext\n\n"
    "para\n```\ncode\n```\nafter\n\n"
    "para\n| a | b |\n|---|---|\n| 1 | 2 |\n"
)


def _pin() -> str:
    return re.search(r"markdownlint-cli2@[\d.]+", HOOK.read_text(encoding="utf-8")).group(0)


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
def test_config_enables_all_three_rules(tmp_path):
    (tmp_path / "bad.md").write_text(BAD, encoding="utf-8")
    proc = subprocess.run(
        ["npx", "--yes", _pin(), "--config", str(CONFIG), "bad.md"],
        cwd=tmp_path, capture_output=True, text=True, timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    for rule in ("MD022", "MD031", "MD058"):
        assert rule in out, f"{rule} not reported:\n{out}"


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