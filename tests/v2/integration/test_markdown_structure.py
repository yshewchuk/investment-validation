"""Tests for checks/markdown_structure.py: a thin wrapper over markdownlint-cli2."""
# packages: ops

import re
import shutil
import subprocess

import pytest

from checks import markdown_structure as ms

needs_node = pytest.mark.skipif(
    shutil.which("npx") is None,
    reason="node/npx not installed; the linter cannot run here (CI has node)",
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    # A pre-commit hook run exports these; they would point git at the real repo.
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@example.com")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "old.md").write_text("# Old\nlegacy defect\n", encoding="utf-8")
    _git(tmp_path, "add", "old.md")
    _git(tmp_path, "commit", "-qm", "base")
    monkeypatch.setattr(ms, "ROOT", tmp_path)
    return tmp_path


def test_changed_lines_parses_hunks_and_deletions():
    diff = "@@ -1,0 +2,2 @@\n+a\n+b\n@@ -9,3 +11 @@\n-x\n+y\n@@ -20,2 +22,0 @@\n-gone\n"
    assert ms.changed_lines(diff) == {2, 3, 11, 22, 23}


@needs_node
@pytest.mark.parametrize("rule, text", [
    ("MD022", "intro\n# Bad\n\nbody\n"),
    ("MD031", "intro\n\n```\ncode\n```\nafter\n"),
    ("MD058", "intro\n| a | b |\n|---|---|\n| 1 | 2 |\n\nend\n"),
])
def test_each_enabled_rule_fires_on_a_tiny_fixture(repo, capsys, rule, text):
    (repo / "new.md").write_text(text, encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 1
    assert f": {rule} " in capsys.readouterr().out


@needs_node
def test_only_the_configured_rules_run(repo):
    # Everything else (line length, bare URLs, trailing spaces, ...) is off.
    (repo / "new.md").write_text("# Title\n\n" + "x" * 300 + " \n\nhttp://example.com\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 0


@needs_node
def test_staged_mode_ignores_untouched_legacy_defects(repo, capsys):
    (repo / "new.md").write_text("# New\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 0
    assert ms.main(["--all", "--quiet"]) == 1
    assert "old.md:1: MD022" in capsys.readouterr().out


@needs_node
def test_staged_legacy_file_flags_only_touched_defect(repo, capsys):
    # old.md has a legacy defect on line 1. Change an unrelated line: not reported.
    (repo / "old.md").write_text("# Old\nlegacy defect\n\nbody\n\nmore\n", encoding="utf-8")
    _git(repo, "add", "old.md")
    assert ms.main(["--quiet"]) == 0
    capsys.readouterr()
    # Now touch the defect line itself: reported.
    (repo / "old.md").write_text("# Older\nlegacy defect\n\nbody\n\nmore\n", encoding="utf-8")
    _git(repo, "add", "old.md")
    assert ms.main(["--quiet"]) == 1
    assert "old.md:1: MD022" in capsys.readouterr().out


@needs_node
def test_base_ref_mode_checks_only_the_branch_diff(repo, capsys):
    _git(repo, "branch", "base")
    (repo / "old.md").write_text("# Old\nlegacy defect\n\npara\n| a | b |\n|---|---|\n| 1 | 2 |\n",
                                 encoding="utf-8")
    _git(repo, "commit", "-qam", "edit")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 1
    out = capsys.readouterr().out
    assert "MD058" in out
    assert "MD022" not in out


@needs_node
def test_staged_mode_reads_the_index_not_the_worktree(repo, capsys):
    bad, fine = "intro\n# Bad\n\nbody\n", "# Fine\n\nbody\n"
    f = repo / "w.md"
    # Staged version is bad, the worktree was then fixed but not staged: still flagged.
    f.write_text(bad, encoding="utf-8")
    _git(repo, "add", "w.md")
    f.write_text(fine, encoding="utf-8")
    assert ms.main(["--quiet"]) == 1
    assert "w.md:2: MD022" in capsys.readouterr().out
    # Staged version is fine, the worktree was then broken but not staged: clean.
    f.write_text(fine, encoding="utf-8")
    _git(repo, "add", "w.md")
    f.write_text(bad, encoding="utf-8")
    assert ms.main(["--quiet"]) == 0


@needs_node
def test_base_ref_mode_reads_committed_content_not_the_worktree(repo, capsys):
    bad, fine = "intro\n# Bad\n\nbody\n", "# Fine\n\nbody\n"
    f = repo / "w.md"
    _git(repo, "branch", "base")
    f.write_text(bad, encoding="utf-8")
    _git(repo, "add", "w.md")
    _git(repo, "commit", "-qm", "bad")
    f.write_text(fine, encoding="utf-8")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 1
    assert "w.md:2: MD022" in capsys.readouterr().out
    _git(repo, "commit", "-qam", "fix")
    f.write_text(bad, encoding="utf-8")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 0


@needs_node
def test_base_that_moved_after_branching_does_not_flag_main_s_own_changes(repo, capsys):
    # CI checks out a merge commit; the base branch tip is newer than the PR's
    # recorded base sha. Three-dot against the CURRENT base tip flags only the
    # branch's own lines, while a stale base sha also flags what main added since.
    _git(repo, "branch", "-M", "main")
    stale_base = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "feat.md").write_text("# Ok\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "feat.md")
    _git(repo, "commit", "-qm", "feature work")
    _git(repo, "checkout", "-q", "main")
    (repo / "moved.md").write_text("# Bad\nno blank after heading\n", encoding="utf-8")
    _git(repo, "add", "moved.md")
    _git(repo, "commit", "-qm", "main moved")
    _git(repo, "checkout", "-q", "feature")
    _git(repo, "merge", "-q", "--no-edit", "main")
    assert ms.main(["--base-ref", "main", "--quiet"]) == 0
    assert ms.main(["--base-ref", stale_base, "--quiet"]) == 1
    assert "moved.md:1: MD022" in capsys.readouterr().out


@needs_node
def test_crlf_files_are_linted_like_lf_files(repo, capsys):
    (repo / "ok.md").write_bytes(b"# Title\r\n\r\n```\r\ncode\r\n```\r\n\r\nend\r\n")
    _git(repo, "add", "ok.md")
    assert ms.main(["--quiet"]) == 0
    (repo / "bad.md").write_bytes(b"intro\r\n# Bad\r\n\r\nbody\r\n")
    _git(repo, "add", "bad.md")
    assert ms.main(["--quiet"]) == 1
    assert "bad.md:2: MD022" in capsys.readouterr().out


@needs_node
def test_staged_deletion_of_a_markdown_file_is_not_read(repo):
    _git(repo, "rm", "-q", "old.md")
    assert ms.main(["--quiet"]) == 0


@needs_node
def test_quiet_suppresses_the_clean_message(repo, capsys):
    (repo / "ok.md").write_text("# Ok\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "ok.md")
    assert ms.main(["--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert ms.main([]) == 0
    assert "clean" in capsys.readouterr().out


@needs_node
def test_user_git_color_and_external_diff_config_do_not_hide_violations(repo, monkeypatch, capsys):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "color.ui")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "diff.external")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "false")
    (repo / "new.md").write_text("intro\n# Bad\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 1
    assert "new.md:2: MD022" in capsys.readouterr().out


def test_tool_missing_fails_closed_with_an_install_hint(repo, monkeypatch, capsys):
    (repo / "new.md").write_text("# New\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    monkeypatch.setattr(ms.shutil, "which", lambda name: None)
    assert ms.main(["--quiet"]) == 2
    err = capsys.readouterr().err
    assert "npx" in err and "install" in err


def _fake_linter(monkeypatch, returncode, stdout="", stderr=""):
    real = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "git":
            return real(cmd, *args, **kwargs)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    monkeypatch.setattr(ms.shutil, "which", lambda name: "/usr/bin/npx")
    monkeypatch.setattr(ms.subprocess, "run", run)


def test_linter_crash_fails_closed(repo, monkeypatch, capsys):
    (repo / "new.md").write_text("# New\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    _fake_linter(monkeypatch, 3, stderr="boom")
    assert ms.main(["--quiet"]) == 2
    assert "crashed" in capsys.readouterr().err


def test_linter_failure_without_findings_fails_closed(repo, monkeypatch, capsys):
    (repo / "new.md").write_text("# New\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    _fake_linter(monkeypatch, 1, stderr="config error")
    assert ms.main(["--quiet"]) == 2


def test_linter_timeout_fails_closed(repo, monkeypatch, capsys):
    (repo / "new.md").write_text("# New\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    real = subprocess.run

    def run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "git":
            return real(cmd, *args, **kwargs)
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=300)

    monkeypatch.setattr(ms.shutil, "which", lambda name: "/usr/bin/npx")
    monkeypatch.setattr(ms.subprocess, "run", run)
    assert ms.main(["--quiet"]) == 2
    assert "could not run" in capsys.readouterr().err


def test_parsed_findings_are_filtered_to_touched_lines(repo, monkeypatch, capsys):
    (repo / "new.md").write_text("intro\n# Bad\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    row = "new.md:2 MD022/blanks-around-headings Headings should be surrounded by blank lines [x]"
    _fake_linter(monkeypatch, 1, stderr=row + "\nnew.md:99 MD031/blanks-around-fences untouched")
    assert ms.main(["--quiet"]) == 1
    out = capsys.readouterr().out
    assert "new.md:2: MD022" in out
    assert "new.md:99" not in out


def test_undecodable_file_and_bad_ref_exit_2_never_clean(repo, capsys):
    (repo / "bin.md").write_bytes(b"\xff\xfe bad\n")
    _git(repo, "add", "bin.md")
    assert ms.main(["--quiet"]) == 2
    assert "not valid UTF-8" in capsys.readouterr().err
    assert ms.main(["--base-ref", "no-such-ref", "--quiet"]) == 2


def test_empty_base_ref_is_a_usage_error_not_staged_mode(repo, capsys):
    assert ms.main(["--base-ref", "", "--quiet"]) == 2
    assert "non-empty" in capsys.readouterr().err
    assert ms.main(["--base-ref", "   ", "--quiet"]) == 2


def test_git_timeout_exits_2(repo, monkeypatch, capsys):
    seen = []

    def boom(*args, **kwargs):
        seen.append(kwargs.get("timeout"))
        raise subprocess.TimeoutExpired(cmd="git", timeout=60)

    monkeypatch.setattr(ms.subprocess, "run", boom)
    assert ms.main(["--quiet"]) == 2
    assert "timed out" in capsys.readouterr().err
    # Every git call must carry the timeout, or a stalled git hangs the hook.
    assert seen and all(t == 60 for t in seen)


def test_unreadable_file_exits_2(repo, monkeypatch, capsys):
    real = ms.Path.read_bytes

    def fail(self):
        if self.name == "old.md":
            raise OSError("denied")
        return real(self)

    monkeypatch.setattr(ms.Path, "read_bytes", fail)
    assert ms.main(["--all", "--quiet"]) == 2
    assert "unreadable" in capsys.readouterr().err


@needs_node
def test_the_versioned_pre_commit_hook_blocks_a_staged_violation(tmp_path, monkeypatch):
    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    hook_text = (ms.ROOT / "checks" / "hooks" / "pre-commit").read_text(encoding="utf-8")
    (tmp_path / "checks").mkdir()
    # The other hook steps are not under test: stub each one as a passing script.
    for name in set(re.findall(r"checks/([a-z_0-9]+\.py)", hook_text)):
        (tmp_path / "checks" / name).write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    shutil.copy(ms.__file__, tmp_path / "checks" / "markdown_structure.py")
    shutil.copy(ms.CONFIG, tmp_path / ".markdownlint-cli2.jsonc")
    hook = tmp_path / "pre-commit"
    hook.write_text(hook_text, encoding="utf-8")
    _git(tmp_path, "init", "-q")

    def run_hook():
        return subprocess.run(["sh", str(hook)], cwd=tmp_path, capture_output=True, text=True)

    (tmp_path / "bad.md").write_text("# Title\nbody\n", encoding="utf-8")
    _git(tmp_path, "add", "bad.md")
    blocked = run_hook()
    assert blocked.returncode != 0
    assert "bad.md:1: MD022" in blocked.stdout

    (tmp_path / "bad.md").write_text("# Title\n\nbody\n", encoding="utf-8")
    _git(tmp_path, "add", "bad.md")
    assert run_hook().returncode == 0