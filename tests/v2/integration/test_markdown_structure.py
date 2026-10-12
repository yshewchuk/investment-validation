"""Tests for checks/markdown_structure.py: one failing and one passing fixture per rule."""
# packages: ops

import subprocess

import pytest

from checks import markdown_structure as ms


def rules(text):
    return sorted(v.rule for v in ms.lint_text(text))


def test_heading_needs_blank_before():
    assert rules("para\n# Title\n\nbody\n") == ["heading-before"]
    assert rules("# First line\n\nbody\n\n## Two\n\nbody\n") == []


def test_heading_needs_blank_after_but_may_end_file():
    assert rules("# Title\nbody\n") == ["heading-after"]
    assert rules("body\n\n# Title\n") == []


def test_table_needs_blank_before():
    assert rules("para\n| a | b |\n|---|---|\n| 1 | 2 |\n") == ["table-before"]
    assert rules("para\n\n| a | b |\n|---|---|\n| 1 | 2 |\n") == []


def test_table_must_not_absorb_following_paragraph():
    assert rules("| a | b |\n|---|---|\n| 1 | 2 |\nclosing sentence\n") == ["table-after"]
    assert rules("| a | b |\n|---|---|\n| 1 | 2 |\n\nclosing sentence\n") == []
    assert rules("| a | b |\n|---|---|\n| 1 | 2 |") == []


def test_pipe_in_following_paragraph_is_not_a_table_row():
    text = "| a | b |\n|---|---|\n| 1 | 2 |\nuse a | b here\n"
    assert rules(text) == ["table-after"]
    assert rules("a | b\n--|--\n1 | 2\n\nafter\n") == []
    assert rules("a | b\n--|--\n1 | 2\nx | y\n\nafter\n") == []


def test_fence_needs_blank_before_and_after():
    assert rules("para\n```\ncode\n```\n\nnext\n") == ["fence-before"]
    assert rules("para\n\n```\ncode\n```\nnext\n") == ["fence-after"]
    assert rules("para\n\n```\ncode\n```\n\nnext\n") == []


def test_unclosed_fence_is_reported_at_opening_line():
    found = ms.lint_text("para\n\n```\ncode\n")
    assert [(v.line, v.rule) for v in found] == [(3, "fence-unclosed")]


def test_fenced_content_is_not_read_as_heading_or_table():
    text = "intro\n\n```\n# not a heading\n| a | b |\n|---|---|\nplain\n```\n\nend\n"
    assert rules(text) == []
    tilde = "intro\n\n~~~\n# x\n```\n# still inside\n~~~\n\nend\n"
    assert rules(tilde) == []


def test_shorter_or_other_fence_does_not_close():
    assert rules("a\n\n````\n```\ncode\n````\n\nb\n") == []
    assert [v.rule for v in ms.lint_text("a\n\n````\n```\ncode\n")] == ["fence-unclosed"]


def test_indented_fence_in_list_is_exempt_from_blank_rules():
    assert rules("- item\n  ```\n  code\n  ```\n- next\n") == []


def test_lists_are_not_checked():
    assert rules("para\n- a\n- b\npara two\n") == []


def test_inline_code_span_at_line_start_is_not_a_fence():
    assert rules("para\n\n   ```json``` appendix\nmore\n") == []


def test_changed_lines_parses_hunks_and_deletions():
    diff = "@@ -1,0 +2,2 @@\n+a\n+b\n@@ -9,3 +11 @@\n-x\n+y\n@@ -20,2 +22,0 @@\n-gone\n"
    assert ms.changed_lines(diff) == {2, 3, 11, 22, 23}


def test_scoped_keeps_only_violations_touching_changed_lines():
    found = ms.lint_text("# A\ntext\n\n# B\ntext\n")
    assert len(found) == 2
    assert [v.line for v in ms.scoped(found, {5})] == [4]
    assert ms.scoped(found, set()) == []
    unclosed = ms.lint_text("para\n\n```\ncode\n")
    assert ms.scoped(unclosed, {4}) == unclosed
    assert ms.scoped(found, None) == found


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


def test_staged_mode_ignores_untouched_legacy_defects(repo, capsys):
    (repo / "new.md").write_text("# New\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 0
    assert ms.main(["--all", "--quiet"]) == 1
    assert "old.md:1: heading-after" in capsys.readouterr().out


def test_staged_legacy_file_flags_only_touched_defect(repo, capsys):
    # old.md has a legacy defect on line 1 (heading-after). Change an unrelated line.
    (repo / "old.md").write_text("# Old\nlegacy defect\n\nbody\n\nmore\n", encoding="utf-8")
    _git(repo, "add", "old.md")
    assert ms.main(["--quiet"]) == 0
    capsys.readouterr()
    # Now touch the defect line itself.
    (repo / "old.md").write_text("# Older\nlegacy defect\n\nbody\n\nmore\n", encoding="utf-8")
    _git(repo, "add", "old.md")
    assert ms.main(["--quiet"]) == 1
    assert "old.md:1: heading-after" in capsys.readouterr().out


def test_unclosed_fence_reported_when_closing_fence_deleted(repo, capsys):
    (repo / "f.md").write_text("intro\n\n```\ncode\n```\n\nend\n", encoding="utf-8")
    _git(repo, "add", "f.md")
    _git(repo, "commit", "-qm", "add f")
    (repo / "f.md").write_text("intro\n\n```\ncode\n\nend\n", encoding="utf-8")
    _git(repo, "add", "f.md")
    assert ms.main(["--quiet"]) == 1
    assert "f.md:3: fence-unclosed" in capsys.readouterr().out


def test_unclosed_fence_reported_when_text_added_after_it(repo, capsys):
    (repo / "g.md").write_text("intro\n\n```\ncode\n", encoding="utf-8")
    _git(repo, "add", "g.md")
    _git(repo, "commit", "-qm", "add g (already unclosed)")
    (repo / "g.md").write_text("intro\n\n```\ncode\nmore\n", encoding="utf-8")
    _git(repo, "add", "g.md")
    assert ms.main(["--quiet"]) == 1
    assert "g.md:3: fence-unclosed" in capsys.readouterr().out


def test_staged_mode_flags_a_new_defect(repo, capsys):
    (repo / "new.md").write_text("intro\n# Bad\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 1
    assert "new.md:2: heading-before" in capsys.readouterr().out


def test_base_ref_mode_checks_only_the_branch_diff(repo, capsys):
    _git(repo, "branch", "base")
    (repo / "old.md").write_text("# Old\nlegacy defect\n\n| a | b |\n|---|---|\n| 1 | 2 |\nabsorbed\n",
                                 encoding="utf-8")
    _git(repo, "commit", "-qam", "edit")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 1
    out = capsys.readouterr().out
    assert "old.md:7: table-after" in out
    assert "heading-after" not in out


def test_undecodable_file_and_bad_ref_exit_2_never_clean(repo, capsys):
    (repo / "bin.md").write_bytes(b"\xff\xfe bad\n")
    _git(repo, "add", "bin.md")
    assert ms.main(["--quiet"]) == 2
    assert "not valid UTF-8" in capsys.readouterr().err
    assert ms.main(["--base-ref", "no-such-ref", "--quiet"]) == 2


def test_quiet_suppresses_the_clean_message(repo, capsys):
    (repo / "ok.md").write_text("# Ok\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "ok.md")
    assert ms.main(["--quiet"]) == 0
    assert capsys.readouterr().out == ""
    assert ms.main([]) == 0
    assert "clean" in capsys.readouterr().out


def test_git_timeout_exits_2(repo, monkeypatch, capsys):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=60)

    monkeypatch.setattr(ms.subprocess, "run", boom)
    assert ms.main(["--quiet"]) == 2
    assert "timed out" in capsys.readouterr().err


def test_unreadable_file_exits_2(repo, monkeypatch, capsys):
    real = ms.Path.read_bytes

    def fail(self):
        if self.name == "old.md":
            raise OSError("denied")
        return real(self)

    monkeypatch.setattr(ms.Path, "read_bytes", fail)
    assert ms.main(["--all", "--quiet"]) == 2
    assert "unreadable" in capsys.readouterr().err


def test_user_git_color_and_external_diff_config_do_not_hide_violations(repo, monkeypatch, capsys):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "color.ui")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "always")
    monkeypatch.setenv("GIT_CONFIG_KEY_1", "diff.external")
    monkeypatch.setenv("GIT_CONFIG_VALUE_1", "false")
    (repo / "new.md").write_text("intro\n# Bad\n\nbody\n", encoding="utf-8")
    _git(repo, "add", "new.md")
    assert ms.main(["--quiet"]) == 1
    assert "new.md:2: heading-before" in capsys.readouterr().out


def test_list_content_is_ignored_for_headings_and_tables():
    indented_table = "- item\n  | a | b |\n  |---|---|\n  | 1 | 2 |\n  tail\n"
    assert rules(indented_table) == []
    marker_pipe = "- a | b\n- --|--\n- 1 | 2\nafter\n"
    assert rules(marker_pipe) == []
    assert rules("- item\n  # not a heading\n  text\n") == []
    assert rules("1. item\n   | a | b |\n   |---|---|\n") == []

