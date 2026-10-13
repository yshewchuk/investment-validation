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


def test_unclosed_indented_fence_in_list_item_is_reported():
    found = ms.lint_text("- item\n  ```\n  code\n- next\n")
    assert [(v.line, v.rule) for v in found] == [(2, "fence-unclosed")]
    assert ms.scoped(found, {4}) == found


def test_fence_opening_on_a_list_marker_line_is_recognised():
    assert rules("- ```\n  code\n  ```\n- next\n") == []
    assert rules("1. ```python\n   code\n   ```\n") == []
    found = ms.lint_text("- ```\n  code\n- next\n")
    assert [(v.line, v.rule) for v in found] == [(1, "fence-unclosed")]


def test_empty_base_ref_is_a_usage_error_not_staged_mode(repo, capsys):
    assert ms.main(["--base-ref", "", "--quiet"]) == 2
    assert "non-empty" in capsys.readouterr().err
    assert ms.main(["--base-ref", "   ", "--quiet"]) == 2


def test_pipeless_header_table_uses_column_count_for_rows():
    # Rows with the header's column count stay in the table.
    assert rules("a | b\n--|--\n1 | 2\nx | y\n\nafter\n") == []
    # A pipe-bearing paragraph with a different column count is not a row.
    assert rules("a | b\n--|--\n1 | 2\nsee x | y | z\n") == ["table-after"]
    # A line with a pipe-less body is not a row either.
    assert rules("a | b\n--|--\n1 | 2\nplain text\n") == ["table-after"]


def test_known_limit_cell_count_paragraph_under_pipeless_header_is_a_row():
    # By GFM rules a line with the header's cell count IS a table row, and nothing
    # distinguishes it from an intended one, so it is deliberately not flagged.
    assert rules("a | b\n--|--\n1 | 2\nuse a | b here\n") == []


def test_closer_less_indented_than_a_list_item_opener_does_not_close():
    found = ms.lint_text("- item\n  ```\n  code\n```\nafter\n")
    assert [(v.line, v.rule) for v in found] == [(2, "fence-unclosed")]
    # Same indent as the opener closes it.
    assert rules("- item\n  ```\n  code\n  ```\n\nafter\n") == []
    # A marker-line opener (content column 2) behaves the same.
    found = ms.lint_text("- ```\n  code\n```\nafter\n")
    assert [v.rule for v in found] == ["fence-unclosed"]


def test_gfm_fence_rules_matrix():
    # Opener indent 1-3 spaces at top level: a column-0 closer is valid.
    assert rules("a\n\n   ```\ncode\n```\n\nb\n") == []
    assert rules("- item\n\ntext\n\n   ```\ncode\n```\n\nb\n") == []
    # 4+ spaces outside a list is an indented code block, not a fence.
    assert rules("a\n\n    ```\n    x\n\nb\n") == []
    # Closer needs at least the opener's length and the same character.
    assert [v.rule for v in ms.lint_text("a\n\n````\ncode\n```\n")] == ["fence-unclosed"]
    assert [v.rule for v in ms.lint_text("a\n\n```\ncode\n~~~\n")] == ["fence-unclosed"]
    assert rules("a\n\n```\ncode\n`````\n\nb\n") == []
    # An info string is allowed on the opener only.
    assert rules("a\n\n```python\ncode\n```\n\nb\n") == []
    assert [v.rule for v in ms.lint_text("a\n\n```\ncode\n```python\n")] == ["fence-unclosed"]
    # A tilde opener's info string may contain backticks.
    assert rules("a\n\n~~~ a`b\ncode\n~~~\n\nb\n") == []
    # Closer indent is at most 3 beyond the opener's.
    assert rules("a\n\n```\ncode\n   ```\n\nb\n") == []
    assert [v.rule for v in ms.lint_text("a\n\n```\ncode\n    ```\n")] == ["fence-unclosed"]


def test_staged_mode_reads_the_index_not_the_worktree(repo, capsys):
    bad, fine = "intro\n# Bad\n\nbody\n", "# Fine\n\nbody\n"
    f = repo / "w.md"
    # Staged version is bad, the worktree was then fixed but not staged: still flagged.
    f.write_text(bad, encoding="utf-8")
    _git(repo, "add", "w.md")
    f.write_text(fine, encoding="utf-8")
    assert ms.main(["--quiet"]) == 1
    assert "w.md:2: heading-before" in capsys.readouterr().out
    # Staged version is fine, the worktree was then broken but not staged: clean.
    f.write_text(fine, encoding="utf-8")
    _git(repo, "add", "w.md")
    f.write_text(bad, encoding="utf-8")
    assert ms.main(["--quiet"]) == 0


def test_base_ref_mode_reads_committed_content_not_the_worktree(repo, capsys):
    bad, fine = "intro\n# Bad\n\nbody\n", "# Fine\n\nbody\n"
    f = repo / "w.md"
    _git(repo, "branch", "base")
    # Committed version is bad, the worktree was then fixed (uncommitted): still flagged.
    f.write_text(bad, encoding="utf-8")
    _git(repo, "add", "w.md")
    _git(repo, "commit", "-qm", "bad")
    f.write_text(fine, encoding="utf-8")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 1
    assert "w.md:2: heading-before" in capsys.readouterr().out
    # Committed version is fine, the worktree was then broken (uncommitted): clean.
    _git(repo, "commit", "-qam", "fix")
    f.write_text(bad, encoding="utf-8")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 0


def test_each_blank_line_rule_reports_exact_positions():
    def fields(text):
        return sorted((v.rule, v.line, v.near) for v in ms.lint_text(text))

    cases = [
        ("para\n# Title\n\nbody\n", [("heading-before", 2, 1)], {1}),
        ("# Title\nbody\n", [("heading-after", 1, 2)], {2}),
        ("para\n| a | b |\n|---|---|\n| 1 | 2 |\n", [("table-before", 2, 1)], {1}),
        ("| a | b |\n|---|---|\n| 1 | 2 |\nclosing\n", [("table-after", 4, 3)], {3}),
        ("para\n```\ncode\n```\n\nnext\n", [("fence-before", 2, 1)], {1}),
        ("para\n\n```\ncode\n```\nnext\n", [("fence-after", 5, 6)], {6}),
    ]
    for text, expected, neighbour in cases:
        assert fields(text) == expected, text
        found = ms.lint_text(text)
        # A change to ONLY the neighbouring line is in scope; an unrelated line is not.
        assert ms.scoped(found, neighbour) == found, text
        assert ms.scoped(found, {99}) == [], text


def test_table_scope_covers_the_whole_block():
    text = "| a | b |\n|---|---|\n| 1 | 2 |\nclosing\n"
    found = ms.lint_text(text)
    for line in (1, 2, 3, 4):
        assert ms.scoped(found, {line}) == found, line
    before = ms.lint_text("para\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    assert ms.scoped(before, {3}) == before  # the delimiter row alone


def test_delimiter_only_edit_that_creates_a_table_is_flagged(repo, capsys):
    f = repo / "t.md"
    f.write_text("| a | b |\n| c | d |\n| e | f |\nparagraph\n", encoding="utf-8")
    _git(repo, "add", "t.md")
    _git(repo, "commit", "-qm", "text rows")
    _git(repo, "branch", "base")
    f.write_text("| a | b |\n|---|---|\n| e | f |\nparagraph\n", encoding="utf-8")
    _git(repo, "add", "t.md")
    assert ms.main(["--quiet"]) == 1
    assert "t.md:4: table-after" in capsys.readouterr().out
    _git(repo, "commit", "-qm", "delimiter")
    assert ms.main(["--base-ref", "base", "--quiet"]) == 1
    assert "t.md:4: table-after" in capsys.readouterr().out


def test_escaped_pipes_and_unequal_counts():
    assert ms._cells("a \\| b | c") == 2
    assert ms._cells("| a | b |") == 2
    assert ms._cells("| a \\| b |") == 1
    assert rules("a \\| b | c\n--|--\n1 | 2\n\nafter\n") == []
    assert rules("a \\| b | c\n--|--\n1 | 2\nafter\n") == ["table-after"]
    assert rules("| a \\| b |\n|---|\n| 1 |\nafter\n") == ["table-after"]
    # Header and delimiter cell counts differ: not a table, so no table rules.
    assert rules("para\na | b\n--|--|--\n1 | 2\nafter\n") == []


def test_closer_bounds_use_the_container_indent():
    # Top level, opener indented 3: a 3-space closer closes, a 6-space one is content.
    assert rules("a\n\n   ```\ncode\n   ```\n\nb\n") == []
    assert [v.rule for v in ms.lint_text("a\n\n   ```\ncode\n      ```\n")] == ["fence-unclosed"]
    # Inside "- item" (content column 2): a 4-space opener is closed by a 2-space closer.
    assert rules("- item\n\n    ```\n  code\n  ```\n\nb\n") == []
    # Closers up to 3 beyond the container column close; 4 beyond do not.
    assert rules("- item\n  ```\n  code\n     ```\n") == []
    assert [v.rule for v in ms.lint_text("- item\n  ```\n  code\n      ```\n")] == ["fence-unclosed"]


def test_nested_list_items_pick_the_right_container():
    assert rules("- a\n  - b\n\n    ```\n  x\n    ```\n") == []
    assert rules("- a\n  - b\n\n  ```\n  x\n  ```\n\nz\n") == []


def test_blockquoted_fence_is_not_checked():
    assert rules("> ```\n> # x\n") == []


def test_alignment_delimiters_are_tables():
    assert rules("| a | b |\n|:---|---:|\n| 1 | 2 |\nafter\n") == ["table-after"]
    assert rules("| a | b |\n|:---:|:---:|\n| 1 | 2 |\n\nafter\n") == []
    assert rules("a | b\n:-:|-:\n1 | 2\nafter\n") == ["table-after"]


def test_heading_regex_edges():
    assert rules("para\n#hashtag\n") == []
    assert rules("para\n####### seven\n") == []
    assert rules("para\n#\n") == ["heading-before"]


def test_delimiter_like_lines_are_not_table_headers():
    assert rules("para\n|---|---|\n|---|---|\n") == []
    assert rules("para\na | b\n---\n") == []
    assert rules("1. a | b\n--|--\n1 | 2\nafter\n") == []


def test_every_list_marker_style_opens_a_fence():
    for opener in ("- ```", "* ```", "+ ```", "2) ```", "3. ```"):
        found = ms.lint_text(opener + "\n  code\n")
        assert [v.rule for v in found] == ["fence-unclosed"], opener


def test_closing_fence_must_have_no_info_string():
    assert rules("a\n\n```\ncode\n```python\nmore\n```\n\nb\n") == []
    assert [v.rule for v in ms.lint_text("a\n\n```\ncode\n```python\nmore\n")] == ["fence-unclosed"]


def test_scoped_matches_the_reported_line_as_well_as_the_neighbour():
    found = ms.lint_text("# A\ntext\n\n# B\ntext\n")
    assert [v.line for v in ms.scoped(found, {1})] == [1]
    assert [v.line for v in ms.scoped(found, {2})] == [1]


def test_staged_deletion_of_a_markdown_file_is_not_read(repo):
    _git(repo, "rm", "-q", "old.md")
    assert ms.main(["--quiet"]) == 0


def test_base_that_moved_after_branching_does_not_flag_main_s_own_changes(repo, capsys):
    # CI checks out a merge commit; the base branch tip is newer than the PR's
    # recorded base sha. Three-dot against the CURRENT base tip must flag only the
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
    assert "moved.md:1: heading-after" in capsys.readouterr().out


def test_closer_follows_commonmark_not_the_opener_regex():
    # A list-marker line or a 4-space-indented fence run inside an open fence is content.
    assert rules("a\n\n```\n- ```\ncode\n```\n\nb\n") == []
    assert rules("a\n\n```\n    ```\ncode\n```\n\nb\n") == []
    # A valid closer indented up to 3 spaces still closes.
    assert rules("a\n\n```\ncode\n   ```\n\nb\n") == []
    # Nested list opener (content column 4): closer at the same column closes it ...
    assert rules("- a\n  - ```\n    code\n    ```\n") == []
    # ... but 8 spaces is more than 3 beyond the opener's column, so it is content.
    found = ms.lint_text("- a\n  - ```\n    code\n        ```\n")
    assert [v.rule for v in found] == ["fence-unclosed"]
    # A closer with trailing text is not a closer.
    assert [v.rule for v in ms.lint_text("a\n\n```\ncode\n``` x\n")] == ["fence-unclosed"]


def test_the_versioned_pre_commit_hook_blocks_a_staged_violation(tmp_path, monkeypatch):
    import re
    import shutil

    for var in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(var, raising=False)
    hook_text = (ms.ROOT / "checks" / "hooks" / "pre-commit").read_text(encoding="utf-8")
    (tmp_path / "checks").mkdir()
    # The other hook steps are not under test: stub each one as a passing script.
    for name in set(re.findall(r"checks/([a-z_0-9]+\.py)", hook_text)):
        (tmp_path / "checks" / name).write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    shutil.copy(ms.__file__, tmp_path / "checks" / "markdown_structure.py")
    hook = tmp_path / "pre-commit"
    hook.write_text(hook_text, encoding="utf-8")
    _git(tmp_path, "init", "-q")

    def run_hook():
        return subprocess.run(["sh", str(hook)], cwd=tmp_path, capture_output=True, text=True)

    (tmp_path / "bad.md").write_text("# Title\nbody\n", encoding="utf-8")
    _git(tmp_path, "add", "bad.md")
    blocked = run_hook()
    assert blocked.returncode != 0
    assert "bad.md:1: heading-after" in blocked.stdout

    (tmp_path / "bad.md").write_text("# Title\n\nbody\n", encoding="utf-8")
    _git(tmp_path, "add", "bad.md")
    assert run_hook().returncode == 0


def test_crlf_files_are_linted_like_lf_files():
    clean = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\ncode\n```\n\nend\n"
    assert rules(clean.replace("\n", "\r\n")) == []
    # Violations are still found, on the same line numbers, in a CRLF file.
    bad = "# Title\nbody\n\n```\ncode\n```\nafter\n"
    assert ms.lint_text(bad.replace("\n", "\r\n")) == ms.lint_text(bad)
    assert sorted(v.rule for v in ms.lint_text(bad)) == ["fence-after", "heading-after"]
    # An unclosed fence is still unclosed with CRLF endings.
    assert [v.rule for v in ms.lint_text("a\r\n\r\n```\r\ncode\r\n")] == ["fence-unclosed"]


