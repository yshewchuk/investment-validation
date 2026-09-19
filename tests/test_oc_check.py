"""The pilot's one shell command: ``tools/oc_check.py``.

It is the only command opencode is allowed to run, so its argument filter and
its proof that a report still matches the current worktree are the parts with
teeth. Both are pinned here against a real git repo under ``tmp_path``.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "oc_check", ROOT / "tools" / "oc_check.py"
)
oc_check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oc_check)


@pytest.fixture
def repo(tmp_path):
    """A real git repo with one committed file. ``.oc_logs`` is excluded so
    writing the report does not itself change the tree id."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "a.txt").write_text("one\n")
    git = ["git", "-C", str(root)]
    subprocess.run(git + ["add", "a.txt"], check=True)
    subprocess.run(
        git
        + [
            "-c",
            "user.name=oc-check",
            "-c",
            "user.email=oc-check@example.com",
            "commit",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )
    (root / ".git" / "info" / "exclude").write_text(".oc_logs/\n")
    return root


@pytest.mark.parametrize(
    "target", ["tests/test_a.py", "tests/test_a.py::test_x"]
)
def test_arg_re_accepts_well_formed_targets(target):
    assert oc_check.ARG_RE.match(target)


@pytest.mark.parametrize(
    "target", ["tests/../x.py", "foo.py", "tests/a.py; rm -rf /"]
)
def test_arg_re_rejects_traversal_outside_tests_and_shell_metacharacters(target):
    assert not oc_check.ARG_RE.match(target)


class TestTreeId:
    def test_stable_across_calls_then_tracks_edits_and_new_files(self, repo):
        first = oc_check.tree_id(repo)
        assert oc_check.tree_id(repo) == first
        (repo / "a.txt").write_text("two\n")
        edited = oc_check.tree_id(repo)
        assert edited != first
        (repo / "b.txt").write_text("new\n")
        assert oc_check.tree_id(repo) != edited


class TestVerify:
    def _write_report(self, tree, verdict):
        report = Path(".oc_logs") / "oc_check_report.json"
        report.parent.mkdir(exist_ok=True)
        report.write_text(json.dumps({"tree": tree, "verdict": verdict}))

    def test_missing_report_exits_3(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        with pytest.raises(SystemExit) as excinfo:
            oc_check.verify(repo)
        assert excinfo.value.code == 3

    def test_all_green_at_current_tree_exits_0(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        self._write_report(oc_check.tree_id(repo), "ALL GREEN")
        with pytest.raises(SystemExit) as excinfo:
            oc_check.verify(repo)
        assert excinfo.value.code == 0

    def test_edit_after_the_run_exits_4(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        self._write_report(oc_check.tree_id(repo), "ALL GREEN")
        (repo / "a.txt").write_text("stale\n")
        with pytest.raises(SystemExit) as excinfo:
            oc_check.verify(repo)
        assert excinfo.value.code == 4

    def test_not_green_at_current_tree_exits_1(self, repo, monkeypatch):
        monkeypatch.chdir(repo)
        self._write_report(oc_check.tree_id(repo), "NOT GREEN")
        with pytest.raises(SystemExit) as excinfo:
            oc_check.verify(repo)
        assert excinfo.value.code == 1


def test_opencode_config_denies_the_dangerous_edits_and_everything_else():
    config = json.loads((ROOT / "tools" / "opencode_config.json").read_text())
    assert config["permission"]["bash"]["*"] == "deny"
    assert config["permission"]["edit"]["tools/oc_check.py"] == "deny"
    assert config["permission"]["edit"][".oc_logs/*"] == "deny"