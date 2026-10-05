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

@pytest.mark.parametrize(
    "config_name", ["opencode_config.json", "opencode_review_config.json"]
)
def test_opencode_config_has_no_root_level_models_key(config_name):
    """opencode ignores a root-level "models" key entirely; every per-model
    override must live under provider.openrouter.models or it silently does
    nothing. This must fail if a root "models" key is ever added back."""
    config = json.loads((ROOT / "tools" / config_name).read_text())
    assert "models" not in config, (
        f"{config_name}: root-level 'models' key is ignored by opencode; "
        "move per-model overrides under provider.openrouter.models"
    )


@pytest.mark.parametrize(
    "config_name", ["opencode_config.json", "opencode_review_config.json"]
)
def test_opencode_config_disables_title_and_summary_agents(config_name):
    """The built-in title and summary agents are disabled through their
    supported agent settings, not per-model options."""
    config = json.loads((ROOT / "tools" / config_name).read_text())
    assert config["agent"]["title"]["disable"] is True, (
        f"{config_name}: agent.title.disable must be true"
    )
    assert config["agent"]["summary"]["disable"] is True, (
        f"{config_name}: agent.summary.disable must be true"
    )
    models = config["provider"]["openrouter"]["models"]
    assert models, f"{config_name}: expected at least one openrouter model override"
    for model_id, override in models.items():
        assert "skip-title" not in override.get("options", {}), (
            f"{config_name}: provider.openrouter.models[{model_id!r}] still sets "
            "options.skip-title, which is unsupported; disable the title agent "
            "via agent.title.disable instead"
        )

def test_bounded_run_exit_75_is_a_resource_wait_not_a_test_failure(monkeypatch, capsys):
    done = subprocess.CompletedProcess([], 75, "", "[bounded] RESOURCE WAIT timed out\n")
    monkeypatch.setattr(oc_check.subprocess, "run", lambda *a, **k: done)
    monkeypatch.setattr(oc_check, "STEPS", [])
    assert oc_check.run("pytest", ["x"], {}, 1) is False
    out = capsys.readouterr().out
    assert "resource wait timed out (bounded_run exit 75)" in out and "box busy, retry" in out
    assert "FAIL" not in out
    assert oc_check.STEPS[-1]["result"] == "RESOURCE WAIT TIMEOUT"


def test_exit_75_from_a_gate_is_still_a_plain_failure(monkeypatch, capsys):
    done = subprocess.CompletedProcess([], 75, "", "")
    monkeypatch.setattr(oc_check.subprocess, "run", lambda *a, **k: done)
    monkeypatch.setattr(oc_check, "STEPS", [])
    assert oc_check.run("hygiene", ["x"], {}, 1) is False
    assert "FAIL (rc=75)" in capsys.readouterr().out


def test_pytest_step_bounds_its_resource_wait_below_its_own_timeout():
    source = (ROOT / "tools" / "oc_check.py").read_text()
    assert '"--max-wait-s", "600"' in source


class TestSelfMetrics:
    """One best-effort JSON line per run in $OC_METRICS_DIR/oc_check.jsonl."""

    BOUNDED_WAIT = "[bounded] RESOURCE WAIT: all test slots held; retrying in 5s\n"

    @pytest.fixture
    def wt(self, tmp_path, monkeypatch):
        root = tmp_path / "worktrees" / "wt1"
        (root / "checks").mkdir(parents=True)
        (root / "checks" / "repo_hygiene.py").write_text("")
        (root / "tests").mkdir()
        (root / "tests" / "test_x.py").write_text("")
        monkeypatch.chdir(root)
        monkeypatch.setattr(oc_check, "WORKTREES", tmp_path / "worktrees")
        monkeypatch.setattr(oc_check, "tree_id", lambda r: "t1")
        for name in ("STEPS", "STEP_SECONDS", "RUN"):
            monkeypatch.setattr(oc_check, name, type(getattr(oc_check, name))())
        monkeypatch.setattr(oc_check, "WAIT", [0.0])
        monkeypatch.setenv("OC_METRICS_DIR", str(tmp_path / "metrics"))
        self.metrics = tmp_path / "metrics" / "oc_check.jsonl"
        self.stderr = ""
        monkeypatch.setattr(oc_check.subprocess, "run", self._fake_run)
        return root

    def _fake_run(self, cmd, **kw):
        err = self.stderr if "pytest" in cmd else ""
        return subprocess.CompletedProcess(cmd, 0, "", err)

    def _main(self, monkeypatch, *argv):
        monkeypatch.setattr(oc_check.sys, "argv", ["oc_check.py", *argv])
        with pytest.raises(SystemExit) as excinfo:
            oc_check.main()
        return excinfo.value.code

    def _lines(self):
        return [json.loads(x) for x in self.metrics.read_text().splitlines()]

    def test_normal_run_writes_exactly_one_line_with_the_documented_fields(self, wt, monkeypatch, capsys):
        assert self._main(monkeypatch, "tests/test_x.py") == 0
        (line,) = self._lines()
        assert set(line) == {"ts", "duration_s", "wait_s", "steps", "targets", "verdict", "exit", "mode", "worktree"}
        assert (line["verdict"], line["exit"], line["mode"]) == ("ALL GREEN", 0, "normal")
        assert (line["targets"], line["wait_s"], line["worktree"]) == (1, 0, "wt1")
        assert set(line["steps"]) == {"hygiene", "import_layers", "code_budgets", "package_readmes", "v2_lint", "pytest"}
        assert str(wt) not in json.dumps(line) and "tests/test_x.py" not in json.dumps(line)
        assert capsys.readouterr().out.rstrip().endswith("oc-check: ALL GREEN  (report: .oc_logs/oc_check_report.json)")

    def test_unwritable_metrics_dir_changes_neither_exit_code_nor_output(self, wt, monkeypatch, capsys, tmp_path):
        assert self._main(monkeypatch, "tests/test_x.py") == 0
        good = capsys.readouterr().out
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, so mkdir under it fails")
        monkeypatch.setenv("OC_METRICS_DIR", str(blocker / "sub"))
        assert self._main(monkeypatch, "tests/test_x.py") == 0
        assert capsys.readouterr().out == good

    def test_verify_run_records_mode_verify(self, wt, monkeypatch):
        (wt / ".oc_logs").mkdir()
        (wt / ".oc_logs" / "oc_check_report.json").write_text(json.dumps({"tree": "t1", "verdict": "ALL GREEN"}))
        assert self._main(monkeypatch, "--verify") == 0
        (line,) = self._lines()
        assert (line["mode"], line["verdict"], line["exit"], line["steps"]) == ("verify", "VERIFIED", 0, {})

    def test_two_runs_append_two_lines(self, wt, monkeypatch):
        self._main(monkeypatch, "tests/test_x.py")
        self._main(monkeypatch, "tests/test_x.py")
        assert len(self._lines()) == 2

    def test_a_run_that_waited_for_a_slot_records_wait_s(self, wt, monkeypatch):
        self.stderr = self.BOUNDED_WAIT * 3
        assert self._main(monkeypatch, "tests/test_x.py") == 0
        assert self._lines()[0]["wait_s"] == 15

    def test_unexpected_exception_propagates_and_records_exit_1(self, wt, monkeypatch):
        """A non-SystemExit crash from _main must reach the caller and still land
        one metrics line with exit=1 (main()'s BaseException branch), not swallow it."""
        def boom():
            # mirrors a real run: _main populates RUN after main() has reset
            # per-invocation state, so metrics only get recorded if RUN is
            # filled from inside _main, not before main() is called
            oc_check.RUN["mode"] = "normal"
            raise RuntimeError("unexpected crash")
        monkeypatch.setattr(oc_check, "_main", boom)
        with pytest.raises(RuntimeError):
            oc_check.main()
        (line,) = self._lines()
        assert line["exit"] == 1

    def test_a_malformed_wait_line_neither_crashes_the_run_nor_counts_as_wait(self, wt, monkeypatch):
        """``retrying in 1..2s`` is not a float; parsing it must not kill the run,
        and a line with no parseable number must contribute 0 to wait_s."""
        self.stderr = "[bounded] RESOURCE WAIT: all test slots held; retrying in 1..2s\n"
        assert self._main(monkeypatch, "tests/test_x.py") == 0
        assert self._lines()[0]["wait_s"] == 0

    def test_a_crash_while_writing_the_report_downgrades_the_verdict(self, wt, monkeypatch):
        """tree_id runs after RUN.update(verdict=...), so a crash there leaves the
        green verdict in place without a report behind it; the recorded row must
        say NOT GREEN, not the verdict of a run that never finished."""
        def boom(root):
            raise RuntimeError("report creation failed")
        monkeypatch.setattr(oc_check, "tree_id", boom)
        monkeypatch.setattr(oc_check.sys, "argv", ["oc_check.py", "tests/test_x.py"])
        with pytest.raises(RuntimeError):
            oc_check.main()
        (line,) = self._lines()
        assert (line["exit"], line["verdict"]) == (1, "NOT GREEN")

    def test_a_verify_after_a_normal_run_records_reset_state(self, wt, monkeypatch):
        """STEPS/STEP_SECONDS/WAIT/RUN are module-level; a second run in the same
        process (here --verify over the first run's report) must not inherit the
        first run's wait, steps or target count."""
        self.stderr = self.BOUNDED_WAIT
        assert self._main(monkeypatch, "tests/test_x.py") == 0
        self.stderr = ""
        assert self._main(monkeypatch, "--verify") == 0
        _, second = self._lines()
        assert (second["mode"], second["verdict"], second["exit"], second["targets"],
                second["steps"], second["wait_s"]) == ("verify", "VERIFIED", 0, 0, {}, 0)

    def test_a_pytest_timeout_still_totals_wait_lines_from_bytes_and_text(self, wt, monkeypatch):
        """TimeoutExpired carries stdout as bytes and stderr as text; both wait
        lines must reach the shared total even though the step itself failed."""
        exc = subprocess.TimeoutExpired(
            ["pytest"], 900,
            output=self.BOUNDED_WAIT.replace("in 5s", "in 3s").encode(),
            stderr=self.BOUNDED_WAIT.replace("in 5s", "in 2s"),
        )
        def raiser(*args, **kwargs):
            raise exc
        monkeypatch.setattr(oc_check.subprocess, "run", raiser)
        assert oc_check.run("pytest", ["x"], {}, 900) is False
        assert oc_check.WAIT[0] == 5

    @pytest.mark.parametrize("mode", ["normal", "timeout"])
    @pytest.mark.parametrize("text", [
        "[bounded] RESOURCE WAIT: slots held; retrying in 5seconds\n",
        "> [bounded] RESOURCE WAIT: slots held; retrying in 5s\n",
        "pytest said: [bounded] RESOURCE WAIT: slots held; retrying in 5s\n",
    ])
    def test_only_complete_bounded_run_lines_count_as_wait(self, wt, monkeypatch, mode, text):
        """A malformed suffix, a quoted line or embedded diagnostic text is not a
        bounded_run wait message; both the completed and the timed-out path ignore it."""
        if mode == "normal":
            self.stderr = text
            assert self._main(monkeypatch, "tests/test_x.py") == 0
            assert self._lines()[0]["wait_s"] == 0
        else:
            def raiser(*args, **kwargs):
                raise subprocess.TimeoutExpired(["pytest"], 900, output=b"", stderr=text)
            monkeypatch.setattr(oc_check.subprocess, "run", raiser)
            assert oc_check.run("pytest", ["x"], {}, 900) is False
            assert oc_check.WAIT[0] == 0

    @pytest.mark.parametrize("mode", ["normal", "timeout"])
    def test_streams_are_parsed_separately_and_never_concatenate_a_fabricated_wait(
        self, wt, monkeypatch, mode
    ):
        """CodeRabbit: stdout ending in a partial wait line and stderr opening
        with its remainder fabricate a complete 'retrying in 12s' only once the
        two streams are joined before parsing; each stream alone carries no
        wait message. Both the completed-process and the TimeoutExpired path
        must finish runner handling and contribute 0 to WAIT."""
        prefix = "[bounded] RESOURCE WAIT: slots held; retrying in 1"
        suffix = "2s\n"
        monkeypatch.setattr(oc_check, "STEPS", [])
        monkeypatch.setattr(oc_check, "STEP_SECONDS", {})
        if mode == "normal":
            done = subprocess.CompletedProcess(["x"], 0, prefix, suffix)
            monkeypatch.setattr(oc_check.subprocess, "run", lambda *a, **k: done)
            assert oc_check.run("pytest", ["x"], {}, 900) is True
        else:
            exc = subprocess.TimeoutExpired(["x"], 900, output=prefix.encode(), stderr=suffix)
            def raiser(*args, **kwargs):
                raise exc
            monkeypatch.setattr(oc_check.subprocess, "run", raiser)
            assert oc_check.run("pytest", ["x"], {}, 900) is False
        assert oc_check.WAIT[0] == 0
