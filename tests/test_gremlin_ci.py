"""pytest-gremlins runner: command building, worker sizing, cache/report handling
and the gremlins workflow candidate's shape.

Every test here is a plain mock test: it never runs pytest-gremlins or pytest,
never touches ``data/`` and never mutates a real file. The gremlin subprocess is
stubbed. The cache-fingerprint tests ``git init`` a synthetic tree in ``tmp_path``
so "tracked" is real there -- git is a declared tool input the fingerprint reads,
not the runner under test -- and never touch the repo working tree. The candidate
workflow (``tools/gremlin_workflow_candidate.yml``) is parsed as YAML only. The
exhaustive engine/v2 module-ownership test lives in ``tests/test_mutation_ci.py``
and is unchanged -- gremlin_pilot reuses the same ``mutation_pilot.toml``
partition, so that single ownership test governs both backends.
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gremlin_pilot as gp  # noqa: E402

CANDIDATE = ROOT / "tools" / "gremlin_workflow_candidate.yml"
WORKFLOW = yaml.safe_load(CANDIDATE.read_text())
JOBS = WORKFLOW["jobs"]

DEFAULTS = {"pytest_args": ["-p", "no:cacheprovider"],
            "deselect": ["tests/test_a.py::needs_data"]}


def cfg(**modules):
    return {"defaults": DEFAULTS, "modules": modules or {"toy": {"why": "x"}}}


def run_args(module="toy", *, workers=None, fresh=False, nproc=8, cwd=None):
    return types.SimpleNamespace(module=module, workers=workers, fresh=fresh,
                                 nproc=nproc, cwd=cwd)


class FakeProc:
    def __init__(self, lines=(), rc=0, on_read=None):
        self._lines = list(lines)
        self._rc = rc
        self._on_read = on_read
        self._read = False

    @property
    def stdout(self):
        if not self._read and self._on_read is not None:
            self._on_read()
            self._read = True
        return iter(self._lines)

    def wait(self):
        return self._rc


def stub_popen(monkeypatch, *, rc=0, lines=(), on_read=None):
    calls: list[tuple[list[str], dict]] = []

    def fake_popen(cmd, **kw):
        calls.append((cmd, kw))
        return FakeProc(lines=lines, rc=rc, on_read=on_read)

    monkeypatch.setattr(gp.subprocess, "Popen", fake_popen)
    return calls


def stub_partition(monkeypatch, targets=("engine/a.py", "engine/b.py"),
                   tests=("tests/test_a.py",)):
    monkeypatch.setattr(gp.pilot, "module_cfg", lambda cfg_, name: cfg_["modules"][name])
    monkeypatch.setattr(gp.pilot, "mutate_files",
                        lambda cfg_, name, tracked=None: list(targets))
    monkeypatch.setattr(gp.pilot, "test_files",
                        lambda cfg_, name, tracked=None: list(tests))


# -- command building --------------------------------------------------------

def test_command_selects_targets_and_tests_and_parallel_workers():
    cmd = gp.build_pytest_command(["engine/a.py", "engine/b.py"], ["tests/test_a.py"],
                                  3, fresh=False, deselect=["tests/test_a.py::needs_data"],
                                  pytest_args=["-p", "no:cacheprovider"],
                                  python="python")
    assert cmd[:3] == ["python", "-m", "pytest"]
    assert "--gremlins" in cmd
    assert "--gremlin-targets=engine/a.py,engine/b.py" in cmd   # comma-separated, in order
    assert "--gremlin-workers=3" in cmd
    assert "--gremlin-cache" in cmd
    assert "--gremlin-report=json" in cmd
    assert "-p" in cmd and "no:cacheprovider" in cmd
    assert "no:xdist" not in cmd      # disabling xdist aborts collection under gremlins
    assert "--deselect=tests/test_a.py::needs_data" in cmd
    assert cmd[-1] == "tests/test_a.py"                          # test files are positional args


def test_command_never_enables_batch_mode():
    cmd = gp.build_pytest_command(["engine/a.py"], ["tests/test_a.py"], 2, fresh=True,
                                  deselect=[], pytest_args=["-p", "no:cacheprovider"])
    joined = " ".join(cmd)
    assert "--gremlin-batch" not in joined       # batch unions test pools -> false timeouts
    assert "xdist" not in joined                 # never disabled (gremlins' xdist hooks) and never invoked


def test_command_leaves_xdist_loaded_and_never_passes_n():
    # pytest-gremlins 1.9.0 implements pytest-xdist's hooks (pytest_configure_node):
    # `-p no:xdist` made pluggy abort collection with PluginValidationError: unknown
    # hook (CI run 36059422920 -- pytest exit 3, no coverage/gremlins/gremlins.json,
    # every module correctly a tool failure). So xdist stays loaded. It cannot fight
    # gremlins for cores: parallelism is --gremlin-workers only, and a pytest
    # -n/--numprocesses switch never appears.
    cmd = gp.build_pytest_command(["engine/a.py"], ["tests/test_a.py"], 4, fresh=False,
                                  deselect=[], pytest_args=["-p", "no:cacheprovider"])
    joined = " ".join(cmd)
    assert "no:xdist" not in joined
    assert "-n" not in cmd and "--numprocesses" not in joined
    assert "--gremlin-workers=4" in cmd          # the only parallelism switch


def test_deselects_appear_before_the_test_files():
    cmd = gp.build_pytest_command(["engine/a.py"], ["tests/test_a.py", "tests/test_b.py"], 2,
                                  fresh=False, deselect=["tests/test_b.py::needs_browser"],
                                  pytest_args=["-p", "no:cacheprovider"])
    dsel = cmd.index("--deselect=tests/test_b.py::needs_browser")
    assert dsel < cmd.index("tests/test_a.py")
    assert cmd[-2:] == ["tests/test_a.py", "tests/test_b.py"]


# -- worker sizing -----------------------------------------------------------

def test_workers_default_to_two_locally():
    assert gp.resolve_workers(nproc=8) == 2
    assert gp.resolve_workers(nproc=1) == 1  # never above what the box has


def test_workers_under_ci_are_nproc_capped_at_four():
    assert gp.resolve_workers(ci=True, nproc=2) == 2
    assert gp.resolve_workers(ci=True, nproc=4) == 4
    assert gp.resolve_workers(ci=True, nproc=16) == 4  # capped at 4


def test_explicit_workers_are_capped_not_ignored():
    assert gp.resolve_workers(1, ci=True, nproc=8) == 1     # explicit small wins
    assert gp.resolve_workers(16, ci=True, nproc=8) == 4    # capped at the CI ceiling
    assert gp.resolve_workers(8, ci=False, nproc=4) == 4    # capped at nproc locally
    assert gp.resolve_workers(3, ci=False, nproc=8) == 3    # explicit within bounds is kept


def test_is_ci_reads_the_github_actions_flag():
    assert gp.is_ci({"GITHUB_ACTIONS": "true"})
    assert gp.is_ci({"GITHUB_ACTIONS": "TRUE"})
    assert not gp.is_ci({"GITHUB_ACTIONS": "false"})
    assert not gp.is_ci({})


# -- environment / thread bounds --------------------------------------------

def test_thread_env_pins_every_blas_pool_to_one():
    env = gp.thread_env({"OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "4", "KEEP": "yes"})
    for var in gp._THREAD_VARS:
        assert env[var] == "1"
    assert env["KEEP"] == "yes"


def test_thread_env_leaves_harness_vars_untouched():
    # gremlins/pytest set their own; the parent is not stripped.
    env = gp.thread_env({"MUTANT_UNDER_TEST": "x", "PYTEST_CURRENT_TEST": "t::u (call)"})
    assert env["MUTANT_UNDER_TEST"] == "x"
    assert env["PYTEST_CURRENT_TEST"] == "t::u (call)"
    assert env["OMP_NUM_THREADS"] == "1"


# -- report / cache files ----------------------------------------------------

def test_stale_report_removed(tmp_path):
    p = gp.raw_report_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    assert gp.clear_stale_report(tmp_path) is True
    assert not p.exists()
    assert gp.clear_stale_report(tmp_path) is False  # idempotent when already gone


def test_describe_raw_report_missing_empty_present(tmp_path):
    assert gp.describe_raw_report(tmp_path) == "missing"
    p = gp.raw_report_path(tmp_path)
    p.parent.mkdir(parents=True)
    p.write_text("")
    assert gp.describe_raw_report(tmp_path) == "empty"
    p.write_text('{"gremlins": []}')
    assert gp.describe_raw_report(tmp_path) == "present"


def test_cache_dir_helpers(tmp_path):
    assert gp.clear_cache_dir(tmp_path) is False  # nothing to clear
    (gp.cache_dir(tmp_path) / "inner").mkdir(parents=True)
    (gp.cache_dir(tmp_path) / "x").write_text("1")
    assert gp.clear_cache_dir(tmp_path) is True
    assert not gp.cache_dir(tmp_path).exists()


# -- outer cache-invalidation fingerprint ------------------------------------

def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _build_fp_tree(root: Path) -> None:
    """A minimal repo-shaped tree carrying every input class the fingerprint now
    covers. The digest hashes ALL tracked inputs (``git ls-files``), so this
    ``git init``-ializes ``root`` and force-stages every file: only a TRACKED edit
    moves the namespace. Includes the classes the old narrow hand list missed --
    an imported production module under ``tools/`` (a selected test imports
    ``tools.phase5_datasets``) and an experiment ``run.py`` fixture -- alongside
    the shared test helpers/fixtures (tests/conftest.py, tests/helpers.py) that
    pytest-gremlins' own keys cannot see, gate/check helpers, the dependency locks
    and the mutation config. Each test changes exactly one class and watches the
    resulting namespace."""
    for d in ("engine", "tests", "checks", "tools",
              "experiments/EXP-169_menu7prime_confirmation"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "engine" / "core.py").write_text("def f():\n    return 1\n")
    (root / "engine" / "notes.md").write_text("# prose only\n")
    (root / "tests" / "test_core.py").write_text("def test_ok():\n    assert f() == 1\n")
    (root / "tests" / "conftest.py").write_text("# shared fixtures\n")
    (root / "tests" / "helpers.py").write_text("SHARED = 'v1'\n")
    (root / "checks" / "gate.py").write_text("LIMIT = 1\n")
    (root / "tools" / "phase5_datasets.py").write_text("DATA = 'v1'\n")
    (root / "experiments" / "EXP-169_menu7prime_confirmation" / "run.py").write_text("SEED = 1\n")
    (root / "requirements.txt").write_text("numpy==2.0\n")
    (root / "requirements-dev.txt").write_text("pytest-gremlins==1.9.0\n")
    (root / "tools" / "mutation_pilot.toml").write_text("[defaults]\ndeselect = []\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A", "-f")  # -f: stage everything present, ignore rules aside


def test_fingerprint_is_stable_when_inputs_unchanged(tmp_path):
    # incremental reuse: the identical tree yields the identical namespace.
    _build_fp_tree(tmp_path)
    assert gp.cache_fingerprint("toy", tmp_path) == gp.cache_fingerprint("toy", tmp_path)


def test_fingerprint_moves_on_shared_test_helper(tmp_path):
    # a helper-only push (no source or test change) must NOT reuse a stale outcome.
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "tests" / "helpers.py").write_text("SHARED = 'v2'\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_moves_on_conftest_fixture(tmp_path):
    # conftest.py is invisible to pytest-gremlins' own keys; the outer digest sees it.
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "tests" / "conftest.py").write_text("# shared fixtures\nseed = 2\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_moves_on_imported_production_helper(tmp_path):
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "engine" / "core.py").write_text("def f():\n    return 2\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_moves_on_dependency_lock(tmp_path):
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "requirements-dev.txt").write_text("pytest-gremlins==1.9.1\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_moves_on_mutation_config(tmp_path):
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "tools" / "mutation_pilot.toml").write_text('[defaults]\ndeselect = ["x"]\n')
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_moves_on_imported_tools_module(tmp_path):
    # A selected test imports tools.phase5_datasets (and the other tools modules);
    # the old narrow hand list named only the mutation tools, so editing an
    # imported one reused a stale mutant outcome. All-tracked sees every tools/
    # file, so the digest moves.
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "tools" / "phase5_datasets.py").write_text("DATA = 'v2'\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_moves_on_experiment_run_py(tmp_path):
    # A selected test reads experiments/EXP-169_menu7prime_confirmation/run.py; an
    # edit there is a real input change, so the namespace must move.
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    run_py = tmp_path / "experiments" / "EXP-169_menu7prime_confirmation" / "run.py"
    run_py.write_text("SEED = 2\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_ignores_untracked_generated_cache_and_logs(tmp_path):
    # .gremlins_cache / .oc_logs are generated and never tracked, so git ls-files
    # cannot list them: they must not move the namespace (a restore of the
    # plugin's own cache must not perturb its key).
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / ".gremlins_cache").mkdir()
    (tmp_path / ".gremlins_cache" / "gremlin.bin").write_bytes(b"mutant cache")
    (tmp_path / ".oc_logs").mkdir()
    (tmp_path / ".oc_logs" / "oc_check_report.json").write_text('{"verdict": "x"}')
    assert gp.cache_fingerprint("toy", tmp_path) == base


def test_fingerprint_includes_documentation_edits(tmp_path):
    # Conservative all-tracked semantics replace the old docs-skipping rule: a
    # tracked ``.md`` edit is hashed too, so the digest moves. A doc cannot move a
    # kill or a survive, so this deliberately over-invalidates -- the accepted cost
    # of a fingerprint that never misses a real input (a stale hit would publish a
    # wrong mutation outcome).
    _build_fp_tree(tmp_path)
    base = gp.cache_fingerprint("toy", tmp_path)
    (tmp_path / "engine" / "notes.md").write_text("# rewrote the prose\n")
    assert gp.cache_fingerprint("toy", tmp_path) != base


def test_fingerprint_separates_per_module_namespaces(tmp_path):
    # matrix keeps per-module cache separation even on the identical tree.
    _build_fp_tree(tmp_path)
    assert gp.cache_fingerprint("alpha", tmp_path) != gp.cache_fingerprint("beta", tmp_path)


# -- heartbeat -------------------------------------------------------------

def test_beat_line_labels_elapsed_seconds():
    assert gp.beat_line(125.7) == "[gremlin_pilot] still running... 125s elapsed"


def test_stream_forwards_child_output_and_returns_rc():
    out: list[str] = []
    proc = FakeProc(lines=["a\n", "b\n"], rc=7)
    rc = gp.stream_and_heartbeat(proc, interval=1000, write=out.append)
    assert rc == 7
    assert out == ["a\n", "b\n"]


# -- cmd_run wiring --------------------------------------------------------

def test_run_builds_command_env_and_cwd(monkeypatch, tmp_path, capsys):
    stub_partition(monkeypatch, targets=("engine/a.py", "engine/b.py"),
                   tests=("tests/test_a.py",))
    monkeypatch.setattr(gp, "is_ci", lambda env=None: False)
    calls = stub_popen(monkeypatch, rc=0)
    rc = gp.cmd_run(cfg(), run_args(nproc=8, cwd=tmp_path))
    assert rc == 0
    assert len(calls) == 1  # exactly one subprocess: no rerun
    cmd, kw = calls[0]
    assert "--gremlin-targets=engine/a.py,engine/b.py" in cmd
    assert "--gremlin-workers=2" in cmd          # local default 2
    assert "--gremlin-cache" in cmd and "--gremlin-report=json" in cmd
    assert "--deselect=tests/test_a.py::needs_data" in cmd
    assert cmd[-1] == "tests/test_a.py"
    assert "--gremlin-batch" not in cmd
    assert "no:xdist" not in cmd                 # the default argv must keep xdist loaded (CI run 36059422920)
    assert kw["cwd"] == str(tmp_path)
    assert kw["env"]["OMP_NUM_THREADS"] == "1"   # BLAS pool pinned
    text = capsys.readouterr().out
    assert "per-mutant timeout 30s" in text       # 1.9.0 constraint recorded
    assert "batch mode disabled" in text
    assert "|| true" not in text and "|| true" not in " ".join(cmd)


def test_run_ci_worker_cap_reaches_command(monkeypatch, tmp_path):
    stub_partition(monkeypatch)
    monkeypatch.setattr(gp, "is_ci", lambda env=None: True)
    calls = stub_popen(monkeypatch, rc=0)
    gp.cmd_run(cfg(), run_args(nproc=8, cwd=tmp_path))
    assert "--gremlin-workers=4" in calls[0][0]   # nproc 8 capped to 4 in CI


def test_run_propagates_nonzero_rc_without_rerunning(monkeypatch, tmp_path):
    stub_partition(monkeypatch)
    calls = stub_popen(monkeypatch, rc=1)
    assert gp.cmd_run(cfg(), run_args(cwd=tmp_path)) == 1
    assert len(calls) == 1


def test_run_deletes_stale_raw_report_even_incremental(monkeypatch, tmp_path):
    stub_partition(monkeypatch)
    raw = gp.raw_report_path(tmp_path)
    raw.parent.mkdir(parents=True)
    raw.write_text("stale")
    stub_popen(monkeypatch, rc=0)
    gp.cmd_run(cfg(), run_args(fresh=False, cwd=tmp_path))
    assert not raw.exists()  # cleared before the subprocess even starts


def test_run_fresh_clears_cache_incremental_keeps_it(monkeypatch, tmp_path):
    stub_partition(monkeypatch)
    (gp.cache_dir(tmp_path)).mkdir()
    stub_popen(monkeypatch, rc=0)
    gp.cmd_run(cfg(), run_args(fresh=True, cwd=tmp_path))
    assert not gp.cache_dir(tmp_path).exists()   # fresh cleared it
    (gp.cache_dir(tmp_path)).mkdir()
    stub_popen(monkeypatch, rc=0)
    gp.cmd_run(cfg(), run_args(fresh=False, cwd=tmp_path))
    assert gp.cache_dir(tmp_path).exists()       # incremental reused it


def test_run_keeps_gremlin_cache_flag_on_a_fresh_run(monkeypatch, tmp_path):
    stub_partition(monkeypatch)
    calls = stub_popen(monkeypatch, rc=0)
    gp.cmd_run(cfg(), run_args(fresh=True, cwd=tmp_path))
    assert "--gremlin-cache" in calls[0][0]      # still writes a fresh cache


def test_run_missing_raw_report_warns_but_keeps_rc(monkeypatch, tmp_path, capsys):
    stub_partition(monkeypatch)
    stub_popen(monkeypatch, rc=0)  # produces no report file
    rc = gp.cmd_run(cfg(), run_args(cwd=tmp_path))
    assert rc == 0                               # the run's rc is unchanged
    assert "raw report missing" in capsys.readouterr().out


def test_run_empty_raw_report_is_flagged(monkeypatch, tmp_path, capsys):
    stub_partition(monkeypatch)

    def make_empty():
        raw = gp.raw_report_path(tmp_path)
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text("")

    stub_popen(monkeypatch, rc=0, on_read=make_empty)
    gp.cmd_run(cfg(), run_args(cwd=tmp_path))
    assert "raw report empty" in capsys.readouterr().out


def test_run_present_raw_report_is_not_flagged(monkeypatch, tmp_path, capsys):
    stub_partition(monkeypatch)

    def make_present():
        raw = gp.raw_report_path(tmp_path)
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_text('{"gremlins": []}')

    stub_popen(monkeypatch, rc=0, on_read=make_present)
    gp.cmd_run(cfg(), run_args(cwd=tmp_path))
    assert "raw report" not in capsys.readouterr().out


def test_run_refuses_an_excluded_module(monkeypatch):
    monkeypatch.setattr(gp.pilot, "module_cfg", lambda cfg_, name: {"excluded": "legacy bridge"})
    with pytest.raises(SystemExit) as exc:
        gp.cmd_run(cfg(toy={"excluded": "legacy bridge"}), run_args())
    assert "excluded" in str(exc.value)


def test_run_refuses_module_with_nothing_to_run(monkeypatch):
    stub_partition(monkeypatch, targets=(), tests=())
    with pytest.raises(SystemExit):
        gp.cmd_run(cfg(), run_args(cwd=None))


# -- production pytest_args defaults: the shipped toml + the in-code fallback --

def test_shipped_toml_pytest_args_never_disable_xdist():
    # The command tests above pass an EXPLICIT corrected pytest_args and the
    # cmd_run tests MOCK the defaults, so neither protects the shipped config
    # itself. tools/mutation_pilot.toml [defaults] pytest_args IS production:
    # cmd_run reads it unchanged into the gremlins argv. An edit reintroducing
    # `-p no:xdist` there is the CI run 36059422920 bug (pytest-gremlins 1.9.0
    # implements xdist's hooks, so pluggy aborts collection) and must fail here.
    import mutation_pilot as pilot
    args = pilot.load_config()["defaults"]["pytest_args"]
    assert "no:xdist" not in args
    assert args == ["-p", "no:cacheprovider"]


def test_missing_pytest_args_fallback_never_disables_xdist(monkeypatch, tmp_path):
    # gremlin_pilot's in-code fallback -- cmd_run reads
    # defaults.get("pytest_args", [...]) -- is the other production default, and
    # every other cmd_run cfg carries the key, so the fallback branch was
    # unpinned. Run a defaults dict with NO pytest_args and pin what the real
    # argv gets: xdist stays loaded and the fallback is exactly
    # `-p no:cacheprovider`.
    stub_partition(monkeypatch)
    calls = stub_popen(monkeypatch, rc=0)
    bare = {"defaults": {}, "modules": {"toy": {"why": "x"}}}
    assert gp.cmd_run(bare, run_args(cwd=tmp_path)) == 0
    cmd = calls[0][0]
    assert "no:xdist" not in cmd
    assert cmd[cmd.index("-p") + 1] == "no:cacheprovider"


# -- matrix selection reuses the toml partition ----------------------------

def test_matrix_matches_enabled_modules_and_rejects_excluded():
    import mutation_pilot as pilot
    real = pilot.load_config()
    enabled = pilot.enabled_modules(real)
    assert gp.select_modules(real) == enabled
    assert gp.select_modules(real, f"{enabled[-1]}, {enabled[0]}") == [enabled[0], enabled[-1]]
    excluded = next(name for name, mod in real["modules"].items() if mod.get("excluded"))
    with pytest.raises(SystemExit):
        gp.select_modules(real, excluded)


def test_cmd_matrix_prints_the_json_list(monkeypatch, capsys):
    import mutation_pilot as pilot
    real = pilot.load_config()
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    assert gp.cmd_matrix(real, types.SimpleNamespace(only="")) == 0
    import json
    assert json.loads(capsys.readouterr().out) == ["foundation", "canonical"]


# -- --changed-files: PR module selection delegates to mutation_pilot --------
#
# gremlin_pilot reuses mutation_pilot's changed_modules/read_changed_files
# unchanged (same shared-input rule governs both backends' matrices, exactly
# as it already reuses the module partition). These tests pin the WIRING --
# that select_modules/cmd_matrix pass the flag through -- not the selection
# rule itself, which tests/test_mutation_ci.py already covers in full.

def test_select_modules_without_changed_files_is_unaffected(monkeypatch):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    assert gp.select_modules(cfg()) == ["foundation", "canonical"]
    assert gp.select_modules(cfg(), "", "") == ["foundation", "canonical"]


def test_select_modules_changed_files_delegates_to_pilot_changed_modules(monkeypatch):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    seen = {}

    def fake_changed_modules(cfg_, names, changed):
        seen["names"], seen["changed"] = names, changed
        return ["foundation"]

    monkeypatch.setattr(gp.pilot, "read_changed_files", lambda p: ["engine/x.py"] if p else [])
    monkeypatch.setattr(gp.pilot, "changed_modules", fake_changed_modules)
    result = gp.select_modules(cfg(), "", "some/path.txt")
    assert result == ["foundation"]
    assert seen == {"names": ["foundation", "canonical"], "changed": ["engine/x.py"]}


def test_select_modules_changed_files_applies_after_only_filtering(monkeypatch):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    monkeypatch.setattr(gp.pilot, "read_changed_files", lambda p: ["x"])
    # changed_modules never sees "canonical": --only already dropped it
    monkeypatch.setattr(gp.pilot, "changed_modules",
                        lambda cfg_, names, changed: names)
    assert gp.select_modules(cfg(), "foundation", "some/path.txt") == ["foundation"]


def test_cmd_matrix_passes_changed_files_through(monkeypatch, capsys):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    monkeypatch.setattr(gp.pilot, "read_changed_files", lambda p: ["engine/x.py"] if p else [])
    monkeypatch.setattr(gp.pilot, "changed_modules", lambda cfg_, names, changed: ["foundation"])
    args = types.SimpleNamespace(only="", changed_files="some/path.txt")
    assert gp.cmd_matrix(cfg(), args) == 0
    import json
    assert json.loads(capsys.readouterr().out) == ["foundation"]


def test_cmd_matrix_missing_changed_files_attr_keeps_old_behavior(monkeypatch, capsys):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    assert gp.cmd_matrix(cfg(), types.SimpleNamespace(only="")) == 0
    import json
    assert json.loads(capsys.readouterr().out) == ["foundation", "canonical"]


def test_gremlin_shares_the_mutant_toml_module_map():
    # One partition governs both backends: the gremlin config IS mutation_pilot's.
    import mutation_pilot as pilot
    assert gp.pilot.load_config is pilot.load_config
    assert "foundation" in gp.pilot.load_config()["modules"]


# -- --only validation: a comma-only subset is never an empty matrix --------

def test_only_comma_is_rejected_not_treated_as_an_empty_matrix():
    # A dispatched empty input renders as "" -> "every enabled module"; a literal
    # comma names NOTHING and must error, never quietly reduce the run to zero.
    import mutation_pilot as pilot
    real = pilot.load_config()
    with pytest.raises(SystemExit):
        gp.select_modules(real, ",")


def test_only_blank_still_means_every_enabled_module():
    import mutation_pilot as pilot
    real = pilot.load_config()
    assert gp.select_modules(real, "") == pilot.enabled_modules(real)


def test_cmd_matrix_refuses_comma_only(monkeypatch):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["foundation", "canonical"])
    with pytest.raises(SystemExit):
        gp.cmd_matrix(cfg(), types.SimpleNamespace(only=","))


# -- fingerprint command: one validated name per namespace ------------------

def test_cmd_fingerprint_prints_one_module_digest(monkeypatch, tmp_path, capsys):
    _build_fp_tree(tmp_path)
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["toy"])
    rc = gp.cmd_fingerprint(cfg(), types.SimpleNamespace(module="toy", cwd=tmp_path))
    assert rc == 0
    assert capsys.readouterr().out.strip() == gp.cache_fingerprint("toy", tmp_path)


def test_cmd_fingerprint_refuses_a_comma_list(monkeypatch, tmp_path):
    # one name per namespace, so a renamed module cannot inherit another's cache
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["toy", "other"])
    with pytest.raises(SystemExit):
        gp.cmd_fingerprint(cfg(), types.SimpleNamespace(module="toy,other", cwd=tmp_path))


def test_cmd_fingerprint_refuses_a_blank_module(tmp_path):
    with pytest.raises(SystemExit):
        gp.cmd_fingerprint(cfg(), types.SimpleNamespace(module="", cwd=tmp_path))


def test_cmd_fingerprint_refuses_an_unknown_module(monkeypatch, tmp_path):
    monkeypatch.setattr(gp.pilot, "enabled_modules", lambda cfg_: ["toy"])
    with pytest.raises(SystemExit):
        gp.cmd_fingerprint(cfg(), types.SimpleNamespace(module="ghost", cwd=tmp_path))


# -- candidate workflow shape ------------------------------------------------

def _step(job, predicate):
    return next(s for s in JOBS[job]["steps"] if predicate(s))


def test_candidate_triggers_and_new_concurrency_group():
    on = WORKFLOW.get("on", WORKFLOW.get(True))
    assert on["push"]["branches"] == ["main"]
    assert on["schedule"] and "cron" in on["schedule"][0]
    assert set(on["workflow_dispatch"]["inputs"]) == {"fresh", "modules"}
    # A new group so an old in-flight mutmut workflow cannot block the first gremlins run.
    assert WORKFLOW["concurrency"]["group"] == "mutation-gremlins-${{ github.ref }}"
    assert WORKFLOW["concurrency"]["cancel-in-progress"] is False
    assert WORKFLOW["permissions"] == {"contents": "read"}


def test_candidate_pins_python_314_and_reads_gremlins_backend():
    assert WORKFLOW["env"]["PYTHON_VERSION"] == "3.14.4"
    assert "pytest-gremlins==1.9.0" in (ROOT / "requirements-dev.txt").read_text()
    plan = _step("plan", lambda s: s.get("id") == "plan")["run"]
    assert "pytest-gremlins==" in plan
    assert "tools/gremlin_pilot.py matrix" in plan
    assert "tools/mutation_pilot.py matrix" not in plan   # gremlins, not mutmut
    assert set(JOBS["plan"]["outputs"]) == {"modules", "mode", "gremlins"}


def test_candidate_run_invokes_gremlin_pilot_and_fresh_on_full():
    run = _step("mutate", lambda s: s.get("id") == "run")["run"]
    assert "tools/gremlin_pilot.py run" in run
    assert "--fresh" in run and 'MODE" = "full"' in run
    assert "set +e" in run and "gremlin-rc" in run       # rc captured, scores never silence it


def test_candidate_cache_pins_backend_and_is_incremental_only():
    assert WORKFLOW["env"]["GREMLINS_CACHE"] == ".gremlins_cache"
    key = _step("mutate", lambda s: s.get("id") == "key")["run"]
    for part in ("needs.plan.outputs.gremlins", "steps.py.outputs.python-version",
                 "hashFiles('tools/mutation_pilot.toml')", "matrix.module"):
        assert part in key, part
    # The outer invalidation fingerprint is in the namespace too, and it is
    # assigned under set -e rather than interpolated into the echo (where a
    # failed substitution would be masked by echo's own exit status).
    assert "set -euo pipefail" in key
    assert 'fp=$(python3 tools/gremlin_pilot.py fingerprint "$MODULE")' in key
    prefix_line = next(ln.strip() for ln in key.splitlines() if "prefix=" in ln)
    assert "${fp}" in prefix_line            # the namespace carries the digest
    assert "$(" not in prefix_line           # ... as a variable, never a substitution
    restore = _step("mutate", lambda s: s.get("uses", "").startswith("actions/cache/restore"))
    save = _step("mutate", lambda s: s.get("uses", "").startswith("actions/cache/save"))
    prefix = "${{ steps.key.outputs.prefix }}"
    assert restore["with"]["restore-keys"] == prefix
    assert restore["with"]["key"] == save["with"]["key"]
    assert restore["if"] == "needs.plan.outputs.mode == 'incremental'"  # full: no restore
    assert "always()" in save["if"]
    assert restore["with"]["path"] == "${{ env.GREMLINS_CACHE }}"
    assert save["with"]["path"] == "${{ env.GREMLINS_CACHE }}"


# -- plan step: a failing matrix command must not be masked ------------------

def test_candidate_plan_assigns_the_matrix_before_echoing_it():
    plan = _step("plan", lambda s: s.get("id") == "plan")["run"]
    lines = [ln.strip() for ln in plan.splitlines()]
    assert lines[0] == "set -euo pipefail"
    assign = 'modules=$(python3 tools/gremlin_pilot.py matrix --only "$ONLY")'
    assert assign in lines
    assert 'echo "modules=$modules"' in lines
    assert 'echo "modules=$(python3' not in plan          # never interpolate the command
    assert lines.index(assign) < lines.index('echo "modules=$modules"')


# -- report job: a nonzero merge must still deliver the consolidated report --

def _report_steps():
    merge = _step("report", lambda s: s.get("name") == "Merge module reports")
    upload = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/upload-artifact"))
    gate = _step("report", lambda s: s.get("name", "").startswith("Fail only on a merge error"))
    return merge["run"], upload, gate["run"]


def test_candidate_report_captures_merge_rc_instead_of_skipping_the_upload():
    merge, upload, gate = _report_steps()
    assert "tools/gremlin_results.py merge" in merge
    assert "set +e" in merge and "gremlin-merge-rc" in merge   # rc captured, not swallowed
    assert "|| true" not in merge                              # never silenced either
    assert "[ -f merged/summary.md ]" in merge                 # summary appended IF generated
    assert 'cat merged/summary.md >> "$GITHUB_STEP_SUMMARY"' in merge
    # A merge that exits 0 and writes no summary consolidated nothing: that is a
    # failure, so the rc is forced nonzero.
    assert 'echo 1 > "$RUNNER_TEMP/gremlin-merge-rc"' in merge
    assert upload["if"] == "always()"
    assert upload["with"]["name"] == "mutation-report"
    assert 'rc" != 0' in gate
    assert "score" not in gate                                 # scores still never gate
    steps = JOBS["report"]["steps"]
    at = {label: next(i for i, s in enumerate(steps) if pred(s))
          for label, pred in (("merge", lambda s: s.get("name") == "Merge module reports"),
                              ("upload", lambda s: str(s.get("uses", "")).startswith("actions/upload")),
                              ("gate", lambda s: str(s.get("name", "")).startswith("Fail only")))}
    assert at["merge"] < at["upload"] < at["gate"]   # artifact + summary land, THEN the fail


def test_candidate_report_merges_the_expected_set_even_with_no_module_dirs():
    """Zero module directories used to be a silent pass (``echo 0`` + ``exit 0``):
    a run in which every matrix job died before its upload published nothing and
    the job went green. merge is now always called, with the plan's module list,
    so the empty/partial set becomes an incomplete diagnostic artifact and a
    nonzero rc the gate step fails on."""
    merge, upload, gate = _report_steps()
    assert 'dirs=(modules/mutation-module-*)' in merge
    assert "no module reports to merge" not in merge
    assert 'echo 0 > "$RUNNER_TEMP/gremlin-merge-rc"' not in merge   # nothing merged != pass
    assert "exit 0" not in merge                                     # no early pass path
    assert '--expected-modules "$EXPECTED_MODULES"' in merge
    assert '"${dirs[@]}"' in merge                                   # same call, empty or not
    assert upload["if"] == "always()"
    assert 'rc" != 0' in gate



def test_candidate_calls_results_adapter_export_and_merge():
    export = _step("mutate", lambda s: s.get("name") == "Export report")["run"]
    assert "tools/gremlin_results.py export" in export
    report = _step("report", lambda s: "run" in s and "gremlin_results" in s["run"])["run"]
    assert "tools/gremlin_results.py merge" in report


def test_candidate_uploads_raw_report_even_after_failure():
    raw = _step("mutate", lambda s: s.get("name") == "Upload raw gremlins report")
    assert raw["if"] == "always()"
    assert raw["with"]["path"] == "coverage/gremlins/gremlins.json"
    assert raw["with"]["retention-days"] == 90
    assert raw["with"]["if-no-files-found"] == "warn"


def test_candidate_artifact_names_and_90_day_retention_preserved():
    uploads = [s for j in JOBS.values() for s in j["steps"]
               if str(s.get("uses", "")).startswith("actions/upload-artifact")]
    assert {u["with"]["retention-days"] for u in uploads} == {90}
    assert {u["with"]["name"] for u in uploads} == {
        "gremlin-raw-${{ matrix.module }}", "mutation-module-${{ matrix.module }}",
        "mutation-report"}
    download = _step("report", lambda s: str(s.get("uses", "")).startswith("actions/download"))
    assert download["with"]["pattern"] == "mutation-module-*"


def test_candidate_is_report_only_and_has_no_broad_true():
    gate = _step("mutate", lambda s: s.get("name", "").startswith("Fail only on a tool error"))
    assert 'rc" != 0' in gate["run"]
    assert "score" not in gate["run"]                   # scores never gate
    assert "always()" in JOBS["report"]["if"]
    text = CANDIDATE.read_text()
    assert "|| true" not in text                        # no swallowed failures anywhere
    import tomllib
    assert tomllib.loads((ROOT / "tools" / "mutation_pilot.toml").read_text())  # toml still parses
