"""Stats failures retain bounded evidence without weakening the mutation gate.

# packages: ops
"""
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tools import mutation_pilot as pilot
from tools import mutation_stats_diagnostics as diagnostics

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("mode", ["", "fail", "mutant_generation", "engine.x__mutmut_1"])
def test_plugin_is_inert_outside_stats(monkeypatch, mode):
    monkeypatch.setenv("MUTANT_UNDER_TEST", mode)
    monkeypatch.setenv("MUTATION_PILOT_STATS_DIAGNOSTIC", "/not-opened")
    calls = []
    diagnostics.pytest_configure(SimpleNamespace(pluginmanager=SimpleNamespace(
        register=lambda *args: calls.append(args))))
    assert calls == []


def test_failure_reports_are_bounded_and_passed_reports_are_not_retained(tmp_path):
    path = tmp_path / "stats-failure.txt"
    plugin = diagnostics.StatsDiagnostics(path)
    plugin.pytest_runtest_logreport(SimpleNamespace(failed=False))
    assert not path.exists()
    plugin.pytest_collectreport(SimpleNamespace(failed=True, nodeid="tests/test_bad.py",
                                               longreprtext="SyntaxError: broken fixture"))
    plugin.pytest_runtest_logreport(SimpleNamespace(failed=True, nodeid="tests/test_a.py::test_a",
        when="setup", longreprtext="fixture failed\n" + "é" * diagnostics.LIMIT_BYTES,
        sections=[("Captured stdout call", "typed refusal detail\n")]))
    result = path.read_bytes()
    assert len(result) == diagnostics.LIMIT_BYTES
    assert b"COLLECTION FAILED tests/test_bad.py\nSyntaxError" in result
    assert b"FAILED tests/test_a.py::test_a [setup]\nfixture failed" in result
    assert diagnostics.TRUNCATED in result
    assert result.endswith(b"--- Captured stdout call ---\ntyped refusal detail\n")
    plugin.pytest_internalerror("must not grow an already full diagnostic")
    assert path.read_bytes() == result


@pytest.mark.parametrize("writable", [True, False])
def test_real_forked_stats_failure_retains_traceback_without_debug_or_rerun(tmp_path, writable):
    # A tiny real mutmut run stops at its intentional stats assertion: no
    # mutants are executed and no repository test/data fixture is loaded.
    work = tmp_path / "work"
    for directory in ("engine", "tests", "tools"):
        (work / directory).mkdir(parents=True)
    (work / "engine/__init__.py").touch()
    (work / "engine/toy.py").write_text("def add(a, b):\n    return a + b\n")
    (work / "tools/mutation_stats_diagnostics.py").write_bytes(
        (ROOT / "tools/mutation_stats_diagnostics.py").read_bytes())
    (work / "tests/test_toy.py").write_text(
        "from pathlib import Path\nfrom engine.toy import add\n"
        "def test_failure():\n"
        "    with Path('../executions.txt').open('a') as out: out.write('call\\n')\n"
        "    print('typed refusal captured stdout sentinel')\n"
        "    assert add(2, 3) == 6, 'intentional stats diagnostic sentinel'\n")
    defaults = {"pytest_args": ["-p", "no:cacheprovider"],
                "timeout_constant": 2, "timeout_multiplier": 5}
    (work / "setup.cfg").write_text(pilot.mutmut_config_text(
        defaults, ["engine/toy.py"], ["tests/test_toy.py"], ["tools"]))
    diagnostic = work / "stats-failure.txt"
    if not writable:
        diagnostic.mkdir()  # deterministic EISDIR, including privileged CI users
    env = dict(os.environ, MUTATION_PILOT_STATS_DIAGNOSTIC=str(diagnostic))
    result = subprocess.run([sys.executable, "-u", "-m", "mutmut", "run", "--max-children", "1"],
                            cwd=work, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "failed to collect stats. runner returned 1" in result.stdout
    if writable:
        retained = diagnostic.read_text()
        assert "FAILED tests/test_toy.py::test_failure [call]" in retained
        assert "AssertionError" in retained and "intentional stats diagnostic sentinel" in retained
        assert "typed refusal captured stdout sentinel" in retained
    else:
        assert "stats diagnostic unavailable:" in result.stderr
        assert "INTERNALERROR" not in result.stdout + result.stderr
    assert (work / "executions.txt").read_text() == "call\n"
    assert "debug = true" not in (work / "setup.cfg").read_text()


@pytest.mark.parametrize("source", ["import pathlib\n", "import pandas\n", "\n" * 210])
def test_source_policy_still_checks_authored_imports_and_size(tmp_path, monkeypatch, source):
    from tests.v2.foundation import test_v2_foundation_market_calendar as checks

    original = tmp_path / "original"
    relative = checks.MODULE_PATH.relative_to(checks.ROOT)
    target = original / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setenv("MUTATION_PILOT_SOURCE_ROOT", str(original))
    if source == "import pathlib\n":
        checks.test_module_imports_are_stdlib_only()
    else:
        with pytest.raises(AssertionError):
            checks.test_module_imports_are_stdlib_only()


def test_workflow_retains_stats_diagnostic_even_when_export_fails(tmp_path):
    workflow = (ROOT / ".github/workflows/mutation-mutmut.yml").read_text()
    assert 'if [ "$rc" != 0 ]; then echo "::error::mutmut run exited $rc"; exit 1; fi' in workflow
    steps = yaml.safe_load(workflow)["jobs"]["mutate"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == "Export report")
    diagnostic = tmp_path / ".cache/investing-plan-mutation-pilot/toy/stats-failure.txt"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("original stats failure detail")
    env = dict(os.environ, HOME=str(tmp_path), RUNNER_TEMP=str(tmp_path), MODULE="toy",
               MODE="incremental", PR_OPEN="true", BEFORE="")
    result = subprocess.run(["bash", "-c", "python() { return 7; };\n" + script],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 7, result.stdout + result.stderr
    assert (tmp_path / "report/toy/stats-failure.txt").read_text() == "original stats failure detail"
    assert not (tmp_path / "report/toy/summary.json").exists()  # not a fabricated success


def test_provider_boundary_still_rejects_authored_dynamic_imports(tmp_path, monkeypatch):
    from tests.v2.ops import test_provider_response_boundary as checks

    path = tmp_path / "source.py"
    monkeypatch.setenv("MUTATION_PILOT_SOURCE_ROOT", str(tmp_path))
    path.write_text("from engine.v2.ops.errors import fail\n")
    assert checks._source_imports("source.py") == (
        {"engine.v2.ops.errors"}, {"engine.v2.ops.errors.fail"})
    for source in ("import subprocess\n", "__import__('subprocess')\n", "from . import leaf\n"):
        path.write_text(source)
        with pytest.raises(AssertionError):
            checks._source_imports("source.py")


def test_driver_resets_stale_diagnostic_and_passes_snapshot_root(monkeypatch, tmp_path):
    from tests.v2.integration.test_mutation_ci import _cmd_run

    work = tmp_path / "work"
    work.mkdir()
    diagnostic = work / "stats-failure.txt"
    diagnostic.write_text("stale failure from an earlier run")
    monkeypatch.setenv("MUTATION_PILOT_SOURCE_ROOT", "/stale/checkout")
    monkeypatch.setenv("MUTATION_PILOT_STATS_DIAGNOSTIC", "/stale/report")
    rc, calls = _cmd_run(monkeypatch, tmp_path, rc=1)
    assert rc == 1 and len(calls) == 1
    assert not diagnostic.exists()
    assert calls[0][1]["env"]["MUTATION_PILOT_SOURCE_ROOT"] == str(work)
    assert calls[0][1]["env"]["MUTATION_PILOT_STATS_DIAGNOSTIC"] == str(diagnostic)
