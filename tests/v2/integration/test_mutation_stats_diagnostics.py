"""Stats failures retain bounded evidence without weakening the mutation gate.

# packages: ops
"""
import io
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


def test_plugin_registers_only_with_a_stats_destination(monkeypatch, tmp_path):
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    monkeypatch.delenv("MUTATION_PILOT_STATS_DIAGNOSTIC", raising=False)
    calls = []
    config = SimpleNamespace(pluginmanager=SimpleNamespace(register=lambda *args: calls.append(args)))
    diagnostics.pytest_configure(config)
    assert calls == []
    destination = tmp_path / "stats-failure.txt"
    monkeypatch.setenv("MUTATION_PILOT_STATS_DIAGNOSTIC", str(destination))
    diagnostics.pytest_configure(config)
    assert len(calls) == 1
    plugin, name = calls[0]
    assert isinstance(plugin, diagnostics.StatsDiagnostics)
    assert plugin.destination == destination and name == "stats-diagnostics"


@pytest.mark.parametrize("hook", ["pytest_collectreport", "pytest_runtest_logreport"])
def test_nonfailed_reports_leave_no_diagnostic(tmp_path, hook):
    path = tmp_path / "stats-failure.txt"
    plugin = diagnostics.StatsDiagnostics(path)
    getattr(plugin, hook)(SimpleNamespace(failed=False))
    assert not path.exists()


@pytest.mark.parametrize("phase", ["collection", "setup", "call", "teardown"])
def test_failed_reports_retain_traceback_and_both_captured_streams(tmp_path, phase):
    path = tmp_path / "stats-failure.txt"
    plugin = diagnostics.StatsDiagnostics(path)
    report = SimpleNamespace(failed=True, nodeid="tests/test_bad.py::test_bad", when=phase,
        longreprtext="ValueError: original failure",
        sections=[("Captured stdout " + phase, "stdout detail"),
                  ("Captured stderr " + phase, "stderr detail")])
    if phase == "collection":
        plugin.pytest_collectreport(report)
        heading = "COLLECTION FAILED tests/test_bad.py::test_bad"
    else:
        plugin.pytest_runtest_logreport(report)
        heading = f"FAILED tests/test_bad.py::test_bad [{phase}]"
    retained = path.read_text()
    assert heading in retained
    assert "ValueError: original failure" in retained
    assert f"--- Captured stdout {phase} ---\nstdout detail" in retained
    assert f"--- Captured stderr {phase} ---\nstderr detail" in retained


def test_internal_error_is_retained_on_a_fresh_sidecar(tmp_path):
    path = tmp_path / "stats-failure.txt"
    diagnostics.StatsDiagnostics(path).pytest_internalerror("RuntimeError: plugin exploded")
    assert path.read_text() == "\nPYTEST INTERNAL ERROR\nRuntimeError: plugin exploded\n"


def test_failure_reports_are_bounded_and_passed_reports_are_not_retained(tmp_path):
    path = tmp_path / "stats-failure.txt"
    plugin = diagnostics.StatsDiagnostics(path)
    plugin.pytest_runtest_logreport(SimpleNamespace(failed=False))
    assert not path.exists()
    plugin.pytest_collectreport(SimpleNamespace(failed=True, nodeid="tests/test_bad.py",
        longreprtext="SyntaxError: broken fixture", sections=[]))
    plugin.pytest_runtest_logreport(SimpleNamespace(failed=True, nodeid="tests/test_a.py::test_a",
        when="setup", longreprtext="fixture failed\n" + "é" * 65536,
        sections=[("Captured stdout call", "typed refusal detail\n")]))
    result = path.read_bytes()
    assert diagnostics.LIMIT_BYTES == 65536  # independent public 64-KiB contract
    assert len(result) == 65536
    assert b"COLLECTION FAILED tests/test_bad.py\nSyntaxError" in result
    assert b"FAILED tests/test_a.py::test_a [setup]\nfixture failed" in result
    assert diagnostics.TRUNCATED in result
    assert result.endswith(b"--- Captured stdout call ---\ntyped refusal detail\n")
    plugin.pytest_internalerror("must not grow an already full diagnostic")
    assert path.read_bytes() == result


@pytest.mark.parametrize("remaining", range(1, 31))
def test_partial_truncation_marker_never_appends_a_zero_length_tail(tmp_path, remaining):
    path = tmp_path / "stats-failure.txt"
    prefix = b"x" * (65536 - remaining)
    path.write_bytes(prefix)
    plugin = diagnostics.StatsDiagnostics(path)
    plugin.pytest_internalerror("oversized final report" * 100)
    retained = path.read_bytes()
    assert len(retained) == 65536
    assert retained == prefix + b"\n[stats diagnostic truncated]\n"[:remaining]
    plugin.pytest_internalerror("must not grow an already full diagnostic")
    assert path.read_bytes() == retained


def test_failed_write_warns_once_when_original_stderr_is_available(tmp_path, monkeypatch):
    destination = tmp_path / "directory-not-sidecar"
    destination.mkdir()
    warning = io.StringIO()
    plugin = diagnostics.StatsDiagnostics(destination)
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "__stderr__", warning)
        plugin.pytest_internalerror("original pytest error")
        plugin.pytest_internalerror("unavailable destination must not be retried")
    message = warning.getvalue()
    assert message.startswith("[mutation_pilot] stats diagnostic unavailable: ")
    assert str(destination) in message
    assert message.count("stats diagnostic unavailable:") == 1
    assert plugin.unavailable is True
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("error", [OSError, ValueError])
def test_diagnostic_write_and_warning_failures_do_not_replace_pytest_outcome(tmp_path, monkeypatch,
                                                                         error):
    destination = tmp_path / "directory-not-sidecar"
    destination.mkdir()

    def unavailable(text):
        raise error("original stderr unavailable")

    plugin = diagnostics.StatsDiagnostics(destination)
    with monkeypatch.context() as scoped:
        scoped.setattr(sys, "__stderr__", SimpleNamespace(write=unavailable))
        plugin.pytest_internalerror("original pytest error")
        plugin.pytest_internalerror("do not retry unavailable destination")
    assert plugin.unavailable is True
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize("phase,writable", [("call", True), ("call", False), ("collection", True)])
def test_real_forked_stats_failure_retains_traceback_without_debug_or_rerun(tmp_path, phase, writable):
    # A tiny real mutmut run stops at its intentional stats assertion: no
    # mutants are executed and no repository test/data fixture is loaded.
    work = tmp_path / "work"
    for directory in ("engine", "tests", "tools"):
        (work / directory).mkdir(parents=True)
    (work / "engine/__init__.py").touch()
    (work / "engine/toy.py").write_text("def add(a, b):\n    return a + b\n")
    (work / "tools/mutation_stats_diagnostics.py").write_bytes(
        (ROOT / "tools/mutation_stats_diagnostics.py").read_bytes())
    source = (
        "from pathlib import Path\nfrom engine.toy import add\n"
        "def test_failure():\n"
        "    with Path('../executions.txt').open('a') as out: out.write('call\\n')\n"
        "    print('typed refusal captured stdout sentinel')\n"
        "    assert add(2, 3) == 6, 'intentional stats diagnostic sentinel'\n")
    if phase == "collection":
        source = ("import sys\nfrom pathlib import Path\nfrom engine.toy import add\n"
                  "with Path('../executions.txt').open('a') as out: out.write('collection\\n')\n"
                  "print('import-time stdout sentinel')\n"
                  "print('import-time stderr sentinel', file=sys.stderr)\n"
                  "raise ValueError('intentional collection diagnostic sentinel')\n")
    (work / "tests/test_toy.py").write_text(source)
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
    pytest_exit = 2 if phase == "collection" else 1
    assert f"failed to collect stats. runner returned {pytest_exit}" in result.stdout
    if writable:
        retained = diagnostic.read_text()
        if phase == "call":
            assert "FAILED tests/test_toy.py::test_failure [call]" in retained
            assert "AssertionError" in retained and "intentional stats diagnostic sentinel" in retained
            assert "typed refusal captured stdout sentinel" in retained
        else:
            assert "COLLECTION FAILED tests/test_toy.py" in retained
            assert "ValueError" in retained and "intentional collection diagnostic sentinel" in retained
            assert "--- Captured stdout ---\nimport-time stdout sentinel" in retained
            assert "--- Captured stderr ---\nimport-time stderr sentinel" in retained
    else:
        assert "stats diagnostic unavailable:" in result.stderr
        assert "INTERNALERROR" not in result.stdout + result.stderr
    assert (work / "executions.txt").read_text() == phase + "\n"
    assert "debug = true" not in (work / "setup.cfg").read_text()


@pytest.mark.parametrize("source,allowed", [
    ("import pathlib\n", True), ("\n" * 209, True), ("\n" * 210, False),
    ("import pandas\n", False), ("import numpy\n", False), ("import engine\n", False),
    ("from pandas import DataFrame\n", False), ("from engine import data\n", False),
])
def test_source_policy_still_checks_authored_imports_and_size(tmp_path, monkeypatch, source, allowed):
    from tests.v2.foundation import test_v2_foundation_market_calendar as checks

    original = tmp_path / "original"
    relative = checks.MODULE_PATH.relative_to(checks.ROOT)
    target = original / relative
    target.parent.mkdir(parents=True)
    target.write_text(source)
    monkeypatch.setenv("MUTATION_PILOT_SOURCE_ROOT", str(original))
    if allowed:
        checks.test_module_imports_are_stdlib_only()
    else:
        with pytest.raises(AssertionError):
            checks.test_module_imports_are_stdlib_only()


@pytest.mark.parametrize("housekeeping", ["ok", "mkdir", "cp", "stale"])
def test_workflow_retains_fresh_stats_diagnostic_and_never_blocks_export(tmp_path, housekeeping):
    workflow = (ROOT / ".github/workflows/mutation-mutmut.yml").read_text()
    assert 'if [ "$rc" != 0 ]; then echo "::error::mutmut run exited $rc"; exit 1; fi' in workflow
    steps = yaml.safe_load(workflow)["jobs"]["mutate"]["steps"]
    script = next(step["run"] for step in steps if step.get("name") == "Export report")
    diagnostic = tmp_path / ".cache/investing-plan-mutation-pilot/toy/stats-failure.txt"
    diagnostic.parent.mkdir(parents=True)
    diagnostic.write_text("original stats failure detail")
    marker = tmp_path / "mutation-start"
    marker.write_text("100")
    os.utime(marker, (100, 100))
    os.utime(diagnostic, (50, 50) if housekeeping == "stale" else (200, 200))
    env = dict(os.environ, HOME=str(tmp_path), RUNNER_TEMP=str(tmp_path), MODULE="toy",
               MODE="incremental", PR_OPEN="true", BEFORE="")
    prelude = "python() { echo EXPORT_CALLED; return 7; };\n"
    if housekeeping in {"mkdir", "cp"}:
        prelude += f"{housekeeping}() {{ return 13; }};\n"
    result = subprocess.run(["bash", "-c", prelude + script],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=5)
    assert result.returncode == 7, result.stdout + result.stderr
    assert "EXPORT_CALLED" in result.stdout
    if housekeeping in {"mkdir", "cp"}:
        assert result.stderr == "::warning::Stats diagnostic copy failed; continuing result export\n"
    else:
        assert result.stderr == ""
    artifact = tmp_path / "report/toy/stats-failure.txt"
    if housekeeping == "ok":
        assert artifact.read_text() == "original stats failure detail"
    else:
        assert not artifact.exists()
    assert not (tmp_path / "report/toy/summary.json").exists()  # not a fabricated success


@pytest.mark.parametrize("extra,allowed", [
    ("import pathlib\n", True), ("import subprocess\n", False),
    ("import importlib\n", False), ("import runpy\n", False),
    ("__import__('subprocess')\n", False), ("eval('1')\n", False), ("exec('pass')\n", False),
    ("from . import leaf\n", False),
])
def test_provider_boundary_still_rejects_authored_dynamic_imports(tmp_path, monkeypatch, extra, allowed):
    from tests.v2.ops import test_provider_response_boundary as checks

    sources = {
        "engine/v2/ops/provider_response.py": (
            "from __future__ import annotations\nimport dataclasses\nimport typing\n"
            "from engine.v2.ops.errors import fail\n"),
        "engine/v2/ops/providers/__init__.py": (
            "from engine.v2.ops.providers.nasdaq_calendar import nasdaq_calendar_fetcher\n"
            "from engine.v2.ops.providers.orats_daily_market import orats_daily_market_fetcher\n"
            "from engine.v2.ops.providers.yfinance_edge import yfinance_earnings_fetcher, yfinance_history_fetcher\n"),
        "engine/v2/ops/providers/orats_daily_market.py": (
            "from engine.v2.foundation import canonical_json\nfrom engine.v2.ops.errors import fail\n"
            "from engine.v2.ops.provider_response import classify_response\n"),
        "engine/v2/ops/providers/nasdaq_calendar.py": "from engine.v2.ops.errors import fail\n",
        "engine/v2/ops/providers/yfinance_edge.py": extra,
        "tests/v2/ops/test_provider_response.py": (
            "from engine.v2.ops import provider_response\nfrom engine.v2.ops.errors import OpsError\n"),
    }
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setenv("MUTATION_PILOT_SOURCE_ROOT", str(tmp_path))
    if allowed:
        checks.test_provider_and_pure_test_direct_dependency_contracts()
    else:
        # Every dependency target remains valid, so only the forbidden source
        # construct can reject this otherwise-identical complete source tree.
        with pytest.raises(AssertionError):
            checks.test_provider_and_pure_test_direct_dependency_contracts()


def test_driver_resets_stale_diagnostic_and_passes_snapshot_root(monkeypatch, tmp_path):
    from tests.v2.integration.test_mutation_ci import run_mutation_command

    work = tmp_path / "work"
    work.mkdir()
    diagnostic = work / "stats-failure.txt"
    diagnostic.write_text("stale failure from an earlier run")
    monkeypatch.setenv("MUTATION_PILOT_SOURCE_ROOT", "/stale/checkout")
    monkeypatch.setenv("MUTATION_PILOT_STATS_DIAGNOSTIC", "/stale/report")
    rc, calls = run_mutation_command(monkeypatch, tmp_path, rc=1)
    assert rc == 1 and len(calls) == 1
    assert not diagnostic.exists()
    assert calls[0][1]["env"]["MUTATION_PILOT_SOURCE_ROOT"] == str(work)
    assert calls[0][1]["env"]["MUTATION_PILOT_STATS_DIAGNOSTIC"] == str(diagnostic)


@pytest.mark.parametrize("broken_warning", [False, OSError, ValueError])
def test_driver_reset_failure_still_runs_once_without_echoing_stale_evidence(monkeypatch, tmp_path,
                                                                         capsys, broken_warning):
    from tests.v2.integration.test_mutation_ci import run_mutation_command

    work = tmp_path / "work"
    work.mkdir()
    diagnostic = work / "stats-failure.txt"
    diagnostic.write_text("STALE_FAILURE_MUST_NOT_BE_ECHOED")
    real_unlink = Path.unlink

    def refuse_reset(path, *args, **kwargs):
        if path == diagnostic:
            raise PermissionError("cannot reset stats sidecar")
        return real_unlink(path, *args, **kwargs)

    def unavailable(text):
        raise broken_warning("stderr unavailable")

    warning = io.StringIO()
    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "unlink", refuse_reset)
        if broken_warning:
            scoped.setattr(sys, "stderr", SimpleNamespace(write=unavailable))
            scoped.setattr(sys, "__stderr__", SimpleNamespace(write=unavailable))
        else:
            scoped.setattr(sys, "__stderr__", warning)
        result, calls = run_mutation_command(scoped, tmp_path, rc=17)
    assert result == 17 and len(calls) == 1
    assert calls[0][1]["env"]["MUTATION_PILOT_STATS_DIAGNOSTIC"] == ""
    assert diagnostic.read_text() == "STALE_FAILURE_MUST_NOT_BE_ECHOED"
    captured = capsys.readouterr()
    assert "STALE_FAILURE_MUST_NOT_BE_ECHOED" not in captured.out + captured.err
    if not broken_warning:
        assert warning.getvalue() == (
            "[mutation_pilot] stats diagnostic unavailable: cannot reset stats sidecar\n")


@pytest.mark.parametrize("rc,payload", [(1, b"synthetic stats failure\n"),
    (17, b"x" + "\u00e9".encode() * 65536),
    (124, b"partial stats failure before budget stop\n"), (0, b"must not echo on success")],
    ids=["failure", "bounded-multibyte", "budget-stop", "success"])
def test_driver_echo_is_bounded_and_preserves_backend_result(monkeypatch, tmp_path, capsys,
                                                           rc, payload):
    from tests.v2.integration.test_mutation_ci import run_mutation_command

    def retain_diagnostic(cmd, cwd, env):
        Path(env["MUTATION_PILOT_STATS_DIAGNOSTIC"]).write_bytes(payload)

    monkeypatch.setenv("MASSIVE_API_KEY", "synthetic-env-value-must-not-be-logged")
    result, calls = run_mutation_command(monkeypatch, tmp_path, rc=rc, on_run=retain_diagnostic)
    echoed = capsys.readouterr().err
    assert result == rc and len(calls) == 1
    assert echoed == (payload[:65536].decode("utf-8", "ignore") if rc else "")
    assert len(echoed.encode()) <= 65536


@pytest.mark.parametrize("failure", ["read", OSError, ValueError])
def test_driver_echo_io_failure_preserves_backend_result(monkeypatch, tmp_path, failure):
    from tests.v2.integration.test_mutation_ci import run_mutation_command

    def retain_diagnostic(cmd, cwd, env):
        destination = Path(env["MUTATION_PILOT_STATS_DIAGNOSTIC"])
        if failure == "read":
            destination.mkdir()  # deterministic EISDIR, even for privileged CI users
        else:
            destination.write_text("synthetic failure")

    def unavailable(text):
        raise failure("stderr unavailable")

    with monkeypatch.context() as scoped:
        if failure != "read":
            scoped.setattr(sys, "stderr", SimpleNamespace(write=unavailable))
        result, calls = run_mutation_command(scoped, tmp_path, rc=1, on_run=retain_diagnostic)
    assert result == 1 and len(calls) == 1
