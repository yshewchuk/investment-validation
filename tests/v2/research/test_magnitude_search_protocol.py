"""Real pipe protocol, resource isolation and cleanup controls for the fixed child."""
import json
import signal
import subprocess
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_execution as execution
from tests.v2.research.test_magnitude_search_execution import LIMITS


def wire_child(monkeypatch, script):
    """Replace only the child program while retaining real pipes and lifecycle."""
    original = subprocess.Popen
    observed = []
    def launch(args, **kwargs):
        source = Path(args[3])
        request = json.loads(source.read_bytes())
        child = original([args[0], "-c", script], **kwargs)
        observed.append((child, source.parent, request))
        return child
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    return observed


def run_wire(progress=lambda *args: None):
    return execution.run_fitted_trial({"features": [], "training": [], "preflight": "not child data",
        "training_inputs": "not child data", "stored_inputs_id": "not child data"}, {},
        {"horizons": [1]}, {}, execution.execution_limits(LIMITS), progress)


@pytest.mark.parametrize("script, code, message", [
    ("print('x' * 513)", "RESOURCE_LIMIT_EXCEEDED", "search execution output exceeds its byte limit"),
    ("print('not JSON')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("print('[]')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("print('null')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("print('{\"event\":\"succeeded\",\"forecasts\":[]}\\n{\"event\":\"failed\"}')",
        "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("pass", "RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable"),
    ("print('unterminated', end='')", "RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable"),
    ("print('{\"event\":\"succeeded\",\"forecasts\":[]}'); raise SystemExit(3)",
        "RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable"),
])
def test_child_protocol_refuses_exactly_and_removes_scratch(monkeypatch, script, code, message):
    monkeypatch.setattr(execution, "MAX_BYTES", 512)
    observed = wire_child(monkeypatch, script)
    with pytest.raises(OpsError) as caught:
        run_wire()
    assert caught.value.problem == fail(code, message).problem
    child, scratch, _ = observed[0]
    assert child.poll() is not None
    assert not scratch.exists()


def test_child_receives_only_fit_inputs_and_success_removes_scratch(monkeypatch):
    observed = wire_child(monkeypatch, "print('{\"event\":\"succeeded\",\"forecasts\":[]}')")
    assert run_wire() == {"event": "succeeded", "forecasts": []}
    child, scratch, request = observed[0]
    assert request == {"inputs": {"features": [], "training": []}, "inner": {},
                       "plan": {"horizons": [1]}, "execution": {}}
    assert child.returncode == 0
    assert not scratch.exists()


def test_input_cap_precedes_launch(monkeypatch):
    monkeypatch.setattr(execution, "MAX_BYTES", 1)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("launched oversized input"))
    with pytest.raises(OpsError) as caught:
        run_wire()
    assert caught.value.problem == fail("RESOURCE_LIMIT_EXCEEDED", "search execution input exceeds its byte limit").problem


def test_fixed_worker_independently_rejects_grown_input(monkeypatch):
    original = subprocess.Popen
    observed = []
    def launch(args, **kwargs):
        source = Path(args[3])
        source.write_bytes(b" " * (execution.MAX_BYTES + 1))
        child = original(args, **kwargs)
        observed.append((child, source.parent))
        return child
    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    with pytest.raises(OpsError) as caught:
        run_wire()
    assert caught.value.problem == fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits").problem
    child, scratch = observed[0]
    assert child.poll() is not None
    assert not scratch.exists()


def test_parent_progress_failure_kills_and_reaps_child_before_removing_scratch(monkeypatch):
    observed = wire_child(monkeypatch, "import time; print('{\"event\":\"started\",\"fit\":{}}', flush=True); time.sleep(2)")
    error = fail("RESOURCE_UNAVAILABLE", "synthetic journal failure")
    def refuse(*args):
        raise error
    with pytest.raises(OpsError) as caught:
        run_wire(refuse)
    assert caught.value is error
    child, scratch, _ = observed[0]
    assert child.returncode == -signal.SIGKILL
    with pytest.raises(ChildProcessError):
        execution.os.waitpid(child.pid, execution.os.WNOHANG)
    assert not scratch.exists()


@pytest.mark.parametrize("available", [None, set()])
def test_affinity_unavailable_is_typed(monkeypatch, available):
    if available is None:
        monkeypatch.delattr(execution.os, "sched_getaffinity")
    else:
        monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: available)
    with pytest.raises(OpsError) as caught:
        execution.execution_limits(LIMITS)
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "search execution requires Linux resource containment").problem
