"""Real pipe protocol, resource isolation and cleanup controls for the fixed child."""
import json
import signal
import subprocess
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_execution as execution
from tests.v2.research.test_magnitude_search_execution import LIMITS

STUB_SOURCE = {"manifest": {"stub": True}, "inputs_id": "sha256:" + "0" * 64, "store_root": "/nonexistent-store"}


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
    return execution.run_fitted_trial(STUB_SOURCE, {},
        {"horizons": [1]}, {}, execution.execution_limits(LIMITS), progress)


@pytest.mark.parametrize("script, code, message", [
    ("print('x' * 513)", "RESOURCE_LIMIT_EXCEEDED", "search execution output exceeds its byte limit"),
    ("print('not JSON')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("print('[]')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("print('null')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
    ("print('{}')", "EXPERIMENT_IDENTITY_CONFLICT", "search worker protocol differs"),
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


def test_child_receives_only_a_header_and_success_removes_scratch(monkeypatch):
    observed = wire_child(monkeypatch, "print('{\"event\":\"succeeded\",\"forecasts\":[]}')")
    assert run_wire() == {"event": "succeeded", "forecasts": []}
    child, scratch, request = observed[0]
    assert request == {**STUB_SOURCE, "inner": {}, "plan": {"horizons": [1]}, "execution": {}}
    assert child.returncode == 0
    assert not scratch.exists()


@pytest.mark.parametrize("event", ["started", "completed"])
@pytest.mark.parametrize("malformation", ["missing-fit", "extra-key"])
def test_malformed_progress_refuses_before_callback_and_removes_scratch(monkeypatch, event, malformation):
    message = {"event": event}
    if malformation == "extra-key":
        message.update(fit={}, extra=True)
    terminal = {"event": "succeeded", "forecasts": []}
    wire = json.dumps(message) + "\n" + json.dumps(terminal)
    observed = wire_child(monkeypatch, f"print({wire!r}, flush=True)")
    progress = []
    with pytest.raises(OpsError) as caught:
        run_wire(lambda *args: progress.append(args))
    assert caught.value.problem == fail(
        "RESOURCE_UNAVAILABLE", "search execution infrastructure is unavailable").problem
    assert progress == []
    child, scratch, _ = observed[0]
    assert child.poll() is not None
    assert not scratch.exists()


def test_post_eof_wait_timeout_is_typed_and_kills_and_reaps_child(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: now[0])
    popen = subprocess.Popen
    original_wait = popen.wait
    observed = wire_child(monkeypatch, "import os, time; os.close(1); time.sleep(60)")
    original_read = execution.os.read
    wait_timeouts = []

    def read(fd, count):
        chunk = original_read(fd, count)
        # Popen also reads its error pipe before the launched child is recorded.
        if observed and fd == observed[0][0].stdout.fileno() and not chunk:
            now[0] += LIMITS["wall_seconds"] + 1
        return chunk

    def wait(child, timeout=None):
        if timeout is not None:
            wait_timeouts.append(timeout)
        return original_wait(child, timeout=timeout)

    monkeypatch.setattr(execution.os, "read", read)
    monkeypatch.setattr(popen, "wait", wait)
    with pytest.raises(OpsError) as caught:
        run_wire()
    assert caught.value.problem == fail(
        "RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits").problem
    assert wait_timeouts == [0]
    child, scratch, _ = observed[0]
    assert child.returncode == -signal.SIGKILL
    with pytest.raises(ChildProcessError):
        execution.os.waitpid(child.pid, execution.os.WNOHANG)
    assert not scratch.exists()


@pytest.mark.parametrize("remaining, wait_advance", [(1, 0), (0, 0), (-1, 0), (1, 2)],
                         ids=["before", "at", "after-deadline", "expires-during-wait"])
def test_exited_child_still_requires_unexpired_deadline(monkeypatch, remaining, wait_advance):
    now = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: now[0])
    popen, original_read = subprocess.Popen, execution.os.read
    original_wait = popen.wait
    result = {"event": "succeeded", "forecasts": []}
    observed = wire_child(monkeypatch, f"print({json.dumps(result)!r}, flush=True)")
    wait_timeouts = []

    def read(fd, count):
        chunk = original_read(fd, count)
        if observed and fd == observed[0][0].stdout.fileno() and not chunk:
            # Establish the already-exited branch before advancing the clock.
            assert original_wait(observed[0][0], timeout=5) == 0
            now[0] += LIMITS["wall_seconds"] - remaining
        return chunk

    def wait(child, timeout=None):
        code = original_wait(child, timeout=timeout)
        if timeout is not None:
            wait_timeouts.append(timeout)
            now[0] += wait_advance
        return code

    monkeypatch.setattr(execution.os, "read", read)
    monkeypatch.setattr(popen, "wait", wait)
    if remaining > wait_advance:
        assert run_wire() == result
    else:
        with pytest.raises(OpsError) as caught:
            run_wire()
        assert caught.value.problem == fail(
            "RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits").problem
    assert wait_timeouts == [max(0, remaining)]
    child, scratch, _ = observed[0]
    assert child.returncode == 0
    assert not scratch.exists()


def test_header_is_not_capped_and_only_output_is(monkeypatch):
    """The removed input cap: a header above MAX_BYTES launches; the output cap still applies."""
    monkeypatch.setattr(execution, "MAX_BYTES", 600)
    big = {**STUB_SOURCE, "manifest": {"stub": "x" * 5000}}
    observed = wire_child(monkeypatch, "print('{\"event\":\"succeeded\",\"forecasts\":[]}')")
    result = execution.run_fitted_trial(big, {}, {"horizons": [1]}, {}, execution.execution_limits(LIMITS),
                                        lambda *args: None)
    assert result == {"event": "succeeded", "forecasts": []}
    assert observed[0][2]["manifest"] == {"stub": "x" * 5000}


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


def test_child_wall_lifetime_includes_parent_progress_acknowledgement_time(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(execution.time, "monotonic", lambda: now[0])
    observed = wire_child(monkeypatch, "import sys; "
        "print('{\"event\":\"started\",\"fit\":{}}', flush=True); sys.stdin.readline(); "
        "print('{\"event\":\"succeeded\",\"forecasts\":[]}', flush=True)")

    def commit(state, evidence):
        assert (state, evidence) == ("started", {})
        now[0] += LIMITS["wall_seconds"] + 1

    with pytest.raises(OpsError) as caught:
        run_wire(commit)
    assert caught.value.problem == fail("RESOURCE_LIMIT_EXCEEDED", "search execution exceeded its resource limits").problem
    child, scratch, _ = observed[0]
    assert child.poll() is not None
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


def test_parent_affinity_oserror_is_typed(monkeypatch):
    def affinity(pid):
        assert pid == 0
        raise OSError("synthetic affinity failure")

    monkeypatch.setattr(execution.os, "sched_getaffinity", affinity)
    with pytest.raises(OpsError) as caught:
        execution.execution_limits(LIMITS)
    assert caught.value.problem == fail(
        "RESOURCE_UNAVAILABLE", "search execution requires Linux resource containment").problem
