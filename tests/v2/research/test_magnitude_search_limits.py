"""Inclusive byte limits and observed containment in the real fixed fit child."""
import json
import os
import sys
from pathlib import Path

import pytest

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.executor import THREAD_VARIABLES
from experiments import magnitude_search_execution as execution
from tests.v2.research.magnitude_search_fit_source import fit_source

LIMITS = {"memory_bytes": 2 * 1024 ** 3, "cpu_seconds": 30, "wall_seconds": 60}
SMALL_CAP = 512
SUCCESS = {"event": "succeeded", "forecasts": []}
STATE = {}


@pytest.fixture(autouse=True)
def _store(tmp_path):
    STATE["root"] = tmp_path / "fit-store"


def request(policy=None):
    # No estimator is needed to observe the worker's containment and wire limits.
    return {"inputs": {"features": [], "training": [{"event_id": "train", "absolute_move_pct": 2}]},
            "inner": {"evaluation_event_ids": []}, "plan": {"horizons": []},
            "execution": {"features": []} if policy is None else policy}


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def run(policy=None, limits=None):
    payload = request(policy)
    source = fit_source(payload["inputs"], STATE["root"])
    return execution.run_fitted_trial(source, payload["inner"], payload["plan"],
        payload["execution"], execution.execution_limits(LIMITS) if limits is None else limits,
        lambda *args: pytest.fail("an empty horizon plan emitted fit progress"))


def observe_launch(monkeypatch, *, instrumentation=""):
    """Keep production launch/containment; optionally observe the worker's emitter."""
    original = execution.subprocess.Popen
    observed = []
    script = ("import sys; sys.argv = sys.argv[1:]; "
              "from experiments import magnitude_search_worker as worker; "
              f"exec({instrumentation!r}); worker.main()")

    def launch(args, **kwargs):
        assert args[1:3] == ["-m", "experiments.magnitude_search_worker"]
        source = Path(args[3])
        command = [args[0], "-c", script, *args[2:]] if instrumentation else args
        child = original(command, **kwargs)
        observed.append((child, source.parent))
        return child

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    return observed


def assert_cleaned(observed):
    assert len(observed) == 1
    child, scratch = observed[0]
    assert child.poll() is not None
    assert not scratch.exists()


@pytest.mark.parametrize("extra", [0, 1], ids=["exact-cap", "cap-plus-one"])
def test_parent_output_buffer_cap_counts_newline_inclusively(monkeypatch, extra):
    monkeypatch.setattr(execution, "MAX_BYTES", SMALL_CAP)
    result = {**SUCCESS, "padding": ""}
    result["padding"] = "x" * (SMALL_CAP + extra - len(json_bytes(result)) - 1)
    wire = json_bytes(result) + b"\n"
    assert len(wire) == SMALL_CAP + extra
    # One atomic write makes the boundary independent of pipe chunk scheduling.
    observed = observe_launch(monkeypatch, instrumentation=f"""
def emit(value):
    assert value == {SUCCESS!r}
    payload = {wire!r}
    descriptor = sys.stdout.fileno()
    assert len(payload) <= worker.os.fpathconf(descriptor, 'PC_PIPE_BUF')
    assert worker.os.write(descriptor, payload) == len(payload)
worker.emit = emit
""")
    if extra:
        with pytest.raises(OpsError) as caught:
            run()
        assert caught.value.problem == fail(
            "RESOURCE_LIMIT_EXCEEDED", "search execution output exceeds its byte limit").problem
    else:
        assert run() == result
    assert_cleaned(observed)


@pytest.mark.parametrize("value", [1, sys.maxsize - 1], ids=["minimum", "maximum"])
def test_all_resource_keys_accept_exact_representable_endpoints(monkeypatch, value):
    monkeypatch.setattr(execution.os, "sched_getaffinity", lambda pid: {7})
    limits = dict.fromkeys(LIMITS, value)
    expected = {**limits, "cpu_id": 7}
    assert execution.execution_limits(limits) == expected


@pytest.mark.parametrize("key", list(LIMITS))
@pytest.mark.parametrize("value", [0, sys.maxsize], ids=["below-minimum", "above-maximum"])
def test_each_resource_key_refuses_outside_representable_endpoints(monkeypatch, key, value):
    monkeypatch.setattr(execution.os, "sched_getaffinity",
                        lambda pid: pytest.fail("invalid resource contract read CPU affinity"))
    with pytest.raises(OpsError) as caught:
        execution.execution_limits({**LIMITS, key: value})
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search execution requires explicit positive integer resource limits").problem


@pytest.mark.parametrize("available", [{2, 5, 9}, {3, 6, 10}])
def test_cpu_choice_is_lowest_of_multiple_allowed_cpus(monkeypatch, available):
    def affinity(pid):
        assert pid == 0
        return available

    monkeypatch.setattr(execution.os, "sched_getaffinity", affinity)
    limits = dict(LIMITS)
    expected = {**limits, "cpu_id": sorted(available)[0]}
    assert execution.execution_limits(limits) == expected


@pytest.mark.parametrize("limits", [None, [], {}, LIMITS,
    {**LIMITS, "cpu_id": 0, "extra": 1}, {"cpu_id": 0},
    *({**LIMITS, "cpu_id": value} for value in (None, True, 1.0, "1", -1, 2 ** 31 - 1, sys.maxsize))])
def test_public_fit_helper_refuses_malformed_resolved_limits_before_effects(monkeypatch, limits):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid limits launched child"))
    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", lambda *a, **k: pytest.fail("invalid limits made scratch"))
    with pytest.raises(OpsError) as caught:
        execution.run_fitted_trial(None, None, None, None, limits, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search execution requires resolved limits and a valid Linux CPU").problem


@pytest.mark.parametrize("key", list(LIMITS))
@pytest.mark.parametrize("value", [True, "60", 1.5, 0, sys.maxsize])
def test_public_fit_helper_revalidates_each_resource_before_effects(monkeypatch, key, value):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("invalid limits launched child"))
    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", lambda *a, **k: pytest.fail("invalid limits made scratch"))
    with pytest.raises(OpsError) as caught:
        execution.run_fitted_trial(None, None, None, None, {**LIMITS, "cpu_id": 0, key: value}, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search execution requires explicit positive integer resource limits").problem


@pytest.mark.parametrize("cpu_id", [0, 2 ** 31 - 2])
@pytest.mark.parametrize("value", [1, sys.maxsize - 1])
def test_public_fit_helper_accepts_exact_resolved_endpoints_before_source_admission(monkeypatch, cpu_id, value):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("endpoint probe launched child"))
    with pytest.raises(OpsError) as caught:
        execution.run_fitted_trial(None, None, None, None, {**dict.fromkeys(LIMITS, value), "cpu_id": cpu_id}, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search execution requires an admitted cohort source").problem


@pytest.mark.parametrize("guard, error", [(no_fit_guard, RuntimeFitForbidden), (legacy_guard, LegacyRuntimeFitForbidden)])
def test_public_fit_helper_honors_both_no_fit_guards_before_inputs(monkeypatch, guard, error):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("guarded helper launched child"))
    with guard(), pytest.raises(error, match="run_fitted_trial"):
        execution.run_fitted_trial(None, None, None, None, {**LIMITS, "cpu_id": 0}, None)


def observe_worker_containment(monkeypatch, tmp_path, cpu_id):
    """Observe the real child while retaining launch and cleanup assertions."""
    allowed = os.sched_getaffinity(0)
    resolved = execution.execution_limits(LIMITS)
    assert resolved["cpu_id"] == min(allowed)
    limits = {**resolved, "cpu_id": cpu_id}
    expected = dict(limits)
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "23")
    observation = tmp_path / "worker-containment.json"
    observed = observe_launch(monkeypatch, instrumentation=f"""
original_emit = worker.emit
def emit(value):
    if value['event'] == 'succeeded':
        from pathlib import Path
        Path({str(observation)!r}).write_text(worker.json.dumps({{
            'affinity': sorted(worker.os.sched_getaffinity(0)),
            'threads': {{name: worker.os.environ.get(name) for name in {THREAD_VARIABLES!r}}},
            'memory': worker.resource.getrlimit(worker.resource.RLIMIT_AS),
            'cpu': worker.resource.getrlimit(worker.resource.RLIMIT_CPU),
        }}))
    original_emit(value)
worker.emit = emit
""")
    assert run(limits=limits) == SUCCESS
    assert os.sched_getaffinity(0) == allowed
    assert_cleaned(observed)
    return json.loads(observation.read_text()), expected


@pytest.mark.parametrize("pin", ["lowest", "highest"])
def test_fixed_worker_observes_requested_singleton_cpu(monkeypatch, tmp_path, pin):
    allowed = os.sched_getaffinity(0)
    if len(allowed) < 2:
        pytest.skip("two allowed CPUs are needed to distinguish pinning from inherited affinity")
    # Both variants distinguish pinning from inheritance; the highest variant
    # also defeats hardcoded-lowest pinning.
    cpu_id = max(allowed) if pin == "highest" else min(allowed)
    actual, limits = observe_worker_containment(monkeypatch, tmp_path, cpu_id)
    assert actual["affinity"] == [limits["cpu_id"]]


def test_fixed_worker_observes_all_thread_and_resource_limits(monkeypatch, tmp_path):
    actual, limits = observe_worker_containment(monkeypatch, tmp_path, min(os.sched_getaffinity(0)))
    assert actual["threads"] == dict.fromkeys(THREAD_VARIABLES, "1")
    assert actual["memory"] == [limits["memory_bytes"]] * 2
    assert actual["cpu"] == [limits["cpu_seconds"], limits["cpu_seconds"] + 1]


@pytest.mark.parametrize("source", [None, {}, {"manifest": {}, "inputs_id": "x"},
                                    {"manifest": {}, "inputs_id": "x", "store_root": "/s", "extra": 1}])
def test_public_fit_helper_refuses_a_malformed_source_before_effects(monkeypatch, source):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **k: pytest.fail("bad source launched child"))
    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", lambda *a, **k: pytest.fail("bad source made scratch"))
    with pytest.raises(OpsError) as caught:
        execution.run_fitted_trial(source, {}, {"horizons": []}, {}, {**LIMITS, "cpu_id": 0}, None)
    assert caught.value.problem == fail("INVALID_EXPERIMENT_SPEC",
        "search execution requires an admitted cohort source").problem
