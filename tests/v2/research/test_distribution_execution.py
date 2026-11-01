"""Tiny fixed child, bounded protocol, sealed limits and interrupted claim oracles."""
# ruff: noqa: F811
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.models import no_fit as legacy_guard
from engine.v2.models import no_fit as native_guard
from engine.v2.ops.errors import OpsError
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.foundation import content_hash
from experiments import distribution_execution as execution
from experiments import distribution_fitted as fitted
from tests.v2.research.test_distribution_fitted import case  # noqa: F401
from tests.v2.research.test_distribution_worker import LIMITS, admitted  # noqa: F401


def _error(code, action):
    with pytest.raises(OpsError) as caught:
        action()
    assert caught.value.code == code


def _script(monkeypatch, script):
    original, launched = subprocess.Popen, []

    def launch(args, **kwargs):
        child = original([args[0], "-c", script, *args[3:]], **kwargs)
        launched.append((child, args, kwargs))
        return child

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    return launched


def test_tiny_real_child_shards_results_and_parent_commits_only_verified_success(admitted, monkeypatch):
    original, launches = subprocess.Popen, []

    def launch(args, **kwargs):
        assert any(slot.startswith("distribution:fit-start:") for run, slot in admitted.records)
        assert not any(slot.startswith("distribution:fit-result:") for run, slot in admitted.records)
        launches.append((args, kwargs))
        return original(args, **kwargs)

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    monkeypatch.setattr(fitted, "fit_candidate", lambda *a, **kw: pytest.fail("fit leaked into parent"))
    result = admitted.fits.fit("inner", "2024", 1, "2023", bounded=True)
    assert result["tail"]["excesses"] == [0, 2]
    assert [row["event_id"] for row in result["rows"]] == ["Z", "A"]
    assert result["rows"][0]["point_prediction"] == pytest.approx(10.5625)
    # Constant training features leave the classifier at observed class priors:
    # one label in bin 0, one in bin 1, and two in overflow bin 80.
    assert [row["absolute_move_pct"] for row in admitted.inputs["training"]] == [0, .25, 20, 22]
    expected = [.25, .25] + [0.] * 78 + [.5]
    for row in result["rows"]:
        assert row["probabilities"] == pytest.approx(expected)
    assert result["receipt"]["diagnostics"]["classifier"]["total_trees"] == 300
    assert len(launches) == 1
    args, kwargs = launches[0]
    assert args[1:3] == ["-m", "experiments.distribution_worker"]
    assert {name: kwargs["env"][name] for name in THREAD_VARIABLES} == dict.fromkeys(THREAD_VARIABLES, "1")
    assert set(json.loads(args[-1])) == {*LIMITS, "cpu_id"}
    assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["start_new_session"] is True
    assert not Path(args[3]).parent.exists()
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("replay launched"))
    assert admitted.fits.fit("inner", "2024", 1, "2023", bounded=True) == result
    assert admitted.fits.read("inner", "2024", 1, "2023") == result


@pytest.mark.parametrize("guard", [legacy_guard, native_guard])
def test_parent_guards_precede_context_and_launch(admitted, monkeypatch, guard):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        admitted.fits.fit(object(), object(), object(), bounded=True)
    assert admitted.calls == []
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        execution.run_distribution_fit(None, None, None, None, None)


@pytest.mark.parametrize("limits", [None, {}, {**LIMITS, "wall_seconds": True}, {**LIMITS, "cpu_seconds": 0}])
def test_native_limit_validation_precedes_launch(admitted, monkeypatch, limits):
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))
    _error("INVALID_EXPERIMENT_SPEC", lambda: execution.run_distribution_fit(admitted.source, admitted.recipe, ["Z", "A"], 1, limits))


@pytest.mark.parametrize("script,code", [
    ("print('x' * 5000)", "RESOURCE_LIMIT_EXCEEDED"),
    ("print('not JSON')", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("print('{}')", "EXPERIMENT_IDENTITY_CONFLICT"),
    ("pass", "RESOURCE_UNAVAILABLE"),
    ("import sys; print('{}'); sys.exit(1)", "RESOURCE_UNAVAILABLE"),
    ("import os,signal; os.kill(os.getpid(),signal.SIGXCPU)", "RESOURCE_LIMIT_EXCEEDED"),
    ("print('{\"error\":\"EXPERIMENT_VARIANT_FAILED\"}')", "EXPERIMENT_VARIANT_FAILED"),
])
def test_child_failure_never_commits_or_retries(admitted, monkeypatch, script, code):
    launched = _script(monkeypatch, script)
    _error(code, lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=True))
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=True))
    assert len(launched) == 1 and launched[0][0].poll() is not None
    assert not Path(launched[0][1][3]).parent.exists()
    assert not any(slot.startswith("distribution:fit-result:") for run, slot in admitted.records)


def test_deadline_kills_and_reaps_child_without_completion(admitted, monkeypatch):
    original = admitted.trials.read_fit_context

    def context(*args):
        value = original(*args)
        value["plan"]["budget"]["child_limits"]["wall_seconds"] = 1
        return value

    admitted.trials.read_fit_context = context
    launched = _script(monkeypatch, "import time; time.sleep(60)")
    _error("RESOURCE_LIMIT_EXCEEDED", lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=True))
    assert launched[0][0].poll() is not None
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: admitted.fits.read("inner", "2024", 1, "2023"))


def test_child_sends_memory_failure_not_partial_success(admitted, monkeypatch):
    launched = _script(monkeypatch, "from experiments import distribution_candidate_fit as f; "
        "exec('def refuse(*a, **kw): raise MemoryError'); f.fit_candidate=refuse; "
        "from experiments.distribution_worker import main; main()")
    _error("RESOURCE_LIMIT_EXCEEDED", lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=True))
    assert launched[0][0].poll() == 0
    assert not any(slot.startswith("distribution:fit-result:") for run, slot in admitted.records)


@pytest.mark.parametrize("value", [None, 1, "yes"])
def test_bounded_option_is_explicit_boolean(admitted, value):
    _error("INVALID_EXPERIMENT_SPEC", lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=value))
    assert admitted.calls == []


def test_transport_publisher_has_no_completion_authority(admitted):
    result = fitted.publish_fit_output(admitted.trials.store,
        [{"event_id": "Z", "probabilities": [1.0] + [0.0] * 80, "point_prediction": 0.0}],
        {"policy": "unresolved", "sample_count": 0, "excesses": []}, {})
    assert set(result) == {"forecasts", "tail", "diagnostics"}
    assert admitted.records == {("source-run", "seal"): {"untouched": True}}


def test_public_surface_is_fixed_child_only():
    assert execution.__all__ == ["run_distribution_fit"]


def test_unbounded_completion_cannot_acquire_bounded_provenance(admitted, monkeypatch):
    result = admitted.fits.fit("inner", "2024", 1, "2023")
    assert result["receipt"]["schema_version"] == "distribution_fitted.v1"
    assert set(result["receipt"]) == {"schema_version", "claim", "result_id", "forecasts", "tail",
        "diagnostics", "status", "final_read_authority"}
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("replay launched"))
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=True))
    assert admitted.fits.read("inner", "2024", 1, "2023") == result
    assert not any(slot.startswith("distribution:fit-bound:") for run, slot in admitted.records)


def test_bounded_replay_requires_exact_sealed_execution_marker(admitted, monkeypatch):
    result = admitted.fits.fit("inner", "2024", 1, "2023", bounded=True)
    marker = admitted.records["own-run", "distribution:fit-bound:" + result["receipt"]["claim"]["trial"]["trial_id"]]
    assert marker == {"plan_id": result["receipt"]["claim"]["plan_id"],
                      "worker": "experiments/distribution_worker.py", "limits": LIMITS}
    marker["limits"]["wall_seconds"] += 1
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("replay launched"))
    _error("EXPERIMENT_IDENTITY_CONFLICT", lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=True))


@pytest.mark.parametrize("replay", ["read", "unbounded", "bounded"])
@pytest.mark.parametrize("change", ["missing_marker", "marker_worker", "marker_limits",
    "missing_execution", "execution_worker", "execution_limits", "execution_plan", "extra_execution",
    "unknown_schema", "downgraded_schema", "plan_limits"])
def test_every_bounded_replay_binds_execution_before_shards(admitted, monkeypatch, replay, change):
    result = admitted.fits.fit("inner", "2024", 1, "2023", bounded=True)
    key = result["receipt"]["claim"]["trial"]["trial_id"]
    record = admitted.records["own-run", "distribution:fit-result:" + key]
    marker_key = ("own-run", "distribution:fit-bound:" + key)
    if change == "missing_marker":
        del admitted.records[marker_key]
    elif change == "marker_worker":
        admitted.records[marker_key]["worker"] = "other-worker"
    elif change == "marker_limits":
        admitted.records[marker_key]["limits"]["wall_seconds"] += 1
    elif change == "missing_execution":
        record.pop("execution", None)
    elif change.startswith("execution_") or change == "extra_execution":
        execution_binding = record.setdefault("execution", {})
        if change == "execution_limits":
            execution_binding["limits"] = {**LIMITS, "wall_seconds": LIMITS["wall_seconds"] + 1}
        else:
            member = {"execution_worker": "worker", "execution_plan": "plan_id", "extra_execution": "extra"}[change]
            execution_binding[member] = "changed"
    elif change == "unknown_schema":
        record["schema_version"] = "distribution_fitted.v3"
    elif change == "downgraded_schema":
        record["schema_version"] = "distribution_fitted.v1"
        record.pop("execution", None)
    else:
        original = admitted.trials.read_fit_context

        def context(*args):
            value = original(*args)
            value["plan"]["budget"]["child_limits"]["wall_seconds"] += 1
            return value

        admitted.trials.read_fit_context = context
    record["result_id"] = content_hash({name: value for name, value in record.items() if name != "result_id"})
    calls = admitted.calls.copy()
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("replay launched"))
    monkeypatch.setattr(fitted, "fit_candidate", lambda *a, **kw: pytest.fail("replay fitted"))
    monkeypatch.setattr(fitted, "iter_section", lambda *a, **kw: pytest.fail("read shards before execution validation"))
    action = (lambda: admitted.fits.read("inner", "2024", 1, "2023")) if replay == "read" else (
        lambda: admitted.fits.fit("inner", "2024", 1, "2023", bounded=replay == "bounded"))
    _error("EXPERIMENT_IDENTITY_CONFLICT", action)
    assert admitted.calls[len(calls):] == [("context", "inner", "2024", "2023")]


def test_bounded_completion_schema_and_read_only_replay_under_both_guards(admitted, monkeypatch):
    result = admitted.fits.fit("inner", "2024", 1, "2023", bounded=True)
    receipt = result["receipt"]
    assert receipt["schema_version"] == "distribution_fitted.v2"
    assert receipt["execution"] == {"plan_id": receipt["claim"]["plan_id"],
        "worker": "experiments/distribution_worker.py", "limits": LIMITS}
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("replay launched"))
    monkeypatch.setattr(fitted, "fit_candidate", lambda *a, **kw: pytest.fail("replay fitted"))
    assert admitted.fits.fit("inner", "2024", 1, "2023", bounded=False) == result
    assert admitted.fits.fit("inner", "2024", 1, "2023", bounded=True) == result
    with legacy_guard.no_fit_guard(), native_guard.no_fit_guard():
        assert admitted.fits.read("inner", "2024", 1, "2023") == result


def test_header_cap_refuses_before_launch(admitted, monkeypatch):
    monkeypatch.setattr(execution, "MAX_BYTES", 32)
    monkeypatch.setattr(execution.subprocess, "Popen", lambda *a, **kw: pytest.fail("launched"))
    _error("RESOURCE_LIMIT_EXCEEDED", lambda: execution.run_distribution_fit(
        admitted.source, admitted.recipe, ["Z", "A"], 1, LIMITS))


def test_idle_child_logs_bounded_heartbeat_and_waits_for_success_exit(monkeypatch):
    times = iter([0, 0, 30, 31, 31, 31])
    monkeypatch.setattr(execution.time, "monotonic", lambda: next(times))
    chunks = iter([b'{"reference":"compact"}', b""])
    monkeypatch.setattr(execution.os, "read", lambda *args: next(chunks))
    calls = []

    class Selector:
        def __enter__(self):
            self.round = 0
            return self

        def __exit__(self, *args):
            pass

        def register(self, *args):
            pass

        def select(self, timeout):
            self.round += 1
            calls.append(("select", timeout))
            return [] if self.round == 1 else [True]

    monkeypatch.setattr(execution.selectors, "DefaultSelector", Selector)
    monkeypatch.setattr(execution, "decision", lambda *args: calls.append(("heartbeat", args)))
    child = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 1),
                            wait=lambda **kwargs: calls.append(("wait", kwargs)) or 0)
    assert execution._receive(child, 60) == {"reference": "compact"}
    assert sum(call[0] == "heartbeat" for call in calls) == 1
    assert calls[-1] == ("wait", {"timeout": 29})
