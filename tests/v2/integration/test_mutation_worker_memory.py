"""Instrumented synthetic workers retain explicit, enforced resource budgets.

# packages: ops, data
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.ops import executor
from engine.v2.ops.profiles import DEFAULT_POLICY
from tests.ops_support import TEST_POLICY, FakeClock
from tests.v2.ops import test_massive_snapshot as fixture


def _instrument(monkeypatch, marker="real"):
    def wrapper(*args):
        pass
    # Replace only this test module's view, never mutmut's live global module.
    monkeypatch.setattr(fixture, "sys", SimpleNamespace(modules={
        "mutmut.mutation.trampoline": SimpleNamespace(wrap_in_trampoline=wrapper)}))
    monkeypatch.setattr(fixture.incremental, "_mutmut_mutated",
                        wrapper if marker == "real" else lambda *args: None, raising=False)
    return wrapper


def _memory(policy):
    return next(profile.memory_bytes for profile in policy.profiles if profile.name == "io_fetch")


@pytest.mark.parametrize("mode", ["", "stats", "engine.x__mutmut_1"])
def test_authored_worker_budget_ignores_environment_package_and_path(monkeypatch, mode):
    _instrument(monkeypatch, marker="lookalike")
    monkeypatch.setenv("MUTANT_UNDER_TEST", mode)
    monkeypatch.setattr(fixture.incremental, "__file__", "/tmp/mutants/incremental.py")
    assert fixture.snapshot_worker_policy() is TEST_POLICY
    monkeypatch.delattr(fixture.incremental, "_mutmut_mutated")
    assert fixture.snapshot_worker_policy() is TEST_POLICY
    assert _memory(TEST_POLICY) == 384 << 20
    assert _memory(DEFAULT_POLICY) == 512 << 20


@pytest.mark.parametrize("modules", [{}, {"mutmut.mutation.trampoline": SimpleNamespace()},
    {"mutmut.mutation.trampoline": SimpleNamespace(wrap_in_trampoline=None)}])
def test_authored_worker_without_a_trampoline_keeps_its_budget(monkeypatch, modules):
    monkeypatch.setattr(fixture, "sys", SimpleNamespace(modules=modules))
    monkeypatch.delattr(fixture.incremental, "_mutmut_mutated", raising=False)
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")
    assert fixture.snapshot_worker_policy() is TEST_POLICY
    assert _memory(TEST_POLICY) == 384 << 20
    assert _memory(DEFAULT_POLICY) == 512 << 20


def test_only_instrumented_fixture_io_fetch_memory_changes(monkeypatch):
    _instrument(monkeypatch)
    monkeypatch.delenv("MUTANT_UNDER_TEST", raising=False)  # includes collection before stats mode
    policy = fixture.snapshot_worker_policy()
    assert _memory(policy) == 1 << 30
    assert _memory(TEST_POLICY) == 384 << 20
    assert _memory(DEFAULT_POLICY) == 512 << 20
    expected = replace(TEST_POLICY, profiles=tuple(
        replace(profile, memory_bytes=1 << 30) if profile.name == "io_fetch" else profile
        for profile in TEST_POLICY.profiles))
    assert policy == expected  # all CPU/thread/admission/other-profile settings stay unchanged


@pytest.mark.parametrize("generated_module,expected", [
    ("massive_daily", 384 << 20), ("incremental", 1 << 30),
])
def test_allowance_tracks_incremental_instrumentation_not_its_sibling(
        monkeypatch, generated_module, expected):
    wrapper = _instrument(monkeypatch)
    for module in (fixture.incremental, fixture.massive_daily):
        monkeypatch.delattr(module, "_mutmut_mutated", raising=False)
    monkeypatch.setattr(getattr(fixture, generated_module), "_mutmut_mutated", wrapper, raising=False)
    assert _memory(fixture.snapshot_worker_policy()) == expected
    assert _memory(TEST_POLICY) == 384 << 20
    assert _memory(DEFAULT_POLICY) == 512 << 20


@pytest.mark.parametrize("instrumented", [False, True])
@pytest.mark.parametrize("excess", [0, 1])
def test_actual_watchdog_enforces_both_fixture_limits(monkeypatch, tmp_path, instrumented, excess):
    with monkeypatch.context() as scope:
        _instrument(scope, marker="real" if instrumented else "lookalike")
        limit = _memory(fixture.snapshot_worker_policy())
    assert limit == (1 << 30 if instrumented else 384 << 20)
    observed = limit + excess  # synthetic observation, never allocate a GiB to test rejection
    claim = SimpleNamespace(attempt_id="synthetic", resources=SimpleNamespace(reserved_memory_bytes=limit))
    stops = []
    monkeypatch.setattr(executor, "observe", lambda *args: ((), True, observed))
    monkeypatch.setattr(executor, "persist_members", lambda *args: None)
    monkeypatch.setattr(executor, "stop", lambda *args, **kwargs: stops.append(True))
    with (tmp_path / "worker-result").open("w+b") as result:
        running = executor.Running(claim, SimpleNamespace(poll=lambda: None), result.fileno(), (), 0)
        status = executor.poll(None, running, boot_id="synthetic", clock=FakeClock())
        assert running.failure == ("RESOURCE_LIMIT_EXCEEDED" if excess else None)
        assert running.peak == observed and status["memory"] == observed
        assert stops == ([True] if excess else [])
        assert status["done"] is False
        # A later tick observes the worker and its whole family gone. This
        # positive oracle rejects a completion calculation hardcoded to False.
        exit_code = -15 if excess else 0
        running.process = SimpleNamespace(poll=lambda: exit_code)
        monkeypatch.setattr(executor, "observe", lambda *args: ((), False, 0))
        status = executor.poll(None, running, boot_id="synthetic", clock=FakeClock())
        assert status == {"done": True, "memory": 0, "exit_code": exit_code}
        assert running.failure == ("RESOURCE_LIMIT_EXCEEDED" if excess else None)
        assert running.peak == observed
    assert stops == ([True] if excess else [])


@pytest.mark.parametrize("module,requested,expected", [
    ("data_incremental", None, 1), ("data_incremental", 4, 1), ("data_incremental", 1, 1),
    ("foundation", None, 2), ("foundation", 4, 4), ("foundation", 1, 1),
])
def test_mutmut_child_cap_is_shard_specific(monkeypatch, tmp_path, module, requested, expected):
    from tests.v2.integration.test_mutation_ci import run_mutation_command

    rc, calls = run_mutation_command(monkeypatch, tmp_path, rc=1, module=module, max_children=requested)
    assert rc == 1 and len(calls) == 1
    command = calls[0][0]
    assert command[command.index("--max-children") + 1] == str(expected)
