"""Deterministic immutable job budget across restarts; no sleep or real jobs."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from experiments import size_job_budget as budget


@pytest.fixture
def state(monkeypatch):
    clock = SimpleNamespace(value=datetime(2026, 1, 1, tzinfo=timezone.utc))
    clock.now = lambda: clock.value
    saved = {}
    def read(conn, store, run, slot):
        return deepcopy(saved.get((run, slot)))
    def append(conn, store, run, slot, body):
        key = run, slot
        if key in saved:
            assert saved[key] == body
        saved[key] = deepcopy(body)
        return deepcopy(body)
    monkeypatch.setattr(budget, "read_record", read)
    monkeypatch.setattr(budget, "append_record", append)
    monkeypatch.setattr(budget, "SystemClock", lambda: clock)
    monkeypatch.setattr(budget, "execution_limits", lambda limits: {**limits, "cpu_id": 0})
    registration = SimpleNamespace(run_id="synthetic")
    limits = {"memory_bytes": 1000, "cpu_seconds": 20, "wall_seconds": 60}
    plan = {"plan_id": "plan", "request": {"limits": {"child": limits, "job": {"wall_seconds": 100}}}}
    return SimpleNamespace(clock=clock, saved=saved, registration=registration, plan=plan, limits=limits)


def test_restart_preserves_first_deadline_and_clamps_child(state):
    args = None, None, state.registration, state.plan
    assert budget.remaining(*args) == 100
    original = deepcopy(state.saved)
    state.clock.value += timedelta(seconds=75)
    assert budget.remaining(*args) == 25
    assert budget.resolved_limits(*args, state.limits)["wall_seconds"] == 25
    assert state.saved == original
    state.clock.value += timedelta(seconds=25)
    with pytest.raises(OpsError) as error:
        budget.remaining(*args)
    assert error.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert state.saved == original


def test_no_fit_probe_never_claims_new_job_budget(state):
    with no_fit_guard(), pytest.raises(RuntimeFitForbidden):
        budget.remaining(None, None, state.registration, state.plan)
    assert state.saved == {}


@pytest.mark.parametrize("field", ["memory_bytes", "cpu_seconds", "wall_seconds"])
def test_changed_child_limit_refuses_before_job_claim(state, field):
    limits = {**state.limits, field: state.limits[field] + 1}
    with pytest.raises(OpsError, match="sealed"):
        budget.resolved_limits(None, None, state.registration, state.plan, limits)
    assert state.saved == {}


@pytest.mark.parametrize("change", ["plan", "budget", "backward", "deadline", "extra"])
def test_job_identity_poison_or_clock_rollback_never_extends_budget(state, change):
    args = None, None, state.registration, state.plan
    budget.remaining(*args)
    record = state.saved["synthetic", "size-job-deadline"]
    if change == "plan":
        state.plan["plan_id"] = "changed"
    elif change == "budget":
        state.plan["request"]["limits"]["job"]["wall_seconds"] = 200
    elif change == "backward":
        state.clock.value -= timedelta(seconds=1)
    elif change == "deadline":
        record["deadline"] = "not-a-date"
    else:
        record["extra"] = 1
    before = deepcopy(state.saved)
    with pytest.raises(OpsError):
        budget.remaining(*args)
    assert state.saved == before


def test_deadline_publication_time_consumes_budget(state, monkeypatch):
    original = budget.append_record
    def slow_publication(*args):
        result = original(*args)
        state.clock.value += timedelta(seconds=7)
        return result
    monkeypatch.setattr(budget, "append_record", slow_publication)
    assert budget.remaining(None, None, state.registration, state.plan) == 93


@pytest.mark.parametrize("completed", [False, True])
def test_cli_run_after_deadline_collects_only_complete_replay(state, monkeypatch, capsys, tmp_path, completed):
    from experiments import size_search as cli
    state.plan.update(folds=[{"outer": {"year": "2024"}}])
    budget.remaining(None, None, state.registration, state.plan)
    state.clock.value += timedelta(seconds=101)
    before = deepcopy(state.saved)
    monkeypatch.setattr(cli, "_admit_driver", lambda *a: state.plan)
    monkeypatch.setattr(cli, "read_record", lambda *a: {} if completed else None)
    monkeypatch.setattr(cli, "run_search", lambda *a, **k: pytest.fail("expired dispatch"))
    monkeypatch.setattr(cli.signal, "alarm", lambda *a: pytest.fail("expired execution alarm"))
    monkeypatch.setattr(cli, "collect", lambda *a: {"verified": True})
    context = (None, None, None, state.registration, state.plan["request"], None)
    if completed:
        assert cli._run(context) == {"verified": True}
    else:
        with pytest.raises(OpsError) as caught:
            cli._run(context)
        assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
    connection = SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(cli, "connect", lambda *a, **k: connection)
    monkeypatch.setattr(cli, "ArtifactStore", lambda *a: None)
    monkeypatch.setattr(cli, "Repository", lambda *a, **k: None)
    monkeypatch.setattr(cli, "experiment_spec_from_document", lambda *a: None)
    monkeypatch.setattr(cli, "read_native_registration", lambda *a: state.registration)
    monkeypatch.setattr(cli, "from_document", lambda *a: None)
    monkeypatch.setattr(cli, "_read_document", lambda *a: state.plan["request"])
    monkeypatch.setattr(cli, "_contain", lambda *a: None)
    monkeypatch.setattr(cli.signal, "alarm", lambda *a: 0)  # main finally always cancels its alarm
    status = cli.main(["run", "--spec", "spec", "--request", "request", "--catalog", "catalog",
        "--store-root", str(tmp_path), "--plan-ref", "plan", "--no-ledger"])
    import json
    output = json.loads(capsys.readouterr().out)
    assert status == (0 if completed else 2)
    assert output == {"verified": True} if completed else output["refused"] == "RESOURCE_LIMIT_EXCEEDED"
    assert state.saved == before


def test_cli_partial_comparator_needs_original_budget(state, monkeypatch):
    from experiments import size_search as cli
    state.plan.update(folds=[{"outer": {"year": "2024"}}])
    budget.remaining(None, None, state.registration, state.plan)
    state.clock.value += timedelta(seconds=101)
    monkeypatch.setattr(cli, "_admit_driver", lambda *a: state.plan)
    monkeypatch.setattr(cli, "read_record", lambda *a: None if a[-1].startswith("comparison-") else {})
    monkeypatch.setattr(cli, "run_search", lambda *a, **k: pytest.fail("partial expired dispatch"))
    monkeypatch.setattr(cli, "collect", lambda *a: pytest.fail("partial collection"))
    with pytest.raises(OpsError, match="deadline exhausted"):
        cli._run((None, None, None, state.registration, state.plan["request"], None))


def test_completed_marker_does_not_bypass_readonly_verification(state, monkeypatch):
    from experiments import size_search as cli
    from engine.v2.ops.errors import fail
    state.plan.update(folds=[{"outer": {"year": "2024"}}])
    monkeypatch.setattr(cli, "_admit_driver", lambda *a: state.plan)
    monkeypatch.setattr(cli, "read_record", lambda *a: {"forged": True})
    monkeypatch.setattr(cli, "remaining", lambda *a: pytest.fail("new deadline"))
    monkeypatch.setattr(cli, "run_search", lambda *a, **k: pytest.fail("fallback execution"))
    def refuse(*args):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "completed journal differs")
    monkeypatch.setattr(cli, "collect", refuse)
    with pytest.raises(OpsError, match="completed journal differs"):
        cli._run((None, None, None, state.registration, state.plan["request"], None))
    assert state.saved == {}
