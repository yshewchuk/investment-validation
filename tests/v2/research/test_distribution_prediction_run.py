"""Real single-use journal/CAS with fixed native fitting and report seams."""
import time
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace

import pytest

from engine.v2.ops.errors import OpsError, fail
from experiments import distribution_prediction_run as execution
from tests.v2.research.test_distribution_prediction_plan import case as case


@pytest.fixture
def running(case, monkeypatch):
    case.trials.own.document = {"execution_plan": {"runner": execution.PredictionRun.source_roots[0]}}
    case.trials.recipe = {"horizons": [5, 3, 1]}
    case.native["folds"][0]["inner"] = [{"year": "2023"}]
    run = execution.PredictionRun(case.trials, case.run.request)
    calls, state = [], {"failure": None}
    fitted = set()
    def start(role, year, inner):
        calls.append(("start", role, year, inner))
    def fit(role, year, horizon, inner, *, bounded):
        assert bounded is True and run.record("started") is not None
        calls.append(("fit", role, horizon))
        if state["failure"] is not None:
            raise state["failure"]
        if role == "outer":
            fitted.add(horizon)
    fits = SimpleNamespace(fit=fit)
    def freeze(value, adapter, year):
        assert value is run and adapter is fits
        if fitted != {1, 3, 5}:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "incomplete frozen horizon evidence")
        calls.append(("freeze", year))
    def evaluate(value, adapter, year):
        assert calls[-1] == ("freeze", year)
        calls.append(("evaluate", year))
    groups = [{"outer_year": "2024", "evaluation_id": "fresh-native-evidence"}]
    summaries = {"forecast": {"crps": .125, "population": 2}, "diagnostics": {"coverage": .5}}
    def evidence(value, adapter):
        assert value is run and adapter is fits
        calls.append(("evidence",))
        return deepcopy((groups, summaries))
    @contextmanager
    def contained(limits, seconds):
        calls.append(("contained", deepcopy(limits), seconds))
        yield
    monkeypatch.setattr(case.trials, "start", start, raising=False)
    monkeypatch.setattr(execution, "DistributionFits", lambda trials: fits)
    monkeypatch.setattr(execution, "freeze_outer", freeze)
    monkeypatch.setattr(execution, "evaluate_outer", evaluate)
    monkeypatch.setattr(execution, "report_evidence", evidence)
    monkeypatch.setattr(execution, "contained", contained)
    run.seal(clock=case.clock)
    calls.clear()
    return SimpleNamespace(run=run, calls=calls, state=state, case=case, groups=groups, summaries=summaries)


def snapshot(case):
    return (case.conn.total_changes, tuple(tuple(row) for row in case.conn.execute(
        "SELECT slot,ref_json FROM magnitude_search_records ORDER BY slot")),
        tuple(sorted((str(path.relative_to(case.store.root)), sha256(path.read_bytes()).hexdigest())
            for path in case.store.root.rglob("*") if path.is_file())))


def test_runner_fits_all_horizons_then_freezes_then_evaluates_and_reports(running):
    result = running.run.run(clock=running.case.clock)
    calls = running.calls
    assert [call for call in calls if call[0] == "fit"] == [
        ("fit", role, horizon) for role in ("inner", "outer") for horizon in (5, 3, 1)]
    assert calls[-3:] == [("freeze", "2024"), ("evaluate", "2024"), ("evidence",)]
    assert result["schema_version"] == "distribution_prediction_run.v1"
    assert result["groups"] == running.groups and result["summary"] == running.summaries
    assert result["final_read_authority"] == result["promotion_authority"] == "none"
    assert running.run.record("completed") == {key: value for key, value in result.items() if key != "summary"}
    assert running.run.record("failed") is None
    assert set(vars(running.run)) == {"trials", "request"}
    assert calls[0][0] == "contained" and calls[0][1] == running.run.request["limits"]["parent"]
    assert 0 < calls[0][2] <= running.run.request["limits"]["job_wall_seconds"]


def test_completed_run_and_collection_reverify_without_fit_targets_or_writes(running, monkeypatch):
    expected = running.run.run(clock=running.case.clock)
    before = snapshot(running.case)
    for name in ("freeze_outer", "evaluate_outer", "forbid_fitting", "forbid_legacy_fitting"):
        monkeypatch.setattr(execution, name, lambda *args: pytest.fail("execution during replay"))
    monkeypatch.setattr(running.run, "_execute", lambda: pytest.fail("fresh execution during replay"))
    running.calls.clear()
    assert running.run.collect() == expected
    assert running.run.run(clock=running.case.clock) == expected
    assert [call for call in running.calls if call[0] != "contained"] == [("evidence",), ("evidence",)]
    assert snapshot(running.case) == before


@pytest.mark.parametrize("missing", [1, 3, 5])
def test_incomplete_horizons_refuse_before_evaluation_or_reports(running, missing):
    running.case.trials.recipe["horizons"].remove(missing)
    with pytest.raises(OpsError) as caught:
        running.run.run(clock=running.case.clock)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not any(call[0] in ("evaluate", "evidence") for call in running.calls)
    assert running.run.record("completed") is None and running.run.record("failed") is not None


@pytest.mark.parametrize("change", ["limits", "uncertainty", "malformed"])
def test_mutated_request_refuses_before_containment_or_effects(running, change):
    if change == "limits":
        running.run.request["limits"]["parent"]["cpu_seconds"] += 1
    elif change == "uncertainty":
        running.run.request["uncertainty"]["resamples"] += 1
    else:
        del running.run.request["limits"]
    before = snapshot(running.case)
    with pytest.raises(OpsError) as caught:
        running.run.run(clock=running.case.clock)
    assert caught.value.code == ("INVALID_EXPERIMENT_SPEC" if change == "malformed" else "EXPERIMENT_IDENTITY_CONFLICT")
    assert running.calls == [] and snapshot(running.case) == before


def test_expired_original_job_deadline_cannot_restart(running):
    before = snapshot(running.case)
    with pytest.raises(OpsError) as caught:
        running.run._run_contained(running.case.clock, time.monotonic() - 61)
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"
    assert running.calls == [] and snapshot(running.case) == before


@pytest.mark.parametrize("secondary", [False, True])
@pytest.mark.parametrize("primary", [fail("RESOURCE_LIMIT_EXCEEDED", "private primary"), SystemExit("SECRET primary exit")])
def test_primary_failure_retained_safe_code_and_attempt_consumed(running, monkeypatch, primary, secondary):
    running.state["failure"] = primary
    original = running.run.record
    def record(slot, body=None):
        if secondary and slot == "failed" and body is not None:
            raise OSError("SECRET secondary outage")
        return original(slot, body)
    monkeypatch.setattr(running.run, "record", record)
    if secondary:
        monkeypatch.setattr(execution, "decision", lambda *args: (_ for _ in ()).throw(RuntimeError("telemetry")))
    with pytest.raises(type(primary)) as caught:
        running.run.run(clock=running.case.clock)
    assert caught.value is primary
    expected_code = primary.code if isinstance(primary, OpsError) else "INTERRUPTED"
    assert original("failed") == (None if secondary else {
        "run_plan_id": original("seal")["run_plan_id"], "code": expected_code})
    assert original("completed") is None
    assert "SECRET" not in str([tuple(row) for row in running.case.conn.execute("SELECT * FROM magnitude_search_records")])
    before, calls = snapshot(running.case), list(running.calls)
    running.state["failure"] = None
    with pytest.raises(OpsError) as retry:
        running.run.run(clock=running.case.clock)
    assert retry.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert snapshot(running.case) == before
    assert [call for call in running.calls if call[0] != "contained"] == [call for call in calls if call[0] != "contained"]


@pytest.mark.parametrize("change", ["groups", "report", "authority", "elapsed", "schema"])
def test_collection_refuses_changed_completion_or_rehashed_report(running, monkeypatch, change):
    running.run.run(clock=running.case.clock)
    changed = deepcopy(running.run.record("completed"))
    if change == "groups":
        changed["groups"][0]["evaluation_id"] = "foreign"
    if change == "report":
        changed["reports"]["forecast"] = running.run.publish("distribution_prediction_wrong", [{"crps": 999}], count=1)
    if change == "authority":
        changed["final_read_authority"] = "granted"
    if change == "elapsed":
        changed["elapsed_seconds"] = True
    if change == "schema":
        changed["schema_version"] = "strategy"
    original = running.run.record
    monkeypatch.setattr(running.run, "record", lambda slot, body=None: changed if slot == "completed" else original(slot, body))
    before = snapshot(running.case)
    with pytest.raises(OpsError) as caught:
        running.run.collect()
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and snapshot(running.case) == before


def test_runner_rejects_other_registered_entrypoint_before_seal_or_start(case):
    case.trials.own.document = {"execution_plan": {"runner": "experiments/distribution_run.py"}}
    run = execution.PredictionRun(case.trials, case.run.request)
    before = snapshot(case)
    with pytest.raises(OpsError) as caught:
        run.seal(clock=case.clock)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT" and snapshot(case) == before
