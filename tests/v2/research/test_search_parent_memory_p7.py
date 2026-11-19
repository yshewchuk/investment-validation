"""Search parent memory P7: pre-published cohort inputs adopted by workers, not re-admitted."""
# ruff: noqa: F811
import os
from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import canonical_json, to_document
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials
from experiments import size_search_binding as binding
from experiments import v2_magnitude_search_run as run_mod
from experiments.magnitude_search_plan import canonical_inner, cohort_key
from tests.v2.research.magnitude_trial_support import trial_cases
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401
from tests.v2.research.test_magnitude_search_trials import case  # noqa: F401

REQUEST = {"schema_version": "magnitude_search_request.v4", "limits": {"parent": {"memory_bytes": 1 << 30}}}


@pytest.fixture
def copy(case, tmp_path):
    """A second sealed journal with the same plan; taken before any start so it is unpublished."""
    conn = connect(tmp_path / "copy.sqlite")
    case.source.conn.backup(conn)
    yield conn
    conn.close()


def _recorder(case, conn):
    source = case.source
    return trials.SearchTrials(conn, source.store, Repository(conn, source.store), source.registration,
                               case.request, case.recorder.plan_ref)


def _cohort(plan, args):
    owner, _ = canonical_inner(plan, args[0], args[1])
    return cohort_key(plan, owner, args[1])


def _key(plan, args):
    return trials.SearchTrials._trial(plan, *args)[0]


def test_prepublish_then_adopt_writes_the_same_slots_as_lazy_admission(case, copy):
    published = _recorder(case, copy)
    published.prepublish(trial_cases(case))
    adopt = _recorder(case, copy)
    adopt.adopt_published = True
    for args in trial_cases(case):
        adopt.start(*args)
        case.recorder.start(*args)
    for args in trial_cases(case):
        cohort, key = _cohort(case.plan, args), _key(case.plan, args)
        assert adopt._read("cohort:" + cohort) is not None and adopt._read("inputs:" + key) is not None
        assert canonical_json(adopt._read("cohort:" + cohort)) == canonical_json(case.recorder._read("cohort:" + cohort))
        assert canonical_json(adopt._read("inputs:" + key)) == canonical_json(case.recorder._read("inputs:" + key))


def test_adopting_worker_never_admits_a_prepublished_cohort(case, copy, monkeypatch):
    _recorder(case, copy).prepublish(trial_cases(case))
    worker = _recorder(case, copy)
    worker.adopt_published = True

    def forbidden(*args, **kwargs):
        pytest.fail("worker admitted a pre-published cohort")

    monkeypatch.setattr(worker, "_inputs", forbidden)
    for args in trial_cases(case):
        worker.start(*args)
    assert {_cohort(case.plan, args) for args in trial_cases(case)} == set(worker._published)


def test_adoption_off_still_admits_lazily_after_prepublish(case, copy, monkeypatch):
    _recorder(case, copy).prepublish(trial_cases(case))
    plain = _recorder(case, copy)
    assert plain.adopt_published is False
    admitted, original = [], plain._inputs

    def counted(inner, plan=None):
        admitted.append(inner["year"])
        return original(inner, plan=plan)

    monkeypatch.setattr(plain, "_inputs", counted)
    args = trial_cases(case)[0]
    plain.start(*args)
    assert len(admitted) == 1


def test_prepublish_failure_matches_start_and_leaves_no_cohort_slot(case, copy, monkeypatch):
    def refuse(*args, **kwargs):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "search native history binding changed")

    recorder = _recorder(case, copy)
    monkeypatch.setattr(recorder, "_inputs", refuse)
    with pytest.raises(OpsError) as published:
        recorder.prepublish(trial_cases(case))
    assert all(recorder._read("cohort:" + _cohort(case.plan, args)) is None for args in trial_cases(case))

    monkeypatch.setattr(case.recorder, "_inputs", refuse)
    with pytest.raises(OpsError) as lazy:
        case.recorder.start(*trial_cases(case)[0])
    assert published.value.problem.code == lazy.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_prepublish_visits_each_distinct_cohort_once_and_skips_published(case, copy, monkeypatch):
    seeded = _recorder(case, copy)
    first = trial_cases(case)[0]
    seeded.start(*first)
    published = {_cohort(case.plan, args) for args in trial_cases(case)}
    visited = []
    original = seeded._cohort_inputs

    def counted(cohort, inner, plan=None):
        visited.append(cohort)
        return original(cohort, inner, plan=plan)

    monkeypatch.setattr(seeded, "_cohort_inputs", counted)
    seeded.prepublish(trial_cases(case) + trial_cases(case))
    assert sorted(visited) == sorted(published - {_cohort(case.plan, first)})
    visited.clear()
    seeded.prepublish(trial_cases(case))
    assert visited == []


def _wire(monkeypatch, outcome):
    """Drive _run_search's parallel branch with one fresh trial and a fake prepublish pool."""
    created, abort = [], Mock()

    class Pool:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

        def submit(self, fn, *args):
            future = Future()
            if isinstance(outcome, BaseException):
                future.set_exception(outcome)
            else:
                future.set_result(outcome)
            return future

    monkeypatch.setattr(run_mod, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(run_mod, "_abort_executor", abort)
    monkeypatch.setattr(run_mod, "validate_workers", lambda request: 2)
    monkeypatch.setattr(run_mod, "validate_execution_limits", lambda limits: None)
    monkeypatch.setattr(run_mod, "_admit_driver", lambda *args: {"folds": []})
    monkeypatch.setattr(run_mod, "SearchTrials", lambda *args: Mock())
    monkeypatch.setattr(run_mod, "log_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mod, "panel_memo_holder", lambda: None)
    monkeypatch.setattr(run_mod, "read_record", lambda conn, store, run_id, slot: None)
    monkeypatch.setattr(run_mod, "_roster", lambda plan: [(2024, 2023, "overall")])
    monkeypatch.setattr(run_mod, "trial_key", lambda plan, *args: "key")
    monkeypatch.setattr(run_mod, "possible_slots", lambda plan: [])
    monkeypatch.setattr(run_mod, "_parallel_preflight", lambda conn, request, limits, workers: ([0, 1], ":memory:"))
    monkeypatch.setattr(run_mod, "forbid_fitting", lambda *args: None)
    monkeypatch.setattr(run_mod, "forbid_legacy_fitting", lambda *args: None)
    monkeypatch.setattr(run_mod, "remaining", lambda *args: None)
    return created, abort


def _run_parallel():
    return run_mod.run_search(None, SimpleNamespace(root="store"), None, SimpleNamespace(run_id="run"), REQUEST, None,
                              limits={"memory_bytes": 1})


def test_prepublish_worker_problem_receipt_raises_its_ops_error(monkeypatch):
    receipt = run_mod._worker_problem(fail("EXPERIMENT_IDENTITY_CONFLICT", "search anchor differs"))
    created, abort = _wire(monkeypatch, receipt)
    with pytest.raises(OpsError) as caught:
        _run_parallel()
    assert caught.value.problem == fail("EXPERIMENT_IDENTITY_CONFLICT", "search anchor differs").problem
    assert abort.call_count == 1 and len(created) == 1
    assert created[0].kwargs["max_workers"] == 1


def test_prepublish_broken_pool_maps_to_resource_unavailable_and_aborts(monkeypatch):
    created, abort = _wire(monkeypatch, BrokenProcessPool("gone"))
    with pytest.raises(OpsError) as caught:
        _run_parallel()
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "search worker process exited unexpectedly").problem
    assert abort.call_count == 1 and len(created) == 1


def test_raise_receipt_maps_ops_data_and_junk():
    problem = fail("EXPERIMENT_IDENTITY_CONFLICT", "x").problem
    with pytest.raises(OpsError) as ops:
        run_mod._raise_receipt({"worker_problem": to_document(problem), "worker_error_type": "ops"})
    assert ops.value.problem == problem
    with pytest.raises(DataError) as data:
        run_mod._raise_receipt({"worker_problem": to_document(problem), "worker_error_type": "data"})
    assert data.value.problem == problem


@pytest.mark.parametrize("junk", ["junk", {"worker_error_type": "ops"},
                                  {"worker_problem": {}, "worker_error_type": "bogus"}])
def test_raise_receipt_refuses_junk_as_resource_unavailable(junk):
    with pytest.raises(OpsError) as caught:
        run_mod._raise_receipt(junk)
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "search worker returned an invalid result").problem


def test_prepublish_task_returns_none_and_forwards_the_trials(monkeypatch):
    args = (2024, 2023, "overall")
    trials_obj = Mock()
    monkeypatch.setattr(run_mod, "_PARALLEL", (None, None, trials_obj, None, None))
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None)
    assert run_mod._prepublish_task(([args], 0)) is None
    trials_obj.prepublish.assert_called_once_with([args])


def test_worker_trial_releases_the_panel_memo_after_completing(monkeypatch):
    panel = {"panel": object()}
    monkeypatch.setattr(binding, "_PANEL_MEMO", panel)
    trials_obj = Mock()
    registration = SimpleNamespace(run_id="run")
    monkeypatch.setattr(run_mod, "_PARALLEL", (None, None, trials_obj, registration, {"folds": []}))
    monkeypatch.setattr(run_mod, "trial_key", lambda plan, *args: "key")
    monkeypatch.setattr(os, "sched_setaffinity", lambda pid, cpus: None)
    assert run_mod._parallel_trial_task(((2024, 2023, "overall"), 0, {"memory_bytes": 1})) is None
    trials_obj._execute_borrowed.assert_called_once()
    assert panel == {}
