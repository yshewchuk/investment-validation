"""Per-instance cohort admission memoisation: one native read per cohort, never failures."""
# ruff: noqa: F811
import pytest

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials
from experiments.magnitude_search_inputs import load_inputs, publish_inputs
from tests.v2.research.magnitude_trial_support import trial_cases, trial_forecasts
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401
from tests.v2.research.test_magnitude_search_trials import case  # noqa: F401


def _all(case):
    for args in trial_cases(case):
        case.recorder.start(*args)
        case.recorder.record(*args, forecasts=trial_forecasts(case, args, 7))


def test_one_native_admission_per_cohort_across_all_trials(case, monkeypatch):
    calls = []
    original = trials.admit_fold_inputs

    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(trials, "admit_fold_inputs", counted)
    _all(case)
    cohorts = {(outer_year, inner_year) for outer_year, inner_year, _ in trial_cases(case)}
    assert len(calls) == len(cohorts)
    assert len(calls) < 2 * len(trial_cases(case))


def test_memoised_slots_match_uncached_admission(case):
    _all(case)
    for outer_year, inner_year in {(args[0], args[1]) for args in trial_cases(case)}:
        inner = next(inner for outer in case.plan["folds"] if outer["outer"]["year"] == outer_year
                     for inner in outer["inner"] if inner["year"] == inner_year)
        cohort = content_hash([case.plan["plan_id"], outer_year, inner_year])
        fresh = trials.admit_fold_inputs(case.source.repository, case.source.registration.document, inner,
            history_failure=fail("EXPERIMENT_IDENTITY_CONFLICT", "x"))
        key = trials.SearchTrials._trial(case.plan, outer_year, inner_year,
                                         case.plan["candidates"][0]["candidate_id"])[0]
        inputs_id = case.recorder._read("inputs:" + key)["inputs_id"]
        stored = load_inputs(case.recorder.store, case.recorder._read("cohort:" + cohort), inputs_id)
        assert canonical_json(stored) == canonical_json(fresh)
        assert inputs_id == publish_inputs(case.recorder.store, fresh)[1]


def test_start_returns_detached_inputs_copy(case):
    args = trial_cases(case)[0]
    second = (args[0], args[1], case.plan["candidates"][1]["candidate_id"])
    started = case.recorder.start(*args)
    cohort = content_hash([case.plan["plan_id"], args[0], args[1]])
    started["inputs"]["features"].clear()
    assert case.recorder._memo[cohort]["features"]
    resumed = case.recorder.start(*second)
    assert canonical_json(resumed["inputs"]) == canonical_json(case.recorder._memo[cohort])


def test_internal_borrow_shares_memo_while_public_start_detaches(case):
    args = trial_cases(case)[0]
    second = (args[0], args[1], case.plan["candidates"][1]["candidate_id"])
    cohort = content_hash([case.plan["plan_id"], args[0], args[1]])
    borrowed = case.recorder._start(*args, detach=False)
    assert borrowed["inputs"] is case.recorder._memo[cohort]
    detached = case.recorder.start(*second)
    assert detached["inputs"] is not case.recorder._memo[cohort]
    assert canonical_json(detached["inputs"]) == canonical_json(case.recorder._memo[cohort])


def test_failed_admission_is_not_memoised(case, monkeypatch):
    args = trial_cases(case)[0]
    inner = next(inner for outer in case.plan["folds"] if outer["outer"]["year"] == args[0]
                 for inner in outer["inner"] if inner["year"] == args[1])
    cohort = content_hash([case.plan["plan_id"], args[0], args[1]])
    original = case.recorder._inputs
    state = []

    def flaky(fold):
        if not state:
            state.append(1)
            raise fail("RESOURCE_UNAVAILABLE", "temporary native input failure")
        return original(fold)

    monkeypatch.setattr(case.recorder, "_inputs", flaky)
    with pytest.raises(OpsError) as caught:
        case.recorder._cohort_inputs(cohort, inner)
    assert caught.value.problem == fail("RESOURCE_UNAVAILABLE", "temporary native input failure").problem
    assert cohort not in case.recorder._memo
    admitted = case.recorder._cohort_inputs(cohort, inner)
    assert admitted == case.recorder._memo[cohort]
    assert canonical_json(admitted) == canonical_json(original(inner))
