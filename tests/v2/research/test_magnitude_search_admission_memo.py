"""Per-instance cohort admission memoisation: one native read per cohort, never failures."""
# ruff: noqa: F811
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError, fail
from experiments import magnitude_search_trials as trials
from tests.v2.research.magnitude_trial_support import trial_cases
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401
from tests.v2.research.test_magnitude_search_trials import _all, case  # noqa: F401


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
    for args in trial_cases(case):
        key = trials.SearchTrials._trial(case.plan, *args)[0]
        admitted = case.recorder._read("inputs:" + key)
        assert admitted["inputs_id"] == content_hash(case.recorder._read("cohort:" + admitted["cohort"]))


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
    assert admitted[1] == content_hash(admitted[0])
