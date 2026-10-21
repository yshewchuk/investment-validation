"""Activation of sharded cohort inputs: writers publish shards + manifest (#991 fix 4, split PR 3)."""
# ruff: noqa: F811
import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_shards as shards
from experiments.magnitude_search_inputs import load_inputs
from tests.v2.research.magnitude_trial_support import trial_cases
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401
from tests.v2.research.test_magnitude_search_trial_integrity import (  # noqa: F401
    journal,
    native_inputs,
)
from tests.v2.research.test_magnitude_search_trials import case  # noqa: F401


def test_start_commits_a_manifest_bound_by_the_manifest_hash(case):
    started = case.recorder.start(*trial_cases(case)[0])
    key = content_hash([case.plan["plan_id"], *trial_cases(case)[0][:2], trial_cases(case)[0][2]])
    admitted = case.recorder._read("inputs:" + key)
    manifest = case.recorder._read("cohort:" + admitted["cohort"])
    assert manifest["schema_version"] == shards.MANIFEST_SCHEMA
    assert admitted["inputs_id"] == shards.manifest_id(manifest)
    assert load_inputs(case.recorder.store, manifest, admitted["inputs_id"]) == started["inputs"]


def test_second_trial_of_a_cohort_reuses_the_same_manifest(case):
    first, second = trial_cases(case)[0], None
    for args in trial_cases(case)[1:]:
        if args[:2] == first[:2]:
            second = args
            break
    assert second is not None
    case.recorder.start(*first)
    case.recorder.start(*second)
    ids = {case.recorder._read("inputs:" + content_hash([case.plan["plan_id"], *args]))["inputs_id"]
           for args in (first, second)}
    assert len(ids) == 1


def test_start_refuses_a_cohort_slot_holding_a_whole_cohort_record(journal):
    cohort = content_hash(["plan", "2024", "2023"])
    journal.recorder._append("cohort:" + cohort, native_inputs())
    with pytest.raises(OpsError) as caught:
        journal.recorder.start("2024", "2023", "z-last")
    assert caught.value.problem.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert caught.value.problem.retryable is False
