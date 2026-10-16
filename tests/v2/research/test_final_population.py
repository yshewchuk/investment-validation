"""Private final population reader over real synthetic snapshots; metadata only."""
from dataclasses import replace

import pytest

from engine.v2.contracts import FinalHoldoutDeclaration, FinalHoldoutRequest
from engine.v2.data import errors
from engine.v2.data.errors import DataError
from engine.v2.foundation import artifact_reference
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research._final_population import FinalPopulation, load_final_population
from engine.v2.research.experiment_population import load_population
from tests.v2.research.test_prediction_inputs import (
    MONTH,
    commit_prediction_inputs,
    prediction_event,
)


def _ref(label, schema):
    return artifact_reference(label.encode(), schema)


def _request(snapshot, **overrides):
    registration = _ref("registration", "native_experiment_registration.v1.1")
    value = FinalHoldoutRequest(
        experiment_id="synthetic-experiment", registration_run_id="synthetic-run",
        variant_id=registration.content_hash, registration_ref=registration,
        snapshot_id=snapshot.snapshot_id, holdout_as_of_month=MONTH,
        random_membership_version=ExperimentHoldouts.random_version,
        rolling_membership_version=ExperimentHoldouts.rolling_version,
        execution_plan_hash=_ref("plan", "x.v1.0").content_hash,
        source_closure_hash=_ref("sources", "x.v1.0").content_hash,
        final_read=FinalHoldoutDeclaration(
            evaluation_recipe_ref=_ref("recipe", "final_holdout_recipe.v1.0"),
            completeness_bar_ref=_ref("bar", "final_holdout_completeness.v1.0"),
            frozen_model_refs=(_ref("model", "synthetic_model.v1.0"),)))
    return replace(value, **overrides)


def _events():
    return {
        "eligible": prediction_event("2024-05-02", prefix="E"),
        "random": prediction_event("2024-05-03", random=True, prefix="R"),
        "rolling": prediction_event("2024-12-03", prefix="L"),
        "both": prediction_event("2024-12-04", random=True, prefix="B"),
        "ambiguous": {**prediction_event("2024-12-05", prefix="A"), "date_conflict": True},
    }


def test_sets_are_separate_and_agree_with_shared_population_labels(tmp_path):
    events = _events()
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(events.values()), include_targets=False)
    try:
        before = conn.total_changes
        result = load_final_population(repository, _request(snapshot))
        ids = {name: row["event_id"] for name, row in events.items()}
        assert isinstance(result, FinalPopulation)
        assert result.random_event_ids == tuple(sorted([ids["random"], ids["both"]]))
        assert result.rolling_event_ids == tuple(sorted([ids["rolling"], ids["both"]]))
        assert result.overlap_count == 1
        assert result.ambiguous_count == 1
        assert result.snapshot_id == snapshot.snapshot_id and result.holdout_as_of_month == MONTH
        bound = _request(snapshot)
        assert result.random_membership_version == bound.random_membership_version == ExperimentHoldouts.random_version
        assert result.rolling_membership_version == bound.rolling_membership_version == ExperimentHoldouts.rolling_version
        assert result == load_final_population(repository, _request(snapshot))
        shared = load_population(repository, snapshot, as_of_month=MONTH)
        labels = {item["event_id"]: set(item["memberships"]) for item in shared.attrs["holdout_exclusions"]}
        assert {k for k, v in labels.items() if "random" in v and "ambiguous" not in v} == set(result.random_event_ids)
        assert {k for k, v in labels.items() if "rolling" in v and "ambiguous" not in v} == set(result.rolling_event_ids)
        assert all(name == "earnings_events" for name, _ in repository.scans)
        assert conn.total_changes == before
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["not-a-request", "version", "future-month", "malformed-month"])
def test_invalid_request_refuses_before_any_scan(tmp_path, kind):
    events = _events()
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(events.values()), include_targets=False)
    try:
        request = _request(snapshot)
        request = {"not-a-request": {"snapshot_id": snapshot.snapshot_id},
                   "version": replace(request, random_membership_version="other.v9"),
                   "future-month": replace(request, holdout_as_of_month="2999-01"),
                   "malformed-month": replace(request, holdout_as_of_month="2025-1")}[kind]
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED") as refusal:
            load_final_population(repository, request)
        assert not refusal.value.problem.retryable
        assert repository.scans == []
    finally:
        conn.close()


def test_unresolved_snapshot_is_typed_without_scan(tmp_path):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(_events().values()), include_targets=False)
    try:
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
            load_final_population(repository, _request(snapshot, snapshot_id="missing-snapshot"))
        assert repository.scans == []
    finally:
        conn.close()


def test_population_with_no_held_out_events_is_not_a_success(tmp_path):
    events = _events()
    conn, repository, snapshot = commit_prediction_inputs(
        tmp_path, [events["eligible"], events["ambiguous"]], include_targets=False)
    try:
        with pytest.raises(DataError, match="POPULATION_COLLAPSED"):
            load_final_population(repository, _request(snapshot))
    finally:
        conn.close()


def test_one_empty_set_is_still_a_population(tmp_path):
    events = _events()
    conn, repository, snapshot = commit_prediction_inputs(
        tmp_path, [events["eligible"], events["rolling"]], include_targets=False)
    try:
        result = load_final_population(repository, _request(snapshot))
        assert result.random_event_ids == ()
        assert result.rolling_event_ids == (events["rolling"]["event_id"],)
        assert result.overlap_count == 0
    finally:
        conn.close()


def test_missing_events_table_refuses_typed_without_partial_population(tmp_path):
    events = _events()
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(events.values()), include_targets=True)
    try:
        stripped = replace(snapshot, table_versions={
            name: version for name, version in snapshot.table_versions.items() if name != "earnings_events"})
        repository.resolve = lambda _snapshot_id: stripped
        with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
            load_final_population(repository, _request(snapshot))
    finally:
        conn.close()


def test_resolved_snapshot_with_different_identity_refuses_without_scan(tmp_path):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(_events().values()), include_targets=False)
    try:
        other = replace(snapshot, snapshot_id="different-snapshot")
        repository.resolve = lambda _snapshot_id: other
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED") as refusal:
            load_final_population(repository, _request(snapshot))
        assert not refusal.value.problem.retryable
        assert repository.scans == []
    finally:
        conn.close()


def test_events_table_without_fragments_is_an_empty_collapsed_population(tmp_path):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(_events().values()), include_targets=False)
    try:
        repository.fragment_records = lambda _snapshot, _table: []
        with pytest.raises(DataError, match="POPULATION_COLLAPSED") as refusal:
            load_final_population(repository, _request(snapshot))
        assert not refusal.value.problem.retryable
    finally:
        conn.close()


def test_integrity_refusal_from_resolve_propagates_unchanged(tmp_path):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, list(_events().values()), include_targets=False)
    try:
        def corrupt(_snapshot_id):
            raise errors.fail("MANIFEST_CORRUPT", "synthetic integrity failure")
        repository.resolve = corrupt
        with pytest.raises(DataError) as refusal:
            load_final_population(repository, _request(snapshot))
        assert refusal.value.problem.code == "MANIFEST_CORRUPT"
        assert refusal.value.problem.category == "integrity"
        assert repository.scans == []
    finally:
        conn.close()
