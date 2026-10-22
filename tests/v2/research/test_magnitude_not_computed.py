"""Not-computed (skipped) magnitude labels: opt-in omission in loader and evidence."""
import pytest

from engine.v2.data.computed_moves_table import COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_inputs
from experiments import magnitude_search_evidence as evidence
from tests.v2.research.magnitude_trial_support import (
    commit_history_inputs,
    history_event,
    outcome_move,
)
from tests.v2.research.test_event_history import commit_event_history

MONTH = "2025-01"
CUTOFF = "2024-07-01"


def _load(repository, snapshot, **kwargs):
    return magnitude_inputs.load_magnitude_targets(repository, snapshot,
        **{**dict(as_of_month=MONTH, labels_before=CUTOFF), **kwargs})


def _skipped(event):
    return outcome_move(event, None, skipped=True, available_as_of_date=None)


def test_skipped_row_refuses_by_default(tmp_path):
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [event], [_skipped(event)])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _load(repository, snapshot)
    finally:
        conn.close()


def test_omit_flag_drops_skipped_event_and_binds_its_object(tmp_path):
    skipped = history_event("2024-01-04")
    kept_first, kept_second = history_event("2024-02-05"), history_event("2024-03-05")
    conn, repository, snapshot = commit_event_history(
        tmp_path, [skipped, kept_first, kept_second],
        [_skipped(skipped), outcome_move(kept_first), outcome_move(kept_second)])
    try:
        result = _load(repository, snapshot, omit_not_computed=True)
        assert result.event_id.tolist() == [kept_first["event_id"], kept_second["event_id"]]
        assert result.attrs["omitted_labels"] == [{"event_id": skipped["event_id"], "reason": "skipped"}]
        assert [row["event_id"] for row in result.attrs["consumed_inputs"]] == [
            skipped["event_id"], kept_first["event_id"], kept_second["event_id"]]
        by_event = {row["event_id"]: row["object_id"] for row in result.attrs["consumed_inputs"]}
        records = {record.time_min: record for record in repository.fragment_records(snapshot, "computed_moves")}
        record = records[skipped["event_date"].date().isoformat()]
        omitted_entry = next(row for row in result.attrs["consumed_inputs"]
                             if row["event_id"] == skipped["event_id"])
        assert omitted_entry == {"event_id": skipped["event_id"], "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash}
        assert result.target_object_id.tolist() == [
            by_event[kept_first["event_id"]], by_event[kept_second["event_id"]]]
        assert result.realized_move_pct.tolist() == [2.0, 2.0]
        assert result.absolute_move_pct.tolist() == [2.0, 2.0]
    finally:
        conn.close()


def test_v5_omit_reports_the_writers_reason(tmp_path):
    skipped, kept = history_event("2024-01-04"), history_event("2024-03-05")
    conn, repository, snapshot = commit_history_inputs(
        tmp_path, [skipped, kept],
        [outcome_move(skipped, None, skipped=True, available_as_of_date=None,
                      not_computed_reason="no_session"),
         outcome_move(kept)],
        contract=COMPUTED_MOVES_V5_CONTRACT, include_calendar=False)
    try:
        result = _load(repository, snapshot, omit_not_computed=True)
        assert result.event_id.tolist() == [kept["event_id"]]
        assert result.attrs["omitted_labels"] == [{"event_id": skipped["event_id"], "reason": "no_session"}]
    finally:
        conn.close()


def test_no_omitted_key_when_nothing_skipped(tmp_path):
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [event], [outcome_move(event)])
    try:
        result = _load(repository, snapshot, omit_not_computed=True)
        assert "omitted_labels" not in result.attrs
        assert result.attrs["inputs_id"] == content_hash({
            key: value for key, value in result.attrs.items()
            if key not in ("inputs_id", "holdout_exclusions")})
    finally:
        conn.close()


def test_omitted_labels_are_sorted_by_event_id_not_date_order(tmp_path):
    later_aaa, earlier_zzz = history_event("2024-05-06", ticker="AAA"), history_event("2024-01-04", ticker="ZZZ")
    kept = history_event("2024-03-05")
    conn, repository, snapshot = commit_event_history(
        tmp_path, [kept, later_aaa, earlier_zzz],
        [outcome_move(kept), _skipped(later_aaa), _skipped(earlier_zzz)])
    try:
        result = _load(repository, snapshot, omit_not_computed=True)
        assert result.event_id.tolist() == [kept["event_id"]]
        labels = [item["event_id"] for item in result.attrs["omitted_labels"]]
        assert labels == sorted(labels) == [later_aaa["event_id"], earlier_zzz["event_id"]]
    finally:
        conn.close()


def test_all_skipped_events_yield_empty_result_with_omissions(tmp_path):
    first, second = history_event("2024-01-04"), history_event("2024-03-05")
    conn, repository, snapshot = commit_event_history(
        tmp_path, [first, second], [_skipped(first), _skipped(second)])
    try:
        result = _load(repository, snapshot, omit_not_computed=True)
        assert len(result) == 0
        labels = [item["event_id"] for item in result.attrs["omitted_labels"]]
        assert labels == sorted([first["event_id"], second["event_id"]])
        assert [row["event_id"] for row in result.attrs["consumed_inputs"]] == [
            first["event_id"], second["event_id"]]
    finally:
        conn.close()


def test_non_boolean_omit_flag_refuses(tmp_path):
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [event], [outcome_move(event)])
    try:
        with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
            _load(repository, snapshot, omit_not_computed="false")
    finally:
        conn.close()


def test_null_availability_still_refuses_with_omit_flag(tmp_path):
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(
        tmp_path, [event], [outcome_move(event, None, available_as_of_date=None)])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _load(repository, snapshot, omit_not_computed=True)
    finally:
        conn.close()


COMMON = {"snapshot_id": "snapshot", "holdout_as_of_month": "2026-01",
    "random_membership_version": "canonical-event-sha256.v1", "rolling_membership_version": "calendar-months.v1",
    "population_use": "post-release selection", "purpose": "selection", "computed_moves_version": "moves"}
PHYSICAL = [{"event_id": "event-a", "fragment_id": "fragment-a", "object_id": "object-a", "content_hash": "hash-a"},
            {"event_id": "event-b", "fragment_id": "fragment-b", "object_id": "object-b", "content_hash": "hash-b"}]
FOLD = {"evaluation_event_ids": ["event-a", "event-b"], "evaluation_labels_before": "2025-02-01"}


def _preflight():
    return {**COMMON, "planned_inputs": PHYSICAL}


def _receipt(omitted):
    recipe = magnitude_inputs.magnitude_recipe()
    identity = {"schema_version": "research_magnitude_targets.v1", **COMMON, "labels_before": "2025-02-01",
        "target_recipe": recipe, "target_recipe_id": content_hash(recipe), "consumed_inputs": PHYSICAL,
        "excluded_future_events": []}
    if omitted:
        identity["omitted_labels"] = omitted
    return {**identity, "inputs_id": content_hash(identity), "holdout_exclusions": []}


def test_validate_targets_accepts_omitted_label_receipt():
    omitted = [{"event_id": "event-b", "reason": "skipped"}]
    evidence.validate_targets(_preflight(), FOLD, _receipt(omitted),
        [{"event_id": "event-a", "absolute_move_pct": 2.5}])


@pytest.mark.parametrize("omitted", [
    [{"event_id": "event-c", "reason": "skipped"}],
    [{"event_id": "event-b", "reason": "unavailable"}],
    [{"event_id": "event-b", "reason": "skipped"}, {"event_id": "event-a", "reason": "skipped"}],
    [{"event_id": "event-b", "reason": "skipped"}, {"event_id": "event-b", "reason": "skipped"}],
])
def test_validate_targets_rejects_bad_omitted_labels(omitted):
    with pytest.raises(ValueError):
        evidence.validate_targets(_preflight(), FOLD, _receipt(omitted),
            [{"event_id": "event-a", "absolute_move_pct": 2.5}])
