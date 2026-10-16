"""Synthetic real-reader qualification and registered whole-run preflight oracles."""
from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError, fail
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_inputs
from experiments import magnitude_inputs as admitted_inputs
from experiments import native_registration as native
from tests.v2.research.magnitude_trial_support import (
    PlanRepository,
    calendar_sessions,
    catalog_rows,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)
from tests.v2.research.test_event_history import commit_event_history
from tests.v2.research.test_native_registration import artifact_object_bytes, native_spec

MONTH, BOUNDARY = "2025-01", "2024-06-01"
RECIPE = {"schema_version": "earnings_magnitude.v1", "units": "percentage_points",
    "target": "absolute_earnings_close_to_close_return", "price_basis": "split_neutral_non_dividend_adjusted",
    "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
    "secondary_diagnostics": ["mae", "quantile_calibration"]}


@contextmanager
def _refuses(code):
    """Assert the exact nonretryable envelope, not matching exception text."""
    with pytest.raises(DataError) as caught:
        yield caught
    assert caught.value.code == code
    assert caught.value.problem.retryable is False


def _qualify(repository, snapshot, events, **changes):
    """Call the public training reader with explicit candidate identities."""
    return magnitude_inputs.qualify_magnitude_training_targets(repository, snapshot,
        **{"as_of_month": MONTH, "labels_before": BOUNDARY,
           "event_ids": [event["event_id"] for event in events], **changes})


def _objects_for(repository, snapshot, events):
    """Expected provenance comes from the fixture's committed fragment metadata."""
    records = {record.time_min: record for record in repository.fragment_records(snapshot, "computed_moves")}
    expected = []
    for event in events:
        record = records[event["event_date"].date().isoformat()]
        expected.append({"event_id": event["event_id"], "fragment_id": record.fragment_id,
            "object_id": record.object_ref.object_id, "content_hash": record.object_ref.content_hash})
    return expected


def test_training_qualifies_before_numeric_parse_and_binds_every_opened_object(tmp_path, monkeypatch):
    """Equality is late; zero is data; omissions never become zero-filled targets."""
    events = [history_event(day) for day in ("2024-01-04", "2024-02-05", "2024-03-05",
                                    "2024-04-05", "2024-05-06", "2024-05-20")]
    random, rolling, future = history_event("2024-03-06", protected=True), history_event("2024-12-03"), history_event("2024-07-02")
    moves = [outcome_move(events[0], -3.5, available_as_of_date="2024-05-31"), outcome_move(events[1], 0.0),
        outcome_move(events[2], None, skipped=True, available_as_of_date=None),
        outcome_move(events[3], None, available_as_of_date=None),
        outcome_move(events[4], None, available_as_of_date=BOUNDARY),
        outcome_move(events[5], None, available_as_of_date="2024-06-02")]
    conn, repository, snapshot = commit_event_history(tmp_path, [*events, random, rolling, future],
        [*moves, *[outcome_move(event, None) for event in (random, rolling, future)]])
    try:
        expected_objects = _objects_for(repository, snapshot, events)
        poison_outcomes(monkeypatch, repository, snapshot, ["2024-03-06", "2024-12-03", "2024-07-02"])
        parse, calls = magnitude_inputs._parse_target, []
        def parse_only_qualified(rows, day):
            assert day.isoformat() in ("2024-01-04", "2024-02-05")
            calls.append(day.isoformat())
            return parse(rows, day)
        monkeypatch.setattr(magnitude_inputs, "_parse_target", parse_only_qualified)
        before = catalog_rows(conn), artifact_object_bytes(repository._store)
        result = _qualify(repository, snapshot, list(reversed(events)))
        assert result.event_id.tolist() == [event["event_id"] for event in events[:2]]
        assert result.realized_move_pct.tolist() == [-3.5, 0.0]
        assert result.absolute_move_pct.tolist() == [3.5, 0.0]
        assert result.target_available_on.tolist() == ["2024-05-31", "2024-02-07"]
        assert result.target_object_id.tolist() == [item["object_id"] for item in expected_objects[:2]]
        assert calls == ["2024-01-04", "2024-02-05"]
        identity = {"schema_version": "research_magnitude_training_targets.v1",
            "snapshot_id": snapshot.snapshot_id, "holdout_as_of_month": MONTH,
            "random_membership_version": "canonical-event-sha256.v1", "rolling_membership_version": "calendar-months.v1",
            "population_use": "post-release selection", "purpose": "selection",
            "computed_moves_version": snapshot.table_versions["computed_moves"].dataset_version_id,
            "labels_before": BOUNDARY, "target_recipe": RECIPE, "target_recipe_id": content_hash(RECIPE),
            "candidate_event_ids": [event["event_id"] for event in events],
            "qualified_event_ids": [event["event_id"] for event in events[:2]], "read_inputs": expected_objects,
            "omitted_labels": [{"event_id": event["event_id"], "reason": reason} for event, reason in zip(events[2:],
                ("skipped", "unavailable", "not_available_before_fit", "not_available_before_fit"))]}
        assert {key: result.attrs[key] for key in identity} == identity
        assert result.attrs["inputs_id"] == content_hash(identity)
        again = _qualify(repository, snapshot, events)
        pd.testing.assert_frame_equal(result, again)
        assert result.attrs == again.attrs
        result.attrs["read_inputs"].clear()
        assert _qualify(repository, snapshot, events).attrs["read_inputs"] == expected_objects
        assert (catalog_rows(conn), artifact_object_bytes(repository._store)) == before
        assert all(query.snapshot_id == snapshot.snapshot_id for _, query in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("kind,code", [("random", "HOLDOUT_ACCESS_DENIED"), ("rolling", "HOLDOUT_ACCESS_DENIED"),
    ("alias", "HOLDOUT_ACCESS_DENIED"), ("unknown", "HOLDOUT_ACCESS_DENIED"),
    ("empty", "HOLDOUT_ACCESS_DENIED"), ("same_day", "CONTRACT_MISMATCH"), ("future", "CONTRACT_MISMATCH")])
def test_explicit_training_refusals_precede_all_outcome_access(tmp_path, monkeypatch, kind, code):
    day = {"rolling": "2024-12-03", "same_day": BOUNDARY, "future": "2024-07-02"}.get(kind, "2024-05-06")
    event = history_event(day, protected=kind == "random")
    events = [event, {**event, "event_id": "alias", "session": "BMO"}] if kind == "alias" else [event]
    conn, repository, snapshot = commit_event_history(tmp_path, events, [outcome_move(event)])
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        ids = [] if kind == "empty" else ["unknown" if kind == "unknown" else event["event_id"]]
        with _refuses(code):
            _qualify(repository, snapshot, [], event_ids=ids)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("cutoff", [None, True, [], "20240601", "2024-02-30", "2024-06-01T00:00:00Z"])
def test_invalid_training_clock_refuses_before_repository(cutoff):
    with _refuses("CONTRACT_MISMATCH"):
        _qualify(None, None, [], labels_before=cutoff)


@pytest.mark.parametrize("operation", ["qualify", "preflight"])
def test_implicit_candidates_are_not_a_training_or_preflight_population(operation):
    with _refuses("CONTRACT_MISMATCH"):
        if operation == "qualify":
            _qualify(None, None, [], event_ids=None)
        else:
            magnitude_inputs.preflight_magnitude_targets(None, None, as_of_month=MONTH, event_ids=None)


@pytest.mark.parametrize("value,changes", [(None, {}), (float("nan"), {}), (1.0, {"available_as_of_date": "bad"}),
    (1.0, {"available_as_of_date": "20240508"}), (1.0, {"available_as_of_date": "2024-05-06"}),
    (1.0, {"available_as_of_date": "2024-05-05"})])
def test_eligible_corruption_refuses_instead_of_selective_dropping(tmp_path, value, changes):
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [event], [outcome_move(event, value, **changes)])
    try:
        with _refuses("EXPERIMENT_VARIANT_FAILED"):
            _qualify(repository, snapshot, [event])
    finally:
        conn.close()


@pytest.mark.parametrize("changes", [{"realized_move_pct": value} for value in
    (True, False, "1", "2", float("inf"), -float("inf"))] +
    [{"skipped": value} for value in (None, 0, 1, "false")] +
    [{"event_date": value} for value in ("2024-05-05", "2024-05-07")])
def test_decoded_training_guard_refuses_values_native_table_types_cannot_publish(changes):
    """Direct parser cases complement real scans without pretending invalid Parquet is native."""
    event = SimpleNamespace(event_date=pd.Timestamp("2024-05-06"))
    row = {"event_date": "2024-05-06", "skipped": False,
           "realized_move_pct": 2.0, "available_as_of_date": "2024-05-08", **changes}
    with _refuses("EXPERIMENT_VARIANT_FAILED"):
        magnitude_inputs._training_label([row], event, date(2024, 6, 1))


@pytest.mark.parametrize("reason,changes", [("skipped", {"skipped": True}),
    ("unavailable", {"available_as_of_date": None}),
    ("not_available_before_fit", {"available_as_of_date": BOUNDARY})])
def test_empty_qualification_refuses_with_complete_read_evidence(tmp_path, reason, changes):
    event = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [event], [outcome_move(event, None, **changes)])
    try:
        with _refuses("POPULATION_COLLAPSED") as caught:
            _qualify(repository, snapshot, [event])
        details = caught.value.problem.details
        identity = details["qualification"]
        assert identity["qualified_event_ids"] == []
        assert identity["omitted_labels"] == [{"event_id": event["event_id"], "reason": reason}]
        assert identity["read_inputs"] == _objects_for(repository, snapshot, [event])
        assert details["inputs_id"] == content_hash(identity)
    finally:
        conn.close()


@pytest.mark.parametrize("missing", [False, True])
def test_training_preflights_later_mixed_or_missing_fragment_before_safe_first_read(tmp_path, monkeypatch, missing):
    first, later, protected = history_event("2024-01-04"), history_event("2024-03-05"), history_event("2024-03-06", protected=True)
    moves = [outcome_move(first), outcome_move(protected)] + ([] if missing else [outcome_move(later)])
    conn, repository, snapshot = commit_event_history(tmp_path, [first, later, protected], moves,
                                        mixed_days=("2024-03-05", "2024-03-06"))
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with _refuses("HOLDOUT_ACCESS_DENIED"):
            _qualify(repository, snapshot, [first, later])
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


def test_late_repository_failure_is_not_retried_or_returned_as_partial_training(tmp_path, monkeypatch):
    events = [history_event("2024-01-04"), history_event("2024-02-05")]
    conn, repository, snapshot = commit_event_history(tmp_path, events, [outcome_move(event) for event in events])
    original, calls = repository.scan, []
    def fail_late(query, *, table_name):
        if table_name == "computed_moves":
            calls.append(query.time_interval.start_inclusive)
            if len(calls) == 2:
                raise fail("OBJECT_CORRUPT", "synthetic late failure")
        yield from original(query, table_name=table_name)
    try:
        before = catalog_rows(conn), artifact_object_bytes(repository._store)
        monkeypatch.setattr(repository, "scan", fail_late)
        with _refuses("OBJECT_CORRUPT"):
            _qualify(repository, snapshot, events)
        assert calls == ["2024-01-04", "2024-02-05"]
        assert (catalog_rows(conn), artifact_object_bytes(repository._store)) == before
    finally:
        conn.close()


def _registered(tmp_path, events, *, years, mixed_days):
    """Real native registration followed by the production read-only input boundary."""
    conn, original, snapshot = commit_history_inputs(tmp_path, events, [outcome_move(event, None) for event in events],
        days=calendar_sessions("2023-12-01", "2025-01-10").days, mixed_days=mixed_days)
    repository, root = PlanRepository(conn, original._store), tmp_path / "code"
    root.mkdir()
    (root / "run.py").write_text("raise AssertionError('not an execution test')\n")
    spec = native_spec(price_source="computed_moves", economic_params={}, seed=0, folds=list(years), runner="run.py")
    request = dict(code_root=root, as_of_month="2026-01", magnitude_training_start="2023-12-10",
                   magnitude_labels_before="2025-02-01")
    registered = native.register_native(conn, original._store, repository, spec, **request)
    return SimpleNamespace(conn=conn, repository=repository, store=original._store, snapshot=snapshot,
                           spec=spec, request=request, registered=registered)


def test_registered_union_preflight_reaches_late_fold_before_opening_early_safe_objects(tmp_path, monkeypatch):
    """A second-fold mixed object blocks the whole run before any first-fold label."""
    train, early, later = (history_event(day) for day in ("2023-12-12", "2024-06-12", "2025-01-07"))
    protected = history_event("2025-01-08", protected=True)
    source = _registered(tmp_path, [train, early, later, protected], years=("2024", "2025"),
                         mixed_days=("2025-01-07", "2025-01-08"))
    try:
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        with _refuses("HOLDOUT_ACCESS_DENIED"):
            admitted_inputs.admit_magnitude_inputs(source.conn, source.store, source.repository,
                                                   source.spec, **source.request)
        assert all(name != "computed_moves" for name, _ in source.repository.scans)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()


def test_registered_preflight_omits_unused_window_events_and_returns_exact_detached_identity(tmp_path, monkeypatch):
    """An unused late-December mixed fragment is not part of actual fold membership."""
    train, unused, test = (history_event(day) for day in ("2023-12-12", "2024-12-30", "2025-01-07"))
    protected = history_event("2024-12-31", protected=True)
    source = _registered(tmp_path, [train, unused, test, protected], years=("2025",),
                         mixed_days=("2024-12-30", "2024-12-31"))
    try:
        objects = _objects_for(source.repository, source.snapshot, [train, test])
        poison_outcomes(monkeypatch, source.repository, source.snapshot, (), all_outcomes=True)
        before = catalog_rows(source.conn), artifact_object_bytes(source.store)
        registration, preflight = admitted_inputs.admit_magnitude_inputs(
            source.conn, source.store, source.repository, source.spec, **source.request)
        assert registration == source.registered
        assert [event["event_id"] for event in registration.document["magnitude_plan"]["events"]] == [
            event["event_id"] for event in (train, unused, test)]
        identity = {"schema_version": "research_magnitude_preflight.v1", "snapshot_id": source.snapshot.snapshot_id,
            "holdout_as_of_month": "2026-01", "random_membership_version": "canonical-event-sha256.v1",
            "rolling_membership_version": "calendar-months.v1", "population_use": "post-release selection",
            "purpose": "selection", "computed_moves_version": source.snapshot.table_versions["computed_moves"].dataset_version_id,
            "event_ids": [train["event_id"], test["event_id"]], "planned_inputs": objects}
        assert preflight == {**identity, "preflight_id": content_hash(identity)}
        preflight["event_ids"].clear()
        assert admitted_inputs.admit_magnitude_inputs(source.conn, source.store, source.repository,
            source.spec, **source.request)[1] == {**identity, "preflight_id": content_hash(identity)}
        assert all(name != "computed_moves" for name, _ in source.repository.scans)
        assert (catalog_rows(source.conn), artifact_object_bytes(source.store)) == before
    finally:
        source.conn.close()
