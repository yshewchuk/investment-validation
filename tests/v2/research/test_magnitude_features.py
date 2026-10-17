"""Pinned calendar/history binding with synthetic native objects and poison guards."""
from dataclasses import replace

import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError, fail
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_features
from tests.v2.research.magnitude_trial_support import (
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)

MONTH = "2025-01"
START = "2023-01-01"


def _read(repository, snapshot, target, **changes):
    return magnitude_features.load_magnitude_history(repository, snapshot, **{
        **dict(event_id=target["event_id"], as_of_month=MONTH, history_start=START, horizon=1), **changes})


def test_pinned_calendar_and_history_binding_never_reads_target_or_protected_outcomes(tmp_path, monkeypatch):
    target, first, second, third = (history_event(day) for day in ("2024-06-12", "2023-01-03", "2024-06-04", "2024-06-06"))
    delayed, skipped = history_event("2024-06-10"), history_event("2023-02-03")
    random, rolling = history_event("2023-04-03", protected=True), history_event("2024-12-03")
    cutoff, future = history_event("2024-06-11"), history_event("2024-06-13")
    alias = history_event("2023-05-03")
    sibling = {**alias, "event_id": "ambiguous-sibling", "session": "BMO"}
    forbidden = [target, random, rolling, cutoff, future, alias]
    events = [first, second, third, delayed, skipped, *forbidden, sibling]
    moves = [outcome_move(e, v) for e, v in zip([first, second, third], [-2.0, 4.0, 6.0])]
    moves += [outcome_move(e, None) for e in [delayed, *forbidden]]
    moves += [outcome_move(skipped, None, skipped=True, available_as_of_date=None)]
    conn, repository, snapshot = commit_history_inputs(tmp_path, events, moves)
    try:
        poison_outcomes(monkeypatch, repository, snapshot, [e["event_date"].date().isoformat() for e in forbidden])
        before = conn.total_changes
        result = _read(repository, snapshot, target)
        assert result == _read(repository, snapshot, target)
        assert result["schema_version"] == "magnitude_history_inputs.v1"
        assert result["inputs_id"] == content_hash({k: v for k, v in result.items() if k != "inputs_id"})
        assert result["snapshot_id"] == snapshot.snapshot_id
        assert result["session"] == "AMC" and result["event_id"] == target["event_id"]
        assert result["features"]["mean_prior_abs_move"] == 4.0
        assert result["features"]["n_prior"] == 3
        assert result["features"]["ema4_prior_move"] is None
        assert result["forecast_cutoff"]["source_session_ceiling"] == "2024-06-11"
        assert result["history_input"]["decision_session"] == "2024-06-11"
        assert result["history_input"]["holdout_as_of_month"] == MONTH
        assert result["history_input"]["schema_version"] == "research_event_history.v2"
        used = [r["event_id"] for r in result["history_input"]["history"]]
        read = [r["event_id"] for r in result["history_input"]["read_inputs"]]
        assert used == [e["event_id"] for e in (first, second, third)]
        assert read == [e["event_id"] for e in (first, skipped, second, third, delayed)]
        assert all(r["fragment_id"] and r["content_hash"].startswith("sha256:")
                   for r in result["history_input"]["read_inputs"])
        assert result["availability"] == {"basis": "reconstructed_date_bounds",
            "event_schedule_as_known": "unverified", "source_publication": "unverified"}
        assert result["feature_recipe"]["columns"] == list(result["features"])
        assert result["feature_recipe"]["imputation"] == "none"
        source = result["calendar_input"]
        assert source["dataset_version"] == snapshot.table_versions["daily_market"].dataset_version_id
        assert source["calendar_version"] == snapshot.calendar_version and source["columns"] == ["date"]
        assert {r["object_id"] for r in source["objects"]} == {
            r.object_ref.object_id for r in repository.fragment_records(snapshot, "daily_market")}
        assert all(q.columns == ("date",) for name, q in repository.scans if name == "daily_market")
        assert all(q.snapshot_id == snapshot.snapshot_id for _, q in repository.scans)
        assert conn.total_changes == before
        assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("LEDGER.csv"))
    finally:
        conn.close()


@pytest.mark.parametrize("horizon,ceiling,count,absolute", [
    (5, "2024-06-05", 1, 2.0), (3, "2024-06-07", 2, 3.0), (1, "2024-06-11", 3, 4.0),
])
def test_forecast_horizon_drives_actual_history_availability(tmp_path, horizon, ceiling, count, absolute):
    target = history_event("2024-06-12")
    prior = [history_event(day) for day in ("2023-01-03", "2024-06-04", "2024-06-06")]
    conn, repository, snapshot = commit_history_inputs(tmp_path, [target, *prior], [outcome_move(e, v) for e, v in zip(prior, [-2, 4, 6])])
    try:
        result = _read(repository, snapshot, target, horizon=horizon)
        assert result["forecast_cutoff"]["source_session_ceiling"] == ceiling
        assert result["features"]["n_prior"] == count
        assert result["features"]["mean_prior_abs_move"] == absolute
        different = _read(repository, snapshot, target, horizon=3 if horizon == 1 else 1)
        assert result["inputs_id"] != different["inputs_id"]
    finally:
        conn.close()


@pytest.mark.parametrize("session,ceiling", [("AMC", "2024-06-11"), ("BMO", "2024-06-10")])
def test_session_comes_from_canonical_event_and_empty_history_retains_nulls(tmp_path, session, ceiling):
    target = {**history_event("2024-06-12"), "session": session}
    conn, repository, snapshot = commit_history_inputs(tmp_path, [target], [])
    try:
        result = _read(repository, snapshot, target)
        assert result["forecast_cutoff"]["source_session_ceiling"] == ceiling
        assert result["history_input"]["history_status"] == "empty"
        assert result["history_input"]["read_inputs"] == []
        assert result["features"]["n_prior"] == 0
        assert all(value is None for name, value in result["features"].items() if name != "n_prior")
    finally:
        conn.close()


def test_history_availability_on_source_session_uses_the_existing_inclusive_day_contract(tmp_path):
    target, prior = history_event("2024-06-12"), history_event("2023-01-03")
    conn, repository, snapshot = commit_history_inputs(tmp_path, [target, prior],
        [outcome_move(prior, -7.0, available_as_of_date="2024-06-11")])
    try:
        on_day = _read(repository, snapshot, target)
        earlier = _read(repository, snapshot, target, horizon=3)
        assert on_day["features"]["mean_prior_abs_move"] == 7.0
        assert earlier["features"]["n_prior"] == 0
        assert earlier["history_input"]["read_inputs"] == on_day["history_input"]["read_inputs"]
        assert earlier["history_input"]["omitted_history"] == [{"event_id": prior["event_id"], "reason": "unavailable"}]
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["random", "rolling", "alias", "unknown", "final", "snapshot"])
def test_target_admission_precedes_calendar_and_outcomes(tmp_path, monkeypatch, kind):
    target = history_event("2024-12-03" if kind == "rolling" else "2024-06-12", protected=kind == "random")
    events = [target]
    if kind == "alias":
        events.append({**target, "event_id": "ambiguous", "session": "BMO"})
    conn, repository, snapshot = commit_history_inputs(tmp_path, events, [outcome_move(target)])
    original = objects.verify_object_path
    blocked = {r.object_ref.object_id for table in ("daily_market", "computed_moves")
               for r in repository.fragment_records(snapshot, table)}
    def verify(store, ref):
        assert ref.object_id not in blocked, "calendar/outcome opened before target admission"
        return original(store, ref)
    monkeypatch.setattr(objects, "verify_object_path", verify)
    try:
        changes = {"unknown": {"event_id": "unknown"}, "final": {"purpose": "final"}}.get(kind, {})
        if kind == "snapshot":
            snapshot = replace(snapshot, calendar_version="wrong")
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED" if kind == "snapshot" else "HOLDOUT_ACCESS_DENIED"):
            _read(repository, snapshot, target, **changes)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("kind,code", [
    ("missing", "CONTRACT_MISMATCH"), ("empty", "CALENDAR_UNAVAILABLE"),
    ("intraday", "CONTRACT_MISMATCH"), ("weekend", "CONTRACT_MISMATCH"),
    ("anchor_gap", "CONTRACT_MISMATCH"), ("interior_gap", "CONTRACT_MISMATCH"),
    ("short", "CONTRACT_MISMATCH"), ("horizon", "CONTRACT_MISMATCH"),
])
def test_calendar_refusals_precede_outcome_access(tmp_path, monkeypatch, kind, code):
    target, prior = history_event("2024-06-12"), history_event("2023-01-03")
    days = list(calendar_sessions().days)
    if kind == "empty":
        days = []
    elif kind == "intraday":
        days[-1] += "T12:00:00"
    elif kind == "weekend":
        days.append("2024-06-08")
    elif kind in ("anchor_gap", "interior_gap"):
        days.remove("2024-06-12" if kind == "anchor_gap" else "2024-06-11")
    elif kind == "short":
        days = days[-2:]
    conn, repository, snapshot = commit_history_inputs(tmp_path, [target, prior], [outcome_move(prior)],
        days=days, include_calendar=kind != "missing")
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with pytest.raises(DataError, match=code):
            _read(repository, snapshot, target, horizon=2 if kind == "horizon" else 1)
    finally:
        conn.close()


def test_mixed_history_objects_refuse_before_first_safe_object_opens(tmp_path, monkeypatch):
    first, second, protected, target = (history_event("2023-01-03"), history_event("2023-03-03"),
        history_event("2023-04-03", protected=True), history_event("2024-06-12"))
    conn, repository, snapshot = commit_history_inputs(tmp_path, [first, second, protected, target],
        [outcome_move(e) for e in (first, second, protected)], mixed_days=("2023-03-03", "2023-04-03"))
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            _read(repository, snapshot, target)
        assert not any(name == "computed_moves" for name, _ in repository.scans)
    finally:
        conn.close()


def test_changed_calendar_bytes_bind_identity_but_unused_market_values_are_not_features(tmp_path):
    target, prior = history_event("2024-06-12"), history_event("2023-01-03")
    results = []
    for marker in (1.0, 900.0):
        conn, repository, snapshot = commit_history_inputs(tmp_path / str(marker), [target, prior], [outcome_move(prior)], marker=marker)
        try:
            results.append(_read(repository, snapshot, target))
        finally:
            conn.close()
    assert results[0]["features"] == results[1]["features"]
    assert results[0]["forecast_cutoff"] == results[1]["forecast_cutoff"]
    assert results[0]["inputs_id"] != results[1]["inputs_id"]
    assert results[0]["calendar_input"]["objects"] != results[1]["calendar_input"]["objects"]


def test_late_storage_failure_propagates_without_partial_result_or_retry(tmp_path, monkeypatch):
    first, second, target = history_event("2023-01-03"), history_event("2023-03-03"), history_event("2024-06-12")
    conn, repository, snapshot = commit_history_inputs(tmp_path, [first, second, target], [outcome_move(first), outcome_move(second)])
    original, calls = repository.scan, []
    def scan(query, *, table_name):
        if table_name == "computed_moves":
            calls.append(query.time_interval.start_inclusive)
            if len(calls) == 2:
                raise fail("OBJECT_CORRUPT", "synthetic late failure")
        yield from original(query, table_name=table_name)
    monkeypatch.setattr(repository, "scan", scan)
    try:
        with pytest.raises(DataError, match="OBJECT_CORRUPT"):
            _read(repository, snapshot, target)
        assert calls == ["2023-01-03", "2023-03-03"]
    finally:
        conn.close()


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "2.0"])
def test_invalid_derived_feature_does_not_reach_a_model(value):
    features = dict.fromkeys(magnitude_features.HISTORY_FEATURE_COLUMNS)
    features["n_prior"] = value
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        magnitude_features._feature_recipe(features)


def test_finite_labels_with_overflowing_aggregate_refuse_through_real_reader(tmp_path):
    first, second, target = history_event("2023-01-03"), history_event("2023-03-03"), history_event("2024-06-12")
    conn, repository, snapshot = commit_history_inputs(tmp_path, [first, second, target],
        [outcome_move(first, 1e308), outcome_move(second, 1e308)])
    try:
        with pytest.warns(RuntimeWarning), pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _read(repository, snapshot, target)
    finally:
        conn.close()
