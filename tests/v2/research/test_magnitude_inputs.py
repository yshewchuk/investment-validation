"""Magnitude contracts over synthetic committed snapshots, with outcome poison guards."""
from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from engine.v2.data.computed_moves_table import COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import DataError, fail
from engine.v2.foundation import content_hash
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research import magnitude_inputs
from tests.v2.research.magnitude_trial_support import (
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)
from tests.v2.research.test_event_history import commit_event_history

MONTH = "2025-01"
CUTOFF = "2024-07-01"


def test_approved_magnitude_recipe_is_explicit_and_fixed():
    assert magnitude_inputs.magnitude_recipe() == {
        "schema_version": "earnings_magnitude.v1", "units": "percentage_points",
        "target": "absolute_earnings_close_to_close_return",
        "price_basis": "split_neutral_non_dividend_adjusted",
        "estimand": "conditional_mean", "primary_loss": "pooled_rmse",
        "secondary_diagnostics": ["mae", "quantile_calibration"],
    }


@pytest.mark.parametrize("session,horizon,expected", [
    ("AMC", 5, "2024-06-05"), ("AMC", 3, "2024-06-07"), ("AMC", 1, "2024-06-11"),
    ("BMO", 5, "2024-06-04"), ("BMO", 3, "2024-06-06"), ("BMO", 1, "2024-06-10"),
])
def test_forecast_ceiling_counts_full_sessions_before_the_preprint_close(session, horizon, expected):
    result = magnitude_inputs.forecast_cutoff("2024-06-12", session, horizon, calendar_sessions())
    assert result["source_session_ceiling"] == expected
    assert result["pre_announcement_close"] == ("2024-06-12" if session == "AMC" else "2024-06-11")
    assert result["horizon_sessions"] == horizon
    assert result["cutoff_id"] == content_hash({k: v for k, v in result.items() if k != "cutoff_id"})
    assert result == magnitude_inputs.forecast_cutoff(date(2024, 6, 12), session, horizon, calendar_sessions())


@pytest.mark.parametrize("start,end,event,session,expected", [
    ("2024-05-20", "2024-05-30", "2024-05-28", "BMO", "2024-05-23"),
    ("2024-05-20", "2024-05-30", "2024-05-28", "AMC", "2024-05-24"),
    ("2024-03-25", "2024-04-02", "2024-04-01", "AMC", "2024-03-28"),
    ("2025-01-06", "2025-01-13", "2025-01-10", "AMC", "2025-01-08"),
])
def test_weekends_holidays_and_one_off_closure_use_native_session_calendar(start, end, event, session, expected):
    result = magnitude_inputs.forecast_cutoff(event, session, 1, calendar_sessions(start, end))
    assert result["source_session_ceiling"] == expected


@pytest.mark.parametrize("changes", [
    {"horizon": True}, {"horizon": 1.0}, {"horizon": 0}, {"horizon": 2}, {"horizon": 7},
    {"session": "UNKNOWN"}, {"session": "amc"}, {"event_date": "20240612"},
    {"event_date": "2024-06-12T00:00:00Z"}, {"event_date": pd.Timestamp("2024-06-12 12:00")},
    {"event_date": "2024-06-01"}, {"event_date": "2024-06-17"}, {"calendar": None},
])
def test_invalid_forecast_request_is_typed(changes):
    args = dict(event_date="2024-06-12", session="AMC", horizon=1, calendar=calendar_sessions())
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        magnitude_inputs.forecast_cutoff(**{**args, **changes})


@pytest.mark.parametrize("kind", ["gap", "projected", "short", "duplicate", "weekend", "observed_missing"])
def test_bad_or_incomplete_forecast_calendar_refuses(kind):
    calendar = calendar_sessions()
    if kind == "gap":
        calendar = replace(calendar, days=tuple(d for d in calendar.days if d != "2024-06-11"))
    elif kind == "projected":
        calendar = replace(calendar, observed_through="2024-06-10")
    elif kind == "short":
        calendar = replace(calendar, days=("2024-06-12", "2024-06-13", "2024-06-14"))
    elif kind == "duplicate":
        calendar = replace(calendar, days=calendar.days + ("2024-06-14",))
    elif kind == "weekend":
        calendar = replace(calendar, days=tuple(sorted((*calendar.days, "2024-06-08"))))
    else:
        calendar = replace(calendar, observed_through="2024-06-20")
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        magnitude_inputs.forecast_cutoff("2024-06-12", "AMC", 1, calendar)


@pytest.mark.parametrize("session,missing_anchor", [("AMC", "2024-06-12"), ("BMO", "2024-06-11")])
def test_missing_actual_preannouncement_close_is_not_silently_shifted(session, missing_anchor):
    calendar = calendar_sessions()
    calendar = replace(calendar, days=tuple(d for d in calendar.days if d != missing_anchor))
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        magnitude_inputs.forecast_cutoff("2024-06-12", session, 1, calendar)


def _load(repository, snapshot, **kwargs):
    return magnitude_inputs.load_magnitude_targets(repository, snapshot,
        **{**dict(as_of_month=MONTH, labels_before=CUTOFF), **kwargs})


def test_real_reader_excludes_union_aliases_and_future_objects_before_magnitude_reads(tmp_path, monkeypatch):
    eligible = [history_event(day) for day in ("2024-01-04", "2024-02-05", "2024-03-05")]
    random = history_event("2024-04-05", protected=True)
    rolling = history_event("2024-12-03")
    future = history_event("2024-07-02")
    same_day = history_event(CUTOFF)
    alias = history_event("2024-05-06")
    sibling = {**alias, "event_id": "ambiguous-sibling", "session": "BMO"}
    protected = [random, rolling, future, same_day, alias]
    events = [*eligible, *protected, sibling]
    moves = [outcome_move(event, value) for event, value in zip(eligible, [-2.5, 0.0, 125.0])]
    moves += [outcome_move(event, None) for event in protected]
    conn, repository, snapshot = commit_event_history(tmp_path, events, moves)
    try:
        poison_outcomes(monkeypatch, repository, snapshot, [e["event_date"].date().isoformat() for e in protected])
        before = conn.total_changes
        result = _load(repository, snapshot)
        assert result.event_id.tolist() == [e["event_id"] for e in eligible]
        assert result.realized_move_pct.tolist() == [-2.5, 0.0, 125.0]
        assert result.absolute_move_pct.tolist() == [2.5, 0.0, 125.0]
        assert result.target_available_on.tolist() == ["2024-01-06", "2024-02-07", "2024-03-07"]
        assert result.attrs["snapshot_id"] == snapshot.snapshot_id
        assert result.attrs["computed_moves_version"] == snapshot.table_versions["computed_moves"].dataset_version_id
        assert result.attrs["holdout_as_of_month"] == MONTH
        assert result.attrs["random_membership_version"] == ExperimentHoldouts.random_version
        assert result.attrs["rolling_membership_version"] == ExperimentHoldouts.rolling_version
        assert result.attrs["labels_before"] == CUTOFF
        assert result.attrs["excluded_future_events"] == sorted([future["event_id"], same_day["event_id"]])
        assert result.attrs["target_recipe"] == magnitude_inputs.magnitude_recipe()
        assert result.attrs["target_recipe_id"] == content_hash(magnitude_inputs.magnitude_recipe())
        assert [r["object_id"] for r in result.attrs["consumed_inputs"]] == result.target_object_id.tolist()
        assert all(r["fragment_id"] and r["content_hash"].startswith("sha256:") for r in result.attrs["consumed_inputs"])
        assert len(result.attrs["holdout_exclusions"]) == 4
        again = _load(repository, snapshot)
        pd.testing.assert_frame_equal(result, again)
        assert result.attrs == again.attrs
        later = _load(repository, snapshot, labels_before="2024-06-30")
        assert result.attrs["inputs_id"] != later.attrs["inputs_id"]
        assert all(q.max_result_rows == 1 for name, q in repository.scans if name == "computed_moves")
        assert conn.total_changes == before
        assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("LEDGER.csv"))
    finally:
        conn.close()


@pytest.mark.parametrize("kind,code", [
    ("random", "HOLDOUT_ACCESS_DENIED"), ("rolling", "HOLDOUT_ACCESS_DENIED"),
    ("alias", "HOLDOUT_ACCESS_DENIED"), ("unknown", "HOLDOUT_ACCESS_DENIED"),
    ("future", "CONTRACT_MISMATCH"), ("same_day", "CONTRACT_MISMATCH"),
])
def test_explicit_inadmissible_id_refuses_before_any_outcome(tmp_path, monkeypatch, kind, code):
    day = {"rolling": "2024-12-03", "future": "2024-07-02", "same_day": CUTOFF}.get(kind, "2024-05-06")
    target = history_event(day, protected=kind == "random")
    events = [target]
    if kind == "alias":
        events.append({**target, "event_id": "alias", "session": "BMO"})
    conn, repository, snapshot = commit_event_history(tmp_path, events, [outcome_move(target)])
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        key = "unknown" if kind == "unknown" else target["event_id"]
        with pytest.raises(DataError, match=code):
            _load(repository, snapshot, event_ids=[key])
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("missing", [False, True])
def test_all_objects_preflight_before_first_safe_target_or_mixed_sibling_opens(tmp_path, monkeypatch, missing):
    first, later, protected = history_event("2024-01-04"), history_event("2024-03-05"), history_event("2024-04-05", protected=True)
    moves = [outcome_move(first), outcome_move(protected)] + ([] if missing else [outcome_move(later)])
    conn, repository, snapshot = commit_event_history(tmp_path, [first, later, protected], moves,
                                        mixed_days=("2024-03-05", "2024-04-05"))
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            _load(repository, snapshot)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("value,changes", [
    (None, {}), (float("nan"), {}),
    (2.0, {"skipped": True}), (2.0, {"available_as_of_date": None}),
    (2.0, {"available_as_of_date": "bad"}), (2.0, {"available_as_of_date": "2024-05-06"}),
    (2.0, {"available_as_of_date": CUTOFF}), (2.0, {"available_as_of_date": "2024-07-02"}),
])
def test_invalid_or_not_strictly_available_label_refuses_whole_read(tmp_path, value, changes):
    target = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [target], [outcome_move(target, value, **changes)])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _load(repository, snapshot)
    finally:
        conn.close()


def test_availability_boundary_is_strict_and_precedes_numeric_validation(tmp_path, monkeypatch):
    target = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [target], [outcome_move(target, None, available_as_of_date=CUTOFF)])
    def forbidden(*args):
        pytest.fail("unavailable numeric label must not be parsed")
    monkeypatch.setattr(magnitude_inputs, "_parse_target", forbidden)
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _load(repository, snapshot)
    finally:
        conn.close()


@pytest.mark.parametrize("value", [True, "2.0", float("inf"), -float("inf")])
def test_non_numeric_and_boolean_values_are_not_coerced(value):
    event = pd.Series(history_event("2024-05-06"))
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        magnitude_inputs._label_before([outcome_move(event, value)], event, date(2024, 7, 1))


def test_last_day_before_label_cutoff_is_admitted_and_recipe_is_not_shared_mutable_state(tmp_path):
    target = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [target], [outcome_move(target, -3.0, available_as_of_date="2024-06-30")])
    try:
        result = _load(repository, snapshot)
        assert result.absolute_move_pct.tolist() == [3.0]
        result.attrs["target_recipe"]["secondary_diagnostics"].append("unregistered")
        assert magnitude_inputs.magnitude_recipe()["secondary_diagnostics"] == ["mae", "quantile_calibration"]
    finally:
        conn.close()


@pytest.mark.parametrize("changes,code", [
    ({"as_of_month": None}, "HOLDOUT_ACCESS_DENIED"), ({"purpose": "final"}, "HOLDOUT_ACCESS_DENIED"),
    ({"labels_before": None}, "CONTRACT_MISMATCH"), ({"labels_before": "20240701"}, "CONTRACT_MISMATCH"),
    ({"labels_before": "2024-07-01T00:00:00Z"}, "CONTRACT_MISMATCH"),
    ({"labels_before": "2024-01-01"}, "POPULATION_COLLAPSED"),
])
def test_invalid_context_refuses_before_outcomes(tmp_path, monkeypatch, changes, code):
    target = history_event("2024-05-06")
    conn, repository, snapshot = commit_event_history(tmp_path, [target], [outcome_move(target)])
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with pytest.raises(DataError, match=code):
            _load(repository, snapshot, **changes)
    finally:
        conn.close()


def test_wrong_snapshot_and_late_repository_failure_never_return_partial_or_retry(tmp_path, monkeypatch):
    first, second = history_event("2024-01-04"), history_event("2024-03-05")
    conn, repository, snapshot = commit_event_history(tmp_path, [first, second], [outcome_move(first), outcome_move(second)])
    try:
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
            _load(repository, replace(snapshot, calendar_version="wrong-calendar"))
        original, calls = repository.scan, []
        def scan(query, *, table_name):
            if table_name == "computed_moves":
                calls.append(query.time_interval.start_inclusive)
                if len(calls) == 2:
                    raise fail("OBJECT_CORRUPT", "synthetic late failure")
            yield from original(query, table_name=table_name)
        monkeypatch.setattr(repository, "scan", scan)
        with pytest.raises(DataError, match="OBJECT_CORRUPT"):
            _load(repository, snapshot)
        assert calls == ["2024-01-04", "2024-03-05"]
    finally:
        conn.close()


def test_v5_skipped_training_label_omits_with_the_writers_reason(tmp_path):
    qualified, no_session, no_prices = (history_event(day) for day in
        ("2024-01-04", "2024-03-05", "2024-05-06"))
    conn, repository, snapshot = commit_history_inputs(tmp_path, [qualified, no_session, no_prices], [
        outcome_move(qualified, -2.0),
        outcome_move(no_session, None, skipped=True, available_as_of_date=None,
                     not_computed_reason="no_session"),
        outcome_move(no_prices, None, skipped=True, available_as_of_date=None,
                     not_computed_reason="no_price_history")],
        contract=COMPUTED_MOVES_V5_CONTRACT, include_calendar=False)
    try:
        rows = magnitude_inputs.qualify_magnitude_training_targets(repository, snapshot,
            as_of_month=MONTH, labels_before=CUTOFF,
            event_ids=[qualified["event_id"], no_session["event_id"], no_prices["event_id"]])
        assert rows.event_id.tolist() == [qualified["event_id"]]
        assert rows.attrs["omitted_labels"] == [
            {"event_id": no_session["event_id"], "reason": "no_session"},
            {"event_id": no_prices["event_id"], "reason": "no_price_history"}]
    finally:
        conn.close()
