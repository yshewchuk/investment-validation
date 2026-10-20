"""Protected-object poison tests over committed synthetic native snapshots."""
from dataclasses import replace

import pandas as pd
import pytest

from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import DataError, fail
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research.event_history import load_event_history
from engine.v2.research.prediction_inputs import _target_query, load_prediction_targets
from tests.v2.research.magnitude_trial_support import (
    TrackingRepository,
    commit_history_inputs,
    history_event,
    outcome_move,
    poison_outcomes,
)

MONTH = "2025-01"
DECISION = "2024-06-19"
START = "2023-01-01"


@pytest.mark.parametrize("protected", [False, True])
def test_history_event_overrides_defaults_without_changing_other_fields(protected):
    original = history_event("2024-01-02", protected=protected, ticker="OVERRIDE")
    overrides = {"session": "BMO", "src_orats": False, "date_conflict": True, "extra": "retained"}
    changed = history_event("2024-01-02", protected=protected, ticker="OVERRIDE", **overrides)
    assert changed == {**original, **overrides}
    assert original["session"] == "AMC"
    assert original["src_orats"] is True
    assert original["date_conflict"] is False
    assert ExperimentHoldouts.random_membership(changed["event_id"]) is protected


def commit_event_history(tmp_path, events, moves, *, mixed_days=()):
    return commit_history_inputs(tmp_path, events, moves, mixed_days=mixed_days, include_calendar=False)


def read_event_history(repository, snapshot, target, **kwargs):
    return load_event_history(repository, snapshot, **dict(event_id=target["event_id"],
        as_of_month=MONTH, history_start=START, decision_session=DECISION, **kwargs))


def test_multiple_histories_union_alias_cutoff_provenance_and_determinism(tmp_path, monkeypatch):
    prior = [history_event(day) for day in ("2023-01-02", "2023-06-02", "2023-12-04")]
    random = history_event("2023-02-02", protected=True)
    alias = history_event("2023-03-02")
    sibling = {**alias, "event_id": history_event("2023-03-02", protected=True)["event_id"], "session": "BMO"}
    skipped, delayed = history_event("2023-04-03"), history_event("2023-05-02")
    same_day, future = history_event(DECISION), history_event("2024-06-21")
    rolling, target = history_event("2024-12-03"), history_event("2024-06-20")
    events = [*prior, random, alias, sibling, skipped, delayed, same_day, future, rolling, target]
    moves = [outcome_move(event, value) for event, value in zip(prior, [-2, 4, 6])]
    moves += [outcome_move(event, None) for event in (random, alias, same_day, future, rolling, target)]
    moves += [outcome_move(skipped, None, skipped=True, available_as_of_date=None),
              outcome_move(delayed, None, available_as_of_date="2024-06-21")]
    # Inclusive availability: this oldest label becomes known exactly on the cutoff.
    moves[0]["available_as_of_date"] = DECISION
    conn, repository, snapshot = commit_event_history(tmp_path, events, moves)
    try:
        protected_days = [e["event_date"].date().isoformat() for e in (random, alias, same_day, future, rolling, target)]
        poison_outcomes(monkeypatch, repository, snapshot, protected_days)
        before = conn.total_changes
        result = read_event_history(repository, snapshot, target)
        assert result == read_event_history(repository, snapshot, target)
        assert result["features"]["n_prior"] == 3
        assert result["features"]["mean_prior_move"] == pytest.approx(8 / 3)
        assert result["features"]["mean_prior_abs_move"] == 4
        assert result["features"]["ema2_prior_move"] == pytest.approx(14 / 3)
        assert result["features"]["ema2_prior_abs_move"] == pytest.approx(46 / 9)
        assert result["features"]["ema4_prior_move"] is None
        assert result["history_status"] == "available"
        assert result["schema_version"] == "research_event_history.v2"
        assert [row["event_id"] for row in result["read_inputs"]] == [
            e["event_id"] for e in (prior[0], skipped, delayed, prior[1], prior[2])]
        assert all(row["fragment_id"] and row["content_hash"].startswith("sha256:")
                   for row in result["read_inputs"])
        assert [row["event_id"] for row in result["history"]] == [e["event_id"] for e in prior]
        assert result["history"][0]["target_available_on"] == DECISION
        assert all(row["fragment_id"] and row["content_hash"].startswith("sha256:") for row in result["history"])
        assert result["omitted_history"] == [{"event_id": skipped["event_id"], "reason": "skipped"},
                                              {"event_id": delayed["event_id"], "reason": "unavailable"}]
        assert result["snapshot_id"] == snapshot.snapshot_id
        assert result["computed_moves_version"] == snapshot.table_versions["computed_moves"].dataset_version_id
        assert result["holdout_as_of_month"] == MONTH
        assert result["random_membership_version"] == ExperimentHoldouts.random_version
        assert result["rolling_membership_version"] == ExperimentHoldouts.rolling_version
        assert result["decision_session"] == DECISION
        assert result["history_start"] == START
        excluded = {r["event_id"]: r["memberships"] for r in result["holdout_exclusions"]}
        assert "random" in excluded[random["event_id"]]
        assert "rolling" in excluded[rolling["event_id"]]
        assert "ambiguous" in excluded[alias["event_id"]]
        scans = [q for name, q in repository.scans if name == "computed_moves"]
        assert len(scans) == 10
        assert all(q.snapshot_id == snapshot.snapshot_id and q.max_result_rows == 1 for q in scans)
        assert conn.total_changes == before
        assert not list(tmp_path.rglob("REPORT.md")) and not list(tmp_path.rglob("LEDGER.csv"))
    finally:
        conn.close()


def test_changed_cutoff_or_window_changes_consumed_history_and_digest(tmp_path):
    target = history_event("2024-06-20")
    first, second = history_event("2023-01-02"), history_event("2023-02-02")
    conn, repository, snapshot = commit_event_history(tmp_path, [target, first, second], [
        outcome_move(first, -2), outcome_move(second, 8, available_as_of_date=DECISION)])
    try:
        full = read_event_history(repository, snapshot, target)
        earlier = load_event_history(repository, snapshot, event_id=target["event_id"],
            as_of_month=MONTH, history_start=START, decision_session="2024-06-18")
        shorter = load_event_history(repository, snapshot, event_id=target["event_id"],
            as_of_month=MONTH, history_start="2023-02-01", decision_session=DECISION)
        assert [r["features"]["mean_prior_move"] for r in (full, earlier, shorter)] == [3, -2, 8]
        assert len({r["history_id"] for r in (full, earlier, shorter)}) == 3
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["random", "rolling", "alias", "unknown"])
def test_explicit_protected_target_refuses_before_outcome_access(tmp_path, monkeypatch, kind):
    target = history_event("2024-12-03" if kind == "rolling" else "2024-06-20", protected=kind == "random")
    prior = history_event("2023-01-02")
    events = [prior, target]
    if kind == "alias":
        events.append({**target, "event_id": "conflicting-sibling", "session": "BMO"})
    conn, repository, snapshot = commit_event_history(tmp_path, events, [outcome_move(prior), outcome_move(target)])
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        if kind == "unknown":
            target = {**target, "event_id": "missing"}
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            read_event_history(repository, snapshot, target)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["mixed", "missing", "noncanonical_key"])
def test_every_fragment_is_preflighted_before_even_an_earlier_safe_scan(tmp_path, monkeypatch, kind):
    first, second, protected, target = (history_event("2023-01-02"), history_event("2023-03-02"),
        history_event("2023-04-03", protected=True), history_event("2024-06-20"))
    moves = [outcome_move(first)]
    if kind != "missing":
        moves.append(outcome_move(second, event_date="2023-03-02T13:00:00" if kind == "noncanonical_key" else "2023-03-02"))
    moves.append(outcome_move(protected))
    conn, repository, snapshot = commit_event_history(tmp_path, [first, second, protected, target], moves,
        mixed_days=("2023-03-02", "2023-04-03") if kind == "mixed" else ())
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            read_event_history(repository, snapshot, target)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["no_prior", "protected_only", "skipped", "null_availability", "delayed"])
def test_empty_or_unavailable_history_is_truthful_not_zero_imputed(tmp_path, kind):
    target, prior = history_event("2024-06-20"), history_event("2023-01-02", protected=kind == "protected_only")
    mutation = {"skipped": dict(skipped=True, available_as_of_date=None),
                "null_availability": dict(available_as_of_date=None),
                "delayed": dict(available_as_of_date="2024-06-21")}.get(kind, {})
    events = [target] if kind == "no_prior" else [target, prior]
    moves = [] if kind == "no_prior" else [outcome_move(prior, None, **mutation)]
    conn, repository, snapshot = commit_event_history(tmp_path, events, moves)
    try:
        result = read_event_history(repository, snapshot, target)
        assert result["history_status"] == "empty"
        assert result["history"] == []
        assert result["features"]["n_prior"] == 0
        assert all(value is None for name, value in result["features"].items() if name != "n_prior")
        assert len(result["omitted_history"]) == (0 if kind in ("no_prior", "protected_only") else 1)
    finally:
        conn.close()


@pytest.mark.parametrize("availability", ["bad", "20230104", "2023-01-02", "2023-01-01", "2023-01-04T00:00:00Z"])
def test_malformed_availability_refuses_without_a_partial_result(tmp_path, availability):
    target, prior = history_event("2024-06-20"), history_event("2023-01-02")
    conn, repository, snapshot = commit_event_history(tmp_path, [target, prior], [outcome_move(prior, available_as_of_date=availability)])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            read_event_history(repository, snapshot, target)
    finally:
        conn.close()


def test_unusable_available_numeric_target_refuses(tmp_path):
    target, prior = history_event("2024-06-20"), history_event("2023-01-02")
    conn, repository, snapshot = commit_event_history(tmp_path, [target, prior], [outcome_move(prior, None)])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            read_event_history(repository, snapshot, target)
    finally:
        conn.close()


@pytest.mark.parametrize("changes, code", [
    ({"as_of_month": None}, "HOLDOUT_ACCESS_DENIED"),
    ({"purpose": "final"}, "HOLDOUT_ACCESS_DENIED"),
    ({"decision_session": "2024-06-21"}, "CONTRACT_MISMATCH"),
    ({"history_start": "2024-06-20"}, "CONTRACT_MISMATCH"),
    ({"decision_session": "2024-06-19T12:00:00"}, "CONTRACT_MISMATCH"),
    ({"history_start": None}, "CONTRACT_MISMATCH"),
])
def test_invalid_context_refuses_before_outcome_access(tmp_path, monkeypatch, changes, code):
    target, prior = history_event("2024-06-20"), history_event("2023-01-02")
    conn, repository, snapshot = commit_event_history(tmp_path, [target, prior], [outcome_move(prior)])
    try:
        poison_outcomes(monkeypatch, repository, snapshot, (), all_outcomes=True)
        args = dict(event_id=target["event_id"], as_of_month=MONTH,
                    history_start=START, decision_session=DECISION)
        with pytest.raises(DataError, match=code):
            load_event_history(repository, snapshot, **{**args, **changes})
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


def test_exact_snapshot_is_required_and_late_failure_is_not_retried(tmp_path, monkeypatch):
    first, second, target = history_event("2023-01-02"), history_event("2023-02-02"), history_event("2024-06-20")
    conn, repository, snapshot = commit_event_history(tmp_path, [first, second, target], [outcome_move(first), outcome_move(second)])
    try:
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
            read_event_history(repository, replace(snapshot, calendar_version="uncommitted"), target)
        assert repository.scans == []
        original = repository.scan
        calls = []

        def failing_scan(query, *, table_name):
            yield from original(query, table_name=table_name)
            if table_name == "computed_moves":
                calls.append(query)
                if len(calls) == 2:
                    raise fail("OBJECT_CORRUPT", "synthetic late outcome failure")

        monkeypatch.setattr(repository, "scan", failing_scan)
        with pytest.raises(DataError, match="OBJECT_CORRUPT"):
            read_event_history(repository, snapshot, target)
        assert len(calls) == 2
    finally:
        conn.close()


@pytest.mark.parametrize("reader", ["history", "prediction"])
def test_history_reuses_preflight_bound_and_prediction_keeps_lookup(tmp_path, monkeypatch, reader):
    first, second, target = history_event("2023-01-02"), history_event("2023-02-02"), history_event("2024-06-20")
    conn, repository, snapshot = commit_event_history(tmp_path, [first, second, target], [outcome_move(first), outcome_move(second)])
    bounds, metadata = [], []
    original_bound, original_records = repository.scan_population_bound, repository.fragment_records

    def bound(snapshot_id, **kwargs):
        if kwargs["table_name"] == "computed_moves":
            bounds.append(kwargs)
            assert reader == "prediction", "history repeated its planned population lookup"
        return original_bound(snapshot_id, **kwargs)

    def records(snapshot_ref, table_name):
        if table_name == "computed_moves":
            metadata.append(table_name)
        return original_records(snapshot_ref, table_name)

    monkeypatch.setattr(repository, "scan_population_bound", bound)
    monkeypatch.setattr(repository, "fragment_records", records)
    try:
        if reader == "history":
            assert read_event_history(repository, snapshot, target)["features"]["n_prior"] == 2
            assert bounds == [] and metadata == ["computed_moves"]
        else:
            result = load_prediction_targets(repository, snapshot, as_of_month=MONTH,
                event_ids=[first["event_id"], second["event_id"]])
            assert len(result) == len(bounds) == 2
        assert all(q.max_result_rows == 1 for name, q in repository.scans if name == "computed_moves")
    finally:
        conn.close()


@pytest.mark.parametrize("bound", [0, 1, 3])
def test_supplied_target_bound_preserves_query_limits_without_lookup(tmp_path, monkeypatch, bound):
    target = history_event("2024-06-20")
    conn, repository, snapshot = commit_event_history(tmp_path, [target], [outcome_move(target)])

    def forbidden_lookup(*args, **kwargs):
        pytest.fail("supplied population bound must not trigger another lookup")

    monkeypatch.setattr(repository, "scan_population_bound", forbidden_lookup)
    try:
        query = _target_query(repository, snapshot, COMPUTED_MOVES_CONTRACT,
                              pd.Series(target), population_bound=bound)
        assert query.max_result_rows == min(bound, 2)
        assert query.max_batch_rows == max(1, min(bound, 2))
        assert repository.scans == []
    finally:
        conn.close()


def test_v5_history_omissions_report_the_writers_reason(tmp_path):
    target = history_event("2024-06-20")
    orats, few_events, nameless = (history_event(day) for day in
        ("2023-01-02", "2023-02-02", "2023-03-02"))
    conn, repository, snapshot = commit_history_inputs(tmp_path, [target, orats, few_events, nameless], [
        outcome_move(orats, None, skipped=True, available_as_of_date=None,
                     not_computed_reason="not_orats"),
        outcome_move(few_events, None, skipped=True, available_as_of_date=None,
                     not_computed_reason="ticker_too_few_events"),
        outcome_move(nameless, None, skipped=True, available_as_of_date=None),
        outcome_move(target, None)],
        contract=COMPUTED_MOVES_V5_CONTRACT, include_calendar=False)
    try:
        result = read_event_history(repository, snapshot, target)
        assert result["omitted_history"] == [
            {"event_id": orats["event_id"], "reason": "not_orats"},
            {"event_id": few_events["event_id"], "reason": "ticker_too_few_events"},
            {"event_id": nameless["event_id"], "reason": "skipped"}]
    finally:
        conn.close()


def test_v5_candidate_without_a_fragment_still_refuses_holdout_access(tmp_path):
    prior, target = history_event("2023-01-02"), history_event("2024-06-20")
    conn, repository, snapshot = commit_history_inputs(tmp_path, [prior, target], [],
        contract=COMPUTED_MOVES_V5_CONTRACT, include_calendar=False)
    try:
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            read_event_history(repository, snapshot, target)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()
