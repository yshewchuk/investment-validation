"""Real synthetic snapshots for the prediction/sweep pinned-input prerequisite."""
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from engine.v2.data.computed_moves_table import COMPUTED_MOVES_CONTRACT, COMPUTED_MOVES_V5_CONTRACT
from engine.v2.data.errors import DataError
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.ops.experiment_folds import TrainFoldRule, fit_walk_forward_fold
from engine.v2.research.experiment_population import load_population
from engine.v2.research.prediction_inputs import load_prediction_targets
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.v2.research.magnitude_trial_support import (
    TrackingRepository,
    outcome_move,
)

MONTH = "2025-01"


def _identity(day, *, random=False, prefix="T"):
    for index in range(1000):
        ticker = f"{prefix}{index}"
        key = f"{ticker}_{day}"
        if ExperimentHoldouts.random_membership(key) == random:
            return key, ticker
    raise AssertionError("deterministic fixture identity unavailable")


def prediction_event(day, *, random=False, prefix="T", **overrides):
    key, ticker = _identity(day, random=random, prefix=prefix)
    return dict(event_id=key, ticker=ticker, event_date=pd.Timestamp(day),
                year=int(day[:4]), session="AMC", session_src="orats", src_orats=True,
                src_oquants=False, src_nasdaq=False, src_yfinance=False,
                date_agree=True, date_conflict=False, **overrides)


def commit_prediction_inputs(tmp_path, events, moves=None, *, include_targets=True,
                             contract=COMPUTED_MOVES_CONTRACT):
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn, clock, store = catalog_and_store(tmp_path)
    contracts = {"earnings_events": contract_for("earnings_events")}
    input_rows = {"earnings_events": events}
    if include_targets:
        contracts["computed_moves"] = contract
        input_rows["computed_moves"] = moves if moves is not None else [outcome_move(e) for e in events]
    tables = {}
    for table, contract in contracts.items():
        grouped = {}
        for row in input_rows[table]:
            key = str(row[contract.partition_columns[0]])
            grouped.setdefault(key, []).append(row)
        tables[table] = [publish_and_inspect(store, contract, contract_ref_for(contract),
            sorted(rows, key=lambda row: tuple(row[name] for name in contract.primary_key)), key)
            for key, rows in sorted(grouped.items())]
    snapshot = commit_tables(conn, clock, tables, contracts, store=store)
    return conn, TrackingRepository(conn, store), snapshot


def _targets(repository, snapshot, **kwargs):
    return load_prediction_targets(repository, snapshot, as_of_month=MONTH, **kwargs)


def test_real_snapshot_targets_exclude_union_before_outcome_scans(tmp_path):
    eligible = prediction_event("2024-05-02")
    random = prediction_event("2024-05-03", random=True)
    rolling = prediction_event("2024-12-03")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [eligible, random, rolling], [
        outcome_move(eligible, -2.5), outcome_move(random, float("nan")), outcome_move(rolling, float("nan")),
    ])
    try:
        before = conn.total_changes
        rows = _targets(repository, snapshot)
        assert rows["event_id"].tolist() == [eligible["event_id"]]
        assert rows["realized_move_pct"].tolist() == [-2.5]
        assert rows["positive_move"].tolist() == [0]
        assert rows["target_available_on"].tolist() == ["2024-05-04"]
        assert rows["snapshot_id"].tolist() == [snapshot.snapshot_id]
        assert rows["holdout_as_of_month"].tolist() == [MONTH]
        assert rows["population_use"].tolist() == ["post-release selection"]
        assert {item["event_id"] for item in rows.attrs["holdout_exclusions"]} == {
            random["event_id"], rolling["event_id"]}
        scans = [(name, query) for name, query in repository.scans if name == "computed_moves"]
        assert len(scans) == 1
        assert scans[0][1].key_filter[0].values == (eligible["ticker"],)
        assert scans[0][1].time_interval.start_inclusive == "2024-05-02"
        assert scans[0][1].time_interval.end_exclusive == "2024-05-03"
        assert all(query.snapshot_id == snapshot.snapshot_id for _, query in repository.scans)
        pd.testing.assert_frame_equal(rows, _targets(repository, snapshot))
        assert conn.total_changes == before
        assert not list(tmp_path.rglob("REPORT.md"))
        assert not list(tmp_path.rglob("LEDGER.csv"))
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["random", "rolling", "missing", "empty", "ambiguous"])
def test_explicit_population_refuses_before_any_target_scan(tmp_path, kind):
    event = prediction_event("2024-12-03" if kind == "rolling" else "2024-05-02",
                   random=kind == "random")
    if kind == "ambiguous":
        event["date_conflict"] = True
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event])
    ids = [] if kind == "empty" else ["absent" if kind == "missing" else event["event_id"]]
    try:
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            _targets(repository, snapshot, event_ids=ids)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


@pytest.mark.parametrize("kwargs", [
    {"as_of_month": None}, {"as_of_month": "2025-13"}, {"as_of_month": "9999-01"},
    {"purpose": "final"}, {"purpose": "holdout"}, {"event_ids": "bad"},
])
def test_context_refuses_without_scanning(tmp_path, kwargs):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [prediction_event("2024-05-02")])
    try:
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
            load_prediction_targets(repository, snapshot, **{"as_of_month": MONTH, **kwargs})
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("kind", ["missing", "mismatched", "wrong_type"])
def test_unresolved_snapshot_is_typed_without_fallback(tmp_path, kind):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [prediction_event("2024-05-02")])
    changed = {"missing": replace(snapshot, snapshot_id="absent"),
               "mismatched": replace(snapshot, calendar_version="other"), "wrong_type": None}[kind]
    try:
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED") as refusal:
            _targets(repository, changed)
        assert not refusal.value.problem.retryable
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("mutation", [
    {"realized_move_pct": None}, {"realized_move_pct": float("nan")},
    {"skipped": True},
    {"available_as_of_date": None}, {"available_as_of_date": "not-a-date"},
    {"available_as_of_date": "2024-05-02"}, {"available_as_of_date": "2024-05-01"},
    {"available_as_of_date": "20240504"},
])
def test_invalid_target_or_availability_refuses_whole_population(tmp_path, mutation):
    first, later = prediction_event("2024-05-01"), prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [first, later], [outcome_move(first), outcome_move(later, **mutation)])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _targets(repository, snapshot)
        assert not list(tmp_path.rglob("REPORT.md"))
    finally:
        conn.close()


def test_missing_target_and_missing_table_remain_distinct(tmp_path):
    event = prediction_event("2024-05-02")
    for with_table, code in [(True, "EXPERIMENT_VARIANT_FAILED"), (False, "CONTRACT_MISMATCH")]:
        conn, repository, snapshot = commit_prediction_inputs(tmp_path / str(with_table), [event], [],
                                             include_targets=with_table)
        try:
            with pytest.raises(DataError, match=code):
                _targets(repository, snapshot)
        finally:
            conn.close()


def test_shared_population_needs_no_target_table(tmp_path):
    event = prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event], include_targets=False)
    try:
        rows = load_population(repository, snapshot, as_of_month=MONTH, purpose="sweep")
        assert rows["session"].tolist() == ["AMC"]
        assert rows["ticker"].tolist() == [event["ticker"]]
        assert rows["event_date"].tolist() == [pd.Timestamp("2024-05-02")]
        assert set(name for name, _ in repository.scans) == {"earnings_events"}
    finally:
        conn.close()


def test_target_availability_can_feed_existing_fold_helper(tmp_path):
    events = [prediction_event(day) for day in ["2023-01-02", "2023-04-03", "2023-10-02", "2024-01-02"]]
    moves = [outcome_move(event, value) for event, value in zip(events, [-2, 2, -1, 1])]
    moves[2]["available_as_of_date"] = "2024-02-01"  # An older event whose label is not yet known.
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, events, moves)
    try:
        rows = _targets(repository, snapshot).sort_values("event_date")
        cutoff = date(2024, 1, 1)
        train = rows[pd.to_datetime(rows.target_available_on).dt.date < cutoff]
        test = rows[rows.event_date >= pd.Timestamp(cutoff)]
        assert train.event_id.tolist() == [events[0]["event_id"], events[1]["event_id"]]
        # Synthetic, answer-free feature values exercise the already-shared fit seam;
        # constructing real entry-relative features is deliberately a later slice.
        fit = fit_walk_forward_fold(LogisticRegression(random_state=7), [[-1], [1]],
                                    train.positive_move, [[0]], TrainFoldRule())
        assert len(fit.test_scores) == len(test) == 1
        assert np.isfinite(fit.test_scores).all()
        assert test.positive_move.tolist() == [1]
    finally:
        conn.close()


@pytest.mark.parametrize("explicit", [True, False])
def test_shared_outcome_key_with_excluded_sibling_never_scans_targets(tmp_path, explicit):
    event = prediction_event("2024-05-02")
    sibling = {**event, "event_id": _identity("2024-05-02", random=True)[0], "session": "BMO"}
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event, sibling], [outcome_move(event)])
    try:
        with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED") as refusal:
            _targets(repository, snapshot, event_ids=[event["event_id"]] if explicit else None)
        assert "ambiguous" in refusal.value.problem.details["holdout_exclusions"][0]["memberships"]
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


def test_late_target_scan_failure_returns_no_partial_frame(tmp_path, monkeypatch):
    first, second = prediction_event("2024-05-01"), prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [first, second])
    original = repository.scan
    calls = 0

    def late_failure(request, *, table_name):
        nonlocal calls
        if table_name == "computed_moves":
            calls += 1
        yield from original(request, table_name=table_name)
        if table_name == "computed_moves" and calls == 2:
            from engine.v2.data import errors
            raise errors.fail("OBJECT_CORRUPT", "synthetic late scan failure")

    monkeypatch.setattr(repository, "scan", late_failure)
    try:
        with pytest.raises(DataError, match="OBJECT_CORRUPT"):
            _targets(repository, snapshot)
        assert calls == 2
        assert not list(tmp_path.rglob("REPORT.md"))
    finally:
        conn.close()


@pytest.mark.parametrize("invalid_id", [[], None, "", " "])
def test_malformed_snapshot_id_is_typed(tmp_path, invalid_id):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [prediction_event("2024-05-02")])
    try:
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
            _targets(repository, replace(snapshot, snapshot_id=invalid_id))
        assert repository.scans == []
    finally:
        conn.close()


@pytest.mark.parametrize("evidence", ["{}", "not-json", "[]", "null"])
def test_corrupt_catalog_evidence_is_typed_without_scan(tmp_path, evidence):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [prediction_event("2024-05-02")])
    try:
        conn.execute("DROP TRIGGER data_dataset_versions_no_update")
        conn.execute("PRAGMA ignore_check_constraints=ON")  # Simulate corrupt bytes in this test-only catalog.
        conn.execute("UPDATE data_dataset_versions SET evidence_json=?", (evidence,))
        with pytest.raises(DataError, match="SNAPSHOT_UNRESOLVED"):
            _targets(repository, snapshot)
        assert repository.scans == []
    finally:
        conn.close()


def test_non_date_outcome_key_is_never_coerced_to_canonical_event(tmp_path):
    event = prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event], [
        outcome_move(event, event_date="2024-05-02T13:00:00")])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
            _targets(repository, snapshot)
    finally:
        conn.close()


def test_empty_bulk_calendar_is_not_a_success(tmp_path):
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [], [], include_targets=False)
    try:
        with pytest.raises(DataError, match="POPULATION_COLLAPSED"):
            _targets(repository, snapshot)
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()


def test_v5_skipped_target_refuses_naming_the_writers_reason(tmp_path):
    event = prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event], [
        outcome_move(event, None, skipped=True, available_as_of_date=None,
                     not_computed_reason="not_orats")], contract=COMPUTED_MOVES_V5_CONTRACT)
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED") as refusal:
            _targets(repository, snapshot)
        assert refusal.value.problem.message == "prediction target was not computed: not_orats"
    finally:
        conn.close()


def test_v4_skipped_target_keeps_the_generic_refusal(tmp_path):
    event = prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event], [
        outcome_move(event, None, skipped=True, available_as_of_date=None,
                     not_computed_reason="not_orats")])
    try:
        with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED") as refusal:
            _targets(repository, snapshot)
        assert refusal.value.problem.message == "prediction target or availability is invalid"
    finally:
        conn.close()


def test_foreign_computed_moves_contract_refuses_before_any_target_scan(tmp_path):
    from engine.v2.data.manifests import table_contract_hash
    foreign = replace(COMPUTED_MOVES_CONTRACT, contract_id="computed_moves.v0", semantic_version="0.0.0")
    foreign = replace(foreign, definition_hash=table_contract_hash(foreign))
    event = prediction_event("2024-05-02")
    conn, repository, snapshot = commit_prediction_inputs(tmp_path, [event], [
        outcome_move(event, None, skipped=True, available_as_of_date=None)], contract=foreign)
    try:
        with pytest.raises(DataError, match="CONTRACT_MISMATCH") as refusal:
            _targets(repository, snapshot)
        assert refusal.value.problem.message == "prediction targets require native computed moves"
        assert all(name == "earnings_events" for name, _ in repository.scans)
    finally:
        conn.close()
