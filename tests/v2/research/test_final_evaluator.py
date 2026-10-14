"""Private frozen single-winner final evaluator over real synthetic snapshots.

One evaluated population at a time: the random and rolling held-out sets are
scored separately from the committed winner rows and never pooled, averaged or
re-read through any other variant or provenance.
"""
import inspect
from dataclasses import fields, replace

import pandas as pd
import pytest

from engine.v2.contracts import FinalHoldoutDeclaration, FinalHoldoutRequest
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import artifact_reference
from engine.v2.foundation.experiment_holdouts import ExperimentHoldouts
from engine.v2.research import _final_evaluator
from engine.v2.research._final_evaluator import (
    FinalMetrics,
    SetMetrics,
    evaluate_final_holdout,
)
from engine.v2.research._final_population import FinalPopulation, load_final_population
from engine.v2.research.experiment_population import load_population
from engine.v2.research.experiment_trades import PROVENANCE
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)
from tests.test_v2_research_build_trades import _trade_row
from tests.test_v2_research_replay import _event_rows
from tests.v2.research.test_experiment_trades import (
    _AMBIGUOUS,
    _OVERLAP,
    _RANDOM,
    _ROLLING,
    _SAFE,
    _holdout_snapshot,
)


def _ref(label, schema):
    return artifact_reference(label.encode(), schema)


#: The as-of month the golden members are labelled under (test_experiment_trades).
_AS_OF = "2024-10"

#: The frozen winner variant: the registration hash `_request` binds.
_WINNER_VARIANT = _ref("registration", "native_experiment_registration.v1.1").content_hash


def _request(snapshot, **overrides):
    registration = _ref("registration", "native_experiment_registration.v1.1")
    value = FinalHoldoutRequest(
        experiment_id="synthetic-experiment", registration_run_id="synthetic-run",
        variant_id=registration.content_hash, registration_ref=registration,
        snapshot_id=snapshot.snapshot_id, holdout_as_of_month=_AS_OF,
        random_membership_version=ExperimentHoldouts.random_version,
        rolling_membership_version=ExperimentHoldouts.rolling_version,
        execution_plan_hash=_ref("plan", "x.v1.0").content_hash,
        source_closure_hash=_ref("sources", "x.v1.0").content_hash,
        final_read=FinalHoldoutDeclaration(
            evaluation_recipe_ref=_ref("recipe", "final_holdout_recipe.v1.0"),
            completeness_bar_ref=_ref("bar", "final_holdout_completeness.v1.0"),
            frozen_model_refs=(_ref("model", "synthetic_model.v1.0"),)))
    return replace(value, **overrides)


def _winner_trade(event_id, date, ret, *, trade_id=None, variant=None, provenance=PROVENANCE,
                  ticker=None):
    row = {**_trade_row(trade_id or f"W-{event_id}", "STR-THRU", provenance),
           "event_id": event_id, "event_date": pd.Timestamp(date),
           "variant": _WINNER_VARIANT if variant is None else variant, "ret": ret}
    if ticker is not None:
        row["ticker"] = ticker
    return row


def _evaluate(repository, request):
    population = load_final_population(repository, request)
    return evaluate_final_holdout(repository, request, population)


def test_random_and_rolling_results_stay_separate(tmp_path):
    extra = [
        _winner_trade("EVENT-5", "2024-01-16", 0.5),
        _winner_trade("EVENT-1", "2024-09-15", -0.25),
        _winner_trade("EVENT-30", "2024-09-16", 1.0),
        _winner_trade("EVENT-0", "2024-01-01", 9.0),
    ]
    conn, repository, snapshot = _holdout_snapshot(
        tmp_path, [_RANDOM, _ROLLING, _OVERLAP, _SAFE], extra_trades=extra)
    try:
        result = _evaluate(repository, _request(snapshot))
        assert result.random == SetMetrics(
            n_events=2, n_trades=2, n_without_trade=0, total_ret=1.5,
            mean_ret=0.75, win_rate=1.0)
        assert result.rolling == SetMetrics(
            n_events=2, n_trades=2, n_without_trade=0, total_ret=0.75,
            mean_ret=0.375, win_rate=0.5)
        assert result.overlap_count == 1
        assert result.snapshot_id == snapshot.snapshot_id
        assert result.holdout_as_of_month == _AS_OF
        assert result.variant_id == _WINNER_VARIANT
        assert tuple(field.name for field in fields(FinalMetrics)) == (
            "snapshot_id", "holdout_as_of_month", "variant_id", "random",
            "rolling", "overlap_count", "ambiguous_count")
    finally:
        conn.close()


def test_ambiguous_event_enters_neither_set_and_is_counted(tmp_path):
    extra = [
        _winner_trade("EVENT-5", "2024-01-16", 0.0),
        _winner_trade("EVENT-1", "2024-09-15", 0.5),
        _winner_trade("EVENT-2", "2024-01-15", 7.0),
    ]
    conn, repository, snapshot = _holdout_snapshot(
        tmp_path, [_RANDOM, _ROLLING, _AMBIGUOUS],
        event_overrides={"EVENT-2": {"date_conflict": True}}, extra_trades=extra)
    try:
        result = _evaluate(repository, _request(snapshot))
        assert result.ambiguous_count == 1
        assert result.random == SetMetrics(
            n_events=1, n_trades=1, n_without_trade=0, total_ret=0.0,
            mean_ret=0.0, win_rate=0.0)
        assert result.rolling == SetMetrics(
            n_events=1, n_trades=1, n_without_trade=0, total_ret=0.5,
            mean_ret=0.5, win_rate=1.0)
    finally:
        conn.close()


def test_other_variants_and_provenance_are_never_read(tmp_path):
    extra = [
        _winner_trade("EVENT-5", "2024-01-16", 0.5),
        _winner_trade("EVENT-5", "2024-01-16", 50.0,
                      trade_id="W-other", variant="other.v1"),
        _winner_trade("EVENT-5", "2024-01-16", 70.0,
                      trade_id="W-replay", provenance="engine.replay"),
    ]
    conn, repository, snapshot = _holdout_snapshot(
        tmp_path, [_SAFE, _RANDOM, _ROLLING, _OVERLAP], extra_trades=extra)
    try:
        result = _evaluate(repository, _request(snapshot))
        assert result.random == SetMetrics(
            n_events=2, n_trades=1, n_without_trade=1, total_ret=0.5,
            mean_ret=0.5, win_rate=1.0)
        assert result.rolling.n_trades == 0
    finally:
        conn.close()


def test_unscored_event_counts_without_trade(tmp_path):
    extra = [_winner_trade("EVENT-5", "2024-01-16", 0.5)]
    conn, repository, snapshot = _holdout_snapshot(
        tmp_path, [_RANDOM, _OVERLAP, _ROLLING], extra_trades=extra)
    try:
        result = _evaluate(repository, _request(snapshot))
        assert result.random.n_events == 2
        assert result.random.n_trades == 1
        assert result.random.n_without_trade == 1
        assert result.rolling == SetMetrics(
            n_events=2, n_trades=0, n_without_trade=2, total_ret=0.0,
            mean_ret=None, win_rate=None)
    finally:
        conn.close()


def test_multiple_winner_trades_on_one_event_count_once_as_event(tmp_path):
    extra = [
        _winner_trade("EVENT-5", "2024-01-16", 0.5, trade_id="W-5a"),
        _winner_trade("EVENT-5", "2024-01-16", 0.25, trade_id="W-5b"),
    ]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        result = _evaluate(repository, _request(snapshot))
        assert result.random.n_events == 1
        assert result.random.n_trades == 2
        assert result.random.n_without_trade == 0
        assert result.random.total_ret == 0.75
        assert result.random.mean_ret == 0.375
    finally:
        conn.close()


def test_no_winner_trade_in_either_set_collapses(tmp_path):
    extra = [_winner_trade("EVENT-0", "2024-01-01", 1.0)]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_SAFE, _RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, request, population)
        assert refusal.value.problem.code == "POPULATION_COLLAPSED"
        assert refusal.value.problem.details == {
            "snapshot_id": snapshot.snapshot_id, "variant_id": _WINNER_VARIANT}
    finally:
        conn.close()


def test_non_finite_ret_is_a_contract_mismatch(tmp_path):
    """NaN is the only non-finite ret that can reach the evaluator.

    Infinities are rejected earlier, at publication: the trades table cannot
    hold them, so only NaN survives to this finite check.
    """
    extra = [_winner_trade("EVENT-5", "2024-01-16", float("nan"))]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, request, population)
        assert refusal.value.problem.code == "CONTRACT_MISMATCH"
    finally:
        conn.close()


def test_stale_trade_date_against_canonical_event_is_a_contract_mismatch(tmp_path):
    """A calendar correction keeps the event_id but changes the canonical date."""
    extra = [_winner_trade("EVENT-5", "2024-02-20", 0.5)]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, request, population)
        assert refusal.value.problem.code == "CONTRACT_MISMATCH"
        assert refusal.value.problem.details == {
            "snapshot_id": snapshot.snapshot_id, "event_ids": ["EVENT-5"]}
    finally:
        conn.close()


def test_winner_ticker_mismatch_against_canonical_event_is_a_contract_mismatch(tmp_path):
    extra = [_winner_trade("EVENT-5", "2024-01-16", 0.5, ticker="ZZZZ")]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, request, population)
        assert refusal.value.problem.code == "CONTRACT_MISMATCH"
        assert refusal.value.problem.details == {
            "snapshot_id": snapshot.snapshot_id, "event_ids": ["EVENT-5"]}
    finally:
        conn.close()


def test_infinite_ret_cannot_enter_a_snapshot(tmp_path):
    with pytest.raises(DataError) as refusal:
        _holdout_snapshot(
            tmp_path, [_RANDOM],
            extra_trades=[_winner_trade("EVENT-5", "2024-01-16", float("inf"))])
    assert refusal.value.problem.code == "CONTRACT_MISMATCH"


def test_hand_reconstructed_equal_population_is_accepted(tmp_path):
    extra = [_winner_trade("EVENT-5", "2024-01-16", 0.5)]
    conn, repository, snapshot = _holdout_snapshot(
        tmp_path, [_RANDOM, _ROLLING, _OVERLAP], extra_trades=extra)
    try:
        request = _request(snapshot)
        derived = load_final_population(repository, request)
        reconstructed = FinalPopulation(
            **{field.name: getattr(derived, field.name) for field in fields(derived)})
        assert reconstructed == derived
        assert reconstructed is not derived
        assert (evaluate_final_holdout(repository, request, reconstructed)
                == evaluate_final_holdout(repository, request, derived))
    finally:
        conn.close()


def test_empty_trades_table_converts_to_population_collapsed(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    trades_contract = contract_for("trades")
    events_contract = contract_for("earnings_events")
    events_record = publish_and_inspect(
        store, events_contract, contract_ref_for(events_contract), _event_rows(), "2024")
    snapshot = commit_tables(
        conn, clock, {"trades": [], "earnings_events": [events_record]},
        {"trades": trades_contract, "earnings_events": events_contract}, store=store)
    repository = Repository(conn, store)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, request, population)
        assert refusal.value.problem.code == "POPULATION_COLLAPSED"
        assert refusal.value.problem.details == {"snapshot_id": snapshot.snapshot_id}
    finally:
        conn.close()


def _denied_population(repository, snapshot, population, case):
    """The non-/mismatched population the evaluator must refuse for ``case``."""
    if case.startswith("frame-"):
        return load_population(
            repository, snapshot, as_of_month=_AS_OF, purpose=case.split("-", 1)[1])
    if case == "list":
        return list(population.random_event_ids)
    if case == "tuple":
        return tuple(population.random_event_ids)
    if case == "none":
        return None
    if case == "forged-random":
        return replace(
            population, random_event_ids=population.random_event_ids + ("EVENT-0",))
    if case == "forged-rolling-empty":
        return replace(population, rolling_event_ids=())
    if case == "forged-overlap":
        return replace(population, overlap_count=population.overlap_count + 1)
    return replace(population, holdout_as_of_month="2000-01")


@pytest.mark.parametrize("case", [
    "list", "tuple", "frame-training", "frame-selection", "frame-sweep", "none",
    "forged-random", "forged-rolling-empty", "forged-overlap", "other-month"])
def test_forged_or_mismatched_population_refuses_before_any_trade_read(
        tmp_path, monkeypatch, case):
    conn, repository, snapshot = _holdout_snapshot(
        tmp_path, [_RANDOM, _ROLLING, _OVERLAP, _SAFE])
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        bad = _denied_population(repository, snapshot, population, case)

        def _no_read(*_args, **_kwargs):
            raise AssertionError("outcomes read")

        monkeypatch.setattr(_final_evaluator, "read_existing_trades", _no_read)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, request, bad)
        assert refusal.value.problem.code == "HOLDOUT_ACCESS_DENIED"
    finally:
        conn.close()


def test_fit_and_alternate_arm_attempts_refuse(tmp_path):
    other_ref = _ref("other-registration", "native_experiment_registration.v1.1")
    other = other_ref.content_hash
    extra = [
        _winner_trade("EVENT-5", "2024-01-16", 0.5),
        _winner_trade("EVENT-5", "2024-01-16", 0.9, trade_id="W-alt", variant=other),
    ]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        attempts = [
            lambda: evaluate_final_holdout(
                repository, request, population, fit=lambda: None),
            lambda: evaluate_final_holdout(repository, request, population, arm="other"),
            lambda: evaluate_final_holdout(repository, request, population, threshold=0.1),
            lambda: evaluate_final_holdout(repository, request, population, callback=print),
            lambda: evaluate_final_holdout(
                repository, request, population, variant_id="x"),
        ]
        for attempt in attempts:
            with pytest.raises(TypeError):
                attempt()
        parameters = inspect.signature(evaluate_final_holdout).parameters
        assert list(parameters) == ["repository", "request", "population"]
        assert all(parameter.kind not in (inspect.Parameter.VAR_KEYWORD,
                                          inspect.Parameter.VAR_POSITIONAL)
                   for parameter in parameters.values())
        alternate = evaluate_final_holdout(
            repository, replace(request, variant_id=other, registration_ref=other_ref),
            population)
        assert alternate.random == SetMetrics(
            n_events=1, n_trades=1, n_without_trade=0, total_ret=0.9,
            mean_ret=0.9, win_rate=1.0)
        winner = evaluate_final_holdout(repository, request, population)
        assert winner.random.total_ret == 0.5
    finally:
        conn.close()


def test_invalid_request_refuses(tmp_path):
    extra = [_winner_trade("EVENT-5", "2024-01-16", 0.5)]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        population = load_final_population(repository, _request(snapshot))
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, object(), population)
        assert refusal.value.problem.code == "HOLDOUT_ACCESS_DENIED"
    finally:
        conn.close()


def test_snapshot_mismatch(tmp_path):
    extra = [_winner_trade("EVENT-5", "2024-01-16", 0.5)]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        with pytest.raises(DataError) as refusal:
            evaluate_final_holdout(repository, replace(request, snapshot_id="snap-nope"),
                                   population)
        assert refusal.value.problem.code == "SNAPSHOT_UNRESOLVED"
    finally:
        conn.close()


def test_does_not_write(tmp_path):
    extra = [_winner_trade("EVENT-5", "2024-01-16", 0.5)]
    conn, repository, snapshot = _holdout_snapshot(tmp_path, [_RANDOM], extra_trades=extra)
    try:
        request = _request(snapshot)
        population = load_final_population(repository, request)
        before = conn.total_changes
        first = evaluate_final_holdout(repository, request, population)
        second = evaluate_final_holdout(repository, request, population)
        assert conn.total_changes == before
        assert first == second
        assert first.random == SetMetrics(
            n_events=1, n_trades=1, n_without_trade=0, total_ret=0.5,
            mean_ret=0.5, win_rate=1.0)
    finally:
        conn.close()


def test_set_metrics_requires_none_when_a_set_has_no_trade():
    SetMetrics(n_events=0, n_trades=0, n_without_trade=0, total_ret=0.0,
               mean_ret=None, win_rate=None)
    with pytest.raises(ValueError):
        SetMetrics(n_events=0, n_trades=0, n_without_trade=0, total_ret=0.0,
                   mean_ret=0.0, win_rate=None)
    with pytest.raises(ValueError):
        SetMetrics(n_events=0, n_trades=0, n_without_trade=0, total_ret=0.0,
                   mean_ret=None, win_rate=1.0)
