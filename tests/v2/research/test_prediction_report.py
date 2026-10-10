"""Synthetic native prior-baseline evaluation through real registration, scans and fits."""
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from threadpoolctl import threadpool_info, threadpool_limits

from checks.package_readmes import directive
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, canonical_json
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiment_folds import TrainFoldRule
from engine.v2.ops.experiments import ExperimentSpec
from experiments import prediction_report as prediction
from experiments.native_registration import NativeRegistration, register_native
from tests.v2.research.test_prediction_inputs import MONTH, _commit, _event, _move


def _spec(**changes):
    return ExperimentSpec(**{
        "experiment_id": "synthetic-prediction", "hypothesis": "A fixed prior baseline provides a calibration reference.",
        "primary_arm_id": "prior", "arms": ("prior",), "seed": 7,
        "folds": ("2023", "2024"), "economic_params": {}, "price_source": "computed_moves",
        "runner": prediction.RUNNER, **changes,
    })


def _dataset():
    dates = ["2022-02-01", "2022-03-01", "2022-04-01", "2022-12-01",
             "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01",
             "2024-02-01", "2024-03-01"]
    events = [_event(day) for day in dates]
    events += [_event("2023-06-01", random=True), _event("2024-12-02")]
    labels = [-1, 1, 1, -1, -1, 1, 1, 1, -1, 1, float("nan"), float("nan")]
    moves = [_move(event, label) for event, label in zip(events, labels)]
    moves[3]["available_as_of_date"] = "2024-01-01"
    return events, moves


class RegistrationRepository(Repository):
    def scan(self, query, *, table_name):
        assert table_name == "earnings_events", "registration must not read targets"
        yield from super().scan(query, table_name=table_name)


@dataclass
class Source:
    conn: object
    repository: object
    registration: NativeRegistration
    spec: ExperimentSpec
    events: list
    root: Path


def _source(tmp_path, *, events=None, moves=None, spec=None, event_ids=None):
    if events is None:
        events, moves = _dataset()
    conn, repository, snapshot = _commit(tmp_path, events, moves)
    root = tmp_path / "code"
    runner = root / prediction.RUNNER
    runner.parent.mkdir(parents=True)
    runner.write_text("# Synthetic source-only registration; never executed.\n")
    spec = spec or _spec()
    registration = register_native(conn, ArtifactStore(tmp_path / "store"),
        RegistrationRepository(conn, ArtifactStore(tmp_path / "store")), spec,
        code_root=root, as_of_month=MONTH, event_ids=event_ids)
    assert registration.document["snapshot"]["snapshot_id"] == snapshot.snapshot_id
    return Source(conn, repository, registration, spec, events, tmp_path)


@pytest.fixture
def source(tmp_path):
    value = _source(tmp_path)
    yield value
    value.conn.close()


def _files(path):
    # SQLite read locks update its transient shared-memory index, not catalog data.
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*")
            if p.is_file() and not p.name.endswith("-shm")}


def test_real_registered_fold_prior_and_metrics_are_deterministic_read_only(source):
    before_files, before_changes = _files(source.root), source.conn.total_changes
    binding_bytes = source.registration.binding_json
    result = prediction.prediction_result(source.repository, source.registration)
    assert result == prediction.prediction_result(source.repository, source.registration)
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["registered_population_count"] == 10
    assert result["attempted_variants"] == 1
    assert result["features"] == ["constant_intercept"]
    assert result["target"] == "positive_move"
    assert result["model"] == "DummyClassifier(strategy=prior)"
    assert result["run_id"] == source.registration.run_id
    assert result["variant_id"] == source.registration.variant_id
    assert result["provenance"] == source.registration.document
    assert result["population_labels"] == ["post-release selection"]
    assert result["holdout_results"] == {
        "random": "excluded; not evaluated", "rolling": "excluded; not evaluated"}
    first, second = result["folds"]
    assert [(fold["year"], fold["train_count"], fold["test_count"]) for fold in result["folds"]] == [
        ("2023", 3, 4), ("2024", 7, 2)]
    assert first["train_event_ids"] == sorted(event["event_id"] for event in source.events[:3])
    assert second["train_event_ids"] == sorted(event["event_id"] for event in source.events[:3] + source.events[4:8])
    assert [fold["target_available_before"] for fold in result["folds"]] == ["2023-01-01", "2024-01-01"]
    assert [row["score"] for row in result["predictions"]] == pytest.approx([2 / 3] * 4 + [5 / 7] * 2)
    assert [fold["threshold"] for fold in result["folds"]] == pytest.approx([2 / 3, 5 / 7])
    labels, scores = [0, 1, 1, 1, 0, 1], [2 / 3] * 4 + [5 / 7] * 2
    brier = sum((score - label) ** 2 for score, label in zip(scores, labels)) / 6
    log_loss = -sum(math.log(score if label else 1 - score) for score, label in zip(scores, labels)) / 6
    assert result["metrics"]["count"] == 6
    assert result["metrics"]["brier"] == pytest.approx(brier)
    assert result["metrics"]["log_loss"] == pytest.approx(log_loss)
    assert result["metrics"]["log_loss_clip"] == 1e-15
    assert result["metrics"]["ece"] == pytest.approx((4 * abs(2 / 3 - 3 / 4) + 2 * abs(5 / 7 - 1 / 2)) / 6)
    assert result["metrics"]["reliability"][6] == {
        "bin": 7, "lower": 0.6, "upper": 0.7, "count": 4,
        "mean_probability": 2 / 3, "positive_rate": 3 / 4}
    assert result["metrics"]["reliability"][7]["count"] == 2
    assert len(result["metrics"]["reliability"]) == len(result["metrics"]["rank_deciles"]) == 10
    assert source.conn.total_changes == before_changes
    assert _files(source.root) == before_files
    assert source.registration.binding_json == binding_bytes
    target_queries = [query for name, query in source.repository.scans if name == "computed_moves"]
    assert len(target_queries) == 20
    # Fixtures can reuse a ticker across dates: key and time bounds must jointly exclude holdouts.
    excluded_keys = {(event["ticker"], event["event_date"].date().isoformat()) for event in source.events[-2:]}
    assert all((query.key_filter[0].values[0], query.time_interval.start_inclusive) not in excluded_keys
               for query in target_queries)
    assert all(query.snapshot_id == result["provenance"]["snapshot"]["snapshot_id"] for _, query in source.repository.scans)


def test_fitting_uses_real_shared_helper_constant_features_and_restores_threadpools(source, monkeypatch):
    real_fit = prediction.fit_walk_forward_fold
    calls = []
    environment = dict(os.environ)

    def inspect_fit(estimator, train_features, train_labels, test_features, rule):
        assert isinstance(estimator, DummyClassifier)
        assert estimator.strategy == "prior" and estimator.random_state == 7
        assert isinstance(rule, TrainFoldRule)
        assert rule.top_fraction == 0.5
        assert np.all(train_features == 1) and np.all(test_features == 1)
        assert train_features.shape[1] == test_features.shape[1] == 1
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        calls.append((len(train_features), len(test_features)))
        return real_fit(estimator, train_features, train_labels, test_features, rule)

    monkeypatch.setattr(prediction, "fit_walk_forward_fold", inspect_fit)
    with threadpool_limits(limits=2):
        before = threadpool_info()
        prediction.prediction_result(source.repository, source.registration)
        assert threadpool_info() == before
    assert calls == [(3, 4), (7, 2)]
    assert dict(os.environ) == environment


def test_threadpools_restore_on_fitting_failure(source, monkeypatch):
    def fail_fit(*args):
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        raise RuntimeError("synthetic estimator failure")

    monkeypatch.setattr(prediction, "fit_walk_forward_fold", fail_fit)
    with threadpool_limits(limits=2):
        before = threadpool_info()
        with pytest.raises(RuntimeError, match="synthetic estimator failure"):
            prediction.prediction_result(source.repository, source.registration)
        assert threadpool_info() == before


@pytest.mark.parametrize("changes", [
    {"runner": "experiments/other.py"}, {"runner": "native-gate-replay"},
    {"price_source": "option_chains"}, {"economic_params": {"fill": 0.5}},
    {"economic_params": {"exit": {"kind": "fixed_day", "trading_days": 1}}},
    {"economic_params": {"unused": 1}}, {"input_files": ("external.csv",)},
    {"folds": ()}, {"folds": ("2024", "2023")}, {"folds": ("2023", "2023")},
    {"folds": ("2023-2024",)}, {"folds": ("2023-01-01",)}, {"folds": ("23",)},
    {"folds": ("20230",)}, {"folds": ("0000",)}, {"folds": ("２０２３",)},
    {"folds": ("2023 ",)}, {"folds": (2023,)}, {"folds": "2023"},
    {"seed": True}, {"seed": -1}, {"seed": 2**32}, {"arms": ("prior", "candidate")},
])
def test_public_spec_validator_refuses_unused_or_invalid_declarations(changes):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        prediction.validate_prediction_spec(_spec(**changes))


def test_public_spec_validator_accepts_exact_fixed_arm_and_rejects_wrong_type():
    plan = prediction.validate_prediction_spec(_spec())
    assert plan.folds == ("2023", "2024")
    assert plan.runner == prediction.RUNNER
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        prediction.validate_prediction_spec(None)


@pytest.mark.parametrize("changes", [
    {"runner": "experiments/other.py"}, {"price_source": "daily_market"},
    {"economic_params": {"fill": 0.5}}, {"economic_params": None},
    {"folds": []}, {"folds": ["2024", "2023"]}, {"folds": ["2023", "2023"]},
    {"folds": ["x"]}, {"folds": [2023]}, {"folds": None},
    {"seed": -1}, {"seed": 2**32}, {"seed": True}, {"seed": 1.5}, {"seed": None},
])
def test_registration_declarations_refuse_before_any_reads(source, changes):
    document = source.registration.document
    document["execution_plan"].update(changes)
    registration = NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    before = source.conn.total_changes
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        prediction.prediction_result(source.repository, registration)
    assert source.repository.scans == []
    assert source.conn.total_changes == before


@pytest.mark.parametrize("kind", ["empty_train", "empty_test", "single_class"])
def test_unsuitable_fold_refuses_without_artifacts(tmp_path, kind):
    events, moves = _dataset()
    spec = _spec(folds={"empty_train": ("2022",), "empty_test": ("2025",),
                       "single_class": ("2023",)}[kind])
    if kind == "single_class":
        for move in moves[:3]:
            move["realized_move_pct"] = 1
    source = _source(tmp_path, events=events, moves=moves, spec=spec)
    try:
        before = _files(tmp_path)
        with pytest.raises(OpsError, match="EXPERIMENT_VARIANT_FAILED"):
            prediction.prediction_result(source.repository, source.registration)
        assert _files(tmp_path) == before
    finally:
        source.conn.close()


def test_late_fold_failure_never_returns_partial_result(source):
    document = source.registration.document
    document["execution_plan"]["folds"] = ["2023", "2025"]
    registration = NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    with pytest.raises(OpsError, match="EXPERIMENT_VARIANT_FAILED"):
        prediction.prediction_result(source.repository, registration)
    assert not list(source.root.rglob("REPORT.md"))


@pytest.mark.parametrize("mutated_indices", [[3], [4, 5, 6, 7], [8, 9]])
def test_future_or_not_yet_available_labels_do_not_change_earlier_fit(tmp_path, mutated_indices):
    baseline = _source(tmp_path / "baseline")
    events, moves = _dataset()
    for index in mutated_indices:
        moves[index]["realized_move_pct"] *= -1
    changed = _source(tmp_path / "changed", events=events, moves=moves)
    try:
        original = prediction.prediction_result(baseline.repository, baseline.registration)
        altered = prediction.prediction_result(changed.repository, changed.registration)
        original_scores = [row["score"] for row in original["predictions"]]
        changed_scores = [row["score"] for row in altered["predictions"]]
        assert original_scores[:4] == changed_scores[:4]
        if mutated_indices == [4, 5, 6, 7]:
            assert changed_scores[4:] == pytest.approx([3 / 7] * 2)
            assert original_scores[4:] != changed_scores[4:]
        else:
            assert original_scores == changed_scores
    finally:
        baseline.conn.close()
        changed.conn.close()


def test_report_binds_hypothesis_and_contains_reviewable_metrics_and_limits(source):
    result = prediction.prediction_result(source.repository, source.registration)
    before = _files(source.root)
    for no_ledger in (True, False):
        report = prediction.render_prediction_report(source.registration, result, spec=source.spec,
                                                     no_ledger=no_ledger)
        assert report == prediction.render_prediction_report(source.registration, result, spec=source.spec,
                                                             no_ledger=no_ledger)
        for text in [source.spec.hypothesis, source.spec.spec_hash, source.registration.variant_id,
                     "Attempted variants: 1", "positive_move", "constant intercept", "post-release selection",
                     "Random holdout: excluded", "Rolling holdout: excluded", "Brier", "log-loss", "weighted ECE",
                     "| 2023 | 3 | 4 |", "| 2024 | 7 | 2 |", "## OOF reliability bins", "## OOF rank deciles", "1e-15", "event_id", "P&L or promotion claims"]:
            assert text in report
        assert ("no-ledger smoke; not recorded" if no_ledger else "recorded evaluation") in report
        embedded = report.split("```json\n", 1)[1].split("\n```", 1)[0]
        assert json.loads(embedded) == result
    assert _files(source.root) == before
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        prediction.render_prediction_report(source.registration, result,
            spec=replace(source.spec, hypothesis="A different conclusion"), no_ledger=True)


def test_ten_bins_clipping_and_stable_rank_deciles():
    scores = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1, .5, .5]
    ids = [f"event-{index:02}" for index in range(len(scores))]
    labels = [int(index % 2 == 0) for index in range(len(scores))]
    frame = pd.DataFrame({"event_id": ids[::-1], "positive_move": labels, "score": scores})
    result = prediction._metrics(frame)
    shuffled = prediction._metrics(frame.sample(frac=1, random_state=123))
    assert result["rank_deciles"] == shuffled["rank_deciles"]
    assert [row["count"] for row in result["reliability"]] == [1, 1, 1, 1, 1, 3, 1, 1, 1, 2]
    assert [row["count"] for row in result["rank_deciles"]] == [2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    actual = [event_id for group in result["rank_deciles"] for event_id in group["event_ids"]]
    assert actual == frame.sort_values(["score", "event_id"]).event_id.tolist()
    assert sum(row["count"] for row in result["rank_deciles"]) == len(frame)
    assert math.isfinite(result["log_loss"])
    endpoint = prediction._metrics(pd.DataFrame({"event_id": ["b", "a"], "positive_move": [1, 0], "score": [0., 1.]}))
    assert endpoint["brier"] == 1
    assert endpoint["log_loss"] == pytest.approx(-(math.log(1e-15) + math.log1p(-(1 - 1e-15))) / 2)
    assert endpoint["ece"] == 1
    assert endpoint["reliability"][1] == {
        "bin": 2, "lower": .1, "upper": .2, "count": 0,
        "mean_probability": None, "positive_rate": None}
    assert endpoint["rank_deciles"][-1] == {
        "decile": 10, "count": 0, "mean_probability": None, "positive_rate": None, "event_ids": []}
    json.dumps(endpoint, allow_nan=False)


def test_final_read_is_not_an_interface_and_excluded_identity_stays_refused(source):
    with pytest.raises(TypeError, match="purpose"):
        prediction.prediction_result(source.repository, source.registration, purpose="final")
    document = source.registration.document
    document["event_ids"] += [source.events[-2]["event_id"]]
    registration = NativeRegistration(source.registration.run_id, canonical_json(document).encode())
    with pytest.raises(DataError, match="HOLDOUT_ACCESS_DENIED"):
        prediction.prediction_result(source.repository, registration)
    assert all(name == "earnings_events" for name, _ in source.repository.scans)


def test_fold_helpers_are_documented_public_ops_interfaces():
    root = Path(__file__).resolve().parents[3]
    assert {"TrainFoldRule", "fit_walk_forward_fold"} <= directive(
        (root / "engine/v2/ops/README.md").read_text(), "public-interface")


def test_input_row_order_does_not_change_result(source, monkeypatch):
    baseline = prediction.prediction_result(source.repository, source.registration)
    original_loader = prediction.load_prediction_targets

    def reversed_rows(*args, **kwargs):
        return original_loader(*args, **kwargs).iloc[::-1]

    monkeypatch.setattr(prediction, "load_prediction_targets", reversed_rows)
    assert prediction.prediction_result(source.repository, source.registration) == baseline


@pytest.mark.parametrize("field", ["variant_id", "run_id", "provenance"])
def test_renderer_refuses_results_from_other_registration(source, field):
    result = prediction.prediction_result(source.repository, source.registration)
    result[field] = "another-registration"
    with pytest.raises(OpsError, match="EXPERIMENT_IDENTITY_CONFLICT"):
        prediction.render_prediction_report(source.registration, result, spec=source.spec, no_ledger=True)


def test_explicit_registered_subset_is_the_only_target_read(tmp_path):
    events, moves = _dataset()
    ids = [event["event_id"] for event in events[:3] + events[4:8]]
    # A nonregistered eligible outcome is poisoned; loading all eligible outcomes would fail.
    moves[8]["realized_move_pct"] = float("nan")
    source = _source(tmp_path, events=events, moves=moves, spec=_spec(folds=("2023",)), event_ids=ids)
    try:
        result = prediction.prediction_result(source.repository, source.registration)
        assert result["registered_population_count"] == 7
        assert result["metrics"]["count"] == 4
        assert source.registration.document["event_ids"] == sorted(ids)
        targets = [query for name, query in source.repository.scans if name == "computed_moves"]
        assert len(targets) == 7
    finally:
        source.conn.close()


def test_availability_strict_boundary_changes_only_next_eligible_training_fold(tmp_path):
    events, moves = _dataset()
    moves[3]["available_as_of_date"] = "2023-12-31"
    source = _source(tmp_path, events=events, moves=moves)
    try:
        result = prediction.prediction_result(source.repository, source.registration)
        assert [fold["train_count"] for fold in result["folds"]] == [3, 8]
        assert [row["score"] for row in result["predictions"]] == pytest.approx([2 / 3] * 4 + [5 / 8] * 2)
    finally:
        source.conn.close()
