"""Nonconstant numerical oracles and candidate-recipe wiring through real fit children."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from engine.v2.contracts import ArtifactRef
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock, from_document
from engine.v2.ops.catalog import connect
from experiments import magnitude_search_execution as execution
from experiments import magnitude_search_trials as trials
from experiments.magnitude_search_plan import seal_search_plan
from tests.v2.models.training.test_magnitude_estimators import FEATURES, HGB_PARAMS, RIDGE_PARAMS
from tests.v2.research.test_magnitude_search_plan import source as base_source  # noqa: F401

LIMITS = {"memory_bytes": 2 * 1024 ** 3, "cpu_seconds": 30, "wall_seconds": 60}
OFFSETS = {5: 0.0, 3: 2.0, 1: -2.0}
TRAIN_X = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
TEST_X = [1.5, -1.5, 2.5, -2.5, 3.5, -3.5]
LABELS = [2.0, 2.0, 2.0, 10.0, 10.0, 10.0]
TRAIN_IDS = ["train-c", "train-a", "train-f", "train-b", "train-e", "train-d"]
TEST_IDS = ["eval-z", "eval-b", "eval-y", "eval-a", "eval-x", "eval-c"]


def parameters(kind):
    return ([{"alpha": 0.5}, {"alpha": 5.0}] if kind == "history_ridge" else
            [{"learning_rate": rate, "max_iter": 3, "max_leaf_nodes": 2,
              "min_samples_leaf": 1, "l2_regularization": 2.0} for rate in (0.2, 0.4)])


def recipe(kind, params):
    """Pin the declared policy independently of production recipe construction."""
    ridge = kind == "history_ridge"
    return {"schema_version": "magnitude_fitted_recipe.v1", "model": kind,
        "features": list(FEATURES), "estimator": {
            "class": "Ridge" if ridge else "HistGradientBoostingRegressor",
            "params": {**(RIDGE_PARAMS if ridge else HGB_PARAMS), **params}},
        "preprocessing": {
            "imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True}, "output_features": 22,
        } if ridge else {"missing_values": "native", "scaling": "none",
                        "all_empty_training_columns": "constant_zero_train_and_predict"},
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1}


def synthetic_inputs():
    """Numeric-boundary fixture: equal-sized, deliberately different train/test matrices."""
    rows = []
    for horizon, offset in OFFSETS.items():
        for ids, values in ((TRAIN_IDS, [x + offset for x in TRAIN_X]), (TEST_IDS, TEST_X)):
            rows.extend({"event_id": event, "context_id": f"context:{event}:{horizon}",
                "horizon_sessions": horizon, **dict(zip(FEATURES, [6.0, x] + [None] * 9))}
                for event, x in zip(ids, values))
    return {"training": [{"event_id": event, "absolute_move_pct": label}
                         for event, label in zip(TRAIN_IDS, LABELS)],
            "features": list(reversed(rows))}


def expected_forecasts(kind, params):
    """Closed forms, with no production fit, matrix helper, or sklearn estimator."""
    expected = {}
    for horizon, offset in OFFSETS.items():
        for event, x in zip(TEST_IDS, TEST_X):
            if kind == "history_ridge":
                # Centered sum(x*x)=28, sum(x*y)=48; scaling makes sum(z*z)=6.
                value = max(0.0, 6.0 + (x - offset) * 72 / (7 * (6 + params["alpha"])))
            else:
                # Two pure three-row leaves: each round shrinks the residual by
                # 1 - learning_rate * n_leaf / (n_leaf + l2_regularization).
                residual = 1 - params["learning_rate"] * 3 / (3 + params["l2_regularization"])
                value = 6.0 + (4.0 if x > offset else -4.0) * (1 - residual ** params["max_iter"])
            expected[event, horizon] = value
    return expected


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
def test_fixed_worker_nonconstant_fits_match_independent_closed_forms(monkeypatch, kind):
    original = execution.subprocess.Popen
    payloads = []

    def launch(args, **kwargs):
        assert args[1:3] == ["-m", "experiments.magnitude_search_worker"]
        payloads.append(json.loads(Path(args[3]).read_text()))
        return original(args, **kwargs)

    monkeypatch.setattr(execution.subprocess, "Popen", launch)
    inputs = synthetic_inputs()
    inner, plan = {"evaluation_event_ids": TEST_IDS}, {"horizons": list(OFFSETS)}
    forecasts = []
    for params in parameters(kind):
        policy = {"features": list(FEATURES), "fit_recipe": recipe(kind, params)}
        expected_payload = copy.deepcopy({"inputs": inputs, "inner": inner, "plan": plan,
                                          "execution": policy})
        progress = []
        result = execution.run_fitted_trial(inputs, inner, plan, policy,
            execution.execution_limits(LIMITS), lambda state, fit: progress.append((state, fit)))
        assert payloads[-1] == expected_payload
        assert result["event"] == "succeeded"
        actual = {(row["event_id"], row["horizon_sessions"]): row["predicted_absolute_move_pct"]
                  for row in result["forecasts"]}
        assert len(result["forecasts"]) == len(actual) == 18
        assert actual == pytest.approx(expected_forecasts(kind, params), rel=1e-8, abs=1e-6)
        assert [(state, fit["horizon_sessions"]) for state, fit in progress] == [
            (state, horizon) for horizon in OFFSETS for state in ("started", "completed")]
        for _, fit in progress:
            assert fit["training_event_ids"] == TRAIN_IDS
            assert fit["evaluation_event_ids"] == TEST_IDS
            assert fit["training_matrix_id"] != fit["prediction_matrix_id"]
        forecasts.append(actual)
    assert len(payloads) == 2
    assert forecasts[0] != forecasts[1]


@pytest.mark.parametrize("kind", ["history_ridge", "history_hgb"])
def test_native_search_passes_each_declared_candidate_to_real_child(base_source, tmp_path, monkeypatch, kind):  # noqa: F811
    conn = connect(tmp_path / "catalog.sqlite")
    try:
        base_source.conn.backup(conn)
        shutil.copytree(base_source.store.root, tmp_path / "store")
        store = ArtifactStore(tmp_path / "store")
        variants = parameters(kind)
        axes = {key: sorted({item[key] for item in variants}) for key in variants[0]}
        request = {"schema_version": "magnitude_search_request.v1",
            "inner_years": {"2024": ["2023"], "2025": ["2024"]}, "space": {kind: axes},
            "max_candidates": 2, "max_fits": 18}
        sealed = seal_search_plan(conn, store, base_source.registration, request)
        recorder = trials.SearchTrials(conn, store, Repository(conn, store), base_source.registration,
            request, from_document(ArtifactRef, sealed["plan_ref"]))
        recorder.seal(clock=SystemClock())
        original, observed = trials.run_fitted_trial, []

        def run(inputs, inner, plan, policy, limits, progress):
            observed.append(copy.deepcopy(policy["fit_recipe"]))
            assert policy["fit_recipe"] == selected["recipe"]
            assert len({row["absolute_move_pct"] for row in inputs["training"]}) > 1
            return original(inputs, inner, plan, policy, limits, progress)

        monkeypatch.setattr(trials, "run_fitted_trial", run)
        forecasts = []
        for selected in sealed["plan"]["candidates"]:
            result = recorder.execute("2025", "2024", selected["candidate_id"], limits=LIMITS)
            assert result["fit_attempt_count"] == result["completed_fit_count"] == 3
            assert result["terminal"]["status"] == "succeeded"
            forecasts.append(result["terminal"]["predictions"])
        axis = "alpha" if kind == "history_ridge" else "learning_rate"
        assert sorted(observed, key=lambda item: item["estimator"]["params"][axis]) == [
            recipe(kind, params) for params in variants]
        assert forecasts[0] != forecasts[1]
    finally:
        conn.close()
