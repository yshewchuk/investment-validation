"""Synthetic, source-free ElasticNet oracles; no experiment/search activation."""
import copy
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models import no_fit
from engine.v2.models.magnitude_elasticnet_recipe import (
    elasticnet_recipe,
    validate_elasticnet_recipe,
)
from engine.v2.models.training.magnitude import fit_magnitude_predictions
from engine.v2.models.training.magnitude_elasticnet import fit_elasticnet_predictions

COLUMNS = ["signal", "constant", "empty"]
DICTIONARY = "sha256:" + "1" * 64


def recipe(**overrides):
    return elasticnet_recipe(**{"columns": COLUMNS, "feature_dictionary_hash": DICTIONARY,
        "alpha": 0.1, "l1_ratio": 0.75, **overrides})


def matrix(values):
    return np.asarray([[value, 7, np.nan] for value in values], dtype=float)


def fit(train=None, labels=None, prediction=None, **kwargs):
    return fit_elasticnet_predictions(recipe(), matrix([0, 2, 4]) if train is None else train,
        [1, 5, 9] if labels is None else labels, matrix([1]) if prediction is None else prediction,
        columns=COLUMNS, **kwargs)


def test_recipe_is_closed_and_imports_neither_training_nor_sklearn():
    probe = '''
import importlib.abc
import sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(('engine.v2.models.training', 'sklearn')), fullname
sys.meta_path.insert(0, RejectTraining())
from engine.v2.models.magnitude_elasticnet_recipe import elasticnet_recipe
elasticnet_recipe(columns=['a'], feature_dictionary_hash='sha256:' + '1' * 64, alpha=1, l1_ratio=.25)
'''
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    expected = {
        "schema_version": "magnitude_elasticnet_recipe.v1", "model": "elasticnet",
        "features": COLUMNS, "feature_dictionary_hash": DICTIONARY,
        "estimator": {"class": "ElasticNet", "params": {
            "alpha": .1, "l1_ratio": .75, "fit_intercept": True, "precompute": False,
            "max_iter": 10000, "copy_X": True, "tol": .0001, "warm_start": False,
            "positive": False, "random_state": 0, "selection": "cyclic"}},
        "preprocessing": {"imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True}, "output_features": 6},
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1,
        "nonconvergence": "variant_failure_no_retry",
    }
    assert recipe() == expected
    returned = validate_elasticnet_recipe(expected)
    returned["features"].clear()
    assert expected["features"] == COLUMNS
    assert content_hash(expected) == content_hash(recipe())


@pytest.mark.parametrize("overrides", [
    {"columns": []}, {"columns": "signal"}, {"columns": ["a", "a"]},
    {"columns": ["bad/name"]}, {"columns": [True]}, {"columns": ["a", []]},
    {"feature_dictionary_hash": "sha256:abc"}, {"feature_dictionary_hash": None},
    {"alpha": True}, {"alpha": False}, {"alpha": float("nan")}, {"alpha": float("inf")},
    {"alpha": 0}, {"alpha": -1}, {"alpha": 1e-7}, {"alpha": 1000.0001},
    {"alpha": "1"}, {"l1_ratio": False}, {"l1_ratio": 0}, {"l1_ratio": -1},
    {"l1_ratio": 1e-7}, {"l1_ratio": 1.0001}, {"l1_ratio": float("nan")},
    {"l1_ratio": float("inf")}, {"l1_ratio": "0.5"},
])
def test_recipe_rejects_undeclared_parameters_or_malformed_identity(overrides):
    with pytest.raises(ValueError):
        recipe(**overrides)


@pytest.mark.parametrize("alpha", [1e-6, 0.001, 0.003, 0.03, 100.0, 1000.0])
@pytest.mark.parametrize("l1_ratio", [1e-6, 0.25, 0.75, 1.0])
def test_v3_alpha_and_l1_ratio_intervals_are_accepted_and_validated(alpha, l1_ratio):
    policy = recipe(alpha=alpha, l1_ratio=l1_ratio)
    assert policy["estimator"]["params"]["alpha"] == float(alpha)
    assert policy["estimator"]["params"]["l1_ratio"] == float(l1_ratio)
    assert validate_elasticnet_recipe(json.loads(canonical_json(policy))) == policy


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "magnitude_fitted_recipe.v1"), (("extra",), 1),
    (("model",), "history_ridge"), (("estimator", "class"), "ElasticNetCV"),
    (("estimator", "params", "max_iter"), 10001), (("estimator", "params", "warm_start"), True),
    (("estimator", "params", "random_state"), 1), (("estimator", "params", "fit_intercept"), 1),
    (("preprocessing", "imputer", "strategy"), "mean"),
    (("preprocessing", "missing_indicator", "features"), "missing-only"),
    (("target_transform",), "log1p"), (("prediction_projection",), "none"),
    (("numerical_threads",), 2), (("nonconvergence",), "retry"),
])
def test_changed_or_extra_policy_refuses_before_array_access(path, value):
    damaged = recipe()
    node = damaged
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="recipe"):
        fit_elasticnet_predictions(damaged, Poison(), Poison(), Poison(), columns=COLUMNS)


def test_non_json_policy_cannot_masquerade_as_canonical_strings():
    class Alias:
        def __str__(self):
            return "ElasticNet"

    damaged = recipe()
    damaged["estimator"]["class"] = Alias()
    with pytest.raises(ValueError, match="recipe"):
        validate_elasticnet_recipe(damaged)


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("protected or premature array access")


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
def test_both_no_fit_guards_precede_every_input(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        fit_elasticnet_predictions(Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("columns", [["unknown"], COLUMNS[::-1], COLUMNS + ["extra"], ["signal"] * 3])
def test_unknown_duplicate_or_reordered_matrix_columns_refuse_before_conversion(columns):
    with pytest.raises(ValueError, match="columns"):
        fit_elasticnet_predictions(recipe(), Poison(), Poison(), Poison(), columns=columns)


def test_identity_binds_subset_order_dictionary_and_all_six_parameter_pairs():
    ids = {content_hash(recipe(alpha=alpha, l1_ratio=ratio))
        for alpha in (.01, .1, 1) for ratio in (.25, .75)}
    assert len(ids) == 6
    original = content_hash(recipe())
    for change in ({"columns": COLUMNS[::-1]}, {"columns": COLUMNS[:2]},
                   {"feature_dictionary_hash": "sha256:" + "2" * 64}):
        assert content_hash(recipe(**change)) != original
    restored = json.loads(canonical_json(recipe(alpha=1)))
    assert type(restored["estimator"]["params"]["alpha"]) is int
    assert validate_elasticnet_recipe(restored) == recipe(alpha=1)
    assert fit()["recipe_id"] == original


@pytest.mark.parametrize("alpha", [.01, .1, 1])
@pytest.mark.parametrize("l1_ratio", [.25, .75])
def test_independent_soft_threshold_oracle_sparse_evidence_and_nonnegative_projection(alpha, l1_ratio):
    # For one standardized signal, c=mean(z*y); beta=(c-alpha*l1)/(1+alpha*(1-l1)).
    scale = np.sqrt(8 / 3)
    beta = (2 * scale - alpha * l1_ratio) / (1 + alpha * (1 - l1_ratio))
    result = fit_elasticnet_predictions(recipe(alpha=alpha, l1_ratio=l1_ratio),
        matrix([0, 2, 4]), [1, 5, 9], matrix([-100, 0, 1, 2, 4]), columns=COLUMNS)
    assert result["predictions"] == pytest.approx(
        [0, 5 - 2 * beta / scale, 5 - beta / scale, 5, 5 + 2 * beta / scale])
    assert result["intercept"] == pytest.approx(5)
    assert result["selected_columns"] == ["signal"]
    assert result["coefficients"] == [
        {"column": col, "kind": kind, "coefficient": pytest.approx(beta) if index == 0 else 0.0}
        for index, (kind, col) in enumerate(
            [(kind, col) for kind in ("value", "missing_indicator") for col in COLUMNS])]
    assert result["coefficient_scale"] == "standardized_inputs"
    assert result["converged"] is True and 0 < result["n_iter"] <= 10000
    assert np.isfinite(result["dual_gap"])


@pytest.fixture
def observed(monkeypatch):
    fits, predictions = [], []
    original_fit, original_predict = ElasticNet.fit, Pipeline.predict

    def capture_fit(self, x, y, *args, **kwargs):
        fits.append((self, x.copy(), y.copy(), threadpool_info()))
        return original_fit(self, x, y, *args, **kwargs)

    def capture_predict(self, x, *args, **kwargs):
        predictions.append(self)
        return original_predict(self, x, *args, **kwargs)

    monkeypatch.setattr(ElasticNet, "fit", capture_fit)
    monkeypatch.setattr(Pipeline, "predict", capture_predict)
    return fits, predictions


def test_poisoned_validation_statistics_cannot_change_preprocessing_or_selection(observed):
    train = matrix([0, 2, np.nan])
    first = fit(train=train)
    polluted = matrix([1, 1e12, np.nan])
    polluted[:, 1:] = 1e12
    second = fit(train=train, prediction=polluted)
    fits, pipelines = observed
    assert len(fits) == len(pipelines) == 2  # Exactly one fit, never a refit on prediction.
    for fitted in pipelines:
        union, scaler, estimator = (step for _, step in fitted.steps)
        imputer, indicator = (step for _, step in union.transformer_list)
        assert imputer.statistics_ == pytest.approx([1, 7, 0])
        assert indicator.features_.tolist() == [0, 1, 2]
        assert scaler.mean_ == pytest.approx([1, 7, 0, 1 / 3, 0, 1])
        assert scaler.n_features_in_ == estimator.n_features_in_ == 6
    assert pipelines[0] is not pipelines[1]
    for key in first.keys() - {"predictions"}:
        assert first[key] == second[key]
    for _, x, y, pools in fits:
        assert len(x) == len(y) == 3
        assert y == pytest.approx([1, 5, 9])
        assert all(pool["num_threads"] == 1 for pool in pools)
    # The adapter has no target-bearing validation/outer input and never inspects one.
    with pytest.raises(TypeError):
        fit(validation_labels=Poison())


def test_missing_indicator_can_select_a_column_without_its_value_coefficient():
    result = fit(train=matrix([1, 1, np.nan, np.nan]), labels=[1, 1, 9, 9], prediction=matrix([1, np.nan]))
    assert result["selected_columns"] == ["signal"]
    assert result["coefficients"][0]["coefficient"] == 0
    assert result["coefficients"][3]["coefficient"] > 0
    assert result["predictions"][1] > result["predictions"][0]


def test_all_null_columns_and_new_missingness_remain_stable(observed):
    result = fit(train=np.full((3, 3), np.nan), prediction=np.asarray([[1, 2, 3], [np.nan] * 3]))
    assert result["predictions"] == pytest.approx([5, 5])
    assert result["selected_columns"] == []
    result = fit(prediction=matrix([np.nan]))
    assert result["predictions"] == pytest.approx([5])


def test_replay_is_deterministic_fresh_and_does_not_mutate_inputs():
    train, labels, prediction = matrix([0, 2, 4]), np.asarray([1., 5, 9]), matrix([1, np.nan])
    inputs = (recipe(), train, labels, prediction)
    before = copy.deepcopy(inputs)
    first = fit_elasticnet_predictions(*inputs, columns=COLUMNS)
    assert fit(labels=labels * 3)["coefficients"] != first["coefficients"]
    assert fit_elasticnet_predictions(*inputs, columns=COLUMNS) == first
    assert inputs[0] == before[0]
    for original, unchanged in zip(inputs[1:], before[1:]):
        np.testing.assert_array_equal(original, unchanged)
    assert json.loads(json.dumps(first, allow_nan=False)) == first


@pytest.mark.parametrize("inputs", [
    {"train": []}, {"prediction": []}, {"train": [[0, 1]]}, {"prediction": [[0, 1]]},
    {"train": [[np.inf] * 3]}, {"prediction": [[np.inf] * 3]},
    {"labels": [1]}, {"labels": [1, -1, 3]}, {"labels": [1, np.nan, 3]},
    {"labels": [[1, 5, 9]]},
])
def test_malformed_matrices_or_labels_refuse(inputs):
    with pytest.raises(ValueError, match="ElasticNet"):
        fit(**inputs)


def test_convergence_warning_refuses_without_predict_retry_or_partial_result(monkeypatch):
    original_fit, calls = ElasticNet.fit, []

    def warns(self, x, y, **kwargs):
        calls.append(1)
        original_fit(self, x, y, **kwargs)
        warnings.warn("synthetic convergence failure", ConvergenceWarning)
        return self

    monkeypatch.setattr(ElasticNet, "fit", warns)
    monkeypatch.setattr(Pipeline, "predict", lambda *a, **k: pytest.fail("prediction after failed fit"))
    with pytest.raises(ValueError, match="convergence failed; no retry"):
        fit()
    assert calls == [1]


@pytest.mark.parametrize("field", ["coef_", "intercept_", "dual_gap_"])
def test_nonfinite_fit_evidence_is_refused(monkeypatch, field):
    original_fit = ElasticNet.fit

    def corrupt(self, x, y, **kwargs):
        original_fit(self, x, y, **kwargs)
        setattr(self, field, np.full(6, np.nan) if field == "coef_" else np.nan)
        return self

    monkeypatch.setattr(ElasticNet, "fit", corrupt)
    with pytest.raises(ValueError, match="nonfinite fit evidence"):
        fit()


def test_nonfinite_prediction_refuses_and_v1_dispatch_is_unchanged(monkeypatch):
    monkeypatch.setattr(Pipeline, "predict", lambda *a, **k: [np.nan])
    with pytest.raises(ValueError, match="predictions require"):
        fit()
    with pytest.raises(ValueError, match="unknown fitted magnitude"):
        fit_magnitude_predictions(recipe(), Poison(), Poison(), Poison())
