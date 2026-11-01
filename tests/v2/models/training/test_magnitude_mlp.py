"""Synthetic standalone neural contracts; no source reads or study activation."""
import copy
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_info, threadpool_limits

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models import no_fit
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe, validate_mlp_recipe
from engine.v2.models.training import _magnitude_mlp_estimator as estimators
from engine.v2.models.training import magnitude_mlp as fitting

COLUMNS = ["signal", "constant", "empty"]
PROFILES = [("relu32_l2_0p01", [32], 0.01),
            ("relu64x32_l2_0p01", [64, 32], 0.01),
            ("relu64x32_l2_0p1", [64, 32], 0.1)]


def recipe(**changes):
    return mlp_recipe(**{"columns": COLUMNS, "feature_dictionary_hash": "sha256:" + "1" * 64,
                         "profile": PROFILES[0][0], **changes})


def arrays():
    signal = np.linspace(-2, 2, 80)
    train = np.column_stack([signal, np.full(80, 7.), np.full(80, np.nan)])
    return train, 10 + 2 * signal, np.array([[0., 7., np.nan], [1., 7., 3.]])


def fit(policy=None, values=None, columns=COLUMNS):
    return fitting.fit_mlp_predictions(recipe() if policy is None else policy,
        *(arrays() if values is None else values), columns=columns)


def test_metadata_imports_neither_training_nor_sklearn():
    probe = '''
import importlib.abc
import sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(('engine.v2.models.training', 'sklearn')), fullname
sys.meta_path.insert(0, RejectTraining())
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
mlp_recipe(columns=['a'], feature_dictionary_hash='sha256:' + '1' * 64, profile='relu32_l2_0p01')
'''
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                               text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("profile,layers,alpha", PROFILES)
def test_closed_profiles_match_complete_sklearn_policy(profile, layers, alpha):
    policy = recipe(profile=profile)
    expected = MLPRegressor(hidden_layer_sizes=layers, alpha=alpha, max_iter=400,
        n_iter_no_change=15, early_stopping=False, random_state=20260829).get_params()
    assert policy["estimator"] == {"class": "MLPRegressor", "params": expected}
    assert policy["preprocessing"] == {
        "variance_filter": {"threshold": 0.0},
        "imputer": {"strategy": "median", "keep_empty_features": True},
        "missing_indicator": {"features": "all", "sparse": False},
        "scaler": {"with_mean": True, "with_std": True},
        "union_features": 6, "output_features": "training_nonconstant"}
    assert policy["target_transform"] == {
        "class": "StandardScaler", "with_mean": True, "with_std": True}
    restored = validate_mlp_recipe(json.loads(canonical_json(policy)))
    assert restored == policy
    assert restored["component_fits"] == restored["numerical_threads"] == 1
    assert restored["prediction_projection"] == "clip_zero"
    restored["features"].clear()
    restored["estimator"]["params"]["hidden_layer_sizes"].clear()
    assert recipe(profile=profile) == policy


@pytest.mark.parametrize("changes", [
    {"columns": []}, {"columns": "a"}, {"columns": ["a", "a"]},
    {"columns": ["bad/name"]}, {"columns": [True]}, {"columns": ["a", []]},
    {"feature_dictionary_hash": "sha256:abc"}, {"feature_dictionary_hash": None},
    {"profile": []}, {"profile": True}, {"profile": "relu128"},
])
def test_unknown_profile_or_bad_identity_refuses(changes):
    with pytest.raises(ValueError):
        recipe(**changes)


class Untouchable:
    def __array__(self, *args, **kwargs):
        raise AssertionError("invalid policy must refuse before array access")


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "magnitude_mlp_recipe.v2"), (("model",), "size_v1_4"),
    (("extra",), 1), (("profile",), "relu128"), (("estimator", "class"), "other"),
    (("estimator", "params", "hidden_layer_sizes"), [128]),
    (("estimator", "params", "random_state"), 42),
    (("estimator", "params", "early_stopping"), True),
    (("estimator", "params", "max_iter"), 401),
    (("estimator", "params", "warm_start"), True),
    (("preprocessing", "missing_indicator", "features"), "missing-only"),
    (("preprocessing", "variance_filter", "threshold"), 0.01),
    (("target_transform", "with_std"), False), (("component_fits",), 2),
    (("numerical_threads",), 2), (("prediction_projection",), "none"),
    (("nonconvergence",), "retry"),
])
def test_policy_drift_refuses_before_array_access(path, value):
    policy = recipe()
    node = policy
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match="invalid closed MLP"):
        fit(policy, [Untouchable()] * 3)


@pytest.mark.parametrize("guard", [legacy_no_fit.no_fit_guard, no_fit.no_fit_guard])
def test_both_guards_precede_recipe_and_array_inspection(guard):
    with guard(), pytest.raises(RuntimeError):
        fitting.fit_mlp_predictions(Untouchable(), *([Untouchable()] * 3), columns=None)


@pytest.mark.parametrize("columns", [list(reversed(COLUMNS)), [], "signal"])
def test_column_order_refuses_before_array_access(columns):
    with pytest.raises(ValueError, match="exactly match"):
        fit(values=[Untouchable()] * 3, columns=columns)


@pytest.mark.parametrize("slot,replacement", [
    (0, []), (0, [[1, 2]]), (0, [[float("inf"), 2, 3]]), (0, [[1, 2, object()]]),
    (1, [0]), (1, np.full(80, -1)), (1, np.full(80, float("nan"))),
    (1, np.full(80, float("inf"))), (2, []), (2, [[1, 2]]),
])
def test_invalid_matrices_fail_without_fitting(monkeypatch, slot, replacement):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid input entered sklearn fit")
    monkeypatch.setattr(TransformedTargetRegressor, "fit", forbidden)
    values = list(arrays())
    values[slot] = replacement
    with pytest.raises(ValueError):
        fit(values=values)


@pytest.mark.parametrize("profile,layers,alpha", PROFILES)
def test_real_fit_matches_independent_oracle_and_is_repeatable(profile, layers, alpha):
    train, labels, prediction = arrays()
    before = copy.deepcopy((train, labels, prediction))
    union = FeatureUnion([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("missing_indicator", MissingIndicator(features="all", sparse=False))], n_jobs=1)
    oracle = TransformedTargetRegressor(transformer=StandardScaler(),
        regressor=make_pipeline(union, VarianceThreshold(threshold=0.), StandardScaler(), MLPRegressor(
            hidden_layer_sizes=layers, alpha=alpha, max_iter=400, n_iter_no_change=15,
            early_stopping=False, random_state=20260829)))
    with threadpool_limits(limits=1):
        oracle.fit(train, labels)
        expected = np.maximum(oracle.predict(prediction), 0.).tolist()
    result = fit(recipe(profile=profile))
    assert result == fit(recipe(profile=profile))
    assert result["predictions"] == expected
    assert result["recipe_id"] == content_hash(recipe(profile=profile))
    assert result["converged"] is True and result["component_fits"] == 1
    assert result["schema_version"] == "magnitude_mlp_fit.v1"
    assert result["n_iter"] == oracle.regressor_[-1].n_iter_ <= 400
    assert result["input_features"] == oracle.regressor_[-1].n_features_in_ == 1
    assert result["loss"] == oracle.regressor_[-1].loss_
    assert result["loss_scale"] == "standardized_target"
    for actual, original in zip((train, labels, prediction), before):
        np.testing.assert_array_equal(actual, original)


def test_prediction_rows_cannot_influence_training_or_preprocessing(monkeypatch):
    models = []
    original = TransformedTargetRegressor.fit
    def capture(self, *args, **kwargs):
        assert all(pool["num_threads"] == 1 for pool in threadpool_info())
        value = original(self, *args, **kwargs)
        models.append(self)
        return value
    monkeypatch.setattr(TransformedTargetRegressor, "fit", capture)
    train, labels, prediction = arrays()
    result = fit(values=(train, labels, prediction))
    poisoned = np.vstack([prediction, [1e10, np.nan, 1e10]])
    changed = fit(values=(train, labels, poisoned))
    assert changed["predictions"][:2] == result["predictions"]
    assert changed["loss"] == result["loss"] and changed["n_iter"] == result["n_iter"]
    for model in models:
        union, selector, scaler, network = model.regressor_
        np.testing.assert_allclose(union.transformer_list[0][1].statistics_, [0., 7., 0.], atol=1e-15)
        assert selector.get_support().tolist() == [True, False, False, False, False, False]
        np.testing.assert_allclose(scaler.mean_, [0.], atol=1e-15)
        assert network.n_features_in_ == 1
        assert model.transformer_.mean_[0] == pytest.approx(labels.mean())
        assert model.transformer_.scale_[0] == pytest.approx(labels.std())


@pytest.mark.parametrize("warning,refuses", [(ConvergenceWarning, True), (UserWarning, False)])
@pytest.mark.parametrize("message", ["synthetic warning", "Training interrupted by user."])
def test_only_convergence_warnings_refuse_without_retry(monkeypatch, warning, refuses, message):
    original = TransformedTargetRegressor.fit
    calls = []
    def warned(self, *args, **kwargs):
        calls.append(1)
        result = original(self, *args, **kwargs)
        warnings.warn(message, warning)
        return result
    monkeypatch.setattr(TransformedTargetRegressor, "fit", warned)
    if refuses:
        with pytest.raises(ValueError, match="convergence failed; no retry"):
            fit()
    else:
        assert fit()["converged"] is True
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["loss_", "coefs_", "intercepts_"])
def test_nonfinite_fitted_state_refuses(monkeypatch, field):
    original = TransformedTargetRegressor.fit
    def corrupted(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        network = self.regressor_[-1]
        if field == "loss_":
            network.loss_ = float("nan")
        else:
            getattr(network, field)[0].flat[0] = float("nan")
        return result
    monkeypatch.setattr(TransformedTargetRegressor, "fit", corrupted)
    with pytest.raises(ValueError, match="nonfinite fit evidence"):
        fit()


@pytest.mark.parametrize("values", [[float("nan"), 0], [float("inf"), 0], [1], [[1], [2]]])
def test_inverse_transformed_predictions_must_be_finite_and_correct_shape(monkeypatch, values):
    monkeypatch.setattr(TransformedTargetRegressor, "predict", lambda *args: values)
    with pytest.raises(ValueError, match="one finite value per prediction row"):
        fit()


def test_projection_occurs_only_after_inverse_transform(monkeypatch):
    monkeypatch.setattr(TransformedTargetRegressor, "predict", lambda *args: [-2., 4.])
    assert fit()["predictions"] == [0., 4.]


def test_affine_target_scaling_restores_original_units():
    train, labels, prediction = arrays()
    result = fit(values=(train, labels, prediction))
    scaled = fit(values=(train, labels * 100 + 1000, prediction))
    np.testing.assert_allclose(scaled["predictions"],
        np.asarray(result["predictions"]) * 100 + 1000, rtol=1e-12)
    assert scaled["n_iter"] == result["n_iter"]
    assert scaled["loss"] == pytest.approx(result["loss"])


@pytest.mark.parametrize("train", [[[np.nan]], [[1.], [1.]]])
def test_all_constant_union_refuses_before_neural_fit(monkeypatch, train):
    def forbidden(*args, **kwargs):
        pytest.fail("constant union entered neural fitting")
    monkeypatch.setattr(MLPRegressor, "fit", forbidden)
    with pytest.raises(ValueError, match="No feature in X meets the variance threshold"):
        fit(recipe(columns=["constant"]), (train, [0.] * len(train), [[2.]]), columns=["constant"])


def test_zero_targets_are_valid():
    train, _, prediction = arrays()
    result = fit(values=(train, np.zeros(len(train)), prediction))
    assert np.isfinite(result["predictions"]).all() and result["converged"] is True


def test_training_all_empty_columns_ignore_new_prediction_values_and_missingness():
    train, labels, _ = arrays()
    prediction = np.array([[1., 7., value] for value in (np.nan, 0., 3., 100., 1e100)])
    before = prediction.copy()
    result = fit(values=(train, labels, prediction))
    assert len(set(result["predictions"])) == 1
    np.testing.assert_array_equal(prediction, before)


def test_training_constant_values_and_indicators_ignore_prediction_only_changes():
    train, labels, _ = arrays()
    prediction = np.array([[1., value, np.nan] for value in (np.nan, 7., 0., 1e100)])
    result = fit(values=(train, labels, prediction))
    assert len(set(result["predictions"])) == 1


def test_real_iteration_exhaustion_is_not_success(monkeypatch):
    calls = []
    def exhausted(**params):
        calls.append(1)
        return MLPRegressor(**{**params, "max_iter": 1})
    monkeypatch.setattr(estimators, "MLPRegressor", exhausted)
    with pytest.raises(ValueError, match="convergence failed; no retry"):
        fit()
    assert len(calls) == 1


@pytest.mark.parametrize("width", [11, 28, 43])
def test_variable_width_features_keep_the_informative_last_column(width):
    columns = [f"x{i}" for i in range(width)]
    train = np.full((80, width), 7.)
    signal = np.linspace(-2, 2, 80)
    train[:, -1] = signal
    prediction = np.full((2, width), 7.)
    prediction[:, -1] = [-1., 1.]
    result = fit(recipe(columns=columns), (train, 10 + 2 * signal, prediction), columns=columns)
    assert result["input_features"] == 1
    assert result["predictions"][1] > result["predictions"][0] + 3


@pytest.mark.parametrize("warning_filter", ["always", "ignore", "error"])
def test_interrupted_sklearn_training_preserves_original_interrupt(monkeypatch, warning_filter):
    original = MLPRegressor._backprop
    interrupt, calls = KeyboardInterrupt("synthetic cancellation"), []
    def interrupted(self, *args, **kwargs):
        calls.append(self.n_iter_)
        if len(calls) == 2:
            raise interrupt
        return original(self, *args, **kwargs)
    monkeypatch.setattr(MLPRegressor, "_backprop", interrupted)
    monkeypatch.setattr(TransformedTargetRegressor, "predict",
                        lambda *a, **k: pytest.fail("interrupted fit reached prediction"))
    with warnings.catch_warnings():
        warnings.simplefilter(warning_filter, UserWarning)
        with pytest.raises(KeyboardInterrupt) as caught:
            fit()
    assert caught.value is interrupt
    assert calls == [0, 1]  # One complete epoch, then cancellation; no hidden retry.
