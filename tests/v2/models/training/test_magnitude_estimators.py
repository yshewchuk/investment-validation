"""Independent tiny numeric oracles for fixed magnitude fits, without data I/O."""
import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json
from engine.v2.models import no_fit
from engine.v2.models.training.magnitude import fit_magnitude_predictions
from engine.v2.ops.native_registration import _fitted_magnitude_recipe as fitted_magnitude_recipe

KINDS = ("history_ridge", "history_hgb")
FEATURES = ["n_prior", "mean_prior_move", "mean_prior_abs_move",
            "ema2_prior_move", "ema2_prior_abs_move", "ema4_prior_move",
            "ema4_prior_abs_move", "ema8_prior_move", "ema8_prior_abs_move",
            "ema12_prior_move", "ema12_prior_abs_move"]
RIDGE_PARAMS = {"alpha": 1.0, "fit_intercept": True, "copy_X": True,
                "max_iter": None, "tol": 0.0001, "solver": "svd",
                "positive": False, "random_state": 0}
HGB_PARAMS = {
    "loss": "squared_error", "quantile": None, "learning_rate": 0.1,
    "max_iter": 100, "max_leaf_nodes": 7, "max_depth": None,
    "min_samples_leaf": 20, "l2_regularization": 0.0, "max_features": 1.0,
    "max_bins": 255, "categorical_features": None, "monotonic_cst": None,
    "interaction_cst": None, "warm_start": False, "early_stopping": False,
    "scoring": "loss", "validation_fraction": 0.1, "n_iter_no_change": 10,
    "tol": 1e-7, "verbose": 0, "random_state": 0,
}


def test_registration_recipe_construction_does_not_import_training_or_sklearn():
    probe = """
import importlib.abc
import sys
class RejectTraining(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(('engine.v2.models.training', 'sklearn')), fullname
sys.meta_path.insert(0, RejectTraining())
from engine.v2.ops.native_registration import magnitude_fitted_execution_document
for kind in ('history_ridge', 'history_hgb'):
    assert magnitude_fitted_execution_document('2026-01-01', kind)['model'] == kind
assert not any(name.startswith(('engine.v2.models.training', 'sklearn')) for name in sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def _matrix(values):
    """This numeric-boundary fixture assigns no history-count semantics."""
    matrix = np.full((len(values), 11), np.nan)
    matrix[:, 0] = values
    matrix[:, 1] = 7.0  # Exercise both constant and entirely missing columns.
    return matrix


@pytest.mark.parametrize("kind", KINDS)
def test_recipe_has_closed_independently_pinned_json_policy(kind):
    ridge = kind == "history_ridge"
    expected = {
        "schema_version": "magnitude_fitted_recipe.v1", "model": kind,
        "features": FEATURES,
        "estimator": {"class": "Ridge" if ridge else "HistGradientBoostingRegressor",
                      "params": RIDGE_PARAMS if ridge else HGB_PARAMS},
        "preprocessing": {
            "imputer": {"strategy": "median", "keep_empty_features": True},
            "missing_indicator": {"features": "all", "sparse": False},
            "scaler": {"with_mean": True, "with_std": True}, "output_features": 22,
        } if ridge else {"missing_values": "native", "scaling": "none",
                        "all_empty_training_columns": "constant_zero_train_and_predict"},
        "target_transform": "none", "prediction_projection": "clip_zero",
        "fit_scope": "fold_and_horizon", "numerical_threads": 1,
    }
    recipe = fitted_magnitude_recipe(kind)
    assert recipe == expected
    assert json.loads(json.dumps(recipe, allow_nan=False)) == expected
    recipe["features"].clear()
    recipe["estimator"]["params"]["random_state"] = 99
    recipe["preprocessing"].clear()
    assert fitted_magnitude_recipe(kind) == expected


@pytest.mark.parametrize("kind", ["pooled_mean", "history_mean", "ridge", "HGB", "", None])
def test_unknown_models_refused_by_recipe_and_fit(kind):
    with pytest.raises(ValueError, match="unknown"):
        fitted_magnitude_recipe(kind)
    with pytest.raises(ValueError, match="unknown"):
        fit_magnitude_predictions({"model": kind}, _matrix([0]), [1], _matrix([0]))


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
@pytest.mark.parametrize("kind", KINDS)
def test_no_fit_switch_precedes_input_or_recipe_processing(guard, kind):
    class Poison:
        def __array__(self, *args, **kwargs):
            pytest.fail("input inspected before no-fit guard")

    with guard.no_fit_guard():
        assert fitted_magnitude_recipe(kind)["model"] == kind
        with pytest.raises(guard.RuntimeFitForbidden, match="fit_magnitude_predictions"):
            fit_magnitude_predictions(fitted_magnitude_recipe(kind), Poison(), Poison(), Poison())
        with pytest.raises(guard.RuntimeFitForbidden, match="fit_magnitude_predictions"):
            fit_magnitude_predictions("unknown", [], [], [])


def test_ridge_standardized_closed_form_and_repeated_training_weight():
    # After scaling, sum(z**2)=3; alpha=1 shrinks the slope 2 to 1.5.
    result = fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), _matrix([0, 2, 4]),
                                       [1, 5, 9], _matrix([0, 1, 2, 4, 100]))
    assert result == pytest.approx([2, 3.5, 5, 8, 152], abs=1e-10)
    # Three copies change the penalty/data ratio: slope=2*9/10, intercept=1.4.
    repeated = fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), _matrix([0, 2, 4] * 3),
                                         [1, 5, 9] * 3, _matrix([0, 2, 4]))
    assert repeated == pytest.approx([1.4, 5, 8.6], abs=1e-10)


def test_ridge_median_and_full_indicators_have_independent_numeric_oracle():
    # The imputed x and missingness indicator are orthogonal after centering.
    # Ridge gives 2 + 1.5*x_imputed + 4.5*is_missing, median=1.
    result = fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), _matrix([0, 2, None]),
                                       [1, 5, 9], _matrix([0, 2, None, 100]))
    assert result == pytest.approx([2, 5, 8, 152], abs=1e-10)


def test_hgb_two_leaf_oracle_uses_fixed_iteration_count():
    # Each round multiplies either leaf's residual from its mean by 0.9.
    result = fit_magnitude_predictions(fitted_magnitude_recipe("history_hgb"), _matrix([0] * 20 + [1] * 20),
                                       [2] * 20 + [8] * 20, _matrix([0, 1]))
    assert result == pytest.approx([2 + 3 * 0.9**100, 8 - 3 * 0.9**100], abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_constant_and_entirely_missing_columns_preserve_original_target_units(kind):
    result = fit_magnitude_predictions(fitted_magnitude_recipe(kind), np.full((3, 11), np.nan),
                                       [1, 1, 9], np.full((2, 11), np.nan))
    assert result == pytest.approx([11 / 3, 11 / 3], abs=1e-10)


@pytest.mark.parametrize("kind", KINDS)
def test_canonical_recipe_round_trip_fits_without_mutating_admitted_data(kind):
    recipe = json.loads(canonical_json(fitted_magnitude_recipe(kind)))
    before = canonical_json(recipe)
    if kind == "history_hgb":
        assert type(recipe["estimator"]["params"]["max_features"]) is int
    result = fit_magnitude_predictions(recipe, np.zeros((2, 11)), [2, 8], np.zeros((1, 11)))
    assert result == pytest.approx([5.])
    assert canonical_json(recipe) == before


@pytest.fixture
def observed_models(monkeypatch):
    """Observe actual predict calls; every estimator and preprocessing fit is real."""
    observed = []
    for cls in (Pipeline, HistGradientBoostingRegressor):
        original = cls.predict

        def observe(self, matrix, *args, _original=original, **kwargs):
            observed.append((self, threadpool_info()))
            return _original(self, matrix, *args, **kwargs)

        monkeypatch.setattr(cls, "predict", observe)
    return observed


def test_ridge_training_statistics_unchanged_by_evaluation_extremes(observed_models):
    train = _matrix([0, 2, None])
    fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), train, [1, 5, 9], _matrix([1]))
    extreme = _matrix([1, 1e12, None])
    extreme[0, 1:] = 1e12
    fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), train, [1, 5, 9], extreme)
    first, second = (entry[0] for entry in observed_models)
    assert first is not second
    for fitted in (first, second):
        union, scaler, ridge = (step for _, step in fitted.steps)
        imputer, indicator = (step for _, step in union.transformer_list)
        assert imputer.statistics_ == pytest.approx([1, 7] + [0] * 9)
        assert indicator.features_.tolist() == list(range(11))
        assert union.transform(extreme).shape == (3, 22)
        assert scaler.n_features_in_ == ridge.n_features_in_ == 22
        assert ridge.get_params(deep=False) == RIDGE_PARAMS
        assert scaler.mean_ == pytest.approx([1, 7] + [0] * 9 + [1 / 3, 0] + [1] * 9)
    np.testing.assert_array_equal(first[1].mean_, second[1].mean_)
    np.testing.assert_array_equal(first[1].scale_, second[1].scale_)


def test_ridge_accepts_missingness_in_a_previously_complete_column(observed_models):
    result = fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), _matrix([0, 2, 4]),
                                       [1, 5, 9], _matrix([None]))
    assert result == pytest.approx([5])
    union = observed_models[0][0][0]
    assert union.transform(_matrix([None])).shape == (1, 22)
    assert union.transform(_matrix([None]))[0, 11] == 1


def test_hgb_training_empty_mask_is_frozen_and_partial_nans_remain_native(monkeypatch):
    train = _matrix([0] * 20 + [None] * 20)
    captured = {}
    original_fit = HistGradientBoostingRegressor.fit
    original_predict = HistGradientBoostingRegressor.predict

    def observe_fit(self, matrix, labels, *args, **kwargs):
        captured["train"] = matrix.copy()
        return original_fit(self, matrix, labels, *args, **kwargs)

    def observe_predict(self, matrix, *args, **kwargs):
        captured["test"] = matrix.copy()
        return original_predict(self, matrix, *args, **kwargs)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", observe_fit)
    monkeypatch.setattr(HistGradientBoostingRegressor, "predict", observe_predict)
    test = _matrix([0, None])
    test[:, 2:] = [[1e12] * 9, [-1e12] * 9]
    result = fit_magnitude_predictions(fitted_magnitude_recipe("history_hgb"), train, [2] * 20 + [8] * 20, test)
    assert result == pytest.approx([2 + 3 * 0.9**100, 8 - 3 * 0.9**100], abs=1e-6)
    assert captured["train"].shape == (40, 11) and captured["test"].shape == (2, 11)
    np.testing.assert_array_equal(captured["train"][:, :2], train[:, :2])
    np.testing.assert_array_equal(captured["test"][:, :2], test[:, :2])
    assert np.all(captured["train"][:, 2:] == 0) and np.all(captured["test"][:, 2:] == 0)
    assert np.isnan(train[:, 2:]).all() and np.all(test[:, 2:] != 0)


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_deterministic_fits_use_one_thread_and_closed_parameters(kind, observed_models,
                                                                    monkeypatch):
    fit_pools = []
    cls = Pipeline if kind == "history_ridge" else HistGradientBoostingRegressor
    original_fit = cls.fit

    def observe_fit(self, *args, **kwargs):
        fit_pools.append(threadpool_info())
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(cls, "fit", observe_fit)
    train, test = _matrix([0, 2, 4]), _matrix([0, 2])
    first = fit_magnitude_predictions(fitted_magnitude_recipe(kind), train, [1, 5, 9], test)
    second = fit_magnitude_predictions(fitted_magnitude_recipe(kind), train, [3, 15, 27], test)
    third = fit_magnitude_predictions(fitted_magnitude_recipe(kind), train, [1, 5, 9], test)
    assert second == pytest.approx(np.asarray(first) * 3)
    assert third == first
    assert len({id(model) for model, _ in observed_models}) == 3
    assert len(fit_pools) == 3
    assert all(pools and all(pool["num_threads"] == 1 for pool in pools) for pools in fit_pools)
    for model, pools in observed_models:
        assert pools and all(pool["num_threads"] == 1 for pool in pools)
        estimator = model[-1] if kind == "history_ridge" else model
        assert estimator.get_params(deep=False) == (RIDGE_PARAMS if kind == "history_ridge" else HGB_PARAMS)


@pytest.mark.parametrize("kind", KINDS)
def test_inputs_and_predictions_are_detached(kind):
    train, test, labels = _matrix([0, 2, None]), _matrix([0, 2]), np.array([1., 5., 9.])
    before = [array.copy() for array in (train, labels, test)]
    result = fit_magnitude_predictions(fitted_magnitude_recipe(kind), train, labels, test)
    assert isinstance(result, list) and all(type(value) is float for value in result)
    for actual, original in zip((train, labels, test), before):
        np.testing.assert_array_equal(actual, original)
    train[:] = test[:] = labels[:] = 999
    assert result == fit_magnitude_predictions(fitted_magnitude_recipe(kind), *before)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which,bad", [
    ("train", []), ("train", np.empty((0, 11))), ("train", np.zeros((2, 10))),
    ("train", np.zeros((2, 12))), ("train", np.zeros((2, 1, 11))),
    ("train", np.full((2, 11), np.inf)), ("train", np.full((2, 11), -np.inf)),
    ("test", []), ("test", np.empty((0, 11))), ("test", np.zeros((1, 10))),
    ("test", np.zeros((1, 12))), ("test", np.zeros((1, 1, 11))),
    ("test", np.full((1, 11), np.inf)), ("test", np.full((1, 11), -np.inf)),
    ("labels", []), ("labels", [1]), ("labels", [1, 2, 3]),
    ("labels", [[1], [2]]), ("labels", [1, np.nan]),
    ("labels", [1, np.inf]), ("labels", [1, -np.inf]), ("labels", [1, -1]),
])
def test_invalid_numeric_inputs_refuse(kind, which, bad, observed_models):
    inputs = {"train": _matrix([0, 2]), "labels": [1, 5], "test": _matrix([1])}
    inputs[which] = bad
    with pytest.raises(ValueError, match="magnitude"):
        fit_magnitude_predictions(fitted_magnitude_recipe(kind), inputs["train"], inputs["labels"], inputs["test"])
    assert not observed_models


def test_actual_negative_ridge_extrapolation_is_projected_to_zero():
    result = fit_magnitude_predictions(fitted_magnitude_recipe("history_ridge"), _matrix([0, 2, 4]),
                                       [1, 5, 9], _matrix([-100, 0]))
    assert result == pytest.approx([0, 2])


@pytest.mark.parametrize("kind,cls", [("history_ridge", Ridge),
                                     ("history_hgb", HistGradientBoostingRegressor)])
@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf], [[1]], [1, 2], 1.0])
def test_bad_predictions_refuse_before_nonnegative_projection(kind, cls, bad, monkeypatch):
    # Only corrupt inference output; all input validation and actual fitting still run.
    monkeypatch.setattr(cls, "predict", lambda self, matrix: bad)
    with pytest.raises(ValueError, match="predictions"):
        fit_magnitude_predictions(fitted_magnitude_recipe(kind), _matrix([0, 2, 4]), [1, 5, 9], _matrix([1]))
