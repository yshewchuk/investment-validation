"""Synthetic variable-width fit oracles; no private data or execution admission."""
import copy
import json

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_info, threadpool_limits

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models import no_fit
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe
from engine.v2.models.training.magnitude import fit_magnitude_predictions
from engine.v2.models.training.magnitude_ridge_hgb import fit_ridge_hgb_predictions
from engine.v2.ops.native_registration import magnitude_fitted_execution_document

KINDS = ("history_ridge", "history_hgb")


def _recipe(kind="history_ridge", width=3, **knobs):
    return ridge_hgb_recipe(model=kind, columns=[f"x{i}" for i in range(width)],
        feature_dictionary_hash="sha256:" + "a" * 64,
        **(knobs or ({"alpha": 1} if kind == "history_ridge" else {"max_leaf_nodes": 7})))


def _matrix(values, width=3):
    matrix = np.full((len(values), width), np.nan)
    matrix[:, 0] = values
    if width > 1:
        matrix[:, 1] = 7
    return matrix


def _fit(recipe, train, labels, prediction):
    return fit_ridge_hgb_predictions(recipe, train, labels, prediction, columns=recipe["features"])


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("numeric input inspected before refusal")

    def __getitem__(self, key):
        pytest.fail("recipe inspected before no-fit guard")

    def __iter__(self):
        pytest.fail("columns inspected before no-fit guard")


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
def test_both_no_fit_guards_precede_all_inspection(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden, match="fit_ridge_hgb_predictions"):
        fit_ridge_hgb_predictions(Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("kind", KINDS)
def test_order_and_policy_refusal_precedes_numeric_conversion(kind):
    recipe = _recipe(kind)
    for columns in (["x1", "x0", "x2"], ["x0"], "x0", None):
        with pytest.raises(ValueError, match="columns"):
            fit_ridge_hgb_predictions(recipe, Poison(), Poison(), Poison(), columns=columns)
    recipe["numerical_threads"] = 2
    with pytest.raises(ValueError, match="recipe"):
        _fit(recipe, Poison(), Poison(), Poison())
    with pytest.raises(ValueError, match="recipe"):
        fit_ridge_hgb_predictions(magnitude_fitted_execution_document("2026-01-01", kind)["fit_recipe"], Poison(), Poison(), Poison(), columns=[])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", [1, 3, 11, 28, 42, 43])
def test_arbitrary_width_original_units_and_recipe_bound_output(kind, width):
    recipe = _recipe(kind, width)
    result = _fit(recipe, _matrix([0, 2, None], width), [1, 5, 9], _matrix([0, 2, None, 100], width))
    assert set(result) == {"schema_version", "recipe_id", "predictions"}
    assert result["schema_version"] == "magnitude_ridge_hgb_fit.v1"
    assert result["recipe_id"] == content_hash(recipe)
    expected = [2, 5, 8, 152] if kind == "history_ridge" else [5] * 4
    assert result["predictions"] == pytest.approx(expected, abs=1e-10)
    assert all(type(value) is float for value in result["predictions"])



@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("width", [3, 28, 42, 43])
def test_first_and_last_selected_columns_both_drive_predictions(kind, width):
    # Four balanced corners: two independent signals, including a column beyond H.
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    prediction = np.full((4, width), 7.0)
    prediction[:, [0, -1]] = corners
    train = np.repeat(prediction, 20, axis=0)
    means = np.array([2., 4., 8., 10.])
    result = _fit(_recipe(kind, width), train, np.repeat(means, 20), prediction)
    # Independent Ridge shrinkage and four HGB leaves' residual contraction.
    shrinkage = 80 / 81 if kind == "history_ridge" else 1 - 0.9**100
    assert result["predictions"] == pytest.approx(6 + (means - 6) * shrinkage, abs=1e-6)


@pytest.mark.parametrize("kind", KINDS)
def test_genuine_zero_labels_remain_valid(kind):
    result = _fit(_recipe(kind, 28), np.zeros((3, 28)), [0, 0, 0], np.ones((2, 28)))
    assert result["predictions"] == [0., 0.]


@pytest.mark.parametrize("alpha", [0.1, 1, 10])
def test_ridge_closed_form_for_every_declared_alpha(alpha):
    slope = 2 * 3 / (3 + alpha)
    result = _fit(_recipe(alpha=alpha), _matrix([0, 2, 4]), [1, 5, 9], _matrix([-100, 0, 2, 4]))
    assert result["predictions"] == pytest.approx([0, 5 - 2 * slope, 5, 5 + 2 * slope], abs=1e-10)


@pytest.mark.parametrize("leaves", [7, 15])
def test_hgb_fixed_iteration_two_leaf_oracle(leaves):
    result = _fit(_recipe("history_hgb", max_leaf_nodes=leaves),
                  _matrix([0] * 20 + [1] * 20), [2] * 20 + [8] * 20, _matrix([0, 1]))
    assert result["predictions"] == pytest.approx([2 + 3 * 0.9**100, 8 - 3 * 0.9**100], abs=1e-6)


def test_early_stopping_v2_recipe_fits_deterministically():
    rng = np.random.default_rng(20261030)
    train = rng.normal(size=(200, 3))
    labels = np.abs(rng.normal(size=200))
    prediction = rng.normal(size=(25, 3))
    recipe = ridge_hgb_recipe(model="history_hgb", columns=["x0", "x1", "x2"],
        feature_dictionary_hash="sha256:" + "a" * 64, learning_rate=0.05, max_leaf_nodes=7,
        min_samples_leaf=20, l2_regularization=0.0)
    first = _fit(recipe, train, labels, prediction)
    second = _fit(recipe, train, labels, prediction)
    assert first == second
    assert first["schema_version"] == "magnitude_ridge_hgb_fit.v1"
    assert first["recipe_id"] == content_hash(recipe)
    values = first["predictions"]
    assert len(values) == len(prediction)
    assert all(np.isfinite(values)) and all(value >= 0 for value in values)


def test_early_stopping_v2_matches_independent_estimator_and_recovers_signal():
    rng = np.random.default_rng(20261031)
    train = rng.normal(size=(400, 3))
    labels = np.where(train[:, 0] < 0, 2.0, 8.0)
    prediction = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    recipe = ridge_hgb_recipe(model="history_hgb", columns=["x0", "x1", "x2"],
        feature_dictionary_hash="sha256:" + "a" * 64, learning_rate=0.1, max_leaf_nodes=7,
        min_samples_leaf=20, l2_regularization=0.0)
    result = _fit(recipe, train, labels, prediction)
    expected = HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.1, max_iter=500,
        max_leaf_nodes=7, min_samples_leaf=20, l2_regularization=0.0, early_stopping=True,
        n_iter_no_change=10, validation_fraction=0.1, tol=1e-7, random_state=0)
    with threadpool_limits(limits=1):
        expected.fit(train, labels)
    assert result["predictions"] == pytest.approx(expected.predict(prediction).tolist(), abs=1e-9)
    assert result["predictions"] == pytest.approx([2.0, 8.0], abs=0.3)
    assert expected.n_iter_ < 500


@pytest.mark.parametrize("kind", KINDS)
def test_eleven_column_numeric_parity_with_unchanged_v1(kind):
    old = magnitude_fitted_execution_document("2026-01-01", kind)["fit_recipe"]
    recipe = ridge_hgb_recipe(model=kind, columns=old["features"],
        feature_dictionary_hash="sha256:" + "a" * 64,
        **({"alpha": 1} if kind == "history_ridge" else {"max_leaf_nodes": 7}))
    train, prediction = _matrix([0] * 20 + [1] * 20 + [None] * 20, 11), _matrix([0, 1, None, -100], 11)
    labels = [2] * 20 + [8] * 20 + [3] * 20
    assert _fit(recipe, train, labels, prediction)["predictions"] == fit_magnitude_predictions(
        old, train, labels, prediction)


@pytest.fixture
def observed(monkeypatch):
    calls = []
    for cls in (Pipeline, HistGradientBoostingRegressor):
        for method in ("fit", "predict"):
            original = getattr(cls, method)

            def observe(self, matrix, *args, _original=original, _method=method, **kwargs):
                calls.append((_method, self, matrix.copy(), threadpool_info()))
                return _original(self, matrix, *args, **kwargs)

            monkeypatch.setattr(cls, method, observe)
    return calls


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_single_thread_train_only_fits_and_detached_inputs(kind, observed):
    recipe = json.loads(canonical_json(_recipe(kind, 28)))
    before_recipe = copy.deepcopy(recipe)
    train, labels, prediction = _matrix([0, 2, None], 28), np.array([1, 5, 9]), _matrix([1], 28)
    before = [array.copy() for array in (train, labels, prediction)]
    first = _fit(recipe, train, labels, prediction)
    poisoned = _matrix([1, 1e12, None], 28)
    poisoned[0, 1:] = 1e12
    second = _fit(recipe, train, labels, poisoned)
    third = _fit(recipe, train, labels * 3, prediction)
    assert first["predictions"] == [second["predictions"][0]]
    assert third["predictions"] == pytest.approx(np.array(first["predictions"]) * 3)
    assert recipe == before_recipe
    for actual, original in zip((train, labels, prediction), before):
        np.testing.assert_array_equal(actual, original)
    models = [model for method, model, _, _ in observed if method == "fit"]
    assert len(models) == len({id(model) for model in models}) == 3
    assert all(pools and all(pool["num_threads"] == 1 for pool in pools) for _, _, _, pools in observed)
    for model in models:
        estimator = model[-1] if kind == "history_ridge" else model
        assert estimator.get_params(deep=False) == recipe["estimator"]["params"]
    if kind == "history_ridge":
        for model in models:
            union, scaler, ridge = (step for _, step in model.steps)
            imputer, indicator = (step for _, step in union.transformer_list)
            assert imputer.statistics_ == pytest.approx([1, 7] + [0] * 26)
            assert indicator.features_.tolist() == list(range(28))
            assert union.transform(poisoned).shape == (3, 56)
            assert scaler.n_features_in_ == ridge.n_features_in_ == 56
        np.testing.assert_array_equal(models[0][1].mean_, models[1][1].mean_)
        np.testing.assert_array_equal(models[0][1].scale_, models[1][1].scale_)
    train[:] = labels[:] = prediction[:] = 999
    assert first == _fit(recipe, *before)


def test_hgb_frozen_training_empty_mask_and_native_partial_nulls(observed):
    train, prediction = _matrix([0] * 20 + [None] * 20), _matrix([0, None])
    prediction[:, 2] = [1e12, -1e12]
    result = _fit(_recipe("history_hgb"), train, [2] * 20 + [8] * 20, prediction)
    assert result["predictions"] == pytest.approx([2 + 3 * 0.9**100, 8 - 3 * 0.9**100], abs=1e-6)
    for (_, _, actual, _), original in zip(observed, (train, prediction)):
        np.testing.assert_array_equal(actual[:, :2], original[:, :2])
        assert np.all(actual[:, 2] == 0)
    assert np.isnan(train[:, 2]).all() and np.all(prediction[:, 2] != 0)


@pytest.mark.parametrize("kind", KINDS)
def test_all_null_training_is_frozen_even_when_prediction_is_complete(kind):
    result = _fit(_recipe(kind, 42), np.full((3, 42), np.nan), [1, 1, 9], np.ones((2, 42)))
    assert result["predictions"] == pytest.approx([11 / 3, 11 / 3])


def test_ridge_full_flags_allow_new_prediction_missingness(observed):
    result = _fit(_recipe(), _matrix([0, 2, 4]), [1, 5, 9], _matrix([None]))
    assert result["predictions"] == pytest.approx([5])
    union = observed[0][1][0]
    assert union.transform(_matrix([None]))[0, 3] == 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which,bad", [
    ("train", []), ("train", np.zeros((0, 3))), ("train", np.zeros((2, 2))),
    ("train", np.zeros((2, 4))), ("train", np.zeros((2, 1, 3))),
    ("train", np.full((2, 3), np.inf)), ("train", np.full((2, 3), -np.inf)),
    ("prediction", []), ("prediction", np.zeros((0, 3))), ("prediction", np.zeros((1, 2))),
    ("prediction", np.zeros((1, 4))), ("prediction", np.zeros((1, 1, 3))),
    ("prediction", np.full((1, 3), np.inf)), ("prediction", np.full((1, 3), -np.inf)),
    ("labels", []), ("labels", [1]), ("labels", [1, 2, 3]), ("labels", [[1], [2]]),
    ("labels", [1, np.nan]), ("labels", [1, np.inf]), ("labels", [1, -np.inf]), ("labels", [1, -1]),
])
def test_invalid_numeric_inputs_refuse_before_estimator_fit(kind, which, bad, observed):
    inputs = {"train": _matrix([0, 2]), "labels": [1, 5], "prediction": _matrix([1])}
    inputs[which] = bad
    with pytest.raises(ValueError, match="magnitude"):
        _fit(_recipe(kind), **inputs)
    assert not observed


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("which", ["train", "labels", "prediction"])
@pytest.mark.parametrize("bad,cause", [
    pytest.param(10**400, OverflowError, id="oversized-numeric"),
    pytest.param(object(), TypeError, id="not-convertible"),
])
def test_conversion_failures_refuse_as_value_error_before_fit(kind, which, bad, cause, observed):
    inputs = {"train": [[0]], "labels": [1], "prediction": [[1]]}
    inputs[which] = [bad] if which == "labels" else [[bad]]
    with pytest.raises(ValueError, match="magnitude inputs must be convertible to floating-point values") as exc:
        _fit(_recipe(kind, width=1), **inputs)
    assert type(exc.value.__cause__) is cause
    assert not observed


@pytest.mark.parametrize("kind,cls", [("history_ridge", Ridge), ("history_hgb", HistGradientBoostingRegressor)])
@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf], [[1]], [1, 2], 1.0])
def test_bad_predictions_refuse_after_real_fit(kind, cls, bad, monkeypatch):
    monkeypatch.setattr(cls, "predict", lambda self, matrix: bad)
    with pytest.raises(ValueError, match="predictions"):
        _fit(_recipe(kind), _matrix([0, 2, 4]), [1, 5, 9], _matrix([1]))


@pytest.mark.parametrize("kind,cls", [("history_ridge", Ridge), ("history_hgb", HistGradientBoostingRegressor)])
def test_clip_only_finite_predictions_without_mutation(kind, cls, monkeypatch):
    raw = np.array([-3.5, 0, 2.25, -0.125])
    monkeypatch.setattr(cls, "predict", lambda self, matrix: raw)
    result = _fit(_recipe(kind), _matrix([0, 2, 4]), [1, 5, 9], _matrix([0, 1, 2, 4]))
    assert result["predictions"] == [0., 0., 2.25, 0.]
    np.testing.assert_array_equal(raw, [-3.5, 0, 2.25, -0.125])
