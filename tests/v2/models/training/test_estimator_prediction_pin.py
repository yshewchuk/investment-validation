"""Exact prediction pins for the current loose-argument estimator fit APIs."""
import numpy as np

from engine.v2.models.magnitude_elasticnet_recipe import (
    elasticnet_recipe,
    regression_elasticnet_recipe,
)
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe
from engine.v2.models.regression_ridge_hgb_recipe import regression_recipe
from engine.v2.models.training.magnitude_elasticnet import (
    fit_elasticnet_predictions,
    fit_regression_elasticnet_predictions,
)
from engine.v2.models.training.magnitude_mlp import fit_mlp_predictions
from engine.v2.models.training.magnitude_ridge_hgb import (
    fit_regression_predictions,
    fit_ridge_hgb_predictions,
)

COLUMNS = ["signal", "constant", "empty"]
DICTIONARY = "sha256:" + "1" * 64


def _prediction():
    return np.asarray([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]], dtype=float)


def test_magnitude_elasticnet_prediction_pin():
    recipe = elasticnet_recipe(columns=COLUMNS, feature_dictionary_hash=DICTIONARY,
                               alpha=0.1, l1_ratio=0.75)
    result = fit_elasticnet_predictions(
        recipe, np.full((3, 3), np.nan), [1, 5, 9], _prediction(), columns=COLUMNS)
    assert result["predictions"] == [5.0, 5.0]


def test_regression_elasticnet_prediction_pin():
    recipe = regression_elasticnet_recipe(
        target_domain="signed", columns=COLUMNS, feature_dictionary_hash=DICTIONARY,
        alpha=0.1, l1_ratio=0.75)
    result = fit_regression_elasticnet_predictions(
        recipe, np.full((3, 3), np.nan), [-4, 0, 4], _prediction(), columns=COLUMNS)
    assert result["predictions"] == [0.0, 0.0]


def test_magnitude_ridge_prediction_pin():
    recipe = ridge_hgb_recipe(
        model="history_ridge", columns=COLUMNS, feature_dictionary_hash=DICTIONARY, alpha=1.0)
    result = fit_ridge_hgb_predictions(
        recipe, np.full((3, 3), np.nan), [1, 5, 9], _prediction(), columns=COLUMNS)
    assert result["predictions"] == [5.0, 5.0]


def test_magnitude_hgb_prediction_pin():
    recipe = ridge_hgb_recipe(
        model="history_hgb", columns=COLUMNS, feature_dictionary_hash=DICTIONARY,
        max_leaf_nodes=7)
    result = fit_ridge_hgb_predictions(
        recipe, np.full((3, 3), np.nan), [2, 2, 8], _prediction(), columns=COLUMNS)
    assert result["predictions"] == [4.0, 4.0]


def test_regression_ridge_prediction_pin():
    recipe = regression_recipe(
        target_domain="signed", model="history_ridge", columns=COLUMNS,
        feature_dictionary_hash=DICTIONARY, alpha=1.0)
    result = fit_regression_predictions(
        recipe, np.full((3, 3), np.nan), [-4, 0, 4], _prediction(), columns=COLUMNS)
    assert result["predictions"] == [0.0, 0.0]


def test_regression_hgb_prediction_pin():
    recipe = regression_recipe(
        target_domain="signed", model="history_hgb", columns=COLUMNS,
        feature_dictionary_hash=DICTIONARY, max_leaf_nodes=7)
    result = fit_regression_predictions(
        recipe, np.full((3, 3), np.nan), [-4, 0, 4], _prediction(), columns=COLUMNS)
    assert result["predictions"] == [0.0, 0.0]


def test_mlp_prediction_pin():
    signal = np.linspace(-2, 2, 80)
    train = np.column_stack([signal, np.full(80, 7.0), np.full(80, np.nan)])
    labels = 10 + 2 * signal
    prediction = np.asarray([[0.0, 7.0, np.nan], [1.0, 7.0, 3.0]], dtype=float)
    recipe = mlp_recipe(columns=COLUMNS, feature_dictionary_hash=DICTIONARY,
                        profile="relu32_l2_0p01")
    result = fit_mlp_predictions(
        recipe, train, labels, prediction, columns=COLUMNS)
    assert result["predictions"] == [10.04295718363434, 12.039117667185115]
