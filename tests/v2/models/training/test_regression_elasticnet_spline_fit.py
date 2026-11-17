"""Synthetic signed spline fit wiring and evidence validation."""
import numpy as np
import pytest

from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    SPLINE_COLUMNS, elasticnet_spline_recipe, spline_coefficient_layout)
from engine.v2.models.regression_elasticnet_spline_recipe import (
    regression_elasticnet_spline_recipe)
from engine.v2.models.training.fit_inputs import FitInputs
from engine.v2.models.training.magnitude_elasticnet import fit_elasticnet_predictions
from engine.v2.models.training.regression_candidate import fit_regression_candidate
from engine.v2.ops.errors import OpsError
from experiments.size_fit_diagnostics import evidence

COLS = list(SPLINE_COLUMNS) + ["extra_a", "extra_b"]
H = "sha256:" + "0" * 64


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, len(COLS)))
    prediction = rng.normal(size=(20, len(COLS)))
    labels = rng.normal(loc=-5.0, scale=1.0, size=300)
    return train, labels, prediction


@pytest.fixture
def recipe():
    return regression_elasticnet_spline_recipe(
        target_domain="signed", columns=COLS, feature_dictionary_hash=H,
        alpha=0.1, l1_ratio=0.25, n_knots=4)


def _unsigned():
    return elasticnet_spline_recipe(columns=COLS, feature_dictionary_hash=H,
                                    alpha=0.1, l1_ratio=0.25, n_knots=4)


def signed_fit(recipe, arrays):
    train, labels, prediction = arrays
    return fit_regression_candidate(recipe, FitInputs(train, labels, prediction, COLS))


def test_signed_spline_fit_returns_finite_ordered_evidence(recipe, arrays):
    result = signed_fit(recipe, arrays)
    assert result["schema_version"] == "regression_elasticnet_fit.v1"
    predictions = np.asarray(result["predictions"])
    assert predictions.shape == (20,)
    assert np.isfinite(predictions).all()
    assert (predictions < 0).any()
    assert ([(entry["kind"], entry["column"]) for entry in result["coefficients"]]
            == spline_coefficient_layout(recipe))


def test_unsigned_spline_schema_is_refused_by_the_regression_dispatch(arrays):
    unsigned = _unsigned()
    assert unsigned["schema_version"] == "magnitude_elasticnet_spline_recipe.v1"
    train, labels, prediction = arrays
    with pytest.raises(ValueError, match="supported recipe schema/model pair"):
        fit_regression_candidate(unsigned, FitInputs(train, labels, prediction, COLS))


def test_signed_accepts_negative_labels_while_unsigned_requires_nonnegative(recipe, arrays):
    assert signed_fit(recipe, arrays)["converged"] is True
    train, labels, prediction = arrays
    with pytest.raises(ValueError, match="nonnegative"):
        fit_elasticnet_predictions(_unsigned(), train, labels, prediction, columns=COLS)


def test_signed_spline_fit_tracks_an_independent_linear_signal():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(400, len(COLS)))
    prediction = rng.normal(size=(30, len(COLS)))
    truth = lambda x: -5.0 + 2.0 * x[:, 0]
    labels = truth(train)
    expected = truth(prediction)
    recipe = regression_elasticnet_spline_recipe(
        target_domain="signed", columns=COLS, feature_dictionary_hash=H,
        alpha=1e-4, l1_ratio=0.5, n_knots=4)
    result = fit_regression_candidate(recipe, FitInputs(train, labels, prediction, COLS))
    predictions = np.asarray(result["predictions"])
    assert np.max(np.abs(predictions - expected)) < 0.5
    deep = int(np.argmin(expected))
    assert expected[deep] < -5.0
    assert predictions[deep] < -4.0


def test_evidence_accepts_signed_diagnostics_and_refuses_a_swapped_kind(recipe, arrays):
    result = signed_fit(recipe, arrays)
    detail = {"horizon_sessions": 1,
              **{key: value for key, value in result.items() if key != "predictions"}}
    execution = {"schema_version": "regression_candidate_execution.v1",
                 "fit_recipe": recipe}
    document = {"event": "succeeded", "diagnostics": [detail]}
    assert evidence(document, execution, [1]) == {"diagnostics": document["diagnostics"]}
    document["diagnostics"][0]["coefficients"][0]["kind"] = "missing_indicator"
    with pytest.raises(OpsError) as caught:
        evidence(document, execution, [1])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
