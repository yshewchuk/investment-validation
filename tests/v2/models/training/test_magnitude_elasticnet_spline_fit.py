"""Synthetic spline-fit wiring: evidence order, refusals, logging and dispatch."""
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from engine.v2.data.errors import DataError
from engine.v2.models.magnitude_candidate import magnitude_candidate_family
from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    SPLINE_COLUMNS, elasticnet_spline_recipe)
from engine.v2.models.training import (
    _magnitude_elasticnet_spline_fit as spline_fit)
from engine.v2.models.training._magnitude_elasticnet_estimator import build_estimator
from engine.v2.models.training.magnitude_elasticnet import fit_elasticnet_predictions

COLS = list(SPLINE_COLUMNS) + ["extra_a", "extra_b"]
H = "sha256:" + "0" * 64


@pytest.fixture
def spline_inputs():
    recipe = elasticnet_spline_recipe(columns=COLS, feature_dictionary_hash=H,
                                      alpha=0.1, l1_ratio=0.25, n_knots=4)
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, len(COLS)))
    prediction = rng.normal(size=(20, len(COLS)))
    labels = np.abs(rng.normal(size=300)) + 0.1
    return recipe, train, labels, prediction


def fit(recipe, train, labels, prediction):
    return fit_elasticnet_predictions(recipe, train, labels, prediction, columns=COLS)


def test_spline_fit_returns_finite_ordered_evidence(spline_inputs):
    recipe, train, labels, prediction = spline_inputs
    result = fit(recipe, train, labels, prediction)
    assert np.isfinite(result["predictions"]).all()
    coefficients = result["coefficients"]
    assert len(coefficients) == recipe["preprocessing"]["output_features"]
    kinds = [entry["kind"] for entry in coefficients]
    assert kinds.index("value") < kinds.index("spline") < kinds.index("missing_indicator")
    assert build_estimator(recipe)[-1].precompute is True


def test_tied_training_column_refuses_the_variant(spline_inputs):
    recipe, train, labels, prediction = spline_inputs
    train[:200, 0] = 0.0
    with pytest.raises(DataError) as raised:
        fit(recipe, train, labels, prediction)
    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_spline_convergence_warning_refuses_without_retry(monkeypatch, spline_inputs):
    recipe, train, labels, prediction = spline_inputs
    original_fit, calls = ElasticNet.fit, []

    def warns(self, x, y, **kwargs):
        calls.append(1)
        original_fit(self, x, y, **kwargs)
        warnings.warn("x", ConvergenceWarning)
        return self

    monkeypatch.setattr(ElasticNet, "fit", warns)
    with pytest.raises(DataError) as raised:
        fit(recipe, train, labels, prediction)
    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert calls == [1]


def test_spline_fit_logs_first_fit_only(monkeypatch, capsys, spline_inputs):
    recipe, train, labels, prediction = spline_inputs
    monkeypatch.setattr(spline_fit, "_SPLINE_FITS", 0)
    fit(recipe, train, labels, prediction)
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("[spline-fit]")]
    assert len(lines) == 1
    assert "spline_knots=4" in lines[0] and "inputs=" in lines[0]
    fit(recipe, train, labels, prediction)
    assert "[spline-fit]" not in capsys.readouterr().err


def test_candidate_family_accepts_spline_recipe(spline_inputs):
    recipe, _, _, _ = spline_inputs
    assert magnitude_candidate_family(recipe) == "elasticnet"
