"""Synthetic signed/nonnegative MLP fit wiring and evidence validation."""
import warnings
from datetime import date, timedelta

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_mlp_recipe import mlp_recipe
from engine.v2.models.regression_mlp_recipe import regression_mlp_recipe
from engine.v2.models.training import magnitude_mlp
from engine.v2.models.training.fit_inputs import FitInputs
from engine.v2.models.training.magnitude_mlp import fit_mlp_predictions
from engine.v2.models.training.regression_candidate import fit_regression_candidate
from engine.v2.ops.errors import OpsError
from experiments.size_fit_diagnostics import evidence

COLS = ["a", "b", "c", "d", "e", "f"]
H = "sha256:" + "0" * 64
PROFILE = "fixed200_relu64x32_l2_1"


def _recipe(domain="signed", profile=PROFILE):
    return regression_mlp_recipe(target_domain=domain, columns=COLS,
                                 feature_dictionary_hash=H, profile=profile)


def _dates(size):
    return [(date(2024, 1, 1) + timedelta(days=index)).isoformat()
            for index in range(size)]


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(120, len(COLS)))
    prediction = rng.normal(size=(20, len(COLS)))
    labels = rng.normal(loc=-5.0, scale=1.0, size=120)
    return train, labels, prediction


def _fit(recipe, values, **kwargs):
    train, labels, prediction = values
    return fit_regression_candidate(
        recipe, FitInputs(train, labels, prediction, COLS, **kwargs))


def test_signed_fit_returns_finite_signed_predictions(arrays):
    recipe = _recipe()
    result = _fit(recipe, arrays)
    predictions = np.asarray(result["predictions"])
    assert result["schema_version"] == "regression_mlp_fit.v1"
    assert predictions.shape == (20,)
    assert np.isfinite(predictions).all()
    assert (predictions < 0).any()
    assert result["recipe_id"] == content_hash(recipe)


def test_nonnegative_domain_rejects_negative_labels_and_clips(arrays):
    train, labels, prediction = arrays
    recipe = _recipe("nonnegative")
    with pytest.raises(ValueError, match="nonnegative"):
        _fit(recipe, arrays)
    result = _fit(recipe, (train, np.abs(labels), prediction))
    assert (np.asarray(result["predictions"]) >= 0).all()


def test_magnitude_fit_still_rejects_negative_labels(arrays):
    train, labels, prediction = arrays
    magnitude = mlp_recipe(columns=COLS, feature_dictionary_hash=H, profile=PROFILE)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_mlp_predictions(magnitude, train, labels, prediction, columns=COLS)


def test_chrono_profile_requires_training_event_dates(arrays):
    recipe = _recipe(profile="chrono15_relu32_l2_0p01")
    with pytest.raises(ValueError, match="event dates"):
        _fit(recipe, arrays)
    result = _fit(recipe, arrays, train_event_dates=_dates(len(arrays[0])))
    assert result["schema_version"] == "regression_mlp_fit.v1"


def test_evidence_accepts_regression_mlp_diagnostics_and_refuses_a_swapped_kind(arrays):
    recipe = _recipe()
    result = _fit(recipe, arrays)
    detail = {"horizon_sessions": 1,
              **{key: value for key, value in result.items() if key != "predictions"}}
    execution = {"schema_version": "regression_candidate_execution.v1",
                 "fit_recipe": recipe}
    document = {"event": "succeeded", "diagnostics": [detail]}
    assert evidence(document, execution, [1]) == {"diagnostics": [detail]}
    with pytest.raises(OpsError) as caught:
        evidence(document, {**execution, "schema_version": "size_candidate_execution.v1"}, [1])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_variant_failure_profile_refuses_a_convergence_warning(monkeypatch, arrays):
    class Network:
        n_iter_ = 1

    class Stub:
        regressor_ = [Network()]

        def fit(self, features, labels):
            warnings.warn("forced warning", ConvergenceWarning)

    monkeypatch.setattr(magnitude_mlp, "build_estimator", lambda recipe: Stub())
    with pytest.raises(ValueError, match="convergence failed; no retry"):
        _fit(_recipe(profile="relu32_l2_0p01"), arrays)
