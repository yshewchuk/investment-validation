"""v9 HGB loss/max_iter axes and the gamma-only training-target floor."""
import numpy as np
import pytest

from engine.v2.models.magnitude_ridge_hgb_recipe import (
    ridge_hgb_recipe, validate_ridge_hgb_recipe)
from engine.v2.models.training import magnitude_ridge_hgb
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe as qualified_recipe
from experiments import magnitude_search_request as compiler
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v6 import v8_request

HASH = "sha256:" + "a" * 64
HGB_BASE = {"learning_rate": 0.05, "max_leaf_nodes": 7,
            "min_samples_leaf": 20, "l2_regularization": 0.0}


def v9_request(**hgb_axes):
    request = v8_request(profiles=["chrono15_relu32_l2_0p01"])
    request["schema_version"] = "magnitude_search_request.v9"
    request["feature_sets"] = ["HMO"]
    request["feature_dictionary_hash"] = qualified_recipe("HMO")["dictionary_hash"]
    request["space"]["hgb"] = {**request["space"]["hgb"], **hgb_axes}
    request["max_candidates"] = 1000
    request["max_fits"] = 100_000
    return request


def hgb_recipe(**overrides):
    return ridge_hgb_recipe(model="history_hgb", columns=["x0"],
                            feature_dictionary_hash=HASH, loss=overrides.pop("loss", None),
                            max_iter=overrides.pop("max_iter", None),
                            target_floor_pct=overrides.pop("target_floor_pct", None),
                            **HGB_BASE, **overrides)


def test_v9_hgb_optional_axes_expand_candidates_and_default_only_when_declared():
    baseline = compile_request(v9_request())
    assert baseline["counts"]["candidates"] == 35
    assert "loss" not in baseline["request"]["space"]["hgb"]
    assert "max_iter" not in baseline["request"]["space"]["hgb"]

    loss_only = compile_request(v9_request(loss=["gamma"]))
    assert loss_only["counts"]["candidates"] == 35
    assert loss_only["request"]["space"]["hgb"]["loss"] == ["gamma"]
    assert loss_only["request"]["space"]["hgb"]["max_iter"] == [500]

    both = compile_request(v9_request(loss=["gamma", "squared_error"],
                                      max_iter=[100, 500]))
    assert both["counts"]["candidates"] == 83
    hgb = [spec for spec in both["candidate_specs"] if spec["family"] == "hgb"]
    assert len(hgb) == 64
    assert {(spec["parameters"]["loss"], spec["parameters"]["max_iter"]) for spec in hgb} == {
        (loss, max_iter) for loss in ("gamma", "squared_error") for max_iter in (100, 500)}
    assert both["request"]["space"]["hgb"]["loss"] == ["gamma", "squared_error"]
    assert both["request"]["space"]["hgb"]["max_iter"] == [100, 500]

    for spec in hgb:
        recipe = candidate_recipe("hgb", spec["parameters"], columns=["x0"],
                                  feature_dictionary_hash=HASH)
        assert recipe["estimator"]["params"]["loss"] == spec["parameters"]["loss"]
        assert recipe["estimator"]["params"]["max_iter"] == spec["parameters"]["max_iter"]
        assert recipe["schema_version"] == "magnitude_ridge_hgb_recipe.v2"
        assert ("target_floor_pct" in recipe) is (spec["parameters"]["loss"] == "gamma")
        if spec["parameters"]["loss"] == "gamma":
            assert recipe["target_floor_pct"] == 0.01


@pytest.mark.parametrize("overrides", [
    {"loss": []}, {"loss": ["unknown"]}, {"loss": ["gamma", "gamma"]},
    {"loss": ["squared_error", "gamma"]}, {"max_iter": []}, {"max_iter": [True]},
    {"max_iter": [100.0]}, {"max_iter": [99]}, {"max_iter": [5001]},
    {"max_iter": [500, 100]}, {"max_iter": [100, 100]},
])
def test_v9_refuses_invalid_optional_axes_before_expansion(monkeypatch, overrides):
    monkeypatch.setattr(compiler, "_specifications",
                        lambda *args, **kwargs: pytest.fail("invalid axes expanded"))
    with pytest.raises(OpsError) as caught:
        compile_request(v9_request(**overrides))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_gamma_floor_clips_training_labels_and_other_losses_do_not(monkeypatch):
    captured = {}

    class Model:
        def fit(self, features, labels):
            captured["labels"] = np.asarray(labels).copy()

        def predict(self, features):
            return np.zeros(len(features))

    monkeypatch.setattr(magnitude_ridge_hgb, "build_estimator", lambda recipe: Model())
    train, labels, prediction = [[0.0], [1.0], [2.0]], [0.0, 0.5, 1.0], [[0.0], [1.0]]

    gamma = hgb_recipe(loss="gamma")
    assert gamma["target_floor_pct"] == 0.01
    magnitude_ridge_hgb.fit_ridge_hgb_predictions(
        gamma, train, labels, prediction, columns=["x0"])
    np.testing.assert_array_equal(captured["labels"], [0.01, 0.5, 1.0])

    squared = hgb_recipe(loss="squared_error")
    assert "target_floor_pct" not in squared
    magnitude_ridge_hgb.fit_ridge_hgb_predictions(
        squared, train, labels, prediction, columns=["x0"])
    np.testing.assert_array_equal(captured["labels"], labels)


def test_gamma_recipe_requires_the_fixed_floor_and_other_losses_refuse_it():
    gamma = hgb_recipe(loss="gamma")
    without_floor = {key: value for key, value in gamma.items() if key != "target_floor_pct"}
    with pytest.raises(ValueError):
        validate_ridge_hgb_recipe(without_floor)
    with pytest.raises(OpsError) as caught:
        fit_candidate(without_floor, [[0.0]], [0.0], [[0.0]], columns=["x0"])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"

    with pytest.raises(ValueError, match="^gamma HGB requires target_floor_pct 0.01$"):
        hgb_recipe(loss="gamma", target_floor_pct=0.02)
    with pytest.raises(ValueError, match="^target_floor_pct is only valid for gamma HGB loss$"):
        hgb_recipe(loss="poisson", target_floor_pct=0.01)


@pytest.mark.parametrize("mutate", [
    {"schema_version": "magnitude_ridge_hgb_recipe.v3"}, {"model": "unsupported"}])
def test_gamma_floor_refusal_precedes_family_validation(mutate):
    without_floor = {key: value for key, value in hgb_recipe(loss="gamma").items()
                     if key != "target_floor_pct"}
    with pytest.raises(OpsError) as caught:
        fit_candidate({**without_floor, **mutate}, [[0.0]], [0.0], [[0.0]], columns=["x0"])
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
