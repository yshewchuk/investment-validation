"""Synthetic signed-domain oracles, without source reads or experiment admission."""
import copy
import json
import subprocess
import sys

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models import no_fit
from engine.v2.models.magnitude_ridge_hgb_recipe import ridge_hgb_recipe, validate_ridge_hgb_recipe
from engine.v2.models.regression_ridge_hgb_recipe import regression_recipe, validate_regression_recipe
from engine.v2.models.training.magnitude_ridge_hgb import fit_regression_predictions, fit_ridge_hgb_predictions


def recipe(model="history_ridge", domain="signed"):
    return regression_recipe(target_domain=domain, model=model, columns=["x", "empty"],
        feature_dictionary_hash="sha256:" + "a" * 64,
        **({"alpha": 1} if model == "history_ridge" else {"max_leaf_nodes": 7}))


def matrix(values):
    return np.column_stack([values, np.full(len(values), np.nan)])


def fit(policy, train, labels, prediction):
    return fit_regression_predictions(policy, train, labels, prediction, columns=policy["features"])


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
def test_signed_predictions_have_both_signs_and_no_floor(model):
    policy = json.loads(canonical_json(recipe(model)))
    train, prediction = matrix([0] * 20 + [1] * 20), matrix([0, 1])
    labels = np.array([-200.] * 20 + [20.] * 20)
    before = [v.copy() for v in (train, labels, prediction)]
    original = copy.deepcopy(policy)
    result = fit(policy, train, labels, prediction)
    shrinkage = 40 / 41 if model == "history_ridge" else 1 - 0.9**100
    assert result["predictions"] == pytest.approx(-90 + np.array([-110, 110]) * shrinkage, abs=1e-5)
    assert result["predictions"][0] < -100  # The existing IV recipe permits this.
    assert result["predictions"][1] > 0
    assert result["schema_version"] == "regression_ridge_hgb_fit.v1"
    assert result["recipe_id"] == content_hash(policy)
    assert policy == original
    for actual, expected in zip((train, labels, prediction), before):
        np.testing.assert_array_equal(actual, expected)


def test_signed_ridge_extrapolates_without_clipping():
    result = fit(recipe(), matrix([0, 2, 4]), [-4, 0, 4], matrix([-100, 2, 100]))
    assert result["predictions"] == pytest.approx([-153, 0, 147], abs=1e-10)


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
@pytest.mark.parametrize("value", [-5, 0, 7])
def test_constant_signed_labels_are_preserved(model, value):
    result = fit(recipe(model), matrix([0, 1, 2]), [value] * 3, matrix([0, 3]))
    assert result["predictions"] == pytest.approx([value, value])


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
def test_nonnegative_numerics_match_legacy_and_both_reject_signed_labels(model):
    policy = recipe(model, "nonnegative")
    old = {k: v for k, v in policy.items() if k != "target_domain"}
    old["schema_version"] = "magnitude_ridge_hgb_recipe.v1"
    assert validate_ridge_hgb_recipe(old) == old
    args = (matrix([0, 2, 4]), [1, 5, 9], matrix([-100, 1, 100]))
    expected = fit_ridge_hgb_predictions(old, *args, columns=old["features"])
    assert fit(policy, *args)["predictions"] == expected["predictions"]
    for function, metadata in ((fit_regression_predictions, policy), (fit_ridge_hgb_predictions, old)):
        with pytest.raises(ValueError, match="nonnegative"):
            function(metadata, matrix([0]), [-1], matrix([0]), columns=metadata["features"])


@pytest.mark.parametrize("guard", [no_fit, legacy_no_fit])
def test_guards_precede_any_inspection(guard):
    class Poison:
        def __getitem__(self, key):
            pytest.fail("recipe inspected before guard")

        def __array__(self, *args, **kwargs):
            pytest.fail("array inspected before guard")

    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden, match="fit_regression_predictions"):
        fit_regression_predictions(Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("labels", [[np.nan], [np.inf], [-np.inf], [], [[-1]], [-1, 2]])
def test_signed_labels_still_require_finite_matching_vector(labels):
    with pytest.raises(ValueError, match="labels"):
        fit(recipe(), matrix([0]), labels, matrix([1]))


@pytest.mark.parametrize("model,estimator", [("history_ridge", Ridge),
                                           ("history_hgb", HistGradientBoostingRegressor)])
@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf], [[1]], [1, 2], 1.0])
def test_signed_predictions_still_require_finite_matching_vector(model, estimator, bad, monkeypatch):
    monkeypatch.setattr(estimator, "predict", lambda self, matrix: bad)
    with pytest.raises(ValueError, match="predictions"):
        fit(recipe(model), matrix([0, 1]), [-1, 1], matrix([0]))


@pytest.mark.parametrize("domain", [None, True, [], "absolute", "signed "])
def test_unknown_domains_refuse(domain):
    with pytest.raises(ValueError, match="domain"):
        recipe(domain=domain)


def test_closed_policy_domain_projection_identity_and_detachment():
    policy = recipe()
    validated = validate_regression_recipe(policy)
    assert validated == policy and validated is not policy
    validated["features"].append("other")
    assert policy["features"] == ["x", "empty"]
    assert content_hash(policy) != content_hash(recipe(domain="nonnegative"))
    for key, value in (("target_domain", "nonnegative"), ("prediction_projection", "clip_zero"),
                       ("schema_version", "magnitude_ridge_hgb_recipe.v1"), ("extra", 1),
                       ("target_transform", "log"), ("numerical_threads", 2)):
        altered = {**policy, key: value}
        with pytest.raises(ValueError, match="recipe"):
            validate_regression_recipe(altered)
    for key in policy:
        with pytest.raises(ValueError, match="recipe"):
            validate_regression_recipe({k: v for k, v in policy.items() if k != key})
    with pytest.raises(ValueError, match="recipe"):
        validate_ridge_hgb_recipe(policy)
    with pytest.raises(ValueError, match="recipe"):
        validate_regression_recipe(ridge_hgb_recipe(model="history_ridge", columns=["x"],
            feature_dictionary_hash="sha256:" + "a" * 64, alpha=1))


def test_metadata_has_no_training_or_source_imports():
    probe = '''
import importlib.abc
import sys
class RejectSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("engine.v2.models.training", "sklearn",
                                        "engine.v2.research", "engine.v2.ops")), fullname
sys.meta_path.insert(0, RejectSources())
from engine.v2.models.regression_ridge_hgb_recipe import regression_recipe, validate_regression_recipe
p = regression_recipe(target_domain="signed", model="history_ridge", columns=["x"],
                      feature_dictionary_hash="sha256:" + "a" * 64, alpha=1)
assert validate_regression_recipe(p) == p
'''
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
