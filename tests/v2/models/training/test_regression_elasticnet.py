"""Synthetic signed-domain ElasticNet oracles; no source or experiment reads."""
import copy
import json
import subprocess
import sys
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from threadpoolctl import threadpool_info

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models import no_fit
from engine.v2.models.magnitude_elasticnet_recipe import validate_elasticnet_recipe
from engine.v2.models.magnitude_elasticnet_recipe import (
    regression_elasticnet_recipe,
    validate_regression_elasticnet_recipe,
)
from engine.v2.models.training.magnitude_elasticnet import (
    fit_elasticnet_predictions,
    fit_regression_elasticnet_predictions,
)


def recipe(domain="signed", alpha=.1, l1_ratio=.75):
    return regression_elasticnet_recipe(target_domain=domain, columns=["signal", "empty"],
        feature_dictionary_hash="sha256:" + "a" * 64, alpha=alpha, l1_ratio=l1_ratio)


def matrix(values):
    return np.column_stack([values, np.full(len(values), np.nan)])


def fit(policy, labels=(-4, 0, 4), prediction=None):
    return fit_regression_elasticnet_predictions(policy, matrix([0, 2, 4]), labels,
        matrix([-100, 0, 2, 4, 100]) if prediction is None else prediction,
        columns=policy["features"])


@pytest.mark.parametrize("alpha", [.01, .1, 1])
@pytest.mark.parametrize("l1_ratio", [.25, .75])
def test_signed_soft_threshold_oracle_and_coefficient_evidence(alpha, l1_ratio):
    policy = json.loads(canonical_json(recipe(alpha=alpha, l1_ratio=l1_ratio)))
    scale = np.sqrt(8 / 3)
    beta = (2 * scale - alpha * l1_ratio) / (1 + alpha * (1 - l1_ratio))
    result = fit(policy)
    assert result["predictions"] == pytest.approx(np.array([-102, -2, 0, 2, 98]) * beta / scale)
    assert result["predictions"][0] < -100  # No inferred economic lower bound.
    assert result["schema_version"] == "regression_elasticnet_fit.v1"
    assert result["recipe_id"] == content_hash(policy)
    assert result["intercept"] == pytest.approx(0)
    assert result["coefficients"] == [
        {"column": column, "kind": kind, "coefficient": pytest.approx(beta) if i == 0 else 0.}
        for i, (kind, column) in enumerate((kind, column)
            for kind in ("value", "missing_indicator") for column in policy["features"])]
    assert result["selected_columns"] == ["signal"]
    assert result["coefficient_scale"] == "standardized_inputs"
    assert result["converged"] is True and 0 < result["n_iter"] <= 10000
    assert np.isfinite(result["dual_gap"])


@pytest.mark.parametrize("value", [-200, 7])
def test_constant_labels_retain_both_domains(value):
    assert fit(recipe(), labels=[value] * 3)["predictions"] == pytest.approx([value] * 5)


def test_nonnegative_result_and_evidence_match_legacy():
    policy = recipe("nonnegative")
    old = {k: v for k, v in policy.items() if k != "target_domain"}
    old["schema_version"] = "magnitude_elasticnet_recipe.v1"
    assert validate_elasticnet_recipe(old) == old
    args = (matrix([0, 2, 4]), [1, 5, 9], matrix([-100, 1, 100]))
    old_result = fit_elasticnet_predictions(old, *args, columns=old["features"])
    result = fit_regression_elasticnet_predictions(policy, *args, columns=policy["features"])
    for key in old_result.keys() - {"schema_version", "recipe_id"}:
        assert result[key] == old_result[key]
    assert result["predictions"][0] == 0
    for function, metadata in ((fit_regression_elasticnet_predictions, policy), (fit_elasticnet_predictions, old)):
        with pytest.raises(ValueError, match="nonnegative"):
            function(metadata, matrix([0]), [-1], matrix([0]), columns=metadata["features"])


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("premature array access")

    def __getitem__(self, key):
        pytest.fail("recipe inspected before guard")


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
def test_both_guards_precede_inspection(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        fit_regression_elasticnet_predictions(Poison(), Poison(), Poison(), Poison(), columns=Poison())


def test_closed_domain_projection_schema_and_detached_metadata():
    policy = recipe()
    validated = validate_regression_elasticnet_recipe(policy)
    validated["features"].clear()
    assert policy["features"] == ["signal", "empty"]
    assert content_hash(policy) != content_hash(recipe("nonnegative"))
    for key, value in (("target_domain", "nonnegative"), ("prediction_projection", "clip_zero"),
                       ("schema_version", "magnitude_elasticnet_recipe.v1"), ("extra", 1),
                       ("target_transform", "log"), ("nonconvergence", "retry")):
        with pytest.raises(ValueError, match="recipe"):
            fit_regression_elasticnet_predictions({**policy, key: value}, Poison(), Poison(), Poison(),
                                                 columns=policy["features"])
    for key in policy:
        with pytest.raises(ValueError, match="recipe"):
            validate_regression_elasticnet_recipe({k: v for k, v in policy.items() if k != key})
    with pytest.raises(ValueError, match="recipe"):
        validate_elasticnet_recipe(policy)


@pytest.mark.parametrize("domain", [None, True, [], "absolute", "signed "])
def test_unknown_domains_refuse(domain):
    with pytest.raises(ValueError, match="domain"):
        recipe(domain)


@pytest.mark.parametrize("labels", [[np.nan] * 3, [np.inf] * 3, [-np.inf] * 3, [], [[-1]] * 3])
def test_finite_matching_labels_required(labels):
    with pytest.raises(ValueError, match="labels"):
        fit(recipe(), labels=labels)


def test_nonconvergence_refuses_once_without_prediction(monkeypatch):
    original = ElasticNet.fit
    calls = []

    def warn(self, *args, **kwargs):
        calls.append(self)
        result = original(self, *args, **kwargs)
        warnings.warn("synthetic nonconvergence", ConvergenceWarning)
        return result

    monkeypatch.setattr(ElasticNet, "fit", warn)
    monkeypatch.setattr(ElasticNet, "predict", lambda *args: pytest.fail("prediction after failed fit"))
    with pytest.raises(ValueError, match="convergence failed"):
        fit(recipe())
    assert len(calls) == 1


@pytest.mark.parametrize("bad", [[np.nan], [np.inf], [-np.inf], [[1]], [1, 2], 1.0])
def test_invalid_predictions_refuse(monkeypatch, bad):
    monkeypatch.setattr(ElasticNet, "predict", lambda *args: bad)
    with pytest.raises(ValueError, match="predictions"):
        fit(recipe(), prediction=matrix([0]))


def test_fresh_single_thread_training_only_and_detached_inputs(monkeypatch):
    calls = []
    original = ElasticNet.fit

    def observe(self, x, y, *args, **kwargs):
        calls.append((self, x.copy(), y.copy(), threadpool_info()))
        return original(self, x, y, *args, **kwargs)

    monkeypatch.setattr(ElasticNet, "fit", observe)
    policy, train, labels, prediction = recipe(), matrix([0, 2, np.nan]), [-4, 0, 4], matrix([1])
    before = copy.deepcopy((policy, train, labels, prediction))
    first = fit_regression_elasticnet_predictions(policy, train, labels, prediction, columns=policy["features"])
    polluted = matrix([1, 1e12, np.nan])
    polluted[:, 1] = 1e12
    second = fit_regression_elasticnet_predictions(policy, train, labels, polluted, columns=policy["features"])
    for key in first.keys() - {"predictions"}:
        assert first[key] == second[key]
    assert first["predictions"][0] == second["predictions"][0]
    assert len(calls) == 2 and calls[0][0] is not calls[1][0]
    assert all(pools and all(p["num_threads"] == 1 for p in pools) for _, _, _, pools in calls)
    assert policy == before[0] and labels == before[2]
    np.testing.assert_array_equal(train, before[1])
    np.testing.assert_array_equal(prediction, before[3])


def test_metadata_has_no_training_or_source_imports():
    probe = '''
import importlib.abc
import sys
class RejectSources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("engine.v2.models.training", "sklearn",
                                        "engine.v2.research", "engine.v2.ops")), fullname
sys.meta_path.insert(0, RejectSources())
from engine.v2.models.magnitude_elasticnet_recipe import regression_elasticnet_recipe, validate_regression_elasticnet_recipe
p = regression_elasticnet_recipe(target_domain="signed", columns=["x"],
    feature_dictionary_hash="sha256:" + "a" * 64, alpha=1, l1_ratio=.25)
assert validate_regression_elasticnet_recipe(p) == p
'''
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
