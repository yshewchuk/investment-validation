"""Shared dispatch conformance and SIZE isolation using only synthetic arrays."""
import copy
import subprocess
import sys

import pytest

from engine.models import no_fit as legacy_no_fit
from engine.v2.models import no_fit
from engine.v2.models.magnitude_elasticnet_recipe import regression_elasticnet_recipe
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.models.regression_ridge_hgb_recipe import regression_recipe
from engine.v2.models.training.magnitude_elasticnet import fit_regression_elasticnet_predictions
from engine.v2.models.training.magnitude_ridge_hgb import fit_regression_predictions
from engine.v2.models.training.regression_candidate import fit_regression_candidate
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.magnitude_candidate_recipe import candidate_recipe

FAMILIES = [("ridge", {"alpha": 1.}), ("elasticnet", {"alpha": .1, "l1_ratio": .75}),
            ("hgb", {"max_leaf_nodes": 7})]
FEATURES = dict(columns=["x"], feature_dictionary_hash="sha256:" + "a" * 64)


def policy(family, parameters, domain="signed"):
    return regression_candidate_recipe(family, parameters, target_domain=domain, **FEATURES)


@pytest.mark.parametrize("family,parameters", FAMILIES)
@pytest.mark.parametrize("domain", ["signed", "nonnegative"])
def test_recipes_and_full_results_equal_direct_family_calls(family, parameters, domain):
    original = copy.deepcopy(parameters)
    recipe = policy(family, parameters, domain)
    if family == "elasticnet":
        expected = regression_elasticnet_recipe(target_domain=domain, **FEATURES, **parameters)
        direct = fit_regression_elasticnet_predictions
    else:
        expected = regression_recipe(target_domain=domain,
            model="history_ridge" if family == "ridge" else "history_hgb", **FEATURES, **parameters)
        direct = fit_regression_predictions
    assert recipe == expected and parameters == original
    labels = [-200.] * 20 + [20.] * 20 if domain == "signed" else [0.] * 20 + [20.] * 20
    args = ([[0.]] * 20 + [[1.]] * 20, labels, [[0.], [1.]])
    result = fit_regression_candidate(recipe, *args, columns=["x"])
    assert result == direct(expected, *args, columns=["x"])
    assert result["predictions"][1] > 0
    assert result["predictions"][0] < -100 if domain == "signed" else result["predictions"][0] >= 0
    assert recipe == expected


@pytest.mark.parametrize("family,parameters", FAMILIES)
def test_existing_size_dispatch_remains_nonnegative_and_rejects_new_schema(family, parameters):
    old = candidate_recipe(family, parameters, **FEATURES)
    assert "target_domain" not in old and old["prediction_projection"] == "clip_zero"
    with pytest.raises(ValueError, match="nonnegative"):
        fit_candidate(old, [[0.]], [-1.], [[0.]], columns=["x"])
    with pytest.raises(ValueError, match="schema/model"):
        fit_candidate(policy(family, parameters), [[0.]], [-1.], [[0.]], columns=["x"])
    with pytest.raises(ValueError, match="schema/model"):
        fit_regression_candidate(old, [[0.]], [1.], [[0.]], columns=["x"])


@pytest.mark.parametrize("family,parameters", [("unknown", {}), ([], {}), ("ridge", []),
    ("ridge", {}), ("ridge", {"alpha": 1., "extra": 1}), ("hgb", {"alpha": 1.}),
    ("elasticnet", {"alpha": .1}), ("ridge", {"alpha": True}),
    ("elasticnet", {"alpha": 1e-7, "l1_ratio": .75}),
    ("elasticnet", {"alpha": .2, "l1_ratio": 1.0001})])
def test_closed_family_parameters_refuse(family, parameters):
    with pytest.raises(ValueError):
        policy(family, parameters)


@pytest.mark.parametrize("domain", [None, "SIGNED", "arbitrary", True, []])
def test_explicit_closed_domain(domain):
    with pytest.raises(ValueError):
        policy(*FAMILIES[0], domain)


@pytest.mark.parametrize("guard", [no_fit, legacy_no_fit])
def test_guards_precede_recipe_inspection_and_arrays(guard):
    class Poison(dict):
        def get(self, *args):
            pytest.fail("inspected recipe")
        def __array__(self, *args, **kwargs):
            pytest.fail("inspected matrix")
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        fit_regression_candidate(Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("changes", [{"extra": 1}, {"target_domain": "other"},
    {"prediction_projection": "clip_zero"}, {"model": "elasticnet"}, {"schema_version": []}])
def test_dispatch_does_not_bypass_closed_recipe_validation(changes):
    recipe = {**policy(*FAMILIES[0]), **changes}
    with pytest.raises(ValueError):
        fit_regression_candidate(recipe, [[0.]], [-1.], [[0.]], columns=["x"])


def test_metadata_avoids_training_and_guarded_dispatch_avoids_family_imports():
    code = '''
import importlib.abc
import sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("sklearn", "engine.v2.models.training", "engine.v2.research", "engine.v2.ops")), fullname
sys.meta_path.insert(0, Deny())
from engine.v2.models.regression_candidate import regression_candidate_recipe
sys.meta_path.pop(0)
from engine.v2.models.training.regression_candidate import fit_regression_candidate
from engine.v2.models.no_fit import no_fit_guard, RuntimeFitForbidden
assert "engine.v2.models.training.magnitude_elasticnet" not in sys.modules
assert "engine.v2.models.training.magnitude_ridge_hgb" not in sys.modules
sys.meta_path.insert(0, Deny())
recipe = regression_candidate_recipe("ridge", {"alpha": 1.}, target_domain="signed", columns=["x"], feature_dictionary_hash="sha256:" + "a" * 64)
with no_fit_guard():
    try:
        fit_regression_candidate(recipe, None, None, None, columns=["x"])
    except RuntimeFitForbidden:
        pass
    else:
        raise AssertionError("guard failed")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_delegated_failure_is_preserved_without_retry(monkeypatch):
    from engine.v2.models.training import magnitude_ridge_hgb
    failure = ValueError("synthetic family refusal")
    calls = []
    def refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise failure
    monkeypatch.setattr(magnitude_ridge_hgb, "fit_regression_predictions", refuse)
    recipe = policy(*FAMILIES[0])
    with pytest.raises(ValueError) as caught:
        fit_regression_candidate(recipe, [[0.]], [-1.], [[0.]], columns=["x"])
    assert caught.value is failure and len(calls) == 1
    assert calls[0][0][0] is recipe
