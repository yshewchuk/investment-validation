"""Closed execution routes preserve their existing numeric fit contracts."""
import pytest

from engine.models import no_fit as legacy_guard
from engine.v2.models import no_fit
from engine.v2.models.regression_candidate import regression_candidate_recipe
from engine.v2.models.training import (
    magnitude,
    magnitude_candidate,
    regression_candidate,
)
from engine.v2.models.training.fit_routes import fit_execution
from engine.v2.ops.native_registration import magnitude_fitted_execution_document
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.magnitude_candidate_recipe import candidate_recipe

COLUMNS = ["x"]
FEATURES = dict(columns=COLUMNS, feature_dictionary_hash="sha256:" + "a" * 64)
FAMILIES = [("ridge", {"alpha": 1.}), ("hgb", {"max_leaf_nodes": 7}),
            ("elasticnet", {"alpha": .1, "l1_ratio": .75})]


def test_legacy_wrapper_delegates_result_unchanged(monkeypatch):
    result = {"predictions": [2.], "synthetic_diagnostic": "preserved"}
    calls = []
    def fitted(*args, **kwargs):
        calls.append((args, kwargs))
        return result
    monkeypatch.setattr(magnitude_candidate, "fit_candidate", fitted)
    recipe = candidate_recipe("ridge", {"alpha": 1.}, **FEATURES)
    assert fit_candidate(recipe, [[0.]], [2.], [[0.]], columns=COLUMNS) is result
    assert len(calls) == 1 and calls[0][0][0] is recipe


@pytest.mark.parametrize("family,parameters", FAMILIES)
@pytest.mark.parametrize("domain", ["signed", "nonnegative"])
def test_regression_route_preserves_complete_result(family, parameters, domain):
    recipe = regression_candidate_recipe(family, parameters, target_domain=domain, **FEATURES)
    train, prediction = [[0.]] * 20 + [[1.]] * 20, [[0.], [1.]]
    labels = [-140.] * 20 + [25.] * 20 if domain == "signed" else [0.] * 20 + [25.] * 20
    expected = regression_candidate.fit_regression_candidate(recipe, train, labels, prediction, columns=COLUMNS)
    assert fit_execution("regression_candidate", recipe, train, labels, prediction, columns=COLUMNS) == expected
    if domain == "signed":
        assert expected["predictions"][0] < -100.


@pytest.mark.parametrize("family,parameters", FAMILIES)
def test_magnitude_candidate_route_preserves_complete_result(family, parameters):
    recipe = candidate_recipe(family, parameters, **FEATURES)
    args = ([[0.]] * 20 + [[1.]] * 20, [0.] * 20 + [25.] * 20, [[0.], [1.]])
    assert fit_execution("magnitude_candidate", recipe, *args, columns=COLUMNS) == fit_candidate(recipe, *args, columns=COLUMNS)
    with pytest.raises(ValueError, match="nonnegative"):
        fit_execution("magnitude_candidate", recipe, [[0.]], [-1.], [[0.]], columns=COLUMNS)


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
@pytest.mark.parametrize("container", [list, tuple])
def test_original_magnitude_route_wraps_only_predictions(model, container):
    execution = magnitude_fitted_execution_document("2025-01-01", model)
    matrix = [[0.] * 11] * 3
    args = (execution["fit_recipe"], matrix, [1., 1., 1.], matrix)
    expected = magnitude.fit_magnitude_predictions(*args)
    assert fit_execution("magnitude", *args, columns=container(execution["features"])) == {"predictions": expected}


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("array inspected before refusal")
    def __getitem__(self, key):
        pytest.fail("recipe inspected before refusal")
    def __eq__(self, other):
        pytest.fail("route compared before type check")


@pytest.mark.parametrize("guard", [legacy_guard, no_fit])
def test_both_guards_precede_route_and_input_inspection(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden):
        fit_execution(Poison(), Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("route", [None, [], {}, Poison(), "other", "regression_candidate_execution.v1"])
def test_unknown_routes_refuse_before_inputs(route):
    with pytest.raises(ValueError, match="unsupported numeric"):
        fit_execution(route, Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("route,module,name", [
    ("magnitude", magnitude, "fit_magnitude_predictions"),
    ("magnitude_candidate", magnitude_candidate, "fit_candidate"),
    ("regression_candidate", regression_candidate, "fit_regression_candidate")])
def test_original_errors_propagate_without_retry(route, module, name, monkeypatch):
    calls, failure = [], RuntimeError("synthetic failure")
    def refuse(*args, **kwargs):
        calls.append(1)
        raise failure
    monkeypatch.setattr(module, name, refuse)
    execution = magnitude_fitted_execution_document("2025-01-01", "history_ridge")
    with pytest.raises(RuntimeError) as caught:
        fit_execution(route, execution["fit_recipe"], [], [], [], columns=execution["features"])
    assert caught.value is failure and calls == [1]


@pytest.mark.parametrize("recipe", [None, {}, {"schema_version": "regression_ridge_hgb_recipe.v1"}])
def test_original_route_cannot_silently_consume_signed_or_unknown_recipe(recipe):
    with pytest.raises(ValueError, match="admitted recipe schema"):
        fit_execution("magnitude", recipe, Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("model", ["history_ridge", "history_hgb"])
@pytest.mark.parametrize("mutation", ["swapped", "wrong_name", "duplicate", "missing", "not_sequence"])
def test_original_magnitude_rejects_column_mismatch_before_fit(model, mutation, monkeypatch):
    execution = magnitude_fitted_execution_document("2025-01-01", model)
    columns = list(execution["features"])
    if mutation == "swapped":
        columns[0], columns[1] = columns[1], columns[0]
    elif mutation == "wrong_name":
        columns[0] = "wrong_feature"
    elif mutation == "duplicate":
        columns[0] = columns[1]
    elif mutation == "missing":
        columns.pop()
    else:
        columns = None
    def unexpected_fit(*args, **kwargs):
        pytest.fail("fitter entered before column admission")
    monkeypatch.setattr(magnitude, "fit_magnitude_predictions", unexpected_fit)
    with pytest.raises(ValueError, match="columns must exactly match"):
        fit_execution("magnitude", execution["fit_recipe"], Poison(), Poison(), Poison(), columns=columns)
