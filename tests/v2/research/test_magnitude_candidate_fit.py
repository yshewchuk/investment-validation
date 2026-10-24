"""Synthetic independent numeric oracles and exact single-fit delegation."""
import copy

import numpy as np
import pytest

from engine.models import no_fit as legacy_no_fit
from engine.v2.foundation import content_hash
from engine.v2.models import no_fit
from engine.v2.models.training import magnitude_elasticnet, magnitude_ridge_hgb
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.magnitude_candidate_recipe import candidate_recipe

CASES = [("ridge", {"alpha": a}) for a in (0.1, 1, 10)] + [
    ("elasticnet", {"alpha": a, "l1_ratio": r}) for a in (0.01, 0.1, 1)
    for r in (0.25, 0.75)] + [("hgb", {"max_leaf_nodes": n}) for n in (7, 15)]
DEFAULTS = [CASES[1], CASES[6], CASES[-2]]


def recipe(family, parameters, width=3):
    return candidate_recipe(family, parameters, columns=[f"x{i}" for i in range(width)],
                            feature_dictionary_hash="sha256:" + "a" * 64)


def delegate(family):
    return ((magnitude_elasticnet, "fit_elasticnet_predictions") if family == "elasticnet"
            else (magnitude_ridge_hgb, "fit_ridge_hgb_predictions"))


class Poison:
    def __array__(self, *args, **kwargs):
        pytest.fail("array inspected before refusal")

    def __getitem__(self, key):
        pytest.fail("recipe inspected before guard")

    def __iter__(self):
        pytest.fail("columns inspected before guard")


@pytest.mark.parametrize("guard", [legacy_no_fit, no_fit])
def test_both_guards_precede_all_inspection(guard):
    with guard.no_fit_guard(), pytest.raises(guard.RuntimeFitForbidden, match="fit_candidate"):
        fit_candidate(Poison(), Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("bad", [None, [], Poison(), {},
    {"schema_version": [], "model": "elasticnet"},
    {"schema_version": "magnitude_elasticnet_recipe.v1", "model": {}},
    {"schema_version": "magnitude_elasticnet_recipe.v1", "model": "history_ridge"},
    {"schema_version": "magnitude_ridge_hgb_recipe.v1", "model": "elasticnet"},
    {"schema_version": "magnitude_fitted_recipe.v1", "model": "history_ridge"},
])
def test_unsupported_envelope_refuses_before_delegation(bad, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported envelope delegated")
    for family, _ in DEFAULTS:
        module, name = delegate(family)
        monkeypatch.setattr(module, name, forbidden)
    with pytest.raises(ValueError, match="^candidate requires a supported recipe schema/model pair$"):
        fit_candidate(bad, Poison(), Poison(), Poison(), columns=Poison())


@pytest.mark.parametrize("family,parameters", DEFAULTS)
def test_closed_policy_and_order_refuse_before_arrays(family, parameters):
    policy = recipe(family, parameters)
    prefix = "ElasticNet" if family == "elasticnet" else "magnitude"
    with pytest.raises(ValueError, match=f"^{prefix} matrix columns must exactly match the ordered recipe subset$"):
        fit_candidate(policy, Poison(), Poison(), Poison(), columns=list(reversed(policy["features"])))
    policy["numerical_threads"] = 2
    message = "invalid closed ElasticNet recipe" if family == "elasticnet" else "invalid closed variable-width magnitude recipe"
    with pytest.raises(ValueError, match="^" + message + "$"):
        fit_candidate(policy, Poison(), Poison(), Poison(), columns=policy["features"])


@pytest.mark.parametrize("family,parameters", CASES)
@pytest.mark.parametrize("width", [3, 28, 42, 43])
def test_every_grid_choice_has_bound_id_and_informative_final_column(family, parameters, width):
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    prediction = np.full((4, width), 7.)
    prediction[:, [0, -1]] = corners
    train = np.repeat(prediction, 20, axis=0)
    means = np.array([2., 6., 10., 14.])
    policy = recipe(family, parameters, width)
    result = fit_candidate(policy, train, np.repeat(means, 20), prediction, columns=policy["features"])
    if family == "ridge":
        expected = 8 + (means - 8) * 80 / (80 + parameters["alpha"])
    elif family == "hgb":
        expected = 8 + (means - 8) * (1 - 0.9**100)
    else:
        a, r = parameters["alpha"], parameters["l1_ratio"]
        coefficients = (np.array([2., 4.]) - a * r) / (1 + a * (1 - r))
        expected = 8 + (2 * corners - 1) @ coefficients
    assert result["predictions"] == pytest.approx(expected, abs=1e-6)
    assert result["recipe_id"] == content_hash(policy)
    assert result["schema_version"] == ("magnitude_elasticnet_fit.v1" if family == "elasticnet" else "magnitude_ridge_hgb_fit.v1")
    if family == "elasticnet":
        expected_coefficients = np.zeros(2 * width)
        expected_coefficients[[0, width - 1]] = coefficients
        assert [(e["column"], e["kind"]) for e in result["coefficients"]] == [
            (column, kind) for kind in ("value", "missing_indicator") for column in policy["features"]]
        assert [e["coefficient"] for e in result["coefficients"]] == pytest.approx(expected_coefficients)
        assert result["selected_columns"] == ["x0", f"x{width - 1}"]
        assert result["coefficient_scale"] == "standardized_inputs" and result["intercept"] == pytest.approx(8.)
        assert result["converged"] is True and 0 < result["n_iter"] <= policy["estimator"]["params"]["max_iter"]
        assert np.isfinite(result["dual_gap"]) and abs(result["dual_gap"]) < 1e-6


@pytest.mark.parametrize("family,parameters", CASES)
@pytest.mark.parametrize("width", [1, 11])
def test_complete_native_result_null_policy_and_input_detachment(family, parameters, width):
    policy = recipe(family, parameters, width)
    train, prediction = np.full((60, width), 7.), np.full((2, width), 7.)
    train[:, 0], prediction[:, 0] = [0] * 20 + [1] * 20 + [np.nan] * 20, [0, np.nan]
    if width > 1:
        train[:, -1], prediction[:, -1] = np.nan, [1e12, -1e12]
    labels = np.repeat([2., 8., 3.], 20)
    before = [value.copy() for value in (train, labels, prediction)]
    saved = copy.deepcopy(policy)
    module, name = delegate(family)
    direct = getattr(module, name)(policy, train, labels, prediction, columns=policy["features"])
    result = fit_candidate(policy, train, labels, prediction, columns=policy["features"])
    assert result == direct
    assert policy == saved
    for actual, expected in zip((train, labels, prediction), before):
        np.testing.assert_array_equal(actual, expected)
    if family == "elasticnet":
        assert {"coefficients", "selected_columns", "intercept", "converged", "n_iter", "dual_gap"} <= result.keys()
        assert len(result["coefficients"]) == 2 * width and result["converged"] is True


@pytest.mark.parametrize("family,parameters", DEFAULTS)
def test_exactly_one_delegation_preserves_arguments_and_complete_result(family, parameters, monkeypatch):
    policy, columns = recipe(family, parameters), ["x0", "x1", "x2"]
    arrays, calls = [object(), object(), object()], []
    returned = {"predictions": [3.], "recipe_id": "opaque", "family_evidence": object()}
    module, name = delegate(family)
    def observed(*args, **kwargs):
        calls.append((args, kwargs))
        return returned
    monkeypatch.setattr(module, name, observed)
    assert fit_candidate(policy, *arrays, columns=columns) is returned
    assert len(calls) == 1
    assert len(calls[0][0]) == 4 and set(calls[0][1]) == {"columns"}
    assert all(actual is expected for actual, expected in zip(calls[0][0], [policy, *arrays]))
    assert calls[0][1]["columns"] is columns


@pytest.mark.parametrize("family,parameters", DEFAULTS)
@pytest.mark.parametrize("error", [ValueError("native refusal"), TypeError("native type"), OverflowError("native conversion"), RuntimeError("native failure")])
def test_native_exception_identity_no_retry(family, parameters, error, monkeypatch):
    module, name = delegate(family)
    calls = []
    def refusing(*args, **kwargs):
        calls.append(1)
        raise error
    monkeypatch.setattr(module, name, refusing)
    with pytest.raises(type(error)) as caught:
        fit_candidate(recipe(family, parameters), Poison(), Poison(), Poison(), columns=["x0", "x1", "x2"])
    assert caught.value is error and calls == [1]


@pytest.mark.parametrize("family,parameters", DEFAULTS)
@pytest.mark.parametrize("error", [TypeError, OverflowError])
@pytest.mark.parametrize("position", range(3))
def test_actual_family_conversion_policy_is_not_normalized(family, parameters, error, position):
    class Unconvertible:
        def __float__(self):
            raise error("native conversion sentinel")
    policy = recipe(family, parameters, 1)
    arrays = [[[0]], [1], [[0]]]
    arrays[position] = [Unconvertible()] if position == 1 else [[Unconvertible()]]
    expected = error if family == "elasticnet" else ValueError
    message = ("native conversion sentinel" if family == "elasticnet" else
               "magnitude inputs must be convertible to floating-point values")
    with pytest.raises(expected, match="^" + message + "$") as caught:
        fit_candidate(policy, *arrays, columns=["x0"])
    if family == "elasticnet":
        assert caught.value.__cause__ is None
    else:
        assert type(caught.value.__cause__) is error
        assert str(caught.value.__cause__) == "native conversion sentinel"
