"""Strict neural worker-claim admission on fresh receipt and no-fit replay paths."""
import copy
import json

import pytest

from engine.models.no_fit import no_fit_guard as legacy_guard
from engine.v2.foundation import canonical_json, content_hash
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.models.training import magnitude_mlp
from engine.v2.ops.errors import OpsError
from experiments.size_fit_diagnostics import evidence
from tests.v2.research.test_magnitude_mlp_dispatch import PROFILES, recipe


def execution():
    return {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe()}


def result():
    return {"event": "succeeded", "diagnostics": [
        {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe()),
         "horizon_sessions": h, "component_fits": 1, "converged": True, "n_iter": 200,
         "input_features": 2, "loss": .125, "loss_scale": "standardized_target"}
        for h in (5, 1)]}


def refuses(value, policy=None, horizons=(5, 1)):
    with pytest.raises(OpsError) as caught:
        evidence(value, policy or execution(), horizons)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_fresh_and_serialized_replay_preserve_evidence_without_fit(monkeypatch):
    monkeypatch.setattr(magnitude_mlp, "fit_mlp_predictions", lambda *a, **k: pytest.fail("refit"))
    original = result()
    before = copy.deepcopy(original)
    with no_fit_guard(), legacy_guard():
        fresh = evidence(original, execution(), [5, 1])
        replay = evidence(json.loads(canonical_json(original)), execution(), [5, 1])
    assert fresh == replay == {"diagnostics": original["diagnostics"]}
    assert original == before


@pytest.mark.parametrize("field,value", [
    ("schema_version", "magnitude_ridge_hgb_fit.v1"), ("schema_version", "magnitude_mlp_fit.v2"),
    ("recipe_id", "sha256:" + "0" * 64), ("horizon_sessions", True), ("horizon_sessions", 1.),
    ("component_fits", True), ("component_fits", 1.), ("component_fits", 0), ("component_fits", 2),
    ("converged", 1), ("converged", False), ("n_iter", True), ("n_iter", 1.),
    ("n_iter", 0), ("n_iter", 401), ("input_features", True), ("input_features", 2.),
    ("input_features", 0), ("input_features", 7), ("loss", True), ("loss", "0.1"),
    ("loss", -1.), ("loss", float("nan")), ("loss", float("inf")), ("loss", 10 ** 1000),
    ("loss_scale", "original_target"),
])
def test_tampered_worker_claims_refuse(field, value):
    document = result()
    document["diagnostics"][0][field] = value
    refuses(document)


@pytest.mark.parametrize("kind", ["extra", "missing", "duplicate", "reverse", "short", "long", "tuple", "null", "row"])
def test_closed_keys_and_exact_ordered_horizon_grid(kind):
    document = result()
    rows = document["diagnostics"]
    if kind == "extra":
        rows[0]["extra"] = 1
    elif kind == "missing":
        del rows[0]["component_fits"]
    elif kind == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif kind == "reverse":
        rows.reverse()
    elif kind == "short":
        rows.pop()
    elif kind == "long":
        rows.append(copy.deepcopy(rows[0]))
    elif kind == "tuple":
        document["diagnostics"] = tuple(rows)
    elif kind == "null":
        document["diagnostics"] = None
    else:
        rows[0] = None
    refuses(document)


@pytest.mark.parametrize("field,value", [
    ("model", "history_ridge"), ("schema_version", "magnitude_mlp_recipe.v2"),
    ("profile", PROFILES[1]), ("numerical_threads", 2), ("component_fits", 2),
])
def test_bound_recipe_policy_drift_refuses_even_with_matching_reported_hash(field, value):
    policy, document = execution(), result()
    policy["fit_recipe"][field] = value
    for row in document["diagnostics"]:
        row["recipe_id"] = content_hash(policy["fit_recipe"])
    refuses(document, policy)


def test_regression_execution_does_not_gain_neural_diagnostics():
    policy = execution()
    policy["schema_version"] = "regression_candidate_execution.v1"
    refuses(result(), policy)


@pytest.mark.parametrize("iterations,width,loss", [(1, 1, 0), (400, 6, 1.7976931348623157e308)])
def test_exact_declared_numeric_evidence_bounds(iterations, width, loss):
    document = result()
    for row in document["diagnostics"]:
        row.update(n_iter=iterations, input_features=width, loss=loss)
    assert evidence(document, execution(), [5, 1]) == {"diagnostics": document["diagnostics"]}


@pytest.mark.parametrize("event", ["failed", "unavailable", "resource_limit"])
def test_failed_attempts_do_not_acquire_success_diagnostics(event):
    assert evidence({"event": event, "diagnostics": None}, execution(), [5, 1]) == {}
