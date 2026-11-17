import json

import pytest

from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v13 import _request

_EN = {"alpha": [0.003, 0.01, 0.03], "l1_ratio": [0.75], "spline_knots": [0, 6]}
_MLP = {"profile": ["chrono10_relu64x32_l2_1", "chrono15_relu64x32_l2_1"]}
_BAD = [
    {"elasticnet": {**_EN, "alpha": [0.5]}},
    {"elasticnet": {**_EN, "alpha": [0.03, 0.003]}},
    {"elasticnet": {**_EN, "alpha": [0.003, 0.003]}},
    {"elasticnet": {**_EN, "alpha": []}},
    {"elasticnet": {"alpha": [0.003], "spline_knots": [0]}},
    {"elasticnet": {**_EN, "wibble": [1]}},
    {"hgb": {"max_leaf_nodes": [7]}},
    {},
    [],
    True,
    {"mlp": {"profile": ["relu32_l2_0p01"]}},
    {"elasticnet": {**_EN, "spline_knots": [0, 4]}},
    {"elasticnet": {**_EN, "spline_knots": [0, True]}},
    {"elasticnet": {**_EN, "alpha": ["0.01"]}},
]


def _rich(log_target=True):
    request = _request(14)
    request["space"]["elasticnet"] = json.loads(json.dumps(_EN))
    request["space"]["mlp"] = json.loads(json.dumps(_MLP))
    request["feature_sets"] = ["HMOSX", "HMOSXL", "HMOSXLP"]
    request["max_candidates"] = 1000
    request["max_fits"] = 100000
    if log_target:
        request["space"]["log_target"] = {
            "elasticnet": json.loads(json.dumps(_EN)),
            "mlp": json.loads(json.dumps(_MLP))}
    return request


def _parameters(specs):
    return sorted(json.dumps(spec["parameters"], sort_keys=True) for spec in specs)


def test_log_target_adds_twenty_four_twins():
    twin = compile_request(_rich())
    base = compile_request(_rich(log_target=False))
    assert twin["counts"]["candidates"] == base["counts"]["candidates"] + 24
    assert len(twin["candidate_specs"]) == len(base["candidate_specs"]) + 24
    twins = [spec for spec in twin["candidate_specs"] if "target" in spec["parameters"]]
    assert len(twins) == 24
    assert {spec["family"] for spec in twins} == {"elasticnet", "mlp"}
    assert all(spec["parameters"]["target"] == "log1p_median" for spec in twins)
    plain = [spec for spec in twin["candidate_specs"] if "target" not in spec["parameters"]]
    assert _parameters(plain) == _parameters(base["candidate_specs"])


def test_v14_without_log_target_matches_v13_counts_and_schedule():
    v14 = compile_request(_request(14))
    v13 = compile_request(_request(13))
    assert v14["counts"] == v13["counts"]
    assert v14["schedule"] == v13["schedule"]


def test_log_target_counts_toward_both_caps():
    twin = compile_request(_rich())
    count = twin["counts"]["candidates"]
    low = _rich()
    low["max_candidates"] = count - 1
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(low)
    fits = _rich()
    fits["max_fits"] = 3 * (count * twin["counts"]["inner_folds"])
    assert fits["max_fits"] < twin["counts"]["search_fits"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(fits)


@pytest.mark.parametrize("log_target", _BAD)
def test_log_target_refusals(log_target):
    request = _rich(log_target=False)
    request["space"]["log_target"] = log_target
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_log_target_requires_its_base_family():
    request = _rich(log_target=False)
    del request["space"]["mlp"]
    request["space"]["log_target"] = {"mlp": json.loads(json.dumps(_MLP))}
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_v13_refuses_log_target():
    request = _request(13)
    request["space"]["log_target"] = {"elasticnet": {"alpha": [0.01], "l1_ratio": [0.25]}}
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_log_target_canonical_form():
    compiled = compile_request(_rich())
    alpha = compiled["request"]["space"]["log_target"]["elasticnet"]["alpha"]
    assert alpha == [0.003, 0.01, 0.03]
    assert all(type(value) is float for value in alpha)
    request = _request(14)
    request["space"]["elasticnet"] = {"alpha": [1], "l1_ratio": [0.25]}
    request["space"]["mlp"] = {"profile": ["chrono15_relu32_l2_0p01"]}
    request["max_candidates"] = 1000
    request["max_fits"] = 100000
    request["space"]["log_target"] = {"elasticnet": {"alpha": [1], "l1_ratio": [0.25]}}
    compiled = compile_request(request)
    assert compiled["request"]["space"]["log_target"]["elasticnet"]["alpha"] == [1.0]
