import pytest

from engine.v2.models.magnitude_elasticnet_spline_recipe import SPLINE_COLUMNS
from engine.v2.ops.errors import OpsError
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v13 import _request


_SPLINE_SETS = ["HMOSX", "HMOSXL", "HMOSXLP"]
_HASH = "sha256:" + "0" * 64


def _spline_request(knots, *, version=13, feature_sets=None):
    request = _request(version, feature_sets=feature_sets)
    request["space"]["elasticnet"]["spline_knots"] = knots
    request["max_candidates"] = 1000
    request["max_fits"] = 3 * (1000 * 2 + 1)
    return request


def test_v13_absent_axis_is_unchanged():
    compiled = compile_request(_request(13))
    assert compiled["request_id"] == (
        "sha256:98195717610b45596e2af186822452fe65913b6e2cda62024707dce5cb6359c4")
    assert set(compiled["request"]["space"]["elasticnet"]) == {"alpha", "l1_ratio"}


def test_v13_spline_axis_expands_elasticnet_candidates():
    compiled = compile_request(_spline_request([0, 4, 6], feature_sets=_SPLINE_SETS))
    without = compile_request(_request(13, feature_sets=_SPLINE_SETS))
    assert compiled["counts"]["candidates"] == (
        without["counts"]["candidates"] + len(_SPLINE_SETS) * 2)
    assert compiled["request_id"] != without["request_id"]
    elastics = [spec for spec in compiled["candidate_specs"]
                if spec["family"] == "elasticnet"]
    assert len(elastics) == len(_SPLINE_SETS) * 3
    for subset in _SPLINE_SETS:
        per_set = [spec for spec in elastics
                   if spec["feature_recipe"]["subset"] == subset]
        assert len(per_set) == 3
    for spec in elastics:
        assert set(spec["parameters"]) == {"alpha", "l1_ratio", "spline_knots"}


def test_v13_spline_elasticnet_candidates_pass_candidate_recipe():
    compiled = compile_request(_spline_request([0, 4, 6], feature_sets=_SPLINE_SETS))
    for spec in compiled["candidate_specs"]:
        if spec["family"] != "elasticnet":
            continue
        built = candidate_recipe("elasticnet", spec["parameters"],
            columns=list(SPLINE_COLUMNS), feature_dictionary_hash=_HASH)
        assert built["schema_version"] == (
            "magnitude_elasticnet_recipe.v1"
            if spec["parameters"]["spline_knots"] == 0
            else "magnitude_elasticnet_spline_recipe.v1")


def test_v13_accepts_zero_knots_for_default_sets():
    compiled = compile_request(_spline_request([0]))
    elasticnet = compiled["request"]["space"]["elasticnet"]
    assert set(elasticnet) == {"alpha", "l1_ratio", "spline_knots"}
    assert elasticnet["spline_knots"] == [0]


@pytest.mark.parametrize("knots", [[4], [8], [7], [3], [4, 4], [6, 4],
                                   [4.0], [True], [], 4])
def test_v13_refuses_invalid_spline_knots(knots):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_spline_request(knots))


def test_v13_refuses_positive_knots_with_partial_spline_sets():
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_spline_request([4], feature_sets=["HMOS", "HMOSX"]))


@pytest.mark.parametrize("version", [9, 12])
@pytest.mark.parametrize("knots", [[0], [4]])
def test_older_versions_refuse_the_spline_axis(version, knots):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_spline_request(knots, version=version))


def test_v13_omitting_ridge_still_accepts_the_spline_axis():
    request = _spline_request([0, 4, 6], feature_sets=_SPLINE_SETS)
    del request["space"]["ridge"]
    compiled = compile_request(request)
    assert not [spec for spec in compiled["candidate_specs"]
                if spec["family"] == "ridge"]
    assert set(compiled["request"]["space"]["elasticnet"]) == {
        "alpha", "l1_ratio", "spline_knots"}
