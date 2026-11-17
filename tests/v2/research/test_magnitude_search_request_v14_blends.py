import json

import pytest

from engine.v2.ops.errors import OpsError
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v13 import _request

_EN = {"alpha": [0.03], "l1_ratio": [0.75], "spline_knots": [0, 6]}
_HGB = {"learning_rate": [0.03], "l2_regularization": [10.0],
        "loss": ["absolute_error", "squared_error"], "max_iter": [3000],
        "max_leaf_nodes": [15], "min_samples_leaf": [300, 1000]}
_MLP = {"profile": ["chrono15_relu64x32_l2_1"]}
_SETS = ["HMOSX", "HMOSXL", "HMOSXLP"]
_BAD_LISTS = [[], ["size_b2_v1", "size_b1_v1"], ["size_b1_v1", "size_b1_v1"],
              ["size_b3_v1"], "size_b1_v1", None, [3]]


def _blend_request():
    request = _request(14)
    request["space"]["elasticnet"] = json.loads(json.dumps(_EN))
    request["space"]["hgb"] = json.loads(json.dumps(_HGB))
    request["space"]["mlp"] = json.loads(json.dumps(_MLP))
    request["feature_sets"] = list(_SETS)
    request["selection"]["winner_rules"] = ["overall", "rmse", "mae"]
    request["max_candidates"] = 1000
    request["max_fits"] = 100000
    return request


def _with_blends(blends=("size_b1_v1", "size_b2_v1")):
    request = _blend_request()
    request["blends"] = list(blends)
    return request


def test_blends_compile_with_sorted_echo_and_count():
    compiled = compile_request(_with_blends())
    assert compiled["request"]["blends"] == ["size_b1_v1", "size_b2_v1"]
    counts = compiled["counts"]
    assert counts["blends"] == 2
    assert counts["search_fits"] == 3 * (
        counts["candidates"] * counts["inner_folds"] + counts["outer_folds"] * 8)


def test_v14_without_blends_has_no_blends_count():
    compiled = compile_request(_blend_request())
    assert "blends" not in compiled["counts"]
    assert "blends" not in compiled["request"]


def test_v13_refuses_blends():
    request = _request(13)
    request["blends"] = ["size_b2_v1"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_blends_require_selection_winner_rules():
    request = _with_blends()
    del request["selection"]["winner_rules"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_blends_require_selection():
    request = _with_blends()
    del request["selection"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_blend_refuses_missing_mlp_profile():
    request = _with_blends()
    request["space"]["mlp"] = {"profile": ["chrono10_relu64x32_l2_1"]}
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_blend_refuses_without_hmosxlp_feature_set():
    request = _with_blends()
    request["feature_sets"] = ["HMOSX", "HMOSXL"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


@pytest.mark.parametrize("blends", _BAD_LISTS)
def test_blends_refuse_bad_lists(blends):
    request = _blend_request()
    request["blends"] = blends
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_blends_refuse_max_fits_below_refit_bound():
    base = compile_request(_blend_request())["counts"]
    request = _with_blends()
    request["max_fits"] = 3 * (base["candidates"] * base["inner_folds"] + base["outer_folds"])
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)
