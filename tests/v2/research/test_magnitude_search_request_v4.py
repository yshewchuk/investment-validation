"""Closed v4 full HGB-axis grid compilation and refusal-before-expansion contracts."""
import copy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_request as compiler
from experiments import magnitude_search_request_versions as versions
from tests.v2.research.test_magnitude_search_request import request as v2_request
from tests.v2.research.test_magnitude_search_request_v3 import (
    ELASTICNET_ALPHA, ELASTICNET_L1_RATIO, RIDGE_ALPHA, fail_if_called, set_value, v3_request)

HGB_LEARNING_RATE = [0.05, 0.1]
HGB_MAX_LEAF_NODES = [7, 15]
HGB_MIN_SAMPLES_LEAF = [20, 100]
HGB_L2_REGULARIZATION = [0, 1]


def v4_request():
    document = v3_request()
    document["schema_version"] = "magnitude_search_request.v4"
    document["space"] = {
        "ridge": {"alpha": list(RIDGE_ALPHA)},
        "elasticnet": {"alpha": list(ELASTICNET_ALPHA),
            "l1_ratio": list(ELASTICNET_L1_RATIO)},
        "hgb": {"learning_rate": list(HGB_LEARNING_RATE),
            "max_leaf_nodes": list(HGB_MAX_LEAF_NODES),
            "min_samples_leaf": list(HGB_MIN_SAMPLES_LEAF),
            "l2_regularization": list(HGB_L2_REGULARIZATION)}}
    document["max_candidates"] = 99
    document["max_fits"] = 1791
    document["selection"] = {
        "metric_weights": {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1},
        "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
        "large_move_threshold_pct": 10.0,
        "tie_band_floor": 0.005,
    }
    return document


def test_v4_full_grid_counts_identities_and_canonical_types():
    document = v4_request()
    inner_folds = sum(len(years) for years in document["inner_years"].values())
    outer_folds = len(document["outer_years"])
    assert document["max_fits"] == 3 * (99 * inner_folds + outer_folds)
    result = versions.compile_request(document)
    assert result["counts"]["candidates"] == 99
    assert result["counts"]["search_fits"] == document["max_fits"]
    assert result["request_id"] == content_hash(result["request"])
    specs = result["candidate_specs"]
    ids = [spec["candidate_spec_id"] for spec in specs]
    assert len(ids) == len(set(ids)) == 99 and ids == sorted(ids)
    for subset in ("H", "HM", "HMO"):
        chosen = [spec for spec in specs if spec["feature_recipe"]["subset"] == subset]
        assert len(chosen) == 33
        hgb = [spec for spec in chosen if spec["family"] == "hgb"]
        assert len(hgb) == 16
        assert {(spec["parameters"]["learning_rate"],
                 spec["parameters"]["max_leaf_nodes"],
                 spec["parameters"]["min_samples_leaf"],
                 spec["parameters"]["l2_regularization"]) for spec in hgb} == {
            (learning_rate, leaves, min_samples, l2)
            for learning_rate in HGB_LEARNING_RATE for leaves in HGB_MAX_LEAF_NODES
            for min_samples in HGB_MIN_SAMPLES_LEAF for l2 in HGB_L2_REGULARIZATION}
        for spec in hgb:
            params = spec["parameters"]
            assert type(params["learning_rate"]) is float
            assert type(params["max_leaf_nodes"]) is int
            assert type(params["min_samples_leaf"]) is int
            assert type(params["l2_regularization"]) is float
    space = result["request"]["space"]["hgb"]
    assert space == {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 15],
                     "min_samples_leaf": [20, 100], "l2_regularization": [0.0, 1.0]}
    assert all(type(value) is int for value in space["max_leaf_nodes"])
    assert all(type(value) is int for value in space["min_samples_leaf"])
    assert all(type(value) is float for value in space["learning_rate"])
    assert all(type(value) is float for value in space["l2_regularization"])


@pytest.mark.parametrize("workers", [1, 8])
def test_v4_workers_accepted_without_changing_request_identity(workers):
    baseline = versions.compile_request(v4_request())
    document = v4_request()
    document["workers"] = workers
    worker_result = versions.compile_request(document)
    assert worker_result["request"] == baseline["request"]
    assert worker_result["request_id"] == baseline["request_id"]
    assert ([spec["candidate_spec_id"] for spec in worker_result["candidate_specs"]]
            == [spec["candidate_spec_id"] for spec in baseline["candidate_specs"]])
    assert worker_result["request_id"] == content_hash(baseline["request"])
    expected_ids = [
        content_hash({"request_id": baseline["request_id"],
                      "family": spec["family"],
                      "feature_recipe": spec["feature_recipe"],
                      "parameters": spec["parameters"]})
        for spec in baseline["candidate_specs"]]
    assert [spec["candidate_spec_id"] for spec in baseline["candidate_specs"]] == expected_ids


@pytest.mark.parametrize("workers", [0, 9, True])
def test_v4_invalid_workers_refuse_before_expansion(monkeypatch, workers):
    monkeypatch.setattr(compiler, "_specifications", fail_if_called)
    document = v4_request()
    document["workers"] = workers
    with pytest.raises(OpsError) as caught:
        versions.compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "stage-2 workers requires an integer from 1 through 8"
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("path,value", [
    (("space", "ridge", "alpha"), [0.1, 0.01]),
    (("space", "ridge", "alpha"), [0.1, 0.1]),
    (("space", "ridge", "alpha"), []),
    (("space", "ridge", "alpha"), [float(value) for value in range(1, 34)]),
    (("space", "hgb", "learning_rate"), [0.1, 0.05]),
    (("space", "hgb", "max_leaf_nodes"), [15, 7]),
    (("space", "hgb", "min_samples_leaf"), [20, 20]),
    (("space", "hgb", "min_samples_leaf"), []),
    (("space", "hgb", "l2_regularization"), [float(value) for value in range(0, 33)]),
    (("space", "hgb", "learning_rate"), [0.5, 2.0]),
    (("space", "hgb", "min_samples_leaf"), [0, 20]),
    (("space", "hgb", "l2_regularization"), [-1, 0]),
    (("space", "hgb", "max_leaf_nodes"), [1, 7]),
    (("space", "hgb", "max_leaf_nodes"), [65, 70]),
    (("space", "hgb", "max_leaf_nodes"), [7.0, 15.0]),
    (("space", "hgb", "learning_rate"), [True, 0.1]),
    (("space", "hgb", "max_leaf_nodes"), [True, 15]),
    (("space", "hgb", "min_samples_leaf"), [True, 20]),
    (("space", "hgb", "l2_regularization"), [True, 1]),
    (("space", "hgb", "learning_rate"), [float("nan"), 0.1]),
    (("space", "hgb", "learning_rate"), [0.1, float("inf")]),
    (("space", "hgb"), {"max_leaf_nodes": [7, 15], "min_samples_leaf": [20, 100],
                        "l2_regularization": [0, 1]}),
    (("space", "hgb"), {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [7, 15],
                        "min_samples_leaf": [20, 100], "l2_regularization": [0, 1],
                        "max_depth": [2, 3]}),
    (("space", "hgb"), {"max_leaf_nodes": [7, 15]}),
])
def test_v4_invalid_grids_refuse_before_expansion(monkeypatch, path, value):
    monkeypatch.setattr(compiler, "_specifications", fail_if_called)
    document = v4_request()
    set_value(document, path, value)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


def test_v4_candidate_ceiling_refuses_before_expansion(monkeypatch):
    monkeypatch.setattr(compiler, "_specifications", fail_if_called)
    document = v4_request()
    document["space"]["hgb"] = {
        "learning_rate": [round(0.001 * (index + 1), 4) for index in range(32)],
        "max_leaf_nodes": list(range(2, 34)),
        "min_samples_leaf": list(range(1, 33)),
        "l2_regularization": [float(index) for index in range(32)]}
    document["max_candidates"] = 10**9
    document["max_fits"] = 10**12
    with pytest.raises(OpsError) as caught:
        versions.compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert "ceiling" in caught.value.problem.message


def test_v4_design_grid_below_ceiling_still_compiles():
    result = versions.compile_request(v4_request())
    assert result["counts"]["candidates"] == 99


def test_v4_single_horizon_selector_scales_horizon_dependent_counts():
    default = versions.compile_request(v4_request())
    single = versions.compile_request(v4_request(), horizons=(1,))
    assert single["horizons"] == [1]
    assert single["counts"]["candidates"] == default["counts"]["candidates"] == 99
    assert single["counts"]["search_fits"] * 3 == default["counts"]["search_fits"]
    assert single["request_id"] == content_hash(single["request"])


@pytest.mark.parametrize("horizons", [[], [1, 1], [2], [True]])
def test_v4_invalid_horizons_refuse_through_versions(horizons):
    with pytest.raises(OpsError) as caught:
        versions.compile_request(v4_request(), horizons=horizons)
    assert caught.value.problem.code == "INVALID_EXPERIMENT_SPEC"


def test_v3_replay_and_v4_axis_refusal_through_versions(monkeypatch):
    document = v3_request()
    result = versions.compile_request(document)
    assert result["counts"]["candidates"] == 57
    assert result["request_id"] == content_hash(result["request"])
    assert versions.compile_request(v2_request())["counts"]["candidates"] == 33
    invalid = v3_request()
    invalid["space"]["hgb"] = copy.deepcopy(v4_request()["space"]["hgb"])
    monkeypatch.setattr(compiler, "_specifications", fail_if_called)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(invalid)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
