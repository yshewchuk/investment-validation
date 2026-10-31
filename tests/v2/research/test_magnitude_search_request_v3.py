"""Closed v3 bounded-grid compilation and refusal-before-expansion contracts."""
import copy

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_request as compiler
from experiments import magnitude_search_request_versions as versions
from tests.v2.research.test_magnitude_search_request import request


RIDGE_ALPHA = [0.01, 0.1, 1.0, 10.0, 100.0]
ELASTICNET_ALPHA = [0.001, 0.003, 0.01, 0.03, 0.1, 1.0]
ELASTICNET_L1_RATIO = [0.25, 0.75]


def v3_request():
    document = request()
    document["schema_version"] = "magnitude_search_request.v3"
    document["space"] = {
        "ridge": {"alpha": list(RIDGE_ALPHA)},
        "elasticnet": {"alpha": list(ELASTICNET_ALPHA),
            "l1_ratio": list(ELASTICNET_L1_RATIO)},
        "hgb": {"max_leaf_nodes": [7, 15]}}
    document["max_candidates"] = 57
    document["max_fits"] = 1035
    return document


def set_value(document, path, value):
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


def fail_if_called(*args, **kwargs):
    raise AssertionError("unexpected candidate expansion")


def test_v3_closed_grid_counts_and_identities():
    result = versions.compile_request(v3_request())
    assert result["counts"]["candidates"] == 57
    assert result["counts"]["search_fits"] == 1035
    assert result["request_id"] == content_hash(result["request"])
    specs = result["candidate_specs"]
    ids = [spec["candidate_spec_id"] for spec in specs]
    assert len(ids) == len(set(ids)) == 57 and ids == sorted(ids)
    for subset in ("H", "HM", "HMO"):
        chosen = [spec for spec in specs if spec["feature_recipe"]["subset"] == subset]
        assert len(chosen) == 19
        assert {spec["parameters"]["alpha"] for spec in chosen
                if spec["family"] == "ridge"} == set(RIDGE_ALPHA)
        assert {(spec["parameters"]["alpha"], spec["parameters"]["l1_ratio"])
                for spec in chosen if spec["family"] == "elasticnet"} == {
                    (alpha, ratio) for alpha in ELASTICNET_ALPHA
                    for ratio in ELASTICNET_L1_RATIO}
        assert {spec["parameters"]["max_leaf_nodes"] for spec in chosen
                if spec["family"] == "hgb"} == {7, 15}


@pytest.mark.parametrize("dictionary_hash,label", [
    (versions._V2_HASH, "v2"),
    (versions._V3_HASH, "v3")])
def test_v3_recognized_dictionary_adapts_recipes(dictionary_hash, label):
    document = v3_request()
    document["feature_dictionary_hash"] = dictionary_hash
    caller_input = copy.deepcopy(document)
    result = versions.compile_request(document)
    assert document == caller_input
    assert result["request"]["feature_dictionary_hash"] == dictionary_hash
    assert result["request_id"] == content_hash(result["request"])
    for spec in result["candidate_specs"]:
        recipe = spec["feature_recipe"]
        assert recipe["dictionary_hash"] == content_hash(recipe["dictionary"])
        assert spec["candidate_spec_id"] == content_hash({
            "request_id": result["request_id"], "family": spec["family"],
            "feature_recipe": spec["feature_recipe"], "parameters": spec["parameters"]})


def test_v3_narrower_in_bounds_grid_changes_every_identity():
    full = versions.compile_request(v3_request())
    narrow = v3_request()
    narrow["space"]["ridge"]["alpha"] = [0.1, 10.0]
    narrow["space"]["elasticnet"]["alpha"] = [0.01, 0.1, 1.0]
    narrow["space"]["elasticnet"]["l1_ratio"] = [0.25]
    result = versions.compile_request(narrow)
    assert result["counts"]["candidates"] == 21
    assert result["request_id"] != full["request_id"]
    assert {spec["candidate_spec_id"] for spec in full["candidate_specs"]}.isdisjoint(
        spec["candidate_spec_id"] for spec in result["candidate_specs"])


@pytest.mark.parametrize("path,value", [
    (("space", "ridge", "alpha"), [0.0, 1.0]),
    (("space", "ridge", "alpha"), [-1.0, 1.0]),
    (("space", "ridge", "alpha"), [1e-9, 1.0]),
    (("space", "ridge", "alpha"), [1.0, 1001.0]),
    (("space", "ridge", "alpha"), [10.0, 1.0]),
    (("space", "ridge", "alpha"), [1.0, 1.0]),
    (("space", "ridge", "alpha"), [True, 1]),
    (("space", "ridge", "alpha"), [float("nan"), 1.0]),
    (("space", "ridge", "alpha"), [float(value) for value in range(1, 34)]),
    (("space", "ridge", "alpha"), [10**1000]),
    (("space", "elasticnet", "l1_ratio"), [0.5, 1.5]),
    (("space", "hgb", "max_leaf_nodes"), [7, 31])])
def test_v3_invalid_grids_refuse_before_expansion(monkeypatch, path, value):
    monkeypatch.setattr(compiler, "_specifications", fail_if_called)
    document = v3_request()
    set_value(document, path, value)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


def test_v2_fixture_replays_identically_through_both_entrypoints():
    document = request()
    assert versions.compile_request(document) == compiler.compile_request(document)


@pytest.mark.parametrize("workers", [1, 3])
def test_v3_worker_count_does_not_change_search_identity(workers):
    baseline = versions.compile_request(v3_request())
    document = {**v3_request(), "workers": workers}
    result = versions.compile_request(document)
    assert result["request"] == baseline["request"]
    assert result["request_id"] == baseline["request_id"]
    assert [spec["candidate_spec_id"] for spec in result["candidate_specs"]] == [
        spec["candidate_spec_id"] for spec in baseline["candidate_specs"]]


def test_v3_omitted_workers_preserves_request_identity():
    document = v3_request()
    result = versions.compile_request(document)
    assert result["request"] == document and "workers" not in result["request"]
    assert result["request_id"] == content_hash(document)


@pytest.mark.parametrize("workers", [0, 9, True, 1.0, "3"])
def test_v3_invalid_workers_refuse_nonretryably(workers):
    document = v3_request()
    document["workers"] = workers
    with pytest.raises(OpsError) as caught:
        versions.compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
