"""v5 fixed size_v1_4 family declaration and refusal-before-expansion contracts."""
import json
import subprocess
import sys

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_request as compiler
from experiments import magnitude_search_request_versions as versions
from tests.v2.research.test_magnitude_search_request_v4 import v4_request


def v5_request():
    request = v4_request()
    request["schema_version"] = "magnitude_search_request.v5"
    request["space"]["size_v1_4"] = {}
    inner = sum(len(years) for years in request["inner_years"].values())
    outer = len(request["outer_years"])
    request["max_candidates"] = 100
    request["max_fits"] = 3 * (100 * inner + outer)
    return request


def test_v5_declares_fixed_size_v1_4_family_without_changing_v4():
    result = versions.compile_request(v5_request())
    assert result["request"]["space"]["size_v1_4"] == {}
    assert result["counts"]["candidates"] == len(result["candidate_specs"]) == 100
    size = [spec for spec in result["candidate_specs"] if spec["family"] == "size_v1_4"]
    assert len(size) == 1
    assert size[0]["parameters"] == {}
    assert size[0]["feature_recipe"]["subset"] == "HMO"
    assert versions.compile_request(v5_request()) == result
    assert versions.compile_request(v4_request())["counts"]["candidates"] == 99


@pytest.mark.parametrize("horizons,logical,components", [((5, 3, 1), 1809, 1854), ((1,), 603, 618)])
def test_v5_counts_one_blend_and_worst_case_selected_outer_blend(horizons, logical, components):
    counts = versions.compile_request(v5_request(), horizons=horizons)["counts"]
    # Six inner folds: 99 single estimators + one two-component blend each.
    # Three outer folds: allow a two-component winner AND a two-component comparator.
    assert counts["inner_folds"] == 6 and counts["outer_folds"] == 3
    assert counts["search_fits"] == logical
    assert counts["total_logical_fits"] == logical + 3 * len(horizons)
    assert counts["total_component_fits"] == components


def test_v5_declaration_does_not_import_runtime_recipe_or_training():
    probe = '''
import importlib.abc
import json
import sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("experiments.size_comparator", "engine.v2.models.training")), fullname
sys.meta_path.insert(0, RejectRuntime())
from experiments.magnitude_search_request_versions import compile_request
assert compile_request(json.loads(sys.argv[1]))["counts"]["candidates"] == 100
'''
    result = subprocess.run([sys.executable, "-c", probe, json.dumps(v5_request())],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def _fail_if_called(*args, **kwargs):
    raise AssertionError("specifications must not expand during refusal")


def test_v5_missing_size_v1_4_refuses_before_expansion(monkeypatch):
    request = v5_request()
    del request["space"]["size_v1_4"]
    monkeypatch.setattr(compiler, "_specifications", _fail_if_called)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("field,value", [("max_candidates", 99), ("max_fits", 1808)])
def test_v5_count_caps_refuse_before_expansion(monkeypatch, field, value):
    request = v5_request()
    request[field] = value
    monkeypatch.setattr(compiler, "_specifications", _fail_if_called)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v5_candidate_ceiling_still_refuses_before_expansion(monkeypatch):
    request = v5_request()
    request["space"]["elasticnet"] = {"alpha": list(range(1, 33)),
                                    "l1_ratio": [index / 16 for index in range(1, 17)]}
    request.update(max_candidates=10000, max_fits=1000000)
    monkeypatch.setattr(compiler, "_specifications", _fail_if_called)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("workers", [1, 8])
def test_v5_workers_do_not_change_request_identity(workers):
    baseline = versions.compile_request(v5_request())
    request = v5_request()
    request["workers"] = workers
    worker_result = versions.compile_request(request)
    assert worker_result["request"] == baseline["request"]
    assert worker_result["request_id"] == baseline["request_id"]
    assert ([spec["candidate_spec_id"] for spec in worker_result["candidate_specs"]]
            == [spec["candidate_spec_id"] for spec in baseline["candidate_specs"]])
    size = [spec for spec in worker_result["candidate_specs"] if spec["family"] == "size_v1_4"]
    assert len(size) == 1
    assert size[0]["feature_recipe"]["subset"] == "HMO"
    assert worker_result["request_id"] == content_hash(baseline["request"])
    expected_candidate_ids = [
        content_hash({"request_id": baseline["request_id"], "family": spec["family"],
                      "feature_recipe": spec["feature_recipe"], "parameters": spec["parameters"]})
        for spec in baseline["candidate_specs"]]
    assert [spec["candidate_spec_id"] for spec in baseline["candidate_specs"]] == expected_candidate_ids


@pytest.mark.parametrize("workers", [0, 9, True])
def test_v5_invalid_workers_refuse_before_expansion(monkeypatch, workers):
    request = v5_request()
    request["workers"] = workers
    monkeypatch.setattr(compiler, "_specifications", _fail_if_called)
    with pytest.raises(OpsError) as caught:
        versions.compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == "stage-2 workers requires an integer from 1 through 8"
    assert request["workers"] == workers
