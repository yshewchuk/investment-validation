"""Opt-in neural profiles: bounded source-free declarations and legacy isolation."""
import copy
import json
import subprocess
import sys

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_search_request as compiler
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v5 import v5_request

PROFILES = ["relu32_l2_0p01", "relu64x32_l2_0p01", "relu64x32_l2_0p1"]


def v6_request(profiles=PROFILES):
    request = v5_request()
    request["schema_version"] = "magnitude_search_request.v6"
    request["space"]["mlp"] = {"profile": list(profiles)}
    request["max_candidates"] = 100 + 3 * len(profiles)
    request["max_fits"] = 3 * (request["max_candidates"] * 6 + 3)
    return request


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("horizons", [(5, 3, 1), (1,)])
def test_closed_profiles_counts_recipes_and_detached_identity(n, horizons):
    request = v6_request(PROFILES[:n])
    original = copy.deepcopy(request)
    result = compile_request(request, horizons=horizons)
    count, leads = 100 + 3 * n, len(horizons)
    assert result["counts"] == {
        "candidates": count, "inner_folds": 6, "outer_folds": 3,
        "search_fits": leads * (count * 6 + 3),
        "baseline_logical_fits": leads * 3, "baseline_component_fits": leads * 6,
        "total_logical_fits": leads * (count * 6 + 6),
        "total_component_fits": leads * (count * 6 + 18)}
    assert result["request_id"] == content_hash(result["request"])
    specs = result["candidate_specs"]
    ids = [s["candidate_spec_id"] for s in specs]
    assert len(ids) == len(set(ids)) == count and ids == sorted(ids)
    neural = [s for s in specs if s["family"] == "mlp"]
    assert {(s["feature_recipe"]["subset"], s["parameters"]["profile"]) for s in neural} == {
        (subset, profile) for subset in ("H", "HM", "HMO") for profile in PROFILES[:n]}
    assert len([s for s in specs if s["family"] == "size_v1_4"]) == 1
    for spec in neural:
        recipe = candidate_recipe("mlp", spec["parameters"], columns=["signal"],
                                  feature_dictionary_hash=request["feature_dictionary_hash"])
        assert recipe["component_fits"] == 1
        assert recipe["estimator"]["params"]["max_iter"] == 400
        assert recipe["estimator"]["params"]["early_stopping"] is False
    assert request == original and compile_request(request, horizons=horizons) == result
    result["request"]["space"]["mlp"]["profile"].clear()
    assert request == original


def _no_expansion(*args, **kwargs):
    pytest.fail("invalid request expanded specifications")


@pytest.mark.parametrize("profiles", [[], ["unknown"], [1], [True], [None], [{}],
    [PROFILES[0], PROFILES[0]], list(reversed(PROFILES)), PROFILES * 11, "relu32_l2_0p01"])
def test_invalid_profiles_refuse_before_expansion(monkeypatch, profiles):
    request = v6_request()
    request["space"]["mlp"]["profile"] = profiles
    monkeypatch.setattr(compiler, "_specifications", _no_expansion)
    with pytest.raises(OpsError) as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("change", [
    lambda r: r["space"].pop("mlp"), lambda r: r["space"].pop("size_v1_4"),
    lambda r: r["space"]["mlp"].update(alpha=[0.1]), lambda r: r.pop("selection"),
    lambda r: r.update(max_candidates=108), lambda r: r.update(max_fits=1970),
    lambda r: r.update(workers=1), lambda r: r.update(workers=2),
    lambda r: r["limits"]["child"].update(cpu_seconds=0),
    lambda r: r["limits"]["parent"].update(memory_bytes=0),
])
def test_v6_closed_shape_existing_caps_and_serial_policy(monkeypatch, change):
    request = v6_request()
    change(request)
    monkeypatch.setattr(compiler, "_specifications", _no_expansion)
    with pytest.raises(OpsError) as caught:
        compile_request(request)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v6_total_grid_ceiling_precedes_expansion(monkeypatch):
    request = v6_request()
    request["space"]["elasticnet"] = {"alpha": list(range(1, 33)),
                                     "l1_ratio": [i / 16 for i in range(1, 17)]}
    request.update(max_candidates=10000, max_fits=1000000)
    monkeypatch.setattr(compiler, "_specifications", _no_expansion)
    with pytest.raises(OpsError, match="1000-candidate"):
        compile_request(request)


def test_v6_requires_lead_one_before_expansion(monkeypatch):
    monkeypatch.setattr(compiler, "_specifications", _no_expansion)
    with pytest.raises(OpsError, match="lead-1"):
        compile_request(v6_request(), horizons=(5, 3))


def test_compilation_does_not_import_neural_training_or_comparator():
    probe = '''
import importlib.abc
import json
import sys
class RejectRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert not fullname.startswith(("experiments.size_comparator", "engine.v2.models.training")), fullname
sys.meta_path.insert(0, RejectRuntime())
from experiments.magnitude_search_request_versions import compile_request
assert compile_request(json.loads(sys.argv[1]))["counts"]["candidates"] == 109
'''
    result = subprocess.run([sys.executable, "-c", probe, json.dumps(v6_request())],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
