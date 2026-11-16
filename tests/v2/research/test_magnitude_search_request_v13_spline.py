import pytest

from engine.v2.models.magnitude_elasticnet_spline_recipe import SPLINE_COLUMNS
from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments.magnitude_candidate_recipe import candidate_recipe
from experiments.magnitude_search_request_versions import compile_request


_G = 4
_V9_SETS = ["H", "HM", "HMO", "MO-", "M"]


def _request(version, feature_sets=None, hash_subset=None):
    sets = (list(feature_sets) if feature_sets is not None else
            _V9_SETS + (["HMOS"] if version >= 10 else [])
            + (["HMOSX"] if version >= 11 else [])
            + (["HMOSXL", "HMOSXLP"] if version >= 12 else []))
    count = len(sets) * _G + (1 if "HMO" in sets else 0)
    if hash_subset is None:
        hash_subset = ("HMOSXLP" if version >= 12 else
                       "HMOSX" if version >= 11 else
                       "HMOS" if version >= 10 else "HMO")
    return {
        "schema_version": f"magnitude_search_request.v{version}",
        "feature_snapshot_id": "snap_" + "0" * 32,
        "history_snapshot_id": "snap_" + "1" * 32,
        "feature_dictionary_hash": recipe(hash_subset)["dictionary_hash"],
        "eligibility_policy_ref": {
            "schema_ref": "group_eligibility_plan.v1",
            "plan_id": "sha256:" + "a" * 64,
            "policy_id": eligibility_policy(
                version="grouped_ticker_embargo.v1", embargo_days=30)["policy_id"],
        },
        "feature_sets": sets,
        "space": {
            "ridge": {"alpha": [0.1]},
            "elasticnet": {"alpha": [0.01], "l1_ratio": [0.25]},
            "hgb": {"learning_rate": [0.1], "max_leaf_nodes": [7],
                    "min_samples_leaf": [2], "l2_regularization": [0.0]},
            "size_v1_4": {},
            "mlp": {"profile": ["chrono15_relu32_l2_0p01"]},
        },
        "outer_years": ["2023"],
        "inner_years": {"2023": ["2021", "2022"]},
        "max_candidates": count,
        "max_fits": 3 * (count * 2 + 1),
        "comparison": {
            "reference": "size_v1_4_causal_refit", "sanity_reference": "history_mean",
            "max_logical_fits": 3, "max_component_fits": 15,
        },
        "selection": {
            "metric_weights": {"rmse": 0.6, "mae": 0.3, "large_move_underprediction": 0.1},
            "lead_weights": {"1": 1.0, "3": 0.0, "5": 0.0},
            "large_move_threshold_pct": 10.0, "tie_band_floor": 0.005,
        },
        "limits": {
            "child": {"cpu_seconds": 100, "memory_bytes": 1000000000, "wall_seconds": 100},
            "parent": {"memory_bytes": 1000000000, "wall_seconds": 100},
            "build": {"memory_bytes": 1000000000, "wall_seconds": 100},
            "job": {"wall_seconds": 100},
        },
        "workers": 4,
    }


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
    assert {spec["parameters"]["spline_knots"] for spec in elastics} == {0, 4, 6}


def test_v13_spline_elasticnet_candidates_pass_candidate_recipe():
    compiled = compile_request(_spline_request([0, 4, 6], feature_sets=_SPLINE_SETS))
    emitted = set()
    for spec in compiled["candidate_specs"]:
        if spec["family"] != "elasticnet":
            continue
        knots = spec["parameters"]["spline_knots"]
        emitted.add(knots)
        built = candidate_recipe("elasticnet", spec["parameters"],
            columns=list(SPLINE_COLUMNS), feature_dictionary_hash=_HASH)
        assert built["schema_version"] == (
            "magnitude_elasticnet_recipe.v1"
            if knots == 0
            else "magnitude_elasticnet_spline_recipe.v1")
        if knots > 0:
            assert built["preprocessing"]["spline"]["n_knots"] == knots
        else:
            assert "spline" not in built["preprocessing"]
    assert emitted == {0, 4, 6}


def test_v13_accepts_zero_knots_for_default_sets():
    compiled = compile_request(_spline_request([0]))
    elasticnet = compiled["request"]["space"]["elasticnet"]
    assert set(elasticnet) == {"alpha", "l1_ratio", "spline_knots"}
    assert elasticnet["spline_knots"] == [0]


@pytest.mark.parametrize("knots", [[8], [7], [3], [4, 4], [6, 4],
                                   [4.0], [True], [], 4])
def test_v13_refuses_invalid_spline_knots(knots):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_spline_request(knots, feature_sets=_SPLINE_SETS))


def test_v13_refuses_positive_knots_with_partial_spline_sets():
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(_spline_request([4], feature_sets=["HMOS", "HMOSX"]))


@pytest.mark.parametrize("version", [9, 10, 11, 12])
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
