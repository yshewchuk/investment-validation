import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import recipe
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


def test_v11_and_v12_request_ids_are_unchanged():
    assert compile_request(_request(12))["request_id"] == (
        "sha256:da972b03e329a2ad101f4787f5289f2aadcdaa7ff7df0f56bc5f53eb66ad130c")
    assert compile_request(_request(11))["request_id"] == (
        "sha256:a58616fec23a3c84b3e981968e6d34a10793593775ccddc0890e1fde4f0fb001")


def test_v13_with_ridge_keeps_the_v12_candidate_count_but_changes_identity():
    with_ridge = compile_request(_request(13))
    v12 = compile_request(_request(12))
    assert with_ridge["counts"]["candidates"] == v12["counts"]["candidates"]
    assert with_ridge["request_id"] != v12["request_id"]


def test_v13_omitting_ridge_drops_only_ridge_candidates():
    with_ridge = compile_request(_request(13))
    request = _request(13)
    del request["space"]["ridge"]
    without_ridge = compile_request(request)
    assert not [spec for spec in without_ridge["candidate_specs"] if spec["family"] == "ridge"]
    assert without_ridge["counts"]["candidates"] == (
        with_ridge["counts"]["candidates"] - len(request["feature_sets"]) * 1)
    assert without_ridge["request_id"] != with_ridge["request_id"]
    assert request["max_candidates"] >= without_ridge["counts"]["candidates"]
    assert request["max_fits"] >= without_ridge["counts"]["search_fits"]


@pytest.mark.parametrize("version", [9, 12])
def test_older_versions_still_require_ridge(version):
    request = _request(version)
    del request["space"]["ridge"]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


@pytest.mark.parametrize("family", ["elasticnet", "hgb", "mlp", "size_v1_4"])
def test_v13_refuses_omitting_any_other_family(family):
    request = _request(13)
    del request["space"][family]
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_v13_refuses_an_unknown_extra_family():
    request = _request(13)
    request["space"]["unknown"] = {}
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        compile_request(request)


def test_v13_accepts_the_new_mlp_profiles_and_v12_refuses_them():
    profiles = ["chrono15_relu128x64_l2_1", "chrono15_relu128x64x32_l2_1"]
    single = _request(13)
    single["space"]["mlp"] = {"profile": [profiles[0]]}
    assert compile_request(single)["counts"]["candidates"] == 37
    both = _request(13)
    both["space"]["mlp"] = {"profile": profiles}
    both["max_candidates"] = 46
    both["max_fits"] = 3 * (46 * 2 + 1)
    assert compile_request(both)["counts"]["candidates"] == 46
    for profile in profiles:
        older = _request(12)
        older["space"]["mlp"] = {"profile": [profile]}
        with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
            compile_request(older)
