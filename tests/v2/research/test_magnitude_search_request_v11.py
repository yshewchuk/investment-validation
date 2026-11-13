import hashlib
import json

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments.magnitude_search_request import validate_workers
from experiments.magnitude_search_request_versions import compile_request
from experiments.magnitude_search_request_v3 import _space
from experiments.size_search_binding import _panel_columns, _panel_subset


_G = 4
_V9_SETS = ["H", "HM", "HMO", "MO-", "M"]


def _request(version, feature_sets=None, hash_subset=None):
    sets = (list(feature_sets) if feature_sets is not None else
            _V9_SETS + (["HMOS"] if version >= 10 else [])
            + (["HMOSX"] if version >= 11 else []))
    count = len(sets) * _G + (1 if "HMO" in sets else 0)
    if hash_subset is None:
        hash_subset = ("HMOSX" if version >= 11 else
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


def _candidate_digest(compiled):
    return hashlib.sha256(json.dumps(
        sorted(s["candidate_spec_id"] for s in compiled["candidate_specs"])
    ).encode()).hexdigest()[:16]


def test_v9_compiles_stably():
    assert compile_request(_request(9)) == compile_request(_request(9))
    compiled = compile_request(_request(9))
    assert compiled["request_id"] == (
        "sha256:eae25eb35a95417579ae220d192fec62b5aa6a6dc4eada955a3a2f77d57fcd02")
    assert _candidate_digest(compiled) == "d69cb507ac599b40"


def test_v10_uses_the_hmos_dictionary_and_stays_stable():
    first = compile_request(_request(10))
    assert first == compile_request(_request(10))
    assert first["request"]["feature_dictionary_hash"] == recipe("HMOS")["dictionary_hash"]
    assert all(spec["feature_recipe"] == recipe("HMOS")
               for spec in first["candidate_specs"]
               if spec["feature_recipe"]["subset"] == "HMOS")
    assert first["request_id"] == (
        "sha256:8087b31da42d0eb300472c75d696f87f5bd27edd0ab61d09fb23bb10efd28d45")
    assert _candidate_digest(first) == "88c60a0f3bbebbd5"


def test_v10_and_v11_request_ids_differ_for_the_same_grid():
    same_grid = _V9_SETS + ["HMOS"]
    assert (compile_request(_request(10, feature_sets=same_grid))["request_id"]
            != compile_request(_request(11, feature_sets=same_grid))["request_id"])


def test_v11_compiles_all_seven_sets():
    compiled = compile_request(_request(11))
    sets = _V9_SETS + ["HMOS", "HMOSX"]
    assert compiled["request"]["feature_sets"] == sets
    assert {spec["feature_recipe"]["subset"] for spec in compiled["candidate_specs"]} == set(sets)
    assert all(spec["feature_recipe"] == recipe("HMOSX")
               for spec in compiled["candidate_specs"]
               if spec["feature_recipe"]["subset"] == "HMOSX")
    assert not [spec for spec in compiled["candidate_specs"]
                if spec["family"] == "size_v1_4"
                and spec["feature_recipe"]["subset"] in ("HMOS", "HMOSX")]


@pytest.mark.parametrize("feature_sets, expected", [
    (["HMOS"], 4),
    (["HMOS", "HMOSX"], 8),
    (["HMOSX"], 4),
    (_V9_SETS + ["HMOS", "HMOSX"], 29),
    (_V9_SETS + ["HMOS"], 25),
])
def test_v11_candidate_counts(feature_sets, expected):
    compiled = compile_request(_request(11, feature_sets=feature_sets))
    assert compiled["counts"]["candidates"] == expected


@pytest.mark.parametrize("hash_subset", ["HMOS", "HMO"])
def test_v11_requires_the_hmosx_hash(hash_subset):
    with pytest.raises(OpsError) as caught:
        compile_request(_request(11, hash_subset=hash_subset))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v10_refuses_the_hmosx_hash():
    with pytest.raises(OpsError) as caught:
        compile_request(_request(10, hash_subset="HMOSX"))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("feature_sets", [
    ["HMOSX", "HMOS"], ["HMOS", "HMOS"], ["X"], []])
def test_v11_refuses_empty_duplicate_unknown_or_out_of_order_feature_sets(feature_sets):
    with pytest.raises(OpsError, match="nonempty ordered subset"):
        _space({"schema_version": "magnitude_search_request.v11",
                "feature_sets": feature_sets, "space": {}})


def test_v10_refuses_the_hmosx_set():
    with pytest.raises(OpsError, match="nonempty ordered subset"):
        _space({"schema_version": "magnitude_search_request.v10",
                "feature_sets": ["HMOSX"], "space": {}})


def test_panel_subset_tracks_the_request_version():
    assert _panel_subset("magnitude_search_request.v11") == "HMOSX"
    assert _panel_subset("magnitude_search_request.v10") == "HMOS"
    assert _panel_subset("magnitude_search_request.v9") == "HMO"


def test_v11_panel_projection_is_unique_and_complete():
    model_columns = recipe("HMOSX")["model_columns"]
    columns = _panel_columns("magnitude_search_request.v11", model_columns)
    assert len(columns) == len(set(columns))
    assert all(name in columns for name in model_columns)


def test_v11_workers_use_the_existing_bounded_worker_contract():
    assert validate_workers(
        {"schema_version": "magnitude_search_request.v11", "workers": 4}) == 4
