import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import feature_dictionary, recipe
from experiments.magnitude_search_request import validate_workers
from experiments.magnitude_search_request_versions import compile_request
from experiments.magnitude_search_request_v3 import _space


_V9_SETS = ["H", "HM", "HMO", "MO-", "M"]


def test_v9_feature_subsets_contain_only_the_declared_nonhistory_features():
    mo = feature_dictionary("MO-")
    market = feature_dictionary("M")
    assert len(market) == 15
    assert len(mo) == 28
    assert {"ema12r_abs", "mean_prior_or_implied"}.isdisjoint(
        {field["name"] for field in mo + market})
    for field in mo + market:
        assert not {dependency["role"] for dependency in field["source_dependencies"]} & {
            "stored_history", "prior_outcomes"
        }


def test_v9_request_sets_and_expanded_candidate_fit_budgets():
    request = {
        "schema_version": "magnitude_search_request.v9",
        "feature_snapshot_id": "snap_" + "0" * 32,
        "history_snapshot_id": "snap_" + "1" * 32,
        "feature_dictionary_hash": recipe("HMO")["dictionary_hash"],
        "eligibility_policy_ref": {
            "schema_ref": "group_eligibility_plan.v1",
            "plan_id": "sha256:" + "a" * 64,
            "policy_id": eligibility_policy(
                version="grouped_ticker_embargo.v1", embargo_days=30)["policy_id"],
        },
        "feature_sets": list(_V9_SETS),
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
        "max_candidates": 21,
        "max_fits": 129,
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
    compiled = compile_request(request)
    assert compiled["request"]["feature_sets"] == _V9_SETS
    assert "workers" not in compiled["request"]
    assert compiled["counts"]["candidates"] == 21
    assert compiled["counts"]["search_fits"] == 129
    assert compiled["counts"]["total_logical_fits"] == 132
    assert compiled["counts"]["total_component_fits"] == 144
    assert len(compiled["candidate_specs"]) == 21
    assert {spec["feature_recipe"]["subset"] for spec in compiled["candidate_specs"]} == set(_V9_SETS)
    for subsets, candidate_count in ((["MO-", "M"], 9), (["M"], 5), (["HMO", "M"], 9)):
        request["feature_sets"] = subsets
        request["max_candidates"] = candidate_count
        request["max_fits"] = 3 * (candidate_count * 2 + 1)
        selected = compile_request(request)
        assert selected["request"]["feature_sets"] == subsets
        assert selected["counts"]["candidates"] == candidate_count
        assert selected["counts"]["search_fits"] == request["max_fits"]
        assert {spec["feature_recipe"]["subset"] for spec in selected["candidate_specs"]} == set(subsets)


def test_v9_workers_use_the_existing_bounded_worker_contract():
    assert validate_workers({"schema_version": "magnitude_search_request.v9"}) == 1
    assert validate_workers({"schema_version": "magnitude_search_request.v9", "workers": 4}) == 4


@pytest.mark.parametrize("feature_sets", [[], ["M", "MO-"], ["M", "M"], ["MO-", "unknown"]])
def test_v9_refuses_empty_duplicate_unknown_or_out_of_order_feature_sets(feature_sets):
    with pytest.raises(OpsError):
        _space({"schema_version": "magnitude_search_request.v9",
                "feature_sets": feature_sets, "space": {}})
