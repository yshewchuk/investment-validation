import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import recipe
from experiments.magnitude_search_request_versions import compile_request
from experiments.magnitude_search_request_v3 import _space
from experiments.size_search_binding import (
    _dictionary_subset,
    _panel_columns,
    _panel_subset,
    _refuse_unwired_derived_sets,
)


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


def test_v12_compiles_all_nine_sets():
    compiled = compile_request(_request(12))
    sets = _V9_SETS + ["HMOS", "HMOSX", "HMOSXL", "HMOSXLP"]
    assert compiled["request"]["feature_sets"] == sets
    assert {spec["feature_recipe"]["subset"] for spec in compiled["candidate_specs"]} == set(sets)
    for subset in ("HMOSXL", "HMOSXLP"):
        assert all(spec["feature_recipe"] == recipe(subset)
                   for spec in compiled["candidate_specs"]
                   if spec["feature_recipe"]["subset"] == subset)
    assert not [spec for spec in compiled["candidate_specs"]
                if spec["family"] == "size_v1_4"
                and spec["feature_recipe"]["subset"] in ("HMOSXL", "HMOSXLP")]


@pytest.mark.parametrize("feature_sets, expected", [
    (["HMOSX", "HMOSXL", "HMOSXLP"], 12),
    (["HMOS", "HMOSX", "HMOSXL", "HMOSXLP"], 16),
    (_V9_SETS + ["HMOS", "HMOSX", "HMOSXL", "HMOSXLP"], 37),
])
def test_v12_candidate_counts(feature_sets, expected):
    compiled = compile_request(_request(12, feature_sets=feature_sets))
    assert compiled["counts"]["candidates"] == expected


@pytest.mark.parametrize("hash_subset", ["HMOSX", "HMO"])
def test_v12_requires_the_hmosxlp_hash(hash_subset):
    with pytest.raises(OpsError) as caught:
        compile_request(_request(12, hash_subset=hash_subset))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v11_refuses_the_hmosxlp_hash():
    with pytest.raises(OpsError) as caught:
        compile_request(_request(11, hash_subset="HMOSXLP"))
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


@pytest.mark.parametrize("version", [10, 11])
@pytest.mark.parametrize("feature_sets", [["HMOSXL"], ["HMOSXLP"]])
def test_v10_and_v11_refuse_the_derived_sets(version, feature_sets):
    with pytest.raises(OpsError, match="nonempty ordered subset") as caught:
        _space({"schema_version": f"magnitude_search_request.v{version}",
                "feature_sets": feature_sets, "space": {}})
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v12_refuses_out_of_order_derived_sets():
    with pytest.raises(OpsError, match="nonempty ordered subset") as caught:
        _space({"schema_version": "magnitude_search_request.v12",
                "feature_sets": ["HMOSXL", "HMOSX"], "space": {}})
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_refuse_unwired_derived_sets_rejects_derived_sets():
    schedule = compile_request(_request(12, feature_sets=["HMOSX", "HMOSXL"]))
    with pytest.raises(OpsError) as caught:
        _refuse_unwired_derived_sets(schedule)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_refuse_unwired_derived_sets_allows_wired_sets():
    _refuse_unwired_derived_sets(compile_request(_request(12, feature_sets=["HMOSX"])))


def test_dictionary_and_panel_subsets_track_the_request_version():
    assert [_dictionary_subset(f"magnitude_search_request.v{v}") for v in (9, 10, 11, 12)] == [
        "HMO", "HMOS", "HMOSX", "HMOSXLP"]
    assert [_panel_subset(f"magnitude_search_request.v{v}") for v in (9, 10, 11, 12)] == [
        "HMO", "HMOS", "HMOSX", "HMOSX"]


def test_v12_panel_projection_matches_v11_without_derived_names():
    v12_columns = _panel_columns("magnitude_search_request.v12",
                                 recipe(_panel_subset("magnitude_search_request.v12"))["model_columns"])
    v11_columns = _panel_columns("magnitude_search_request.v11", recipe("HMOSX")["model_columns"])
    assert v12_columns == v11_columns
    hmossx = set(recipe("HMOSX")["model_columns"])
    assert not [name for name in recipe("HMOSXLP")["model_columns"]
                if name not in hmossx and name in v12_columns]


def test_v12_request_id_differs_from_v11_and_compiles_stably():
    same_grid = _V9_SETS + ["HMOS", "HMOSX"]
    assert (compile_request(_request(12, feature_sets=same_grid))["request_id"]
            != compile_request(_request(11, feature_sets=same_grid))["request_id"])
    assert compile_request(_request(12)) == compile_request(_request(12))
