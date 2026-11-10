import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_qualified_table import admit_columns, recipe
from experiments.magnitude_search_request import validate_workers
from experiments.magnitude_search_request_schema import LATEST
from experiments.magnitude_search_request_versions import compile_request
from experiments.magnitude_search_request_v3 import _space
from experiments.size_search_binding import _panel_columns


_V9_SETS = ["H", "HM", "HMO", "MO-", "M"]


def _request(version):
    sets = _V9_SETS + (["HMOS"] if version >= 10 else [])
    count = len(sets) * 4 + 1
    return {
        "schema_version": f"magnitude_search_request.v{version}",
        "feature_snapshot_id": "snap_" + "0" * 32,
        "history_snapshot_id": "snap_" + "1" * 32,
        "feature_dictionary_hash": recipe("HMOS" if version >= 10 else "HMO")["dictionary_hash"],
        "eligibility_policy_ref": {
            "schema_ref": "group_eligibility_plan.v1",
            "plan_id": "sha256:" + "a" * 64,
            "policy_id": eligibility_policy(
                version="grouped_ticker_embargo.v1", embargo_days=30)["policy_id"],
        },
        "feature_sets": list(sets),
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


def test_v9_request_id_and_candidate_hashes_are_stable():
    compiled = compile_request(_request(9))
    assert compiled["request_id"] == (
        "sha256:eae25eb35a95417579ae220d192fec62b5aa6a6dc4eada955a3a2f77d57fcd02")
    assert content_hash([spec["candidate_spec_id"] for spec in compiled["candidate_specs"]]) == (
        "sha256:04e9efe21864eb3c56efdddceec44d9e99ee2010cb891de7022d789df639909f")


def test_hmos_dictionary_hash_is_registered_and_differs_from_hmo():
    assert recipe("HMOS")["dictionary_hash"] == (
        "sha256:ab953ea4b9370f5d848871aac7bf827040c36e021515f7cf85af63c267f94b2e")
    assert recipe("HMOS")["dictionary_hash"] != recipe("HMO")["dictionary_hash"]


def test_latest_request_version_is_ten():
    assert LATEST == 10


def test_v10_compiles_all_six_sets_with_only_hmo_size_candidate():
    compiled = compile_request(_request(10))
    assert compiled["counts"]["candidates"] == 25
    assert compiled["request"]["feature_sets"] == _V9_SETS + ["HMOS"]
    assert {spec["feature_recipe"]["subset"] for spec in compiled["candidate_specs"]} == set(
        _V9_SETS + ["HMOS"])
    assert all(spec["feature_recipe"] == recipe("HMOS") for spec in compiled["candidate_specs"]
               if spec["feature_recipe"]["subset"] == "HMOS")
    assert not [spec for spec in compiled["candidate_specs"] if spec["family"] == "size_v1_4"
                and spec["feature_recipe"]["subset"] == "HMOS"]


def test_v10_compiles_the_hmos_set_alone():
    request = _request(10)
    request.update(feature_sets=["HMOS"], max_candidates=4, max_fits=27)
    compiled = compile_request(request)
    assert compiled["counts"]["candidates"] == 4
    assert {spec["feature_recipe"]["subset"] for spec in compiled["candidate_specs"]} == {"HMOS"}


def test_v10_requires_the_hmos_hash_and_v9_requires_the_hmo_hash():
    v10 = _request(10)
    v10["feature_dictionary_hash"] = recipe("HMO")["dictionary_hash"]
    with pytest.raises(OpsError) as caught:
        compile_request(v10)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    v9 = _request(9)
    v9["feature_dictionary_hash"] = recipe("HMOS")["dictionary_hash"]
    with pytest.raises(OpsError) as caught:
        compile_request(v9)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"


def test_v10_accepts_the_hmos_hash_without_the_hmos_set():
    request = _request(10)
    request.update(feature_sets=_V9_SETS, max_candidates=21, max_fits=129)
    compiled = compile_request(request)
    assert compiled["request"]["feature_sets"] == _V9_SETS
    assert compiled["counts"]["candidates"] == 21


def test_prepare_accepts_hmos_candidates(monkeypatch):
    from experiments import size_search_binding

    class ReachedRegistration(Exception):
        pass

    request = _request(10)
    monkeypatch.setattr(size_search_binding, "_sources", lambda: {})
    monkeypatch.setattr(size_search_binding, "validate_panel_request",
                        lambda panel_request: {"horizons": [1],
                                                "feature_snapshot_id": request["feature_snapshot_id"],
                                                "history_snapshot": request["history_snapshot_id"],
                                                "as_of_month": "2023-01-01", "event_ids": []})
    monkeypatch.setattr(size_search_binding, "compile_request",
                        lambda request, *, horizons: {"horizons": [1]})
    def reach_registration(*args, **kwargs):
        raise ReachedRegistration("past the guard")

    monkeypatch.setattr(size_search_binding, "prepare_native_registration", reach_registration)
    with pytest.raises(ReachedRegistration):
        size_search_binding.prepare(None, {}, request, {"horizons": [1]},
                                    scope=None, training_start=None, labels_before=None)


def test_hmos_candidates_get_season_columns_and_hmo_do_not():
    from engine.v2.research.magnitude_panel_qualified_table import SEASON
    compiled = compile_request(_request(10))
    for spec in compiled["candidate_specs"]:
        feature_recipe = spec["feature_recipe"]
        if feature_recipe["subset"] == "HMOS":
            assert all(name in feature_recipe["model_columns"] for name in SEASON)
        elif feature_recipe["subset"] == "HMO":
            assert not any(name in feature_recipe["model_columns"] for name in SEASON)
    hmos_projection = _panel_columns("magnitude_search_request.v10", recipe("HMOS")["model_columns"])
    assert all(name in hmos_projection for name in SEASON)
    admit_columns(hmos_projection, season=True)
    hmo_projection = _panel_columns("magnitude_search_request.v9", recipe("HMO")["model_columns"])
    assert not any(name in hmo_projection for name in SEASON)


@pytest.mark.parametrize("feature_sets", [[], ["HMOS", "M"], ["M", "M"], ["HMO", "x"]])
def test_v10_refuses_empty_duplicate_unknown_or_out_of_order_feature_sets(feature_sets):
    with pytest.raises(OpsError, match="nonempty ordered subset"):
        _space({"schema_version": "magnitude_search_request.v10",
                "feature_sets": feature_sets, "space": {}})


def test_v9_refuses_the_hmos_set():
    with pytest.raises(OpsError, match="nonempty ordered subset"):
        _space({"schema_version": "magnitude_search_request.v9",
                "feature_sets": ["HMOS"], "space": {}})


def test_v10_panel_projection_is_admissible():
    admit_columns(_panel_columns(
        "magnitude_search_request.v10", recipe("HMOS")["model_columns"]), season=True)


def test_v10_workers_use_the_existing_bounded_worker_contract():
    assert validate_workers(
        {"schema_version": "magnitude_search_request.v10", "workers": 4}) == 4


@pytest.mark.parametrize("workers", [0, 65])
def test_v10_workers_refuse_what_v9_refuses(workers):
    with pytest.raises(OpsError):
        validate_workers({"schema_version": "magnitude_search_request.v9", "workers": workers})
    with pytest.raises(OpsError):
        validate_workers({"schema_version": "magnitude_search_request.v10", "workers": workers})
