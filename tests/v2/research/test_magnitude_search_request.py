"""Independent closed-grid/count oracles for source-free stage-2 declarations."""
import copy
import json
import sqlite3
import subprocess
import sys
from decimal import Decimal

import pytest

from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, content_hash
from engine.v2.ops.errors import OpsError
from engine.v2.research.eligibility_policy import eligibility_policy
from engine.v2.research.magnitude_panel_table import recipe
from experiments import magnitude_search_request as compiler


def request():
    """Synthetic identity tokens and deliberately tiny budgets, not a runnable request."""
    policy = eligibility_policy(version="grouped_ticker_embargo.v1", embargo_days=30)
    return {"schema_version": "magnitude_search_request.v2",
        "feature_snapshot_id": "snap_" + "1" * 32, "history_snapshot_id": "snap_" + "2" * 32,
        "feature_dictionary_hash": recipe()["dictionary_hash"],
        "eligibility_policy_ref": {"schema_ref": "group_eligibility_plan.v1",
            "plan_id": "sha256:" + "3" * 64, "policy_id": policy["policy_id"]},
        "feature_sets": ["H", "HM", "HMO"], "space": {
            "ridge": {"alpha": [0.1, 1.0, 10.0]},
            "elasticnet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.25, 0.75]},
            "hgb": {"max_leaf_nodes": [7, 15]}},
        "outer_years": ["2022", "2023", "2024"],
        "inner_years": {"2022": ["2020", "2021"], "2023": ["2021", "2022"], "2024": ["2022", "2023"]},
        "max_candidates": 33, "max_fits": 603,
        "comparison": {"reference": "size_v1_4_causal_refit", "sanity_reference": "history_mean",
            "max_logical_fits": 9, "max_component_fits": 18},
        "limits": {"child": {"memory_bytes": 1, "cpu_seconds": 2, "wall_seconds": 3},
            "parent": {"memory_bytes": 4, "wall_seconds": 5},
            "build": {"memory_bytes": 6, "wall_seconds": 7}, "job": {"wall_seconds": 8}}}


def set_value(document, path, value):
    for key in path[:-1]:
        document = document[key]
    document[path[-1]] = value


def fail_if_called(*args, **kwargs):
    raise AssertionError("unexpected expansion, source access, write or fit")


@pytest.fixture
def no_expansion(monkeypatch):
    monkeypatch.setattr(compiler, "_specifications", fail_if_called)


def refuses(document, message):
    with pytest.raises(OpsError) as caught:
        compiler.compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.message == message


def test_exact_closed_grid_and_independent_schedule_counts():
    document = request()
    result = compiler.compile_request(document)
    assert set(result) == {"schema_version", "request", "request_id", "horizons",
                           "candidate_specs", "schedule", "counts"}
    assert result["schema_version"] == "magnitude_search_schedule.v2"
    assert result["request"] == document
    assert result["request_id"] == content_hash(document)
    assert result["horizons"] == [5, 3, 1]
    assert result["counts"] == {"candidates": 33, "inner_folds": 6, "outer_folds": 3,
        "search_fits": 603, "baseline_logical_fits": 9, "baseline_component_fits": 18,
        "total_logical_fits": 612, "total_component_fits": 621}
    assert result["schedule"] == [
        {"outer_year": "2022", "inner_years": ["2020", "2021"]},
        {"outer_year": "2023", "inner_years": ["2021", "2022"]},
        {"outer_year": "2024", "inner_years": ["2022", "2023"]}]
    specs = result["candidate_specs"]
    ids = [s["candidate_spec_id"] for s in specs]
    assert len(ids) == len(set(ids)) == 33 and ids == sorted(ids)
    for subset, width in (("H", 11), ("HM", 27), ("HMO", 40)):
        chosen = [s for s in specs if s["feature_recipe"]["subset"] == subset]
        assert len(chosen) == 11
        for spec in chosen:
            assert set(spec) == {"family", "feature_recipe", "parameters", "candidate_spec_id"}
            assert spec["feature_recipe"] == recipe(subset)
            fields = spec["feature_recipe"]["dictionary"]
            assert len({f["name"] for f in fields}) == len(fields) == width
            assert spec["feature_recipe"]["quote_indicator"] is (subset == "HMO")
            assert spec["candidate_spec_id"] == content_hash({"request_id": result["request_id"],
                **{k: v for k, v in spec.items() if k != "candidate_spec_id"}})
        assert {s["parameters"]["alpha"] for s in chosen if s["family"] == "ridge"} == {0.1, 1, 10}
        assert {(s["parameters"]["alpha"], s["parameters"]["l1_ratio"])
                for s in chosen if s["family"] == "elasticnet"} == {
                    (0.01, 0.25), (0.01, 0.75), (0.1, 0.25), (0.1, 0.75), (1, 0.25), (1, 0.75)}
        assert {s["parameters"]["max_leaf_nodes"] for s in chosen if s["family"] == "hgb"} == {7, 15}


def test_counts_are_schedule_specific_not_universally_603():
    value = request()
    value.update(outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=102)
    value["comparison"].update(max_logical_fits=3, max_component_fits=6)
    result = compiler.compile_request(value)
    assert result["counts"] == {"candidates": 33, "inner_folds": 1, "outer_folds": 1,
        "search_fits": 102, "baseline_logical_fits": 3, "baseline_component_fits": 6,
        "total_logical_fits": 105, "total_component_fits": 108}
    value.update(max_candidates=34, max_fits=103)
    assert compiler.compile_request(value)["counts"] == result["counts"]


def test_canonical_aliases_key_order_replay_and_complete_detachment():
    original = request()
    saved = copy.deepcopy(original)
    first = compiler.compile_request(original)
    alias = dict(reversed(list(original.items())))
    alias["space"] = copy.deepcopy(original["space"])
    alias["space"]["ridge"]["alpha"] = [0.1, 1, 10]
    alias["space"]["hgb"]["max_leaf_nodes"] = [7.0, 15.0]
    assert compiler.compile_request(alias) == first
    assert original == saved
    first["request"]["limits"]["child"]["memory_bytes"] = 999
    first["schedule"][0]["inner_years"].append("2099")
    first["candidate_specs"][0]["feature_recipe"]["dictionary"][0]["name"] = "poison"
    assert compiler.compile_request(original) == compiler.compile_request(saved)
    assert original == saved
    assert json.dumps(compiler.compile_request(alias)) == json.dumps(compiler.compile_request(saved))


@pytest.mark.parametrize("path,value", [
    (("feature_snapshot_id",), "snap_" + "a" * 32),
    (("history_snapshot_id",), "snap_" + "b" * 32),
    (("eligibility_policy_ref", "plan_id"), "sha256:" + "c" * 64),
    (("inner_years", "2022"), ["2019", "2021"]),
    (("max_candidates",), 34), (("max_fits",), 604),
    (("comparison", "max_logical_fits"), 10), (("comparison", "max_component_fits"), 19),
    (("limits", "child", "cpu_seconds"), 20), (("limits", "parent", "memory_bytes"), 20),
    (("limits", "build", "wall_seconds"), 20), (("limits", "job", "wall_seconds"), 20)])
def test_changed_declarations_change_request_and_every_spec_id(path, value):
    document = request()
    before = compiler.compile_request(document)
    set_value(document, path, value)
    after = compiler.compile_request(document)
    assert before["request_id"] != after["request_id"]
    assert {s["candidate_spec_id"] for s in before["candidate_specs"]}.isdisjoint(
        s["candidate_spec_id"] for s in after["candidate_specs"])


@pytest.mark.parametrize("path,value,message", [
    (("schema_version",), "magnitude_search_request.v1", "request requires its closed v2 schema"),
    (("feature_snapshot_id",), "latest", "snapshot IDs require native immutable ID syntax"),
    (("history_snapshot_id",), "snap_" + "A" * 32, "snapshot IDs require native immutable ID syntax"),
    (("feature_dictionary_hash",), "sha256:" + "4" * 64, "feature dictionary differs from the supported panel dictionary"),
    (("eligibility_policy_ref", "plan_id"), "sha256:bad", "eligibility reference requires the declared 30-day policy"),
    (("eligibility_policy_ref", "schema_ref"), "group_eligibility_plan.v2", "eligibility reference requires the declared 30-day policy"),
    (("eligibility_policy_ref", "policy_id"), "sha256:" + "5" * 64, "eligibility reference requires the declared 30-day policy"),
    (("feature_sets",), ["HM", "H", "HMO"], "requires exactly ordered H/HM/HMO feature sets"),
    (("feature_sets",), ["H", "H", "HMO"], "requires exactly ordered H/HM/HMO feature sets"),
    (("feature_sets",), ["H", "HM", "unknown_column"], "requires exactly ordered H/HM/HMO feature sets"),
    (("space",), {"ridge": {}}, "requires the closed three-family grid"),
    (("space", "hgb"), {"unknown": [7, 15]}, "grid axes differ from the closed policy"),
    (("space", "ridge", "alpha"), [0.1, True, 10], "grid values differ from the closed policy"),
    (("space", "elasticnet", "alpha"), [1, 0.1, 0.01], "grid values differ from the closed policy"),
    (("space", "elasticnet", "l1_ratio"), [0.25, 0.25], "grid values differ from the closed policy"),
    (("space", "hgb", "max_leaf_nodes"), [7, 15, 31], "grid values differ from the closed policy"),
    (("comparison", "reference"), "latest_model", "comparison requires the fixed causal legacy and history-mean references"),
    (("comparison", "sanity_reference"), "median", "comparison requires the fixed causal legacy and history-mean references")])
def test_closed_semantics_refuse_before_expansion(no_expansion, path, value, message):
    document = request()
    set_value(document, path, value)
    refuses(document, "stage-2 " + message)


@pytest.mark.parametrize("key", list(request()))
def test_every_top_level_field_required(no_expansion, key):
    document = request()
    del document[key]
    refuses(document, "stage-2 request requires its closed v2 schema")


def test_extra_keys_and_non30_day_policy_refuse(no_expansion):
    document = request()
    document["columns"] = ["unknown"]
    refuses(document, "stage-2 request requires its closed v2 schema")
    document = request()
    document["eligibility_policy_ref"]["policy_id"] = eligibility_policy(
        version="grouped_ticker_embargo.v1", embargo_days=29)["policy_id"]
    refuses(document, "stage-2 eligibility reference requires the declared 30-day policy")


@pytest.mark.parametrize("path,message", [
    (("eligibility_policy_ref",), "stage-2 eligibility reference requires the declared 30-day policy"),
    (("comparison",), "stage-2 comparison requires the fixed causal legacy and history-mean references"),
    (("limits",), "stage-2 requires explicit child, parent, build and job limits"),
    (("limits", "child"), "search execution requires explicit positive integer resource limits"),
    (("limits", "parent"), "stage-2 parent/build/job limits require explicit bounded positive integers"),
    (("limits", "build"), "stage-2 parent/build/job limits require explicit bounded positive integers"),
    (("limits", "job"), "stage-2 parent/build/job limits require explicit bounded positive integers"),
    (("inner_years",), "stage-2 requires ordered outer years and exact inner-year keys")])
@pytest.mark.parametrize("mutation", ["extra", "missing", "scalar"])
def test_nested_shapes_are_closed_before_expansion(no_expansion, path, message, mutation):
    document = request()
    nested = document
    for key in path:
        nested = nested[key]
    if mutation == "extra":
        nested["unexpected"] = 1
    elif mutation == "missing":
        del nested[next(iter(nested))]
    else:
        set_value(document, path, None)
    refuses(document, message)


@pytest.mark.parametrize("value", [[], ["2024", "2022"], ["2022", "2022"], [2022],
    ["２０２２"], ["0001"], ["9999"], ["22"], None])
def test_bad_outer_years_refuse(no_expansion, value):
    document = request()
    document["outer_years"] = value
    refuses(document, "stage-2 requires ordered outer years and exact inner-year keys")


@pytest.mark.parametrize("value", [[], ["2021", "2020"], ["2020", "2020"], ["2022"], ["2023"], [2020]])
def test_bad_inner_years_refuse(no_expansion, value):
    document = request()
    document["inner_years"]["2022"] = value
    refuses(document, "stage-2 inner years must be ordered and strictly earlier than their outer year")


@pytest.mark.parametrize("key,minimum", [("max_candidates", 33), ("max_fits", 603)])
@pytest.mark.parametrize("bad", [0, -1, True, None, "33", "below", "float"])
def test_search_caps_refuse_before_expansion(no_expansion, key, minimum, bad):
    document = request()
    document[key] = minimum - 1 if bad == "below" else float(minimum) if bad == "float" else bad
    refuses(document, "stage-2 candidate or search-fit cap is insufficient")


@pytest.mark.parametrize("key,minimum", [("max_logical_fits", 9), ("max_component_fits", 18)])
@pytest.mark.parametrize("bad", [True, 0, "below", "float"])
def test_comparison_caps_refuse_before_expansion(no_expansion, key, minimum, bad):
    document = request()
    document["comparison"][key] = minimum - 1 if bad == "below" else float(minimum) if bad == "float" else bad
    refuses(document, "stage-2 comparison fit cap is insufficient")


@pytest.mark.parametrize("group", ["child", "parent", "build", "job"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.0, None, sys.maxsize, "1"])
def test_resource_limit_types_and_bounds(no_expansion, group, bad):
    document = request()
    document["limits"][group]["wall_seconds"] = bad
    message = ("search execution requires explicit positive integer resource limits" if group == "child"
        else "stage-2 parent/build/job limits require explicit bounded positive integers")
    refuses(document, message)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), Decimal("1"), (1,), {1: "bad"}])
def test_non_json_and_nonfinite_refuse_before_normalization(no_expansion, value):
    document = request()
    document["extra"] = value
    refuses(document, "stage-2 request requires plain finite JSON")


def test_container_subclasses_and_cycles_refuse(no_expansion):
    class Array(list):
        pass
    class Object(dict):
        pass
    for value in (Array([1]), Object(a=1)):
        document = request()
        document["extra"] = value
        refuses(document, "stage-2 request requires plain finite JSON")
    document = request()
    document["cycle"] = document
    refuses(document, "stage-2 request requires plain finite JSON")


def test_valid_preview_has_no_io_fit_or_native_admission(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    for owner, name in ((Repository, "resolve"), (Repository, "scan"), (ArtifactStore, "publish_bytes"),
                        (ArtifactStore, "read_verified"), (sqlite3, "connect"), (subprocess, "Popen"),
                        (HistGradientBoostingRegressor, "__init__"), (ElasticNet, "__init__"), (Ridge, "__init__"),
                        (HistGradientBoostingRegressor, "fit"), (ElasticNet, "fit"), (Ridge, "fit")):
        monkeypatch.setattr(owner, name, fail_if_called)
    assert compiler.compile_request(request())["counts"]["search_fits"] == 603


def test_public_child_validator_receives_exact_original_scalar_types(monkeypatch):
    seen, original = [], compiler.validate_execution_limits
    def observe(limits):
        seen.append(copy.deepcopy(limits))
        original(limits)
    monkeypatch.setattr(compiler, "validate_execution_limits", observe)
    document = request()
    compiler.compile_request(document)
    assert seen == [document["limits"]["child"]]
    document["limits"]["child"]["memory_bytes"] = 1.0
    refuses(document, "search execution requires explicit positive integer resource limits")
    assert type(seen[-1]["memory_bytes"]) is float


def test_v1_planner_does_not_accept_the_new_request():
    from experiments.magnitude_search_plan import _request
    with pytest.raises(OpsError) as caught:
        _request(request())
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "search requires explicit valid axes, inner years and count caps"
