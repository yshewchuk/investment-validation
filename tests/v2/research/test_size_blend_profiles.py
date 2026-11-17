import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import size_blend_profiles as profiles
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.test_magnitude_search_request_v14_blends import _blend_request

_NAMES = ["size_b1_v1", "size_b2_v1"]


def _specs(**changes):
    request = _blend_request()
    request["blends"] = list(_NAMES)
    request.update(changes)
    return compile_request(request)["candidate_specs"]


def _records(specs):
    resolved = profiles.resolve_blends(_NAMES, specs)
    candidate_id_by_spec = {s["candidate_spec_id"]: "cand:" + s["candidate_spec_id"] for s in specs}
    return profiles.blend_records(resolved, candidate_id_by_spec)


def test_validate_accepts_sorted_unique_known_lists():
    assert profiles.validate_blends(["size_b1_v1", "size_b2_v1"]) == ["size_b1_v1", "size_b2_v1"]
    assert profiles.validate_blends(["size_b2_v1"]) == ["size_b2_v1"]


@pytest.mark.parametrize("value", [
    [], ["size_b2_v1", "size_b1_v1"], ["size_b1_v1", "size_b1_v1"], ["size_b3_v1"],
    "size_b1_v1", None])
def test_validate_refuses(value):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        profiles.validate_blends(value)


def test_resolve_matches_one_base_candidate_per_component():
    specs = _specs()
    resolved = profiles.resolve_blends(_NAMES, specs)
    assert len(resolved["size_b1_v1"]) == 4
    assert len(resolved["size_b2_v1"]) == 3
    ids = {s["candidate_spec_id"] for s in specs}
    assert set(resolved["size_b1_v1"]) <= ids and set(resolved["size_b2_v1"]) <= ids


def test_resolve_never_matches_a_log_target_twin():
    request = _blend_request()
    request["blends"] = list(_NAMES)
    request["space"]["log_target"] = {"elasticnet": {"alpha": [0.03], "l1_ratio": [0.75],
                                                     "spline_knots": [0, 6]}}
    specs = compile_request(request)["candidate_specs"]
    assert any("target" in s["parameters"] for s in specs)
    by_id = {s["candidate_spec_id"]: s for s in specs}
    resolved = profiles.resolve_blends(_NAMES, specs)
    for ids in resolved.values():
        assert all("target" not in by_id[spec_id]["parameters"] for spec_id in ids)


def test_blend_id_is_deterministic_and_differs_by_profile():
    specs = _specs()
    first, second = _records(specs), _records(specs)
    assert first == second
    assert [r["name"] for r in first] == _NAMES
    assert len({r["blend_id"] for r in first}) == 2
    assert all(r["inner_scores_optimistic"] is True for r in first)
    assert all(r["aggregation"] == "equal_mean" for r in first)


def test_blend_id_matches_hand_built_oracle_for_size_b2_v1():
    resolved = {"size_b2_v1": ["s1", "s2", "s3"]}
    candidate_id_by_spec = {"s1": "c3", "s2": "c1", "s3": "c2"}
    records = profiles.blend_records(resolved, candidate_id_by_spec)
    assert len(records) == 1
    record = records[0]
    assert record["component_candidate_ids"] == ["c1", "c2", "c3"]
    expected_profile = [
        {"family": "hgb", "parameters": {"learning_rate": 0.03, "l2_regularization": 10.0,
                                         "loss": "absolute_error", "max_iter": 3000,
                                         "max_leaf_nodes": 15, "min_samples_leaf": 300}},
        {"family": "hgb", "parameters": {"learning_rate": 0.03, "l2_regularization": 10.0,
                                         "loss": "squared_error", "max_iter": 3000,
                                         "max_leaf_nodes": 15, "min_samples_leaf": 1000}},
        {"family": "mlp", "parameters": {"profile": "chrono15_relu64x32_l2_1"}},
    ]
    assert record["blend_id"] == content_hash({"profile": expected_profile,
                                               "aggregation": "equal_mean",
                                               "component_candidate_ids": ["c1", "c2", "c3"]})


def test_blend_id_changes_when_one_component_id_changes():
    resolved = {"size_b2_v1": ["s1", "s2", "s3"]}
    original = profiles.blend_records(resolved, {"s1": "c3", "s2": "c1", "s3": "c2"})[0]
    changed = profiles.blend_records(resolved, {"s1": "c3", "s2": "c1", "s3": "c4"})[0]
    assert changed["component_candidate_ids"] != original["component_candidate_ids"]
    assert changed["blend_id"] != original["blend_id"]


def test_distinct_component_counts():
    assert profiles.distinct_component_count(["size_b1_v1", "size_b2_v1"]) == 4
    assert profiles.distinct_component_count(["size_b2_v1"]) == 3
