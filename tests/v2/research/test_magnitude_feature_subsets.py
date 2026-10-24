"""Independent static-neighborhood, identity and pre-expansion bound oracles."""
import builtins
import copy
import hashlib
import itertools
import json
import sqlite3
from decimal import Decimal

import pytest

from engine.v2.foundation import content_hash
from engine.v2.ops.errors import OpsError
from experiments import magnitude_feature_subsets as planner


def request(blocks=4):
    dictionary = ["history", "core_tail"] + [f"f{index}" for index in range(blocks)]
    return {"schema_version": "magnitude_feature_subsets_request.v1",
        "dictionary": dictionary, "column_list_id": content_hash(dictionary),
        "core": dictionary[:2], "optional_blocks": [
            {"name": f"block{index:02}", "columns": [f"f{index}"]} for index in range(blocks)],
        "model_configuration_ids": ["sha256:" + f"{index:064x}" for index in range(11)],
        "horizons": [5, 3, 1], "outer_years": ["2022", "2023", "2024"],
        "inner_years": {"2022": ["2020", "2021"], "2023": ["2021", "2022"],
                        "2024": ["2022", "2023"]},
        "max_subsets": 130, "max_candidates": 1430, "max_fits": 25749}


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected hashing, expansion, source access or write")


def refuses(document):
    with pytest.raises(OpsError) as caught:
        planner.plan_subsets(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False


@pytest.mark.parametrize("blocks", range(7))
def test_independent_combinations_oracle_dedup_core_order_and_identity(blocks):
    document = request(blocks)
    result = planner.plan_subsets(document)
    expected = set()
    for size in {0, 1, blocks - 1, blocks}:
        if 0 <= size <= blocks:
            for selected in itertools.combinations(range(blocks), size):
                expected.add(tuple(document["core"] + [f"f{i}" for i in selected]))
    actual = {tuple(subset["columns"]) for subset in result["subsets"]}
    assert actual == expected
    assert len(actual) == len(result["subsets"])
    assert result["schema_version"] == "magnitude_feature_subset_plan.v1"
    assert result["plan_id"] == content_hash(document)
    assert result["request"] == document and result["horizons"] == [5, 3, 1]
    assert result["counts"] == {"subsets": len(expected), "model_configurations": 11,
        "candidates": 11 * len(expected), "inner_folds": 6, "outer_folds": 3,
        "search_fits": 198 * len(expected) + 9}
    ids = [subset["subset_id"] for subset in result["subsets"]]
    assert ids == sorted(set(ids))
    for subset in result["subsets"]:
        assert set(subset) == {"schema_version", "column_list_id", "columns", "subset_id"}
        assert subset["schema_version"] == "magnitude_feature_subset.v1"
        assert subset["column_list_id"] == document["column_list_id"]
        assert subset["subset_id"] == content_hash({key: value for key, value in subset.items()
                                                   if key != "subset_id"})


def test_exact_example_and_schedule_specific_counts():
    document = request()
    document.update(max_subsets=10, max_candidates=110, max_fits=1989)
    result = planner.plan_subsets(document)
    assert result["counts"] == {"subsets": 10, "model_configurations": 11,
        "candidates": 110, "inner_folds": 6, "outer_folds": 3, "search_fits": 1989}
    document.update(outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=333)
    assert planner.plan_subsets(document)["counts"]["search_fits"] == 333
    document["model_configuration_ids"] = document["model_configuration_ids"][:1]
    document.update(max_candidates=10, max_fits=33)
    assert planner.plan_subsets(document)["counts"]["search_fits"] == 33


def test_interleaved_multicolumn_blocks_preserve_dictionary_order_not_block_order():
    document = request(2)
    document["dictionary"] = ["f0", "history", "a", "f1", "core_tail", "b"]
    document["column_list_id"] = content_hash(document["dictionary"])
    document["optional_blocks"][0]["columns"] = ["f0", "b"]
    document["optional_blocks"][1]["columns"] = ["a", "f1"]
    assert {tuple(subset["columns"]) for subset in planner.plan_subsets(document)["subsets"]} == {
        ("history", "core_tail"), ("f0", "history", "core_tail", "b"),
        ("history", "a", "f1", "core_tail"), tuple(document["dictionary"])}


def test_bounded_maximum_blocks_and_columns_are_linear():
    document = request(64)
    document["dictionary"] += [f"extra{i}" for i in range(446)]
    document["optional_blocks"][-1]["columns"] += document["dictionary"][66:]
    document["column_list_id"] = content_hash(document["dictionary"])
    result = planner.plan_subsets(document)
    assert len(result["subsets"]) == 130
    assert result["counts"]["candidates"] == 1430
    assert result["counts"]["search_fits"] == 25749
    assert max(len(subset["columns"]) for subset in result["subsets"]) == 512


def test_no_io_and_deterministic_detached_replay(monkeypatch):
    document = request()
    saved = copy.deepcopy(document)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    first = planner.plan_subsets(document)
    second = planner.plan_subsets(dict(reversed(list(document.items()))))
    assert json.dumps(first) == json.dumps(second)
    first["request"]["dictionary"].append("poison")
    first["request"]["inner_years"]["2024"].append("2025")
    first["subsets"][0]["columns"].append("poison")
    first["horizons"].append(7)
    assert document == saved
    assert planner.plan_subsets(document) == second


@pytest.mark.parametrize("field,value", [
    ("max_subsets", 131), ("max_candidates", 1431), ("max_fits", 25750),
    ("outer_years", ["2022", "2023", "2025"]),
    ("model_configuration_ids", ["sha256:" + "f" * 64]),
])
def test_changed_declarations_change_plan_not_intrinsic_subset_ids(field, value):
    document = request()
    before = planner.plan_subsets(document)
    document[field] = value
    if field == "outer_years":
        document["inner_years"]["2025"] = document["inner_years"].pop("2024")
    after = planner.plan_subsets(document)
    assert before["plan_id"] != after["plan_id"]
    assert before["subsets"] == after["subsets"]


def test_changed_dictionary_order_changes_every_subset_identity():
    document = request()
    before = planner.plan_subsets(document)
    document["dictionary"][:2] = reversed(document["dictionary"][:2])
    document["core"].reverse()
    document["column_list_id"] = content_hash(document["dictionary"])
    after = planner.plan_subsets(document)
    assert before["plan_id"] != after["plan_id"]
    assert {s["subset_id"] for s in before["subsets"]}.isdisjoint(
        s["subset_id"] for s in after["subsets"])


@pytest.mark.parametrize("field,value", [
    ("schema_version", "magnitude_search_request.v2"), ("dictionary", []),
    ("dictionary", ["a"] * 513), ("dictionary", ["a", "a"]),
    ("dictionary", ["A"]), ("dictionary", ["a" * 65]), ("dictionary", [1]),
    ("dictionary", [Decimal("1")]), ("dictionary", ("a",)),
    ("column_list_id", "sha256:" + "A" * 64), ("column_list_id", "x" * 10000),
    ("core", []), ("core", ["history", "history"]), ("core", ["unknown"]),
    ("core", ["core_tail", "history"]), ("core", [True]),
    ("optional_blocks", [None] * 65), ("optional_blocks", {}),
    ("model_configuration_ids", []), ("model_configuration_ids", ["x"] * 257),
    ("model_configuration_ids", [True]), ("model_configuration_ids", ["x"]),
    ("horizons", [5.0, 3, 1]), ("horizons", [5, 3, True]), ("horizons", [1, 3, 5]),
    ("horizons", (5, 3, 1)), ("outer_years", []), ("outer_years", ["2024"] * 65),
    ("outer_years", ["2023", "2022", "2024"]), ("outer_years", [2022, 2023, 2024]),
    ("outer_years", ["0001"]), ("outer_years", ["9999"]),
    ("inner_years", {}), ("inner_years", []),
    ("max_subsets", 0), ("max_candidates", True), ("max_fits", 1989.0),
    ("max_fits", 2**31), ("max_subsets", 9), ("max_candidates", 109), ("max_fits", 1988),
])
def test_bad_shapes_bounds_and_caps_refuse_before_hash_or_materialization(monkeypatch, field, value):
    document = request()
    document[field] = value
    monkeypatch.setattr(planner, "content_hash", forbidden)
    monkeypatch.setattr(planner, "_subsets", forbidden)
    refuses(document)


@pytest.mark.parametrize("change", [
    lambda r: r.update(extra=1), lambda r: r.pop("max_fits"),
    lambda r: r["optional_blocks"][0].update(extra=1),
    lambda r: r["optional_blocks"][0].update(columns=[]),
    lambda r: r["optional_blocks"][0].update(columns=["history"]),
    lambda r: r["optional_blocks"][0].update(columns=["f1"]),
    lambda r: r["optional_blocks"][0].update(columns=["unknown"]),
    lambda r: r["optional_blocks"][0].update(columns=["f0", "f0"]),
    lambda r: r["optional_blocks"][0].update(name="block01"),
    lambda r: r["optional_blocks"].reverse(), lambda r: r["optional_blocks"].pop(),
    lambda r: r["model_configuration_ids"].reverse(),
    lambda r: r["model_configuration_ids"].append(r["model_configuration_ids"][0]),
    lambda r: r["inner_years"].update({"2024": []}),
    lambda r: r["inner_years"].update({"2024": ["2024"]}),
    lambda r: r["inner_years"].update({"2024": ["2025"]}),
    lambda r: r["inner_years"].update({"2024": ["2022", "2022"]}),
    lambda r: r["inner_years"].update({"2024": ["2023", "2022"]}),
    lambda r: r["inner_years"].update({"2024": ["2020"] * 65}),
    lambda r: r["inner_years"].update({"2024": ["2e03"]}),
])
def test_invalid_nested_declarations_before_expansion(monkeypatch, change):
    document = request()
    change(document)
    monkeypatch.setattr(planner, "content_hash", forbidden)
    monkeypatch.setattr(planner, "_subsets", forbidden)
    refuses(document)


def test_wrong_name_hash_refuses_before_expansion(monkeypatch):
    document = request()
    document["column_list_id"] = "sha256:" + "f" * 64
    monkeypatch.setattr(planner, "_subsets", forbidden)
    refuses(document)


def test_literal_identity_oracle_without_native_hashing():
    document = request(0)
    result = planner.plan_subsets(document)
    # This fixture contains ASCII strings and integers only: sorted compact JSON
    # is canonical independently of the production canonical-JSON implementation.
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    expected = "sha256:ec68ba7522b4bb3638fd9a47257ade1ef3abb40b21563a1874bc22482f5a0ec2"
    assert "sha256:" + hashlib.sha256(encoded).hexdigest() == expected
    assert result["plan_id"] == expected
    assert result["subsets"][0]["subset_id"] == (
        "sha256:23893a836e07799987bd8776ed7423c87e9405d1570046702495acc8c350944b")


def test_nonplain_subclasses_cycles_and_nonfinite_values_refuse_before_hash(monkeypatch):
    class Subdict(dict):
        pass

    class Sublist(list):
        pass

    documents = [Subdict(request())]
    for field in ("dictionary", "core", "optional_blocks", "model_configuration_ids",
                  "horizons", "outer_years"):
        document = request()
        document[field] = Sublist(document[field])
        documents.append(document)
    for value in (float("nan"), float("inf"), -float("inf")):
        document = request()
        document["max_fits"] = value
        documents.append(document)
    for field in ("dictionary", "optional_blocks"):
        document = request()
        document[field] = [document]
        documents.append(document)
    monkeypatch.setattr(planner, "content_hash", forbidden)
    monkeypatch.setattr(planner, "_subsets", forbidden)
    for document in documents:
        refuses(document)


@pytest.mark.parametrize("value", [None, [], "request", True, {"x": "y"}])
def test_nonrequests_refuse(value):
    refuses(value)
