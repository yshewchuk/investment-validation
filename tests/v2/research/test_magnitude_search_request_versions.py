"""Public-API replay/refusal parity and independent version-specific identities."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from engine.v2.foundation import canonical_json
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.research import magnitude_panel_table as v1
from engine.v2.research import magnitude_panel_v2_table as v2
from experiments import magnitude_search_request_leads as leads
from experiments import magnitude_search_request_versions as versions
from tests.v2.research import test_magnitude_search_request as legacy
from tests.v2.research.magnitude_panel_policy_support import forbid_policy_io
from tests.v2.research.test_magnitude_panel_v2_table import _expected_dictionary

V1_COMPILE, V1_REQUEST = legacy.compiler.compile_request, legacy.request
V2_HASH = "sha256:21a88956f12177613742de95834c7f310216825e01da1f2d813dba3589991505"
no_expansion = legacy.no_expansion


def _hash(value):
    return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _request():
    return {**V1_REQUEST(), "feature_dictionary_hash": V2_HASH}


@pytest.mark.parametrize("bad_horizons", [[], None, [1, 1], [2], [True], [1.0], [3, 1, 5]])
def test_v2_lead_compiler_refuses_invalid_horizons(bad_horizons):
    with pytest.raises(OpsError) as actual:
        leads.compile_request(_request(), horizons=bad_horizons)
    assert actual.value.problem.code == "INVALID_EXPERIMENT_SPEC"
    assert actual.value.problem.message == "stage-2 horizons must be a nonempty unique subset of integer leads 1, 3, 5"


@pytest.mark.parametrize("horizons,expected", [([1], [1]), ([1, 5], [5, 1]), ((5, 3, 1), [5, 3, 1])])
def test_v2_lead_compiler_accepts_valid_horizons_in_order(horizons, expected):
    actual = versions.compile_request(_request(), horizons=horizons)
    assert actual["horizons"] == expected


def _parity(document):
    # Only the reference validator sees the surrogate; compare the whole Problem.
    reference = ({**document, "feature_dictionary_hash": v1.recipe()["dictionary_hash"]}
        if type(document) is dict and type(document.get("feature_dictionary_hash")) is str
        and document["feature_dictionary_hash"] == V2_HASH else document)
    try:
        expected = V1_COMPILE(reference)
    except OpsError as error:
        with pytest.raises(OpsError) as actual:
            versions.compile_request(document)
        assert actual.value.problem == error.problem
        raise
    actual = versions.compile_request(document)
    assert json.dumps(actual) == json.dumps(expected)
    return actual


class TestV1Replay:
    """Run the existing fixture/refusal matrix through both public entrypoints."""
    @pytest.fixture(autouse=True)
    def parity(self, monkeypatch):
        monkeypatch.setattr(legacy.compiler, "compile_request", _parity)

    test_grid = staticmethod(legacy.test_exact_closed_grid_and_independent_schedule_counts)
    test_small = staticmethod(legacy.test_counts_are_schedule_specific_not_universally_603)
    test_aliases = staticmethod(legacy.test_canonical_aliases_key_order_replay_and_complete_detachment)
    test_changes = staticmethod(legacy.test_changed_declarations_change_request_and_every_spec_id)
    test_semantics = staticmethod(legacy.test_closed_semantics_refuse_before_expansion)
    test_required = staticmethod(legacy.test_every_top_level_field_required)
    test_extra = staticmethod(legacy.test_extra_keys_and_non30_day_policy_refuse)
    test_shapes = staticmethod(legacy.test_nested_shapes_are_closed_before_expansion)
    test_outer = staticmethod(legacy.test_bad_outer_years_refuse)
    test_inner = staticmethod(legacy.test_bad_inner_years_refuse)
    test_search_caps = staticmethod(legacy.test_search_caps_refuse_before_expansion)
    test_comparison_caps = staticmethod(legacy.test_comparison_caps_refuse_before_expansion)
    test_resources = staticmethod(legacy.test_resource_limit_types_and_bounds)
    test_nonjson = staticmethod(legacy.test_non_json_and_nonfinite_refuse_before_normalization)
    test_cycles = staticmethod(legacy.test_container_subclasses_and_cycles_refuse)
    test_no_io = staticmethod(legacy.test_valid_preview_has_no_io_fit_or_native_admission)
    test_native_refusal = staticmethod(legacy.test_v1_planner_does_not_accept_the_new_request)


class TestV2Refusals:
    @pytest.fixture(autouse=True)
    def parity(self, monkeypatch):
        monkeypatch.setattr(legacy, "request", _request)
        monkeypatch.setattr(legacy.compiler, "compile_request", _parity)
        monkeypatch.setattr(versions, "v2_recipe", legacy.fail_if_called)

    test_semantics = staticmethod(legacy.test_closed_semantics_refuse_before_expansion)
    test_required = staticmethod(legacy.test_every_top_level_field_required)
    test_extra = staticmethod(legacy.test_extra_keys_and_non30_day_policy_refuse)
    test_shapes = staticmethod(legacy.test_nested_shapes_are_closed_before_expansion)
    test_outer = staticmethod(legacy.test_bad_outer_years_refuse)
    test_inner = staticmethod(legacy.test_bad_inner_years_refuse)
    test_search_caps = staticmethod(legacy.test_search_caps_refuse_before_expansion)
    test_comparison_caps = staticmethod(legacy.test_comparison_caps_refuse_before_expansion)
    test_resources = staticmethod(legacy.test_resource_limit_types_and_bounds)
    test_nonjson = staticmethod(legacy.test_non_json_and_nonfinite_refuse_before_normalization)
    test_cycles = staticmethod(legacy.test_container_subclasses_and_cycles_refuse)


@pytest.mark.parametrize("schema", ["regression_search_request.v1", "magnitude_search_request.v1"])
@pytest.mark.parametrize("horizons", [(1,), None])
def test_legacy_schemas_refuse_external_horizon_selector(schema, horizons):
    with pytest.raises(OpsError) as actual:
        versions.compile_request({"schema_version": schema}, horizons=horizons)
    assert actual.value.problem.code == "INVALID_EXPERIMENT_SPEC"
    assert actual.value.problem.message == "stage-2 legacy requests do not accept a horizon selector"


@pytest.mark.parametrize("schema", ["regression_search_request.v1", "magnitude_search_request.v1"])
@pytest.mark.parametrize("horizons", [(5, 3, 1), [5, 3, 1]])
def test_legacy_schemas_preserve_default_dispatch(schema, horizons, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(versions, "compile_v1_request", lambda request: sentinel)
    assert versions.compile_request({"schema_version": schema}, horizons=horizons) is sentinel


def test_pinned_complete_v1_output_and_current_source_bytes(monkeypatch):
    def no_whole_file_read(*_, **__):
        pytest.fail("source digest must not materialize a whole file")
    monkeypatch.setattr(Path, "read_bytes", no_whole_file_read)
    expected = "sha256:31b7f493b3f2742e4a948767a352f3fa2bbe408b01898fcb9e0371c4b598fac0"
    assert _hash(V1_COMPILE(V1_REQUEST())) == expected
    assert _hash(versions.compile_request(V1_REQUEST())) == expected
    root = Path(__file__).resolve().parents[3]
    for path, digest in (
        ("experiments/magnitude_search_request.py", "5525b7511b2e0a28ba7a2393b37252bd5b45fc4ca7b77b53f37bbf7156d1f64b"),
        ("engine/v2/research/magnitude_panel_table.py", "687d4d5341dd4e43879aee9d122507b8289150d7e73f1ccc527299182fe5aad4"),
        ("engine/v2/research/magnitude_panel_v2_table.py", "c753255acba0af3872f6a7bf1cbbdc59eb3b1097a2a06d89f6517d4c34111ce4")):
        with (root / path).open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == digest


def test_v2_golden_recipes_columns_counts_and_every_identity():
    document = _request()
    actual = versions.compile_request(document)
    surrogate = V1_COMPILE(V1_REQUEST())
    assert actual["request"] == document and actual["request_id"] == _hash(document)
    for key in ("schema_version", "horizons", "schedule", "counts"):
        assert actual[key] == surrogate[key]
    assert actual["counts"] == {"candidates": 33, "inner_folds": 6, "outer_folds": 3,
        "search_fits": 603, "baseline_logical_fits": 9, "baseline_component_fits": 18,
        "total_logical_fits": 612, "total_component_fits": 621}
    recipes = {"H": "91bfbb589cb732b09d1ef315a2e3c5431d93f5cc901883c431435fd1a026cf46",
        "HM": "bd3fe95c79f492dc7fc0af5294701dc4e57854f7596747cded43cdebb4a311ee",
        "HMO": "59c80937103e378d6d3ea6bfe976bc2c2b83d4a159db44a2414f236405028ed0"}
    ids = [spec["candidate_spec_id"] for spec in actual["candidate_specs"]]
    assert len(set(ids)) == 33 and ids == sorted(ids)
    assert actual["request_id"] != surrogate["request_id"]
    assert set(ids).isdisjoint(spec["candidate_spec_id"] for spec in surrogate["candidate_specs"])
    for subset, width in (("H", 11), ("HM", 28), ("HMO", 42)):
        chosen = [spec for spec in actual["candidate_specs"] if spec["feature_recipe"]["subset"] == subset]
        assert len(chosen) == 11
        assert sorted((s["family"], json.dumps(s["parameters"], sort_keys=True)) for s in chosen) == sorted(
            (s["family"], json.dumps(s["parameters"], sort_keys=True)) for s in surrogate["candidate_specs"]
            if s["feature_recipe"]["subset"] == subset)
        for spec in chosen:
            recipe = spec["feature_recipe"]
            assert _hash(recipe) == "sha256:" + recipes[subset]
            assert recipe["dictionary"] == _expected_dictionary(subset)
            names = [field["name"] for field in recipe["dictionary"]]
            assert len(names) == width and "has_implied_quote" not in names
            assert recipe["model_columns"] == names + (["has_implied_quote"] if subset == "HMO" else [])
            assert recipe["dictionary_hash"] == _hash(recipe["dictionary"])
            assert recipe != v1.recipe(subset)
            assert spec["candidate_spec_id"] == _hash({"request_id": actual["request_id"],
                **{key: value for key, value in spec.items() if key != "candidate_spec_id"}})


def test_v2_normalization_smaller_schedule_and_detachment():
    original = _request()
    saved = copy.deepcopy(original)
    expected = versions.compile_request(original)
    baseline = copy.deepcopy(expected)
    aliases = dict(reversed(list(copy.deepcopy(original).items())))
    aliases["space"] = {family: dict(reversed(list(axes.items())))
        for family, axes in reversed(list(aliases["space"].items()))}
    aliases["space"]["ridge"]["alpha"] = [0.1, 1, 10]
    aliases["space"]["elasticnet"]["alpha"] = [0.01, 0.1, 1]
    aliases["space"]["hgb"]["max_leaf_nodes"] = [7.0, 15.0]
    assert versions.compile_request(aliases) == expected and original == saved
    expected["request"]["limits"]["child"]["memory_bytes"] = 999
    expected["schedule"][0]["inner_years"].append("2099")
    recipes = [spec["feature_recipe"] for spec in expected["candidate_specs"]]
    recipes[0]["dictionary"][0]["source_dependencies"][0]["policy"]["rebuild"] = "poison"
    assert all(recipe["dictionary"][0]["source_dependencies"][0]["policy"]["rebuild"] == "forbidden"
        for recipe in recipes[1:])
    recipes[0]["model_columns"].clear()
    quote_recipes = [recipe for recipe in recipes if recipe["subset"] == "HMO"]
    quote_recipes[0]["quote_indicator"]["false_statuses"].clear()
    assert all(recipe["quote_indicator"]["false_statuses"] for recipe in quote_recipes[1:])
    assert versions.compile_request(original) == baseline
    assert original == saved
    detached = versions.compile_request(original)
    original["space"]["ridge"]["alpha"][0] = 999
    assert detached == baseline
    original = copy.deepcopy(saved)
    original.update(outer_years=["2024"], inner_years={"2024": ["2023"]}, max_fits=102)
    original["comparison"].update(max_logical_fits=3, max_component_fits=6)
    assert versions.compile_request(original)["counts"] == {"candidates": 33, "inner_folds": 1,
        "outer_folds": 1, "search_fits": 102, "baseline_logical_fits": 3,
        "baseline_component_fits": 6, "total_logical_fits": 105, "total_component_fits": 108}


@pytest.mark.parametrize("path,value", [(("feature_snapshot_id",), "snap_" + "a" * 32),
    (("history_snapshot_id",), "snap_" + "b" * 32),
    (("eligibility_policy_ref", "plan_id"), "sha256:" + "c" * 64),
    (("inner_years", "2022"), ["2019", "2021"]), (("limits", "job", "wall_seconds"), 9),
    (("max_candidates",), 34), (("max_fits",), 604),
    (("comparison", "max_logical_fits"), 10), (("comparison", "max_component_fits"), 19),
    (("limits", "child", "cpu_seconds"), 20), (("limits", "parent", "memory_bytes"), 20),
    (("limits", "build", "wall_seconds"), 20)])
def test_v2_changed_declarations_change_all_ids(path, value):
    before = versions.compile_request(_request())
    document = _request()
    legacy.set_value(document, path, value)
    after = versions.compile_request(document)
    assert before["request_id"] != after["request_id"] == _hash(after["request"])
    assert {spec["candidate_spec_id"] for spec in before["candidate_specs"]}.isdisjoint(
        spec["candidate_spec_id"] for spec in after["candidate_specs"])


@pytest.mark.parametrize("bad", [None, {}, [], 1, V2_HASH.upper(), "latest",
    v1.recipe("H")["dictionary_hash"], v1.recipe("HM")["dictionary_hash"],
    v2.recipe("H")["dictionary_hash"], v2.recipe("HM")["dictionary_hash"]])
def test_non_full_hashes_delegate_exactly(bad, monkeypatch):
    monkeypatch.setattr(versions, "v2_recipe", legacy.fail_if_called)
    with pytest.raises(OpsError):
        _parity({**_request(), "feature_dictionary_hash": bad})


def test_nonplain_roots_and_hash_subclasses_do_not_dispatch(monkeypatch):
    class Object(dict):
        pass
    class Text(str):
        pass
    monkeypatch.setattr(versions, "v2_recipe", legacy.fail_if_called)
    for document in (None, [], "{}", Object(_request()),
                     {**_request(), "feature_dictionary_hash": Text(V2_HASH)}):
        with pytest.raises(OpsError):
            _parity(document)


@pytest.mark.parametrize("factory", [V1_REQUEST, _request])
def test_exact_once_public_validation_and_no_io_or_fit(factory, monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    seen, validate = [], legacy.compiler.validate_execution_limits
    def observe(limits):
        seen.append(copy.deepcopy(limits))
        validate(limits)
    monkeypatch.setattr(legacy.compiler, "validate_execution_limits", observe)
    document = factory()
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        for owner, name in ((legacy.Repository, "resolve"), (legacy.Repository, "scan"),
                (legacy.ArtifactStore, "publish_bytes"), (legacy.ArtifactStore, "read_verified"),
                (legacy.subprocess, "Popen"), *[(model, method) for model in (
                    Ridge, ElasticNet, HistGradientBoostingRegressor) for method in ("__init__", "fit")]):
            guarded.setattr(owner, name, legacy.fail_if_called)
        assert versions.compile_request(document)["counts"]["search_fits"] == 603
    assert seen == [document["limits"]["child"]]
    document["limits"]["child"]["memory_bytes"] = 1.0
    with pytest.raises(OpsError) as caught:
        versions.compile_request(document)
    assert caught.value.problem == make_problem("INVALID_EXPERIMENT_SPEC",
        "search execution requires explicit positive integer resource limits")
    assert type(seen[-1]["memory_bytes"]) is float
