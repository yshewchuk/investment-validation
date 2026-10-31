"""Synthetic IV registration-to-sealed-plan path; never source payloads or fits."""
from copy import deepcopy
from dataclasses import replace

import pytest

from engine.models.no_fit import no_fit_guard as legacy_no_fit
from engine.v2.foundation import canonical_json, content_hash, from_document
from engine.v2.contracts import ArtifactRef
from engine.v2.models.no_fit import no_fit_guard
from engine.v2.ops.errors import OpsError
from engine.v2.research.magnitude_panel_qualified_table import recipe
from engine.v2.research.magnitude_panel_v2_table import recipe as v2_recipe
from experiments import magnitude_search_plan as plans
from experiments import magnitude_search_request as compiler
from experiments import native_registration as native
from experiments.magnitude_search_request_versions import compile_request
from tests.v2.research.magnitude_panel_policy_support import forbid_policy_io
from tests.v2.research.magnitude_trial_support import catalog_rows
from tests.v2.research.test_iv_crush_registration import prepared  # noqa: F401
from tests.v2.research.test_magnitude_panel_qualified import panel_data  # noqa: F401
from tests.v2.research.test_magnitude_search_request import request as size_request
from tests.v2.research.test_native_registration import artifact_object_bytes
from tests.v2.research.test_size_review_contract import panel_request as wire_panel_request


def request(data):
    value = size_request()
    panel = {**data.panel_request, "feature_snapshot_id": data.features.snapshot_id,
        "history_snapshot": data.panel_request["history_snapshot"].snapshot_id,
        "price_history_snapshot": data.panel_request["price_history_snapshot"].snapshot_id,
        "event_ids": data.registration.document["event_ids"]}
    value.update(schema_version="regression_search_request.v1", panel_request=panel,
        feature_snapshot_id=data.features.snapshot_id, history_snapshot_id=panel["history_snapshot"],
        feature_dictionary_hash=recipe("HMO")["dictionary_hash"], outer_years=["2024"],
        inner_years={"2024": ["2023"]}, max_fits=102,
        comparison={"reference": "iv_crush_causal_refit", "sanity_reference": "training_mean",
                    "max_logical_fits": 3, "max_component_fits": 3})
    return value


def test_signed_native_seal_and_read_only_replay(prepared):  # noqa: F811
    data = prepared
    registered = native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    before = catalog_rows(data.conn)
    document = request(data)
    with no_fit_guard(), legacy_no_fit():
        sealed = plans.seal_search_plan(data.conn, data.store, registered, document)
        plan = sealed["plan"]
        ref = from_document(ArtifactRef, sealed["plan_ref"])
        objects = artifact_object_bytes(data.store)
        assert plans.read_search_plan(data.conn, data.store, registered, document, ref) == plan
        assert plans.seal_search_plan(data.conn, data.store, registered, document) == sealed
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects
    assert plan["plan_id"] == content_hash({k: v for k, v in plan.items() if k != "plan_id"})
    assert plan["target"] == registered.document["regression_plan"]["target_recipe"]
    assert plan["horizons"] == [5, 3, 1]
    assert plan["counts"] == {"candidates": 33, "inner_folds": 1, "outer_folds": 1,
        "search_fits": 102, "baseline_logical_fits": 3, "baseline_component_fits": 3,
        "total_logical_fits": 105, "total_component_fits": 105}
    assert plan["selection"] == {"metric": "inner_pooled_rmse", "ties": "candidate_id_ascending",
        "retries": 0, "outer": "frozen_selected_only", "final_read_authority": "none"}
    candidates = plan["candidates"]
    assert len({candidate["candidate_id"] for candidate in candidates}) == 33
    assert {candidate["feature_recipe"]["subset"] for candidate in candidates} == {"H", "HM", "HMO"}
    for candidate in candidates:
        assert candidate["recipe"]["target_domain"] == "signed"
        assert candidate["recipe"]["prediction_projection"] == "none"
        assert candidate["source_binding_id"] == registered.variant_id
        assert candidate["target_id"] == plan["target"]["target_id"]
        assert candidate["candidate_id"] == content_hash({k: v for k, v in candidate.items() if k != "candidate_id"})
    group = plan["folds"][0]
    inner = group["inner"][0]
    assert set(inner["training_candidate_event_ids"] + inner["evaluation_event_ids"]) <= set(group["outer"]["training_candidate_event_ids"])
    assert inner["evaluation_labels_before"] == group["outer"]["training_labels_before"]


@pytest.mark.parametrize("change", ["features", "month", "events", "outer", "comparison", "caps", "shape"])
def test_changed_or_malformed_request_refuses_without_publication(prepared, change):  # noqa: F811
    data = prepared
    native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    document = request(data)
    if change == "features":
        document["panel_request"]["feature_snapshot_id"] = data.targets.snapshot_id
        document["feature_snapshot_id"] = data.targets.snapshot_id
    elif change == "month":
        document["panel_request"]["as_of_month"] = "2026-02"
    elif change == "events":
        document["panel_request"]["event_ids"] = document["panel_request"]["event_ids"][:-1]
    elif change == "outer":
        document.update(outer_years=["2025"], inner_years={"2025": ["2023"]})
    elif change == "comparison":
        document["comparison"]["reference"] = "size_v1_4_causal_refit"
    elif change == "caps":
        document["max_fits"] = 101
    else:
        document["panel_request"]["producer"] = {}
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        plans.seal_search_plan(data.conn, data.store, data.registration, document)
    assert caught.value.code == ("INVALID_EXPERIMENT_SPEC" if change in {"comparison", "caps", "shape"}
                                 else "EXPERIMENT_IDENTITY_CONFLICT")
    if change == "comparison":
        assert caught.value.problem.message == "regression comparison requires iv_crush_causal_refit and training_mean references"
        assert caught.value.problem.retryable is False
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


def test_changed_plan_request_cannot_replay_old_ref(prepared):  # noqa: F811
    data = prepared
    native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    original = request(data)
    sealed = plans.seal_search_plan(data.conn, data.store, data.registration, original)
    changed = deepcopy(original)
    changed["limits"]["job"]["wall_seconds"] += 1
    with pytest.raises(OpsError) as caught:
        plans.read_search_plan(data.conn, data.store, data.registration, changed,
                               from_document(ArtifactRef, sealed["plan_ref"]))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert compile_request(original)["request_id"] != compile_request(changed)["request_id"]


@pytest.mark.parametrize("change", ["source", "empty", "runner", "policy", "planner", "target"])
def test_changed_source_or_target_anchor_refuses_before_plan_write(prepared, change):  # noqa: F811
    data = prepared
    document = data.registration.document
    if change == "source":
        document["source_closure"]["experiments/iv_crush_search.py"] = "sha256:" + "0" * 64
    elif change == "empty":
        document["source_closure"] = {}
    elif change in {"runner", "policy", "planner"}:
        path = {"runner": "experiments/iv_crush_search.py",
            "policy": "experiments/iv_crush_registration.py",
            "planner": "experiments/magnitude_search_plan.py"}[change]
        del document["source_closure"][path]
    else:
        temporal = document["regression_plan"]
        target = temporal["target_recipe"]
        target["domain"] = "nonnegative"
        target["target_id"] = content_hash({k: v for k, v in target.items() if k != "target_id"})
        temporal["plan_id"] = content_hash({k: v for k, v in temporal.items() if k != "plan_id"})
    changed = replace(data.registration, binding_json=canonical_json(document).encode())
    if change == "target":
        objects = artifact_object_bytes(data.store)
        with pytest.raises(OpsError) as caught:
            native.reserve_native_registration(data.conn, data.store, changed, data.spec)
        assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert artifact_object_bytes(data.store) == objects
        return
    native.reserve_native_registration(data.conn, data.store, changed, data.spec)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with no_fit_guard(), legacy_no_fit(), pytest.raises(OpsError) as caught:
        plans.seal_search_plan(data.conn, data.store, changed, request(data))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("change", ["domain", "runner", "price_source"])
def test_iv_identity_refuses_before_shared_plan(prepared, monkeypatch, change):  # noqa: F811
    from experiments import regression_search_binding
    data = prepared
    document = data.registration.document
    if change == "domain":
        temporal = document["regression_plan"]
        target = temporal["target_recipe"]
        target["domain"] = "nonnegative"
        target["target_id"] = content_hash({k: v for k, v in target.items() if k != "target_id"})
        temporal["plan_id"] = content_hash({k: v for k, v in temporal.items() if k != "plan_id"})
    else:
        document["execution_plan"][change] = "other"
    changed = replace(data.registration, binding_json=canonical_json(document).encode())
    def poison(*args, **kwargs):
        raise AssertionError("invalid IV identity reached shared planning")
    monkeypatch.setattr(regression_search_binding, "prepare_plan", poison)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with pytest.raises(OpsError) as caught:
        plans.seal_search_plan(data.conn, data.store, changed, request(data))
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


@pytest.mark.parametrize("change", ["null", "empty", "extra", "date", "events", "horizons", "policy", "limits"])
def test_panel_request_refuses_before_expansion_without_mutation(prepared, monkeypatch, change):  # noqa: F811
    data = prepared
    document = deepcopy(request(data))
    panel = document["panel_request"]
    if change in {"null", "empty"}:
        document["panel_request"] = None if change == "null" else {}
    elif change == "extra":
        panel["producer"] = {}
    elif change == "date":
        panel["as_of_month"] = "2025-13"
    elif change == "events":
        panel["event_ids"].append(panel["event_ids"][0])
    elif change == "horizons":
        panel["horizons"] = [True, 3, 5]
    elif change == "policy":
        panel["policy"]["embargo_days"] = 29
    else:
        panel["limits"]["max_source_rows"] = True
    original = deepcopy(document)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    def no_expansion(*args, **kwargs):
        pytest.fail("invalid panel request reached candidate expansion")
    monkeypatch.setattr(compiler, "_specifications", no_expansion)
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with no_fit_guard(), legacy_no_fit(), pytest.raises(OpsError) as caught:
            compile_request(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "invalid closed SIZE panel request"
    assert caught.value.problem.retryable is False
    assert document == original
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects


def test_valid_regression_preview_is_detached_without_io_or_fits(prepared, monkeypatch):  # noqa: F811
    document = request(prepared)
    original = deepcopy(document)
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with no_fit_guard(), legacy_no_fit():
            result = compile_request(document)
    assert result["request"] == original == document
    assert len(result["candidate_specs"]) == 33
    result["request"]["panel_request"]["event_ids"].clear()
    assert document == original


def _preview_request():
    value = size_request()
    panel = wire_panel_request()
    panel.update(feature_snapshot_id=value["feature_snapshot_id"],
                 history_snapshot=value["history_snapshot_id"],
                 price_history_snapshot="snap_" + "4" * 32)
    value.update(schema_version="regression_search_request.v1", panel_request=panel,
                 feature_dictionary_hash=recipe("HMO")["dictionary_hash"])
    value["comparison"].update(reference="iv_crush_causal_refit", sanity_reference="training_mean")
    return value


@pytest.mark.parametrize("compile", [compiler.compile_request, compile_request], ids=["core", "versions"])
@pytest.mark.parametrize("dictionary", [compiler.panel_recipe("HMO")["dictionary_hash"],
    v2_recipe("HMO")["dictionary_hash"], recipe("H")["dictionary_hash"], "sha256:" + "0" * 64],
    ids=["v1", "v2", "v3-subset", "unknown"])
def test_regression_dictionary_refuses_before_expansion(compile, dictionary, monkeypatch):
    document = _preview_request()
    document["feature_dictionary_hash"] = dictionary
    original = deepcopy(document)
    def no_expansion(*args, **kwargs):
        pytest.fail("unsupported regression dictionary reached candidate expansion")
    monkeypatch.setattr(compiler, "_specifications", no_expansion)
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with no_fit_guard(), legacy_no_fit(), pytest.raises(OpsError) as caught:
            compile(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.message == "stage-2 feature dictionary differs from the supported panel dictionary"
    assert caught.value.problem.retryable is False
    assert document == original


def test_both_regression_compilers_preserve_exact_qualified_preview(monkeypatch):
    document = _preview_request()
    original = deepcopy(document)
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with no_fit_guard(), legacy_no_fit():
            core = compiler.compile_request(document)
            versioned = compile_request(document)
    assert core == versioned
    assert content_hash(core) == "sha256:25920c14a044ae245c1e8cc37fdac2bcc1f5ecde6f0d30453fcf737d4badc891"
    assert core["request"] == original == document
    for candidate in core["candidate_specs"]:
        assert candidate["feature_recipe"] == recipe(candidate["feature_recipe"]["subset"])
    core["request"]["panel_request"]["event_ids"].clear()
    assert document == original


@pytest.mark.parametrize("compile", [compiler.compile_request, compile_request], ids=["core", "versions"])
@pytest.mark.parametrize("field,value", [("reference", "size_v1_4_causal_refit"),
    ("sanity_reference", "history_mean"), ("reference", None), ("extra", True),
    (None, None), ("max_logical_fits", 0), ("max_component_fits", 0)])
def test_regression_comparison_refusal_names_its_policy(compile, field, value, monkeypatch):
    document = _preview_request()
    if field is None:
        document["comparison"] = value
    else:
        document["comparison"][field] = value
    original = deepcopy(document)
    def no_expansion(*args, **kwargs):
        pytest.fail("invalid comparison reached candidate expansion")
    monkeypatch.setattr(compiler, "_specifications", no_expansion)
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with no_fit_guard(), legacy_no_fit(), pytest.raises(OpsError) as caught:
            compile(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    expected = ("stage-2 comparison fit cap is insufficient" if field in
        ("max_logical_fits", "max_component_fits") else
        "regression comparison requires iv_crush_causal_refit and training_mean references")
    assert caught.value.problem.message == expected
    assert caught.value.problem.retryable is False
    assert document == original


def _invalid_local_wire(document, change):
    if change == "memory":
        document["limits"]["child"]["memory_bytes"] = document["limits"]["parent"]["memory_bytes"] + 1
    else:
        name = {"price": "price_history_snapshot", "features": "feature_snapshot_id",
                "history": "history_snapshot"}[change]
        document["panel_request"][name] = "not-a-snapshot" if change == "price" else "snap_" + "f" * 32


@pytest.mark.parametrize("compile", [compiler.compile_request, compile_request], ids=["core", "versions"])
@pytest.mark.parametrize("change", ["price", "features", "history", "memory"])
def test_regression_local_wire_refuses_before_expansion(compile, change, monkeypatch):
    document = _preview_request()
    _invalid_local_wire(document, change)
    original = deepcopy(document)
    def no_expansion(*args, **kwargs):
        pytest.fail("invalid regression local wire reached candidate expansion")
    monkeypatch.setattr(compiler, "_specifications", no_expansion)
    with monkeypatch.context() as guarded:
        forbid_policy_io(guarded)
        with no_fit_guard(), legacy_no_fit(), pytest.raises(OpsError) as caught:
            compile(document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert caught.value.problem.retryable is False
    assert document == original


@pytest.mark.parametrize("change", ["price", "features", "history", "memory"])
def test_regression_local_wire_refuses_before_plan_publication(prepared, change):  # noqa: F811
    data = prepared
    native.reserve_native_registration(data.conn, data.store, data.registration, data.spec)
    document = request(data)
    _invalid_local_wire(document, change)
    original = deepcopy(document)
    before, objects = catalog_rows(data.conn), artifact_object_bytes(data.store)
    with no_fit_guard(), legacy_no_fit(), pytest.raises(OpsError) as caught:
        plans.seal_search_plan(data.conn, data.store, data.registration, document)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert document == original
    assert catalog_rows(data.conn) == before and artifact_object_bytes(data.store) == objects
