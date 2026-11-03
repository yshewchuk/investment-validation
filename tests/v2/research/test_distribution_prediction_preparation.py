"""Prediction registration keeps native identity and never reserves or executes."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from engine.v2.contracts import SnapshotRef
from engine.v2.foundation import canonical_json, to_document
from engine.v2.ops.errors import OpsError
from experiments import distribution_prediction_preparation as preparation
from experiments.distribution_candidate_recipe import candidate_recipe
from experiments.native_registration import NativeRegistration
from tests.v2.research.magnitude_trial_support import prediction_spec


@pytest.fixture
def case(monkeypatch):
    snapshot = SnapshotRef(snapshot_id="synthetic", manifest_hash="synthetic-manifest", table_versions={},
        calendar_version="calendar", source_priority_version="priority", finality_receipt_refs=(),
        knowledge_mode_by_table={})
    common = {"snapshot": to_document(snapshot), "scope": "selection", "event_ids": ["train", "inner", "outer"],
        "holdouts": {"holdout_as_of_month": "2025-01", "policy": "synthetic-unchanged"}}
    binding = {**common, "schema_version": "native_experiment_registration.v1.7"}
    own = {**deepcopy(common), "schema_version": "native_experiment_registration.v1.0"}
    source = NativeRegistration("source", canonical_json(binding).encode())
    spec = prediction_spec(runner="experiments/distribution_prediction_run.py", folds=("2024",))
    native = {"plan_id": "synthetic-plan", "horizons": [5, 3, 1],
        "folds": [{"outer": {"year": "2024"}, "inner": [{"year": "2023"}]}]}
    calls, source_request, ref = [], {"explicit": "source-request"}, object()
    conn, store, repository = object(), object(), object()
    def read(*args):
        assert args == (conn, store, source, source_request, ref)
        calls.append("read-source")
        return native
    def prepare(actual_repository, actual_spec, **kwargs):
        assert actual_repository is repository and actual_spec is spec
        assert kwargs["snapshot"] == snapshot
        assert kwargs["scope"] == binding["scope"] and kwargs["event_ids"] == binding["event_ids"]
        assert kwargs["as_of_month"] == binding["holdouts"]["holdout_as_of_month"]
        assert kwargs["code_root"].joinpath("experiments/distribution_prediction_preparation.py").is_file()
        calls.append("prepare-own")
        return NativeRegistration("prediction-own", canonical_json(own).encode()), spec
    monkeypatch.setattr(preparation, "read_search_plan", read)
    monkeypatch.setattr(preparation, "prepare_native_registration", prepare)
    caps = {"max_fits": 12, "max_trees": 49200,
        "child_limits": {"memory_bytes": 1024, "cpu_seconds": 3, "wall_seconds": 5}}
    return SimpleNamespace(args=(conn, store, repository, spec, source, source_request, ref, candidate_recipe("H"), caps),
        own=own, source=source, native=native, calls=calls)


def test_preparation_uses_exact_source_identity_and_native_fit_budget(case):
    registration, spec, result = preparation._prepare(*case.args)
    assert spec is case.args[3] and registration.run_id == "prediction-own"
    assert result["schema_version"] == "distribution_prediction_preparation.v1"
    assert result["run_id"] == registration.run_id and result["variant_id"] == registration.variant_id
    assert result["source_plan_id"] == case.native["plan_id"] and result["final_read_authority"] == "none"
    assert result["fit_budget"]["counts"]["total_fits"] == 12
    assert case.calls == ["read-source", "prepare-own"]


@pytest.mark.parametrize("field,value", [("snapshot", {}), ("scope", "other"), ("event_ids", ["outer"]),
    ("holdouts", {}), ("schema_version", "native_experiment_registration.v1.7"), ("final_read", None)])
def test_changed_own_binding_or_final_read_refuses(case, field, value):
    case.own[field] = value
    with pytest.raises(OpsError) as caught:
        preparation._prepare(*case.args)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
    assert case.calls == ["read-source", "prepare-own"]


@pytest.mark.parametrize("fault", ["schema", "horizon", "folds", "caps", "caps_type", "recipe", "budget"])
def test_invalid_source_or_budget_refuses_before_own_preparation(case, fault):
    args = list(case.args)
    if fault == "schema":
        value = case.source.document
        value["schema_version"] = "native_experiment_registration.v1.6"
        # The source reader's identity seam still expects this exact source object.
        object.__setattr__(case.source, "binding_json", canonical_json(value).encode())
    elif fault == "horizon":
        case.native["horizons"].remove(1)
    elif fault == "folds":
        case.native["folds"][0]["outer"]["year"] = "2023"
    elif fault == "caps":
        args[-1] = {**args[-1], "unknown": 1}
    elif fault == "caps_type":
        args[-1] = None
    elif fault == "recipe":
        args[-2] = {**args[-2], "horizons": [1]}
    else:
        args[-1] = {**args[-1], "max_fits": 1}
    with pytest.raises(OpsError) as caught:
        preparation._prepare(*args)
    assert caught.value.code == ("RESOURCE_LIMIT_EXCEEDED" if fault == "budget" else "INVALID_EXPERIMENT_SPEC")
    assert case.calls == ["read-source"]


def test_preparation_preserves_native_registration_byte_cap(case, monkeypatch):
    monkeypatch.setattr(preparation, "MAX_NATIVE_EVIDENCE_BYTES", 1)
    with pytest.raises(OpsError) as caught:
        preparation._prepare(*case.args)
    assert caught.value.code == "RESOURCE_LIMIT_EXCEEDED"


def test_source_run_cannot_be_reused_as_prediction_owner(case, monkeypatch):
    monkeypatch.setattr(preparation, "prepare_native_registration", lambda *args, **kwargs:
        (NativeRegistration(case.source.run_id, canonical_json(case.own).encode()), case.args[3]))
    with pytest.raises(OpsError) as caught:
        preparation._prepare(*case.args)
    assert caught.value.code == "INVALID_EXPERIMENT_SPEC"
