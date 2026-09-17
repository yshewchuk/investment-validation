from __future__ import annotations

import copy
import hashlib

import pytest

from checks.phase4_checkpoints import (
    REQUIRED_BRANCHES,
    SCHEMA_VERSION,
    CheckpointError,
    load_bundle,
    validate_bundle,
)
from engine.v2.foundation import content_hash


def _hashed(value):
    return {"value": value, "content_hash": content_hash(value)}


def _case(case_id, branches):
    request = {
        "event_id": case_id,
        "strategy_version": "STR-THRU",
        "mode": "replay",
    }
    checkpoints = {
        "features": _hashed({
            "feature_vector": {"spot": 100.0},
            "missing_mask": {"spot": False},
            "model_identity": {"artifact": "model-a"},
        }),
        "selection_pricing": _hashed({
            "selected_legs": [{"right": "call", "side": "long", "quantity": 1}],
            "entry_cost": 2.5,
        }),
        "simulation": _hashed({
            "horizon": "planned_exit",
            "capital_denominator": 2.5,
            "residual_population_identity": {"ref": "residual-a"},
            "draw_count": 1000,
            "seed": 7,
        }),
        "gate_inputs": _hashed({"entry_cost": 2.5}),
    }
    row = {
        "case_id": case_id,
        "request": request,
        "request_hash": content_hash(request),
        "strategy": "STR-THRU",
        "branches": branches,
        "resource_refs": ["models"],
        "executable_inputs": "available",
        "disposition": "compared",
        "checkpoints": checkpoints,
    }
    return {**row, "case_hash": content_hash(row)}


def _resign_case(case):
    case["case_hash"] = content_hash({
        key: value for key, value in case.items() if key != "case_hash"
    })


def _bundle(tmp_path):
    raw = b'{"artifact":"frozen"}\n'
    (tmp_path / "models.json").write_bytes(raw)
    branches = sorted(REQUIRED_BRANCHES)
    row = _case("case-1", branches)
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "release_id": "phase4-test-release",
        "resources": [{
            "resource_id": "models",
            "path": "models.json",
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
        }],
        "coverage": {
            "strategies": {"STR-THRU": ["case-1"]},
            "branches": {branch: ["case-1"] for branch in branches},
        },
        "cases": [row],
    }
    return {**bundle, "manifest_hash": content_hash(bundle)}


def _resign_bundle(bundle):
    bundle["manifest_hash"] = content_hash({
        key: value for key, value in bundle.items() if key != "manifest_hash"
    })


def test_valid_bundle_has_shared_resources_and_complete_coverage(tmp_path):
    verified = validate_bundle(_bundle(tmp_path), tmp_path)
    assert verified["case_ids"] == ("case-1",)
    assert verified["strategies"] == ("STR-THRU",)
    assert set(verified["branches"]) == REQUIRED_BRANCHES


@pytest.mark.parametrize("mutation, message", [
    (
        lambda bundle: bundle["resources"][0].update({"path": "../models.json"}),
        "escapes release root",
    ),
    (
        lambda bundle: bundle["resources"][0].update({"sha256": "sha256:" + "0" * 64}),
        "sha256: mismatch",
    ),
])
def test_resource_path_and_hash_corruption_are_rejected(tmp_path, mutation, message):
    bundle = _bundle(tmp_path)
    mutation(bundle)
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match=message):
        validate_bundle(bundle, tmp_path)


@pytest.mark.parametrize("mutation, message", [
    (
        lambda bundle: bundle["cases"].append(copy.deepcopy(bundle["cases"][0])),
        "case_id: duplicate",
    ),
    (
        lambda bundle: bundle["cases"][0].pop("case_id"),
        "unexpected or missing fields",
    ),
    (
        lambda bundle: bundle["cases"][0]["checkpoints"].pop("gate_inputs"),
        "missing required group",
    ),
])
def test_duplicate_ids_and_missing_checkpoint_group_are_rejected(tmp_path, mutation, message):
    bundle = _bundle(tmp_path)
    mutation(bundle)
    for case in bundle["cases"]:
        _resign_case(case)
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match=message):
        validate_bundle(bundle, tmp_path)


def test_incomplete_declared_coverage_is_rejected(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["coverage"]["branches"].pop("ties")
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="incomplete critical branches"):
        validate_bundle(bundle, tmp_path)


@pytest.mark.parametrize("value, message", [
    (float("nan"), "non-finite value"),
    (("not", "a", "json", "list"), "not JSON-safe"),
])
def test_malformed_json_and_nonfinite_values_are_rejected(tmp_path, value, message):
    bundle = _bundle(tmp_path)
    checkpoint = bundle["cases"][0]["checkpoints"]["features"]
    checkpoint["value"]["feature_vector"]["spot"] = value
    checkpoint["content_hash"] = content_hash(checkpoint["value"])
    _resign_case(bundle["cases"][0])
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match=message):
        validate_bundle(bundle, tmp_path)


def test_missing_executable_inputs_cannot_be_success(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["cases"][0]["executable_inputs"] = "missing"
    _resign_case(bundle["cases"][0])
    _resign_bundle(bundle)
    with pytest.raises(CheckpointError, match="must be incomparable"):
        validate_bundle(bundle, tmp_path)


def test_load_bundle_rejects_malformed_json(tmp_path):
    path = tmp_path / "checkpoints.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="invalid JSON"):
        load_bundle(path)
