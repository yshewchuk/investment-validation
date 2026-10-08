"""resolve_gate_policy reads one gate's staged, hash-verified registry threshold."""
import dataclasses
import json

import pytest

from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
)
from engine.v2.models import deployment
from engine.v2.ops import native_score_batch
from engine.v2.ops.native_score_batch import run_native_score_batch_worker
from engine.v2.scoring.release_bindings import (
    ModelNotReady,
    resolve_gate_policy,
    resolve_release_binding,
)
from tests.test_v2_ops_native_score_batch import (
    _event_doc,
    _linear_payload,
    _sha,
    _worker_parameters,
    _write_empty_catalog,
)

_STANDARD_REGISTRY = json.dumps({"models": [
    {"id": "m-other", "threshold": 9.5}, {"id": "m-gate", "threshold": 0.375},
]}).encode()


def _registry(*entries) -> bytes:
    return json.dumps({"models": list(entries)}).encode()


def _stage_gate_release(tmp_path, *, registry_payload: bytes | None, release_id="r1"):
    """Stage + promote a synthetic STR-THRU release with one binding per role
    ("driver", "gate"), both clocked "entry-close"; when ``registry_payload``
    is given, the gate binding additionally carries a hash-verified
    ``threshold`` member holding those exact bytes. Returns the real resolved
    ScoringReleaseBinding."""
    bindings = []
    artifacts = []
    release_bindings = []
    requirements = []
    payloads = {}
    for role in ("driver", "gate"):
        payload = _linear_payload()
        member_hash = _sha(payload)
        payloads[member_hash] = payload
        members = [ArtifactMember(name="estimator", path="unused.json",
                                  content_hash=member_hash)]
        inventory_members = [ArtifactInventoryMember(
            member_id=f"m-{role}:estimator", kind="estimator",
            artifact_ref=f"artifact://m-{role}", content_hash=member_hash)]
        required_kinds = ("estimator",)
        if role == "gate" and registry_payload is not None:
            members.append(ArtifactMember(name="threshold", path="unused.json",
                                          content_hash=_sha(registry_payload)))
            payloads[_sha(registry_payload)] = registry_payload
            inventory_members.append(ArtifactInventoryMember(
                member_id="m-gate:threshold", kind="threshold",
                artifact_ref="artifact://m-gate-threshold",
                content_hash=_sha(registry_payload)))
            required_kinds = ("estimator", "threshold")
        bindings.append(ModelBinding(
            binding_id=f"b-{role}", model_id=f"m-{role}", role=role,
            strategy_id="STR-THRU", decision_clock_id="entry-close",
            adapter="json-linear.v1", feature_order=("x",),
            output_names=("prediction",), members=tuple(members)))
        artifacts.append(ModelArtifactInventory(
            artifact_id=f"m-{role}", role=role, strategy_ids=("STR-THRU",),
            compatible_clock_ids=("entry-close",), target_contract_ref="return.v1",
            ordered_features=("x",), members=tuple(inventory_members)))
        release_bindings.append(ReleaseBinding(
            role=role, strategy_id="STR-THRU", clock_id="entry-close",
            artifact_id=f"m-{role}", ordered_features=("x",),
            required_member_kinds=required_kinds))
        requirements.append(ReleaseRequirement(
            role=role, strategy_id="STR-THRU", clock_id="entry-close"))
    release = ModelRelease(release_id=release_id, deployment_id="d1",
                           bindings=tuple(bindings))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=tuple(artifacts), bindings=tuple(release_bindings),
        requirements=tuple(requirements), artifact_manifest_ref="manifest://r",
        evidence_refs=("evidence://r",))
    dep_root = tmp_path / "deployment"
    deployment.stage_release(dep_root, release, inventory, payloads)
    deployment.mark_staging_succeeded(dep_root, release_id)
    deployment.promote(dep_root, release_id)
    _write_empty_catalog(tmp_path, release_id)
    return resolve_release_binding(tmp_path)


def _stage_events(tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    (root / "events.json").write_text(json.dumps([_event_doc()]))
    return root


def _capture_gate_policy(monkeypatch) -> dict:
    """Wrap assemble_score_batch_inputs so the worker's resolved gate_policy
    argument is recorded while the real assembly still runs."""
    recorded = {}
    original = native_score_batch.assemble_score_batch_inputs

    def wrapper(**kwargs):
        recorded["gate_policy"] = kwargs["gate_policy"]
        return original(**kwargs)

    monkeypatch.setattr(native_score_batch, "assemble_score_batch_inputs", wrapper)
    return recorded


def test_threshold_resolved_from_the_release_registry_member(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    assert resolve_gate_policy(binding, tmp_path) == {"STR-THRU": {"threshold": 0.375}}


def test_no_threshold_member_omits_the_strategy(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=None)
    assert resolve_gate_policy(binding, tmp_path) == {}


def test_hash_mismatch_raises_model_not_ready(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    gate_binding = next(b for b in binding.model_release.bindings if b.role == "gate")
    member = next(m for m in gate_binding.members if m.name == "threshold")
    (tmp_path / "deployment" / member.path).write_bytes(b"tampered after resolve")
    with pytest.raises(ModelNotReady) as exc:
        resolve_gate_policy(binding, tmp_path)
    assert exc.value.member_id == "model:gate:STR-THRU"


def test_duplicate_threshold_member_raises_model_not_ready(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    gate_binding = next(b for b in binding.model_release.bindings if b.role == "gate")
    threshold = next(m for m in gate_binding.members if m.name == "threshold")
    duplicated = dataclasses.replace(gate_binding, members=(*gate_binding.members, threshold))
    release = dataclasses.replace(
        binding.model_release,
        bindings=tuple(duplicated if b.binding_id == gate_binding.binding_id else b
                       for b in binding.model_release.bindings))
    patched = dataclasses.replace(binding, model_release=release)
    with pytest.raises(ModelNotReady) as exc:
        resolve_gate_policy(patched, tmp_path)
    assert exc.value.member_id == "model:gate:STR-THRU"


def test_duplicate_gate_binding_raises_model_not_ready(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    gate_binding = next(b for b in binding.model_release.bindings if b.role == "gate")
    release = dataclasses.replace(
        binding.model_release,
        bindings=binding.model_release.bindings + (gate_binding,))
    patched = dataclasses.replace(binding, model_release=release)
    with pytest.raises(ModelNotReady) as exc:
        resolve_gate_policy(patched, tmp_path)
    assert exc.value.member_id == "model:gate:STR-THRU"


def test_driver_binding_sharing_the_gate_binding_id_is_not_a_match(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    driver = next(b for b in binding.model_release.bindings if b.role == "driver")
    gate = next(b for b in binding.model_release.bindings if b.role == "gate")
    shadowed_driver = dataclasses.replace(driver, binding_id=gate.binding_id)
    others = tuple(b for b in binding.model_release.bindings if b.role != "driver")
    release = dataclasses.replace(
        binding.model_release, bindings=(shadowed_driver, *others))
    patched = dataclasses.replace(binding, model_release=release)
    assert resolve_gate_policy(patched, tmp_path) == {"STR-THRU": {"threshold": 0.375}}


_BAD_REGISTRIES = [
    pytest.param(b"not json", id="not-json"),
    pytest.param(b"[" * 200000, id="too-deeply-nested"),
    pytest.param(b"[]", id="array-document"),
    pytest.param(json.dumps({"models": "x"}).encode(), id="models-not-a-list"),
    pytest.param(_registry({"id": "m-other", "threshold": 9.5}), id="no-gate-entry"),
    pytest.param(_registry({"id": "m-gate", "threshold": 0.375},
                           {"id": "m-gate", "threshold": 9.5}), id="two-gate-entries"),
    pytest.param(_registry({"id": "m-gate"}), id="no-threshold-key"),
    pytest.param(_registry({"id": "m-gate", "threshold": None}), id="threshold-null"),
    pytest.param(_registry({"id": "m-gate", "threshold": True}), id="threshold-bool"),
    pytest.param(_registry({"id": "m-gate", "threshold": "0.375"}), id="threshold-string"),
    pytest.param(b'{"models": [{"id": "m-gate", "threshold": NaN}]}', id="threshold-nan"),
    pytest.param(b'{"models": [{"id": "m-gate", "threshold": Infinity}]}',
                 id="threshold-infinity"),
    pytest.param(b'{"models": [{"id": "m-gate", "threshold": 1e999}]}',
                 id="threshold-overflow"),
]


@pytest.mark.parametrize("payload", _BAD_REGISTRIES)
def test_bad_registry_raises_model_not_ready(tmp_path, payload):
    binding = _stage_gate_release(tmp_path, registry_payload=payload)
    with pytest.raises(ModelNotReady) as exc:
        resolve_gate_policy(binding, tmp_path)
    assert exc.value.member_id == "model:gate:STR-THRU"
    assert str(tmp_path) not in str(exc.value)


def test_resolution_is_deterministic(tmp_path):
    binding = _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    first = resolve_gate_policy(binding, tmp_path)
    assert resolve_gate_policy(binding, tmp_path) == first
    second = resolve_release_binding(tmp_path)
    assert resolve_gate_policy(second, tmp_path) == first


def test_worker_uses_release_threshold_and_gets_past_the_gate_policy_refusal(
        tmp_path, monkeypatch):
    _stage_gate_release(tmp_path, registry_payload=_STANDARD_REGISTRY)
    root = _stage_events(tmp_path)
    recorded = _capture_gate_policy(monkeypatch)
    parameters = _worker_parameters(tmp_path, expected_ids=["native_score_batch"])
    del parameters["gate_policy"]
    run_native_score_batch_worker(parameters, root)
    assert recorded["gate_policy"] == {"STR-THRU": {"threshold": 0.375}}
    refusals_document = json.loads((root / "refusals.json").read_text())
    assert refusals_document["refusals"] == {}
    records_document = json.loads((root / "records.json").read_text())
    assert list(records_document["records"]) == ["TEST|STR-THRU|2026-01-15|am"]


def test_worker_supplied_policy_wins_and_release_is_not_consulted(
        tmp_path, monkeypatch):
    _stage_gate_release(tmp_path, registry_payload=b"not json")
    root = _stage_events(tmp_path)
    recorded = _capture_gate_policy(monkeypatch)
    run_native_score_batch_worker(
        _worker_parameters(tmp_path, expected_ids=["native_score_batch"]), root)
    assert recorded["gate_policy"] == {"STR-THRU": {"threshold": 0.0}}


def test_worker_without_threshold_member_keeps_gate_policy_not_staged(tmp_path):
    _stage_gate_release(tmp_path, registry_payload=None)
    root = _stage_events(tmp_path)
    parameters = _worker_parameters(tmp_path, expected_ids=["native_score_batch"])
    del parameters["gate_policy"]
    run_native_score_batch_worker(parameters, root)
    refusals_document = json.loads((root / "refusals.json").read_text())
    refusal = refusals_document["refusals"]["TEST|STR-THRU|2026-01-15|am"]
    assert refusal["code"] == "GATE_POLICY_NOT_STAGED"
    records_document = json.loads((root / "records.json").read_text())
    assert records_document["records"] == {}


def test_worker_bad_registry_raises_model_not_ready(tmp_path):
    _stage_gate_release(tmp_path, registry_payload=b"not json")
    root = _stage_events(tmp_path)
    parameters = _worker_parameters(tmp_path, expected_ids=["native_score_batch"])
    del parameters["gate_policy"]
    with pytest.raises(ModelNotReady):
        run_native_score_batch_worker(parameters, root)
    assert not (root / "records.json").exists()
