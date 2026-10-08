"""Issue 192 slice 1: deployment.promote's optional expected-incumbent guard (synthetic releases only)."""
import hashlib
import inspect
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
    deployment,
)
from engine.v2.models.deployment import ConcurrentPromote, DeploymentError


def _linear_payload(intercept, coefficient):
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": intercept,
                     "coefficients": [coefficient]}],
    }
    return json.dumps(payload, sort_keys=True).encode()


def _hash(payload):
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _fixture(release_id, *, intercept=1.0, coefficient=2.0):
    """A one-binding release plus its matching completeness inventory."""
    payload = _linear_payload(intercept, coefficient)
    member_hash = _hash(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,),
    )
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))

    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(
        artifact_id="m1", role="size", strategy_ids=("*",),
        compatible_clock_ids=("entry-close",),
        target_contract_ref="return.v1", ordered_features=("x",), members=(inv_member,),
    )
    inv_binding = ReleaseBinding(
        role="size", strategy_id="*", clock_id="entry-close", artifact_id="m1",
        ordered_features=("x",), required_member_kinds=("estimator",),
    )
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",),
    )
    payloads = {member_hash: payload}
    return release, inventory, payloads


def _staged_pair(root):
    """Stage and mark two distinct synthetic releases r1 and r2."""
    r1, inv1, pay1 = _fixture("r1", intercept=1.0, coefficient=2.0)
    r2, inv2, pay2 = _fixture("r2", intercept=10.0, coefficient=20.0)
    deployment.stage_release(root, r1, inv1, pay1)
    deployment.stage_release(root, r2, inv2, pay2)
    deployment.mark_staging_succeeded(root, "r1")
    deployment.mark_staging_succeeded(root, "r2")


def test_guarded_promote_succeeds_when_pointer_names_expected_incumbent(tmp_path):
    _staged_pair(tmp_path)
    deployment.promote(tmp_path, "r1")

    state = deployment.promote(tmp_path, "r2", expected_previous_release_id="r1")

    assert state.release_id == "r2"
    assert state.previous_release_id == "r1"
    assert deployment.current_pointer(tmp_path).release_id == "r2"
    assert [item.release_id for item in deployment.pointer_history(tmp_path)] == ["r1", "r2"]


def test_guarded_promote_refuses_when_incumbent_changed_and_writes_nothing(tmp_path):
    _staged_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    deployment.promote(tmp_path, "r2")
    before_pointer = deployment.current_pointer(tmp_path)
    before_history = deployment.pointer_history(tmp_path)

    with pytest.raises(ConcurrentPromote) as error:
        deployment.promote(tmp_path, "r1", expected_previous_release_id="r1")

    assert error.value.code == "CONCURRENT_PROMOTE"
    assert isinstance(error.value, DeploymentError)
    assert error.value.retryable is False
    assert "r1" in str(error.value)
    assert deployment.current_pointer(tmp_path) == before_pointer
    assert deployment.current_pointer(tmp_path).release_id == "r2"
    assert deployment.pointer_history(tmp_path) == before_history
    assert len(before_history) == 2


def test_guarded_promote_refuses_when_no_pointer_exists_yet(tmp_path):
    r1, inv1, pay1 = _fixture("r1")
    deployment.stage_release(tmp_path, r1, inv1, pay1)
    deployment.mark_staging_succeeded(tmp_path, "r1")

    with pytest.raises(ConcurrentPromote) as error:
        deployment.promote(tmp_path, "r1", expected_previous_release_id="r0")

    assert error.value.code == "CONCURRENT_PROMOTE"
    assert deployment.current_pointer(tmp_path) is None
    assert deployment.pointer_history(tmp_path) == ()


def test_same_target_noop_wins_over_a_stale_expected_incumbent(tmp_path):
    _staged_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    prior = deployment.promote(tmp_path, "r2")

    repeated = deployment.promote(tmp_path, "r2", expected_previous_release_id="r1")

    assert repeated == prior
    assert len(deployment.pointer_history(tmp_path)) == 2
    assert deployment.current_pointer(tmp_path).release_id == "r2"


def test_omitted_guard_keeps_the_previous_blind_swap_behavior(tmp_path):
    _staged_pair(tmp_path)
    deployment.promote(tmp_path, "r1")

    state = deployment.promote(tmp_path, "r2")

    assert state.release_id == "r2"
    assert state.previous_release_id == "r1"
    assert len(deployment.pointer_history(tmp_path)) == 2


def test_rollback_never_takes_the_expected_incumbent_guard(tmp_path):
    _staged_pair(tmp_path)
    deployment.promote(tmp_path, "r1")
    deployment.promote(tmp_path, "r2")

    assert "expected_previous_release_id" not in inspect.signature(
        deployment.rollback).parameters
    state = deployment.rollback(tmp_path)
    assert state.release_id == "r1"
    assert deployment.current_pointer(tmp_path).release_id == "r1"
