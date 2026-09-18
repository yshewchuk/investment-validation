import hashlib
import json
import os

import pytest

from engine.v2.models import (
    MODEL_READY,
    ArtifactInventoryMember,
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ModelReleaseRefusal,
    NoPriorRelease,
    ReleaseBinding,
    ReleaseNotStaged,
    ReleaseRequirement,
    StagingRefused,
    current_pointer,
    current_release,
    pointer_history,
    promote,
    resolve_release,
    rollback,
    stage_release,
)
from engine.v2.models import deployment as deployment_module


def _linear_payload(intercept, coefficient):
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": intercept, "coefficients": [coefficient]}],
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
        artifact_id="m1", role="size", strategy_ids=("*",), compatible_clock_ids=("entry-close",),
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


# --------------------------------------------------------------------------
# staging refusals
# --------------------------------------------------------------------------


def test_partial_release_missing_a_required_binding_refuses(tmp_path):
    release, inventory, payloads = _fixture("r1")
    # inventory requires a second (gate, STR-THRU) binding the release never supplies.
    gate_member = ArtifactInventoryMember(member_id="g1:estimator", kind="estimator",
                                           artifact_ref="artifact://g1", content_hash=_hash(b"gate"))
    gate_artifact = ModelArtifactInventory(
        artifact_id="g1", role="gate", strategy_ids=("STR-THRU",), compatible_clock_ids=("entry-close",),
        target_contract_ref="return.v1", ordered_features=("x",), members=(gate_member,),
    )
    gate_binding = ReleaseBinding(
        role="gate", strategy_id="STR-THRU", clock_id="entry-close", artifact_id="g1",
        ordered_features=("x",), required_member_kinds=("estimator",),
    )
    inventory = ModelReleaseInventory(
        release_id=inventory.release_id, deployment_id=inventory.deployment_id,
        known_clock_ids=inventory.known_clock_ids,
        artifacts=inventory.artifacts + (gate_artifact,),
        bindings=inventory.bindings + (gate_binding,),
        requirements=inventory.requirements + (
            ReleaseRequirement(role="gate", strategy_id="STR-THRU", clock_id="entry-close"),
        ),
        artifact_manifest_ref=inventory.artifact_manifest_ref, evidence_refs=inventory.evidence_refs,
    )
    with pytest.raises(StagingRefused) as error:
        stage_release(tmp_path, release, inventory, payloads)
    assert "MISSING_INFERENCE_BINDING" in [item.code for item in error.value.issues]
    assert not (tmp_path / "releases" / "r1" / "manifest.json").exists()


def test_incompatible_feature_order_refuses(tmp_path):
    release, inventory, payloads = _fixture("r1")
    binding = release.bindings[0]
    reordered = ModelBinding(
        binding_id=binding.binding_id, model_id=binding.model_id, role=binding.role,
        strategy_id=binding.strategy_id, decision_clock_id=binding.decision_clock_id,
        adapter=binding.adapter, feature_order=("y",), output_names=binding.output_names,
        members=binding.members,
    )
    release = ModelRelease(release_id=release.release_id, deployment_id=release.deployment_id,
                            bindings=(reordered,))
    with pytest.raises(StagingRefused) as error:
        stage_release(tmp_path, release, inventory, payloads)
    assert "INCOMPATIBLE_FEATURE_ORDER" in [item.code for item in error.value.issues]


def test_incomplete_inventory_itself_refuses_via_require_complete_release(tmp_path):
    release, inventory, payloads = _fixture("r1")
    inventory = ModelReleaseInventory(
        release_id=inventory.release_id, deployment_id=inventory.deployment_id,
        known_clock_ids=inventory.known_clock_ids, artifacts=inventory.artifacts,
        bindings=(), requirements=inventory.requirements,
        artifact_manifest_ref=inventory.artifact_manifest_ref, evidence_refs=inventory.evidence_refs,
    )
    with pytest.raises(ModelReleaseRefusal):
        stage_release(tmp_path, release, inventory, payloads)


def test_staging_never_touches_the_live_pointer(tmp_path):
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    assert current_pointer(tmp_path) is None
    assert pointer_history(tmp_path) == ()


# --------------------------------------------------------------------------
# promote / rollback / replay
# --------------------------------------------------------------------------


def test_promote_then_rollback_restores_the_exact_prior_state(tmp_path):
    r1, inv1, pay1 = _fixture("r1", intercept=1.0, coefficient=2.0)
    r2, inv2, pay2 = _fixture("r2", intercept=10.0, coefficient=20.0)
    stage_release(tmp_path, r1, inv1, pay1)
    stage_release(tmp_path, r2, inv2, pay2)

    promote(tmp_path, "r1")
    assert current_pointer(tmp_path).release_id == "r1"
    promote(tmp_path, "r2")
    assert current_pointer(tmp_path).release_id == "r2"

    state = rollback(tmp_path)
    assert state.release_id == "r1"
    assert current_pointer(tmp_path).release_id == "r1"
    assert current_release(tmp_path) == resolve_release(tmp_path, "r1")


def test_rollback_with_no_prior_release_refuses(tmp_path):
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    promote(tmp_path, "r1")
    with pytest.raises(NoPriorRelease):
        rollback(tmp_path)


def test_promote_unstaged_release_refuses(tmp_path):
    with pytest.raises(ReleaseNotStaged):
        promote(tmp_path, "ghost")


def test_replay_after_later_promotion_resolves_the_old_release_by_id(tmp_path):
    r1, inv1, pay1 = _fixture("r1", intercept=1.0, coefficient=2.0)
    r2, inv2, pay2 = _fixture("r2", intercept=100.0, coefficient=200.0)
    stage_release(tmp_path, r1, inv1, pay1)
    stage_release(tmp_path, r2, inv2, pay2)
    promote(tmp_path, "r1")
    resolved_r1_before = resolve_release(tmp_path, "r1")

    promote(tmp_path, "r2")
    assert current_pointer(tmp_path).release_id == "r2"

    # r1's exact members are still resolvable by id, unaffected by promoting r2.
    resolved_r1_after = resolve_release(tmp_path, "r1")
    assert resolved_r1_after == resolved_r1_before

    inference = FrozenInference(tmp_path)
    request = InferenceRequest(release_id="r1", binding_id="b1", feature_order=("x",), rows=((3.0,),))
    result = inference.infer(resolved_r1_after, request)
    assert result.status == MODEL_READY
    assert result.predictions == ((7.0,),)  # 1.0 + 2.0*3.0 -- r1's own coefficients, not r2's


def test_pointer_history_is_append_only(tmp_path):
    r1, inv1, pay1 = _fixture("r1")
    r2, inv2, pay2 = _fixture("r2", intercept=5.0, coefficient=6.0)
    stage_release(tmp_path, r1, inv1, pay1)
    stage_release(tmp_path, r2, inv2, pay2)

    promote(tmp_path, "r1")
    first_entry = (tmp_path / "history" / "000000.json").read_bytes()
    promote(tmp_path, "r2")
    # the earlier history file must be byte-identical after a later promotion.
    assert (tmp_path / "history" / "000000.json").read_bytes() == first_entry
    rollback(tmp_path)

    history = pointer_history(tmp_path)
    assert [item.sequence for item in history] == [0, 1, 2]
    assert [item.release_id for item in history] == ["r1", "r2", "r1"]
    assert [item.action for item in history] == ["promote", "promote", "rollback"]


def test_interrupted_promote_leaves_the_old_pointer(tmp_path, monkeypatch):
    release, inventory, payloads = _fixture("r1")
    other, inv2, pay2 = _fixture("r2", intercept=9.0, coefficient=9.0)
    stage_release(tmp_path, release, inventory, payloads)
    stage_release(tmp_path, other, inv2, pay2)
    promote(tmp_path, "r1")
    before = current_pointer(tmp_path)

    real_replace = os.replace

    def _fail_on_pointer_swap(src, dst):
        if os.path.basename(str(dst)) == "DEPLOYED":
            raise OSError("simulated crash between temp write and rename")
        return real_replace(src, dst)

    monkeypatch.setattr(deployment_module.os, "replace", _fail_on_pointer_swap)

    with pytest.raises(OSError):
        promote(tmp_path, "r2")

    monkeypatch.setattr(deployment_module.os, "replace", real_replace)
    assert current_pointer(tmp_path) == before
    assert current_pointer(tmp_path).release_id == "r1"
    # no orphan history entry was recorded for the failed promotion.
    assert len(pointer_history(tmp_path)) == 1
    # no leftover temp file for the pointer itself.
    leftovers = [p for p in tmp_path.glob(".DEPLOYED.tmp-*")]
    assert leftovers == []


def test_staging_same_release_id_twice_with_same_content_is_a_noop(tmp_path):
    release, inventory, payloads = _fixture("r1")
    first = stage_release(tmp_path, release, inventory, payloads)
    second = stage_release(tmp_path, release, inventory, payloads)
    assert first == second


def test_staging_same_release_id_with_different_content_refuses(tmp_path):
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    other, other_inv, other_pay = _fixture("r1", intercept=42.0, coefficient=42.0)
    with pytest.raises(StagingRefused) as error:
        stage_release(tmp_path, other, other_inv, other_pay)
    assert error.value.issues[0].code == "RELEASE_ID_REUSED"
