import dataclasses
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
from engine.v2.models.deployment import (
    MissingReleaseRoot,
    StaleReleaseHash,
    production_release_root,
    restage_semantic_hash,
)


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


@pytest.mark.parametrize(
    "field,value",
    [
        ("binding_id", "b2"),
        ("model_id", "m2"),
        ("role", "gate"),
        ("strategy_id", "STR-THRU"),
        ("decision_clock_id", "next-open"),
        ("adapter", "other-adapter.v1"),
        ("feature_order", ("y",)),
        ("output_names", ("other-output",)),
        ("schema_version", "model_binding.v2.0"),
    ],
)
def test_release_hash_includes_every_binding_semantic(field, value):
    release, _, _ = _fixture("r1")
    original = release.bindings[0]
    changed = dataclasses.replace(original, **{field: value})
    changed_release = dataclasses.replace(release, bindings=(changed,))

    assert deployment_module._release_hash(changed_release) != deployment_module._release_hash(release)


def test_release_hash_includes_member_identity_but_not_staged_storage_path():
    release, _, _ = _fixture("r1")
    binding = release.bindings[0]
    member = binding.members[0]
    renamed = dataclasses.replace(member, name="transform")
    changed_member_release = dataclasses.replace(
        release, bindings=(dataclasses.replace(binding, members=(renamed,)),),
    )
    stored_member = dataclasses.replace(member, path="objects/sha256-object")
    stored_release = dataclasses.replace(
        release, bindings=(dataclasses.replace(binding, members=(stored_member,)),),
    )

    assert deployment_module._release_hash(changed_member_release) != deployment_module._release_hash(release)
    assert deployment_module._release_hash(stored_release) == deployment_module._release_hash(release)


def test_release_hash_preserves_ordered_feature_semantics():
    release, _, _ = _fixture("r1")
    binding = dataclasses.replace(release.bindings[0], feature_order=("x", "y"))
    ordered = dataclasses.replace(release, bindings=(binding,))
    permuted = dataclasses.replace(
        release, bindings=(dataclasses.replace(binding, feature_order=("y", "x")),),
    )

    assert deployment_module._release_hash(ordered) != deployment_module._release_hash(permuted)


def test_reused_explicit_id_with_same_bytes_but_different_binding_refuses(tmp_path):
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    changed_binding = dataclasses.replace(release.bindings[0], model_id="other-model")
    conflicting_release = dataclasses.replace(release, bindings=(changed_binding,))

    with pytest.raises(StagingRefused) as error:
        stage_release(tmp_path, conflicting_release, inventory, payloads)

    assert error.value.issues[0].code == "RELEASE_ID_REUSED"


def test_repeated_promote_is_idempotent_and_rolls_back_to_the_real_predecessor(tmp_path):
    r1, inv1, pay1 = _fixture("r1", intercept=1.0, coefficient=2.0)
    r2, inv2, pay2 = _fixture("r2", intercept=10.0, coefficient=20.0)
    stage_release(tmp_path, r1, inv1, pay1)
    stage_release(tmp_path, r2, inv2, pay2)

    promote(tmp_path, "r1")
    prior_b = promote(tmp_path, "r2")
    repeated = promote(tmp_path, "r2")

    assert repeated == prior_b
    history = pointer_history(tmp_path)
    assert len(history) == 2
    assert [item.release_id for item in history] == ["r1", "r2"]
    assert history[1].previous_release_id == "r1"

    state = rollback(tmp_path)
    assert state.release_id == "r1"
    assert current_pointer(tmp_path).release_id == "r1"


def test_promote_after_crash_between_pointer_write_and_history_repairs_history(tmp_path, monkeypatch):
    r1, inv1, pay1 = _fixture("r1", intercept=1.0, coefficient=2.0)
    r2, inv2, pay2 = _fixture("r2", intercept=10.0, coefficient=20.0)
    stage_release(tmp_path, r1, inv1, pay1)
    stage_release(tmp_path, r2, inv2, pay2)
    promote(tmp_path, "r1")

    real_append = deployment_module._append_history

    def _crash(root, state):
        raise RuntimeError("simulated crash between pointer write and history append")

    monkeypatch.setattr(deployment_module, "_append_history", _crash)
    with pytest.raises(RuntimeError):
        promote(tmp_path, "r2")
    monkeypatch.setattr(deployment_module, "_append_history", real_append)

    promote(tmp_path, "r2")

    history = pointer_history(tmp_path)
    assert len(history) == 2
    assert history[1].release_id == "r2"
    assert history[1].previous_release_id == "r1"

    state = rollback(tmp_path)
    assert state.release_id == "r1"
    assert current_pointer(tmp_path).release_id == "r1"


def test_rollback_refuses_a_pointer_that_is_its_own_predecessor(tmp_path):
    state = deployment_module.PointerState(
        sequence=0, release_id="r1", previous_release_id="r1",
        action="promote", at="2024-01-01T00:00:00Z",
    )
    deployment_module._atomic_write_bytes(
        deployment_module._pointer_path(tmp_path), deployment_module._encode(state),
    )

    with pytest.raises(NoPriorRelease):
        rollback(tmp_path)


# --------------------------------------------------------------------------
# production_release_root (config key)
# --------------------------------------------------------------------------


def test_production_release_root_reads_the_env_var(monkeypatch, tmp_path):
    """A configured MODEL_RELEASE_ROOT that is already absolute is returned
    as-is."""
    monkeypatch.setenv("MODEL_RELEASE_ROOT", str(tmp_path))
    assert production_release_root() == tmp_path


def test_production_release_root_missing_env_var_refuses(monkeypatch):
    """An unset MODEL_RELEASE_ROOT refuses MissingReleaseRoot."""
    monkeypatch.delenv("MODEL_RELEASE_ROOT", raising=False)
    with pytest.raises(MissingReleaseRoot):
        production_release_root()


def test_production_release_root_blank_env_var_refuses(monkeypatch):
    """A whitespace-only MODEL_RELEASE_ROOT refuses MissingReleaseRoot, same
    as unset."""
    monkeypatch.setenv("MODEL_RELEASE_ROOT", "   ")
    with pytest.raises(MissingReleaseRoot):
        production_release_root()


# --------------------------------------------------------------------------
# release-hash versioning: promote/rollback require the current version
# --------------------------------------------------------------------------


def _stage_legacy_hashed(tmp_path, release_id):
    """Stage a real release, then rewrite its manifest on disk to look like
    a pre-versioning (RELEASE_HASH_MEMBER_V1) manifest -- the exact shape
    the real /root/p5-6/release-B manifest has (no release_hash_version key
    at all, and release_hash computed by the legacy member-only function).
    Returns the STAGED release (rewritten member paths, matching what
    resolve_release/current_release will later return), not the original
    pre-staging release object."""
    release, inventory, payloads = _fixture(release_id)
    staged = stage_release(tmp_path, release, inventory, payloads).release
    manifest_path = deployment_module._manifest_path(tmp_path, release_id)
    document = json.loads(manifest_path.read_text())
    document["release_hash"] = deployment_module._legacy_release_hash(staged)
    document.pop("release_hash_version", None)
    manifest_path.write_text(json.dumps(document))
    return staged


def test_promote_refuses_a_legacy_hashed_release_and_restage_clears_it(tmp_path):
    """promote() refuses a legacy-hashed manifest with StaleReleaseHash;
    restage_semantic_hash() upgrades it in place and promote() then
    succeeds."""
    _stage_legacy_hashed(tmp_path, "r1")
    with pytest.raises(StaleReleaseHash) as error:
        promote(tmp_path, "r1")
    assert error.value.release_id == "r1"
    assert error.value.hash_version == deployment_module.RELEASE_HASH_MEMBER_V1

    restaged = restage_semantic_hash(tmp_path, "r1")
    assert restaged.release_hash_version == deployment_module.RELEASE_HASH_SEMANTIC_V2

    state = promote(tmp_path, "r1")
    assert state.release_id == "r1"
    assert current_pointer(tmp_path).release_id == "r1"


def test_rollback_refuses_when_the_previous_release_is_legacy_hashed(tmp_path):
    """rollback() refuses StaleReleaseHash when the release it would roll
    back TO is legacy-hashed, not only the one being replaced."""
    _stage_legacy_hashed(tmp_path, "r1")
    restage_semantic_hash(tmp_path, "r1")
    promote(tmp_path, "r1")

    r2, inv2, pay2 = _fixture("r2")
    stage_release(tmp_path, r2, inv2, pay2)
    promote(tmp_path, "r2")

    # r1 is legacy-hashed on disk again: simulate it having never been
    # restaged by rewriting it back after the promote above.
    manifest_path = deployment_module._manifest_path(tmp_path, "r1")
    document = json.loads(manifest_path.read_text())
    document["release_hash"] = deployment_module._legacy_release_hash(
        deployment_module._read_manifest(tmp_path, "r1").release)
    document.pop("release_hash_version", None)
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(StaleReleaseHash) as error:
        rollback(tmp_path)
    assert error.value.release_id == "r1"


def test_repromoting_the_currently_deployed_release_refuses_once_its_manifest_turns_legacy(tmp_path):
    """The hash-version gate runs BEFORE the already-deployed no-op check
    in _swap_pointer, so even a repeat promote of the CURRENTLY live
    release is refused once its on-disk manifest looks stale -- proving
    the gate is not bypassed by the no-op short-circuit."""
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    promote(tmp_path, "r1")
    assert current_pointer(tmp_path).release_id == "r1"

    manifest_path = deployment_module._manifest_path(tmp_path, "r1")
    document = json.loads(manifest_path.read_text())
    document["release_hash"] = deployment_module._legacy_release_hash(release)
    document.pop("release_hash_version", None)
    manifest_path.write_text(json.dumps(document))

    with pytest.raises(StaleReleaseHash):
        promote(tmp_path, "r1")


# --------------------------------------------------------------------------
# restage_semantic_hash
# --------------------------------------------------------------------------


def test_restage_semantic_hash_is_a_noop_on_an_already_modern_manifest(tmp_path):
    """Restaging a manifest already at RELEASE_HASH_SEMANTIC_V2 returns it
    unchanged."""
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    before = deployment_module._read_manifest(tmp_path, "r1")
    after = restage_semantic_hash(tmp_path, "r1")
    assert after == before
    assert after.release_hash_version == deployment_module.RELEASE_HASH_SEMANTIC_V2


def test_restage_semantic_hash_refuses_an_unstaged_release(tmp_path):
    """Restaging a release_id with no staged manifest refuses
    ReleaseNotStaged."""
    with pytest.raises(ReleaseNotStaged):
        restage_semantic_hash(tmp_path, "ghost")


def test_restage_semantic_hash_refuses_a_tampered_legacy_manifest(tmp_path):
    """A legacy manifest whose own declared hash does not verify refuses
    StagingRefused(RELEASE_ID_REUSED) before any rewrite."""
    _stage_legacy_hashed(tmp_path, "r1")
    manifest_path = deployment_module._manifest_path(tmp_path, "r1")
    document = json.loads(manifest_path.read_text())
    document["release_hash"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(document))
    with pytest.raises(StagingRefused) as error:
        restage_semantic_hash(tmp_path, "r1")
    assert error.value.issues[0].code == "RELEASE_ID_REUSED"


def test_restage_semantic_hash_touches_only_the_manifest_file(tmp_path):
    """Restaging rewrites only manifest.json -- staged object bytes/mtimes,
    the DEPLOYED pointer and history are all untouched."""
    release = _stage_legacy_hashed(tmp_path, "r1")
    object_path = tmp_path / "objects" / release.bindings[0].members[0].content_hash.removeprefix("sha256:")
    before_object_bytes = object_path.read_bytes()
    before_object_mtime = object_path.stat().st_mtime_ns

    restage_semantic_hash(tmp_path, "r1")

    assert object_path.read_bytes() == before_object_bytes
    assert object_path.stat().st_mtime_ns == before_object_mtime
    assert current_pointer(tmp_path) is None
    assert pointer_history(tmp_path) == ()


def test_restage_semantic_hash_then_promote_then_resolve_release_round_trip(tmp_path):
    """Restage, promote, then resolve_release/current_release all return
    the same ModelRelease the legacy manifest originally staged."""
    original = _stage_legacy_hashed(tmp_path, "r1")
    restage_semantic_hash(tmp_path, "r1")
    promote(tmp_path, "r1")
    resolved = resolve_release(tmp_path, "r1")
    assert resolved == original
    assert current_release(tmp_path) == original


def test_production_release_root_expands_and_resolves_a_relative_value(monkeypatch, tmp_path):
    """A relative MODEL_RELEASE_ROOT is resolved to an absolute path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MODEL_RELEASE_ROOT", "relative/releases")
    assert production_release_root() == (tmp_path / "relative/releases").resolve()


def test_restage_semantic_hash_refuses_an_unreadable_existing_manifest(tmp_path):
    """A manifest.json that fails to parse refuses
    StagingRefused(MANIFEST_UNREADABLE) instead of raising a bare decode
    error."""
    release, inventory, payloads = _fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    manifest_path = deployment_module._manifest_path(tmp_path, "r1")
    manifest_path.write_text("not json")
    with pytest.raises(StagingRefused) as error:
        restage_semantic_hash(tmp_path, "r1")
    assert error.value.issues[0].code == "MANIFEST_UNREADABLE"


def test_promote_refuses_a_release_it_never_saw_regardless_of_hash_version(tmp_path):
    """An unstaged release_id still refuses ReleaseNotStaged -- the
    hash-version gate never masks this existing refusal."""
    # Sanity: the new hash-version gate in _swap_pointer never masks the
    # existing ReleaseNotStaged refusal for a release_id with no manifest
    # at all -- StaleReleaseHash is only ever raised once a manifest exists.
    with pytest.raises(ReleaseNotStaged):
        promote(tmp_path, "ghost")
