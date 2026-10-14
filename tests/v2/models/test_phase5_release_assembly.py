"""Pool release assembly: derive/stage a carried-forward candidate catalog (#192 slice 2)."""
from __future__ import annotations

import dataclasses
import json

import pytest

from checks import phase5_release
from checks import phase5_release_assembly as assembly
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

UNRELATED_MEMBER_ID = "payoff_line:STR-THRU"


def _fixture(release_id):
    payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                          "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}]},
                         sort_keys=True).encode()
    member_hash = phase5_release.sha256_bytes(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(binding_id="b1", model_id="m1", role="size", strategy_id="*",
                           decision_clock_id="entry-close", adapter="json-linear.v1",
                           feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))
    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(artifact_id="m1", role="size", strategy_ids=("*",),
                                      compatible_clock_ids=("entry-close",),
                                      target_contract_ref="return.v1", ordered_features=("x",),
                                      members=(inv_member,))
    inv_binding = ReleaseBinding(role="size", strategy_id="*", clock_id="entry-close",
                                 artifact_id="m1", ordered_features=("x",),
                                 required_member_kinds=("estimator",))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def _spec(member_id):
    return next(spec for spec in phase5_release.STATE_SPECS if spec.member_id == member_id)


def _stage_prior(root, release_id="prior1"):
    release, inventory, payloads = _fixture(release_id)
    deployment.stage_release(phase5_release.deployment_root(root), release, inventory, payloads)
    rows = []
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID, UNRELATED_MEMBER_ID):
        payload = f"prior-{member_id}".encode()
        content_hash, rel = phase5_release.write_object(root, payload)
        obj = {"name": "prior.bin", "path": rel, "content_hash": content_hash, "bytes": len(payload)}
        rows.append(phase5_release.member_row(_spec(member_id), phase5_release.STAGED, [obj], "prior"))
    body = phase5_release.manifest_body(release_id, "d1", rows, {"pool_cutoff": "2026-09-01"})
    phase5_release.write_manifest(root, body)
    return inventory, body


def _carry_kwargs(args):
    return dict(outputs=args["outputs"], chooser=args["chooser"], cutoff=args["cutoff"])


def _valid_derive_args(root, *, cutoff="2026-10-01", prefix="new",
                       expected_prior_release_id="prior1"):
    outputs = []
    for member_id in assembly.POOL_MEMBER_IDS:
        payload = f"{prefix}-{member_id}".encode()
        content_hash, _ = phase5_release.write_object(root, payload)
        outputs.append(assembly.PoolOutput(member_id=member_id, name="pool.bin",
                                           content_hash=content_hash, cutoff=cutoff,
                                           payload_bytes=len(payload)))
    chooser_payload = f"{prefix}-chooser".encode()
    chooser_hash, _ = phase5_release.write_object(root, chooser_payload)
    chooser = assembly.PoolOutput(member_id=assembly.CHOOSER_MEMBER_ID, name="chooser.bin",
                                  content_hash=chooser_hash, cutoff=cutoff,
                                  payload_bytes=len(chooser_payload))
    return dict(release_id="cand1", expected_prior_release_id=expected_prior_release_id,
                outputs=outputs, chooser=chooser, cutoff=cutoff)


def test_happy_path_carries_the_pools_forward(tmp_path):
    inventory, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    prior_path = phase5_release._catalog_path(tmp_path, "prior1")
    prior_bytes = prior_path.read_bytes()
    root_manifest = (tmp_path / "phase5_release.json").read_bytes()

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == phase5_release._catalog_path(tmp_path, "cand1")
    candidate = phase5_release._read_catalog(path, "cand1")
    rows = {row["member_id"]: row for row in candidate["members"]}
    expected = {output.member_id: output.content_hash for output in (*args["outputs"], args["chooser"])}
    for member_id, content_hash in expected.items():
        assert rows[member_id]["status"] == phase5_release.STAGED
        assert rows[member_id]["objects"][0]["content_hash"] == content_hash
        assert rows[member_id]["objects"][0]["path"] == phase5_release.object_relpath(content_hash)
    prior_unrelated = next(row for row in prior_body["members"]
                           if row["member_id"] == UNRELATED_MEMBER_ID)
    assert rows[UNRELATED_MEMBER_ID] == prior_unrelated
    assert candidate["sources"]["pool_cutoff"] == "2026-10-01"
    assert candidate["sources"]["carried_from"] == "prior1"
    assert not (phase5_release.deployment_root(tmp_path) / "DEPLOYED").exists()
    assert prior_path.read_bytes() == prior_bytes
    assert (tmp_path / "phase5_release.json").read_bytes() == root_manifest


def test_derive_is_deterministic_and_carry_forward_is_idempotent(tmp_path):
    inventory, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    assert assembly.derive_catalog(prior_body, **args) == assembly.derive_catalog(prior_body, **args)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    first = assembly.carry_forward_release(tmp_path, prior_release_id="prior1",
                                           release_id="cand1", inventory=inventory, **_carry_kwargs(args))
    first_bytes = first.read_bytes()
    second = assembly.carry_forward_release(tmp_path, prior_release_id="prior1",
                                            release_id="cand1", inventory=inventory, **_carry_kwargs(args))
    assert second == first
    assert second.read_bytes() == first_bytes


def test_prior_mismatch_on_wrong_expected_id(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, expected_prior_release_id="someone-else")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "PRIOR_MISMATCH"


def test_pool_set_mismatch_with_five_outputs(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["outputs"] = args["outputs"][:5]
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "POOL_SET_MISMATCH"


def test_chooser_mismatch(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = dataclasses.replace(args["chooser"], member_id="admissible_table:dyn_sv")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CHOOSER_MISMATCH"


def test_cutoff_mismatch_on_one_output(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], cutoff="2026-09-15")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_MISMATCH"


def test_cutoff_not_advanced_when_equal(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, cutoff="2026-09-01")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_NOT_ADVANCED"


def test_object_unverified_leaves_no_candidate_catalog(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["outputs"][0] = dataclasses.replace(args["outputs"][0],
                                             content_hash="sha256:" + "0" * 64, payload_bytes=0)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    candidate_path = phase5_release._catalog_path(tmp_path, "cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "OBJECT_UNVERIFIED"
    assert not candidate_path.exists()


def test_id_conflict_when_candidate_id_equals_prior(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["release_id"] = "prior1"
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "ID_CONFLICT"


def test_id_conflict_when_candidate_catalog_is_corrupt(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    candidate_path = phase5_release._catalog_path(tmp_path, "cand1")
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_path.write_bytes(b"not a valid catalog")
    garbage = candidate_path.read_bytes()
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "ID_CONFLICT"
    assert candidate_path.read_bytes() == garbage


def test_id_conflict_when_candidate_catalog_exists_with_different_content(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    rows = [phase5_release.member_row(_spec(member_id), phase5_release.STAGED, [], "old")
            for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)]
    different = phase5_release.manifest_body("cand1", "d1", rows, {"pool_cutoff": "2026-09-01"})
    phase5_release._write_catalog(phase5_release._catalog_path(tmp_path, "cand1"), different)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "ID_CONFLICT"
