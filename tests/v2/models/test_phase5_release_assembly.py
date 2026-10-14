"""Pool release assembly: derive/stage a carried-forward candidate catalog (#192 slice 2)."""
from __future__ import annotations

import copy
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


def _candidate_path(root, release_id="cand1"):
    return phase5_release._catalog_path(root, release_id)


def _expected_row(spec, output, detail):
    return {
        "member_id": spec.member_id,
        "kind": spec.kind,
        "strategies": list(spec.strategies),
        "status": phase5_release.STAGED,
        "objects": [{
            "name": output.name,
            "path": phase5_release.object_relpath(output.content_hash),
            "content_hash": output.content_hash,
            "bytes": output.payload_bytes,
        }],
        "detail": detail,
    }


def _expected_catalog(prior_body, args, release_id):
    by_member = {output.member_id: output for output in (*args["outputs"], args["chooser"])}
    rows = [_expected_row(_spec(member_id), by_member[member_id],
                          f"carried forward {args['cutoff']}")
            for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)]
    rows.append(next(row for row in prior_body["members"]
                     if row["member_id"] == UNRELATED_MEMBER_ID))
    sources = {**prior_body["sources"], "pool_cutoff": args["cutoff"],
               "carried_from": args["expected_prior_release_id"]}
    return phase5_release.manifest_body(release_id, prior_body["deployment_id"], rows, sources)


def _mutate_prior(body, mode):
    members = [dict(row) for row in body["members"]]
    sources = dict(body["sources"])
    if mode == "missing-row":
        members = [row for row in members if row["member_id"] != "paired_residual_pool"]
    elif mode == "duplicate-row":
        members = members * 2
    elif mode == "non-staged-row":
        next(r for r in members
             if r["member_id"] == "paired_residual_pool")["status"] = phase5_release.MISSING
    elif mode == "absent-cutoff":
        sources.pop("pool_cutoff")
    elif mode == "invalid-cutoff":
        sources["pool_cutoff"] = "not-a-date"
    return phase5_release.manifest_body(body["release_id"], body["deployment_id"], members, sources)


@pytest.mark.parametrize("release_id", ["cand1", "cand2"])
def test_derive_catalog_matches_independent_oracle(tmp_path, release_id):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["release_id"] = release_id
    snapshot = copy.deepcopy(prior_body)
    derived = assembly.derive_catalog(prior_body, **args)
    assert derived == _expected_catalog(prior_body, args, release_id)
    assert prior_body == snapshot


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
    assert phase5_release._read_catalog(path, "cand1") == _expected_catalog(prior_body, args, "cand1")
    store = phase5_release.deployment_root(tmp_path)
    prior_model = deployment.resolve_release(store, "prior1")
    candidate_model = deployment.resolve_release(store, "cand1")
    assert dataclasses.replace(candidate_model, release_id="prior1") == prior_model
    assert not (store / "DEPLOYED").exists()
    assert prior_path.read_bytes() == prior_bytes
    assert (tmp_path / "phase5_release.json").read_bytes() == root_manifest


def test_carry_forward_is_idempotent(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    first = assembly.carry_forward_release(tmp_path, prior_release_id="prior1",
                                           release_id="cand1", inventory=inventory,
                                           **_carry_kwargs(args))
    first_bytes = first.read_bytes()
    second = assembly.carry_forward_release(tmp_path, prior_release_id="prior1",
                                            release_id="cand1", inventory=inventory,
                                            **_carry_kwargs(args))
    assert second == first
    assert second.read_bytes() == first_bytes


@pytest.mark.parametrize("mode", ["wrong-expected-id", "missing-row", "duplicate-row",
                                  "non-staged-row", "absent-cutoff", "invalid-cutoff"])
def test_prior_mismatch_on_edited_prior_catalog(tmp_path, mode):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    if mode == "wrong-expected-id":
        args["expected_prior_release_id"] = "someone-else"
        body = prior_body
    else:
        body = _mutate_prior(prior_body, mode)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(body, **args)
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("mutate", [
    lambda outputs: outputs[:5],
    lambda outputs: [*outputs, outputs[0]],
    lambda outputs: [dataclasses.replace(outputs[0], content_hash="sha256:nothex"), *outputs[1:]],
], ids=["five-outputs", "duplicate", "malformed-hash"])
def test_pool_set_mismatch(tmp_path, mutate):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["outputs"] = mutate(args["outputs"])
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "POOL_SET_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("mutate", [
    lambda chooser: dataclasses.replace(chooser, member_id="admissible_table:dyn_sv"),
    lambda chooser: "not-a-pool-output",
], ids=["wrong-member", "not-a-pool-output"])
def test_chooser_mismatch(tmp_path, mutate):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = mutate(args["chooser"])
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CHOOSER_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("kind", ["output", "chooser", "invalid-date"])
def test_cutoff_mismatch(tmp_path, kind):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, cutoff="not-a-date" if kind == "invalid-date" else "2026-10-01")
    if kind == "output":
        args["outputs"][0] = dataclasses.replace(args["outputs"][0], cutoff="2026-09-15")
    elif kind == "chooser":
        args["chooser"] = dataclasses.replace(args["chooser"], cutoff="2026-09-15")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("cutoff", ["2026-09-01", "2026-08-15"])
def test_cutoff_not_advanced(tmp_path, cutoff):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, cutoff=cutoff)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_NOT_ADVANCED"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("mode", ["missing", "hash", "length"])
def test_object_unverified_leaves_no_candidate_catalog(tmp_path, mode):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    output = args["outputs"][0]
    if mode == "missing":
        args["outputs"][0] = dataclasses.replace(output, content_hash="sha256:" + "0" * 64,
                                                 payload_bytes=0)
    elif mode == "hash":
        (phase5_release.deployment_root(tmp_path)
         / phase5_release.object_relpath(output.content_hash)).write_bytes(b"x" * output.payload_bytes)
    else:
        args["outputs"][0] = dataclasses.replace(output, payload_bytes=output.payload_bytes + 1)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "OBJECT_UNVERIFIED"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("release_id", ["prior1", "", "a/b"])
def test_id_conflict_on_bad_candidate_id(tmp_path, release_id):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["release_id"] = release_id
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "ID_CONFLICT"
    assert not _candidate_path(tmp_path).exists()


def test_id_conflict_when_inventory_release_id_mismatches(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="someone-else")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "ID_CONFLICT"
    assert not _candidate_path(tmp_path).exists()


def _corrupt_candidate_catalog(root):
    path = _candidate_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid catalog")


def _different_candidate_catalog(root):
    rows = [phase5_release.member_row(_spec(member_id), phase5_release.STAGED, [], "old")
            for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)]
    phase5_release._write_catalog(_candidate_path(root),
                                  phase5_release.manifest_body("cand1", "d1", rows,
                                                               {"pool_cutoff": "2026-09-01"}))


def _stage_other_candidate_release(root):
    release, other_inventory, payloads = _fixture("cand1")
    deployment.stage_release(phase5_release.deployment_root(root),
                             dataclasses.replace(release, deployment_id="d2"),
                             dataclasses.replace(other_inventory, deployment_id="d2"), payloads)


@pytest.mark.parametrize("setup, no_catalog", [
    (_corrupt_candidate_catalog, False),
    (_different_candidate_catalog, False),
    (_stage_other_candidate_release, True),
], ids=["corrupt-catalog", "different-catalog", "id-already-staged"])
def test_id_conflict_when_candidate_already_exists(tmp_path, setup, no_catalog):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    setup(tmp_path)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=dataclasses.replace(inventory, release_id="cand1"),
                                       **_carry_kwargs(args))
    assert error.value.code == "ID_CONFLICT"
    if no_catalog:
        assert not _candidate_path(tmp_path).exists()


def test_incomplete_inventory_on_retry_is_refused(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    published = path.read_bytes()
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=dataclasses.replace(inventory, artifacts=()),
                                       **_carry_kwargs(args))
    assert error.value.code == "INVENTORY_MISMATCH"
    assert path.read_bytes() == published


def test_retry_restages_a_deleted_candidate_manifest(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    published = path.read_bytes()
    deployment._manifest_path(phase5_release.deployment_root(tmp_path), "cand1").unlink()
    again = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
    assert again == path
    assert again.read_bytes() == published


def test_prior_mismatch_when_prior_model_object_is_deleted(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    prior = deployment.resolve_release(store, "prior1")
    (store / prior.bindings[0].members[0].path).unlink()
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_id_conflict_when_candidate_content_changes_on_retry(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    published = path.read_bytes()
    other = _valid_derive_args(tmp_path, prefix="other", cutoff="2026-10-05")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(other))
    assert error.value.code == "ID_CONFLICT"
    assert path.read_bytes() == published
