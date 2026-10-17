"""Pool release assembly: derive/stage a carried-forward candidate catalog (#192 slice 2)."""
from __future__ import annotations

import builtins
import copy
import dataclasses
import fcntl
import json
import os
import stat
import sys
import threading
import time

import pytest

from checks import phase5_release_assembly as assembly
from engine.v2.models import release_layout as phase5_release
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


def _fixture(release_id, deployment_id="d1", binding_id="b1"):
    payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                          "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}]},
                         sort_keys=True).encode()
    member_hash = phase5_release.sha256_bytes(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(binding_id=binding_id, model_id="m1", role="size", strategy_id="*",
                           decision_clock_id="entry-close", adapter="json-linear.v1",
                           feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id=deployment_id, bindings=(binding,))
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
        release_id=release_id, deployment_id=deployment_id, known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def _spec(member_id):
    return next(spec for spec in phase5_release.STATE_SPECS if spec.member_id == member_id)


def _stage_prior(root, release_id="prior1", deployment_id="d1", binding_id="b1",
                 prior_names=None, prior_sources=None):
    release, inventory, payloads = _fixture(release_id, deployment_id=deployment_id,
                                            binding_id=binding_id)
    deployment.stage_release(phase5_release.deployment_root(root), release, inventory, payloads)
    rows = []
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID, UNRELATED_MEMBER_ID):
        payload = f"prior-{member_id}".encode()
        content_hash, rel = phase5_release.write_object(root, payload)
        name = "prior.bin" if prior_names is None else prior_names[member_id]
        obj = {"name": name, "path": rel, "content_hash": content_hash, "bytes": len(payload)}
        rows.append(phase5_release.member_row(_spec(member_id), phase5_release.STAGED, [obj], "prior"))
    sources = {"pool_cutoff": "2026-09-01"} if prior_sources is None else prior_sources
    body = phase5_release.manifest_body(release_id, deployment_id, rows, sources)
    phase5_release.write_manifest(root, body)
    return inventory, body


def _carry_kwargs(args):
    return dict(outputs=args["outputs"], chooser=args["chooser"], cutoff=args["cutoff"])


def _valid_derive_args(root, *, cutoff="2026-10-01", prefix="new",
                       expected_prior_release_id="prior1", output_names=None,
                       chooser_name="chooser.bin"):
    outputs = []
    for member_id in assembly.POOL_MEMBER_IDS:
        payload = f"{prefix}-{member_id}".encode()
        content_hash, _ = phase5_release.write_object(root, payload)
        name = "pool.bin" if output_names is None else output_names[member_id]
        outputs.append(assembly.PoolOutput(member_id=member_id, name=name,
                                           content_hash=content_hash, cutoff=cutoff,
                                           payload_bytes=len(payload)))
    chooser_payload = f"{prefix}-chooser".encode()
    chooser_hash, _ = phase5_release.write_object(root, chooser_payload)
    chooser = assembly.PoolOutput(member_id=assembly.CHOOSER_MEMBER_ID, name=chooser_name,
                                  content_hash=chooser_hash, cutoff=cutoff,
                                  payload_bytes=len(chooser_payload))
    return dict(release_id="cand1", expected_prior_release_id=expected_prior_release_id,
                outputs=outputs, chooser=chooser, cutoff=cutoff)


def _candidate_path(root, release_id="cand1"):
    return phase5_release.catalog_path(root, release_id)


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


@pytest.mark.parametrize("deployment_id, binding_id", [("d1", "b1"), ("deployZ", "bindZ")])
def test_happy_path_carries_the_pools_forward(tmp_path, deployment_id, binding_id):
    inventory, prior_body = _stage_prior(tmp_path, deployment_id=deployment_id, binding_id=binding_id)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    prior_path = phase5_release.catalog_path(tmp_path, "prior1")
    prior_bytes = prior_path.read_bytes()
    root_manifest = (tmp_path / "phase5_release.json").read_bytes()

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == phase5_release.catalog_path(tmp_path, "cand1")
    catalog = phase5_release.read_catalog(path, "cand1")
    assert catalog == _expected_catalog(prior_body, args, "cand1")
    assert catalog["deployment_id"] == deployment_id
    store = phase5_release.deployment_root(tmp_path)
    prior_model = deployment.resolve_release(store, "prior1")
    candidate_model = deployment.resolve_release(store, "cand1")
    assert dataclasses.replace(candidate_model, release_id="prior1") == prior_model
    assert candidate_model.deployment_id == catalog["deployment_id"]
    assert prior_model.deployment_id == deployment_id
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


def _replace_output(args, **changes):
    outputs = list(args["outputs"])
    outputs[0] = dataclasses.replace(outputs[0], **changes)
    return {**args, "outputs": outputs}


def _replace_chooser(args, **changes):
    return {**args, "chooser": dataclasses.replace(args["chooser"], **changes)}


def _output_with(args, target_member, **changes):
    return {**args, "outputs": [dataclasses.replace(output, **changes)
                                if output.member_id == target_member else output
                                for output in args["outputs"]]}


_POOL_REFUSALS = [
    (lambda a: {**a, "outputs": "not-a-sequence"}, "POOL_SET_MISMATCH"),
    (lambda a: {**a, "outputs": a["outputs"][:5]}, "POOL_SET_MISMATCH"),
    (lambda a: {**a, "outputs": [*a["outputs"], a["outputs"][0]]}, "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, content_hash="sha256:nothex"), "POOL_SET_MISMATCH"),
    (lambda a: {**a, "outputs": [*a["outputs"][:5], "not-an-output"]}, "POOL_SET_MISMATCH"),
    (lambda a: {**a, "outputs": [*a["outputs"], "not-an-output"]}, "POOL_SET_MISMATCH"),
    (lambda a: {**a, "chooser": [a["chooser"], "not-an-output"]}, "CHOOSER_MISMATCH"),
    (lambda a: _replace_output(a, member_id=5), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, member_id=["x"]), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, name=""), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, payload_bytes=-1), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, payload_bytes=True), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, payload_bytes="12"), "POOL_SET_MISMATCH"),
    (lambda a: _replace_chooser(a, member_id="admissible_table:dyn_sv"), "CHOOSER_MISMATCH"),
    (lambda a: {**a, "chooser": "not-a-pool-output"}, "CHOOSER_MISMATCH"),
    (lambda a: _replace_chooser(a, content_hash="sha256:nothex"), "CHOOSER_MISMATCH"),
    (lambda a: _replace_chooser(a, name=""), "CHOOSER_MISMATCH"),
    (lambda a: _replace_chooser(a, payload_bytes=-1), "CHOOSER_MISMATCH"),
    (lambda a: _output_with(a, "trailing_pnl_cutoff", member_id="bogus"), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, content_hash="sha256:" + "g" * 64), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, content_hash="sha256:" + "A" * 64), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, name=5), "POOL_SET_MISMATCH"),
]


@pytest.mark.parametrize("mutate, code", _POOL_REFUSALS, ids=[
    "outputs-not-a-sequence", "five-outputs", "duplicate", "malformed-hash", "element-not-a-pool-output",
    "extra-output-element", "chooser-list-with-junk",
    "member-id-int", "member-id-list", "empty-name", "negative-bytes", "bool-bytes", "str-bytes",
    "chooser-wrong-member", "chooser-not-a-pool-output", "chooser-bad-hash", "chooser-empty-name",
    "chooser-bad-bytes", "wrong-member-id", "nonhex-hash-g", "uppercase-hash", "non-string-name"])
def test_derive_catalog_rejects_malformed_outputs(tmp_path, mutate, code):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **mutate(args))
    assert error.value.code == code
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


@pytest.mark.parametrize("cutoff", ["20261001", "2026-W41-3"])
def test_cutoff_not_extended_iso_date_is_cutoff_mismatch(tmp_path, cutoff):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, cutoff=cutoff)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_prior_pool_cutoff_not_extended_iso_date_is_prior_mismatch(tmp_path):
    _, prior_body = _stage_prior(tmp_path, prior_sources={"pool_cutoff": "20260101"})
    args = _valid_derive_args(tmp_path)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("cutoff", ["2026-09-01", "2026-08-15"])
def test_cutoff_not_advanced(tmp_path, cutoff):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, cutoff=cutoff)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_NOT_ADVANCED"
    store = phase5_release.deployment_root(tmp_path)
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(store, "cand1").exists()


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


@pytest.mark.parametrize("release_id", ["prior1", "", "a/b", ".hidden", "a\\b", "a\x00b"])
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
    phase5_release.write_catalog(_candidate_path(root),
                                  phase5_release.manifest_body("cand1", "d1", rows,
                                                               {"pool_cutoff": "2026-09-01"}))


def _stage_other_candidate_release(root):
    release, other_inventory, payloads = _fixture("cand1")
    deployment.stage_release(phase5_release.deployment_root(root),
                             dataclasses.replace(release, deployment_id="d2"),
                             dataclasses.replace(other_inventory, deployment_id="d2"), payloads)


@pytest.mark.parametrize("setup, no_catalog, staged_absent", [
    (_corrupt_candidate_catalog, False, True),
    (_different_candidate_catalog, False, True),
    (_stage_other_candidate_release, True, False),
], ids=["corrupt-catalog", "different-catalog", "id-already-staged"])
def test_id_conflict_when_candidate_already_exists(tmp_path, setup, no_catalog, staged_absent):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    setup(tmp_path)
    catalog_path = _candidate_path(tmp_path)
    existing_bytes = catalog_path.read_bytes() if catalog_path.exists() else None
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=dataclasses.replace(inventory, release_id="cand1"),
                                       **_carry_kwargs(args))
    assert error.value.code == "ID_CONFLICT"
    if no_catalog:
        assert not catalog_path.exists()
    else:
        assert catalog_path.read_bytes() == existing_bytes
    if staged_absent:
        store = phase5_release.deployment_root(tmp_path)
        assert not phase5_release.staged_manifest_path(store, "cand1").exists()


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
    store = phase5_release.deployment_root(tmp_path)
    phase5_release.staged_manifest_path(store, "cand1").unlink()
    again = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
    assert again == path
    assert again.read_bytes() == published
    resolved = deployment.resolve_release(store, "cand1")
    assert dataclasses.replace(resolved, release_id="prior1") == deployment.resolve_release(store, "prior1")


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


def _gate(real_stage, entered, proceed, first):
    def wrapper(*args, **kwargs):
        if not first["done"]:
            first["done"] = True
            entered.set()
            assert proceed.wait(10)
        return real_stage(*args, **kwargs)
    return wrapper


def test_publication_lock_serialises_concurrent_same_candidate(tmp_path, monkeypatch):
    inventory, prior_body = _stage_prior(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    args_x = _valid_derive_args(tmp_path, prefix="x")
    args_y = _valid_derive_args(tmp_path, prefix="y", cutoff="2026-10-02")
    entered = threading.Event()
    b_started = threading.Event()
    proceed, first = threading.Event(), {"done": False}
    real_stage = assembly.deployment.stage_release
    monkeypatch.setattr(assembly.deployment, "stage_release", _gate(real_stage, entered, proceed, first))
    results = {}

    def run(tag, kwargs):
        if tag == "b":
            b_started.set()
        try:
            results[tag] = assembly.carry_forward_release(
                tmp_path, prior_release_id="prior1", release_id="cand1",
                inventory=inventory, **_carry_kwargs(kwargs))
        except BaseException as exc:
            results[tag] = exc

    a = threading.Thread(target=run, args=("a", args_x))
    a.start()
    assert entered.wait(10)
    b = threading.Thread(target=run, args=("b", args_y))
    b.start()
    deadline = time.monotonic() + 5
    while not b_started.is_set() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert b_started.is_set()
    hold_until = time.monotonic() + 0.2
    while time.monotonic() < hold_until:
        assert b.is_alive()
        time.sleep(0.01)
    proceed.set()
    a.join(20)
    b.join(20)
    assert not a.is_alive() and not b.is_alive()
    assert results["a"] == _candidate_path(tmp_path)
    assert isinstance(results["b"], assembly.ReleaseAssemblyRefusal)
    assert results["b"].code == "ID_CONFLICT"
    assert phase5_release.read_catalog(_candidate_path(tmp_path), "cand1") == _expected_catalog(
        prior_body, args_x, "cand1")


def _corrupt_candidate_manifest(root, args, monkeypatch):
    _stage_other_candidate_release(root)
    store = phase5_release.deployment_root(root)
    phase5_release.staged_manifest_path(store, "cand1").write_bytes(b"garbage manifest")


def _corrupt_prior_payload(root, args, monkeypatch):
    store = phase5_release.deployment_root(root)
    prior = deployment.resolve_release(store, "prior1")
    (store / prior.bindings[0].members[0].path).write_bytes(b"tampered bytes")


def _lock_is_directory(root, args, monkeypatch):
    (phase5_release.deployment_root(root) / ".carry_forward.lock").mkdir()


def _staging_oserror(root, args, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(assembly.deployment, "stage_release", boom)


def _write_catalog_oserror(root, args, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(assembly, "write_catalog", boom)


def _symlink_supplied_object(root, args, monkeypatch):
    store = phase5_release.deployment_root(root)
    path = store / phase5_release.object_relpath(args["outputs"][0].content_hash)
    original = path.read_bytes()
    target = root / "valid-object.bin"
    target.write_bytes(original)
    path.unlink()
    path.symlink_to(target)


def _symlink_prior_payload(root, args, monkeypatch):
    store = phase5_release.deployment_root(root)
    member_path = store / deployment.resolve_release(store, "prior1").bindings[0].members[0].path
    original = member_path.read_bytes()
    target = root / "prior-copy.bin"
    target.write_bytes(original)
    member_path.unlink()
    member_path.symlink_to(target)


@pytest.mark.parametrize("setup, code", [
    (_corrupt_candidate_manifest, "ID_CONFLICT"),
    (_corrupt_prior_payload, "PRIOR_MISMATCH"),
    (_lock_is_directory, "PUBLICATION_FAILED"),
    (_staging_oserror, "PUBLICATION_FAILED"),
    (_write_catalog_oserror, "PUBLICATION_FAILED"),
    (_symlink_supplied_object, "OBJECT_UNVERIFIED"),
    (_symlink_prior_payload, "PRIOR_MISMATCH"),
], ids=["corrupt-manifest", "corrupt-prior-payload", "lock-is-directory", "staging-oserror",
        "write-catalog-oserror", "supplied-object-symlink", "prior-payload-symlink"])
def test_refusal_classification(tmp_path, setup, code, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    setup(tmp_path, args, monkeypatch)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == code
    assert not _candidate_path(tmp_path).exists()


def test_inventory_none_is_inventory_mismatch(tmp_path):
    _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=None, **_carry_kwargs(args))
    assert error.value.code == "INVENTORY_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_inventory_mismatch_on_different_deployment_id(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1", deployment_id="d2")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "INVENTORY_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_inventory_with_none_artifacts_is_inventory_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1", artifacts=None)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "INVENTORY_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_inventory_without_release_id_is_inventory_mismatch(tmp_path):
    _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=object(), **_carry_kwargs(args))
    assert error.value.code == "INVENTORY_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_derive_catalog_carries_distinct_identities_and_names(tmp_path):
    prior_names = {member_id: f"prior-{member_id}.bin"
                   for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID,
                                     UNRELATED_MEMBER_ID)}
    prior_sources = {"pool_cutoff": "2026-09-01", "panel_sha": "x", "note": "y"}
    _, prior_body = _stage_prior(tmp_path, release_id="priorX", deployment_id="dX",
                                 prior_names=prior_names, prior_sources=prior_sources)
    output_names = {member_id: f"pool-{member_id}.bin" for member_id in assembly.POOL_MEMBER_IDS}
    args = _valid_derive_args(tmp_path, expected_prior_release_id="priorX",
                              output_names=output_names, chooser_name="chooser-X.bin")
    derived = assembly.derive_catalog(prior_body, **args)
    assert derived["deployment_id"] == "dX"
    assert derived["sources"] == {**prior_body["sources"], "pool_cutoff": args["cutoff"],
                                  "carried_from": "priorX"}
    assert derived == _expected_catalog(prior_body, args, "cand1")


def test_prior_mismatch_when_prior_catalog_file_is_missing(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").unlink()
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_prior_mismatch_when_prior_staged_manifest_is_deleted(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    phase5_release.staged_manifest_path(store, "prior1").unlink()
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(store, "cand1").exists()


def _assert_prior_mismatch_on_retry(tmp_path, inventory, args, store):
    for _ in range(2):
        with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
            assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
        assert error.value.code == "PRIOR_MISMATCH"
        assert not _candidate_path(tmp_path).exists()
        assert not phase5_release.staged_manifest_path(store, "cand1").exists()


def test_prior_mismatch_when_prior_manifest_hash_is_invalid(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    path = phase5_release.staged_manifest_path(store, "prior1")
    manifest = json.loads(path.read_bytes())
    manifest["release_hash"] = "sha256:" + "0" * 64
    path.write_bytes(json.dumps(manifest).encode())
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _assert_prior_mismatch_on_retry(tmp_path, inventory, args, store)


def test_prior_mismatch_when_prior_manifest_declares_different_release(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    other_release, other_inventory, other_payloads = _fixture("other1")
    deployment.stage_release(store, other_release, other_inventory, other_payloads)
    phase5_release.staged_manifest_path(store, "prior1").write_bytes(
        phase5_release.staged_manifest_path(store, "other1").read_bytes())
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _assert_prior_mismatch_on_retry(tmp_path, inventory, args, store)


def test_prior_mismatch_when_prior_catalog_deployment_id_differs(tmp_path):
    inventory, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    body = phase5_release.manifest_body(prior_body["release_id"], "dX",
                                        prior_body["members"], prior_body["sources"])
    phase5_release.write_catalog(phase5_release.catalog_path(tmp_path, "prior1"), body)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(phase5_release.deployment_root(tmp_path),
                                         "cand1").exists()


def test_carry_forward_leaves_existing_deployed_pointer_untouched(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    pointer = phase5_release.deployment_root(tmp_path) / "DEPLOYED"
    pointer.write_bytes(b'{"x":1}')

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == phase5_release.catalog_path(tmp_path, "cand1")
    assert pointer.read_bytes() == b'{"x":1}'


@pytest.mark.parametrize("value", [None, ""])
def test_derive_catalog_requires_non_empty_expected_prior_id(tmp_path, value):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["expected_prior_release_id"] = value
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("value", [None, ""])
def test_carry_forward_requires_non_empty_prior_release_id(tmp_path, value):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id=value, release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


_REQUIRED_MEMBER_IDS = (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)


def _prior_row(prior_body, member_id):
    return next(row for row in prior_body["members"] if row["member_id"] == member_id)


def _rebuilt_prior(prior_body, members):
    return phase5_release.manifest_body(prior_body["release_id"], prior_body["deployment_id"],
                                        members, prior_body["sources"])


@pytest.mark.parametrize("mode", ["missing", "duplicate", "missing-status"])
@pytest.mark.parametrize("member_id", _REQUIRED_MEMBER_IDS)
def test_prior_mismatch_for_each_required_member(tmp_path, member_id, mode):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    members = [dict(row) for row in prior_body["members"]]
    if mode == "missing":
        members = [row for row in members if row["member_id"] != member_id]
    elif mode == "duplicate":
        members.append(dict(_prior_row(prior_body, member_id)))
    else:
        next(row for row in members if row["member_id"] == member_id)["status"] = phase5_release.MISSING
    body = _rebuilt_prior(prior_body, members)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(body, **args)
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


@pytest.mark.parametrize("member_id", _REQUIRED_MEMBER_IDS)
def test_cutoff_mismatch_for_each_required_member(tmp_path, member_id):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    if member_id == assembly.CHOOSER_MEMBER_ID:
        args["chooser"] = dataclasses.replace(args["chooser"], cutoff="2026-09-15")
    else:
        args["outputs"] = [dataclasses.replace(output, cutoff="2026-09-15")
                           if output.member_id == member_id else output
                           for output in args["outputs"]]
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


_RICH_BINDINGS = (("b1", "m1", "size"), ("b2", "m2", "implied_t1"))
_RICH_MEMBER_KINDS = ("estimator", "calibration")


def _rich_fixture(release_id, deployment_id="d1"):
    payloads, bindings, artifacts, inv_bindings, requirements = {}, [], [], [], []
    for binding_id, model_id, role in _RICH_BINDINGS:
        members, inv_members = [], []
        for name in _RICH_MEMBER_KINDS:
            payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                                  "model": model_id, "member": name}, sort_keys=True).encode()
            content_hash = phase5_release.sha256_bytes(payload)
            payloads[content_hash] = payload
            members.append(ArtifactMember(name=name, path="unused.json", content_hash=content_hash))
            inv_members.append(ArtifactInventoryMember(member_id=f"{model_id}:{name}", kind=name,
                                                       artifact_ref=f"artifact://{model_id}",
                                                       content_hash=content_hash))
        bindings.append(ModelBinding(binding_id=binding_id, model_id=model_id, role=role,
                                     strategy_id="*", decision_clock_id="entry-close",
                                     adapter="json-linear.v1", feature_order=("x",),
                                     output_names=("prediction",), members=tuple(members)))
        artifacts.append(ModelArtifactInventory(artifact_id=model_id, role=role, strategy_ids=("*",),
                                                compatible_clock_ids=("entry-close",),
                                                target_contract_ref="return.v1",
                                                ordered_features=("x",), members=tuple(inv_members)))
        inv_bindings.append(ReleaseBinding(role=role, strategy_id="*", clock_id="entry-close",
                                           artifact_id=model_id, ordered_features=("x",),
                                           required_member_kinds=_RICH_MEMBER_KINDS))
        requirements.append(ReleaseRequirement(role=role, strategy_id="*", clock_id="entry-close"))
    release = ModelRelease(release_id=release_id, deployment_id=deployment_id,
                           bindings=tuple(bindings))
    inventory = ModelReleaseInventory(release_id=release_id, deployment_id=deployment_id,
                                      known_clock_ids=("entry-close",), artifacts=tuple(artifacts),
                                      bindings=tuple(inv_bindings), requirements=tuple(requirements),
                                      artifact_manifest_ref="manifest://r",
                                      evidence_refs=("evidence://r",))
    return release, inventory, payloads


def _unrelated_ids(count=3):
    return tuple(spec.member_id for spec in phase5_release.STATE_SPECS
                 if spec.member_id not in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID,
                                           UNRELATED_MEMBER_ID))[:count]


def _stage_rich_prior(root, *, release_id="prior1", deployment_id="d1",
                      prior_cutoff="2026-09-01", extra_ids=()):
    release, inventory, payloads = _rich_fixture(release_id, deployment_id=deployment_id)
    deployment.stage_release(phase5_release.deployment_root(root), release, inventory, payloads)
    rows = []
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID, *extra_ids):
        payload = f"prior-{member_id}".encode()
        content_hash, rel = phase5_release.write_object(root, payload)
        obj = {"name": "prior.bin", "path": rel, "content_hash": content_hash, "bytes": len(payload)}
        rows.append(phase5_release.member_row(_spec(member_id), phase5_release.STAGED, [obj], "prior"))
    body = phase5_release.manifest_body(release_id, deployment_id, rows, {"pool_cutoff": prior_cutoff})
    phase5_release.write_manifest(root, body)
    return inventory, body


@pytest.mark.parametrize("release_id", ["candA", "candB"])
@pytest.mark.parametrize("prior_cutoff, cutoff", [("2026-08-20", "2026-10-01"),
                                                  ("2026-07-15", "2026-12-31")])
def test_rich_carry_forward_preserves_untouched_rows_and_release(tmp_path, release_id, prior_cutoff,
                                                                 cutoff):
    inventory, prior_body = _stage_rich_prior(tmp_path, prior_cutoff=prior_cutoff,
                                              extra_ids=_unrelated_ids())
    args = _valid_derive_args(tmp_path, cutoff=cutoff, prefix="rich")
    args["release_id"] = release_id
    inventory = dataclasses.replace(inventory, release_id=release_id)
    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id=release_id,
                                          inventory=inventory, **_carry_kwargs(args))
    catalog = phase5_release.read_catalog(path, release_id)
    carrier = (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)
    assert catalog["sources"]["pool_cutoff"] == cutoff
    assert catalog["sources"]["carried_from"] == "prior1"
    supplied = _supplied_by_member(args)
    for member_id in carrier:
        expected = _expected_row(_spec(member_id), supplied[member_id],
                                 f"carried forward {cutoff}")
        assert next(r for r in catalog["members"] if r["member_id"] == member_id) == expected
    for row in prior_body["members"]:
        if row["member_id"] not in carrier:
            assert next(r for r in catalog["members"]
                        if r["member_id"] == row["member_id"]) == row
    store = phase5_release.deployment_root(tmp_path)
    prior_model = deployment.resolve_release(store, "prior1")
    candidate_model = deployment.resolve_release(store, release_id)
    assert dataclasses.replace(candidate_model, release_id="prior1") == prior_model


def _object_path(root, target):
    return phase5_release.deployment_root(root) / phase5_release.object_relpath(target.content_hash)


def _supplied_by_member(args):
    return {obj.member_id: obj for obj in (*args["outputs"], args["chooser"])}


@pytest.mark.parametrize("member_id", (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID))
def test_object_preflight_rejects_each_supplied_object(tmp_path, member_id):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    target = _supplied_by_member(args)[member_id]
    _object_path(tmp_path, target).write_bytes(b"x" * max(target.payload_bytes, 1))
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "OBJECT_UNVERIFIED"
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(phase5_release.deployment_root(tmp_path),
                                         "cand1").exists()


@pytest.mark.parametrize("member_id", [assembly.POOL_MEMBER_IDS[0], assembly.CHOOSER_MEMBER_ID])
def test_object_preflight_after_publication_preserves_catalog(tmp_path, member_id):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    published = path.read_bytes()
    obj_path = _object_path(tmp_path, _supplied_by_member(args)[member_id])
    original = obj_path.read_bytes()
    obj_path.write_bytes(b"damaged:" + original)
    try:
        with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
            assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
        assert error.value.code == "OBJECT_UNVERIFIED"
        assert path.read_bytes() == published
    finally:
        obj_path.write_bytes(original)


def test_inventory_mismatch_on_feature_order_disagreement(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    artifact = dataclasses.replace(inventory.artifacts[0], ordered_features=("x", "y"))
    binding = dataclasses.replace(inventory.bindings[0], ordered_features=("x", "y"))
    inventory = dataclasses.replace(inventory, release_id="cand1", artifacts=(artifact,),
                                    bindings=(binding,))
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "INVENTORY_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_republish_after_write_catalog_oserror(tmp_path, monkeypatch):
    inventory, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")

    def boom(*a, **k):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(assembly, "write_catalog", boom)
        with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
            assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
        assert error.value.code == "PUBLICATION_FAILED"

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == _candidate_path(tmp_path)
    assert phase5_release.read_catalog(path, "cand1") == _expected_catalog(prior_body, args, "cand1")
    store = phase5_release.deployment_root(tmp_path)
    resolved = deployment.resolve_release(store, "cand1")
    assert dataclasses.replace(resolved, release_id="prior1") == deployment.resolve_release(store,
                                                                                            "prior1")


def test_carry_forward_prefers_named_prior_over_staged_decoy(tmp_path):
    chosen_id = "release-2026-09-a"
    inventory, chosen_body = _stage_prior(tmp_path, release_id=chosen_id, deployment_id="d1",
                                          binding_id="b1")
    decoy_names = {member_id: "decoy.bin"
                   for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID,
                                     UNRELATED_MEMBER_ID)}
    _, decoy_body = _stage_prior(tmp_path, release_id="prior1", deployment_id="d9",
                                 binding_id="b9", prior_names=decoy_names,
                                 prior_sources={"pool_cutoff": "2026-08-01"})
    args = _valid_derive_args(tmp_path, expected_prior_release_id=chosen_id)
    inventory = dataclasses.replace(inventory, release_id="cand1")

    path = assembly.carry_forward_release(tmp_path, prior_release_id=chosen_id, release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    first_bytes = path.read_bytes()
    catalog = phase5_release.read_catalog(path, "cand1")
    assert catalog["sources"]["carried_from"] == chosen_id
    assert catalog["deployment_id"] == "d1"
    assert _prior_row(catalog, UNRELATED_MEMBER_ID) == _prior_row(chosen_body, UNRELATED_MEMBER_ID)
    assert _prior_row(catalog, UNRELATED_MEMBER_ID) != _prior_row(decoy_body, UNRELATED_MEMBER_ID)

    store = phase5_release.deployment_root(tmp_path)
    resolved = deployment.resolve_release(store, "cand1")
    chosen_model = deployment.resolve_release(store, chosen_id)
    decoy_model = deployment.resolve_release(store, "prior1")
    assert dataclasses.replace(resolved, release_id=chosen_id) == chosen_model
    assert dataclasses.replace(resolved, release_id=chosen_id) != decoy_model

    again = assembly.carry_forward_release(tmp_path, prior_release_id=chosen_id, release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
    assert again == path
    assert again.read_bytes() == first_bytes


def _deny_reads_of(monkeypatch, target):
    real = os.open

    def guarded(path, flags, *args, **kwargs):
        if str(path) == str(target):
            raise PermissionError("denied")
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(assembly.os, "open", guarded)


_READ_FAILURE_CASES = (
    ("supplied-pool", assembly.POOL_MEMBER_IDS[0], "OBJECT_UNVERIFIED"),
    ("chooser", assembly.CHOOSER_MEMBER_ID, "OBJECT_UNVERIFIED"),
    ("prior-model", None, "PRIOR_MISMATCH"),
)


@pytest.mark.parametrize("label, member_id, code", _READ_FAILURE_CASES,
                         ids=[case[0] for case in _READ_FAILURE_CASES])
def test_read_failure_classification(tmp_path, monkeypatch, label, member_id, code):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    if label == "prior-model":
        prior = deployment.resolve_release(store, "prior1")
        target = store / prior.bindings[0].members[0].path
    else:
        target = _object_path(tmp_path, _supplied_by_member(args)[member_id])
    _deny_reads_of(monkeypatch, target)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == code
    if label == "prior-model":
        assert "a prior model object is missing or unreadable" in str(error.value)
    else:
        assert "unreadable" in str(error.value)
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(store, "cand1").exists()


_POOL_OBJECT_ORDER = (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)
_PRIOR_MULTI_COUNTS = dict(zip(_POOL_OBJECT_ORDER, (2, 2, 2, 2, 4, 3, 2)))
_SUPPLIED_MULTI_COUNTS = dict(zip(_POOL_OBJECT_ORDER, (3, 3, 3, 2, 2, 4, 2)))


def _stage_prior_with_multi_objects(root, *, release_id="prior1", deployment_id="d1",
                                    prior_cutoff="2026-09-01"):
    inventory, _ = _stage_prior(root, release_id=release_id, deployment_id=deployment_id)
    catalog = phase5_release.read_catalog(phase5_release.catalog_path(root, release_id), release_id)
    members = []
    for row in catalog["members"]:
        member_id = row["member_id"]
        count = _PRIOR_MULTI_COUNTS.get(member_id, 0)
        if not count:
            members.append(row)
            continue
        objects = []
        for index in range(count):
            payload = f"prior-{member_id}-{index}".encode()
            content_hash, rel = phase5_release.write_object(root, payload)
            objects.append({"name": f"prior-{member_id}-{index}.bin", "path": rel,
                            "content_hash": content_hash, "bytes": len(payload)})
        members.append(phase5_release.member_row(_spec(member_id), phase5_release.STAGED,
                                                 objects, "prior"))
    body = phase5_release.manifest_body(catalog["release_id"], catalog["deployment_id"], members,
                                        {**catalog["sources"], "pool_cutoff": prior_cutoff})
    phase5_release.write_manifest(root, body)
    return inventory, body


def _multi_objects_args(root, *, cutoff="2026-10-01", prefix="multi"):
    outputs = []
    for member_id in assembly.POOL_MEMBER_IDS:
        for index in range(_SUPPLIED_MULTI_COUNTS[member_id]):
            payload = f"{prefix}-{member_id}-{index}".encode()
            content_hash, _ = phase5_release.write_object(root, payload)
            outputs.append(assembly.PoolOutput(member_id=member_id,
                                               name=f"{prefix}-{member_id}-{index}.bin",
                                               content_hash=content_hash, cutoff=cutoff,
                                               payload_bytes=len(payload)))
    chooser = []
    for index in range(_SUPPLIED_MULTI_COUNTS[assembly.CHOOSER_MEMBER_ID]):
        payload = f"{prefix}-chooser-{index}".encode()
        content_hash, _ = phase5_release.write_object(root, payload)
        chooser.append(assembly.PoolOutput(member_id=assembly.CHOOSER_MEMBER_ID,
                                           name=f"{prefix}-chooser-{index}.bin",
                                           content_hash=content_hash, cutoff=cutoff,
                                           payload_bytes=len(payload)))
    return dict(release_id="cand1", expected_prior_release_id="prior1",
                outputs=outputs, chooser=chooser, cutoff=cutoff)


def _supplied_by_member_multi(args):
    by_member = {}
    for obj in (*args["outputs"], *args["chooser"]):
        by_member.setdefault(obj.member_id, []).append(obj)
    return by_member


def _objects_of(outputs):
    return sorted(
        [{"name": obj.name, "path": phase5_release.object_relpath(obj.content_hash),
          "content_hash": obj.content_hash, "bytes": obj.payload_bytes} for obj in outputs],
        key=lambda entry: entry["name"])


@pytest.mark.parametrize("derive", [True, False],
                         ids=["derive_catalog", "carry_forward_release"])
def test_multiple_objects_per_pool_row_carried_forward(tmp_path, derive):
    inventory, prior_body = _stage_prior_with_multi_objects(tmp_path)
    args = _multi_objects_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    if derive:
        catalog = assembly.derive_catalog(prior_body, **args)
    else:
        path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1",
                                              release_id="cand1", inventory=inventory,
                                              **_carry_kwargs(args))
        catalog = phase5_release.read_catalog(path, "cand1")
    supplied = _supplied_by_member_multi(args)
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID):
        row = _prior_row(catalog, member_id)
        assert row["objects"] == _objects_of(supplied[member_id])
        prior_names = {obj["name"] for obj in _prior_row(prior_body, member_id)["objects"]}
        assert not prior_names & {obj["name"] for obj in row["objects"]}
    assert _prior_row(catalog, UNRELATED_MEMBER_ID) == _prior_row(prior_body, UNRELATED_MEMBER_ID)


def test_duplicate_object_with_same_member_and_name_is_refused(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args = {**args, "outputs": [*args["outputs"], args["outputs"][0]]}
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "POOL_SET_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def _extra_chooser_object(root, *, cutoff="2026-10-01", prefix="extra"):
    payload = f"{prefix}-chooser".encode()
    content_hash, _ = phase5_release.write_object(root, payload)
    return assembly.PoolOutput(member_id=assembly.CHOOSER_MEMBER_ID, name=f"{prefix}.bin",
                               content_hash=content_hash, cutoff=cutoff,
                               payload_bytes=len(payload))


def test_chooser_sequence_of_objects_is_accepted(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = [args["chooser"], _extra_chooser_object(tmp_path)]
    derived = assembly.derive_catalog(prior_body, **args)
    row = _prior_row(derived, assembly.CHOOSER_MEMBER_ID)
    assert row["objects"] == _objects_of(args["chooser"])


@pytest.mark.parametrize("chooser", [[], ()], ids=["empty-list", "empty-tuple"])
def test_empty_chooser_sequence_is_refused(tmp_path, chooser):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = chooser
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CHOOSER_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_duplicate_chooser_names_are_refused(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = [args["chooser"], args["chooser"]]
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CHOOSER_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_chooser_sequence_wrong_cutoff_on_second_object(tmp_path):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = [args["chooser"], _extra_chooser_object(tmp_path, cutoff="2026-09-15")]
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_MISMATCH"
    assert not _candidate_path(tmp_path).exists()


def test_chooser_sequence_damaged_second_object_is_unverified(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["chooser"] = [args["chooser"], _extra_chooser_object(tmp_path)]
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _object_path(tmp_path, args["chooser"][1]).write_bytes(b"damaged")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "OBJECT_UNVERIFIED"
    store = phase5_release.deployment_root(tmp_path)
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(store, "cand1").exists()


@pytest.mark.parametrize("cutoff", ["2026-03-15", "2026-03-14"])
def test_carry_forward_cutoff_not_advanced_leaves_no_candidate(tmp_path, cutoff):
    inventory, _ = _stage_prior(tmp_path, prior_sources={"pool_cutoff": "2026-03-15"})
    args = _valid_derive_args(tmp_path, cutoff=cutoff)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "CUTOFF_NOT_ADVANCED"
    store = phase5_release.deployment_root(tmp_path)
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(store, "cand1").exists()


def _assert_no_candidate(tmp_path):
    store = phase5_release.deployment_root(tmp_path)
    assert not _candidate_path(tmp_path).exists()
    assert not phase5_release.staged_manifest_path(store, "cand1").exists()


def _run_carry(tmp_path, inventory, args):
    return assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))


def test_supplied_object_longer_than_its_claim_is_refused(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    output = args["outputs"][0]
    payload = b"z" * (output.payload_bytes + 16)
    content_hash = phase5_release.sha256_bytes(payload)
    (phase5_release.deployment_root(tmp_path)
     / phase5_release.object_relpath(content_hash)).write_bytes(payload)
    args["outputs"][0] = dataclasses.replace(output, content_hash=content_hash,
                                             payload_bytes=len(payload) - 1)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "OBJECT_UNVERIFIED"
    _assert_no_candidate(tmp_path)


def test_supplied_object_path_replaced_by_directory_is_refused(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    path = _object_path(tmp_path, args["outputs"][0])
    path.unlink()
    path.mkdir()
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "OBJECT_UNVERIFIED"
    _assert_no_candidate(tmp_path)


def _make_fifo(path):
    path.unlink()
    try:
        os.mkfifo(path)
    except (AttributeError, OSError):
        pytest.skip("os.mkfifo is not supported on this platform")


def _refused_without_hanging(tmp_path, inventory, args, code):
    outcome = {}

    def worker():
        try:
            _run_carry(tmp_path, inventory, args)
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive(), "carry_forward hung reading a FIFO"
    assert isinstance(outcome.get("error"), assembly.ReleaseAssemblyRefusal)
    assert outcome["error"].code == code


def test_supplied_object_path_fifo_is_refused_without_hanging(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    _make_fifo(_object_path(tmp_path, args["outputs"][0]))
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _refused_without_hanging(tmp_path, inventory, args, "OBJECT_UNVERIFIED")
    _assert_no_candidate(tmp_path)


def test_prior_catalog_fifo_is_prior_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    _make_fifo(phase5_release.catalog_path(tmp_path, "prior1"))
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _refused_without_hanging(tmp_path, inventory, args, "PRIOR_MISMATCH")
    _assert_no_candidate(tmp_path)


def test_candidate_catalog_fifo_is_id_conflict(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _run_carry(tmp_path, inventory, args)
    _make_fifo(_candidate_path(tmp_path))
    _refused_without_hanging(tmp_path, inventory, args, "ID_CONFLICT")
    assert not _candidate_path(tmp_path).is_file()


def _forbid_nonregular_reads(monkeypatch):
    real = os.fdopen

    def watch(fd, *a, **k):
        assert stat.S_ISREG(os.fstat(fd).st_mode), "non-regular file was read"
        return real(fd, *a, **k)

    monkeypatch.setattr(os, "fdopen", watch)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
@pytest.mark.parametrize("site, code", [
    (lambda root, args: _object_path(root, args["outputs"][0]), "OBJECT_UNVERIFIED"),
    (lambda root, _args: phase5_release.catalog_path(root, "prior1"), "PRIOR_MISMATCH"),
    (lambda root, _args: phase5_release.staged_manifest_path(
        phase5_release.deployment_root(root), "prior1"), "PRIOR_MISMATCH"),
], ids=["supplied-object", "prior-catalog", "prior-manifest"])
def test_nonregular_inputs_never_reach_a_read(tmp_path, monkeypatch, site, kind, code):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    path = site(tmp_path, args)
    if kind == "directory":
        path.unlink()
        path.mkdir()
        holder = None
    else:
        _make_fifo(path)
        holder = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    _forbid_nonregular_reads(monkeypatch)
    try:
        _refused_without_hanging(tmp_path, dataclasses.replace(inventory, release_id="cand1"),
                                 args, code)
    finally:
        if holder is not None:
            os.close(holder)
    _assert_no_candidate(tmp_path)


def test_prior_payload_larger_than_max_object_bytes_is_prior_mismatch(tmp_path, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    supplied = (*args["outputs"], args["chooser"])
    monkeypatch.setattr(assembly, "MAX_OBJECT_BYTES",
                        max(output.payload_bytes for output in supplied) + 1)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


@pytest.mark.parametrize("mode", ["claim-above-sys-maxsize", "above-max-object-bytes"])
def test_object_unverified_when_payload_bytes_exceed_max(tmp_path, monkeypatch, mode):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    output = args["outputs"][0]
    if mode == "claim-above-sys-maxsize":
        args["outputs"][0] = dataclasses.replace(output, payload_bytes=sys.maxsize + 1)
    else:
        limit = _prior_payload_size()
        monkeypatch.setattr(assembly, "MAX_OBJECT_BYTES", limit)
        payload = b"q" * (limit + 1)
        content_hash, _ = phase5_release.write_object(tmp_path, payload)
        args["outputs"][0] = dataclasses.replace(
            output, content_hash=content_hash, payload_bytes=limit + 1)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "OBJECT_UNVERIFIED"
    _assert_no_candidate(tmp_path)


def test_inventory_with_none_bindings_is_inventory_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1", bindings=None)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "INVENTORY_MISMATCH"
    _assert_no_candidate(tmp_path)


def _pad_beyond(path, limit):
    data = path.read_bytes()
    if len(data) <= limit:
        path.write_bytes(data + b" " * (limit + 1 - len(data)))


def _oversize_prior_catalog(root, monkeypatch):
    store = phase5_release.deployment_root(root)
    path = phase5_release.catalog_path(root, "prior1")
    limit = phase5_release.staged_manifest_path(store, "prior1").stat().st_size
    _pad_beyond(path, limit)
    monkeypatch.setattr(assembly, "MAX_CATALOG_BYTES", limit)


def _oversize_prior_manifest(root, monkeypatch):
    store = phase5_release.deployment_root(root)
    manifest = phase5_release.staged_manifest_path(store, "prior1")
    limit = phase5_release.catalog_path(root, "prior1").stat().st_size
    _pad_beyond(manifest, limit)
    monkeypatch.setattr(assembly, "MAX_MANIFEST_BYTES", limit)


def _oversize_candidate_catalog(root, monkeypatch):
    store = phase5_release.deployment_root(root)
    prior = json.loads(phase5_release.catalog_path(root, "prior1").read_text())
    path = _candidate_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    phase5_release.write_catalog(path, assembly.derive_catalog(prior, **_valid_derive_args(root)))
    limit = max(phase5_release.catalog_path(root, "prior1").stat().st_size,
                phase5_release.staged_manifest_path(store, "prior1").stat().st_size)
    monkeypatch.setattr(assembly, "MAX_CATALOG_BYTES", limit)
    _pad_beyond(path, limit)


@pytest.mark.parametrize("setup, code", [
    (_oversize_prior_catalog, "PRIOR_MISMATCH"),
    (_oversize_prior_manifest, "PRIOR_MISMATCH"),
    (_oversize_candidate_catalog, "ID_CONFLICT"),
], ids=["prior-catalog", "prior-manifest", "candidate-catalog"])
def test_meta_oversize_refusal(tmp_path, monkeypatch, setup, code):
    inventory, _ = _stage_prior(tmp_path)
    setup(tmp_path, monkeypatch)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"),
                   _valid_derive_args(tmp_path))
    assert error.value.code == code
    if code == "PRIOR_MISMATCH":
        _assert_no_candidate(tmp_path)


def test_meta_oversize_candidate_manifest_refusal(tmp_path, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _run_carry(tmp_path, inventory, args)
    store = phase5_release.deployment_root(tmp_path)
    limit = max(phase5_release.catalog_path(tmp_path, "prior1").stat().st_size,
                phase5_release.staged_manifest_path(store, "prior1").stat().st_size,
                _candidate_path(tmp_path).stat().st_size)
    _pad_beyond(phase5_release.staged_manifest_path(store, "cand1"), limit)
    monkeypatch.setattr(assembly, "MAX_MANIFEST_BYTES", limit)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "ID_CONFLICT"


def test_prior_catalog_symlink_is_prior_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    prior_path = phase5_release.catalog_path(tmp_path, "prior1")
    copy = tmp_path / "prior-catalog-copy.json"
    copy.write_bytes(prior_path.read_bytes())
    prior_path.unlink()
    prior_path.symlink_to(copy)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_prior_manifest_symlink_is_prior_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    manifest = phase5_release.staged_manifest_path(store, "prior1")
    copy = tmp_path / "prior-manifest-copy.json"
    copy.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(copy)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, _valid_derive_args(tmp_path))
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_carried_object_with_wrong_bytes_claim_is_prior_mismatch(tmp_path):
    inventory, prior_body = _stage_prior(tmp_path)
    members = copy.deepcopy(prior_body["members"])
    next(row for row in members
         if row["member_id"] == UNRELATED_MEMBER_ID)["objects"][0]["bytes"] += 1
    body = _rebuilt_prior(prior_body, members)
    phase5_release.write_catalog(phase5_release.catalog_path(tmp_path, "prior1"), body)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"),
                   _valid_derive_args(tmp_path))
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_metadata_readers_accept_files_exactly_at_the_cap(tmp_path, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _run_carry(tmp_path, inventory, args)
    store = phase5_release.deployment_root(tmp_path)
    catalog_cap = max(phase5_release.catalog_path(tmp_path, rid).stat().st_size
                      for rid in ("prior1", "cand1"))
    manifest_cap = max(phase5_release.staged_manifest_path(store, rid).stat().st_size
                       for rid in ("prior1", "cand1"))
    monkeypatch.setattr(assembly, "MAX_CATALOG_BYTES", catalog_cap)
    monkeypatch.setattr(assembly, "MAX_MANIFEST_BYTES", manifest_cap)
    _run_carry(tmp_path, inventory, args)
    exact = {"MAX_CATALOG_BYTES": catalog_cap, "MAX_MANIFEST_BYTES": manifest_cap}
    for name, cap in (("MAX_CATALOG_BYTES", catalog_cap - 1),
                      ("MAX_MANIFEST_BYTES", manifest_cap - 1)):
        monkeypatch.setattr(assembly, name, cap)
        with pytest.raises(assembly.ReleaseAssemblyRefusal):
            _run_carry(tmp_path, inventory, args)
        monkeypatch.setattr(assembly, name, exact[name])


@pytest.mark.parametrize("mode", ["catalog", "manifest"])
def test_candidate_symlink_is_id_conflict(tmp_path, mode):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    if mode == "catalog":
        path = _candidate_path(tmp_path)
        source = phase5_release.catalog_path(tmp_path, "prior1")
    else:
        path = phase5_release.staged_manifest_path(store, "cand1")
        source = phase5_release.staged_manifest_path(store, "prior1")
    path.parent.mkdir(parents=True, exist_ok=True)
    copy = tmp_path / f"candidate-{mode}-copy.json"
    copy.write_bytes(source.read_bytes())
    path.symlink_to(copy)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"), args)
    assert error.value.code == "ID_CONFLICT"
    if mode == "catalog":
        assert path.is_symlink() and path.read_bytes() == copy.read_bytes()
    else:
        assert not _candidate_path(tmp_path).exists()


def _lock_held(root):
    lock = phase5_release.deployment_root(root) / ".carry_forward.lock"
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _lock_symlinked(root):
    lock = phase5_release.deployment_root(root) / ".carry_forward.lock"
    target = root / "lock-target"
    target.write_bytes(b"")
    lock.symlink_to(target)
    return None


def _lock_fifo(root):
    lock = phase5_release.deployment_root(root) / ".carry_forward.lock"
    if lock.exists() or lock.is_symlink():
        lock.unlink()
    try:
        os.mkfifo(lock)
    except (AttributeError, OSError):
        pytest.skip("os.mkfifo is not supported on this platform")
    return None


@pytest.mark.parametrize("setup, wait",
                         [(_lock_held, 0.2), (_lock_symlinked, None), (_lock_fifo, None)],
                         ids=["held-timeout", "symlink", "fifo"])
def test_publication_lock_failure_is_publication_failed(tmp_path, monkeypatch, setup, wait):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    handle = setup(tmp_path)
    if wait is not None:
        monkeypatch.setattr(assembly, "LOCK_WAIT_S", wait)
    try:
        inventory = dataclasses.replace(inventory, release_id="cand1")
        with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
            _run_carry(tmp_path, inventory, args)
        assert error.value.code == "PUBLICATION_FAILED"
        _assert_no_candidate(tmp_path)
    finally:
        if handle is not None:
            fcntl.flock(handle, fcntl.LOCK_UN)
            os.close(handle)


@pytest.mark.parametrize("mode", ("missing", "corrupt", "truncated"))
def test_damaged_untouched_carried_object_is_prior_mismatch(tmp_path, mode):
    inventory, body = _stage_prior(tmp_path)
    row = next(r for r in body["members"] if r["member_id"] == UNRELATED_MEMBER_ID)
    obj = row["objects"][0]
    path = phase5_release.deployment_root(tmp_path) / obj["path"]
    if mode == "missing":
        path.unlink()
    elif mode == "corrupt":
        path.write_bytes(b"x" * obj["bytes"])
    else:
        path.write_bytes(b"x" * (obj["bytes"] - 1))
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"),
                   _valid_derive_args(tmp_path))
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_inventory_hash_not_matching_prior_payload_is_inventory_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    artifact = inventory.artifacts[0]
    member = dataclasses.replace(artifact.members[0], content_hash="sha256:" + "a" * 64)
    artifact = dataclasses.replace(artifact, members=(member,) + artifact.members[1:])
    inventory = dataclasses.replace(inventory, release_id="cand1",
                                    artifacts=(artifact,) + inventory.artifacts[1:])
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, _valid_derive_args(tmp_path))
    assert error.value.code == "INVENTORY_MISMATCH"
    _assert_no_candidate(tmp_path)


def _reversed_object_order(args):
    return {**args, "outputs": list(reversed(args["outputs"])),
            "chooser": list(reversed(args["chooser"]))}


def test_object_order_does_not_change_derived_catalog(tmp_path):
    inventory, prior_body = _stage_prior_with_multi_objects(tmp_path)
    args = _multi_objects_args(tmp_path)
    first = assembly.derive_catalog(prior_body, **args)
    assert first == assembly.derive_catalog(prior_body, **_reversed_object_order(args))
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID):
        names = [obj["name"] for obj in _prior_row(first, member_id)["objects"]]
        assert names == sorted(names)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    path = _run_carry(tmp_path, inventory, args)
    published = path.read_bytes()
    again = _run_carry(tmp_path, inventory, _reversed_object_order(args))
    assert again == path
    assert again.read_bytes() == published


def test_stage_release_type_error_is_inventory_mismatch(tmp_path, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)

    def boom(*a, **k):
        raise TypeError("unexpected staging shape")

    monkeypatch.setattr(assembly.deployment, "stage_release", boom)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "INVENTORY_MISMATCH"
    _assert_no_candidate(tmp_path)


def _break_manifest(root, path, mode):
    copy = root / "manifest-copy.json"
    copy.write_bytes(path.read_bytes())
    path.unlink()
    if mode == "symlink":
        path.symlink_to(copy)
    elif mode == "directory":
        path.mkdir()
    else:
        try:
            os.mkfifo(path)
        except (AttributeError, OSError):
            pytest.skip("os.mkfifo is not supported on this platform")


def _manifest_unreadable(store, release_id):
    with pytest.raises(deployment.StagingRefused) as error:
        deployment.resolve_release(store, release_id)
    assert {issue.code for issue in error.value.issues} == {"MANIFEST_UNREADABLE"}


@pytest.mark.parametrize("mode", ["symlink", "directory", "fifo", "oversize", "missing"])
def test_resolve_release_refuses_bad_manifest(tmp_path, monkeypatch, mode):
    _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    path = phase5_release.staged_manifest_path(store, "prior1")
    if mode == "missing":
        path.unlink()
        with pytest.raises(deployment.ReleaseNotStaged):
            deployment.resolve_release(store, "prior1")
        return
    if mode == "oversize":
        monkeypatch.setattr(deployment, "MAX_MANIFEST_BYTES", path.stat().st_size - 1)
    else:
        _break_manifest(tmp_path, path, mode)
    _manifest_unreadable(store, "prior1")


@pytest.mark.parametrize("mode", ["oversize", "symlink"])
def test_phase5_read_catalog_refuses_bad_catalog(tmp_path, monkeypatch, mode):
    _stage_prior(tmp_path)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    if mode == "oversize":
        monkeypatch.setattr(phase5_release, "MAX_CATALOG_BYTES", path.stat().st_size - 1)
    else:
        copy = tmp_path / "catalog-copy.json"
        copy.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(copy)
    with pytest.raises(phase5_release.ReleaseLayoutError):
        phase5_release.read_catalog(path, "prior1")


def _fstat_raising_in_verify_objects(real_fstat):
    def boom(fd):
        frame = sys._getframe(1)
        while frame is not None:
            if frame.f_code.co_name == "_verify_objects":
                raise OSError("simulated fstat failure")
            frame = frame.f_back
        return real_fstat(fd)
    return boom


def test_verify_objects_fstat_oserror_is_object_unverified(tmp_path, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    monkeypatch.setattr(assembly.os, "fstat",
                        _fstat_raising_in_verify_objects(assembly.os.fstat))
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "OBJECT_UNVERIFIED"
    _assert_no_candidate(tmp_path)


def _prior_payload_size():
    return len(next(iter(_fixture("prior1")[2].values())))


@pytest.mark.parametrize("delta, refused", [(0, False), (1, True)], ids=["at-cap", "over-cap"])
def test_prior_payload_max_object_bytes_boundary(tmp_path, monkeypatch, delta, refused):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    monkeypatch.setattr(assembly, "MAX_OBJECT_BYTES", _prior_payload_size() - delta)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    if not refused:
        assert _run_carry(tmp_path, inventory, args) == _candidate_path(tmp_path)
        return
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def _output_at_size(root, output, size):
    content_hash, _ = phase5_release.write_object(root, b"q" * size)
    return dataclasses.replace(output, content_hash=content_hash, payload_bytes=size)


@pytest.mark.parametrize("delta, refused", [(0, False), (1, True)], ids=["at-cap", "over-cap"])
def test_supplied_object_max_object_bytes_boundary(tmp_path, monkeypatch, delta, refused):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    size = _prior_payload_size()
    monkeypatch.setattr(assembly, "MAX_OBJECT_BYTES", size)
    target = assembly.POOL_MEMBER_IDS[0]
    args["outputs"] = [_output_at_size(tmp_path, output, size + delta)
                       if output.member_id == target else output for output in args["outputs"]]
    inventory = dataclasses.replace(inventory, release_id="cand1")
    if not refused:
        assert _run_carry(tmp_path, inventory, args) == _candidate_path(tmp_path)
        return
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "OBJECT_UNVERIFIED"
    _assert_no_candidate(tmp_path)


@pytest.mark.parametrize("bad", ["a/b", "a\x00b", ".hidden"])
def test_unsafe_prior_release_id_is_prior_mismatch(tmp_path, bad):
    inventory, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    args["expected_prior_release_id"] = bad
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "PRIOR_MISMATCH"
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id=bad, release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_deeply_nested_prior_catalog_is_prior_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").write_bytes(b"[" * 200000)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_shared_readers_accept_exact_cap_and_refuse_one_below(tmp_path, monkeypatch):
    _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    catalog = phase5_release.catalog_path(tmp_path, "prior1")
    size = catalog.stat().st_size
    monkeypatch.setattr(phase5_release, "MAX_CATALOG_BYTES", size)
    assert phase5_release.read_catalog(catalog, "prior1")["release_id"] == "prior1"
    monkeypatch.setattr(phase5_release, "MAX_CATALOG_BYTES", size - 1)
    with pytest.raises(phase5_release.ReleaseLayoutError):
        phase5_release.read_catalog(catalog, "prior1")
    manifest = phase5_release.staged_manifest_path(store, "prior1")
    msize = manifest.stat().st_size
    monkeypatch.setattr(deployment, "MAX_MANIFEST_BYTES", msize)
    assert deployment.resolve_release(store, "prior1").release_id == "prior1"
    monkeypatch.setattr(deployment, "MAX_MANIFEST_BYTES", msize - 1)
    with pytest.raises(deployment.StagingRefused) as error:
        deployment.resolve_release(store, "prior1")
    assert {issue.code for issue in error.value.issues} == {"MANIFEST_UNREADABLE"}


def test_inventory_members_malformed_is_inventory_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    bad = dataclasses.replace(inventory.artifacts[0], members=None)
    inventory = dataclasses.replace(inventory, release_id="cand1",
                                    artifacts=(bad,) + inventory.artifacts[1:])
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, _valid_derive_args(tmp_path))
    assert error.value.code == "INVENTORY_MISMATCH"
    _assert_no_candidate(tmp_path)


@pytest.mark.parametrize("target", ["catalog", "manifest"])
@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_shared_readers_never_read_non_regular_files(tmp_path, monkeypatch, kind, target):
    _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    if target == "catalog":
        path = phase5_release.catalog_path(tmp_path, "prior1")
        call, expected = (lambda: phase5_release.read_catalog(path, "prior1")), phase5_release.ReleaseLayoutError
    else:
        path = phase5_release.staged_manifest_path(store, "prior1")
        call, expected = (lambda: deployment.resolve_release(store, "prior1")), deployment.StagingRefused
    path.unlink()
    holder = None
    if kind == "directory":
        path.mkdir()
    else:
        try:
            os.mkfifo(path)
        except (AttributeError, OSError):
            pytest.skip("os.mkfifo is not supported on this platform")
        holder = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    real_fdopen, real_read, real_open = os.fdopen, os.read, os.open
    tracked = set()

    def watch_open(opened, *args, **kwargs):
        fd = real_open(opened, *args, **kwargs)
        if str(opened) == str(path):
            tracked.add(fd)
        return fd

    def watch_fdopen(fd, *args, **kwargs):
        assert stat.S_ISREG(os.fstat(fd).st_mode), "non-regular file was read"
        return real_fdopen(fd, *args, **kwargs)

    def watch_read(fd, *args, **kwargs):
        if fd in tracked:
            assert stat.S_ISREG(os.fstat(fd).st_mode), "non-regular file was read"
        return real_read(fd, *args, **kwargs)

    monkeypatch.setattr(os, "open", watch_open)
    monkeypatch.setattr(os, "fdopen", watch_fdopen)
    monkeypatch.setattr(os, "read", watch_read)
    outcome = {}

    def run():
        try:
            call()
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(5)
    try:
        assert not worker.is_alive()
    finally:
        if holder is not None:
            os.close(holder)
    assert isinstance(outcome.get("error"), expected)


class _CountingFile:
    """Records every read request and the bytes it returned."""

    def __init__(self, handle, log):
        self._handle, self._log = handle, log

    def read(self, size=-1):
        data = self._handle.read(size)
        self._log.append((size, len(data)))
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._handle.close()

    def __getattr__(self, name):
        return getattr(self._handle, name)


def test_catalog_reader_never_reads_past_its_cap(tmp_path, monkeypatch):
    _stage_prior(tmp_path)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    cap = 64
    path.write_bytes(b" " * (cap * 100))
    monkeypatch.setattr(phase5_release, "MAX_CATALOG_BYTES", cap)
    log = []
    real = os.fdopen
    monkeypatch.setattr(os, "fdopen", lambda fd, *a, **k: _CountingFile(real(fd, *a, **k), log))
    with pytest.raises(phase5_release.ReleaseLayoutError):
        phase5_release.read_catalog(path, "prior1")
    assert log and all(size <= cap + 1 for size, _ in log)
    assert sum(got for _, got in log) <= cap + 1


def test_manifest_reader_refuses_oversize_manifests_at_the_budget(tmp_path, monkeypatch):
    _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    manifest = phase5_release.staged_manifest_path(store, "prior1")
    limit = 1024 * 1024
    monkeypatch.setattr(deployment, "MAX_MANIFEST_BYTES", limit)
    base = manifest.read_bytes()

    reads = []
    real_open = builtins.open

    class _ManifestReadSpy:
        def __init__(self, handle):
            self._handle = handle

        def read(self, *args, **kwargs):
            data = self._handle.read(*args, **kwargs)
            reads.append(len(data))
            return data

        def readinto(self, *args, **kwargs):
            count = self._handle.readinto(*args, **kwargs)
            reads.append(count)
            return count

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *exc):
            return self._handle.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def spy(file, *args, **kwargs):
        stream = real_open(file, *args, **kwargs)
        if str(file) == str(manifest):
            return _ManifestReadSpy(stream)
        return stream

    monkeypatch.setattr(builtins, "open", spy)

    manifest.write_bytes(base + b" " * (limit - len(base)))
    assert deployment.resolve_release(store, "prior1").release_id == "prior1"
    assert reads and sum(reads) == limit
    reads.clear()
    manifest.write_bytes(base + b" " * (limit + 1 - len(base)))
    with pytest.raises(deployment.StagingRefused) as error:
        deployment.resolve_release(store, "prior1")
    assert {issue.code for issue in error.value.issues} == {"MANIFEST_UNREADABLE"}
    assert sum(reads) <= limit + 1
    reads.clear()
    manifest.write_bytes(base + b" " * (5 * limit))
    with pytest.raises(deployment.StagingRefused) as error:
        deployment.resolve_release(store, "prior1")
    assert {issue.code for issue in error.value.issues} == {"MANIFEST_UNREADABLE"}
    assert sum(reads) <= limit + 1


@pytest.mark.parametrize("exc", [TypeError, AttributeError, KeyError])
def test_stage_time_stray_errors_are_inventory_mismatch(tmp_path, monkeypatch, exc):
    inventory, _ = _stage_prior(tmp_path)

    def boom(*args, **kwargs):
        raise exc("x")

    monkeypatch.setattr(assembly.deployment, "stage_release", boom)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"),
                   _valid_derive_args(tmp_path))
    assert error.value.code == "INVENTORY_MISMATCH"
    _assert_no_candidate(tmp_path)


@pytest.mark.parametrize("bad", ["members-none", "artifact-none", "hash-unhashable"])
def test_malformed_inventory_members_are_inventory_mismatch(tmp_path, bad):
    inventory, _ = _stage_prior(tmp_path)
    artifact = inventory.artifacts[0]
    if bad == "members-none":
        artifacts = (dataclasses.replace(artifact, members=None),) + inventory.artifacts[1:]
    elif bad == "artifact-none":
        artifacts = (None,) + inventory.artifacts[1:]
    else:
        member = dataclasses.replace(artifact.members[0], content_hash=["x"])
        artifacts = (dataclasses.replace(artifact, members=(member,) + artifact.members[1:]),
                     ) + inventory.artifacts[1:]
    inventory = dataclasses.replace(inventory, release_id="cand1", artifacts=artifacts)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, _valid_derive_args(tmp_path))
    assert error.value.code == "INVENTORY_MISMATCH"
    _assert_no_candidate(tmp_path)


_DEEP = b"[" * 200_000 + b"]" * 200_000


def test_balanced_deeply_nested_prior_catalog_is_prior_mismatch(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").write_bytes(_DEEP)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"),
                   _valid_derive_args(tmp_path))
    assert error.value.code == "PRIOR_MISMATCH"
    _assert_no_candidate(tmp_path)


def test_deeply_nested_candidate_catalog_is_id_conflict(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    args = _valid_derive_args(tmp_path)
    _run_carry(tmp_path, inventory, args)
    _candidate_path(tmp_path).write_bytes(_DEEP)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, inventory, args)
    assert error.value.code == "ID_CONFLICT"


def test_deeply_nested_catalog_through_shared_reader_is_layout_error(tmp_path):
    path = tmp_path / "deep.json"
    path.write_bytes(_DEEP)
    with pytest.raises(phase5_release.ReleaseLayoutError):
        phase5_release.read_catalog(path, "prior1")


def test_supplied_object_read_stops_at_its_claim(tmp_path, monkeypatch):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    target = args["outputs"][0]
    path = _object_path(tmp_path, target)
    path.write_bytes(path.read_bytes() + b"x" * 100_000)
    log, tracked = [], set()
    real_open, real_fdopen = os.open, os.fdopen

    def watch_open(opened, *a, **k):
        fd = real_open(opened, *a, **k)
        if str(opened) == str(path):
            tracked.add(fd)
        return fd

    def watch_fdopen(fd, *a, **k):
        handle = real_fdopen(fd, *a, **k)
        return _CountingFile(handle, log) if fd in tracked else handle

    monkeypatch.setattr(os, "open", watch_open)
    monkeypatch.setattr(os, "fdopen", watch_fdopen)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        _run_carry(tmp_path, dataclasses.replace(inventory, release_id="cand1"), args)
    assert error.value.code == "OBJECT_UNVERIFIED"
    assert log and all(0 <= size <= target.payload_bytes + 1 for size, _ in log)
    assert sum(got for _, got in log) <= target.payload_bytes + 1


def _carry_args(tmp_path, carried_members=(), chooser_carried=False):
    """derive args where ``carried_members`` (and optionally the chooser) are the prior objects, carried."""
    _inventory, body = _stage_prior(tmp_path)
    cutoff_prior, carried = assembly.prior_carry_set(tmp_path, "prior1")
    assert cutoff_prior == "2026-09-01"
    args = _valid_derive_args(tmp_path)
    by_key = {item.member_id: item for item in carried}
    args["outputs"] = [by_key[o.member_id] if o.member_id in carried_members else o
                       for o in args["outputs"]]
    if chooser_carried:
        args["chooser"] = by_key[assembly.CHOOSER_MEMBER_ID]
    return body, args


def test_prior_carry_set_returns_every_prior_object_as_carried(tmp_path):
    _stage_prior(tmp_path)
    cutoff, carried = assembly.prior_carry_set(tmp_path, "prior1")
    assert cutoff == "2026-09-01"
    members = {item.member_id for item in carried}
    assert members == {*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID}
    assert all(item.carried and item.cutoff == "2026-09-01" and item.name == "prior.bin"
               for item in carried)


@pytest.mark.parametrize("release_id", ["missing", "", "../x", ".hidden"])
def test_prior_carry_set_refuses_unknown_or_unsafe_prior(tmp_path, release_id):
    _stage_prior(tmp_path)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.prior_carry_set(tmp_path, release_id)
    assert err.value.code == "PRIOR_MISMATCH"


def test_carried_objects_keep_their_prior_rows(tmp_path):
    body, args = _carry_args(
        tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:3], chooser_carried=True)
    candidate = assembly.derive_catalog(body, **args)
    prior_rows = {row["member_id"]: row for row in body["members"]}
    rows = {row["member_id"]: row for row in candidate["members"]}
    for member_id in (*assembly.POOL_MEMBER_IDS[:3], assembly.CHOOSER_MEMBER_ID):
        assert rows[member_id]["objects"] == prior_rows[member_id]["objects"]
    for member_id in assembly.POOL_MEMBER_IDS[3:]:
        assert rows[member_id]["objects"] != prior_rows[member_id]["objects"]
    assert candidate["sources"]["pool_cutoff"] == "2026-10-01"


def test_carried_object_not_in_prior_row_refuses_pool_set_mismatch(tmp_path):
    body, args = _carry_args(tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:1])
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], content_hash=args["outputs"][-1].content_hash)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "POOL_SET_MISMATCH"


def test_carried_chooser_not_in_prior_row_refuses_chooser_mismatch(tmp_path):
    body, args = _carry_args(tmp_path, chooser_carried=True)
    args["chooser"] = dataclasses.replace(args["chooser"], name="other.bin")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "CHOOSER_MISMATCH"


@pytest.mark.parametrize("cutoff", ["2026-10-02", "2026-9-1", "bad"])
def test_carried_cutoff_after_session_or_malformed_refuses(tmp_path, cutoff):
    body, args = _carry_args(tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:1])
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], cutoff=cutoff)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "CUTOFF_MISMATCH"


def test_non_carried_object_with_old_cutoff_still_refuses(tmp_path):
    body, args = _carry_args(tmp_path)
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], cutoff="2026-09-01")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "CUTOFF_MISMATCH"


def test_carried_flag_must_be_a_bool(tmp_path):
    body, args = _carry_args(tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:1])
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], carried="yes")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "POOL_SET_MISMATCH"
    assert "carried flag" in str(err.value)


def test_carried_object_with_wrong_payload_bytes_refuses(tmp_path):
    body, args = _carry_args(tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:1])
    args["outputs"][0] = dataclasses.replace(
        args["outputs"][0], payload_bytes=args["outputs"][0].payload_bytes + 1)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "POOL_SET_MISMATCH"


def test_carried_object_cutoff_equal_to_session_cutoff_is_accepted(tmp_path):
    body, args = _carry_args(tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:1])
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], cutoff=args["cutoff"])
    candidate = assembly.derive_catalog(body, **args)
    assert candidate["sources"]["pool_cutoff"] == args["cutoff"]


def _rewrite_prior(tmp_path, mutate):
    _inventory, body = _stage_prior(tmp_path)
    mutate(body)
    body = phase5_release.manifest_body(
        body["release_id"], body["deployment_id"], body["members"], body["sources"])
    phase5_release.write_catalog(phase5_release.catalog_path(tmp_path, "prior1"), body)


def _drop_pool_cutoff(body):
    body["sources"] = {}


def _bad_pool_cutoff(body):
    body["sources"]["pool_cutoff"] = "2026-9-1"


def _drop_object_bytes(body):
    del body["members"][0]["objects"][0]["bytes"]


def _drop_members(body):
    body["members"] = []


@pytest.mark.parametrize(
    "mutate", [_drop_pool_cutoff, _bad_pool_cutoff, _drop_object_bytes, _drop_members])
def test_prior_carry_set_refuses_malformed_prior_catalog(tmp_path, mutate):
    _rewrite_prior(tmp_path, mutate)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.prior_carry_set(tmp_path, "prior1")
    assert err.value.code == "PRIOR_MISMATCH"


def test_carried_pool_object_with_a_changed_name_refuses(tmp_path):
    body, args = _carry_args(tmp_path, carried_members=assembly.POOL_MEMBER_IDS[:1])
    args["outputs"][0] = dataclasses.replace(args["outputs"][0], name="renamed.bin")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "POOL_SET_MISMATCH"


def test_earlier_session_cutoff_with_carried_objects_is_not_advanced(tmp_path):
    _stage_prior(tmp_path)
    _cutoff, carried = assembly.prior_carry_set(tmp_path, "prior1")
    args = _valid_derive_args(tmp_path, cutoff="2026-08-01")
    by_key = {item.member_id: item for item in carried}
    args["outputs"] = [by_key[o.member_id] if o.member_id == assembly.POOL_MEMBER_IDS[0] else o
                       for o in args["outputs"]]
    body = json.loads(phase5_release.catalog_path(tmp_path, "prior1").read_bytes())
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as err:
        assembly.derive_catalog(body, **args)
    assert err.value.code == "CUTOFF_NOT_ADVANCED"


def _add_second_object(body):
    row = next(r for r in body["members"] if r["member_id"] == assembly.POOL_MEMBER_IDS[0])
    row["objects"].append({**row["objects"][0], "name": "second.bin"})


def test_carried_object_matches_any_object_of_a_multi_object_prior_row(tmp_path):
    _rewrite_prior(tmp_path, _add_second_object)
    _cutoff, carried = assembly.prior_carry_set(tmp_path, "prior1")
    member = assembly.POOL_MEMBER_IDS[0]
    second = next(item for item in carried if item.member_id == member and item.name == "second.bin")
    args = _valid_derive_args(tmp_path)
    args["outputs"] = [second if o.member_id == member else o for o in args["outputs"]]
    body = json.loads(phase5_release.catalog_path(tmp_path, "prior1").read_bytes())
    candidate = assembly.derive_catalog(body, **args)
    row = next(r for r in candidate["members"] if r["member_id"] == member)
    assert [o["name"] for o in row["objects"]] == ["second.bin"]


def test_release_caps_are_pinned():
    assert deployment.MAX_MANIFEST_BYTES == 64 * 1024 * 1024
    assert assembly.MAX_CATALOG_BYTES == 64 * 1024 * 1024
    assert assembly.MAX_OBJECT_BYTES == 512 * 1024 * 1024
    assert assembly.LOCK_WAIT_S == 30.0


def test_symlinked_staged_manifest_is_refused(tmp_path):
    _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    manifest = phase5_release.staged_manifest_path(store, "prior1")
    real = tmp_path / "real_manifest.json"
    real.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(real)
    with pytest.raises(deployment.StagingRefused) as error:
        deployment.resolve_release(store, "prior1")
    assert {issue.code for issue in error.value.issues} == {"MANIFEST_UNREADABLE"}
