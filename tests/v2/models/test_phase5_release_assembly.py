"""Pool release assembly: derive/stage a carried-forward candidate catalog (#192 slice 2)."""
from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import threading
import time

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


@pytest.mark.parametrize("deployment_id, binding_id", [("d1", "b1"), ("deployZ", "bindZ")])
def test_happy_path_carries_the_pools_forward(tmp_path, deployment_id, binding_id):
    inventory, prior_body = _stage_prior(tmp_path, deployment_id=deployment_id, binding_id=binding_id)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    prior_path = phase5_release._catalog_path(tmp_path, "prior1")
    prior_bytes = prior_path.read_bytes()
    root_manifest = (tmp_path / "phase5_release.json").read_bytes()

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == phase5_release._catalog_path(tmp_path, "cand1")
    catalog = phase5_release._read_catalog(path, "cand1")
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
    (lambda a: _replace_output(a, member_id=5), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, member_id=["x"]), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, name=""), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, payload_bytes=-1), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, payload_bytes=True), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, payload_bytes="12"), "POOL_SET_MISMATCH"),
    (lambda a: _replace_chooser(a, member_id="admissible_table:dyn_sv"), "CHOOSER_MISMATCH"),
    (lambda a: {**a, "chooser": "not-a-pool-output"}, "CHOOSER_MISMATCH"),
    (lambda a: _replace_chooser(a, content_hash="sha256:nothex"), "POOL_SET_MISMATCH"),
    (lambda a: _replace_chooser(a, name=""), "CHOOSER_MISMATCH"),
    (lambda a: _replace_chooser(a, payload_bytes=-1), "CHOOSER_MISMATCH"),
    (lambda a: _output_with(a, "trailing_pnl_cutoff", member_id="bogus"), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, content_hash="sha256:" + "g" * 64), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, content_hash="sha256:" + "A" * 64), "POOL_SET_MISMATCH"),
    (lambda a: _replace_output(a, name=5), "POOL_SET_MISMATCH"),
]


@pytest.mark.parametrize("mutate, code", _POOL_REFUSALS, ids=[
    "outputs-not-a-sequence", "five-outputs", "duplicate", "malformed-hash", "element-not-a-pool-output",
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


@pytest.mark.parametrize("cutoff", ["2026-09-01", "2026-08-15"])
def test_cutoff_not_advanced(tmp_path, cutoff):
    _, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path, cutoff=cutoff)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_NOT_ADVANCED"
    store = phase5_release.deployment_root(tmp_path)
    assert not _candidate_path(tmp_path).exists()
    assert not deployment._manifest_path(store, "cand1").exists()


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
    deployment._manifest_path(store, "cand1").unlink()
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
    entered, proceed, first = threading.Event(), threading.Event(), {"done": False}
    real_stage = assembly.deployment.stage_release
    monkeypatch.setattr(assembly.deployment, "stage_release", _gate(real_stage, entered, proceed, first))
    results = {}

    def run(tag, kwargs):
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
    time.sleep(0.5)
    assert b.is_alive()
    proceed.set()
    a.join(20)
    b.join(20)
    assert not a.is_alive() and not b.is_alive()
    assert results["a"] == _candidate_path(tmp_path)
    assert isinstance(results["b"], assembly.ReleaseAssemblyRefusal)
    assert results["b"].code == "ID_CONFLICT"
    assert phase5_release._read_catalog(_candidate_path(tmp_path), "cand1") == _expected_catalog(
        prior_body, args_x, "cand1")


def _corrupt_candidate_manifest(root, args, monkeypatch):
    _stage_other_candidate_release(root)
    store = phase5_release.deployment_root(root)
    deployment._manifest_path(store, "cand1").write_bytes(b"garbage manifest")


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
    monkeypatch.setattr(assembly, "_write_catalog", boom)


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
    phase5_release._catalog_path(tmp_path, "prior1").unlink()
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
    deployment._manifest_path(store, "prior1").unlink()
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"


def _assert_prior_mismatch_on_retry(tmp_path, inventory, args, store):
    for _ in range(2):
        with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
            assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
        assert error.value.code == "PRIOR_MISMATCH"
        assert not _candidate_path(tmp_path).exists()
        assert not deployment._manifest_path(store, "cand1").exists()


def test_prior_mismatch_when_prior_manifest_hash_is_invalid(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    path = deployment._manifest_path(store, "prior1")
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
    deployment._manifest_path(store, "prior1").write_bytes(
        deployment._manifest_path(store, "other1").read_bytes())
    inventory = dataclasses.replace(inventory, release_id="cand1")
    _assert_prior_mismatch_on_retry(tmp_path, inventory, args, store)


def test_prior_mismatch_when_prior_catalog_deployment_id_differs(tmp_path):
    inventory, prior_body = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    body = phase5_release.manifest_body(prior_body["release_id"], "dX",
                                        prior_body["members"], prior_body["sources"])
    phase5_release._write_catalog(phase5_release._catalog_path(tmp_path, "prior1"), body)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                       inventory=inventory, **_carry_kwargs(args))
    assert error.value.code == "PRIOR_MISMATCH"
    assert not _candidate_path(tmp_path).exists()
    assert not deployment._manifest_path(phase5_release.deployment_root(tmp_path),
                                         "cand1").exists()


def test_carry_forward_leaves_existing_deployed_pointer_untouched(tmp_path):
    inventory, _ = _stage_prior(tmp_path)
    args = _valid_derive_args(tmp_path)
    inventory = dataclasses.replace(inventory, release_id="cand1")
    pointer = phase5_release.deployment_root(tmp_path) / "DEPLOYED"
    pointer.write_bytes(b'{"x":1}')

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == phase5_release._catalog_path(tmp_path, "cand1")
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
    catalog = phase5_release._read_catalog(path, release_id)
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


@pytest.mark.parametrize("cutoff", ["2026-03-15", "2026-03-14"])
def test_cutoff_not_advanced_refusal(tmp_path, cutoff):
    _, prior_body = _stage_prior(tmp_path, prior_sources={"pool_cutoff": "2026-03-15"})
    args = _valid_derive_args(tmp_path, cutoff=cutoff)
    with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
        assembly.derive_catalog(prior_body, **args)
    assert error.value.code == "CUTOFF_NOT_ADVANCED"
    store = phase5_release.deployment_root(tmp_path)
    assert not _candidate_path(tmp_path).exists()
    assert not deployment._manifest_path(store, "cand1").exists()


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
    assert not deployment._manifest_path(phase5_release.deployment_root(tmp_path),
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
        patch.setattr(assembly, "_write_catalog", boom)
        with pytest.raises(assembly.ReleaseAssemblyRefusal) as error:
            assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                           inventory=inventory, **_carry_kwargs(args))
        assert error.value.code == "PUBLICATION_FAILED"

    path = assembly.carry_forward_release(tmp_path, prior_release_id="prior1", release_id="cand1",
                                          inventory=inventory, **_carry_kwargs(args))
    assert path == _candidate_path(tmp_path)
    assert phase5_release._read_catalog(path, "cand1") == _expected_catalog(prior_body, args, "cand1")
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
    catalog = phase5_release._read_catalog(path, "cand1")
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
    real = pathlib.Path.read_bytes

    def guarded(self):
        if str(self) == str(target):
            raise PermissionError("denied")
        return real(self)

    monkeypatch.setattr(pathlib.Path, "read_bytes", guarded)


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
    assert not _candidate_path(tmp_path).exists()
    assert not deployment._manifest_path(store, "cand1").exists()


_POOL_OBJECT_ORDER = (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID)
_PRIOR_MULTI_COUNTS = dict(zip(_POOL_OBJECT_ORDER, (2, 2, 2, 2, 4, 3, 2)))
_SUPPLIED_MULTI_COUNTS = dict(zip(_POOL_OBJECT_ORDER, (3, 3, 3, 2, 2, 4, 2)))


def _stage_prior_with_multi_objects(root, *, release_id="prior1", deployment_id="d1",
                                    prior_cutoff="2026-09-01"):
    inventory, _ = _stage_prior(root, release_id=release_id, deployment_id=deployment_id)
    catalog = phase5_release._read_catalog(phase5_release._catalog_path(root, release_id), release_id)
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
        catalog = phase5_release._read_catalog(path, "cand1")
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
    assert not deployment._manifest_path(store, "cand1").exists()


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
    assert not deployment._manifest_path(store, "cand1").exists()
