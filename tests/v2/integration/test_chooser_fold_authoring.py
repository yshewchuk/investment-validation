"""Offline chooser fold declarations using only synthetic temporary artifacts."""
# packages: engine.v2.models, engine.v2.data, engine.v2.foundation
import builtins
import copy
import dataclasses
import errno
import hashlib
import io
import json
import os
import stat
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import joblib
import pytest

from engine.v2.foundation import DocumentError, content_hash, from_document, to_document
from engine.v2.models import (
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    ModelRelease,
    deployment,
    frozen_state_io,
    release_layout as layout,
    release_objects,
)
from engine.v2.models import serving_fold_preflight as folds
from engine.v2.models.serving_folds import (
    ChooserFoldPolicy,
    ChooserServingFoldDescriptor,
    ServingFoldDescriptor,
    SizeFoldPolicy,
)
from tests.v2.models.test_phase5_pinned_panel import ALL_ONE, build_pinned
from tests.v2.models.serving_fold_support import (
    catalog_object,
    chooser_policy,
    fold_bytes,
    fold_name,
    fold_states,
    published_manifest,
    release_inputs,
)
from tools import phase5_prepare_release as prep

OUTPUTS = {"implied_t1": ("driver_prediction", "pred_im_t1_d14"),
           "runup_move": ("pred_runup_abs_move_d14", "pred_runup_abs_move_d14")}


def test_shared_object_publication_orders_durability_and_preserves_retry(tmp_path, monkeypatch):
    payload = b"synthetic non-model state"
    digest = hashlib.sha256(payload).hexdigest()
    store = tmp_path / "deployment"
    dest = store / "objects" / digest
    dest.parent.mkdir(parents=True)
    old_temp = dest.with_suffix(".tmp")
    old_temp.write_bytes(b"unrelated former temporary file")
    events = []
    real_sync, real_link, real_dir_sync = os.fsync, os.link, release_objects.fsync_directory

    def sync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            events.append(("file", os.fstat(fd).st_size))
        real_sync(fd)

    def link(source, target):
        events.append(("link", Path(source)))
        assert Path(source).parent.parent == store / "tmp"
        real_link(source, target)

    def sync_directory(path):
        real_dir_sync(path)
        events.append(("directory", path))

    monkeypatch.setattr(release_objects.os, "fsync", sync)
    monkeypatch.setattr(release_objects.os, "link", link)
    monkeypatch.setattr(release_objects, "fsync_directory", sync_directory)
    expected = ("sha256:" + digest, "objects/" + digest)
    assert layout.write_object(tmp_path, payload) == expected
    assert [event[0] for event in events] == ["file", "link", "directory", "directory"]
    assert events[0][1] == len(payload)  # Flush precedes the file fsync.
    assert events[-2:] == [("directory", dest.parent), ("directory", store)]
    before = dest.stat()
    assert dest.read_bytes() == payload
    assert list((store / "tmp").iterdir()) == []
    events.clear()
    assert layout.write_object(tmp_path, payload) == expected
    after = dest.stat()
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert events == [("directory", dest.parent), ("directory", store)]
    assert old_temp.read_bytes() == b"unrelated former temporary file"


@pytest.mark.parametrize("same_payload", [False, True])
def test_shared_object_concurrent_publishers_use_unique_no_clobber_temps(tmp_path, monkeypatch, same_payload):
    payloads = [b"first object", b"first object" if same_payload else b"second object"]
    barrier, sources = threading.Barrier(2), []
    real_link = os.link

    def concurrent_link(source, dest):
        sources.append(Path(source))
        barrier.wait(timeout=5)
        real_link(source, dest)

    monkeypatch.setattr(release_objects.os, "link", concurrent_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda data: layout.write_object(tmp_path, data), payloads))
    assert len(set(sources)) == 2
    store = tmp_path / "deployment"
    for data, result in zip(payloads, results):
        digest = hashlib.sha256(data).hexdigest()
        assert result == ("sha256:" + digest, "objects/" + digest)
        assert (store / result[1]).read_bytes() == data
    assert len(list((store / "objects").iterdir())) == (1 if same_payload else 2)
    assert list((store / "tmp").iterdir()) == []


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("failure", ["write", "flush", "file-sync", "link", "object-dir", "store-dir"])
def test_object_publication_failure_never_publishes_catalog_and_retry_is_safe(
        tmp_path, role, failure, monkeypatch):
    cache, out = _cache(role), tmp_path / "release"
    data = fold_bytes(cache)
    dest = out / "deployment" / "objects" / hashlib.sha256(data).hexdigest()
    error = OSError(errno.EXDEV if failure == "link" else errno.EIO, "synthetic publication failure")
    handles = []
    real_open, real_fsync, real_link = Path.open, os.fsync, os.link
    real_dir_sync = release_objects.fsync_directory

    class FileProbe:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            self.handle.__enter__()
            return self

        def __exit__(self, *args):
            return self.handle.__exit__(*args)

        def write(self, value):
            if failure == "write":
                raise error
            return self.handle.write(value)

        def flush(self):
            if failure == "flush":
                raise error
            return self.handle.flush()

        def fileno(self):
            return self.handle.fileno()

    def open_file(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if path.parent.name.startswith("phase5-object-"):
            handles.append(handle)
            return FileProbe(handle)
        return handle

    def sync(fd):
        if failure == "file-sync" and any(not h.closed and h.fileno() == fd for h in handles):
            raise error
        real_fsync(fd)

    def link(source, target):
        if failure == "link":
            raise error
        real_link(source, target)

    def sync_directory(path):
        if path == (dest.parent if failure == "object-dir" else dest.parent.parent) and failure.endswith("dir"):
            raise error
        real_dir_sync(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", open_file)
        patch.setattr(release_objects.os, "fsync", sync)
        patch.setattr(release_objects.os, "link", link)
        patch.setattr(release_objects, "fsync_directory", sync_directory)
        with pytest.raises(layout.ReleaseLayoutError) as caught:
            prep.write_release(out, *release_inputs(role, cache), fold_states(role, cache))
    assert type(caught.value) is layout.ReleaseLayoutError
    assert str(caught.value) == "phase5 object could not be published"
    assert caught.value.__cause__ is error
    assert handles and all(handle.closed for handle in handles)
    assert list((out / "deployment" / "tmp").iterdir()) == []
    assert not (out / "phase5_release.json").exists()
    assert not (out / "deployment" / "releases" / "r1" / "phase5_release.json").exists()
    assert deployment.current_pointer(out / "deployment") is None
    linked = failure in ("object-dir", "store-dir")
    assert dest.exists() is linked
    before = dest.stat() if linked else None
    if linked:
        assert dest.read_bytes() == data
    synced = []

    def retry_sync(path):
        real_dir_sync(path)
        synced.append(path)

    monkeypatch.setattr(release_objects, "fsync_directory", retry_sync)
    prep.write_release(out, *release_inputs(role, cache), fold_states(role, cache))
    _assert_published_descriptor(out, role, cache)
    assert synced == [dest.parent, dest.parent.parent]
    if linked:
        after = dest.stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
    assert list((out / "deployment" / "tmp").iterdir()) == []


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("shape", ["wrong", "directory", "dangling", "fifo", "symlink"])
def test_object_publication_never_clobbers_a_racing_destination(tmp_path, role, shape, monkeypatch):
    cache, out = _cache(role), tmp_path / "release"
    real_link, planted = os.link, []
    outside = tmp_path / "outside"
    outside.write_bytes(b"independent target")

    def collision(source, target):
        target = Path(target)
        if shape == "wrong":
            target.write_bytes(b"wrong winner")
        elif shape == "directory":
            target.mkdir()
        elif shape == "fifo":
            os.mkfifo(target)
        else:
            target.symlink_to(outside if shape == "symlink" else tmp_path / "missing")
        planted.append((target, target.lstat()))
        real_link(source, target)

    monkeypatch.setattr(release_objects.os, "link", collision)
    expected = prep.PrepareRefused if shape in ("wrong", "symlink") else layout.ReleaseLayoutError
    with pytest.raises(expected) as caught:
        prep.write_release(out, *release_inputs(role, cache), fold_states(role, cache))
    assert type(caught.value) is expected
    if shape == "wrong":
        assert str(caught.value) == "chooser fold object bytes differ from preflight"
        assert caught.value.__cause__ is None
    elif shape == "symlink":
        assert str(caught.value) == "serving fold cannot be read"
        assert type(caught.value.__cause__) is OSError
        assert caught.value.__cause__.errno == errno.ELOOP
    else:
        assert str(caught.value) == "phase5 object could not be published"
        assert type(caught.value.__cause__) is FileExistsError
    assert len(planted) == 1
    target, before = planted[0]
    after = target.lstat()
    assert (after.st_ino, after.st_mtime_ns, after.st_mode) == (before.st_ino, before.st_mtime_ns, before.st_mode)
    if shape == "wrong":
        assert target.read_bytes() == b"wrong winner"
    assert outside.read_bytes() == b"independent target"
    assert list((out / "deployment" / "tmp").iterdir()) == []
    assert not (out / "phase5_release.json").exists()
    assert not (out / "deployment" / "releases" / "r1" / "phase5_release.json").exists()
    assert deployment.current_pointer(out / "deployment") is None


class NoInference:
    def predict(self, rows):
        raise AssertionError("authoring must not predict")

    def fit(self, *args):
        raise AssertionError("authoring must not fit")


def _cache(role, panel="a" * 64):
    return {"model_id": f"synthetic-{role}", "features": ["x", "y"],
            "tier3_snapshot": panel, "fold_start": "2026-02-01", "training_cutoff": "2026-01-31",
            "estimator": NoInference(), "pool_pred": [1.0, 3.0], "pool_res": [-0.5, 0.25]}


def _retry_inputs(role, cache):
    parts = [release_inputs(role, cache), release_inputs("auxiliary", _cache("auxiliary"), output=("left", "right"))]
    bindings = tuple(dataclasses.replace(part[0].bindings[0], members=(
        part[0].bindings[0].members[0], dataclasses.replace(part[0].bindings[0].members[0], name="metadata")))
        for part in parts)
    release = dataclasses.replace(parts[0][0], bindings=bindings)
    inventory = dataclasses.replace(parts[0][1], **{
        field: tuple(getattr(part[1], field)[0] for part in parts)
        for field in ("artifacts", "bindings", "requirements")})
    return release, inventory, {key: value for part in parts for key, value in part[2].items()}


def _assert_published_descriptor(root, role, cache, *, release_id="r1", clock="entry-close", floor=0.0, count=2,
                                 data=None):
    body, obj = catalog_object(root, role, name=fold_name(cache))
    data = fold_bytes(cache) if data is None else data
    digest = hashlib.sha256(data).hexdigest()
    parent = published_manifest(root / "deployment", release_id)
    expected = {
        "schema_version": "chooser_serving_fold_descriptor.v1.0",
        "parent_release_id": release_id, "parent_release_hash": parent.release_hash,
        "policy": {"schema_version": "chooser_fold_policy.v1.0", "role": role,
                   "model_id": cache["model_id"], "feature_order": cache["features"],
                   "panel_sha256": cache["tier3_snapshot"], "interval_floor": floor,
                   "interval_policy": "chooser-heldout-deciles.v1"},
        "fold_start": cache["fold_start"], "decision_clock_id": clock, "pool_count": count,
        "pool_pred_field": "pool_pred", "pool_res_field": "pool_res",
        "output_name": OUTPUTS[role][0] if role != "size" else "forecast_abs_move",
        "estimator": {"schema_version": "artifact_member.v1.0", "name": "estimator",
                      "path": f"objects/{digest}", "content_hash": f"sha256:{digest}"},
    }
    descriptor_type = ChooserServingFoldDescriptor
    if role == "size":
        descriptor_type = ServingFoldDescriptor
        expected.update(schema_version="serving_fold_descriptor.v1.0", role="size")
        expected["policy"].update(schema_version="size_fold_policy.v1.0", interval_policy="size-heldout-deciles.v1")
        del expected["policy"]["role"]
    else:
        expected["training_cutoff"] = cache["training_cutoff"]
        expected["producer_output"] = OUTPUTS[role][1]
    assert obj["serving_fold"] == expected
    descriptor = from_document(descriptor_type, obj["serving_fold"])
    assert to_document(descriptor) == expected
    assert descriptor.descriptor_hash == content_hash(descriptor if role == "size" else expected)
    assert obj["path"] == f"objects/{digest}"
    assert obj["content_hash"] == f"sha256:{digest}"
    assert obj["name"] == fold_name(cache)
    assert obj["bytes"] == len(data)
    return body, obj


def test_carried_chooser_fold_loads_from_verified_original_release(tmp_path, monkeypatch):
    from engine.v2.scoring import release_bindings as rb

    ModelNotReady = rb.ModelNotReady
    resolve_release_binding = rb.resolve_release_binding
    monkeypatch.setattr(rb, "_resolve_state_group", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rb, "_resolve_driver_residual_pools", lambda *_args: {})
    monkeypatch.setattr(rb, "_resolve_chooser_state_group", lambda *_args, **_kwargs: ())

    role = "implied_t1"
    cache = _cache(role)
    states = prep.build_states(
        {f"tier4_folds:{role}": {fold_name(cache): fold_bytes(cache)}},
        chooser_fold_policies={role: chooser_policy(role, cache)})
    prep.write_release(tmp_path, *release_inputs(role, cache, release_id="origin"), states)
    origin_body, origin_obj = _assert_published_descriptor(
        tmp_path, role, cache, release_id="origin")
    store = tmp_path / "deployment"
    deployment.mark_staging_succeeded(store, "origin")
    deployment.promote(store, "origin")

    original = resolve_release_binding(
        tmp_path, include_chooser_fold_references=True)
    origin_ref = original.chooser_fold_references[
        (OUTPUTS[role][1], cache["fold_start"])]
    assert origin_ref.release_id == origin_ref.carrying_release_id == "origin"
    assert origin_ref.training_cutoff == cache["training_cutoff"]

    release, inventory, payloads = release_inputs(role, cache, release_id="carried")
    deployment.stage_release(store, release, inventory, payloads)
    deployment.mark_staging_succeeded(store, "carried")
    deployment.promote(store, "carried")

    carried_body = copy.deepcopy(origin_body)
    carried_body["release_id"] = "carried"
    carried_body["sources"]["carried_from"] = "origin"

    def write_catalog(release_id, body):
        body["manifest_hash"] = content_hash(
            {key: value for key, value in body.items() if key != "manifest_hash"})
        path = deployment.manifest_path(store, release_id).with_name("phase5_release.json")
        path.write_text(json.dumps(body, sort_keys=True))

    write_catalog("carried", carried_body)
    carried = resolve_release_binding(
        tmp_path, include_chooser_fold_references=True)
    carried_ref = carried.chooser_fold_references[
        (OUTPUTS[role][1], cache["fold_start"])]
    assert carried_ref.release_id == "origin"
    assert carried_ref.carrying_release_id == "carried"
    assert carried_ref.descriptor_hash == content_hash(origin_obj["serving_fold"])
    assert carried_ref.training_cutoff == cache["training_cutoff"]

    from engine.v2.models import loader, serving_fold_loader

    references, contexts = rb.resolve_chooser_fold_authoring_contexts(tmp_path, "carried")
    replay_ref = references[(OUTPUTS[role][1], cache["fold_start"])]
    assert replay_ref.release_id == "origin"
    assert replay_ref.carrying_release_id == "carried"
    authoring_manifest, authoring_catalog = contexts[carried_ref.descriptor_hash]
    assert authoring_manifest.release.release_id == "origin"
    assert authoring_catalog["release_id"] == "origin"
    descriptor = from_document(ChooserServingFoldDescriptor, origin_obj["serving_fold"])
    verified = serving_fold_loader.load_chooser_serving_fold(
        loader.FrozenInference(store), descriptor,
        verified_parent_manifest=authoring_manifest,
        verified_catalog=dict(authoring_catalog))
    assert verified.descriptor.descriptor_hash == carried_ref.descriptor_hash

    carried_body_without_link = copy.deepcopy(carried_body)
    del carried_body_without_link["sources"]["carried_from"]
    write_catalog("carried", carried_body_without_link)
    with pytest.raises(ModelNotReady) as refusal:
        resolve_release_binding(tmp_path, include_chooser_fold_references=True)
    assert refusal.value.code == rb.MODEL_NOT_READY
    write_catalog("carried", copy.deepcopy(carried_body))

    carried_body_with_changed_membership = copy.deepcopy(carried_body)
    changed_row = next(
        row for row in carried_body_with_changed_membership["members"]
        if row["member_id"] == f"tier4_folds:{role}")
    changed_row["objects"][0]["serving_fold"]["parent_release_hash"] = "sha256:" + "1" * 64
    write_catalog("carried", carried_body_with_changed_membership)
    with pytest.raises(ModelNotReady) as refusal:
        resolve_release_binding(tmp_path, include_chooser_fold_references=True)
    assert refusal.value.code == rb.MODEL_NOT_READY
    write_catalog("carried", copy.deepcopy(carried_body))

    tampered_origin = copy.deepcopy(origin_body)
    tampered_carried = copy.deepcopy(carried_body)
    for release_id, body in (("origin", tampered_origin), ("carried", tampered_carried)):
        row = next(row for row in body["members"]
                   if row["member_id"] == f"tier4_folds:{role}")
        row["objects"][0]["serving_fold"]["parent_release_hash"] = "sha256:" + "0" * 64
        write_catalog(release_id, body)

    with pytest.raises(ModelNotReady) as refusal:
        resolve_release_binding(tmp_path, include_chooser_fold_references=True)
    assert refusal.value.code == rb.MODEL_NOT_READY


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("count,month,clock,features,floor,release_id,training_cutoff", [
    (2, "2026-02-01", "entry-close", ["x", "y"], 0.0, "r1", "2026-01-31"),
    (0, "2026-05-01", "next-open", ["b", "c", "a"], None, "empty-release", "2026-04-30"),
    (1, "2025-11-01", "next-open", ["y", "x"], 1.25, "thin-release", "2025-10-31"),
])
def test_real_authoring_preserves_exact_role_identity_and_metadata(
        tmp_path, role, count, month, clock, features, floor, release_id, training_cutoff):
    cache = _cache(role)
    cache.update(model_id=f"{release_id}-{role}", fold_start=month, features=features,
                 training_cutoff=training_cutoff, pool_pred=[1.0] * count,
                 pool_res=[-0.5] * count)
    args = release_inputs(role, cache, clock=clock, release_id=release_id)
    states = prep.build_states({f"tier4_folds:{role}": {fold_name(cache): fold_bytes(cache)}},
                               chooser_fold_policies={role: chooser_policy(role, cache, floor=floor)})
    prep.write_release(tmp_path, *args, states)
    body, obj = _assert_published_descriptor(tmp_path, role, cache, release_id=release_id,
                                            clock=clock, floor=floor, count=count)
    descriptor = from_document(ChooserServingFoldDescriptor, obj["serving_fold"])
    assert to_document(descriptor) == obj["serving_fold"]
    with pytest.raises(DocumentError) as error:
        from_document(ServingFoldDescriptor, obj["serving_fold"])
    assert error.value.code == "UNSUPPORTED_VERSION"
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.policy.role = "size"
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.parent_release_id = "mutated"
    changed = dataclasses.replace(descriptor, policy=dataclasses.replace(
        descriptor.policy, interval_floor=2.5))
    assert changed.descriptor_hash != descriptor.descriptor_hash
    unsigned = {key: value for key, value in body.items() if key != "manifest_hash"}
    assert content_hash(unsigned) == body["manifest_hash"]
    prep.write_release(tmp_path, *args, states)
    assert layout.read_manifest(tmp_path) == body
    assert deployment.current_pointer(tmp_path / "deployment") is None


@pytest.mark.parametrize("identity", ["parent_release_id", "parent_release_hash", "role", "estimator", "training_cutoff"])
def test_descriptor_hash_binds_parent_role_estimator_and_training_cutoff(tmp_path, identity):
    role, cache = "implied_t1", _cache("implied_t1")
    prep.write_release(tmp_path, *release_inputs(role, cache), fold_states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    expected = json.loads(json.dumps(obj["serving_fold"]))
    original = from_document(ChooserServingFoldDescriptor, expected)
    if identity == "role":
        expected["policy"]["role"] = "runup_move"
        expected.update(output_name="pred_runup_abs_move_d14", producer_output="pred_runup_abs_move_d14")
        changed = dataclasses.replace(original, policy=dataclasses.replace(original.policy, role="runup_move"),
                                      output_name="pred_runup_abs_move_d14", producer_output="pred_runup_abs_move_d14")
    elif identity == "estimator":
        expected["estimator"]["content_hash"] = "sha256:" + "b" * 64
        expected["estimator"]["path"] = "objects/" + "b" * 64
        changed = dataclasses.replace(original, estimator=dataclasses.replace(
            original.estimator, content_hash="sha256:" + "b" * 64, path="objects/" + "b" * 64))
    elif identity == "training_cutoff":
        value = "2026-01-30"
        expected[identity] = value
        changed = dataclasses.replace(original, training_cutoff=value)
    else:
        value = "other-release" if identity == "parent_release_id" else "sha256:" + "c" * 64
        expected[identity] = value
        changed = dataclasses.replace(original, **{identity: value})
    assert changed.descriptor_hash == content_hash(expected)
    assert changed.descriptor_hash != original.descriptor_hash


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("via_document", [False, True])
@pytest.mark.parametrize("bad_panel", ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64,
                                       "sha256:" + "a" * 64, "a" * 64 + "\n"])
def test_chooser_policy_refuses_noncanonical_panel_digest(role, via_document, bad_panel):
    valid = chooser_policy(role, _cache(role))
    with pytest.raises(DocumentError) as error:
        if via_document:
            from_document(ChooserFoldPolicy, {**to_document(valid), "panel_sha256": bad_panel})
        else:
            dataclasses.replace(valid, panel_sha256=bad_panel)
    assert type(error.value) is DocumentError
    assert error.value.code == "BAD_PANEL_HASH"
    assert error.value.path == "$.panel_sha256"
    assert str(error.value) == "BAD_PANEL_HASH at $.panel_sha256: expected 64 lowercase hex digits"
    assert error.value.__cause__ is None


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("via_document", [False, True])
@pytest.mark.parametrize("defect", ["output", "producer", "both", "path", "hash", "uppercase",
                                    "prefix", "short", "nonhex", "absolute", "traversal", "panel"])
def test_chooser_descriptor_refuses_inconsistent_identity(tmp_path, role, via_document, defect):
    cache = _cache(role)
    prep.write_release(tmp_path, *release_inputs(role, cache), fold_states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    document = obj["serving_fold"]
    if defect in ("output", "producer", "both"):
        other = next(other for other in OUTPUTS if other != role)
        if defect in ("output", "both"):
            document["output_name"] = OUTPUTS[other][0]
        if defect in ("producer", "both"):
            document["producer_output"] = OUTPUTS[other][1]
        code, path, message = "CHOOSER_OUTPUT_MISMATCH", "$", "outputs disagree with chooser role"
    elif defect == "panel":
        document["policy"]["panel_sha256"] = "A" * 64
        code, path, message = ("BAD_PANEL_HASH", "$.policy.panel_sha256" if via_document else "$.panel_sha256",
                               "expected 64 lowercase hex digits")
    else:
        member = document["estimator"]
        if defect in ("path", "absolute", "traversal"):
            member["path"] = {"path": "objects/" + "b" * 64, "absolute": "/synthetic/elsewhere",
                              "traversal": "objects/../elsewhere"}[defect]
        else:
            member["content_hash"] = {"hash": "sha256:" + "b" * 64, "uppercase": "sha256:" + "B" * 64,
                                      "prefix": "b" * 64, "short": "sha256:" + "b" * 63,
                                      "nonhex": "sha256:" + "g" * 64}[defect]
            if defect != "hash":
                member["path"] = "objects/" + member["content_hash"].removeprefix("sha256:")
        code, path, message = "BAD_FOLD_OBJECT", "$.estimator", "expected matching content-addressed cache object"
    with pytest.raises(DocumentError) as error:
        if via_document:
            from_document(ChooserServingFoldDescriptor, document)
        else:
            fields = {key: value for key, value in document.items() if key not in ("policy", "estimator")}
            ChooserServingFoldDescriptor(**fields, policy=ChooserFoldPolicy(**document["policy"]),
                                          estimator=ArtifactMember(**document["estimator"]))
    assert type(error.value) is DocumentError
    assert (error.value.code, error.value.path) == (code, path)
    assert str(error.value) == f"{code} at {path}: {message}"
    if defect == "panel" and via_document:
        assert type(error.value.__cause__) is DocumentError
        assert error.value.__cause__.path == "$.panel_sha256"
    else:
        assert error.value.__cause__ is None


@pytest.mark.parametrize("role", OUTPUTS)
def test_policy_absence_keeps_opaque_catalog_compatibility(tmp_path, role):
    cache = _cache(role)
    prep.write_release(tmp_path, *release_inputs(role, cache), fold_states(role, cache, explicit=False))
    _body, obj = catalog_object(tmp_path, role)
    assert "serving_fold" not in obj


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("pinned_mode", [False, True])
@pytest.mark.parametrize("bad_later_month", [False, True])
def test_every_month_is_preflighted_and_published_before_catalog(
        tmp_path, role, pinned_mode, bad_later_month):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, data = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(data).hexdigest()
    first = _cache(role, panel_hash)
    later = _cache(role, panel_hash)
    later.update(fold_start="2026-03-01", pool_pred=[4.5],
                 pool_res=[float("nan") if bad_later_month else -1.25])
    # Insert later first so successful ordering cannot rely on mapping insertion order.
    payloads = {fold_name(later): fold_bytes(later), fold_name(first): fold_bytes(first)}
    states = prep.build_states({f"tier4_folds:{role}": payloads},
                               chooser_fold_policies={role: chooser_policy(role, first)})
    out = tmp_path / "out"
    args = release_inputs(role, first)
    if bad_later_month:
        with pytest.raises(prep.PrepareRefused) as error:
            prep.write_release(out, *args, states, pinned_panel=pinned)
        assert type(error.value) is prep.PrepareRefused
        assert str(error.value) == "chooser fold refused: invalid serving fold held-out pool"
        assert type(error.value.__cause__) is ValueError
        assert str(error.value.__cause__) == "invalid serving fold held-out pool"
        assert not out.exists()
        assert deployment.current_pointer(out / "deployment") is None
        return
    prep.write_release(out, *args, states, pinned_panel=pinned)
    body = layout.read_manifest(out)
    row = next(row for row in body["members"] if row["member_id"] == f"tier4_folds:{role}")
    assert [obj["name"] for obj in row["objects"]] == [fold_name(first), fold_name(later)]
    first_obj = _assert_published_descriptor(out, role, first, count=2)[1]
    later_obj = _assert_published_descriptor(out, role, later, count=1)[1]
    assert first_obj["content_hash"] != later_obj["content_hash"]
    assert content_hash(first_obj["serving_fold"]) != content_hash(later_obj["serving_fold"])
    assert content_hash({key: value for key, value in body.items() if key != "manifest_hash"}) == body["manifest_hash"]
    if pinned_mode:
        assert body["sources"]["panel_copy_sha256"] == panel_hash
    prep.write_release(out, *args, states, pinned_panel=pinned)
    assert layout.read_manifest(out) == body
    assert deployment.current_pointer(out / "deployment") is None


def test_size_and_both_chooser_roles_coexist_without_relabeling(tmp_path):
    caches = {role: _cache(role) for role in OUTPUTS}
    parts = [release_inputs(role, cache) for role, cache in caches.items()]
    size = _cache("size")
    parts.append(release_inputs("size", size, output="forecast_abs_move"))
    release = dataclasses.replace(parts[0][0], bindings=tuple(p[0].bindings[0] for p in parts))
    inventory = dataclasses.replace(parts[0][1], **{
        field: tuple(getattr(p[1], field)[0] for p in parts)
        for field in ("artifacts", "bindings", "requirements")})
    payloads = {key: value for p in parts for key, value in p[2].items()}
    states = prep.build_states(
        {"tier4_folds:size": {fold_name(size): fold_bytes(size)},
         **{f"tier4_folds:{role}": {fold_name(cache): fold_bytes(cache)} for role, cache in caches.items()}},
        size_fold_policy=SizeFoldPolicy(model_id=size["model_id"], feature_order=("x", "y"),
                                       panel_sha256="a" * 64, interval_floor=0.0),
        chooser_fold_policies={role: chooser_policy(role, cache) for role, cache in caches.items()})
    prep.write_release(tmp_path, release, inventory, payloads, states)
    descriptors = [_assert_published_descriptor(tmp_path, role, caches[role])[1]["serving_fold"]
                   for role in OUTPUTS]
    assert [d["policy"]["role"] for d in descriptors] == ["implied_t1", "runup_move"]
    assert [d["producer_output"] for d in descriptors] == ["pred_im_t1_d14", "pred_runup_abs_move_d14"]
    assert len({d["estimator"]["content_hash"] for d in descriptors}) == 2
    size_descriptor = catalog_object(tmp_path, "size")[1]["serving_fold"]
    assert size_descriptor["schema_version"] == "serving_fold_descriptor.v1.0"
    assert size_descriptor["role"] == "size"
    assert size_descriptor["output_name"] == "forecast_abs_move"
    assert {d["schema_version"] for d in descriptors} == {"chooser_serving_fold_descriptor.v1.0"}
    assert len({d["parent_release_hash"] for d in [size_descriptor, *descriptors]}) == 1


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("pinned_mode", [False, True])
@pytest.mark.parametrize("reorder", ["bindings", "members", "both"])
def test_semantically_identical_reordered_retries_keep_original_manifest_and_catalog(
        tmp_path, role, pinned_mode, reorder):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, data = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(data).hexdigest()
    cache = _cache(role, panel_hash)
    release, inventory, payloads = _retry_inputs(role, cache)
    states, out = fold_states(role, cache), tmp_path / "out"
    prep.write_release(out, release, inventory, payloads, states, pinned_panel=pinned)
    catalog, _obj = _assert_published_descriptor(out, role, cache)
    manifest = published_manifest(out / "deployment")
    bindings = release.bindings
    if reorder in ("members", "both"):
        bindings = tuple(dataclasses.replace(binding, members=tuple(reversed(binding.members)))
                         for binding in bindings)
    if reorder in ("bindings", "both"):
        bindings = tuple(reversed(bindings))
    reordered = dataclasses.replace(release, bindings=bindings)
    assert reordered != release
    prep.write_release(out, reordered, inventory, payloads, states, pinned_panel=pinned)
    assert published_manifest(out / "deployment") == manifest
    assert _assert_published_descriptor(out, role, cache)[0] == catalog
    assert deployment.current_pointer(out / "deployment") is None


@pytest.mark.parametrize("role,expected", [
    ("implied_t1", "sha256:d046274a8fe00a5ab5b167671fb592ec510d29083b4ef555fd622a38ee02b8a0"),
    ("runup_move", "sha256:9e9f90638fe05f02f1a57569a79acba6380a72e7f9c3e32f2a71f3370d1f6adb"),
])
def test_public_semantic_release_hash_preserves_preexisting_identity(role, expected):
    # Goldens captured from the existing calculation before exposing its public name.
    release, _inventory, _payloads = _retry_inputs(role, _cache(role))
    assert "semantic_release_hash" in deployment.__all__
    assert deployment.semantic_release_hash(release) == expected
    reordered = dataclasses.replace(release, bindings=tuple(dataclasses.replace(
        binding, members=tuple(dataclasses.replace(member, path=f"relocated/{member.name}")
                               for member in reversed(binding.members)))
        for binding in reversed(release.bindings)))
    assert deployment.semantic_release_hash(reordered) == expected


def test_public_semantic_release_hash_accepts_incomplete_release_without_io(monkeypatch):
    release = ModelRelease(release_id="incomplete", deployment_id="offline", bindings=())
    canonical = (b'{"bindings":[],"deployment_id":"offline","release_id":"incomplete",'
                 b'"schema_version":"model_release.v1.0"}')
    expected = "sha256:" + hashlib.sha256(canonical).hexdigest()

    def forbidden(*args, **kwargs):
        pytest.fail("pure release identity must not perform I/O or readiness validation")

    monkeypatch.setattr(deployment, "require_complete_release", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    assert deployment.semantic_release_hash(release) == expected
    assert release.bindings == ()


@pytest.mark.parametrize("role", OUTPUTS)
def test_descriptor_parent_calls_public_release_identity(tmp_path, role, monkeypatch):
    cache = _cache(role)
    prep.write_release(tmp_path, *_retry_inputs(role, cache), fold_states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    manifest = published_manifest(tmp_path / "deployment")
    policy = chooser_policy(role, cache)
    validated = folds.preflight_chooser_fold(fold_bytes(cache), fold_name(cache), policy, manifest.release)
    calls = []

    def public_hash(release):
        calls.append(release)
        return deployment.semantic_release_hash(release)

    monkeypatch.setattr(folds, "semantic_release_hash", public_hash)
    descriptor = folds.descriptor_from_preflight(validated, policy, manifest)
    assert calls == [manifest.release, validated.release]
    assert to_document(descriptor) == obj["serving_fold"]


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("legacy_hash", [False, True])
def test_parent_storage_paths_do_not_change_semantics_or_manifest_hash_ownership(tmp_path, role, legacy_hash):
    cache = _cache(role)
    inputs = _retry_inputs(role, cache)
    prep.write_release(tmp_path, *inputs, fold_states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    manifest = published_manifest(tmp_path / "deployment")
    if legacy_hash:
        historical_hashes = {
            "implied_t1": "sha256:f26c05ef8b88b7503efdc86001c94bdb746a813e6b167accbfa046e285038494",
            "runup_move": "sha256:48886726edaadf45cef16494b3d57dd449eed0aa1f74a14585ba624ac0ab0b95",
        }
        manifest = dataclasses.replace(manifest,
            release_hash=historical_hashes[role],
            release_hash_version=deployment.RELEASE_HASH_MEMBER_V1)
        (tmp_path / "deployment" / "releases" / "r1" / "manifest.json").write_text(
            json.dumps(to_document(manifest)))
    # Idempotent staging validates the persisted semantic/historical hash through its public boundary.
    assert deployment.stage_release(tmp_path / "deployment", *inputs) == manifest
    relocated = dataclasses.replace(manifest.release, bindings=tuple(dataclasses.replace(
        binding, members=tuple(dataclasses.replace(member, path=f"relocated/{member.name}")
                               for member in binding.members)) for binding in manifest.release.bindings))
    assert relocated != manifest.release
    policy = chooser_policy(role, cache)
    validated = folds.preflight_chooser_fold(fold_bytes(cache), fold_name(cache), policy, relocated)
    descriptor = folds.descriptor_from_preflight(validated, policy, manifest)
    expected = {**obj["serving_fold"], "parent_release_hash": manifest.release_hash}
    assert to_document(descriptor) == expected
    assert descriptor.descriptor_hash == content_hash(expected)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("binding_index", [0, 1], ids=["validated-role", "auxiliary-role"])
@pytest.mark.parametrize("identity", ["release_id", "deployment_id", "model_id", "adapter", "clock",
                                     "feature_order", "output_order", "member_hash", "member_name"])
def test_retained_parent_guard_rejects_actual_identity_changes(tmp_path, role, identity, binding_index):
    cache, policy = _cache(role), chooser_policy(role, _cache(role))
    prep.write_release(tmp_path, *_retry_inputs(role, cache), fold_states(role, cache))
    manifest = published_manifest(tmp_path / "deployment")
    validated = folds.preflight_chooser_fold(fold_bytes(cache), fold_name(cache), policy, manifest.release)
    release, selected = manifest.release, manifest.release.bindings[binding_index]
    if identity in ("release_id", "deployment_id"):
        release = dataclasses.replace(release, **{identity: "different"})
    else:
        changes = {"model_id": {"model_id": "different"}, "adapter": {"adapter": "different"},
                   "clock": {"decision_clock_id": "different"},
                   "feature_order": {"feature_order": tuple(reversed(selected.feature_order))},
                   "output_order": {"output_names": tuple(reversed(selected.output_names))
                                    if len(selected.output_names) > 1 else ("different",)}}
        if identity in ("member_hash", "member_name"):
            field, value = ("content_hash", "sha256:" + "b" * 64) if identity == "member_hash" else ("name", "different")
            changed = dataclasses.replace(selected.members[0], **{field: value})
            changes[identity] = {"members": (changed, *selected.members[1:])}
        bindings = list(release.bindings)
        bindings[binding_index] = dataclasses.replace(selected, **changes[identity])
        release = dataclasses.replace(release, bindings=tuple(bindings))
    # Keep the asserted hash unchanged: compare actual releases, not just caller-supplied hash strings.
    substituted = dataclasses.replace(manifest, release=release)
    with pytest.raises(ValueError) as error:
        folds.descriptor_from_preflight(validated, policy, substituted)
    assert type(error.value) is ValueError
    assert str(error.value) == "staged release disagrees with validated fold"
    assert error.value.__cause__ is None
    assert published_manifest(tmp_path / "deployment") == manifest


def test_chooser_preflight_refuses_missing_training_cutoff():
    cache = _cache("implied_t1")
    cache.pop("training_cutoff")
    data = fold_bytes(cache)
    policy = chooser_policy("implied_t1", cache)
    release = release_inputs("implied_t1", cache)[0]
    with pytest.raises(ValueError, match="invalid chooser training cutoff"):
        folds.preflight_chooser_fold(data, fold_name(cache), policy, release)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_thin_pools_keep_explicit_counts(role, count):
    cache = _cache(role)
    cache.update(pool_pred=[1.0] * count, pool_res=[-0.5] * count)
    cache["training_cutoff"] = "2026-01-31"
    data, policy, release = fold_bytes(cache), chooser_policy(role, cache), release_inputs(role, cache)[0]
    validated = folds.preflight_chooser_fold(data, fold_name(cache), policy, release)
    expected = {"fold_start": "2026-02-01", "decision_clock_id": "entry-close",
                "digest": hashlib.sha256(data).hexdigest(), "pool_count": count,
                "policy": policy, "release": release}
    expected["training_cutoff"] = "2026-01-31"
    assert vars(validated) == expected
    assert {field.name for field in dataclasses.fields(validated)} == set(expected)
    with pytest.raises(dataclasses.FrozenInstanceError):
        validated.pool_count = 100


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("defect", ["role", "model", "features", "clock-release", "output",
                                    "duplicate", "schema", "panel",
                                    "month", "name", "estimator", "absent", "shape", "length", "nan", "decode"])
def test_bad_inputs_refuse_before_output_creation(tmp_path, role, defect):
    cache = _cache(role)
    release, inv, payloads = release_inputs(role, cache)
    policy, name = chooser_policy(role, cache), fold_name(cache)
    if defect in ("role", "model", "features", "output", "clock-release", "duplicate"):
        binding = release.bindings[0]
        changes = {"role": {"role": "size"}, "model": {"model_id": "wrong"},
                   "features": {"feature_order": ("y", "x")},
                   "output": {"output_names": ("runup_move_prediction",)},
                   "clock-release": {"decision_clock_id": "wrong"}, "duplicate": {}}
        binding = dataclasses.replace(binding, **changes[defect])
        release = dataclasses.replace(release, bindings=(binding,) * (2 if defect == "duplicate" else 1))
    elif defect == "schema":
        policy = dataclasses.replace(policy, schema_version="wrong")
    elif defect == "panel":
        cache["tier3_snapshot"] = "a" * 12 + "b" * 52
    elif defect == "month":
        cache["fold_start"] = "2026-02-02"
    elif defect == "name":
        name = "wrong.joblib"
    elif defect in ("estimator", "absent"):
        cache.pop("estimator" if defect == "estimator" else "pool_res")
    elif defect != "decode":
        cache["pool_res"] = {"shape": [[1.0, 2.0]], "length": [1.0],
                             "nan": [float("nan"), 1.0]}[defect]
    data = b"not a joblib artifact" if defect == "decode" else fold_bytes(cache)
    states = prep.build_states({f"tier4_folds:{role}": {name: data}},
                               chooser_fold_policies={role: policy})
    out = tmp_path / "out"
    expected_type = deployment.StagingRefused if defect == "clock-release" else prep.PrepareRefused
    with pytest.raises(expected_type) as error:
        prep.write_release(out, release, inv, payloads, states)
    assert type(error.value) is expected_type
    if defect == "clock-release":
        assert error.value.code == "MODEL_DEPLOYMENT_REFUSED"
        assert {(issue.path, issue.code) for issue in error.value.issues} == {
            ("$.bindings", "MISSING_INFERENCE_BINDING"),
            (f"$.bindings[{role}:*]", "UNBOUND_IN_INVENTORY"),
        }
    else:
        if defect in ("role", "model", "features", "output", "duplicate"):
            diagnostic = f"{role} serving policy disagrees with release binding"
        elif defect == "schema":
            diagnostic = "invalid chooser serving policy"
        elif defect in ("panel", "estimator"):
            diagnostic = "serving fold header disagrees with policy"
        elif defect == "month":
            diagnostic = "invalid serving fold month"
        elif defect == "name":
            diagnostic = "serving fold filename disagrees with header"
        elif defect == "decode":
            diagnostic = "serving fold cannot be decoded"
        else:
            diagnostic = "invalid serving fold held-out pool"
        assert str(error.value) == f"chooser fold refused: {diagnostic}"
        assert type(error.value.__cause__) is ValueError
        assert str(error.value.__cause__) == diagnostic
    if defect != "clock-release":  # Deployment, not the cache header, owns clock-inventory compatibility.
        assert not out.exists()
    assert not (out / "phase5_release.json").exists()
    assert deployment.current_pointer(out / "deployment") is None


@pytest.mark.parametrize("role", OUTPUTS)
def test_bound_precedes_decode_and_each_authoring_decodes_once(tmp_path, role, monkeypatch):
    cache = _cache(role)
    data, name, policy = fold_bytes(cache), fold_name(cache), chooser_policy(role, cache)
    calls, real = [], joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(folds, "MAX_FOLD_BYTES", len(data) - 1)
    with pytest.raises(ValueError) as error:
        folds.preflight_chooser_fold(data, name, policy, release_inputs(role, cache)[0])
    assert type(error.value) is ValueError
    assert str(error.value) == "serving fold exceeds byte limit"
    assert error.value.__cause__ is None
    assert calls == []
    monkeypatch.setattr(folds, "MAX_FOLD_BYTES", len(data))
    prep.write_release(tmp_path, *release_inputs(role, cache), fold_states(role, cache))
    assert calls == [1]
    prep.write_release(tmp_path, *release_inputs(role, cache), fold_states(role, cache))
    assert calls == [1, 1]


@pytest.mark.parametrize("role", ["size", *OUTPUTS])
@pytest.mark.parametrize("pinned_mode", [False, True])
@pytest.mark.parametrize("protocol", [0, 1, 2, 4, 5])
def test_uncompressed_protocols_preserve_complete_published_fold_identity(
        tmp_path, monkeypatch, role, pinned_mode, protocol):
    import numpy as np

    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, panel = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(panel).hexdigest()
    cache = _cache(role, panel_hash)
    cache.update(pool_pred=np.array(cache["pool_pred"]), pool_res=np.array(cache["pool_res"]))
    raw = fold_bytes(cache, compress=0, protocol=protocol)
    inputs = release_inputs(role, cache, output="forecast_abs_move" if role == "size" else None)
    states = fold_states(role, cache, data=raw)
    loads, real_load = [], joblib.load

    def load(stream, *args, **kwargs):
        loads.append(stream.getvalue())
        return real_load(stream, *args, **kwargs)

    monkeypatch.setattr(joblib, "load", load)
    out = tmp_path / "out"
    prep.write_release(out, *inputs, states, pinned_panel=pinned)
    body, obj = _assert_published_descriptor(out, role, cache, data=raw)
    assert (out / "deployment" / obj["path"]).read_bytes() == raw
    assert body["manifest_hash"] == content_hash({key: value for key, value in body.items()
                                                if key != "manifest_hash"})
    if pinned is not None:
        assert body["sources"]["panel_copy_sha256"] == panel_hash
        assert body["sources"]["panel_copy_snapshot_id"] == pinned.snapshot_id
        assert body["sources"]["panel_copy_mode"] == "COPY"
    else:
        assert "panel_copy_sha256" not in body["sources"]
    prep.write_release(out, *inputs, states, pinned_panel=pinned)
    assert layout.read_manifest(out) == body
    assert loads == [raw, raw]
    assert deployment.current_pointer(out / "deployment") is None


class ConstantChampion(NoInference):
    def predict(self, rows):
        return [2.5] * len(rows)


@pytest.mark.parametrize("role", ["size", *OUTPUTS])
@pytest.mark.parametrize("codec", ["zlib", "gzip", "bz2", "lzma", "xz"])
def test_compressed_champion_remains_supported_by_real_preparation_and_loader(
        tmp_path, monkeypatch, role, codec):
    from engine.data.features import tier4

    cache = _cache(role)
    _release, inventory, _payloads = release_inputs(
        role, cache, output="forecast_abs_move" if role == "size" else None)
    champion = fold_bytes(SimpleNamespace(features=tuple(cache["features"]), model=ConstantChampion()),
                      compress=(codec, 3))
    champion_hash = "sha256:" + hashlib.sha256(champion).hexdigest()
    source = tmp_path / "champion.joblib"
    source.write_bytes(champion)
    artifact = inventory.artifacts[0]
    member = dataclasses.replace(artifact.members[0], artifact_ref=str(source), content_hash=champion_hash)
    inventory = dataclasses.replace(inventory, artifacts=(dataclasses.replace(artifact, members=(member,)),))

    def never(*args, **kwargs):
        pytest.fail("registry, fitting, or live data read in synthetic champion compatibility test")

    monkeypatch.setattr(tier4.registry_mod, "load_registry", never)
    monkeypatch.setattr(tier4.store, "read_table", never)
    monkeypatch.setattr(tier4.store, "file_sha256", never)
    monkeypatch.setattr(tier4, "serving_model", never)
    release, payloads = prep.model_release(inventory)
    out = tmp_path / "out"
    prep.write_release(out, release, inventory, payloads, fold_states(role, cache))
    _assert_published_descriptor(out, role, cache)
    staged = published_manifest(out / "deployment")
    binding = staged.release.bindings[0]
    assert binding.adapter == "joblib-estimator.v1"
    assert binding.members[0].content_hash == champion_hash
    assert (out / "deployment" / binding.members[0].path).read_bytes() == champion
    result = FrozenInference(out / "deployment").infer(staged.release, InferenceRequest(
        release_id="r1", binding_id=f"{role}:*", feature_order=tuple(cache["features"]),
        rows=((1.0, 3.0), (2.0, 4.0))))
    assert result.status == "READY"
    assert result.predictions == ((2.5,), (2.5,))
    assert result.output_names == binding.output_names
    assert result.artifact_hashes == (champion_hash,)


@pytest.mark.parametrize("role", OUTPUTS)
def test_pinned_authoring_requires_full_panel_hash_and_retained_destination(tmp_path, role, monkeypatch):
    (tmp_path / "pinned").mkdir()
    pinned, _store, _snapshot, _records, data = build_pinned(tmp_path / "pinned", ALL_ONE)
    panel_hash = hashlib.sha256(data).hexdigest()
    cache = _cache(role, panel_hash)
    out = tmp_path / "good"
    prep.write_release(out, *release_inputs(role, cache), fold_states(role, cache), pinned_panel=pinned)
    body, obj = catalog_object(out, role)
    assert body["sources"]["panel_copy_sha256"] == panel_hash
    assert obj["serving_fold"]["policy"]["panel_sha256"] == panel_hash
    bad = _cache(role, panel_hash[:12] + "0" * 52)
    refused = tmp_path / "bad-panel"
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(refused, *release_inputs(role, bad), fold_states(role, bad), pinned_panel=pinned)
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold policy does not match the pinned panel"
    assert error.value.__cause__ is None
    assert not refused.exists()
    real = prep.write_object

    def wrong_destination(root, payload):
        digest, rel = real(root, payload)
        (layout.deployment_root(root) / rel).write_bytes(b"substituted")
        return digest, rel

    monkeypatch.setattr(prep, "write_object", wrong_destination)
    refused = tmp_path / "bad-destination"
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(refused, *release_inputs(role, cache), fold_states(role, cache), pinned_panel=pinned)
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold object bytes differ from preflight"
    assert error.value.__cause__ is None
    assert not (refused / "phase5_release.json").exists()
    assert deployment.current_pointer(refused / "deployment") is None


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("panel_mode", ["unpinned-default", "unpinned-distinct", "pinned-auto", "pinned-explicit"])
@pytest.mark.parametrize("failure,features,floor", [
    (None, ("x", "y"), 0.0), (None, ("y", "x"), 1.25),
    (None, ("b", "c", "a"), None), ("output", ("b", "c", "a"), 1.25),
    ("recipe", ("b", "c", "a"), None), ("oversize", ("x", "y"), 0.0),
    ("symlink", ("x", "y"), 0.0), ("cache", ("x", "y"), 0.0),
])
def test_real_cli_obtains_role_policy_without_fitting_or_data_reads(
        tmp_path, role, monkeypatch, capsys, failure, features, floor, panel_mode):
    from engine.data.features import tier4
    from engine.v2.models import inventory

    pinned, panel_hash = None, "a" * 64
    if panel_mode.startswith("pinned-"):
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, panel_bytes = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(panel_bytes).hexdigest()
    elif panel_mode == "unpinned-distinct":
        panel_hash = hashlib.sha256(b"distinct synthetic unpinned panel bytes").hexdigest()
    if panel_mode != "unpinned-default":
        assert panel_hash != "a" * 64
    requested_snapshot = ("auto" if panel_mode == "pinned-auto" else
                          "sha256:" + panel_hash if panel_mode == "pinned-explicit" else panel_hash)
    cache = _cache(role, panel_hash)
    cache["features"] = list(features)
    if failure == "cache":
        cache["pool_res"] = [float("nan"), 0.0]
    release, inv, payloads = release_inputs(role, cache)
    monkeypatch.setattr(inventory, "current_release_inventory", lambda: (inv, ()))
    monkeypatch.setattr(prep, "model_release", lambda *a, **k: (release, payloads))
    monkeypatch.setattr(prep, "modules_available", lambda *a: (False, "unused"))
    collect_payloads = prep._collect_payloads

    def collect_with_pool(*args, **kwargs):
        found = collect_payloads(*args, **kwargs)
        found["chooser_analog_pool"] = {"chooser_pool|2026-09-30": b"pool"}
        return found

    monkeypatch.setattr(prep, "_collect_payloads", collect_with_pool)
    monkeypatch.setattr(tier4, "size_feature_model", lambda: SimpleNamespace(
        model_id="unused-size", features=("size-x",), interval_floor=0.0, produces="pred_abs_move"))
    calls = []
    recipe_error = tier4.Tier4Error("private recipe feature or target detail")

    def producer():
        calls.append(role)
        if failure == "recipe":
            raise recipe_error
        return SimpleNamespace(model_id=cache["model_id"], features=features,
                               interval_floor=floor, produces="scaled" if failure == "output" else OUTPUTS[role][1])

    def never(*args, **kwargs):
        pytest.fail("unrelated producer, fitting or live data read")

    for candidate, method in (("implied_t1", "im_t1_feature_model"), ("runup_move", "runup_move_feature_model")):
        monkeypatch.setattr(tier4, method, producer if role == candidate else never)
    monkeypatch.setattr(tier4, "serving_model", never)
    monkeypatch.setattr(tier4.registry_mod, "load_registry", never)
    monkeypatch.setattr(tier4.store, "read_table", never)
    monkeypatch.setattr(tier4.store, "file_sha256", never)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / fold_name(cache)).write_bytes(fold_bytes(cache, compress=0))
    if failure == "symlink":
        target = tmp_path / "outside-cache"
        (cache_dir / fold_name(cache)).rename(target)
        (cache_dir / fold_name(cache)).symlink_to(target)
    if failure == "oversize":
        monkeypatch.setattr(prep, "MAX_FOLD_BYTES", len(fold_bytes(cache)) - 1)
        monkeypatch.setattr(joblib, "load", never)
    out = tmp_path / "out"
    argv = ["--release-id", "r1", "--out", str(out), "--tier4-dir", str(cache_dir),
            "--tier3-snapshot", requested_snapshot]
    if pinned is not None:
        argv += ["--panel-catalog", str(pinned.catalog_path), "--panel-snapshot-id", pinned.snapshot_id,
                 "--panel-artifact-root", str(pinned.artifact_root)]
    refusals = []
    if failure == "recipe":
        initialize = prep.PrepareRefused.__init__

        def record_refusal(error, *args):
            initialize(error, *args)
            refusals.append(error)

        monkeypatch.setattr(prep.PrepareRefused, "__init__", record_refusal)
    assert prep.main(argv) == (2 if failure else 0)
    assert calls == [role] * (0 if failure in ("oversize", "symlink") else 1)
    if failure == "recipe":
        assert len(refusals) == 1
        assert type(refusals[0]) is prep.PrepareRefused
        assert str(refusals[0]) == f"chooser {role} producer disagrees with registered recipe"
        assert refusals[0].__cause__ is recipe_error
    if failure:
        diagnostic = ("serving fold exceeds byte limit" if failure == "oversize" else
                      "serving fold cannot be read" if failure == "symlink" else
                      "chooser fold refused: invalid serving fold held-out pool" if failure == "cache" else
                      "unsupported chooser serving output" if failure == "output"
                      else f"chooser {role} producer disagrees with registered recipe")
        assert capsys.readouterr().err == f"refused: PrepareRefused: {diagnostic}\n"
        assert not out.exists()
        return
    assert capsys.readouterr().err == ""
    body, obj = _assert_published_descriptor(out, role, cache, floor=floor)
    assert obj["serving_fold"]["policy"]["panel_sha256"] == panel_hash
    if pinned is not None:
        assert body["sources"]["panel_copy_sha256"] == panel_hash
        assert body["sources"]["panel_copy_snapshot_id"] == pinned.snapshot_id
        assert body["sources"]["panel_copy_mode"] == "COPY"
    else:
        assert "panel_copy_sha256" not in body["sources"]


@pytest.mark.parametrize("role", OUTPUTS)
def test_pinned_plan_only_refuses_chooser_panel_mismatch_before_plan(tmp_path, role, monkeypatch, capsys):
    pinned, _store, _snapshot, _records, _data = build_pinned(tmp_path, ALL_ONE)
    cache = _cache(role)
    monkeypatch.setattr(prep, "_real_inputs", lambda *a: (*release_inputs(role, cache), fold_states(role, cache)))
    out = tmp_path / "out"
    assert prep.main(["--release-id", "r1", "--out", str(out), "--plan-only",
                      "--panel-catalog", str(pinned.catalog_path),
                      "--panel-snapshot-id", pinned.snapshot_id,
                      "--panel-artifact-root", str(pinned.artifact_root)]) == 2
    assert capsys.readouterr().err == "refused: PrepareRefused: chooser fold policy does not match the pinned panel\n"
    assert not out.exists()


@pytest.mark.parametrize("role", ["size", "unsupported", "runup_move"])
def test_explicit_policy_mapping_cannot_relabel_role(role):
    policy = chooser_policy("implied_t1", _cache("implied_t1"))
    with pytest.raises(prep.PrepareRefused) as error:
        prep.build_states({}, chooser_fold_policies={role: policy})
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == ("chooser fold policy role disagrees with member" if role == "runup_move"
                                else "unsupported chooser fold role")
    if role == "runup_move":
        assert error.value.__cause__ is None
    else:
        assert type(error.value.__cause__) is ValueError
        assert str(error.value.__cause__) == "unsupported chooser fold role"


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("kind", ["none", "mapping", "duck-policy", "size-policy"])
def test_policy_mapping_requires_exact_chooser_policy_type(role, kind):
    cache = _cache(role)
    fields = dataclasses.asdict(chooser_policy(role, cache))
    invalid = {"none": None, "mapping": fields, "duck-policy": SimpleNamespace(**fields),
               "size-policy": SizeFoldPolicy(model_id=cache["model_id"], feature_order=("x", "y"),
                                             panel_sha256="a" * 64, interval_floor=0.0)}[kind]
    with pytest.raises(prep.PrepareRefused) as error:
        prep.build_states({f"tier4_folds:{role}": {fold_name(cache): fold_bytes(cache)}},
                          chooser_fold_policies={role: invalid})
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "invalid chooser fold policy type"
    assert error.value.__cause__ is None


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("defect", ["policy-type", "member", "duplicate", "bytearray", "memoryview"])
def test_direct_state_guards_refuse_before_any_publication(tmp_path, role, defect):
    cache = _cache(role)
    states = fold_states(role, cache)
    build = next(row for row in states if row.spec.member_id == f"tier4_folds:{role}")
    if defect in ("policy-type", "member"):
        # Empty state isolates the boundary guard from deeper cache/binding validation.
        build.payloads.clear()
        if defect == "policy-type":
            build.chooser_fold_policy = SimpleNamespace(**dataclasses.asdict(build.chooser_fold_policy))
        else:
            other_role = next(other for other in OUTPUTS if other != role)
            build.spec = next(row.spec for row in states if row.spec.member_id == f"tier4_folds:{other_role}")
    elif defect == "duplicate":
        states.append(dataclasses.replace(build, payloads=dict(build.payloads)))
    else:
        convert = bytearray if defect == "bytearray" else memoryview
        build.payloads = {fold_name(cache): convert(fold_bytes(cache))}
    out = tmp_path / "out"
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(out, *release_inputs(role, cache), states)
    assert type(error.value) is prep.PrepareRefused
    expected = {"policy-type": "invalid chooser fold policy type",
                "member": "chooser fold policy role disagrees with member"}
    assert str(error.value) == expected.get(defect, "duplicate or non-byte chooser fold object")
    assert error.value.__cause__ is None
    assert not out.exists()
    assert deployment.current_pointer(out / "deployment") is None


def test_existing_size_cli_case_with_synthetic_registered_producer(tmp_path, monkeypatch):
    from engine.data.features import tier4
    from tests.v2.models.test_phase5_serving_folds import test_cli_gets_explicit_registered_policy as original

    cache = _cache("size")
    monkeypatch.setattr(tier4, "size_feature_model", lambda: SimpleNamespace(
        model_id=cache["model_id"], features=("x", "y"), interval_floor=0.0, produces="pred_abs_move"))

    def never(*args, **kwargs):
        pytest.fail("registry or live data read in synthetic CLI compatibility test")

    monkeypatch.setattr(tier4.registry_mod, "load_registry", never)
    monkeypatch.setattr(tier4.store, "read_table", never)
    monkeypatch.setattr(tier4.store, "file_sha256", never)
    original(tmp_path, cache, monkeypatch)


@pytest.mark.parametrize("role", OUTPUTS)
def test_ordinary_destination_substitution_refuses_before_catalog(tmp_path, role, monkeypatch):
    cache, real = _cache(role), prep.write_object

    def wrong_destination(root, payload):
        digest, rel = real(root, payload)
        (layout.deployment_root(root) / rel).write_bytes(b"substituted")
        return digest, rel

    monkeypatch.setattr(prep, "write_object", wrong_destination)
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(tmp_path, *release_inputs(role, cache), fold_states(role, cache))
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold object bytes differ from preflight"
    assert not (tmp_path / "phase5_release.json").exists()
    assert deployment.current_pointer(tmp_path / "deployment") is None


@pytest.mark.parametrize("role,pinned_mode", [
    ("implied_t1", False), ("implied_t1", True),
    ("runup_move", False), ("runup_move", True), ("size", False), ("size", True),
])
@pytest.mark.parametrize("symlink", [False, True])
def test_corrupt_immutable_address_stays_refused_and_fresh_preparation_succeeds(
        tmp_path, role, pinned_mode, symlink):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, panel = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(panel).hexdigest()
    cache = _cache(role, panel_hash)
    raw = fold_bytes(cache)
    digest = hashlib.sha256(raw).hexdigest()
    if role == "size":
        inputs = release_inputs(role, cache, output="forecast_abs_move")
        policy = SizeFoldPolicy(model_id=cache["model_id"], feature_order=cache["features"],
                                panel_sha256=panel_hash, interval_floor=0.0)
        states = prep.build_states({"tier4_folds:size": {fold_name(cache): raw}}, size_fold_policy=policy)
    else:
        inputs, states = release_inputs(role, cache), fold_states(role, cache)
    corrupt, fresh = tmp_path / "corrupt-root", tmp_path / "fresh-root"
    address = corrupt / "deployment" / "objects" / digest
    address.parent.mkdir(parents=True)
    target = tmp_path / "existing-object" if symlink else address
    wrong = b"preexisting wrong bytes: preserve for store-owner investigation"
    target.write_bytes(wrong)
    if symlink:
        address.symlink_to(target)
    original_stat = target.stat()
    prefix = f"{prep.COPY_PREFIX}refused: " if pinned_mode else ""
    expected = (f"{prefix}size fold object bytes differ" if role == "size"
                else "chooser fold object bytes differ from preflight")
    if symlink:
        expected = "serving fold cannot be read"
    for _attempt in range(2):
        with pytest.raises(prep.PrepareRefused) as error:
            prep.write_release(corrupt, *inputs, states, pinned_panel=pinned)
        assert type(error.value) is prep.PrepareRefused
        assert str(error.value) == expected
        if symlink:
            assert type(error.value.__cause__) is OSError
            assert error.value.__cause__.errno == errno.ELOOP
        else:
            assert error.value.__cause__ is None
        assert address.read_bytes() == target.read_bytes() == wrong
        assert target.stat().st_ino == original_stat.st_ino
        assert target.stat().st_mtime_ns == original_stat.st_mtime_ns
        assert address.is_symlink() is symlink
        assert not (corrupt / "phase5_release.json").exists()
        assert not (corrupt / "deployment" / "releases" / "r1" / "phase5_release.json").exists()
        assert deployment.current_pointer(corrupt / "deployment") is None
    prep.write_release(fresh, *inputs, states, pinned_panel=pinned)
    body, obj = catalog_object(fresh, role)
    assert (fresh / "deployment" / "objects" / digest).read_bytes() == raw
    assert obj["content_hash"] == "sha256:" + digest
    assert obj["path"] == "objects/" + digest and obj["bytes"] == len(raw)
    if role in OUTPUTS:
        _assert_published_descriptor(fresh, role, cache)
    else:
        parent = published_manifest(fresh / "deployment")
        assert obj["serving_fold"] == {
            "schema_version": "serving_fold_descriptor.v1.0", "role": "size",
            "parent_release_id": "r1", "parent_release_hash": parent.release_hash,
            "policy": {"schema_version": "size_fold_policy.v1.0", "model_id": cache["model_id"],
                       "feature_order": ["x", "y"], "panel_sha256": panel_hash,
                       "interval_policy": "size-heldout-deciles.v1", "interval_floor": 0.0},
            "fold_start": "2026-02-01", "decision_clock_id": "entry-close",
            "pool_count": 2, "pool_pred_field": "pool_pred", "pool_res_field": "pool_res",
            "output_name": "forecast_abs_move",
            "estimator": {"schema_version": "artifact_member.v1.0", "name": "estimator",
                          "path": "objects/" + digest, "content_hash": "sha256:" + digest},
        }
    assert body["manifest_hash"] == content_hash({key: value for key, value in body.items()
                                                if key != "manifest_hash"})
    prep.write_release(fresh, *inputs, states, pinned_panel=pinned)
    assert layout.read_manifest(fresh) == body
    assert deployment.current_pointer(fresh / "deployment") is None
    assert address.read_bytes() == target.read_bytes() == wrong


@pytest.mark.parametrize("role", OUTPUTS)
def test_ordinary_plan_only_refuses_malformed_cache_before_plan(tmp_path, role, monkeypatch, capsys):
    cache = _cache(role)
    cache["pool_res"] = [float("nan"), 0.0]
    monkeypatch.setattr(prep, "_real_inputs", lambda *a: (*release_inputs(role, cache), fold_states(role, cache)))
    out = tmp_path / "out"
    assert prep.main(["--release-id", "r1", "--out", str(out), "--plan-only"]) == 2
    assert capsys.readouterr().err == (
        "refused: PrepareRefused: chooser fold refused: invalid serving fold held-out pool\n")
    assert not out.exists()


@pytest.mark.parametrize("role", OUTPUTS)
def test_ordinary_authoring_snapshots_caller_payloads_before_staging(tmp_path, role, monkeypatch):
    cache = _cache(role)
    states, real = fold_states(role, cache), deployment.stage_release
    original = next(build for build in states if build.spec.member_id == f"tier4_folds:{role}")

    def mutate_caller(*args, **kwargs):
        original.payloads.clear()
        return real(*args, **kwargs)

    monkeypatch.setattr(deployment, "stage_release", mutate_caller)
    prep.write_release(tmp_path, *release_inputs(role, cache), states)
    _assert_published_descriptor(tmp_path, role, cache)


def _watch_fold_reads(monkeypatch, *, chunk_size=None, when=None):
    reads, handles = [], []

    class WatchedStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, count):
            assert count > 0
            result = self.stream.read(min(count, chunk_size) if chunk_size else count)
            reads.append((count, len(result)))
            return result

    def watched_open(*args, **kwargs):
        if when is not None and not when(Path(args[0])):
            return builtins.open(*args, **kwargs)
        assert kwargs.get("buffering") == 0
        stream = builtins.open(*args, **kwargs)
        handles.append(stream)
        return WatchedStream(stream)

    monkeypatch.setattr(frozen_state_io, "open", watched_open, raising=False)
    return reads, handles


@pytest.mark.parametrize("replacement", ["same-length", "short", "long", "symlink"])
def test_opaque_state_retry_verifies_address_and_preserves_existing_catalog(tmp_path, replacement):
    raw, name = b"synthetic opaque state", "opaque.json"
    cache = _cache("implied_t1")
    inputs = release_inputs("implied_t1", cache)
    states = prep.build_states({"admissible_table:dyn_sv": {name: raw}})
    out, fresh = tmp_path / "retained", tmp_path / "fresh"
    prep.write_release(out, *inputs, states)
    catalog = out / "phase5_release.json"
    scoped = out / "deployment" / "releases" / "r1" / "phase5_release.json"
    before_catalogs = catalog.read_bytes(), scoped.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    dest = out / "deployment" / "objects" / digest
    wrong = {"same-length": b"!" * len(raw), "short": b"bad", "long": raw + b"!",
             "symlink": raw}[replacement]
    target = tmp_path / "opaque-target" if replacement == "symlink" else dest
    if replacement == "symlink":
        dest.unlink()
        dest.symlink_to(target)
    target.write_bytes(wrong)
    before = target.stat()
    for _ in range(2):
        with pytest.raises(layout.ObjectVerificationError) as caught:
            prep.write_release(out, *inputs, states)
        assert type(caught.value) is layout.ObjectVerificationError
        if replacement in ("long", "symlink"):
            assert str(caught.value) == "phase5 object bytes could not be verified"
            if replacement == "long":
                assert type(caught.value.__cause__) is frozen_state_io.BoundedStateReadError
                assert caught.value.__cause__.code == "SIZE_LIMIT"
            else:
                assert type(caught.value.__cause__) is OSError
                assert caught.value.__cause__.errno == errno.ELOOP
        else:
            assert str(caught.value) == "phase5 object bytes differ from payload"
            assert caught.value.__cause__ is None
        assert (catalog.read_bytes(), scoped.read_bytes()) == before_catalogs
        assert target.read_bytes() == wrong
        after = target.stat()
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert dest.is_symlink() is (replacement == "symlink")
        assert deployment.current_pointer(out / "deployment") is None
    prep.write_release(fresh, *inputs, states)
    body = layout.read_manifest(fresh)
    row = next(row for row in body["members"] if row["member_id"] == "admissible_table:dyn_sv")
    assert row["objects"] == [{"name": name, "path": "objects/" + digest,
                               "content_hash": "sha256:" + digest, "bytes": len(raw)}]
    assert (fresh / "deployment" / "objects" / digest).read_bytes() == raw
    assert body["manifest_hash"] == content_hash({k: v for k, v in body.items() if k != "manifest_hash"})


@pytest.mark.parametrize("payload", [b"", b"ordinary opaque payload"])
@pytest.mark.parametrize("growth", [False, True])
def test_shared_writer_exact_payload_bound_applies_to_retained_objects(tmp_path, monkeypatch, payload, growth):
    digest = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "deployment" / "objects" / digest
    dest.parent.mkdir(parents=True)
    dest.write_bytes(payload)
    reads, handles = _watch_fold_reads(monkeypatch, chunk_size=3)
    real_fstat = os.fstat

    def after_admission(fd):
        info = real_fstat(fd)
        if growth and handles and fd == handles[-1].fileno():
            with dest.open("ab") as stream:
                stream.write(b"additional bytes")
        return info

    monkeypatch.setattr(os, "fstat", after_admission)
    if growth:
        with pytest.raises(layout.ObjectVerificationError) as caught:
            layout.write_object(tmp_path, payload)
        assert str(caught.value) == "phase5 object bytes could not be verified"
        assert type(caught.value.__cause__) is frozen_state_io.BoundedStateReadError
        assert caught.value.__cause__.code == "SIZE_LIMIT"
    else:
        assert layout.write_object(tmp_path, payload) == ("sha256:" + digest, "objects/" + digest)
        assert dest.read_bytes() == payload
    consumed = 0
    for requested, returned in reads:
        assert requested == len(payload) + 1 - consumed
        consumed += returned
    assert consumed == len(payload) + int(growth)
    assert len(handles) == 1 and handles[0].closed


@pytest.mark.parametrize("role", ["size", *OUTPUTS])
@pytest.mark.parametrize("pinned_mode", [False, True])
def test_retained_oversize_fold_refuses_at_exact_payload_bound_before_any_read(
        tmp_path, monkeypatch, role, pinned_mode):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, panel = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(panel).hexdigest()
    cache = _cache(role, panel_hash)
    raw = fold_bytes(cache)
    if role == "size":
        inputs = release_inputs(role, cache, output="forecast_abs_move")
        policy = SizeFoldPolicy(model_id=cache["model_id"], feature_order=cache["features"],
                                panel_sha256=panel_hash, interval_floor=0.0)
        states = prep.build_states({"tier4_folds:size": {fold_name(cache): raw}}, size_fold_policy=policy)
    else:
        inputs, states = release_inputs(role, cache), fold_states(role, cache)
    out = tmp_path / "out"
    dest = out / "deployment" / "objects" / hashlib.sha256(raw).hexdigest()
    dest.parent.mkdir(parents=True)
    with dest.open("wb") as stream:
        stream.truncate(32 * 1024 * 1024 + 1)
    before = dest.stat()
    reads, handles = _watch_fold_reads(monkeypatch, when=lambda path: path == dest)
    with pytest.raises(prep.PrepareRefused) as caught:
        prep.write_release(out, *inputs, states, pinned_panel=pinned)
    prefix = f"{prep.COPY_PREFIX}refused: " if pinned_mode else ""
    assert type(caught.value) is prep.PrepareRefused
    assert str(caught.value) == (f"{prefix}size fold object bytes differ" if role == "size"
                                 else "chooser fold object bytes differ from preflight")
    assert caught.value.__cause__ is None
    assert reads == [] and len(handles) == 1 and handles[0].closed
    after = dest.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_size) == (before.st_ino, before.st_mtime_ns, before.st_size)
    assert not (out / "phase5_release.json").exists()
    assert deployment.current_pointer(out / "deployment") is None


@pytest.mark.parametrize("entry", ["library", "cli", "plan"])
@pytest.mark.parametrize("defect", ["pool", "duplicate", "nonbytes"])
def test_unpinned_size_preflight_refuses_before_any_output(tmp_path, monkeypatch, capsys, entry, defect):
    cache = _cache("size")
    inputs = release_inputs("size", cache, output="forecast_abs_move")
    if defect == "pool":
        cache["pool_res"] = [float("nan"), 0.0]
    policy = SizeFoldPolicy(model_id=cache["model_id"], feature_order=cache["features"],
                            panel_sha256=cache["tier3_snapshot"], interval_floor=0.0)
    states = prep.build_states({"tier4_folds:size": {fold_name(cache): fold_bytes(cache)}}, size_fold_policy=policy)
    build = next(build for build in states if build.spec.member_id == "tier4_folds:size")
    if defect == "duplicate":
        states.append(build)
    elif defect == "nonbytes":
        build.payloads[fold_name(cache)] = bytearray(fold_bytes(cache))
    out = tmp_path / "out"
    diagnostic = ("size fold refused: invalid serving fold held-out pool" if defect == "pool"
                  else "duplicate or non-byte size fold object")
    if entry == "library":
        with pytest.raises(prep.PrepareRefused) as caught:
            prep.write_release(out, *inputs, states)
        assert type(caught.value) is prep.PrepareRefused
        assert str(caught.value) == diagnostic
        if defect == "pool":
            assert type(caught.value.__cause__) is ValueError
            assert str(caught.value.__cause__) == "invalid serving fold held-out pool"
        else:
            assert caught.value.__cause__ is None
    else:
        monkeypatch.setattr(prep, "_real_inputs", lambda *a: (*inputs, states))
        flags = ["--plan-only"] if entry == "plan" else []
        assert prep.main(["--release-id", "r1", "--out", str(out), *flags]) == 2
        captured = capsys.readouterr()
        assert captured.err == f"refused: PrepareRefused: {diagnostic}\n"
        assert captured.out == ""
    assert not out.exists()


@pytest.mark.parametrize("role", ["size", *OUTPUTS, "iv_crush"])
@pytest.mark.parametrize("target_kind", ["regular", "directory", "dangling", "fifo"])
def test_fold_collector_never_opens_final_symlink_target(tmp_path, monkeypatch, role, target_kind):
    cache = _cache(role)
    source = tmp_path / "cache"
    source.mkdir()
    target = tmp_path / "outside-cache"
    if target_kind == "regular":
        target.write_bytes(fold_bytes(cache))
    elif target_kind == "directory":
        target.mkdir()
    elif target_kind == "fifo":
        os.mkfifo(target)
    (source / fold_name(cache)).symlink_to(target)
    reads, handles = _watch_fold_reads(monkeypatch)
    real_open, seen = os.open, []

    def record_open(path, flags, *args, **kwargs):
        seen.append(flags)
        assert flags & os.O_NONBLOCK  # A defective no-follow mutation must not block on FIFO.
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", record_open)
    with pytest.raises(prep.PrepareRefused) as error:
        prep.tier4_fold_payloads(source, {role: cache["model_id"]}, "a" * 64, "202602")
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "serving fold cannot be read"
    assert type(error.value.__cause__) is OSError
    assert error.value.__cause__.errno == errno.ELOOP
    assert reads == handles == []
    assert len(seen) == 1 and seen[0] & os.O_NOFOLLOW
    if target_kind == "regular":
        assert target.read_bytes() == fold_bytes(cache)


@pytest.mark.parametrize("role", ["size", *OUTPUTS, "iv_crush"])
@pytest.mark.parametrize("slack,chunk_size", [(0, None), (8, 17)])
def test_fold_collector_preserves_bounded_regular_file_bytes(tmp_path, monkeypatch, role, slack, chunk_size):
    cache = _cache(role)
    raw, name = fold_bytes(cache), fold_name(cache)
    target = tmp_path / name
    target.write_bytes(raw)
    limit = len(raw) + slack
    monkeypatch.setattr(prep, "MAX_FOLD_BYTES", limit)
    reads, handles = _watch_fold_reads(monkeypatch, chunk_size=chunk_size)
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("unbounded read_bytes"))
    assert prep.tier4_fold_payloads(tmp_path, {role: cache["model_id"]}, "a" * 64, "202602") == {
        f"tier4_folds:{role}": {name: raw}}
    consumed = 0
    for requested, returned in reads:
        assert requested == limit + 1 - consumed
        consumed += returned
    assert consumed == len(raw)
    assert reads[-1][1] == 0
    assert len(handles) == 1 and handles[0].closed


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("growth", [False, True])
def test_fold_collector_refuses_oversize_before_decode_and_bounds_growth(tmp_path, monkeypatch, role, growth):
    cache = _cache(role)
    path = tmp_path / fold_name(cache)
    path.write_bytes(fold_bytes(cache))
    limit = len(fold_bytes(cache)) if growth else folds.MAX_FOLD_BYTES
    assert folds.MAX_FOLD_BYTES == 32 * 1024 * 1024
    monkeypatch.setattr(prep, "MAX_FOLD_BYTES", limit)
    if growth:
        real_fstat = os.fstat

        def append_after_fstat(fd):
            prior = real_fstat(fd)
            with path.open("ab") as stream:
                stream.write(b"growth past limit")
            return prior

        monkeypatch.setattr(os, "fstat", append_after_fstat)
    else:
        with path.open("ab") as stream:
            stream.truncate(limit + 1)
    reads, handles = _watch_fold_reads(monkeypatch)
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("oversize reached decode"))
    with pytest.raises(prep.PrepareRefused) as error:
        prep.tier4_fold_payloads(tmp_path, {role: cache["model_id"]}, "a" * 64, "202602")
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "serving fold exceeds byte limit"
    assert error.value.__cause__ is None
    assert reads == ([(limit + 1, limit + 1)] if growth else [])
    assert len(handles) == 1 and handles[0].closed


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("symlink", [False, True])
def test_fold_collector_refuses_fifo_without_blocking_or_reading(tmp_path, monkeypatch, role, symlink):
    cache = _cache(role)
    fifo = tmp_path / ("fifo" if symlink else fold_name(cache))
    os.mkfifo(fifo)
    if symlink:
        (tmp_path / fold_name(cache)).symlink_to(fifo)
    reads, handles = _watch_fold_reads(monkeypatch)
    real_open, flags_seen = os.open, []

    def checked_open(path, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK  # Fail before a mutation can block opening the FIFO.
        flags_seen.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", checked_open)
    with pytest.raises(prep.PrepareRefused) as error:
        prep.tier4_fold_payloads(tmp_path, {role: cache["model_id"]}, "a" * 64, "202602")
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == ("serving fold cannot be read" if symlink else "serving fold is not a regular file")
    if symlink:
        assert type(error.value.__cause__) is OSError
        assert error.value.__cause__.errno == errno.ELOOP
    else:
        assert error.value.__cause__ is None
    assert reads == [] and len(flags_seen) == 1
    assert len(handles) == (0 if symlink else 1)
    assert all(handle.closed for handle in handles)


@pytest.mark.parametrize("role", OUTPUTS)
def test_fold_collector_refuses_unreadable_match_and_filters_before_open(tmp_path, monkeypatch, role):
    cache = _cache(role)
    path = tmp_path / fold_name(cache)
    path.symlink_to(tmp_path / "missing-target")
    with pytest.raises(prep.PrepareRefused) as error:
        prep.tier4_fold_payloads(tmp_path, {role: cache["model_id"]}, "a" * 64, "202602")
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "serving fold cannot be read"
    assert type(error.value.__cause__) is OSError
    assert error.value.__cause__.errno == errno.ELOOP
    monkeypatch.setattr(frozen_state_io, "open", lambda *a, **k: pytest.fail("unmatched candidate opened"), raising=False)
    assert prep.tier4_fold_payloads(tmp_path, {role: "other-model"}, None, None) == {}
    assert prep.tier4_fold_payloads(tmp_path, {role: cache["model_id"]}, "b" * 64, None) == {}
    assert prep.tier4_fold_payloads(tmp_path, {role: cache["model_id"]}, None, "202603") == {}
    assert prep.tier4_fold_payloads(tmp_path / "missing", {role: cache["model_id"]}, None, None) == {}


@pytest.mark.parametrize("role,pinned_mode", [
    ("implied_t1", False), ("implied_t1", True),
    ("runup_move", False), ("runup_move", True), ("size", True),
])
@pytest.mark.parametrize("replacement", [
    "exact", "short-reads", "regular-symlink", "mismatch", "missing", "unreadable",
    "oversize", "growth", "fifo", "fifo-symlink",
])
def test_destination_reverification_is_bounded_on_every_writer_path(
        tmp_path, monkeypatch, role, pinned_mode, replacement):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, panel = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(panel).hexdigest()
    cache = _cache(role, panel_hash)
    raw = fold_bytes(cache)
    limit = folds.MAX_FOLD_BYTES if replacement == "oversize" else len(raw)
    assert prep.MAX_FOLD_BYTES == folds.MAX_FOLD_BYTES == 32 * 1024 * 1024
    monkeypatch.setattr(prep, "MAX_FOLD_BYTES", limit)
    if role == "size":
        inputs = release_inputs(role, cache, output="forecast_abs_move")
        policy = SizeFoldPolicy(model_id=cache["model_id"], feature_order=cache["features"],
                                panel_sha256=panel_hash, interval_floor=0.0)
        states = prep.build_states({"tier4_folds:size": {fold_name(cache): raw}}, size_fold_policy=policy)
    else:
        inputs, states = release_inputs(role, cache), fold_states(role, cache)
    out, destinations = tmp_path / "out", []
    real_write, real_open, real_path_open, real_fstat = prep.write_object, os.open, Path.open, os.fstat
    reads, handles = _watch_fold_reads(monkeypatch, chunk_size=17 if replacement == "short-reads" else None,
                                     when=lambda path: path in destinations)

    def substitute(root, payload):
        digest, rel = real_write(root, payload)
        path = layout.deployment_root(root) / rel
        destinations.append(path)
        if replacement in ("missing", "fifo", "fifo-symlink", "regular-symlink"):
            path.unlink()
        if replacement == "fifo":
            os.mkfifo(path)
        elif replacement in ("fifo-symlink", "regular-symlink"):
            target = tmp_path / "replacement"
            if replacement == "fifo-symlink":
                os.mkfifo(target)
            else:
                target.write_bytes(payload)
            path.symlink_to(target)
        elif replacement == "oversize":
            with path.open("ab") as stream:
                stream.truncate(limit + 1)
        elif replacement == "mismatch":
            path.write_bytes(b"different bytes")
        return digest, rel

    def checked_open(path, flags, *args, **kwargs):
        if Path(path) in destinations:
            assert flags & os.O_NONBLOCK  # Stop a defective FIFO open before it can hang.
            if replacement == "unreadable":
                raise PermissionError("synthetic private destination path")
        return real_open(path, flags, *args, **kwargs)

    def checked_path_open(path, mode="r", *args, **kwargs):
        if path in destinations and mode == "rb":
            pytest.fail("destination used an uncapped Path.open read")
        return real_path_open(path, mode, *args, **kwargs)

    def grow_after_fstat(fd):
        prior = real_fstat(fd)
        if replacement == "growth" and handles and fd == handles[-1].fileno():
            with destinations[-1].open("ab") as stream:
                stream.write(b"growth after admission")
        return prior

    monkeypatch.setattr(prep, "write_object", substitute)
    monkeypatch.setattr(os, "open", checked_open)
    monkeypatch.setattr(Path, "open", checked_path_open)
    monkeypatch.setattr(os, "fstat", grow_after_fstat)
    if replacement in ("exact", "short-reads"):
        prep.write_release(out, *inputs, states, pinned_panel=pinned)
        if role in OUTPUTS:
            _assert_published_descriptor(out, role, cache)
        body, obj = catalog_object(out, role)
        assert obj["content_hash"] == "sha256:" + hashlib.sha256(raw).hexdigest()
        assert obj["bytes"] == len(raw)
        assert obj["serving_fold"]["pool_count"] == 2
        assert body["release_id"] == "r1"
        consumed = 0
        for requested, returned in reads:
            assert requested == limit + 1 - consumed
            consumed += returned
        assert consumed == len(raw) and reads[-1][1] == 0
    else:
        with pytest.raises(prep.PrepareRefused) as error:
            prep.write_release(out, *inputs, states, pinned_panel=pinned)
        assert type(error.value) is prep.PrepareRefused
        expected = {
            "missing": "serving fold cannot be read", "unreadable": "serving fold cannot be read",
            "oversize": "serving fold exceeds byte limit", "growth": "serving fold exceeds byte limit",
            "fifo": "serving fold is not a regular file", "fifo-symlink": "serving fold cannot be read",
            "regular-symlink": "serving fold cannot be read",
            "mismatch": (f"{prep.COPY_PREFIX}refused: size fold object bytes differ" if role == "size"
                         else "chooser fold object bytes differ from preflight"),
        }[replacement]
        assert str(error.value) == expected
        cause = {"missing": FileNotFoundError, "unreadable": PermissionError,
                 "fifo-symlink": OSError, "regular-symlink": OSError}.get(replacement)
        assert type(error.value.__cause__) is (type(None) if cause is None else cause)
        if replacement in ("fifo-symlink", "regular-symlink"):
            assert error.value.__cause__.errno == errno.ELOOP
        assert not (out / "phase5_release.json").exists()
        if replacement == "growth":
            assert reads == [(limit + 1, limit + 1)]
        elif replacement in ("missing", "unreadable", "oversize", "fifo", "fifo-symlink", "regular-symlink"):
            assert reads == []
    assert len(destinations) == 1
    assert len(handles) == (0 if replacement in ("missing", "unreadable", "fifo-symlink", "regular-symlink") else 1)
    assert all(stream.closed for stream in handles)
    assert deployment.current_pointer(out / "deployment") is None

