"""Artifact-only chooser preparation; independent synthetic numeric expectations."""

import builtins
import copy
import hashlib
import io
import os
import socket
import sys
from dataclasses import FrozenInstanceError, fields, replace

import joblib
import numpy as np
import pytest

from engine.v2.foundation import content_hash, to_document
from engine.v2.models import (
    MODEL_NOT_READY,
    AdapterError,
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    ModelBinding,
    ModelRelease,
    no_fit_guard,
)
from engine.v2.models import serving_fold_loader as loader
from engine.v2.models.deployment import (
    RELEASE_HASH_SEMANTIC_V2,
    StagedManifest,
    semantic_release_hash,
)
from engine.v2.models.serving_folds import ChooserFoldPolicy, ChooserServingFoldDescriptor
from tests.v2.models.serving_fold_support import OUTPUTS


class Predictor:
    def predict(self, rows):
        return np.asarray(rows).sum(axis=1) * 0.01

    def fit(self, *args, **kwargs):
        pytest.fail("runtime fit")


def _raw(cache):
    stream = io.BytesIO()
    joblib.dump(cache, stream, compress=0)
    return stream.getvalue()


def _rehash(catalog):
    catalog["manifest_hash"] = content_hash(
        {k: v for k, v in catalog.items() if k != "manifest_hash"}
    )
    return catalog


def _store(tmp_path, cache, parent, role):
    raw = _raw(cache)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    member = ArtifactMember(name="estimator", path=f"objects/{digest[7:]}", content_hash=digest)
    path = tmp_path / member.path
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    output, producer = OUTPUTS[role]
    descriptor = ChooserServingFoldDescriptor(
        parent_release_id=parent.release.release_id,
        parent_release_hash=parent.release_hash,
        policy=ChooserFoldPolicy(
            role=role,
            model_id=cache["model_id"],
            feature_order=("x", "y"),
            panel_sha256="a" * 64,
            interval_floor=0.0,
        ),
        fold_start="2026-02-01",
        decision_clock_id="entry-close",
        estimator=member,
        pool_count=len(cache["pool_pred"]),
        output_name=output,
        producer_output=producer,
    )
    catalog = _rehash(
        {
            "schema_version": "phase5_staged_release.v1.0",
            "release_id": "r1",
            "deployment_id": "d1",
            "sources": {},
            "members": [
                {
                    "member_id": f"tier4_folds:{role}",
                    "status": "STAGED",
                    "objects": [
                        {
                            "name": f"{cache['model_id']}_202602_{'a' * 12}.joblib",
                            "path": member.path,
                            "content_hash": digest,
                            "bytes": len(raw),
                            "serving_fold": to_document(descriptor),
                        }
                    ],
                }
            ],
        }
    )
    return descriptor, catalog, path


@pytest.fixture(params=OUTPUTS)
def setup(tmp_path, request):
    role = request.param
    binding = ModelBinding(
        binding_id=f"{role}:*",
        model_id=f"synthetic-{role}",
        role=role,
        strategy_id="*",
        decision_clock_id="entry-close",
        adapter="json-linear.v1",
        feature_order=("x", "y"),
        output_names=(OUTPUTS[role][0],),
        members=(
            ArtifactMember(
                name="estimator", path="champion", content_hash=content_hash("champion")
            ),
        ),
    )
    release = ModelRelease(release_id="r1", deployment_id="d1", bindings=(binding,))
    parent = StagedManifest(
        release=release,
        release_hash=semantic_release_hash(release),
        release_hash_version=RELEASE_HASH_SEMANTIC_V2,
        staged_at="2026-02-02T00:00:00Z",
    )
    cache = {
        "estimator": Predictor(),
        "model_id": binding.model_id,
        "features": ["x", "y"],
        "tier3_snapshot": "a" * 64,
        "fold_start": "2026-02-01",
        "pool_pred": [0.1, 0.2],
        "pool_res": [-0.01, 0.02],
    }
    descriptor, catalog, path = _store(tmp_path, cache, parent, role)
    return FrozenInference(tmp_path), descriptor, parent, catalog, path, cache


def _load(setup, **overrides):
    inference, descriptor, parent, catalog, _, _ = setup
    return loader.load_chooser_serving_fold(
        inference,
        overrides.pop("descriptor", descriptor),
        verified_parent_manifest=overrides.pop("parent", parent),
        verified_catalog=overrides.pop("catalog", catalog),
        **overrides,
    )


def _infer(ref):
    binding = ref.release.bindings[0]
    return ref.inference.infer(
        ref.release,
        InferenceRequest(
            release_id=ref.release.release_id,
            binding_id=binding.binding_id,
            feature_order=("x", "y"),
            rows=((2.0, 3.0),),
        ),
    )


def test_one_decode_raw_outputs_immutable_and_no_source_admission(setup, monkeypatch):
    original, calls = joblib.load, []
    monkeypatch.setattr(joblib, "load", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    before = copy.deepcopy(setup[3])
    ref = _load(setup)
    assert setup[3] == before
    assert _load(setup) == ref
    assert _infer(ref).predictions == ((0.05,),)  # Raw D14: no horizon scaling.
    assert _infer(ref).output_names == (OUTPUTS[ref.descriptor.policy.role][0],)
    assert ref.descriptor.producer_output == OUTPUTS[ref.descriptor.policy.role][1]
    assert ref.pool_pred == (0.1, 0.2) and ref.pool_res == (-0.01, 0.02)
    assert calls == [1] and ref.inference.cache_size == 1
    assert ref.inference is setup[0] and setup[2].release.bindings[0].adapter == "json-linear.v1"
    assert {field.name for field in fields(ref)} == {
        "descriptor", "catalog_hash", "pool_pred", "pool_res", "inference", "release"
    }
    assert ref.catalog_hash == before["manifest_hash"]
    expected_id = content_hash({
        "parent": setup[2].release, "catalog": before["manifest_hash"], "descriptor": setup[1]
    })
    assert ref.release.release_id == expected_id
    assert ref.release.bindings[0].binding_id == expected_id
    assert set(to_document(ref.descriptor)) == set(
        setup[3]["members"][0]["objects"][0]["serving_fold"]
    )
    with pytest.raises(FrozenInstanceError):
        ref.catalog_hash = "other"
    with pytest.raises(TypeError):
        ref.pool_res[0] = 1
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        loader.load_serving_fold(
            setup[0],
            ref.descriptor,
            verified_parent_manifest=setup[2],
            catalog_hash=ref.catalog_hash,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"role": "size"},
        {"strategy_id": "TWIN-P"},
        {"model_id": "other"},
        {"feature_order": ("y", "x")},
        {"decision_clock_id": "next-open"},
        {"output_names": ("wrong",)},
    ],
)
def test_parent_mismatch_even_under_rehashed_manifest(setup, change, monkeypatch):
    parent = setup[2]
    release = replace(parent.release, bindings=(replace(parent.release.bindings[0], **change),))
    parent = replace(parent, release=release, release_hash=semantic_release_hash(release))
    descriptor = replace(setup[1], parent_release_hash=parent.release_hash)
    catalog = copy.deepcopy(setup[3])
    catalog["members"][0]["objects"][0]["serving_fold"] = to_document(descriptor)
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("invalid parent decoded"))
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup, parent=parent, descriptor=descriptor, catalog=_rehash(catalog))
    assert caught.value.code == MODEL_NOT_READY


@pytest.mark.parametrize("field,value", [
    ("parent_release_id", "other-parent"),
    ("parent_release_hash", content_hash("other-parent")),
])
def test_descriptor_parent_identity_refuses_with_matching_catalog(setup, field, value, monkeypatch):
    descriptor = replace(setup[1], **{field: value})
    catalog = copy.deepcopy(setup[3])
    catalog["members"][0]["objects"][0]["serving_fold"] = to_document(descriptor)
    monkeypatch.setattr(setup[0], "prepare_serving_fold",
                        lambda *a, **k: pytest.fail("wrong descriptor parent reached preparation"))
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup, descriptor=descriptor, catalog=_rehash(catalog))
    assert caught.value.code == MODEL_NOT_READY
    assert setup[0].cache_size == 0


def test_descriptor_pool_count_refuses_with_matching_catalog(setup):
    descriptor = replace(setup[1], pool_count=setup[1].pool_count + 1)
    catalog = copy.deepcopy(setup[3])
    catalog["members"][0]["objects"][0]["serving_fold"] = to_document(descriptor)
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup, descriptor=descriptor, catalog=_rehash(catalog))
    assert caught.value.code == MODEL_NOT_READY
    assert setup[0].cache_size == 1  # Valid bytes decoded; descriptor agreement refused.


@pytest.mark.parametrize(
    "defect",
    [
        "hash",
        "schema",
        "release",
        "deployment",
        "missing",
        "row-duplicate",
        "object-duplicate",
        "descriptor-duplicate",
        "name-duplicate",
        "status",
        "wrong-role",
        "path",
        "object-hash",
        "name",
        "bytes",
        "zero-bytes",
        "negative-bytes",
        "oversize-bytes",
        "fractional-bytes",
        "opaque",
        "descriptor",
        "members",
        "objects",
    ],
)
def test_catalog_membership_refuses_before_decode(setup, defect, monkeypatch):
    catalog = copy.deepcopy(setup[3])
    row, obj = catalog["members"][0], catalog["members"][0]["objects"][0]
    if defect in {"schema", "release", "deployment"}:
        catalog[
            {"schema": "schema_version", "release": "release_id", "deployment": "deployment_id"}[
                defect
            ]
        ] = "wrong"
    elif defect == "missing":
        catalog["members"] = []
    elif defect == "row-duplicate":
        catalog["members"].append(copy.deepcopy(row))
    elif defect == "object-duplicate":
        row["objects"].append(copy.deepcopy(obj))
    elif defect == "descriptor-duplicate":
        other = copy.deepcopy(obj)
        other["name"] += ".duplicate"
        row["objects"].append(other)
    elif defect == "name-duplicate":
        other = copy.deepcopy(obj)
        other["serving_fold"]["policy"]["interval_floor"] = 1.0
        row["objects"].append(other)
    elif defect == "status":
        row["status"] = "PENDING"
    elif defect == "wrong-role":
        row["member_id"] = "chooser_analog_pool"
    elif defect == "path":
        obj["path"] = "champion"
    elif defect == "object-hash":
        obj["content_hash"] = content_hash("other")
    elif defect == "name":
        obj["name"] = "other.joblib"
    elif defect in {"bytes", "zero-bytes", "negative-bytes", "oversize-bytes", "fractional-bytes"}:
        obj["bytes"] = {"bytes": True, "zero-bytes": 0, "negative-bytes": -1,
                        "oversize-bytes": loader.MAX_FOLD_BYTES + 1, "fractional-bytes": 1.5}[defect]
    elif defect == "opaque":
        del obj["serving_fold"]
    elif defect == "descriptor":
        obj["serving_fold"]["policy"]["interval_floor"] = 1.0
    elif defect == "members":
        catalog["members"] = [None]
    elif defect == "objects":
        row["objects"] = [None]
    if defect != "hash":
        _rehash(catalog)
    else:
        catalog["manifest_hash"] = content_hash("wrong")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("invalid catalog decoded"))
    monkeypatch.setattr(setup[0], "prepare_serving_fold",
                        lambda *a, **k: pytest.fail("invalid catalog reached preparation"))
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup, catalog=catalog)
    assert caught.value.code == MODEL_NOT_READY
    assert setup[0].cache_size == 0


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_id", "wrong"),
        ("features", ["y", "x"]),
        ("tier3_snapshot", "b" * 64),
        ("fold_start", "2026-03-01"),
        ("pool_res", [float("nan"), 0]),
        ("pool_res", [1]),
        ("pool_pred", [1]),
        ("pool_res", [[1, 2]]),
        ("pool_res", [[1], [2]]),
        ("pool_res", [True, False]),
        ("pool_res", ["1", "2"]),
        ("pool_res", None),
        ("estimator", object()),
    ],
)
def test_rehashed_header_and_pool_tamper_refuses(setup, key, value):
    inference, descriptor, parent, catalog, path, cache = setup
    cache = dict(cache, **{key: value})
    raw = _raw(cache)
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    member = replace(descriptor.estimator, content_hash=digest, path=f"objects/{digest[7:]}")
    # Keep every expected header/pool field independent of the tampered payload.
    descriptor = replace(descriptor, estimator=member)
    new_path = path.parent.parent / member.path
    new_path.write_bytes(raw)
    catalog = copy.deepcopy(catalog)
    catalog["members"][0]["objects"][0].update(
        path=member.path, content_hash=digest, bytes=len(raw), serving_fold=to_document(descriptor)
    )
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load((inference, descriptor, parent, _rehash(catalog), new_path, cache))
    assert caught.value.code == MODEL_NOT_READY


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("defect", ["oversize", "hash", "missing", "symlink", "fifo", "directory"])
def test_cold_warm_bytes_refuse_and_no_stale_success(setup, warm, defect, monkeypatch):
    inference, _, _, _, path, _ = setup
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", path.stat().st_size)
    ref = _load(setup) if warm else None
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("bad bytes decoded"))
    raw = path.read_bytes()
    path.unlink()
    if defect == "oversize":
        path.write_bytes(raw + b"growth")
    elif defect == "hash":
        path.write_bytes(b"x" * len(raw))
    elif defect == "symlink":
        target = path.with_name("same-bytes")
        target.write_bytes(raw)
        path.symlink_to(target)
    elif defect == "fifo":
        os.mkfifo(path)
    elif defect == "directory":
        path.mkdir()
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        _load(setup)
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        _load(setup)
    if ref is not None:
        result = _infer(ref)
        assert result.status == MODEL_NOT_READY
        assert str(inference._root) not in result.detail


@pytest.mark.parametrize("limit", [None, 0, -1, True, 1.5, sys.maxsize])
def test_public_preparation_rejects_bad_limits_without_io(setup, limit, monkeypatch):
    monkeypatch.setattr(
        setup[0], "_load_artifact", lambda *a, **k: pytest.fail("invalid limit load")
    )
    with pytest.raises(AdapterError, match=(
        "^artifact byte limit must be a positive integer below sys.maxsize$"
    )) as caught:
        setup[0].prepare_serving_fold(setup[2].release.bindings[0], max_member_bytes=limit)
    assert type(caught.value) is AdapterError
    assert setup[0].cache_size == 0


def test_identity_catalog_and_policy_changes_share_only_decoded_cache(setup):
    first = _load(setup)
    catalog = copy.deepcopy(setup[3])
    catalog["sources"]["extra"] = "changed"
    other = _load(setup, catalog=_rehash(catalog))
    descriptor = replace(setup[1], policy=replace(setup[1].policy, interval_floor=None))
    catalog["members"][0]["objects"][0]["serving_fold"] = to_document(descriptor)
    third = _load(setup, descriptor=descriptor, catalog=_rehash(catalog))
    assert len({ref.release.release_id for ref in (first, other, third)}) == 3
    assert setup[0].cache_size == 1
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        _load(setup, parent=replace(setup[2], release_hash=content_hash("wrong")))
    parent = replace(
        setup[2], release=replace(setup[2].release, bindings=setup[2].release.bindings * 2)
    )
    parent = replace(parent, release_hash=semantic_release_hash(parent.release))
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        _load(setup, parent=parent)


def test_no_fit_network_write_or_source_ready_claim(setup, monkeypatch):
    original = builtins.open
    root = setup[4].parent.parent

    def tree():
        return {p.relative_to(root): p.read_bytes() if p.is_file() else None
                for p in root.rglob("*")}

    before = tree()
    catalog_before = copy.deepcopy(setup[3])

    def read_only(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        return original(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", read_only)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: pytest.fail("network access"))
    with no_fit_guard():
        ref = _load(setup)
        assert _infer(ref).predictions == ((0.05,),)
        metadata = ref.inference.prepare_serving_fold(
            ref.release.bindings[0], max_member_bytes=loader.MAX_FOLD_BYTES
        )
        assert {field.name for field in fields(ref)} == {
            "descriptor", "catalog_hash", "pool_pred", "pool_res", "inference", "release"
        }
        assert {field.name for field in fields(metadata)} == {
            "model_id", "fold_start", "panel_sha256", "feature_order", "pool_pred", "pool_res",
            "training_cutoff"
        }
        assert not any(hasattr(metadata, key) for key in ("fit", "estimator", "header"))
        assert not any(hasattr(value, key) for value in (ref, metadata)
                       for key in ("source_ready", "admission", "ready"))
    assert tree() == before  # Include sidecars and directories anywhere under the root.
    assert setup[3] == catalog_before  # Artifact loading adds no source evidence.


@pytest.mark.parametrize("count", [0, 1])
def test_empty_thin_artifact_pools_do_not_invent_intervals(setup, count):
    inference, _, parent, _, path, cache = setup
    cache = dict(cache, pool_pred=[0.1] * count, pool_res=[-0.01] * count)
    descriptor, catalog, path = _store(path.parent.parent, cache, parent, setup[1].policy.role)
    ref = _load((inference, descriptor, parent, catalog, path, cache))
    assert ref.pool_pred == (0.1,) * count and ref.pool_res == (-0.01,) * count


def test_standard_authoring_catalog_is_consumed_without_relabeling(setup, tmp_path):
    from engine.v2.foundation import from_document
    from tests.v2.models.serving_fold_support import (
        catalog_object,
        fold_states,
        published_manifest,
        release_inputs,
    )
    from tools.phase5_prepare_release import write_release

    role, cache = setup[1].policy.role, setup[5]
    root = tmp_path / "standard"
    write_release(root, *release_inputs(role, cache), fold_states(role, cache))
    catalog, obj = catalog_object(root, role)
    parent = published_manifest(root / "deployment")
    descriptor = from_document(ChooserServingFoldDescriptor, obj["serving_fold"])
    ref = loader.load_chooser_serving_fold(
        FrozenInference(root / "deployment"),
        descriptor,
        verified_parent_manifest=parent,
        verified_catalog=catalog,
    )
    assert ref.descriptor == descriptor
    assert _infer(ref).predictions == ((0.05,),)


def test_public_metadata_cannot_poison_hash_verified_cached_state(setup):
    ref = _load(setup)
    owner, binding = ref.inference, ref.release.bindings[0]
    metadata = owner.prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES)
    assert metadata.feature_order == ("x", "y")
    assert all(type(v) is float for v in metadata.pool_pred + metadata.pool_res)
    with pytest.raises(TypeError):
        metadata.pool_res[0] = 999.0
    with pytest.raises(FrozenInstanceError):
        metadata.pool_res = (999.0, 0.02)
    # Even a forced mutation of this detached value cannot reach the cache.
    object.__setattr__(metadata, "pool_res", (999.0, 0.02))
    object.__setattr__(metadata, "feature_order", ("changed",))
    again = _load(setup)
    assert again.pool_res == (-0.01, 0.02)
    assert _infer(again).predictions == ((0.05,),)
    assert owner.cache_size == 1


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("delta", [-1, 1])
def test_chooser_wrong_declared_size_refuses_before_decode_and_preserves_warm_ceiling(
    setup, warm, delta, monkeypatch
):
    owner = setup[0]
    if warm:
        _load(setup)
    cached = next(iter(owner._cache.values())) if warm else None
    ceiling = None if cached is None else cached.member_byte_limit
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("wrong declared bytes decoded"))
    catalog = copy.deepcopy(setup[3])
    catalog["members"][0]["objects"][0]["bytes"] += delta
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", loader.MAX_FOLD_BYTES // 2)
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        _load(setup, catalog=_rehash(catalog))
    if warm:
        assert cached.member_byte_limit == ceiling
        assert _infer(_load(setup)).predictions == ((0.05,),)
    else:
        assert owner.cache_size == 0


@pytest.mark.parametrize("expected", [0, -1, True, 1.5, sys.maxsize, loader.MAX_FOLD_BYTES + 1])
def test_bad_expected_size_refuses_before_cache_access(setup, expected, monkeypatch):
    monkeypatch.setattr(
        setup[0], "_load_artifact", lambda *a, **k: pytest.fail("invalid size load")
    )
    with pytest.raises(AdapterError, match="^invalid serving fold declared byte count$") as caught:
        setup[0].prepare_serving_fold(
            setup[2].release.bindings[0],
            max_member_bytes=loader.MAX_FOLD_BYTES,
            expected_bytes=expected,
        )
    assert type(caught.value) is AdapterError


def test_public_metadata_refuses_non_fold_or_missing_adapter(setup, monkeypatch):
    owner = setup[0]
    binding = setup[2].release.bindings[0]
    monkeypatch.setattr(
        owner, "_load_artifact", lambda *a, **k: pytest.fail("unsupported adapter load")
    )
    with pytest.raises(AdapterError, match="^unsupported serving fold adapter$") as caught:
        owner.prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES)
    assert type(caught.value) is AdapterError
    owner._adapters.clear()
    with pytest.raises(AdapterError, match="^unsupported serving fold adapter$") as caught:
        owner.prepare_serving_fold(
            replace(binding, adapter="tier4-serving-fold.v1"),
            max_member_bytes=loader.MAX_FOLD_BYTES,
        )
    assert type(caught.value) is AdapterError


@pytest.mark.parametrize("field", ["output_name", "producer_output"])
def test_forged_descriptor_output_refuses_at_public_loader_boundary(setup, field, monkeypatch):
    descriptor = copy.copy(setup[1])
    object.__setattr__(descriptor, field, "wrong-output")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("wrong output decoded"))
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup, descriptor=descriptor)
    assert caught.value.code == MODEL_NOT_READY
    assert setup[0].cache_size == 0


@pytest.mark.parametrize("defect", ["missing", "oversize", "fifo", "symlink"])
def test_public_preparation_normalizes_bounded_read_failures(setup, defect):
    ref = _load(setup)
    path = setup[4]
    raw = path.read_bytes()
    path.unlink()
    if defect == "oversize":
        path.write_bytes(raw + b"growth")
    elif defect == "fifo":
        os.mkfifo(path)
    elif defect == "symlink":
        target = path.with_name("same-bytes")
        target.write_bytes(raw)
        path.symlink_to(target)
    with pytest.raises(AdapterError, match="^bounded artifact verification failed$") as caught:
        ref.inference.prepare_serving_fold(ref.release.bindings[0], max_member_bytes=len(raw))
    assert type(caught.value) is AdapterError
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("binding", [None, {}, object()])
def test_public_preparation_refuses_wrong_binding_type(setup, binding):
    with pytest.raises(AdapterError, match="^unsupported serving fold adapter$") as caught:
        setup[0].prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES)
    assert type(caught.value) is AdapterError


@pytest.mark.parametrize("warm", [False, True])
def test_chooser_exact_byte_ceiling_accepts_and_next_byte_refuses(setup, warm, monkeypatch):
    owner, _, _, catalog, path, _ = setup
    if warm:
        _load(setup)
    ceiling = len(path.read_bytes())
    assert catalog["members"][0]["objects"][0]["bytes"] == ceiling
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", ceiling - 1)
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup)
    assert caught.value.code == MODEL_NOT_READY
    assert owner.cache_size == int(warm)  # Cold refusal cannot populate the cache.
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", ceiling)
    ref = _load(setup)  # Declared and actual bytes equal the chooser ceiling.
    assert ref.pool_pred == (0.1, 0.2)
    assert _infer(ref).predictions == ((0.05,),)
    assert owner.cache_size == 1
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("oversize object decoded"))
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", ceiling - 1)
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as caught:
        _load(setup)  # The same valid object is now limit + 1 bytes.
    assert caught.value.code == MODEL_NOT_READY
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", ceiling)
    assert _load(setup) == ref  # Refusal cannot poison an otherwise-valid reference.


@pytest.mark.parametrize("warm", [False, True])
def test_public_expected_bytes_at_ceiling_accepts_and_next_byte_refuses(setup, warm, monkeypatch):
    owner, descriptor, parent, _, path, _ = setup
    binding = replace(parent.release.bindings[0], adapter="tier4-serving-fold.v1",
                      members=(descriptor.estimator,))
    ceiling = len(path.read_bytes())
    if warm:
        owner.prepare_serving_fold(binding, max_member_bytes=ceiling + 1)
    with monkeypatch.context() as guard:
        guard.setattr(owner, "_load_artifact", lambda *a, **k: pytest.fail("invalid size load"))
        with pytest.raises(AdapterError, match="^invalid serving fold declared byte count$") as caught:
            owner.prepare_serving_fold(binding, max_member_bytes=ceiling, expected_bytes=ceiling + 1)
        assert type(caught.value) is AdapterError
        assert owner.cache_size == int(warm)
    metadata = owner.prepare_serving_fold(
        binding, max_member_bytes=ceiling, expected_bytes=ceiling
    )
    assert metadata.pool_pred == (0.1, 0.2)
    monkeypatch.setattr(owner, "_load_artifact", lambda *a, **k: pytest.fail("invalid size load"))
    with pytest.raises(AdapterError, match="^invalid serving fold declared byte count$") as caught:
        owner.prepare_serving_fold(binding, max_member_bytes=ceiling, expected_bytes=ceiling + 1)
    assert type(caught.value) is AdapterError


def test_chooser_passes_exact_limit_and_no_follow_to_shared_reader(setup, monkeypatch):
    from engine.v2.models import loader as inference_loader

    original, reads = inference_loader.read_bounded_state, []

    def read(path, limit, *, no_follow=False):
        reads.append((path, limit, no_follow))
        return original(path, limit, no_follow=no_follow)

    monkeypatch.setattr(inference_loader, "read_bounded_state", read)
    ref = _load(setup)
    _load(setup)
    _infer(ref)
    assert reads == [(setup[4], loader.MAX_FOLD_BYTES, True)] * 3


def test_public_prepare_and_infer_cannot_widen_cold_byte_ceiling(setup, monkeypatch):
    from engine.v2.models import loader as inference_loader

    owner, descriptor, parent, _, path, _ = setup
    binding = replace(parent.release.bindings[0], adapter="tier4-serving-fold.v1",
                      members=(descriptor.estimator,))
    release = replace(parent.release, bindings=(binding,))
    ceiling = path.stat().st_size
    assert ceiling < loader.MAX_FOLD_BYTES
    original, reads = inference_loader.read_bounded_state, []

    def read(path, limit, *, no_follow=False):
        reads.append((path, limit, no_follow))
        return original(path, limit, no_follow=no_follow)

    monkeypatch.setattr(inference_loader, "read_bounded_state", read)
    metadata = owner.prepare_serving_fold(binding, max_member_bytes=ceiling)
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("warm artifact decoded"))
    assert owner.prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES) == metadata
    result = owner.infer(release, InferenceRequest(
        release_id=release.release_id, binding_id=binding.binding_id,
        feature_order=("x", "y"), rows=((2.0, 3.0),),
    ))
    assert result.status == "READY" and result.predictions == ((0.05,),)
    assert reads == [(path, ceiling, True)] * 3
    assert owner.cache_size == 1


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_descriptor_matches_among_two_same_role_folds(setup, reverse):
    owner, descriptor, parent, catalog, path, cache = setup
    other_cache = dict(cache, fold_start="2026-03-01", pool_pred=[4.0, 5.0])
    other, other_catalog, _ = _store(path.parent.parent, other_cache, parent, descriptor.policy.role)
    other = replace(other, fold_start="2026-03-01")
    obj = other_catalog["members"][0]["objects"][0]
    obj.update(name=f"{cache['model_id']}_202603_{'a' * 12}.joblib", serving_fold=to_document(other))
    catalog = copy.deepcopy(catalog)
    objects = catalog["members"][0]["objects"]
    objects.append(obj)
    if reverse:
        objects.reverse()
    _rehash(catalog)
    first = _load(setup, catalog=catalog)
    second = _load(setup, descriptor=other, catalog=catalog)
    assert first.descriptor == descriptor and first.pool_pred == (0.1, 0.2)
    assert second.descriptor == other and second.pool_pred == (4.0, 5.0)
    assert first.release.release_id != second.release.release_id
    assert owner.cache_size == 2


@pytest.mark.parametrize("path, message", [
    ("../outside", "artifact path escapes artifact root"),
    ("objects/\0bad", "invalid bounded artifact path"),
])
def test_public_preparation_path_errors_are_typed_and_hide_cause(setup, path, message):
    binding = replace(setup[2].release.bindings[0], adapter="tier4-serving-fold.v1",
                      members=(replace(setup[1].estimator, path=path),))
    with pytest.raises(AdapterError) as caught:
        setup[0].prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES)
    assert type(caught.value) is AdapterError
    assert str(caught.value) == message
    assert caught.value.__cause__ is None


def test_public_preparation_normalizes_path_resolution_runtime_error(setup, monkeypatch):
    from pathlib import Path

    binding = replace(setup[2].release.bindings[0], adapter="tier4-serving-fold.v1",
                      members=(setup[1].estimator,))

    def loop(path, *args, **kwargs):
        raise RuntimeError(f"symlink loop: {path}")

    monkeypatch.setattr(Path, "resolve", loop)
    with pytest.raises(AdapterError, match="^invalid bounded artifact path$") as caught:
        setup[0].prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES)
    assert type(caught.value) is AdapterError
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("key,value", [
    ("model_id", 123), ("fold_start", 123), ("tier3_snapshot", 123),
    ("features", "xy"), ("features", [1, 2]),
])
def test_public_metadata_rejects_header_types_after_adapter_decode(setup, key, value, monkeypatch):
    owner, descriptor, parent, _, path, cache = setup
    raw = _raw(dict(cache, **{key: value}))
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    member = replace(descriptor.estimator, content_hash=digest, path=f"objects/{digest[7:]}")
    (path.parent.parent / member.path).write_bytes(raw)
    binding = replace(parent.release.bindings[0], adapter="tier4-serving-fold.v1", members=(member,))
    if key == "features":
        # Both malformed feature containers pass the adapter's tuple-equality check.
        binding = replace(binding, feature_order=tuple(value))
    original, loads = joblib.load, []
    monkeypatch.setattr(joblib, "load", lambda *a, **k: (loads.append(1), original(*a, **k))[1])
    for _ in range(2):
        with pytest.raises(AdapterError, match="^invalid serving fold metadata$") as caught:
            owner.prepare_serving_fold(binding, max_member_bytes=loader.MAX_FOLD_BYTES,
                                       expected_bytes=len(raw))
        assert type(caught.value) is AdapterError
        assert caught.value.__cause__ is None
    assert loads == [1] and owner.cache_size == 1  # Reached metadata on cold and warm calls.
