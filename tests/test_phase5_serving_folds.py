"""Offline declarations from a captured size-cache header, synthetic bounded values."""
import dataclasses
import copy
import io
import json

import joblib
import numpy as np
import pytest

from checks import phase5_release as layout
from engine.v2.foundation import from_document, to_document
from engine.v2.foundation import content_hash
from engine.v2.models import (
    ArtifactInventoryMember, ArtifactMember, ModelArtifactInventory, ModelBinding,
    ModelRelease, ModelReleaseInventory, ReleaseBinding, ReleaseRequirement, deployment,
)
from engine.v2.models.serving_folds import ServingFoldDescriptor, SizeFoldPolicy
from tests.fixtures.size_serving_header import HEADER
from tools import phase5_prepare_release as prep
from tools import phase5_serving_folds as folds


class NoPrediction:
    def predict(self, rows):
        raise AssertionError("offline authoring must never predict")


@pytest.fixture
def cache():
    header = copy.deepcopy(HEADER)
    captured = header.pop("capture")
    # Actual captured keys/header; replace estimator and large real pools only.
    trimmed = dict(header, estimator=NoPrediction(), pool_pred=np.array([0.1, 0.2]),
                   pool_res=np.array([-0.01, 0.02]))
    assert sorted(trimmed) == captured["keys"]
    return trimmed


def _bytes(cache):
    stream = io.BytesIO()
    joblib.dump(cache, stream)
    return stream.getvalue()


def _policy(cache):
    return SizeFoldPolicy(model_id=cache["model_id"], feature_order=tuple(cache["features"]),
                          panel_sha256=cache["tier3_snapshot"], interval_floor=0.0)


def _name(cache):
    model, month, panel = cache["model_id"], cache["fold_start"].replace("-", "")[:6], cache["tier3_snapshot"]
    return f"{model}_{month}_{panel[:12]}.joblib"


def _inputs(cache):
    features, model_id = tuple(cache["features"]), cache["model_id"]
    raw = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": features,
                      "outputs": [{"name": "forecast_abs_move", "intercept": 0,
                                   "coefficients": [0] * len(features)}]}).encode()
    digest = layout.sha256_bytes(raw)
    binding = ModelBinding(binding_id="size:*", model_id=model_id, role="size", strategy_id="*",
                           decision_clock_id="entry-close", adapter="json-linear.v1",
                           feature_order=features, output_names=("forecast_abs_move",),
                           members=(ArtifactMember(name="estimator", path="source", content_hash=digest),))
    inventory = ModelReleaseInventory(
        release_id="r1", deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(ModelArtifactInventory(
            artifact_id=model_id, role="size", strategy_ids=("*",), compatible_clock_ids=("entry-close",),
            target_contract_ref="return.v1", ordered_features=features,
            members=(ArtifactInventoryMember(member_id="size:estimator", kind="estimator",
                                              artifact_ref="source", content_hash=digest),)),),
        bindings=(ReleaseBinding(role="size", strategy_id="*", clock_id="entry-close", artifact_id=model_id,
                                 ordered_features=features, required_member_kinds=("estimator",)),),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="registry", evidence_refs=("evidence",))
    return ModelRelease(release_id="r1", deployment_id="d1", bindings=(binding,)), inventory, {digest: raw}


@pytest.fixture
def staged(tmp_path, cache):
    return deployment.stage_release(tmp_path / "deployment", *_inputs(cache))


def test_descriptor_round_trip_and_identity(cache, staged):
    descriptor = folds.describe_size_fold(_bytes(cache), _name(cache), _policy(cache), staged)
    assert from_document(ServingFoldDescriptor, to_document(descriptor)) == descriptor
    assert descriptor.pool_count == 2
    assert descriptor.parent_release_hash == staged.release_hash
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.fold_start = "2000-01-01"
    changes = [dict(parent_release_id="r2"), dict(parent_release_hash="sha256:" + "a" * 64),
               dict(fold_start="2000-01-01"), dict(pool_count=3)]
    changes += [dict(policy=dataclasses.replace(descriptor.policy, **change)) for change in (
        dict(panel_sha256="a" * 64), dict(feature_order=tuple(reversed(cache["features"]))),
        dict(interval_floor=None))]
    for change in changes:
        assert dataclasses.replace(descriptor, **change).descriptor_hash != descriptor.descriptor_hash
    altered = dict(cache, pool_res=np.array([0.03, 0.04]))
    other = folds.describe_size_fold(_bytes(altered), _name(cache), _policy(cache), staged)
    assert other.estimator.content_hash != descriptor.estimator.content_hash
    assert other.descriptor_hash != descriptor.descriptor_hash


@pytest.mark.parametrize("defect", ["model", "features", "panel", "date", "month", "filename",
                                     "estimator", "absent", "shape", "length", "numeric", "nan"])
def test_malformed_cache_refuses(cache, staged, defect):
    original_name, policy = _name(cache), _policy(cache)
    if defect == "model":
        cache["model_id"] += "-wrong"
    elif defect == "features":
        cache["features"] = list(reversed(cache["features"]))
    elif defect == "panel":
        cache["tier3_snapshot"] = cache["tier3_snapshot"][:12] + "f" * 52
    elif defect in ("date", "month"):
        cache["fold_start"] = "2026-09-02" if defect == "date" else "2026-09-01"
    elif defect == "filename":
        original_name = "wrong.joblib"
    elif defect in ("estimator", "absent"):
        cache.pop("estimator" if defect == "estimator" else "pool_res")
    else:
        cache["pool_res"] = {"shape": [[1, 2]], "length": [1], "numeric": ["bad", "bad"],
                             "nan": [float("nan"), 1]}[defect]
    with pytest.raises(ValueError) as error:
        folds.describe_size_fold(_bytes(cache), original_name, policy, staged)
    assert "/" not in str(error.value)


def test_bound_checked_before_deserialization(cache, staged, monkeypatch):
    raw = _bytes(cache)
    monkeypatch.setattr(folds, "MAX_FOLD_BYTES", len(raw) - 1)
    monkeypatch.setattr(joblib, "load", lambda *a: pytest.fail("oversized bytes decoded"))
    with pytest.raises(ValueError, match="byte limit"):
        folds.describe_size_fold(raw, _name(cache), _policy(cache), staged)


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_thin_pools_are_declared(cache, staged, count):
    cache.update(pool_pred=np.zeros(count), pool_res=np.zeros(count))
    descriptor = folds.describe_size_fold(_bytes(cache), _name(cache), _policy(cache), staged)
    assert descriptor.pool_count == count
    assert descriptor.policy.interval_floor == 0.0


@pytest.mark.parametrize("explicit", [False, True])
def test_real_preparer_emits_catalog_hashed_descriptor(tmp_path, cache, explicit):
    name, raw = _name(cache), _bytes(cache)
    states = prep.build_states({"tier4_folds:size": {name: raw}},
                               size_fold_policy=_policy(cache) if explicit else None)
    args = _inputs(cache)
    prep.write_release(tmp_path, *args, states)
    body = layout.read_manifest(tmp_path)
    row = next(row for row in body["members"] if row["member_id"] == "tier4_folds:size")
    obj = row["objects"][0]
    assert ("serving_fold" in obj) == explicit
    if explicit:
        descriptor = from_document(ServingFoldDescriptor, obj["serving_fold"])
        assert descriptor.estimator.content_hash == obj["content_hash"]
        assert descriptor.estimator.path == obj["path"]
        assert descriptor.policy.interval_floor == 0.0
        changed = json.loads(json.dumps(body))
        changed.pop("manifest_hash")
        changed_row = next(row for row in changed["members"] if row["member_id"] == "tier4_folds:size")
        changed_row["objects"][0]["serving_fold"]["policy"]["interval_floor"] = 1.0
        assert content_hash(changed) != body["manifest_hash"]
    prep.write_release(tmp_path, *args, states)
    assert layout.read_manifest(tmp_path) == body
    assert deployment.current_pointer(tmp_path / "deployment") is None


def test_cli_gets_explicit_registered_policy(tmp_path, cache, monkeypatch):
    from engine.data.features import tier4
    from engine.v2.models import inventory

    producer = tier4.size_feature_model()
    assert producer.model_id == cache["model_id"]
    assert producer.features == tuple(cache["features"])
    assert producer.interval_floor == 0.0
    release, inv, payloads = _inputs(cache)
    monkeypatch.setattr(inventory, "current_release_inventory", lambda: (inv, ()))
    monkeypatch.setattr(prep, "model_release", lambda *a, **k: (release, payloads))
    monkeypatch.setattr(prep, "modules_available", lambda *a: (False, "unused"))
    monkeypatch.setattr(tier4, "serving_model", lambda *a, **k: pytest.fail("runtime fitting path"))
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / _name(cache)).write_bytes(_bytes(cache))
    out = tmp_path / "out"
    assert prep.main(["--release-id", "r1", "--out", str(out), "--tier4-dir", str(cache_dir),
                      "--tier3-snapshot", cache["tier3_snapshot"]]) == 0
    catalog = layout.read_manifest(out)
    row = next(row for row in catalog["members"] if row["member_id"] == "tier4_folds:size")
    assert row["objects"][0]["serving_fold"]["policy"]["interval_floor"] == producer.interval_floor


def test_policy_binding_disagreement_refuses(cache, staged):
    wrong = dataclasses.replace(_policy(cache), model_id="wrong")
    with pytest.raises(ValueError, match="release binding"):
        folds.describe_size_fold(_bytes(cache), _name(cache), wrong, staged)
    bad_floor = dataclasses.replace(_policy(cache), interval_floor=float("nan"))
    with pytest.raises(ValueError, match="invalid size serving policy"):
        folds.describe_size_fold(_bytes(cache), _name(cache), bad_floor, staged)
