"""Inference refuses an artifact member above the bounded read limit."""
from __future__ import annotations

import hashlib
import json

import pytest

from engine.v2.models import (
    MODEL_NOT_READY,
    ArtifactMember,
    FrozenInference,
    InferenceRequest,
    JsonLinearAdapter,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models import loader as loader_module

_SMALL_LIMIT = 64


def _release(root):
    payload = {
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}],
        "padding": "x" * 256,
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    assert len(raw) > _SMALL_LIMIT
    (root / "estimator.json").write_bytes(raw)
    member = ArtifactMember(
        name="estimator",
        path="estimator.json",
        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
    )
    binding = ModelBinding(
        binding_id="b1",
        model_id="m1",
        role="size",
        strategy_id="*",
        decision_clock_id="entry-close",
        adapter="json-linear.v1",
        feature_order=("x",),
        output_names=("prediction",),
        members=(member,),
    )
    return ModelRelease(release_id="r1", deployment_id="d1", bindings=(binding,))


def _must_not_run(*args, **kwargs):
    pytest.fail("oversized artifact reached adapter parsing or inference")


def test_oversized_member_is_refused_before_adapter_and_cache(tmp_path, monkeypatch):
    assert loader_module._INFER_MAX_MEMBER_BYTES == 256 * 1024 * 1024
    release = _release(tmp_path)
    request = InferenceRequest(
        release_id="r1", binding_id="b1", feature_order=("x",), rows=((3.0,),)
    )
    inference = FrozenInference(tmp_path)

    monkeypatch.setattr(loader_module, "_INFER_MAX_MEMBER_BYTES", _SMALL_LIMIT)
    monkeypatch.setattr(JsonLinearAdapter, "load", _must_not_run)
    monkeypatch.setattr(JsonLinearAdapter, "predict", _must_not_run)
    result = inference.infer(release, request)

    assert result.status == MODEL_NOT_READY
    assert result.reason_codes == ("ARTIFACT_INVALID",)
    assert result.predictions == ()
    assert inference.cache_size == 0
