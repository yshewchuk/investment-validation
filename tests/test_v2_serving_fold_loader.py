"""Real captured cache shape (see #247 fixture); only numeric payloads are synthetic."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from engine.data.features.tier4 import ServingModel
from engine.v2.foundation import content_hash
from engine.v2.models import FrozenInference, InferenceRequest, MODEL_NOT_READY
from engine.v2.models import serving_fold_loader as loader
from engine.v2.scoring.frozen_executor import FrozenStageExecutor
from tests.test_phase5_serving_folds import cache, staged, _bytes, _name, _policy  # noqa: F401
from tools.phase5_serving_folds import describe_size_fold


class Predictor:
    def predict(self, rows):
        return np.asarray(rows).sum(axis=1) * 0.01

    def fit(self, *args):
        raise AssertionError("runtime fitting is forbidden")


@pytest.fixture
def setup(tmp_path, cache, staged):
    cache["estimator"] = Predictor()
    raw = _bytes(cache)
    descriptor = describe_size_fold(raw, _name(cache), _policy(cache), staged)
    path = tmp_path / descriptor.estimator.path
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(raw)
    return FrozenInference(tmp_path), descriptor, staged, path, cache


def _load(setup, **kwargs):
    inference, descriptor, parent, _, _ = setup
    return loader.load_serving_fold(inference, kwargs.pop("descriptor", descriptor),
                                   verified_parent_manifest=kwargs.pop("parent", parent),
                                   catalog_hash=kwargs.pop("catalog_hash", content_hash("catalog")), **kwargs)


def _infer(ref):
    binding = ref.release.bindings[0]
    return ref.inference.infer(ref.release, InferenceRequest(
        release_id=ref.release.release_id, binding_id=binding.binding_id,
        feature_order=binding.feature_order, rows=((1.0,) * len(binding.feature_order),)))


def test_one_decode_existing_executor_matches_legacy(setup, monkeypatch):
    calls, original = [], joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    first, repeated = _load(setup), _load(setup)
    assert first == repeated
    assert first.inference is setup[0]
    assert len(first.release.bindings) == 1
    binding = first.release.bindings[0]
    values = dict.fromkeys(binding.feature_order, 1.0)
    result = FrozenStageExecutor(inference=first.inference, release=first.release,
                                 binding_id=binding.binding_id).execute(values)
    stored = setup[4]
    legacy = ServingModel(estimator=stored["estimator"], model_id=stored["model_id"],
                          fold_start=pd.Timestamp(stored["fold_start"]),
                          tier3_snapshot=stored["tier3_snapshot"], features=tuple(stored["features"]))
    assert result.outputs["forecast_abs_move"] == legacy.predict(pd.DataFrame([values]))[0]
    assert _infer(first).predictions == ((result.outputs["forecast_abs_move"],),)
    assert calls == [1]
    assert first.inference.cache_size == 1
    assert first.pool_pred == (0.1, 0.2)
    with pytest.raises(FrozenInstanceError):
        first.catalog_hash = "changed"
    with pytest.raises(TypeError):
        first.pool_res[0] = 9
    assert setup[2].release.bindings[0].adapter == "json-linear.v1"


def test_warm_legacy_entry_reused_and_limit_never_relaxes(setup, monkeypatch):
    inference, descriptor, parent, path, _ = setup
    binding = replace(parent.release.bindings[0], adapter="tier4-serving-fold.v1",
                      members=(descriptor.estimator,))
    calls, original = [], joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    inference._load_artifact(binding)
    entry = next(iter(inference._cache.values()))
    assert entry.member_byte_limit is None
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", path.stat().st_size)
    ref = _load(setup)
    assert entry is next(iter(inference._cache.values()))
    ceiling = entry.member_byte_limit
    inference._load_artifact(binding, max_member_bytes=ceiling + 100)
    assert entry.member_byte_limit == ceiling
    assert _infer(ref).status == "READY"
    assert calls == [1]
    with pytest.raises(ValueError):
        inference._load_artifact(binding, max_member_bytes=1)
    assert entry.member_byte_limit == ceiling


@pytest.mark.parametrize("change", [
    {"parent_release_id": "other"}, {"parent_release_hash": "sha256:" + "a" * 64},
    {"schema_version": "serving_fold_descriptor.v2.0"}, {"role": "iv_crush"},
    {"output_name": "wrong"}, {"pool_pred_field": "wrong"}, {"pool_count": -1},
    {"pool_count": 3}, {"decision_clock_id": "wrong"}, {"fold_start": "2026-10-02"},
])
def test_descriptor_refusals(setup, change):
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$") as err:
        _load(setup, descriptor=replace(setup[1], **change))
    assert err.value.code == MODEL_NOT_READY


@pytest.mark.parametrize("change", [
    {"panel_sha256": "b" * 64}, {"feature_order": ("wrong",)}, {"model_id": "wrong"},
    {"interval_floor": float("nan")}, {"interval_floor": True}, {"interval_policy": "wrong"},
])
def test_policy_refusals(setup, change):
    descriptor = replace(setup[1], policy=replace(setup[1].policy, **change))
    with pytest.raises(loader.ServingFoldError):
        _load(setup, descriptor=descriptor)


def test_parent_catalog_and_identity(setup):
    first = _load(setup)
    other = _load(setup, catalog_hash=content_hash("other-catalog"))
    assert other.release.release_id != first.release.release_id
    changed = replace(setup[1], policy=replace(setup[1].policy, interval_floor=None))
    assert _load(setup, descriptor=changed).release.release_id != first.release.release_id
    assert setup[0].cache_size == 1
    for parent in (replace(setup[2], release_hash="sha256:" + "a" * 64),
                   replace(setup[2], release=replace(setup[2].release, release_id="wrong"))):
        with pytest.raises(loader.ServingFoldError):
            _load(setup, parent=parent)
    with pytest.raises(loader.ServingFoldError):
        _load(setup, catalog_hash="bad")


def test_legacy_minimal_fold_keeps_prediction_acceptance(setup, monkeypatch):
    import hashlib

    inference, descriptor, parent, path, stored = setup
    raw = _bytes({key: stored[key] for key in ("estimator", "features")})
    path.write_bytes(raw)
    member = replace(descriptor.estimator, content_hash="sha256:" + hashlib.sha256(raw).hexdigest())
    binding = replace(parent.release.bindings[0], adapter="tier4-serving-fold.v1", members=(member,))
    release = replace(parent.release, bindings=(binding,))
    request = InferenceRequest(release_id=release.release_id, binding_id=binding.binding_id,
                               feature_order=binding.feature_order, rows=((1.0,) * len(binding.feature_order),))
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", 1)
    assert inference.infer(release, request).status == "READY"
    with pytest.raises(loader.ServingFoldError):
        _load(setup, descriptor=replace(descriptor, estimator=member))
    assert next(iter(inference._cache.values())).member_byte_limit is None
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", len(raw))
    with pytest.raises(loader.ServingFoldError):
        _load(setup, descriptor=replace(descriptor, estimator=member))
    assert inference.infer(release, request).status == "READY"


@pytest.mark.parametrize("key,value", [
    ("model_id", "wrong"), ("fold_start", "2026-11-01"), ("tier3_snapshot", "a" * 64),
    ("pool_pred", [[1, 2]]), ("pool_res", [float("nan"), 0]), ("pool_res", [1]),
    ("pool_res", ["bad", "bad"]), ("pool_res", None),
    ("pool_res", np.array([np.finfo(np.longdouble).max, 0], dtype=np.longdouble)),
])
def test_rehashed_invalid_header_and_pools_refuse(setup, key, value):
    inference, descriptor, parent, path, stored = setup
    stored[key] = value
    if value is None:
        stored.pop(key)
    raw = _bytes(stored)
    path.write_bytes(raw)
    import hashlib

    member = replace(descriptor.estimator, content_hash="sha256:" + hashlib.sha256(raw).hexdigest())
    with pytest.raises(loader.ServingFoldError):
        _load((inference, replace(descriptor, estimator=member), parent, path, stored))


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_thin_pools_are_preserved(setup, count):
    inference, _, parent, _, stored = setup
    stored["pool_pred"] = np.zeros(count)
    stored["pool_res"] = np.ones(count)
    raw = _bytes(stored)
    descriptor = describe_size_fold(raw, _name(stored), _policy(stored), parent)
    path = inference._root / descriptor.estimator.path
    path.write_bytes(raw)
    ref = _load((inference, descriptor, parent, path, stored))
    assert ref.pool_pred == (0.0,) * count
    assert ref.pool_res == (1.0,) * count


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("defect", ["oversize", "hash", "missing", "escape"])
def test_bytes_refused_before_decode_and_warm_inference(setup, monkeypatch, warm, defect):
    inference, descriptor, _, path, _ = setup
    size = path.stat().st_size
    monkeypatch.setattr(loader, "MAX_FOLD_BYTES", size)
    ref = _load(setup) if warm else None
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("must reject before decode"))
    if defect == "oversize":
        path.write_bytes(path.read_bytes() + b"growth")
    elif defect == "hash":
        path.write_bytes(b"x" * size)
    elif defect == "missing":
        path.unlink()
    else:
        descriptor = replace(descriptor, estimator=replace(descriptor.estimator, path="../outside"))
    with pytest.raises(loader.ServingFoldError, match="^serving fold is not ready$"):
        _load(setup, descriptor=descriptor)
    if warm and defect != "escape":
        result = _infer(ref)
        assert result.status == MODEL_NOT_READY
        assert str(inference._root) not in result.detail


def test_bounded_read_uses_limit_plus_one(setup, monkeypatch):
    original, reads = Path.open, []

    class Reader:
        def __init__(self, path, *args, **kwargs):
            self.stream = original(path, *args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size):
            reads.append(size)
            return self.stream.read(size)

    monkeypatch.setattr(Path, "open", lambda path, *a, **k: Reader(path, *a, **k))
    ref = _load(setup)
    _infer(ref)
    assert reads == [loader.MAX_FOLD_BYTES + 1] * 2
