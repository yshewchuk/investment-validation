"""Synthetic size-fold indexing and gate forecast selection (#1104 slice B)."""
from __future__ import annotations

import copy
from datetime import date, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from engine.v2.foundation import content_hash, to_document
from engine.v2.models import deployment
from engine.v2.models.loader import FrozenInference
from engine.v2.models.serving_fold_loader import MAX_FOLD_BYTES, load_serving_fold
from engine.v2.models.serving_folds import SizeFoldPolicy
from engine.v2.scoring.gate_fold_selection import (
    SizeFoldCatalog,
    index_size_folds,
    select_gate_forecast_source,
)
from engine.v2.scoring.release_bindings import ModelNotReady
from tests.fixtures.size_serving_header import HEADER
from tests.v2.models.serving_fold_support import fold_bytes, fold_name, release_inputs
from tools.phase5_serving_folds import describe_size_fold

MEMBER = "tier4_folds:size"


class Predictor:
    def predict(self, rows):
        return np.asarray(rows).sum(axis=1) * 0.01

    def fit(self, *args):
        raise AssertionError("runtime fitting is forbidden")


def _cache(fold_start="2026-10-01"):
    cache = copy.deepcopy(HEADER)
    cache.pop("capture")
    cache.update(estimator=Predictor(), pool_pred=np.array([0.1, 0.2]),
                 pool_res=np.array([-0.01, 0.02]), fold_start=fold_start)
    return cache


def _policy(cache):
    return SizeFoldPolicy(model_id=cache["model_id"], feature_order=tuple(cache["features"]),
                          panel_sha256=cache["tier3_snapshot"], interval_floor=0.0)


def _staged(tmp_path, cache):
    return deployment.stage_release(tmp_path / "deployment", *release_inputs("size", cache))


def _descriptor(cache, parent):
    return describe_size_fold(fold_bytes(cache), fold_name(cache), _policy(cache), parent)


def _object(descriptor, *, name=None, size=1):
    if name is None:
        name = (f"{descriptor.policy.model_id}_"
                f"{date.fromisoformat(descriptor.fold_start):%Y%m}_"
                f"{descriptor.policy.panel_sha256[:12]}.joblib")
    return {"path": descriptor.estimator.path,
            "content_hash": descriptor.estimator.content_hash,
            "name": name, "bytes": size,
            "serving_fold": to_document(descriptor)}


def _row(descriptors, status="STAGED"):
    return {"member_id": MEMBER, "status": status,
            "objects": [_object(descriptor) for descriptor in descriptors]}


def _catalog(*rows, digest=None):
    return {"manifest_hash": digest or content_hash("catalog"),
            "members": list(rows)}


def _fold(tmp_path, fold_start):
    cache = _cache(fold_start)
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    path = tmp_path / descriptor.estimator.path
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(fold_bytes(cache))
    catalog_hash = content_hash("catalog")
    catalog = SizeFoldCatalog(parent=parent, catalog_hash=catalog_hash,
                              descriptors={descriptor.fold_start: descriptor})
    binding = SimpleNamespace(frozen_inference=FrozenInference(tmp_path), size_folds=catalog)
    return SimpleNamespace(binding=binding, descriptor=descriptor, parent=parent,
                           path=path, catalog_hash=catalog_hash, cache=cache)


@pytest.fixture
def fold(tmp_path):
    return _fold(tmp_path, "2026-10-01")


@pytest.fixture
def november_fold(tmp_path):
    return _fold(tmp_path, "2026-11-01")


def test_index_two_real_folds(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    october = _descriptor(cache, parent)
    november = _descriptor(_cache("2026-11-01"), parent)
    indexed = index_size_folds(_catalog(_row([october, november])), parent)
    assert indexed.parent is parent
    assert indexed.catalog_hash == content_hash("catalog")
    assert indexed.descriptors == {"2026-10-01": october, "2026-11-01": november}


def test_index_without_member_is_empty(tmp_path):
    parent = _staged(tmp_path, _cache())
    catalog = _catalog(digest=content_hash("empty"))
    indexed = index_size_folds(catalog, parent)
    assert indexed.descriptors == {}
    assert indexed.parent is parent
    assert indexed.catalog_hash == catalog["manifest_hash"] == content_hash("empty")


def test_index_non_staged_member_is_empty(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    catalog = _catalog(_row([_descriptor(cache, parent)], "MISSING"), digest=content_hash("missing"))
    indexed = index_size_folds(catalog, parent)
    assert indexed.descriptors == {}
    assert indexed.catalog_hash == catalog["manifest_hash"] == content_hash("missing")


def test_index_multiple_member_rows_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(_row([descriptor]), _row([descriptor])), parent)
    assert caught.value.member_id == MEMBER
    assert "size fold catalog is not valid" in caught.value.detail


def test_index_object_without_serving_fold_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    row = {"member_id": MEMBER, "status": "STAGED",
           "objects": [{"path": "objects/x", "content_hash": "sha256:" + "a" * 64}]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(row), parent)
    assert caught.value.member_id == MEMBER
    assert "size fold catalog is not valid" in caught.value.detail


def test_index_estimator_disagreement_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    obj = _object(_descriptor(cache, parent))
    obj["content_hash"] = "sha256:" + "b" * 64
    row = {"member_id": MEMBER, "status": "STAGED", "objects": [obj]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(row), parent)
    assert caught.value.member_id == MEMBER
    assert "size fold catalog is not valid" in caught.value.detail


@pytest.mark.parametrize("name,size", [
    ("wrong.joblib", 1),
    (None, 0),
    (None, MAX_FOLD_BYTES + 1),
    (None, "1"),
])
def test_index_invalid_catalog_object_refuses(tmp_path, name, size):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    obj = _object(descriptor, name=name, size=size)
    row = {"member_id": MEMBER, "status": "STAGED", "objects": [obj]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(row), parent)
    assert caught.value.member_id == MEMBER
    assert "size fold catalog is not valid" in caught.value.detail


def test_index_duplicate_fold_start_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(_row([descriptor, descriptor])), parent)
    assert caught.value.member_id == MEMBER
    assert "size fold catalog is not valid" in caught.value.detail


@pytest.mark.parametrize("event,as_of", [
    ("2026-11-15", "2026-10-01"),
    ("2026-10-20", "2026-11-05"),
    ("2026-10-01", "2026-10-31"),
    (date(2026, 11, 15), datetime(2026, 10, 2, 12, 30)),
])
def test_selection_uses_earlier_month(fold, event, as_of):
    source = select_gate_forecast_source(fold.binding, event, as_of)
    ref = load_serving_fold(fold.binding.frozen_inference, fold.descriptor,
                            verified_parent_manifest=fold.parent,
                            catalog_hash=fold.catalog_hash)
    assert source.binding_id == ref.release.bindings[0].binding_id
    assert source.release == ref.release
    assert source.inference is fold.binding.frozen_inference
    assert source.pool_pred == (0.1, 0.2)
    assert source.pool_res == (-0.01, 0.02)
    assert source.interval_floor == 0.0


def test_selection_missing_month_never_falls_back(fold):
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(fold.binding, "2026-12-05", "2026-12-06")
    assert caught.value.member_id == MEMBER
    assert "no staged size fold for 2026-12-01" in caught.value.detail


def test_selection_corrupt_object_refuses(fold):
    fold.path.write_bytes(b"not a fold")
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(fold.binding, "2026-10-05", "2026-10-06")
    assert caught.value.member_id == MEMBER
    assert "size fold could not be verified or loaded" in caught.value.detail


def test_selection_accepts_iso_datetime_strings(fold):
    source = select_gate_forecast_source(
        fold.binding, "2026-10-15T09:30:00", "2026-10-16T10:00:00")
    ref = load_serving_fold(fold.binding.frozen_inference, fold.descriptor,
                            verified_parent_manifest=fold.parent,
                            catalog_hash=fold.catalog_hash)
    assert source.binding_id == ref.release.bindings[0].binding_id
    assert source.release == ref.release


def test_selection_picks_non_october_fold(november_fold):
    source = select_gate_forecast_source(november_fold.binding, "2026-12-05", "2026-11-02")
    ref = load_serving_fold(november_fold.binding.frozen_inference, november_fold.descriptor,
                            verified_parent_manifest=november_fold.parent,
                            catalog_hash=november_fold.catalog_hash)
    assert ref.descriptor.fold_start == "2026-11-01"
    assert source.binding_id == ref.release.bindings[0].binding_id
    assert source.release == ref.release


def test_selection_uses_fold_policy_interval_floor(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    policy = SizeFoldPolicy(model_id=cache["model_id"],
                            feature_order=tuple(cache["features"]),
                            panel_sha256=cache["tier3_snapshot"],
                            interval_floor=0.037)
    descriptor = describe_size_fold(fold_bytes(cache), fold_name(cache), policy, parent)
    path = tmp_path / descriptor.estimator.path
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(fold_bytes(cache))
    catalog_hash = content_hash("catalog")
    catalog = SizeFoldCatalog(parent=parent, catalog_hash=catalog_hash,
                              descriptors={descriptor.fold_start: descriptor})
    binding = SimpleNamespace(frozen_inference=FrozenInference(tmp_path), size_folds=catalog)
    source = select_gate_forecast_source(binding, "2026-10-05", "2026-10-06")
    assert source.interval_floor == 0.037


@pytest.mark.parametrize("event,as_of", [
    (20261001, "2026-10-01"),
    ("2026-10-01", object()),
    ("not-a-date", "2026-10-01"),
    ("2026-10-01", "2026-13-01"),
])
def test_selection_bad_dates_raise_value_error(fold, event, as_of):
    with pytest.raises(ValueError, match="event_date and as_of must be ISO dates") as caught:
        select_gate_forecast_source(fold.binding, event, as_of)
    assert not isinstance(caught.value, ModelNotReady)


def test_selection_without_catalog_refuses(tmp_path):
    binding = SimpleNamespace(frozen_inference=FrozenInference(tmp_path), size_folds=None)
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(binding, "2026-10-05", "2026-10-06")
    assert caught.value.member_id == MEMBER
    assert "no staged size fold for 2026-10-01" in caught.value.detail


def test_selection_without_inference_refuses(fold):
    binding = SimpleNamespace(frozen_inference=None, size_folds=fold.binding.size_folds)
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(binding, "2026-10-05", "2026-10-06")
    assert caught.value.member_id == MEMBER
    assert "no staged size fold for 2026-10-01" in caught.value.detail
