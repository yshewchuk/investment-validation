"""Synthetic size-fold indexing and gate forecast selection (#1104 slice B)."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import date, datetime
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from engine.v2.foundation import content_hash, to_document
from engine.v2.models import deployment
from engine.v2.models.loader import FrozenInference
from engine.v2.models.serving_fold_loader import (
    MAX_FOLD_BYTES,
    ServingFoldError,
    load_serving_fold,
    validate_serving_fold_descriptor,
)
from engine.v2.models.serving_folds import SizeFoldPolicy
from engine.v2.scoring.gate_fold_selection import (
    SizeFoldCatalog,
    index_size_folds,
    select_gate_forecast_source,
)
from engine.v2.scoring.release_bindings import ModelNotReady, resolve_release_binding
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


_SCHEMA = "phase5_staged_release.v1.0"


def _sealed(body):
    body = dict(body)
    body["manifest_hash"] = content_hash({k: v for k, v in body.items() if k != "manifest_hash"})
    return body


def _catalog(parent, *rows, digest=None, schema_version=_SCHEMA, release_id=None):
    body = _sealed({
        "schema_version": schema_version,
        "release_id": parent.release.release_id if release_id is None else release_id,
        "members": list(rows),
    })
    if digest is not None:
        body["manifest_hash"] = digest
    return body


def _assert_model_not_ready(caught, detail=None):
    assert caught.value.code == "MODEL_NOT_READY"
    assert caught.value.member_id == MEMBER
    if detail is not None:
        assert detail in caught.value.detail


def _fold(tmp_path, fold_start):
    cache = _cache(fold_start)
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    path = tmp_path / descriptor.estimator.path
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(fold_bytes(cache))
    catalog_dict = _catalog(parent, _row([descriptor]))
    catalog = index_size_folds(catalog_dict, parent)
    catalog_hash = catalog.catalog_hash
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
    catalog = _catalog(parent, _row([october, november]))
    indexed = index_size_folds(catalog, parent)
    assert indexed.parent is parent
    assert indexed.catalog_hash == catalog["manifest_hash"]
    assert indexed.descriptors == {"2026-10-01": october, "2026-11-01": november}


def test_index_without_member_is_empty(tmp_path):
    parent = _staged(tmp_path, _cache())
    catalog = _catalog(parent)
    indexed = index_size_folds(catalog, parent)
    assert indexed.descriptors == {}
    assert indexed.parent is parent
    assert indexed.catalog_hash == catalog["manifest_hash"]


def test_index_non_staged_member_is_empty(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    catalog = _catalog(parent, _row([_descriptor(cache, parent)], "MISSING"))
    indexed = index_size_folds(catalog, parent)
    assert indexed.descriptors == {}
    assert indexed.catalog_hash == catalog["manifest_hash"]


def test_index_multiple_member_rows_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, _row([descriptor]), _row([descriptor])), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_object_without_serving_fold_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    row = {"member_id": MEMBER, "status": "STAGED",
           "objects": [{"path": "objects/x", "content_hash": "sha256:" + "a" * 64}]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, row), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_estimator_disagreement_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    obj = _object(_descriptor(cache, parent))
    obj["content_hash"] = "sha256:" + "b" * 64
    row = {"member_id": MEMBER, "status": "STAGED", "objects": [obj]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, row), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


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
        index_size_folds(_catalog(parent, row), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_duplicate_fold_start_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, _row([descriptor, descriptor])), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_size_fold_catalog_descriptors_are_read_only(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    catalog = index_size_folds(_catalog(parent, _row([descriptor])), parent)
    with pytest.raises(TypeError):
        catalog.descriptors["2026-11-01"] = descriptor


def test_size_fold_catalog_requires_issuing(tmp_path):
    parent = _staged(tmp_path, _cache())
    with pytest.raises(TypeError):
        SizeFoldCatalog(parent=parent, catalog_hash="x", descriptors={})


@pytest.mark.parametrize("mutate", [
    lambda d: replace(d, pool_count=True),
    lambda d: replace(d, role="iv_crush"),
    lambda d: replace(d, policy=replace(d.policy, interval_floor=float("nan"))),
    lambda d: replace(d, policy=replace(d.policy, interval_floor=float("inf"))),
])
def test_validate_serving_fold_descriptor_refuses_untyped_fields(tmp_path, mutate):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    with pytest.raises(ServingFoldError):
        validate_serving_fold_descriptor(mutate(descriptor))


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
    _assert_model_not_ready(caught, "no staged size fold for 2026-12-01")


def test_selection_corrupt_object_refuses(fold):
    fold.path.write_bytes(b"not a fold")
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(fold.binding, "2026-10-05", "2026-10-06")
    _assert_model_not_ready(caught, "size fold could not be verified or loaded")


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


def test_selection_refuses_catalog_keyed_to_wrong_month(november_fold, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("load_serving_fold must not run for a miskeyed catalog")

    monkeypatch.setattr("engine.v2.scoring.gate_fold_selection.load_serving_fold", fail)
    catalog = index_size_folds(
        _catalog(november_fold.parent, _row([november_fold.descriptor])),
        november_fold.parent)
    object.__setattr__(
        catalog, "descriptors",
        MappingProxyType({"2026-10-01": november_fold.descriptor}))
    binding = SimpleNamespace(
        frozen_inference=november_fold.binding.frozen_inference, size_folds=catalog)
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(binding, "2026-10-05", "2026-10-06")
    _assert_model_not_ready(caught, "no staged size fold for 2026-10-01")


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
    catalog = index_size_folds(_catalog(parent, _row([descriptor])), parent)
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
    _assert_model_not_ready(caught, "no staged size fold for 2026-10-01")


def test_selection_without_inference_refuses(fold):
    binding = SimpleNamespace(frozen_inference=None, size_folds=fold.binding.size_folds)
    with pytest.raises(ModelNotReady) as caught:
        select_gate_forecast_source(binding, "2026-10-05", "2026-10-06")
    _assert_model_not_ready(caught, "no staged size fold for 2026-10-01")



@pytest.mark.parametrize("schema_version", ["phase5_staged_release.v9.9", None, 1])
def test_index_wrong_schema_version_refuses(tmp_path, schema_version):
    parent = _staged(tmp_path, _cache())
    catalog = _catalog(parent, schema_version=schema_version)
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(catalog, parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_wrong_release_id_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    catalog = _catalog(parent, release_id="some-other-release")
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(catalog, parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_wrong_manifest_hash_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    catalog = _catalog(parent, digest="sha256:" + "0" * 64)
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(catalog, parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_non_mapping_catalog_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds([], parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_members_not_a_list_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    catalog = _sealed({"schema_version": _SCHEMA, "release_id": parent.release.release_id,
                       "members": "not-a-list"})
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(catalog, parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_non_mapping_member_row_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    catalog = _sealed({"schema_version": _SCHEMA, "release_id": parent.release.release_id,
                       "members": ["not-a-row"]})
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(catalog, parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_descriptor_mid_month_refuses(tmp_path):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    descriptor = replace(_descriptor(cache, parent), fold_start="2026-10-15")
    row = {"member_id": MEMBER, "status": "STAGED", "objects": [_object(descriptor)]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, row), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


@pytest.mark.parametrize("mutate", [
    lambda obj: obj.update(path=1),
    lambda obj: obj.pop("content_hash"),
    lambda obj: obj.pop("serving_fold"),
])
def test_index_object_odd_types_refuse(tmp_path, mutate):
    cache = _cache()
    parent = _staged(tmp_path, cache)
    obj = _object(_descriptor(cache, parent))
    mutate(obj)
    row = {"member_id": MEMBER, "status": "STAGED", "objects": [obj]}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, row), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def test_index_objects_not_a_list_refuses(tmp_path):
    parent = _staged(tmp_path, _cache())
    row = {"member_id": MEMBER, "status": "STAGED", "objects": "not-a-list"}
    with pytest.raises(ModelNotReady) as caught:
        index_size_folds(_catalog(parent, row), parent)
    _assert_model_not_ready(caught, "size fold catalog is not valid")


def _three_folds(tmp_path):
    parent = _staged(tmp_path, _cache("2026-08-01"))
    descriptors = {}
    pools = {}
    for index, month in enumerate(("2026-08-01", "2026-09-01", "2026-10-01")):
        cache = _cache(month)
        cache.update(pool_pred=np.array([0.1 + index, 0.2 + index]),
                     pool_res=np.array([-0.01 - index, 0.02 + index]))
        descriptor = _descriptor(cache, parent)
        path = tmp_path / descriptor.estimator.path
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(fold_bytes(cache))
        descriptors[month] = descriptor
        pools[month] = (tuple(cache["pool_pred"]), tuple(cache["pool_res"]))
    catalog = _catalog(parent, _row(list(descriptors.values())))
    indexed = index_size_folds(catalog, parent)
    binding = SimpleNamespace(frozen_inference=FrozenInference(tmp_path), size_folds=indexed)
    return SimpleNamespace(binding=binding, parent=parent, descriptors=descriptors,
                           catalog=indexed, pools=pools)


@pytest.fixture
def three_folds(tmp_path):
    return _three_folds(tmp_path)


@pytest.mark.parametrize("event,as_of,expected", [
    ("2026-08-15", "2026-09-02", "2026-08-01"),
    ("2026-09-20", "2026-10-05", "2026-09-01"),
    ("2026-10-10", "2026-11-06", "2026-10-01"),
    ("2026-11-15", "2026-10-31", "2026-10-01"),
    ("2026-10-05", "2026-09-06", "2026-09-01"),
    ("2026-09-30", "2026-09-01", "2026-09-01"),
    ("2026-08-31", "2026-08-01", "2026-08-01"),
])
def test_selection_among_three_folds(three_folds, event, as_of, expected):
    source = select_gate_forecast_source(three_folds.binding, event, as_of)
    descriptor = three_folds.descriptors[expected]
    ref = load_serving_fold(three_folds.binding.frozen_inference, descriptor,
                            verified_parent_manifest=three_folds.parent,
                            catalog_hash=three_folds.catalog.catalog_hash)
    assert source.binding_id == ref.release.bindings[0].binding_id
    assert source.release == ref.release
    assert source.pool_pred == ref.pool_pred
    assert source.pool_res == ref.pool_res
    expected_pools = three_folds.pools[expected]
    assert source.pool_pred == expected_pools[0]
    assert source.pool_res == expected_pools[1]
    other_pools = [pools for month, pools in three_folds.pools.items() if month != expected]
    assert all(pools[0] != expected_pools[0] for pools in other_pools)


def _stage_and_promote(tmp_path, cache):
    dep_root = tmp_path / "deployment"
    release, inventory, payloads = release_inputs("size", cache)
    manifest = deployment.stage_release(dep_root, release, inventory, payloads)
    deployment.mark_staging_succeeded(dep_root, release.release_id)
    deployment.promote(dep_root, release.release_id)
    return manifest


def test_resolve_release_binding_indexes_size_folds(tmp_path):
    cache = _cache()
    parent = _stage_and_promote(tmp_path, cache)
    october = _descriptor(cache, parent)
    november = _descriptor(_cache("2026-11-01"), parent)
    catalog = _catalog(parent, _row([october, november]))
    (tmp_path / "phase5_release.json").write_text(json.dumps(catalog, sort_keys=True))
    binding = resolve_release_binding(tmp_path)
    size_folds = binding.size_folds
    assert size_folds is not None
    assert size_folds.catalog_hash == catalog["manifest_hash"]
    assert size_folds.descriptors == {"2026-10-01": october, "2026-11-01": november}
    for key, descriptor in size_folds.descriptors.items():
        assert key == descriptor.fold_start
        assert descriptor.estimator.name == "estimator"


def test_resolve_release_binding_refuses_bad_catalog_hash(tmp_path):
    cache = _cache()
    parent = _stage_and_promote(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    catalog = _catalog(parent, _row([descriptor]), digest="sha256:" + "0" * 64)
    (tmp_path / "phase5_release.json").write_text(json.dumps(catalog, sort_keys=True))
    with pytest.raises(ModelNotReady) as caught:
        resolve_release_binding(tmp_path)
    assert caught.value.code == "MODEL_NOT_READY"
    assert caught.value.member_id == "phase5_release.json"
    assert "manifest_hash" in caught.value.detail


def test_resolve_release_binding_refuses_malformed_size_fold_row(tmp_path):
    cache = _cache()
    parent = _stage_and_promote(tmp_path, cache)
    descriptor = _descriptor(cache, parent)
    row = _row([descriptor])
    row["objects"] = "not-a-list"
    catalog = _catalog(parent, row)
    (tmp_path / "phase5_release.json").write_text(json.dumps(catalog, sort_keys=True))
    with pytest.raises(ModelNotReady) as caught:
        resolve_release_binding(tmp_path)
    _assert_model_not_ready(caught, "size fold catalog is not valid")
