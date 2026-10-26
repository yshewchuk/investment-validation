"""Causal size-fold indexing and gate forecast selection (see ARCHITECTURE.md)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from engine.v2.foundation import content_hash, from_document, to_document
from engine.v2.models.deployment import StagedManifest
from engine.v2.models.serving_fold_loader import (
    ServingFoldError,
    load_serving_fold,
    validate_serving_fold_descriptor,
    validate_serving_fold_object,
)
from engine.v2.models.serving_folds import ServingFoldDescriptor
from engine.v2.scoring.gate_forecast_source import GateForecastSource

__all__ = ["SizeFoldCatalog", "index_size_folds", "select_gate_forecast_source"]

_MEMBER_ID = "tier4_folds:size"
_CATALOG_MEMBER_ID = "phase5_release.json"
_INVALID = "size fold catalog is not valid"
_PHASE5_RELEASE_SCHEMA = "phase5_staged_release.v1.0"


def _proof(catalog_hash: str, descriptors: Mapping[str, ServingFoldDescriptor],
           object_bytes: Mapping[str, int]) -> str:
    return content_hash({
        "catalog": catalog_hash,
        "descriptors": {k: to_document(v) for k, v in sorted(descriptors.items())},
        "object_bytes": {k: v for k, v in sorted(object_bytes.items())},
    })


@dataclass(frozen=True, kw_only=True)
class SizeFoldCatalog:
    """The staged size folds of one verified release, descriptors and object bytes."""

    parent: StagedManifest
    catalog_hash: str
    descriptors: Mapping[str, ServingFoldDescriptor]
    object_bytes: Mapping[str, int]
    _issued: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._issued != _proof(self.catalog_hash, dict(self.descriptors),
                                  dict(self.object_bytes)):
            raise TypeError("SizeFoldCatalog proof does not match its contents")
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))
        object.__setattr__(self, "object_bytes", MappingProxyType(dict(self.object_bytes)))


def index_size_folds(catalog: Mapping, parent: StagedManifest) -> SizeFoldCatalog:
    """Index the ``tier4_folds:size`` row's descriptors without loading a fold."""
    from engine.v2.scoring.release_bindings import ModelNotReady

    if not isinstance(catalog, Mapping):
        raise ModelNotReady(_CATALOG_MEMBER_ID, _INVALID)
    try:
        if catalog.get("schema_version") != _PHASE5_RELEASE_SCHEMA:
            raise ValueError("schema_version")
        if catalog.get("release_id") != parent.release.release_id:
            raise ValueError("release_id")
        digest = catalog.get("manifest_hash")
        if digest != content_hash({k: v for k, v in catalog.items() if k != "manifest_hash"}):
            raise ValueError("manifest_hash")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ModelNotReady(_CATALOG_MEMBER_ID, _INVALID) from None
    rows = _size_fold_rows(catalog)
    try:
        if len(rows) != 1 or rows[0].get("status") != "STAGED":
            return SizeFoldCatalog(
                parent=parent, catalog_hash=digest, descriptors={}, object_bytes={},
                _issued=_proof(digest, {}, {}))
        objects = rows[0].get("objects")
        if not isinstance(objects, list):
            raise ValueError("objects")
        descriptors, object_bytes = _fold_entries(objects)
        return SizeFoldCatalog(
            parent=parent, catalog_hash=digest, descriptors=descriptors,
            object_bytes=object_bytes, _issued=_proof(digest, descriptors, object_bytes))
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ModelNotReady(_MEMBER_ID, _INVALID) from None


def _size_fold_rows(catalog: Mapping) -> list:
    from engine.v2.scoring.release_bindings import ModelNotReady

    members = catalog.get("members")
    if not isinstance(members, list):
        raise ModelNotReady(_CATALOG_MEMBER_ID, _INVALID)
    if any(not isinstance(row, Mapping) for row in members):
        raise ModelNotReady(_CATALOG_MEMBER_ID, _INVALID)
    if any(not isinstance(row.get("member_id", ""), str) for row in members):
        raise ModelNotReady(_CATALOG_MEMBER_ID, _INVALID)
    rows = [row for row in members if row.get("member_id") == _MEMBER_ID]
    if len(rows) > 1:
        raise ModelNotReady(_MEMBER_ID, _INVALID)
    return rows


def _descriptor_of(obj: Any) -> ServingFoldDescriptor:
    if not isinstance(obj, Mapping):
        raise ValueError("object")
    path, digest = obj.get("path"), obj.get("content_hash")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise ValueError("identity")
    descriptor = from_document(ServingFoldDescriptor, obj["serving_fold"])
    try:
        validate_serving_fold_descriptor(descriptor)
    except ServingFoldError:
        raise ValueError("descriptor") from None
    if (descriptor.estimator.path != path
            or descriptor.estimator.content_hash != digest):
        raise ValueError("estimator")
    try:
        validate_serving_fold_object(obj, descriptor)
    except ServingFoldError:
        raise ValueError("catalog object") from None
    return descriptor


def _fold_entries(objects: list) -> tuple[dict[str, ServingFoldDescriptor], dict[str, int]]:
    descriptors: dict[str, ServingFoldDescriptor] = {}
    object_bytes: dict[str, int] = {}
    for obj in objects:
        descriptor = _descriptor_of(obj)
        size = obj.get("bytes")
        if type(size) is not int:
            raise ValueError("bytes")
        if descriptor.fold_start in descriptors:
            raise ValueError("duplicate")
        descriptors[descriptor.fold_start] = descriptor
        object_bytes[descriptor.fold_start] = size
    return descriptors, object_bytes


def select_gate_forecast_source(binding, event_date, as_of) -> GateForecastSource:
    """Load the fold for the earlier of the event's and decision's month."""
    from engine.v2.scoring.release_bindings import ModelNotReady

    fold_start = min(_month(event_date), _month(as_of)).isoformat()
    catalog = getattr(binding, "size_folds", None)
    descriptor = (catalog.descriptors.get(fold_start)
                  if isinstance(catalog, SizeFoldCatalog) else None)
    if (descriptor is None or descriptor.fold_start != fold_start
            or getattr(binding, "frozen_inference", None) is None):
        raise ModelNotReady(_MEMBER_ID, f"no staged size fold for {fold_start}")
    try:
        ref = load_serving_fold(
            binding.frozen_inference, descriptor,
            verified_parent_manifest=catalog.parent, catalog_hash=catalog.catalog_hash,
            expected_bytes=catalog.object_bytes[fold_start])
    except ServingFoldError:
        raise ModelNotReady(
            _MEMBER_ID, "size fold could not be verified or loaded") from None
    selected = ref.release.bindings[0]
    return GateForecastSource(
        binding_id=selected.binding_id,
        artifact_hash=ref.descriptor.estimator.content_hash,
        release=ref.release, inference=ref.inference,
        pool_pred=ref.pool_pred, pool_res=ref.pool_res,
        interval_floor=ref.descriptor.policy.interval_floor)


def _month(value: Any) -> date:
    if isinstance(value, datetime):
        value = value.date()
    elif isinstance(value, str):
        try:
            value = (datetime.fromisoformat(value).date() if len(value) > 10 and value[10] in "T "
                     else date.fromisoformat(value))
        except ValueError:
            raise ValueError("event_date and as_of must be ISO dates") from None
    elif not isinstance(value, date):
        raise ValueError("event_date and as_of must be ISO dates")
    return value.replace(day=1)
