"""Causal size-fold indexing and gate forecast selection (see ARCHITECTURE.md)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType
from typing import Any

from engine.v2.foundation import from_document
from engine.v2.models.deployment import StagedManifest
from engine.v2.models.serving_fold_loader import ServingFoldError, load_serving_fold
from engine.v2.models.serving_folds import ServingFoldDescriptor
from engine.v2.scoring.gate_forecast_source import GateForecastSource

__all__ = ["SizeFoldCatalog", "index_size_folds", "select_gate_forecast_source"]

_MEMBER_ID = "tier4_folds:size"
_INVALID = "size fold catalog is not valid"


@dataclass(frozen=True, kw_only=True)
class SizeFoldCatalog:
    """The staged size folds of one verified release, descriptors only."""

    parent: StagedManifest
    catalog_hash: str
    descriptors: Mapping[str, ServingFoldDescriptor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", MappingProxyType(dict(self.descriptors)))


def index_size_folds(catalog: Mapping, parent: StagedManifest) -> SizeFoldCatalog:
    """Index the ``tier4_folds:size`` row's descriptors without loading a fold."""
    from engine.v2.scoring.release_bindings import ModelNotReady

    if not isinstance(catalog, Mapping):
        raise ModelNotReady(_MEMBER_ID, _INVALID)
    try:
        rows = _size_fold_rows(catalog)
        if len(rows) != 1 or rows[0].get("status") != "STAGED":
            return SizeFoldCatalog(
                parent=parent, catalog_hash=catalog.get("manifest_hash"), descriptors={})
        objects = rows[0].get("objects")
        if not isinstance(objects, list):
            raise ValueError("objects")
        descriptors: dict[str, ServingFoldDescriptor] = {}
        for obj in objects:
            descriptor = _descriptor_of(obj)
            if descriptor.fold_start in descriptors:
                raise ValueError("duplicate")
            descriptors[descriptor.fold_start] = descriptor
        return SizeFoldCatalog(
            parent=parent, catalog_hash=catalog.get("manifest_hash"), descriptors=descriptors)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ModelNotReady(_MEMBER_ID, _INVALID) from None


def _size_fold_rows(catalog: Mapping) -> list:
    from engine.v2.scoring.release_bindings import ModelNotReady

    members = catalog.get("members")
    rows = ([row for row in members
             if isinstance(row, Mapping) and row.get("member_id") == _MEMBER_ID]
            if isinstance(members, list) else [])
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
    if (descriptor.estimator.path != path
            or descriptor.estimator.content_hash != digest):
        raise ValueError("estimator")
    return descriptor


def select_gate_forecast_source(binding, event_date, as_of) -> GateForecastSource:
    """Load the fold for the earlier of the event's and decision's month."""
    from engine.v2.scoring.release_bindings import ModelNotReady

    fold_start = min(_month(event_date), _month(as_of)).isoformat()
    catalog = binding.size_folds
    descriptor = catalog.descriptors.get(fold_start) if catalog is not None else None
    if descriptor is None:
        raise ModelNotReady(_MEMBER_ID, f"no staged size fold for {fold_start}")
    try:
        ref = load_serving_fold(
            binding.frozen_inference, descriptor,
            verified_parent_manifest=catalog.parent, catalog_hash=catalog.catalog_hash)
    except ServingFoldError:
        raise ModelNotReady(
            _MEMBER_ID, "size fold could not be verified or loaded") from None
    selected = ref.release.bindings[0]
    return GateForecastSource(
        binding_id=selected.binding_id, release=ref.release, inference=ref.inference,
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
