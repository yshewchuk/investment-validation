"""Verify one catalog-declared fold using the existing inference cache owner."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from engine.v2.foundation import content_hash, from_document, to_document

from .contracts import MODEL_NOT_READY, ModelRelease
from .deployment import StagedManifest, _manifest_hash_matches
from .loader import FrozenInference
from .serving_folds import ChooserServingFoldDescriptor, ServingFoldDescriptor

MAX_FOLD_BYTES = 32 * 1024 * 1024


class ServingFoldError(ValueError):
    code = MODEL_NOT_READY

    def __init__(self):
        super().__init__("serving fold is not ready")


@dataclass(frozen=True, kw_only=True)
class ServingFoldRef:
    descriptor: ServingFoldDescriptor
    catalog_hash: str
    pool_pred: tuple[float, ...]
    pool_res: tuple[float, ...]
    inference: FrozenInference
    release: ModelRelease


@dataclass(frozen=True, kw_only=True)
class ChooserServingFoldRef:
    """Verified artifact only: not evidence of causal/source-ready admission."""

    descriptor: ChooserServingFoldDescriptor
    catalog_hash: str
    pool_pred: tuple[float, ...]
    pool_res: tuple[float, ...]
    inference: FrozenInference
    release: ModelRelease


def _hash_valid(text, prefix="sha256:"):
    return (isinstance(text, str) and text.startswith(prefix) and len(text) == len(prefix) + 64
            and all(c in "0123456789abcdef" for c in text[len(prefix):]))


def _validate_descriptor(descriptor):
    # Size descriptors need these shared checks; chooser intrinsic validation overlaps.
    policy = descriptor.policy
    month = date.fromisoformat(descriptor.fold_start)
    if (month.day != 1 or month.isoformat() != descriptor.fold_start
            or not _hash_valid(policy.panel_sha256, "")
            or not policy.feature_order or any(not f for f in policy.feature_order)
            or len(set(policy.feature_order)) != len(policy.feature_order)
            or descriptor.pool_count < 0 or descriptor.estimator.name != "estimator"
            or not _hash_valid(descriptor.estimator.content_hash)):
        raise ServingFoldError()


def validate_serving_fold_descriptor(descriptor):
    """Validate one serving-fold descriptor's intrinsic fields."""
    try:
        _validate_descriptor(descriptor)
    except ServingFoldError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ServingFoldError() from None


def _view(descriptor, parent, catalog_hash, *, role="size"):
    policy = descriptor.policy
    if (not _hash_valid(catalog_hash) or not _manifest_hash_matches(parent)
            or descriptor.parent_release_id != parent.release.release_id
            or descriptor.parent_release_hash != parent.release_hash):
        raise ServingFoldError()
    bindings = [b for b in parent.release.bindings if b.role == role and b.strategy_id == "*"]
    if (len(bindings) != 1 or bindings[0].model_id != policy.model_id
            or bindings[0].feature_order != policy.feature_order
            or bindings[0].decision_clock_id != descriptor.decision_clock_id
            or bindings[0].output_names != (descriptor.output_name,)):
        raise ServingFoldError()
    identity = content_hash({"parent": parent.release, "catalog": catalog_hash, "descriptor": descriptor})
    binding = replace(bindings[0], binding_id=identity, adapter="tier4-serving-fold.v1",
                      members=(descriptor.estimator,))
    return replace(parent.release, release_id=identity, bindings=(binding,))


def _pools(metadata, descriptor):
    # Recheck detached metadata defensively, including the adapter-checked feature order.
    policy = descriptor.policy
    if (metadata.model_id != policy.model_id or metadata.fold_start != descriptor.fold_start
            or metadata.panel_sha256 != policy.panel_sha256
            or metadata.feature_order != policy.feature_order
            or len(metadata.pool_pred) != descriptor.pool_count
            or len(metadata.pool_res) != descriptor.pool_count):
        raise ServingFoldError()
    return metadata.pool_pred, metadata.pool_res


def load_serving_fold(inference: FrozenInference, descriptor: ServingFoldDescriptor, *,
                      verified_parent_manifest: StagedManifest, catalog_hash: str) -> ServingFoldRef:
    """Caller verifies catalog hash/membership; this API verifies its declared object."""
    try:
        descriptor = from_document(ServingFoldDescriptor, to_document(descriptor))
        _validate_descriptor(descriptor)
        release = _view(descriptor, verified_parent_manifest, catalog_hash)
        artifact = inference.prepare_serving_fold(release.bindings[0], max_member_bytes=MAX_FOLD_BYTES)
        pred, res = _pools(artifact, descriptor)
        return ServingFoldRef(descriptor=descriptor, catalog_hash=catalog_hash,
                              pool_pred=pred, pool_res=res, inference=inference, release=release)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ServingFoldError() from None


def _catalog_objects(catalog, role):
    members = catalog.get("members")
    if not isinstance(members, list) or any(not isinstance(row, dict) for row in members):
        raise ServingFoldError()
    rows = [row for row in members if row.get("member_id") == f"tier4_folds:{role}"]
    if len(rows) != 1 or rows[0].get("status") != "STAGED":
        raise ServingFoldError()
    objects = rows[0].get("objects")
    if not isinstance(objects, list) or any(not isinstance(obj, dict) for obj in objects):
        raise ServingFoldError()
    names = [obj.get("name") for obj in objects]
    if len(set(names)) != len(names):
        raise ServingFoldError()
    return objects


def _validate_catalog_object(obj, descriptor):
    member = descriptor.estimator
    name = (f"{descriptor.policy.model_id}_{date.fromisoformat(descriptor.fold_start):%Y%m}_"
            f"{descriptor.policy.panel_sha256[:12]}.joblib")
    if (obj.get("path") != member.path or obj.get("content_hash") != member.content_hash
            or obj.get("name") != name or type(obj.get("bytes")) is not int
            or not 0 < obj["bytes"] <= MAX_FOLD_BYTES):
        raise ServingFoldError()


def validate_serving_fold_object(obj, descriptor):
    """Validate a catalog object against the serving-fold descriptor."""
    _validate_catalog_object(obj, descriptor)


def _chooser_catalog_hash(catalog, descriptor, parent):
    if (not isinstance(catalog, dict)
            or catalog.get("schema_version") != "phase5_staged_release.v1.0"
            or catalog.get("release_id") != parent.release.release_id
            or catalog.get("deployment_id") != parent.release.deployment_id):
        raise ServingFoldError()
    digest = catalog.get("manifest_hash")
    if (not _hash_valid(digest)
            or digest != content_hash({k: v for k, v in catalog.items() if k != "manifest_hash"})):
        raise ServingFoldError()
    objects = _catalog_objects(catalog, descriptor.policy.role)
    matches = [obj for obj in objects if obj.get("serving_fold") == to_document(descriptor)]
    if len(matches) != 1:
        raise ServingFoldError()
    _validate_catalog_object(matches[0], descriptor)
    return digest, matches[0]["bytes"]


def load_chooser_serving_fold(inference: FrozenInference, descriptor: ChooserServingFoldDescriptor, *,
                             verified_parent_manifest: StagedManifest,
                             verified_catalog: dict) -> ChooserServingFoldRef:
    """Verify the exact declared chooser artifact, never select/admit sources.

    Caller supplies its verified catalog/parent; membership and semantic identity
    are rechecked here. Source finality/publication and causal training bounds are
    not in the descriptor and cannot be established by this artifact-only API.
    """
    try:
        descriptor = from_document(ChooserServingFoldDescriptor, to_document(descriptor))
        _validate_descriptor(descriptor)
        catalog_hash, expected_bytes = _chooser_catalog_hash(verified_catalog, descriptor, verified_parent_manifest)
        release = _view(descriptor, verified_parent_manifest, catalog_hash, role=descriptor.policy.role)
        artifact = inference.prepare_serving_fold(release.bindings[0], max_member_bytes=MAX_FOLD_BYTES,
                                                  expected_bytes=expected_bytes)
        pred, res = _pools(artifact, descriptor)
        return ChooserServingFoldRef(descriptor=descriptor, catalog_hash=catalog_hash,
                                     pool_pred=pred, pool_res=res, inference=inference, release=release)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ServingFoldError() from None
