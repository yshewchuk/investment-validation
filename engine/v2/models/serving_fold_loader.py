"""Verify one catalog-declared fold using the existing inference cache owner."""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date

from engine.v2.foundation import content_hash, from_document, to_document

from .contracts import MODEL_NOT_READY, ModelRelease
from .deployment import StagedManifest, _manifest_hash_matches
from .loader import FrozenInference
from .serving_folds import ServingFoldDescriptor

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


def _hash_valid(text, prefix="sha256:"):
    return (isinstance(text, str) and text.startswith(prefix) and len(text) == len(prefix) + 64
            and all(c in "0123456789abcdef" for c in text[len(prefix):]))


def _validate_descriptor(descriptor):
    policy = descriptor.policy
    month = date.fromisoformat(descriptor.fold_start)
    if (month.day != 1 or month.isoformat() != descriptor.fold_start
            or not _hash_valid(policy.panel_sha256, "")
            or not policy.feature_order or any(not f for f in policy.feature_order)
            or len(set(policy.feature_order)) != len(policy.feature_order)
            or descriptor.pool_count < 0 or descriptor.estimator.name != "estimator"
            or not _hash_valid(descriptor.estimator.content_hash)):
        raise ServingFoldError()


def _view(descriptor, parent, catalog_hash):
    policy = descriptor.policy
    if (not _hash_valid(catalog_hash) or not _manifest_hash_matches(parent)
            or descriptor.parent_release_id != parent.release.release_id
            or descriptor.parent_release_hash != parent.release_hash):
        raise ServingFoldError()
    bindings = [b for b in parent.release.bindings if b.role == "size" and b.strategy_id == "*"]
    if (len(bindings) != 1 or bindings[0].model_id != policy.model_id
            or bindings[0].feature_order != policy.feature_order
            or bindings[0].decision_clock_id != descriptor.decision_clock_id
            or bindings[0].output_names != (descriptor.output_name,)):
        raise ServingFoldError()
    identity = content_hash({"parent": parent.release, "catalog": catalog_hash, "descriptor": descriptor})
    binding = replace(bindings[0], binding_id=identity, adapter="tier4-serving-fold.v1",
                      members=(descriptor.estimator,))
    return replace(parent.release, release_id=identity, bindings=(binding,))


def _pools(artifact, descriptor):
    import numpy as np

    header, policy = artifact.header, descriptor.policy
    if (not callable(getattr(artifact.estimator, "predict", None))
            or header.get("model_id") != policy.model_id
            or header.get("fold_start") != descriptor.fold_start
            or header.get("tier3_snapshot") != policy.panel_sha256
            or not isinstance(header.get("features"), (tuple, list))
            or tuple(header["features"]) != policy.feature_order):
        raise ServingFoldError()
    pools = [np.asarray(header[key]) for key in (descriptor.pool_pred_field, descriptor.pool_res_field)]
    if any(pool.ndim != 1 or pool.dtype.kind not in "fiu"
           or len(pool) != descriptor.pool_count or not np.isfinite(pool).all() for pool in pools):
        raise ServingFoldError()
    return tuple(tuple(float(v) for v in pool) for pool in pools)


def load_serving_fold(inference: FrozenInference, descriptor: ServingFoldDescriptor, *,
                      verified_parent_manifest: StagedManifest, catalog_hash: str) -> ServingFoldRef:
    """Caller verifies catalog hash/membership; this API verifies its declared object."""
    try:
        descriptor = from_document(ServingFoldDescriptor, to_document(descriptor))
        _validate_descriptor(descriptor)
        release = _view(descriptor, verified_parent_manifest, catalog_hash)
        artifact = inference._load_artifact(release.bindings[0], max_member_bytes=MAX_FOLD_BYTES)
        pred, res = _pools(artifact, descriptor)
        return ServingFoldRef(descriptor=descriptor, catalog_hash=catalog_hash,
                              pool_pred=pred, pool_res=res, inference=inference, release=release)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ServingFoldError() from None
