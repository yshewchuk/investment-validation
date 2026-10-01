"""Offline size-fold declaration from bounded cache bytes and registered policy.

The descriptor is preparation metadata, not a runtime loader or inference view.
Legacy cache decoding stays outside the native model package.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass
from datetime import date

from engine.v2.models import ArtifactMember, ModelRelease
from engine.v2.models.deployment import StagedManifest
from engine.v2.models.serving_folds import ServingFoldDescriptor, SizeFoldPolicy

MAX_FOLD_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ValidatedSizeFold:
    """Immutable header/pool metadata only; never carries estimator or pool arrays."""

    fold_start: str
    decision_clock_id: str
    digest: str
    pool_count: int
    policy: SizeFoldPolicy
    release: ModelRelease


def preflight_size_fold(data: bytes, name: str, policy: SizeFoldPolicy,
                        release: ModelRelease) -> ValidatedSizeFold:
    """Pure authoring validation of bounded bytes against a release; refuse with fixed messages."""
    if len(data) > MAX_FOLD_BYTES:
        raise ValueError("serving fold exceeds byte limit")
    if (not isinstance(policy.panel_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", policy.panel_sha256)
            or not policy.feature_order
            or any(not isinstance(key, str) or not key for key in policy.feature_order)
            or len(set(policy.feature_order)) != len(policy.feature_order)
            or policy.interval_policy != "size-heldout-deciles.v1"
            or (policy.interval_floor is not None
                and (type(policy.interval_floor) not in (int, float)
                     or not math.isfinite(policy.interval_floor)))):
        raise ValueError("invalid size serving policy")
    bindings = [b for b in release.bindings if b.role == "size" and b.strategy_id == "*"]
    if (len(bindings) != 1 or bindings[0].model_id != policy.model_id
            or bindings[0].feature_order != policy.feature_order
            or bindings[0].output_names != ("forecast_abs_move",)):
        raise ValueError("size serving policy disagrees with release binding")
    try:
        import joblib

        stored = joblib.load(io.BytesIO(data))
    except Exception as exc:
        raise ValueError("serving fold cannot be decoded") from exc
    if (not isinstance(stored, dict) or not callable(getattr(stored.get("estimator"), "predict", None))
            or stored.get("model_id") != policy.model_id
            or stored.get("tier3_snapshot") != policy.panel_sha256
            or not isinstance(stored.get("features"), (list, tuple))
            or tuple(stored["features"]) != policy.feature_order):
        raise ValueError("serving fold header disagrees with policy")
    try:
        fold = date.fromisoformat(stored["fold_start"])
        if fold.day != 1 or fold.isoformat() != stored["fold_start"]:
            raise ValueError("not an exact month start")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid serving fold month") from exc
    expected = f"{policy.model_id}_{fold:%Y%m}_{policy.panel_sha256[:12]}.joblib"
    if name != expected:
        raise ValueError("serving fold filename disagrees with header")
    try:
        import numpy as np

        pools = [np.asarray(stored[key]) for key in ("pool_pred", "pool_res")]
        if (any(pool.ndim != 1 or pool.dtype.kind not in "fiu"
                or not np.isfinite(pool).all() for pool in pools)
                or pools[0].shape != pools[1].shape):
            raise ValueError("invalid paired arrays")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid serving fold held-out pool") from exc
    return ValidatedSizeFold(fold_start=fold.isoformat(),
                             decision_clock_id=bindings[0].decision_clock_id,
                             digest=hashlib.sha256(data).hexdigest(), pool_count=len(pools[0]),
                             policy=policy, release=release)


def _descriptor_from_validated(validated: ValidatedSizeFold,
                               manifest: StagedManifest) -> ServingFoldDescriptor:
    """Trusted construction from retained validated metadata; refuses a different staged release."""
    if manifest.release != validated.release:
        raise ValueError("staged release disagrees with validated fold")
    return ServingFoldDescriptor(
        parent_release_id=manifest.release.release_id, parent_release_hash=manifest.release_hash,
        policy=validated.policy, fold_start=validated.fold_start,
        decision_clock_id=validated.decision_clock_id,
        estimator=ArtifactMember(name="estimator", path=f"objects/{validated.digest}",
                                 content_hash=f"sha256:{validated.digest}"),
        pool_count=validated.pool_count)


def descriptor_from_preflight(validated: ValidatedSizeFold, policy: SizeFoldPolicy,
                              manifest: StagedManifest) -> ServingFoldDescriptor:
    """Describe staged folds only under the exact validated policy; no independent substitution."""
    if policy != validated.policy:
        raise ValueError("descriptor policy disagrees with validated fold")
    return _descriptor_from_validated(validated, manifest)


def describe_size_fold(data: bytes, name: str, policy: SizeFoldPolicy,
                       manifest: StagedManifest) -> ServingFoldDescriptor:
    """Validate authoring inputs without fitting; refuse with fixed messages."""
    return _descriptor_from_validated(preflight_size_fold(data, name, policy, manifest.release),
                                      manifest)
