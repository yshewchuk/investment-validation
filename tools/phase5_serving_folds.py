"""Offline role-specific fold declarations from bounded bytes and registered policy.

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

from engine.v2.foundation import from_document, to_document
from engine.v2.models import ArtifactMember, ModelRelease
from engine.v2.models.deployment import StagedManifest, semantic_release_hash
from engine.v2.models.serving_folds import (
    ChooserFoldPolicy, ChooserServingFoldDescriptor, ServingFoldDescriptor, SizeFoldPolicy,
    chooser_fold_outputs,
)

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


@dataclass(frozen=True)
class ValidatedChooserFold:
    """Validated chooser metadata; never reuses a size descriptor or retains arrays."""

    fold_start: str
    decision_clock_id: str
    digest: str
    pool_count: int
    policy: ChooserFoldPolicy
    release: ModelRelease


def preflight_size_fold(data: bytes, name: str, policy: SizeFoldPolicy,
                        release: ModelRelease) -> ValidatedSizeFold:
    """Pure authoring validation of bounded bytes against a release; refuse with fixed messages."""
    return _preflight_fold(data, name, policy, release, role="size",
                           output="forecast_abs_move", interval_policy="size-heldout-deciles.v1",
                           result_type=ValidatedSizeFold)


def preflight_chooser_fold(data: bytes, name: str, policy: ChooserFoldPolicy,
                           release: ModelRelease) -> ValidatedChooserFold:
    try:
        policy = from_document(ChooserFoldPolicy, to_document(policy))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid chooser serving policy") from exc
    output, _column = chooser_fold_outputs(policy.role)
    return _preflight_fold(data, name, policy, release, role=policy.role, output=output,
                           interval_policy="chooser-heldout-deciles.v1", result_type=ValidatedChooserFold)


def _preflight_fold(data, name, policy, release, *, role, output, interval_policy, result_type):
    if len(data) > MAX_FOLD_BYTES:
        raise ValueError("serving fold exceeds byte limit")
    if (not isinstance(policy.panel_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", policy.panel_sha256)
            or not policy.feature_order
            or any(not isinstance(key, str) or not key for key in policy.feature_order)
            or len(set(policy.feature_order)) != len(policy.feature_order)
            or policy.interval_policy != interval_policy
            or (policy.interval_floor is not None
                and (type(policy.interval_floor) not in (int, float)
                     or not math.isfinite(policy.interval_floor)))):
        raise ValueError(f"invalid {role} serving policy")
    bindings = [b for b in release.bindings if b.role == role and b.strategy_id == "*"]
    if (len(bindings) != 1 or bindings[0].model_id != policy.model_id
            or bindings[0].feature_order != policy.feature_order
            or bindings[0].output_names != (output,)):
        raise ValueError(f"{role} serving policy disagrees with release binding")
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
    return result_type(fold_start=fold.isoformat(),
                       decision_clock_id=bindings[0].decision_clock_id,
                       digest=hashlib.sha256(data).hexdigest(), pool_count=len(pools[0]),
                       policy=policy, release=release)


def _descriptor_from_validated(validated: ValidatedSizeFold | ValidatedChooserFold,
                               manifest: StagedManifest) -> ServingFoldDescriptor | ChooserServingFoldDescriptor:
    """Trusted construction from retained validated metadata; refuses a different staged release."""
    if semantic_release_hash(manifest.release) != semantic_release_hash(validated.release):
        raise ValueError("staged release disagrees with validated fold")
    descriptor, extra = ServingFoldDescriptor, {}
    if isinstance(validated, ValidatedChooserFold):
        output, column = chooser_fold_outputs(validated.policy.role)
        descriptor, extra = ChooserServingFoldDescriptor, {"output_name": output, "producer_output": column}
    return descriptor(
        parent_release_id=manifest.release.release_id, parent_release_hash=manifest.release_hash,
        policy=validated.policy, fold_start=validated.fold_start,
        decision_clock_id=validated.decision_clock_id,
        estimator=ArtifactMember(name="estimator", path=f"objects/{validated.digest}",
                                 content_hash=f"sha256:{validated.digest}"),
        pool_count=validated.pool_count, **extra)


def descriptor_from_preflight(validated: ValidatedSizeFold | ValidatedChooserFold,
                              policy: SizeFoldPolicy | ChooserFoldPolicy,
                              manifest: StagedManifest) -> ServingFoldDescriptor | ChooserServingFoldDescriptor:
    """Describe staged folds only under the exact validated policy; no independent substitution."""
    if policy != validated.policy:
        raise ValueError("descriptor policy disagrees with validated fold")
    return _descriptor_from_validated(validated, manifest)


def describe_size_fold(data: bytes, name: str, policy: SizeFoldPolicy,
                       manifest: StagedManifest) -> ServingFoldDescriptor:
    """Validate authoring inputs without fitting; refuse with fixed messages."""
    return _descriptor_from_validated(preflight_size_fold(data, name, policy, manifest.release),
                                      manifest)
