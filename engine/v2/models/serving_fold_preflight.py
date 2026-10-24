"""Offline role-specific fold declarations from bounded bytes and registered policy.

The descriptor is preparation metadata, not a runtime loader or inference view.
The legacy joblib cache is decoded only inside the same-package checks (``serving_fold_checks``). Inputs must be
trusted uncompressed output from the serving-fold writers; byte caps and hashes
do not establish provenance or bound decoded memory.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from engine.v2.foundation import from_document, to_document
from engine.v2.models import ArtifactMember, ModelRelease
from engine.v2.models.deployment import StagedManifest, semantic_release_hash
from engine.v2.models.serving_fold_checks import (
    check_size_policy,
    fold_month,
    held_out_pools,
    load_stored_fold,
)
from engine.v2.models.serving_folds import (
    ChooserFoldPolicy,
    ChooserServingFoldDescriptor,
    ServingFoldDescriptor,
    SizeFoldPolicy,
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


def _release_binding(release, policy, role, output):
    bindings = [b for b in release.bindings if b.role == role and b.strategy_id == "*"]
    if (len(bindings) != 1 or bindings[0].model_id != policy.model_id
            or bindings[0].feature_order != policy.feature_order
            or bindings[0].output_names != (output,)):
        raise ValueError(f"{role} serving policy disagrees with release binding")
    return bindings[0]


def _preflight_fold(data, name, policy, release, *, role, output, interval_policy, result_type):
    if len(data) > MAX_FOLD_BYTES:
        raise ValueError("serving fold exceeds byte limit")
    if role == "size":
        check_size_policy(policy, interval_policy)
    binding = _release_binding(release, policy, role, output)
    stored = load_stored_fold(data, policy)
    fold = fold_month(stored)
    if name != f"{policy.model_id}_{fold:%Y%m}_{policy.panel_sha256[:12]}.joblib":
        raise ValueError("serving fold filename disagrees with header")
    pools = held_out_pools(stored)
    return result_type(fold_start=fold.isoformat(),
                       decision_clock_id=binding.decision_clock_id,
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
