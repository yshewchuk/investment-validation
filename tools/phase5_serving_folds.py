"""Offline size-fold declaration from bounded cache bytes and registered policy.

The descriptor is preparation metadata, not a runtime loader or inference view.
Legacy cache decoding stays outside the native model package.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
from datetime import date

from engine.v2.models import ArtifactMember
from engine.v2.models.deployment import StagedManifest
from engine.v2.models.serving_folds import ServingFoldDescriptor, SizeFoldPolicy

MAX_FOLD_BYTES = 32 * 1024 * 1024


def describe_size_fold(data: bytes, name: str, policy: SizeFoldPolicy,
                       manifest: StagedManifest) -> ServingFoldDescriptor:
    """Validate authoring inputs without fitting; refuse with fixed messages."""
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
    bindings = [b for b in manifest.release.bindings if b.role == "size" and b.strategy_id == "*"]
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
    digest = hashlib.sha256(data).hexdigest()
    return ServingFoldDescriptor(
        parent_release_id=manifest.release.release_id, parent_release_hash=manifest.release_hash,
        policy=policy, fold_start=fold.isoformat(), decision_clock_id=bindings[0].decision_clock_id,
        estimator=ArtifactMember(name="estimator", path=f"objects/{digest}",
                                 content_hash=f"sha256:{digest}"), pool_count=len(pools[0]))
