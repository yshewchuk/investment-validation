"""Validation helpers for ``tools/phase5_serving_folds.py`` (split out to keep both modules small).

Pure checks over a serving-fold policy and its stored joblib header; every refusal is a
``ValueError`` with a fixed message, exactly as the caller expects.
"""
from __future__ import annotations

import io
import math
from datetime import date

from engine.v2.models.serving_folds import FOLD_PANEL_SHA256_PATTERN


def _valid_feature_order(order) -> bool:
    return (bool(order) and all(isinstance(key, str) and key for key in order)
            and len(set(order)) == len(order))


def _valid_interval_floor(floor) -> bool:
    return floor is None or (type(floor) in (int, float) and math.isfinite(floor))


def check_size_policy(policy, interval_policy: str) -> None:
    """Refuse a size policy with a bad panel hash, feature order, interval policy or floor."""
    if not (isinstance(policy.panel_sha256, str)
            and FOLD_PANEL_SHA256_PATTERN.fullmatch(policy.panel_sha256)
            and _valid_feature_order(policy.feature_order)
            and policy.interval_policy == interval_policy
            and _valid_interval_floor(policy.interval_floor)):
        raise ValueError("invalid size serving policy")


def load_stored_fold(data: bytes, policy) -> dict:
    """Decode the joblib bytes and check the header against the policy."""
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
    return stored


def fold_month(stored: dict) -> date:
    """The fold's exact month start."""
    try:
        fold = date.fromisoformat(stored["fold_start"])
        if fold.day != 1 or fold.isoformat() != stored["fold_start"]:
            raise ValueError("not an exact month start")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid serving fold month") from exc
    return fold


def held_out_pools(stored: dict) -> list:
    """The paired held-out prediction/residual arrays: finite, 1-D, numeric, equal shape."""
    try:
        import numpy as np

        pools = [np.asarray(stored[key]) for key in ("pool_pred", "pool_res")]
        if (any(pool.ndim != 1 or pool.dtype.kind not in "fiu"
                or not np.isfinite(pool).all() for pool in pools)
                or pools[0].shape != pools[1].shape):
            raise ValueError("invalid paired arrays")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid serving fold held-out pool") from exc
    return pools
