"""Immutable metadata for release-owned size serving folds; no runtime loading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from engine.v2.foundation import content_hash
from engine.v2.models.contracts import ArtifactMember


@dataclass(frozen=True, kw_only=True)
class SizeFoldPolicy:
    """Explicit registered producer metadata, never inferred from cache omissions."""

    model_id: str
    feature_order: tuple[str, ...]
    panel_sha256: str
    interval_floor: float | None
    interval_policy: Literal["size-heldout-deciles.v1"] = "size-heldout-deciles.v1"
    schema_version: str = "size_fold_policy.v1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_order", tuple(self.feature_order))


@dataclass(frozen=True, kw_only=True)
class ServingFoldDescriptor:
    """Catalog-hashed declaration; embedded pools share the estimator byte identity."""

    parent_release_id: str
    parent_release_hash: str
    policy: SizeFoldPolicy
    fold_start: str
    decision_clock_id: str
    estimator: ArtifactMember
    pool_count: int
    role: Literal["size"] = "size"
    output_name: Literal["forecast_abs_move"] = "forecast_abs_move"
    pool_pred_field: Literal["pool_pred"] = "pool_pred"
    pool_res_field: Literal["pool_res"] = "pool_res"
    schema_version: str = "serving_fold_descriptor.v1.0"

    @property
    def descriptor_hash(self) -> str:
        return content_hash(self)
