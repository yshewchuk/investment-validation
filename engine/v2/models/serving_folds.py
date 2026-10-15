"""Immutable role-specific serving-fold metadata; no runtime loading."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from engine.v2.foundation import DocumentError, content_hash, to_document
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


def chooser_fold_outputs(role: str) -> tuple[str, str]:
    """Exact champion output and raw D14 chooser column for the declared role."""
    if role == "implied_t1":
        return "driver_prediction", "pred_im_t1_d14"
    if role == "runup_move":
        return "pred_runup_abs_move_d14", "pred_runup_abs_move_d14"
    raise ValueError("unsupported chooser fold role")


@dataclass(frozen=True, kw_only=True)
class ChooserFoldPolicy:
    """Registered producer metadata, distinct from the size-only policy."""

    role: Literal["implied_t1", "runup_move"]
    model_id: str
    feature_order: tuple[str, ...]
    panel_sha256: str
    interval_floor: float | None
    interval_policy: Literal["chooser-heldout-deciles.v1"] = "chooser-heldout-deciles.v1"
    schema_version: str = "chooser_fold_policy.v1.0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        if not isinstance(self.panel_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.panel_sha256):
            raise DocumentError("BAD_PANEL_HASH", "$.panel_sha256", "expected 64 lowercase hex digits")


@dataclass(frozen=True, kw_only=True)
class ChooserServingFoldDescriptor:
    """Offline declaration only; not accepted by the existing size-fold loader."""

    parent_release_id: str
    parent_release_hash: str
    policy: ChooserFoldPolicy
    fold_start: str
    decision_clock_id: str
    estimator: ArtifactMember
    pool_count: int
    output_name: Literal["driver_prediction", "pred_runup_abs_move_d14"]
    producer_output: Literal["pred_im_t1_d14", "pred_runup_abs_move_d14"]
    pool_pred_field: Literal["pool_pred"] = "pool_pred"
    pool_res_field: Literal["pool_res"] = "pool_res"
    schema_version: str = "chooser_serving_fold_descriptor.v1.0"

    def __post_init__(self) -> None:
        if (self.output_name, self.producer_output) != chooser_fold_outputs(self.policy.role):
            raise DocumentError("CHOOSER_OUTPUT_MISMATCH", "$", "outputs disagree with chooser role")
        digest = self.estimator.content_hash
        if (not isinstance(digest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
                or self.estimator.path != f"objects/{digest.removeprefix('sha256:')}"):
            raise DocumentError("BAD_FOLD_OBJECT", "$.estimator", "expected matching content-addressed cache object")

    @property
    def descriptor_hash(self) -> str:
        return content_hash(to_document(self))
