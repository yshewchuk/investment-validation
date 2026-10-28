"""Immutable role-specific serving-fold metadata; no loader in this module."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from engine.v2.foundation import DocumentError, content_hash, to_document
from engine.v2.models.contracts import ArtifactMember

FOLD_PANEL_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


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
        if self.role not in ("implied_t1", "runup_move"):
            raise DocumentError("BAD_CHOOSER_ROLE", "$.role", "unsupported chooser fold role")
        if (not isinstance(self.feature_order, (list, tuple)) or not self.feature_order
                or any(not isinstance(key, str) or not key for key in self.feature_order)
                or len(set(self.feature_order)) != len(self.feature_order)):
            raise DocumentError("BAD_FEATURE_ORDER", "$.feature_order", "expected nonempty unique feature names")
        object.__setattr__(self, "feature_order", tuple(self.feature_order))
        if not isinstance(self.panel_sha256, str) or not FOLD_PANEL_SHA256_PATTERN.fullmatch(self.panel_sha256):
            raise DocumentError("BAD_PANEL_HASH", "$.panel_sha256", "expected 64 lowercase hex digits")
        if self.interval_policy != "chooser-heldout-deciles.v1":
            raise DocumentError("BAD_INTERVAL_POLICY", "$.interval_policy", "unsupported chooser interval policy")
        try:
            finite_floor = (self.interval_floor is None or type(self.interval_floor) in (int, float)
                            and math.isfinite(self.interval_floor))
        except OverflowError:
            finite_floor = False
        if not finite_floor:
            raise DocumentError("BAD_INTERVAL_FLOOR", "$.interval_floor", "expected a finite number or null")


def _is_canonical_date(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return parsed.isoformat() == value


def _validate_optional_training_cutoff(value: object) -> None:
    if value is not None and not _is_canonical_date(value):
        raise DocumentError("BAD_TRAINING_CUTOFF", "$.training_cutoff", "expected canonical ISO date")


@dataclass(frozen=True, kw_only=True)
class ChooserServingFoldDescriptor:
    """Role-specific declaration; loaded only by the chooser artifact loader.

    ``training_cutoff`` is the producer-declared upper training day. The runtime
    consumer requires it to be canonical and strictly before the decision session.
    The date alone does not prove source finality or training provenance.
    """

    parent_release_id: str
    parent_release_hash: str
    policy: ChooserFoldPolicy
    fold_start: str
    training_cutoff: str | None = None
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
        try:
            month = date.fromisoformat(self.fold_start)
            valid_month = month.day == 1 and month.isoformat() == self.fold_start
        except (TypeError, ValueError):
            valid_month = False
        if not valid_month:
            raise DocumentError("BAD_FOLD_MONTH", "$.fold_start", "expected canonical ISO month start")
        _validate_optional_training_cutoff(self.training_cutoff)
        if type(self.pool_count) is not int or self.pool_count < 0:
            raise DocumentError("BAD_POOL_COUNT", "$.pool_count", "expected a nonnegative integer")
        if (not isinstance(self.parent_release_hash, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", self.parent_release_hash)):
            raise DocumentError("BAD_PARENT_HASH", "$.parent_release_hash", "expected canonical SHA-256 content hash")
        if not isinstance(self.decision_clock_id, str) or not self.decision_clock_id:
            raise DocumentError("BAD_FOLD_CLOCK", "$.decision_clock_id", "expected a nonempty clock identity")

    @property
    def descriptor_hash(self) -> str:
        return content_hash(to_document(self))
