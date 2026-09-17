"""Immutable contracts for frozen model releases and inference."""
from __future__ import annotations

from dataclasses import dataclass

MODEL_READY = "READY"
MODEL_NOT_READY = "MODEL_NOT_READY"


@dataclass(frozen=True, kw_only=True)
class ArtifactMember:
    name: str
    path: str
    content_hash: str
    schema_version: str = "artifact_member.v1.0"


@dataclass(frozen=True, kw_only=True)
class ModelBinding:
    binding_id: str
    model_id: str
    role: str
    strategy_id: str
    decision_clock_id: str
    adapter: str
    feature_order: tuple[str, ...]
    output_names: tuple[str, ...]
    members: tuple[ArtifactMember, ...]
    schema_version: str = "model_binding.v1.0"


@dataclass(frozen=True, kw_only=True)
class ModelRelease:
    release_id: str
    deployment_id: str
    bindings: tuple[ModelBinding, ...]
    schema_version: str = "model_release.v1.0"


@dataclass(frozen=True, kw_only=True)
class InferenceRequest:
    release_id: str
    binding_id: str
    feature_order: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    schema_version: str = "inference_request.v1.0"


@dataclass(frozen=True, kw_only=True)
class InferenceResult:
    status: str
    release_id: str
    binding_id: str
    model_id: str | None = None
    artifact_hashes: tuple[str, ...] = ()
    output_names: tuple[str, ...] = ()
    predictions: tuple[tuple[float, ...], ...] = ()
    reason_codes: tuple[str, ...] = ()
    detail: str | None = None
    schema_version: str = "inference_result.v1.0"


@dataclass(frozen=True, kw_only=True)
class PredictionFrame:
    """Immutable prediction values plus the verified model lineage."""

    model_id: str
    release_id: str
    binding_id: str
    ordered_outputs: tuple[str, ...]
    rows: tuple[tuple[float, ...], ...]
    artifact_hashes: tuple[str, ...]
    schema_version: str = "prediction_frame.v1.0"
