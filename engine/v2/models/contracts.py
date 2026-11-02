"""Immutable contracts for frozen model releases and inference."""
from __future__ import annotations

from dataclasses import dataclass

from .chooser_cutoff import canonical_day

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


def validated_training_cutoff(value: object, fold_start: str | None = None) -> str | None:
    """Legacy absence stays ``None``; anything else must be one canonical day.

    Parsing is delegated to :func:`chooser_cutoff.canonical_day`, and the value
    is then required to already be exactly the immutable string it validated to,
    so a date, a datetime, a timestamp-like string or any mutable value is
    refused rather than silently rewritten. When ``fold_start`` is supplied it
    must also be one canonical day and the cutoff must fall strictly before it.
    """
    if value is None:
        return None
    day = canonical_day(value)
    if not isinstance(value, str) or day != value:
        raise ValueError("invalid serving fold training cutoff")
    if fold_start is not None:
        fold = canonical_day(fold_start)
        if not isinstance(fold_start, str) or fold != fold_start:
            raise ValueError("invalid serving fold start")
        if day >= fold:
            raise ValueError("serving fold training cutoff is not before its fold")
    return day
