"""Typed execution of one frozen-model scoring stage."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from engine.v2.models.contracts import (
    MODEL_READY,
    InferenceRequest,
    InferenceResult,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.loader import FrozenInference

__all__ = [
    "FrozenStageExecutor",
    "FrozenStageRefusal",
    "FrozenStageResult",
]


class FrozenStageRefusal(ValueError):
    """Explicit refusal to execute or trust a frozen scoring stage."""

    def __init__(
        self,
        code: str,
        detail: str,
        *,
        reason_codes: tuple[str, ...] = (),
        missing_features: tuple[str, ...] = (),
    ) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.reason_codes = reason_codes
        self.missing_features = missing_features


@dataclass(frozen=True, kw_only=True)
class FrozenStageResult:
    """One named prediction row and its verified frozen-artifact lineage."""

    request: InferenceRequest
    model_id: str
    outputs: Mapping[str, float]
    artifact_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outputs",
            MappingProxyType(dict(self.outputs)),
        )


@dataclass(frozen=True, kw_only=True)
class FrozenStageExecutor:
    """Map runtime features into one verified frozen inference request."""

    inference: FrozenInference
    release: ModelRelease
    binding_id: str

    def _binding(self) -> ModelBinding:
        matches = tuple(
            binding
            for binding in self.release.bindings
            if binding.binding_id == self.binding_id
        )
        if len(matches) != 1:
            reason = "BINDING_NOT_FOUND" if not matches else "DUPLICATE_BINDING"
            raise FrozenStageRefusal(
                "MODEL_NOT_READY",
                f"frozen binding cannot be resolved: {self.binding_id}",
                reason_codes=(reason,),
            )
        return matches[0]

    @staticmethod
    def _row(
        binding: ModelBinding,
        features: Mapping[str, float],
    ) -> tuple[float, ...]:
        missing = tuple(
            name for name in binding.feature_order if name not in features
        )
        if missing:
            raise FrozenStageRefusal(
                "MISSING_FEATURES",
                f"missing frozen model features: {', '.join(missing)}",
                reason_codes=("MISSING_FEATURES",),
                missing_features=missing,
            )

        row: list[float] = []
        for name in binding.feature_order:
            try:
                value = float(features[name])
            except (TypeError, ValueError) as exc:
                raise FrozenStageRefusal(
                    "INVALID_FEATURE",
                    f"frozen model feature is not numeric: {name}",
                    reason_codes=("INVALID_FEATURE",),
                ) from exc
            if not isfinite(value):
                raise FrozenStageRefusal(
                    "INVALID_FEATURE",
                    f"frozen model feature is not finite: {name}",
                    reason_codes=("INVALID_FEATURE",),
                )
            row.append(value)
        return tuple(row)

    def _verify(
        self,
        binding: ModelBinding,
        result: InferenceResult,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if result.status != MODEL_READY:
            reasons = tuple(result.reason_codes) or ("MODEL_NOT_READY",)
            detail = result.detail or ", ".join(reasons)
            raise FrozenStageRefusal(
                "MODEL_NOT_READY",
                detail,
                reason_codes=reasons,
            )

        expected_hashes = tuple(member.content_hash for member in binding.members)
        identity = (
            result.release_id,
            result.binding_id,
            result.model_id,
            tuple(result.artifact_hashes),
        )
        expected_identity = (
            self.release.release_id,
            binding.binding_id,
            binding.model_id,
            expected_hashes,
        )
        if identity != expected_identity:
            raise FrozenStageRefusal(
                "ARTIFACT_PROVENANCE_MISMATCH",
                "inference lineage disagrees with the selected frozen binding",
                reason_codes=("ARTIFACT_PROVENANCE_MISMATCH",),
            )

        output_names = tuple(result.output_names)
        predictions = tuple(result.predictions)
        valid_shape = (
            output_names == binding.output_names
            and len(output_names) == len(set(output_names))
            and len(predictions) == 1
            and len(predictions[0]) == len(output_names)
        )
        if not valid_shape:
            raise FrozenStageRefusal(
                "OUTPUT_CONTRACT_MISMATCH",
                "inference output shape or names disagree with the frozen binding",
                reason_codes=("OUTPUT_CONTRACT_MISMATCH",),
            )
        return expected_hashes, tuple(predictions[0])

    def execute(self, features: Mapping[str, float]) -> FrozenStageResult:
        binding = self._binding()
        row = self._row(binding, features)
        request = InferenceRequest(
            release_id=self.release.release_id,
            binding_id=binding.binding_id,
            feature_order=binding.feature_order,
            rows=(row,),
        )
        result = self.inference.infer(self.release, request)
        artifact_hashes, values = self._verify(binding, result)
        return FrozenStageResult(
            request=request,
            model_id=binding.model_id,
            outputs={
                name: float(value)
                for name, value in zip(binding.output_names, values, strict=True)
            },
            artifact_hashes=artifact_hashes,
        )
