"""Typed execution of one frozen-model scoring stage."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

from engine.v2.foundation import untag_nonfinite
from engine.v2.models.contracts import (
    MODEL_READY,
    InferenceRequest,
    InferenceResult,
    ModelBinding,
    ModelRelease,
)
from engine.v2.models.loader import FrozenInference

__all__ = [
    "FrozenRecipeExecutor",
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

        # Legacy semantics (engine/score.py ``_score_model``/``_score_gate``):
        # ``features[...].to_numpy(dtype=float)`` turns a missing value (None
        # or NaN) into NaN, and ``not np.isfinite(X).all()`` then flags
        # MISSING_FEATURES naming EVERY non-finite column -- NaN and +/-inf
        # alike -- and declines. So a non-finite value is a missing feature,
        # not an invalid one (R4-20 gap 4). Only a value that is not a number
        # at all (legacy's float conversion would raise) is INVALID_FEATURE.
        row: list[float] = []
        for name in binding.feature_order:
            raw = untag_nonfinite(features[name])
            try:
                value = float("nan") if raw is None else float(raw)
            except (TypeError, ValueError) as exc:
                raise FrozenStageRefusal(
                    "INVALID_FEATURE",
                    f"frozen model feature is not numeric: {name}",
                    reason_codes=("INVALID_FEATURE",),
                ) from exc
            row.append(value)
        nonfinite = tuple(
            name for name, value in zip(binding.feature_order, row)
            if not isfinite(value)
        )
        if nonfinite:
            raise FrozenStageRefusal(
                "MISSING_FEATURES",
                f"non-finite frozen model features: {', '.join(nonfinite)}",
                reason_codes=("MISSING_FEATURES",),
                missing_features=nonfinite,
            )
        return tuple(row)

    def _verify(
        self,
        binding: ModelBinding,
        result: InferenceResult,
    ) -> tuple[tuple[str, ...], tuple[float, ...]]:
        if result.status != MODEL_READY:
            reasons = tuple(result.reason_codes) or ("MODEL_NOT_READY",)
            detail = getattr(result, "detail", None) or ", ".join(reasons)
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
        role_rows = features.get("role_model_inputs")
        if isinstance(role_rows, Mapping):
            role = str(binding.role)
            vector = role_rows.get(role, role_rows.get(role.split(":", 1)[0]))
            if isinstance(vector, Mapping):
                stage_facts = features.get("_native_stage_facts", {})
                stage_facts = (stage_facts if isinstance(stage_facts, Mapping)
                               else {})
                features = {**vector, **stage_facts}
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

    def predict(self, features: Mapping[str, float]) -> Mapping[str, float]:
        """Return named outputs for the native stage executor contract."""
        return self.execute(features).outputs


class FrozenRecipeExecutor:
    """One source-declared forecast/gate recipe executed by frozen inference.

    ``SourceBundle`` recipes name a release binding (``binding_id``) instead
    of carrying inline coefficients (R4-16). This maps that binding's
    ``source`` output onto the native stage field ``target``, through
    :class:`FrozenStageExecutor` -- verified members, the registered adapter,
    no fitting. A declared recipe whose release or inference was not supplied
    is kept as an unresolved executor that refuses ``MODEL_NOT_READY`` when
    the stage runs, never a silent fallback to another model.

    ``str()`` is the recipe identity used by stage receipts: release, binding,
    model, member hashes and the output mapping -- never an object address.
    """

    def __init__(
        self,
        *,
        target: str,
        binding_id: str,
        source: str | None,
        executor: FrozenStageExecutor | None,
        binding: ModelBinding | None,
    ) -> None:
        self._target = str(target)
        self._binding_id = str(binding_id)
        self._source = source
        self._executor = executor
        self._binding = binding

    @property
    def target(self) -> str:
        return self._target

    @property
    def binding_id(self) -> str:
        return self._binding_id

    @property
    def artifact_hashes(self) -> tuple[str, ...]:
        if self._binding is None:
            return ()
        return tuple(member.content_hash for member in self._binding.members)

    @property
    def adapter(self) -> str | None:
        """The resolved binding's adapter name, ``None`` when unresolved."""
        return None if self._binding is None else self._binding.adapter

    @property
    def feature_order(self) -> tuple[str, ...]:
        """The resolved binding's feature order, ``()`` when unresolved."""
        return () if self._binding is None else tuple(self._binding.feature_order)

    @property
    def role(self) -> str | None:
        """The resolved served role, ``None`` when the binding is unresolved."""
        return None if self._binding is None else str(self._binding.role)

    def __str__(self) -> str:
        if self._executor is None or self._binding is None:
            return f"frozen-recipe:unresolved:{self._binding_id}->{self._target}"
        return (
            f"frozen-recipe:{self._executor.release.release_id}:{self._binding_id}:"
            f"{self._binding.model_id}:{self._source}->{self._target}:"
            f"{','.join(self.artifact_hashes)}"
        )

    __repr__ = __str__

    def predict(self, features: Mapping[str, float]) -> Mapping[str, float]:
        if self._executor is None or self._source is None:
            raise FrozenStageRefusal(
                "MODEL_NOT_READY",
                f"frozen recipe binding is not resolved: {self._binding_id}",
                reason_codes=("MODEL_NOT_READY",),
            )
        try:
            outputs = self._executor.predict(features)
        except FrozenStageRefusal as exc:
            if exc.code != "MODEL_NOT_READY" or "MODEL_NOT_READY" in exc.reason_codes:
                raise
            # An unservable model says so by name (P5-2), ahead of the
            # inference detail (BINDING_NOT_FOUND, ARTIFACT_INVALID, ...).
            raise FrozenStageRefusal(
                exc.code, exc.detail,
                reason_codes=("MODEL_NOT_READY", *exc.reason_codes),
                missing_features=exc.missing_features,
            ) from exc
        return {self._target: float(outputs[self._source])}
