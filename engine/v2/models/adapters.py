"""Prediction-only adapters for verified frozen artifacts."""
from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .contracts import ModelBinding


class AdapterError(ValueError):
    pass


class RuntimeFitForbidden(RuntimeError):
    pass


class InferenceAdapter(Protocol):
    def load(self, members: Mapping[str, bytes], binding: ModelBinding) -> object: ...

    def predict(
        self, artifact: object, rows: Sequence[Sequence[float]], binding: ModelBinding
    ) -> Sequence[Sequence[float]]: ...


class ReadOnlyArtifact:
    """Block fitting and mutation while preserving prediction methods."""

    __slots__ = ("__artifact",)
    _BLOCKED = frozenset(
        {"fit", "fit_predict", "fit_transform", "partial_fit", "set_params"}
    )

    def __init__(self, artifact: object) -> None:
        object.__setattr__(self, "_ReadOnlyArtifact__artifact", artifact)

    def __getattr__(self, name: str) -> Any:
        if name in self._BLOCKED:
            raise RuntimeFitForbidden(f"runtime model mutation is forbidden: {name}")
        return getattr(object.__getattribute__(self, "_ReadOnlyArtifact__artifact"), name)

    def __iter__(self):
        return iter(object.__getattribute__(self, "_ReadOnlyArtifact__artifact"))

    def __setattr__(self, name: str, value: object) -> None:
        raise RuntimeFitForbidden(f"runtime model mutation is forbidden: {name}")


@dataclass(frozen=True)
class _Linear:
    intercept: float
    coefficients: tuple[float, ...]


class JsonLinearAdapter:
    name = "json-linear.v1"

    def load(self, members: Mapping[str, bytes], binding: ModelBinding) -> object:
        raw = members.get("estimator")
        if raw is None:
            raise AdapterError("missing estimator member")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AdapterError("estimator is not valid JSON") from exc
        if document.get("schema_version") != "linear_estimator.v1.0":
            raise AdapterError("unsupported estimator schema")
        if tuple(document.get("feature_order", ())) != binding.feature_order:
            raise AdapterError("artifact feature order disagrees with binding")
        outputs = document.get("outputs", ())
        if len(outputs) != len(binding.output_names):
            raise AdapterError("artifact outputs disagree with binding")
        decoded = []
        for name, item in zip(binding.output_names, outputs, strict=True):
            if item.get("name") != name:
                raise AdapterError("artifact output names disagree with binding")
            coefficients = tuple(float(value) for value in item["coefficients"])
            if len(coefficients) != len(binding.feature_order):
                raise AdapterError("coefficient count disagrees with feature order")
            decoded.append(_Linear(float(item.get("intercept", 0.0)), coefficients))
        return tuple(decoded)

    def predict(self, artifact, rows, binding):
        answer = []
        for row in rows:
            if len(row) != len(binding.feature_order):
                raise AdapterError("row width disagrees with feature order")
            answer.append(tuple(
                output.intercept + sum(
                    weight * float(value)
                    for weight, value in zip(output.coefficients, row, strict=True)
                )
                for output in artifact
            ))
        return answer


class JoblibEstimatorAdapter:
    name = "joblib-estimator.v1"

    def load(self, members: Mapping[str, bytes], binding: ModelBinding) -> object:
        raw = members.get("estimator")
        if raw is None:
            raise AdapterError("missing estimator member")
        try:
            import joblib

            stored = joblib.load(io.BytesIO(raw))
        except Exception as exc:
            raise AdapterError("joblib estimator could not be decoded") from exc
        if tuple(getattr(stored, "features", binding.feature_order)) != binding.feature_order:
            raise AdapterError("artifact feature order disagrees with binding")
        return getattr(stored, "model", stored)

    def predict(self, artifact, rows, binding):
        try:
            values = artifact.predict(rows)
        except RuntimeFitForbidden:
            raise
        except Exception as exc:
            raise AdapterError("estimator prediction failed") from exc
        values = values.tolist() if hasattr(values, "tolist") else list(values)
        if len(binding.output_names) == 1:
            return [
                (float(value[0]) if isinstance(value, (list, tuple)) else float(value),)
                for value in values
            ]
        return [tuple(float(item) for item in value) for value in values]


def default_adapters() -> dict[str, InferenceAdapter]:
    adapters: tuple[InferenceAdapter, ...] = (
        JsonLinearAdapter(),
        JoblibEstimatorAdapter(),
    )
    return {adapter.name: adapter for adapter in adapters}
