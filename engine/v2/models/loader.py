"""Verified, cache-independent loading and read-only inference."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

from .adapters import AdapterError, InferenceAdapter, ReadOnlyArtifact, default_adapters
from .contracts import (
    MODEL_NOT_READY,
    MODEL_READY,
    InferenceRequest,
    InferenceResult,
    ModelRelease,
)


class FrozenInference:
    def __init__(
        self,
        artifact_root: Path,
        *,
        adapters: Mapping[str, InferenceAdapter] | None = None,
    ) -> None:
        self._root = Path(artifact_root).resolve()
        self._adapters = dict(default_adapters() if adapters is None else adapters)
        self._cache: dict[tuple[object, ...], ReadOnlyArtifact] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def infer(self, release: ModelRelease, request: InferenceRequest) -> InferenceResult:
        binding = self._resolve(release, request)
        if isinstance(binding, InferenceResult):
            return binding
        if request.feature_order != binding.feature_order:
            return self._refuse(request, "INCOMPATIBLE_FEATURE_ORDER", binding)
        adapter = self._adapters.get(binding.adapter)
        if adapter is None:
            return self._refuse(request, "UNKNOWN_ADAPTER", binding)
        try:
            members = self._verified_members(binding)
            key = (
                binding.adapter,
                binding.feature_order,
                binding.output_names,
                tuple((item.name, item.content_hash) for item in binding.members),
            )
            artifact = self._cache.get(key)
            if artifact is None:
                artifact = ReadOnlyArtifact(adapter.load(members, binding))
                self._cache[key] = artifact
            predictions = tuple(
                tuple(float(value) for value in row)
                for row in adapter.predict(artifact, request.rows, binding)
            )
            if len(predictions) != len(request.rows):
                raise AdapterError("prediction row count disagrees with request")
            if any(len(row) != len(binding.output_names) for row in predictions):
                raise AdapterError("prediction width disagrees with outputs")
        except (AdapterError, OSError, ValueError) as exc:
            return self._refuse(request, "ARTIFACT_INVALID", binding, str(exc))
        return InferenceResult(
            status=MODEL_READY,
            release_id=release.release_id,
            binding_id=binding.binding_id,
            model_id=binding.model_id,
            artifact_hashes=tuple(item.content_hash for item in binding.members),
            output_names=binding.output_names,
            predictions=predictions,
        )

    def _resolve(self, release, request):
        if request.release_id != release.release_id:
            return self._refuse(request, "RELEASE_MISMATCH")
        matches = [item for item in release.bindings if item.binding_id == request.binding_id]
        if len(matches) != 1:
            reason = "BINDING_NOT_FOUND" if not matches else "DUPLICATE_BINDING"
            return self._refuse(request, reason)
        return matches[0]

    def _verified_members(self, binding):
        names = [item.name for item in binding.members]
        if not names or len(names) != len(set(names)):
            raise AdapterError("artifact member names must be unique and non-empty")
        verified = {}
        for member in binding.members:
            path = (self._root / member.path).resolve()
            try:
                path.relative_to(self._root)
            except ValueError as exc:
                raise AdapterError("artifact path escapes artifact root") from exc
            if not path.is_file():
                raise FileNotFoundError(f"missing artifact member: {member.name}")
            payload = path.read_bytes()
            if "sha256:" + hashlib.sha256(payload).hexdigest() != member.content_hash:
                raise AdapterError(f"artifact hash mismatch: {member.name}")
            verified[member.name] = payload
        return verified

    @staticmethod
    def _refuse(request, reason, binding=None, detail=None):
        return InferenceResult(
            status=MODEL_NOT_READY,
            release_id=request.release_id,
            binding_id=request.binding_id,
            model_id=None if binding is None else binding.model_id,
            artifact_hashes=() if binding is None else tuple(
                item.content_hash for item in binding.members
            ),
            output_names=() if binding is None else binding.output_names,
            reason_codes=(reason,),
            detail=detail,
        )
