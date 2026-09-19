"""Verified loading and release pinning for P5-4 frozen serving state.

One loader for the frozen non-model states this package defines -- the
driver and paired residual pools (:mod:`.residual_artifact`), the
admissible-depth calibration table (:mod:`.admissible_table`) and the board
analog-matcher population (:mod:`.analog_artifact`) -- with the
same verification shape as ``PayoffArtifactLoader`` and ``FrozenInference``:
re-hash the raw bytes against the reference, rebuild the record, and refuse
unless the record's own recomputed hash agrees too.

:func:`serialize_frozen_state` writes the canonical JSON of the payload, so
``sha256(file bytes) == artifact.content_hash`` by construction. That is what
lets a release pin one of these like any model member:
``frozen_release.release_member``/``inventory_member`` give the members.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from engine.v2.foundation.canonical import CONTENT_HASH_PREFIX, canonical_json
from engine.v2.models.frozen_documents import FrozenState, frozen_state_from_document

__all__ = [
    "FrozenState",
    "FrozenStateError",
    "FrozenStateLoader",
    "FrozenStateRef",
    "serialize_frozen_state",
]

class FrozenStateError(ValueError):
    """A frozen state could not be verified or loaded."""


@dataclass(frozen=True, kw_only=True)
class FrozenStateRef:
    path: str
    content_hash: str
    schema_version: str = "frozen_state_ref.v1.0"


def serialize_frozen_state(state: FrozenState) -> bytes:
    """The exact bytes to write: their sha256 IS ``state.content_hash``."""
    return canonical_json(state.payload()).encode("utf-8")


def _from_document(document: Mapping[str, Any]) -> FrozenState:
    return frozen_state_from_document(document)


class FrozenStateLoader:
    """Hash-verified, read-only loading, cached by content hash."""

    def __init__(self, artifact_root: Path | str) -> None:
        self._root = Path(artifact_root).resolve()
        self._cache: dict[str, FrozenState] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def _read(self, ref: FrozenStateRef) -> bytes:
        path = (self._root / ref.path).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise FrozenStateError("artifact path escapes artifact root") from exc
        try:
            return path.read_bytes()
        except OSError as exc:
            raise FrozenStateError(f"missing frozen state: {ref.path}") from exc

    def load(self, ref: FrozenStateRef) -> FrozenState:
        cached = self._cache.get(ref.content_hash)
        if cached is not None:
            return cached
        raw = self._read(ref)
        if CONTENT_HASH_PREFIX + hashlib.sha256(raw).hexdigest() != ref.content_hash:
            raise FrozenStateError(f"frozen state hash mismatch: {ref.path}")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FrozenStateError("frozen state is not valid JSON") from exc
        if not isinstance(document, Mapping):
            raise FrozenStateError("frozen state document must be a JSON object")
        try:
            state = _from_document(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise FrozenStateError(f"frozen state is malformed: {ref.path}") from exc
        if state.content_hash != ref.content_hash:
            raise FrozenStateError(f"frozen state disagrees with its own hash: {ref.path}")
        self._cache[ref.content_hash] = state
        return state
