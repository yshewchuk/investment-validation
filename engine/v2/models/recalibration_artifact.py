"""Versioned, content-hashed win-rate recalibration-map artifacts (P5-4).

Legacy ``engine.recalibrate.fit_recalibration`` fits an ``IsotonicRegression``
``raw_win -> realized outcome`` map live inside every Scorer (P5-1
``NON_MODEL_STATE_ITEMS``), and ``Scorer._score_model`` pushes the STR-THRU
raw model win rate through it. This module is the frozen artifact TYPE that
map belongs in instead, copying :mod:`engine.v2.models.payoff_artifact`'s
pattern: an immutable record of the map and its provenance, a content hash
over its canonical JSON, and a verified read-only loader.

One artifact is bound to ``(strategy, alpha, cutoff)`` -- the identity
legacy's own ``Scorer.recalibration`` cache keys on (``round(alpha, 4)``,
the normalized ``before`` date) and the same one
:func:`engine.v2.models.payoff_artifact.payoff_artifact_key` builds, so a
request's resolved fill alpha and evidence cutoff look up both artifacts the
same way.

**Two states, both frozen.** Legacy returns ``None`` -- and ships the raw
probability unchanged -- when the pairs table is empty or fewer than
``min_pairs`` pairs had closed before the cutoff. That is a real answer, not
a missing model, so it is frozen too: ``fitted=False``, no thresholds, and
:meth:`RecalibrationMapArtifact.transform` returns its input. A MISSING
artifact is a different thing (the release never built this fold) and
scoring refuses it with MODEL_NOT_READY; it never falls back to raw.

Layering (``checks/layer_map.py``): layer 3, so this module holds no fitting
code. The builder that fits the map lives at
``engine/v2/models/training/recalibration.py`` (layer 6); scoring (layer 5)
reads this module only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from engine.v2.foundation.canonical import (
    CONTENT_HASH_PREFIX,
    canonical_json,
    content_hash,
    untag_nonfinite,
)

from .payoff_artifact import PayoffArtifactKey, payoff_artifact_key

__all__ = [
    "RECALIBRATION_MAP_ARTIFACT_V1",
    "RecalibrationArtifactError",
    "RecalibrationArtifactLoader",
    "RecalibrationArtifactRef",
    "RecalibrationMapArtifact",
    "make_recalibration_map_artifact",
    "recalibration_artifact_key",
    "serialize_recalibration_artifact",
]

RECALIBRATION_MAP_ARTIFACT_V1 = "recalibration_map_artifact.v1.0"


def recalibration_artifact_key(strategy: str, alpha: float, cutoff: Any) -> PayoffArtifactKey:
    """``(strategy, alpha at 4dp, cutoff ISO date or None)`` -- payoff's own key."""
    return payoff_artifact_key(strategy, alpha, cutoff)


def _floats(values: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _payload(
    *, strategy: str, alpha: float, cutoff: Any, min_pairs: int, fitted: bool,
    n: int | None, base_rate: float | None, x_thresholds: Sequence[float],
    y_thresholds: Sequence[float], window: tuple[Any, Any] | None,
) -> dict[str, Any]:
    key = recalibration_artifact_key(strategy, alpha, cutoff)
    window_start, window_end = window if window is not None else (None, None)
    x, y = list(_floats(x_thresholds)), list(_floats(y_thresholds))
    if len(x) != len(y):
        raise RecalibrationArtifactError("x_thresholds and y_thresholds differ in length")
    if not fitted and (x or n is not None or base_rate is not None):
        raise RecalibrationArtifactError("an unfitted map carries no thresholds, n or base_rate")
    if fitted and n is None:
        raise RecalibrationArtifactError("a fitted map records its pair count")
    return {
        "schema_version": RECALIBRATION_MAP_ARTIFACT_V1,
        "strategy": key[0],
        "alpha": key[1],
        "cutoff": key[2],
        "min_pairs": int(min_pairs),
        "fitted": bool(fitted),
        "n": None if n is None else int(n),
        "base_rate": None if base_rate is None else float(base_rate),
        "x_thresholds": x,
        "y_thresholds": y,
        "window_start": None if window_start is None else str(window_start)[:10],
        "window_end": None if window_end is None else str(window_end)[:10],
    }


@dataclass(frozen=True, kw_only=True)
class RecalibrationMapArtifact:
    """A frozen ``raw_win -> calibrated_win`` monotone map, or a frozen "no map"."""

    schema_version: str = RECALIBRATION_MAP_ARTIFACT_V1
    strategy: str
    alpha: float
    cutoff: str | None
    min_pairs: int
    fitted: bool
    n: int | None
    base_rate: float | None
    x_thresholds: tuple[float, ...]
    y_thresholds: tuple[float, ...]
    window_start: str | None
    window_end: str | None
    content_hash: str

    @property
    def key(self) -> PayoffArtifactKey:
        return recalibration_artifact_key(self.strategy, self.alpha, self.cutoff)

    def transform(self, raw_win: float) -> float:
        """Legacy ``RecalibrationMap.transform`` for one probability, exactly.

        ``np.interp`` over the isotonic thresholds, clipped to [0, 1]; the
        input unchanged when there is no map (legacy's ``None`` map, and an
        empty threshold set).
        """
        if not self.fitted or not self.x_thresholds:
            return float(raw_win)
        x = np.asarray(self.x_thresholds, dtype=float)
        y = np.asarray(self.y_thresholds, dtype=float)
        calibrated = np.clip(np.interp(np.asarray(raw_win, dtype=float), x, y), 0.0, 1.0)
        return float(np.ravel(calibrated)[0])

    def _payload(self) -> dict[str, Any]:
        return _payload(
            strategy=self.strategy, alpha=self.alpha, cutoff=self.cutoff,
            min_pairs=self.min_pairs, fitted=self.fitted, n=self.n,
            base_rate=self.base_rate, x_thresholds=self.x_thresholds,
            y_thresholds=self.y_thresholds, window=(self.window_start, self.window_end),
        )


def _from_payload(payload: Mapping[str, Any]) -> RecalibrationMapArtifact:
    return RecalibrationMapArtifact(
        strategy=payload["strategy"], alpha=payload["alpha"], cutoff=payload["cutoff"],
        min_pairs=payload["min_pairs"], fitted=payload["fitted"], n=payload["n"],
        base_rate=payload["base_rate"], x_thresholds=_floats(payload["x_thresholds"]),
        y_thresholds=_floats(payload["y_thresholds"]), window_start=payload["window_start"],
        window_end=payload["window_end"], content_hash=content_hash(payload),
    )


def make_recalibration_map_artifact(
    fit: Mapping[str, Any] | None,
    *,
    strategy: str,
    alpha: float,
    cutoff: Any = None,
    min_pairs: int,
    window: tuple[Any, Any] | None = None,
) -> RecalibrationMapArtifact:
    """Wrap an already-fitted map (or legacy's ``None``) as a frozen artifact.

    Pure: fits nothing. ``fit`` carries ``n``, ``base_rate``,
    ``x_thresholds`` and ``y_thresholds`` exactly as the isotonic fit left
    them; ``None`` freezes the "too few closed pairs" answer.
    """
    if fit is None:
        payload = _payload(strategy=strategy, alpha=alpha, cutoff=cutoff, min_pairs=min_pairs,
                           fitted=False, n=None, base_rate=None, x_thresholds=(),
                           y_thresholds=(), window=window)
    else:
        payload = _payload(strategy=strategy, alpha=alpha, cutoff=cutoff, min_pairs=min_pairs,
                           fitted=True, n=fit["n"], base_rate=fit["base_rate"],
                           x_thresholds=fit["x_thresholds"], y_thresholds=fit["y_thresholds"],
                           window=window)
    return _from_payload(payload)


def serialize_recalibration_artifact(artifact: RecalibrationMapArtifact) -> bytes:
    """Canonical bytes: their sha256 IS the artifact's ``content_hash``."""
    return canonical_json(artifact._payload()).encode("utf-8")


class RecalibrationArtifactError(ValueError):
    """A recalibration artifact could not be verified or loaded."""


@dataclass(frozen=True, kw_only=True)
class RecalibrationArtifactRef:
    """Where one recalibration artifact's JSON lives, and what it must hash to."""

    path: str
    content_hash: str
    schema_version: str = "recalibration_artifact_ref.v1.0"


_FIELDS = ("strategy", "alpha", "cutoff", "min_pairs", "fitted", "n", "base_rate",
           "x_thresholds", "y_thresholds", "window_start", "window_end")


def _artifact_from_document(document: Mapping[str, Any]) -> RecalibrationMapArtifact:
    document = untag_nonfinite(document)
    schema = document.get("schema_version")
    if schema != RECALIBRATION_MAP_ARTIFACT_V1:
        raise RecalibrationArtifactError(
            f"unsupported recalibration artifact schema_version: {schema!r}")
    missing = [name for name in _FIELDS if name not in document]
    if missing:
        raise RecalibrationArtifactError(f"recalibration artifact is missing {missing}")
    payload = _payload(
        strategy=document["strategy"], alpha=document["alpha"], cutoff=document["cutoff"],
        min_pairs=document["min_pairs"], fitted=document["fitted"], n=document["n"],
        base_rate=document["base_rate"], x_thresholds=document["x_thresholds"],
        y_thresholds=document["y_thresholds"],
        window=(document["window_start"], document["window_end"]),
    )
    return _from_payload(payload)


class RecalibrationArtifactLoader:
    """Verified, read-only loading, shaped like ``PayoffArtifactLoader``.

    Re-hashes the raw bytes against the ref before parsing, refuses a path
    outside the artifact root, re-derives the artifact's own hash from the
    parsed payload, and caches by content hash (in memory only).
    """

    def __init__(self, artifact_root: Path | str) -> None:
        self._root = Path(artifact_root).resolve()
        self._cache: dict[str, RecalibrationMapArtifact] = {}

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def load(self, ref: RecalibrationArtifactRef) -> RecalibrationMapArtifact:
        cached = self._cache.get(ref.content_hash)
        if cached is not None:
            return cached
        path = (self._root / ref.path).resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise RecalibrationArtifactError("artifact path escapes artifact root") from exc
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RecalibrationArtifactError(f"missing recalibration artifact: {ref.path}") from exc
        if CONTENT_HASH_PREFIX + hashlib.sha256(raw).hexdigest() != ref.content_hash:
            raise RecalibrationArtifactError(f"recalibration artifact hash mismatch: {ref.path}")
        try:
            document = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecalibrationArtifactError("recalibration artifact is not valid JSON") from exc
        if not isinstance(document, Mapping):
            raise RecalibrationArtifactError("recalibration artifact must be a JSON object")
        artifact = _artifact_from_document(document)
        if artifact.content_hash != ref.content_hash:
            raise RecalibrationArtifactError(
                f"recalibration artifact content disagrees with its own hash: {ref.path}")
        self._cache[ref.content_hash] = artifact
        return artifact
