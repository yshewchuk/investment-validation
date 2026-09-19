"""Reading frozen residual state inside a score request (P5-4).

The model and simulation stages used to receive their residual populations
as request-supplied rows and re-derive the pool from them on every request
(``native_payoff.driver_residual_pool`` buckets the driver rows;
``stages._residual_arrays`` sorts the paired rows). When a bundle declares a
frozen artifact instead (``engine.v2.models.residual_artifact``), these
helpers read it -- after the same full causal-key check the payoff artifact
gets in ``stages._artifact_key_mismatch`` -- and never rebuild anything: a
missing, wrong-kind or wrong-key artifact is ``MODEL_NOT_READY``, with no
fallback to the rows path.

Each helper returns ``(value, flag)``: exactly one of the two is ``None``.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from engine.v2.models.residual_artifact import (
    DriverResidualPoolArtifact,
    PairedResidualPoolArtifact,
    driver_residual_pool_key,
    paired_residual_pool_key,
)
from engine.v2.scoring import native_payoff

__all__ = [
    "MODEL_NOT_READY",
    "PAIRED_ARTIFACT_FIELD",
    "PAIRED_KEY_FIELD",
    "RESIDUAL_ARTIFACTS_FIELD",
    "RESIDUAL_KEYS_FIELD",
    "driver_pool_from_artifact",
    "identity_view",
    "paired_arrays_from_artifact",
]

MODEL_NOT_READY = "MODEL_NOT_READY"
#: Simulation-block fields (see ``source_inputs._simulation_block``).
PAIRED_ARTIFACT_FIELD = "paired_residual_artifact"
PAIRED_KEY_FIELD = "paired_residual_key"
#: Model-block fields (see ``source_inputs._residual_members``).
RESIDUAL_ARTIFACTS_FIELD = "model_residual_artifacts"
RESIDUAL_KEYS_FIELD = "model_residual_artifact_recipe"

_PAIRED_KEY_PARTS = ("move_model_id", "crush_model_id", "cutoff")
_DRIVER_KEY_PARTS = ("role", "model_id", "fold")
_ARRAY_CACHE: dict[str, tuple[np.ndarray, ...]] = {}
_ARRAY_CACHE_LIMIT = 4


def _expected_key(expected: Any, parts: tuple[str, ...], build) -> tuple | None:
    """The request's own declared key, or ``None`` when any part is absent.

    A missing identity component can never match -- it is not "match by
    omission" (same rule as the payoff artifact's alpha).
    """
    if not isinstance(expected, Mapping) or any(part not in expected for part in parts):
        return None
    return build(*(expected[part] for part in parts))


def _mismatch(artifact: Any, kind: type, expected: Any, parts, build) -> bool:
    key = _expected_key(expected, parts, build)
    if key is None or not isinstance(artifact, kind) or artifact.key != key:
        return True
    pinned = expected.get("content_hash")
    return pinned is not None and pinned != artifact.content_hash


def paired_arrays_from_artifact(
    block: Mapping[str, Any],
) -> tuple[tuple[np.ndarray, ...] | None, str | None]:
    """``(dates, predicted, err_move, err_crush)`` read from the frozen pool.

    Same arrays ``stages._residual_arrays`` builds from rows, in the
    artifact's own fixed order (already date-sorted). Cached by content hash
    and marked read-only, so a request can neither pay the conversion twice
    nor mutate what the next request reads.
    """
    if block.get("residuals") is not None:
        return None, MODEL_NOT_READY  # two pools declared: refuse, never pick one
    artifact = block.get(PAIRED_ARTIFACT_FIELD)
    if _mismatch(artifact, PairedResidualPoolArtifact, block.get(PAIRED_KEY_FIELD),
                 _PAIRED_KEY_PARTS, paired_residual_pool_key):
        return None, MODEL_NOT_READY
    cached = _ARRAY_CACHE.get(artifact.content_hash)
    if cached is None:
        rows = artifact.rows
        cached = (
            np.asarray([row[0] for row in rows], dtype="datetime64[D]"),
            np.asarray([row[2] for row in rows], dtype=float),
            np.asarray([row[3] for row in rows], dtype=float),
            np.asarray([row[4] for row in rows], dtype=float),
        )
        for array in cached:
            array.setflags(write=False)
        if len(_ARRAY_CACHE) >= _ARRAY_CACHE_LIMIT:
            _ARRAY_CACHE.pop(next(iter(_ARRAY_CACHE)))
        _ARRAY_CACHE[artifact.content_hash] = cached
    return cached, None


def driver_pool_from_artifact(
    block: Mapping[str, Any], slot: str, prediction: float,
) -> tuple[np.ndarray | None, str | None]:
    """The frozen driver pool for ``prediction`` -- a lookup, never a rebuild.

    ``native_payoff.residual_pool_for`` over the artifact's frozen buckets
    and flat pool: identical to ``native_payoff.driver_residual_pool`` on the
    rows the artifact was built from (proved by the builder's parity test),
    without re-bucketing them per request.
    """
    artifacts = block.get(RESIDUAL_ARTIFACTS_FIELD) or {}
    expected = (block.get(RESIDUAL_KEYS_FIELD) or {}).get(slot)
    artifact = artifacts.get(slot)
    if _mismatch(artifact, DriverResidualPoolArtifact, expected,
                 _DRIVER_KEY_PARTS, driver_residual_pool_key):
        return None, MODEL_NOT_READY
    buckets = artifact.buckets
    if buckets is not None:
        buckets = {
            "edges": np.asarray(buckets["edges"], dtype=float),
            "pools": [np.asarray(pool, dtype=float) for pool in buckets["pools"]],
            "min_pool": buckets["min_pool"],
        }
    flat = np.asarray(artifact.flat_residuals, dtype=float)
    pool, _ = native_payoff.residual_pool_for(buckets, prediction, flat)
    if pool.size == 0:
        return None, "MISSING_MODEL_RESIDUALS"
    return pool, None


def identity_view(block: Mapping[str, Any]) -> dict[str, Any]:
    """``block`` with every frozen artifact replaced by its identity string.

    Stage receipts and observers document a block's inputs; a frozen
    artifact is identified by ``schema_version:content_hash`` (its
    ``__str__``), not by expanding every row into the receipt document.
    """
    frozen = (DriverResidualPoolArtifact, PairedResidualPoolArtifact)
    out: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, frozen):
            out[key] = str(value)
        elif isinstance(value, Mapping):
            out[key] = identity_view(value)
        else:
            out[key] = value
    return out
