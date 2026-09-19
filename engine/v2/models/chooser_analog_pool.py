"""The DYN-SV chooser's k-NN analog population as a frozen artifact (P5-4).

The chooser champion (``dyn_sv_chooser_v1_1``) reads five ``analog_*``
columns that are NOT the board's bucket-analog layer: for each menu candidate
they are the 25 nearest PRIOR candidates of the same structure in the space of
``(exp_pnl_sim, width_over_forecast, n_legs, anchor_over_spot, rel_spread)``,
summarized as dollar P&L (legacy ``engine/score.py``
``Scorer._chooser_analogs``). Legacy reads that population from
``data/features/chooser_analog_pool.parquet`` (built by
``tools/build_chooser_pool.py``) through ``Scorer._chooser_analog_pool`` --
an unversioned file no release can pin. This record freezes it:

* one row per prior candidate, ``(exit_date, pnl, *DIMS)``, grouped by
  structure; within a structure the rows keep the SOURCE order. Order is part
  of the state: legacy ``np.argpartition`` picks the K nearest by position
  among ties, and the mean of the picked P&L is summed in that order, so a
  re-sorted pool would not be bit-for-bit the legacy neighbourhood;
* a causal key ``(pool_id, cutoff)``: every row closed strictly before
  ``cutoff``. Scoring still applies legacy's per-request filter (closed
  strictly before the request's entry date); the key is what a request
  declares and the stage checks before trusting the pool;
* its :class:`~engine.v2.models.lineage.Lineage` and a content hash over all
  of it (``sha256(serialize_frozen_state(pool)) == pool.content_hash``).

Layer 3: this wraps already-filtered rows; the legacy filter lives in the
layer-6 builder (``engine/v2/models/training/chooser_pool.py``) and the
neighbourhood arithmetic in scoring (``native_chooser_features``).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.foundation.canonical import content_hash, untag_nonfinite
from engine.v2.models.lineage import Lineage, lineage_from_document

__all__ = [
    "CHOOSER_ANALOG_DIMS",
    "CHOOSER_ANALOG_K",
    "CHOOSER_ANALOG_POOL_ARTIFACT_V1",
    "ChooserAnalogPoolArtifact",
    "ChooserAnalogPoolError",
    "chooser_analog_pool_from_document",
    "chooser_analog_pool_key",
    "make_chooser_analog_pool_artifact",
]

CHOOSER_ANALOG_POOL_ARTIFACT_V1 = "chooser_analog_pool_artifact.v1.0"
#: engine/score.py ``Scorer._CHOOSER_ANALOG_DIMS`` (copied; the artifact test
#: pins the two equal).
CHOOSER_ANALOG_DIMS = ("exp_pnl_sim", "width_over_forecast", "n_legs",
                       "anchor_over_spot", "rel_spread")
#: engine/score.py ``Scorer._CHOOSER_ANALOG_K``.
CHOOSER_ANALOG_K = 25

ChooserAnalogPoolKey = tuple[str, "str | None"]
PoolRow = tuple[str, float, float, float, float, float, float]


class ChooserAnalogPoolError(ValueError):
    """The chooser analog pool is malformed or cannot be verified."""


def _day(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


def chooser_analog_pool_key(pool_id: str, cutoff: Any) -> ChooserAnalogPoolKey:
    return (str(pool_id), _day(cutoff))


def _row(row: Sequence[Any], bound: str | None) -> PoolRow:
    if len(row) != 2 + len(CHOOSER_ANALOG_DIMS):
        raise ChooserAnalogPoolError(
            f"pool row must have {2 + len(CHOOSER_ANALOG_DIMS)} columns")
    day = _day(row[0])
    if not day or len(day) != 10:
        raise ChooserAnalogPoolError("pool row needs an exit date")
    pnl = float(row[1])
    dims = tuple(float(value) for value in row[2:])
    # Legacy drops a MISSING P&L (``dropna``) and keeps an infinite one; it
    # keeps only rows whose five dimensions are all finite.
    if pnl != pnl:
        raise ChooserAnalogPoolError("pool P&L must not be NaN")
    if not all(isfinite(value) for value in dims):
        raise ChooserAnalogPoolError("pool dimensions must be finite")
    if bound is not None and day >= bound:
        raise ChooserAnalogPoolError("pool row closed on/after its own cutoff")
    return (day, pnl, *dims)


@dataclass(frozen=True, kw_only=True)
class ChooserAnalogPoolArtifact:
    """Prior menu candidates with realized P&L, per structure, source order."""

    schema_version: str = CHOOSER_ANALOG_POOL_ARTIFACT_V1
    pool_id: str
    cutoff: str | None
    dims: tuple[str, ...]
    k: int
    strategies: tuple[tuple[str, tuple[PoolRow, ...]], ...]
    lineage: Lineage
    content_hash: str

    def __str__(self) -> str:
        return f"{self.schema_version}:{self.content_hash}"

    @property
    def key(self) -> ChooserAnalogPoolKey:
        return chooser_analog_pool_key(self.pool_id, self.cutoff)

    def rows_for(self, strategy: str) -> tuple[PoolRow, ...] | None:
        for name, rows in self.strategies:
            if name == strategy:
                return rows
        return None

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "pool_id": self.pool_id,
            "cutoff": self.cutoff,
            "columns": ["exit_date", "pnl", *self.dims],
            "k": self.k,
            "strategies": {name: [list(row) for row in rows]
                           for name, rows in self.strategies},
            "lineage": self.lineage.document(),
        }


def make_chooser_analog_pool_artifact(
    *,
    pool_id: str,
    cutoff: Any,
    strategies: Mapping[str, Sequence[Sequence[Any]]],
    lineage: Lineage,
) -> ChooserAnalogPoolArtifact:
    """Freeze per-structure rows, keeping each structure's row order.

    A row closed on/after ``cutoff``, a NaN P&L or a non-finite dimension is
    refused, not dropped: dropping is the builder's filter, and an artifact
    that receives such a row was built wrong. A structure with no row is
    omitted (legacy serves it no neighbourhood either).
    """
    bound = _day(cutoff)
    frozen = tuple(sorted(
        (str(name), tuple(_row(row, bound) for row in rows))
        for name, rows in strategies.items() if rows
    ))
    draft = ChooserAnalogPoolArtifact(
        pool_id=str(pool_id), cutoff=bound, dims=CHOOSER_ANALOG_DIMS,
        k=CHOOSER_ANALOG_K, strategies=frozen, lineage=lineage.canonical(),
        content_hash="",
    )
    return replace(draft, content_hash=content_hash(draft.payload()))


def chooser_analog_pool_from_document(
    document: Mapping[str, Any],
) -> ChooserAnalogPoolArtifact:
    """Rebuild from the JSON document (hash recomputed, never trusted)."""
    document = untag_nonfinite(dict(document))
    if document.get("schema_version") != CHOOSER_ANALOG_POOL_ARTIFACT_V1:
        raise ChooserAnalogPoolError(
            "unsupported chooser analog pool schema_version: "
            f"{document.get('schema_version')!r}")
    columns = tuple(document.get("columns", ()))
    if columns != ("exit_date", "pnl", *CHOOSER_ANALOG_DIMS):
        raise ChooserAnalogPoolError("chooser analog pool columns do not match")
    if int(document.get("k", -1)) != CHOOSER_ANALOG_K:
        raise ChooserAnalogPoolError("chooser analog pool k does not match")
    return make_chooser_analog_pool_artifact(
        pool_id=document["pool_id"], cutoff=document.get("cutoff"),
        strategies=document["strategies"],
        lineage=lineage_from_document(document.get("lineage")),
    )
