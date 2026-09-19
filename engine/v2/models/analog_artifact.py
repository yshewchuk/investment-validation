"""The frozen board analog-matcher population (P5-4).

Legacy ``engine.analogs.AnalogMatcher`` is built inside every Scorer from the
Scorer's own enriched trades (``Scorer._enrich``: a left merge of the LOADED
panel, then ``bucket_frame``). For each request it takes the ``(strategy,
alpha)`` pool, keeps the trades closed strictly before the request's evidence
cutoff, re-derives the implied-ratio tercile edges from that causal slice and
re-buckets it. A bounded Scorer (the nightly's) loads a narrower panel, so
the same request matched a different population: the context-width defect.

This artifact is that causal slice, frozen once from the full universe by
``engine/v2/models/training/analogs.py``: one record per causal key
``(strategy, alpha, cutoff)``. It holds the slice's rows, already bucketed on
the slice's own causal edges, plus the two edge pairs the request side needs
(the causal edges, and the population edges legacy falls back to when the
slice is empty). The native analog stage reads it after a full key check and
never rebuilds it (``engine.v2.scoring.native_analog``).

Layer 3: no bucketing or quantile math lives here; the constructor wraps
rows the layer-6 builder already derived, as ``residual_artifact`` does.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.foundation.canonical import content_hash, untag_nonfinite
from engine.v2.models.lineage import Lineage, lineage_from_document

__all__ = [
    "BOARD_ANALOG_POOL_ARTIFACT_V1",
    "BOARD_ANALOG_COLUMNS",
    "AnalogArtifactError",
    "BoardAnalogPoolArtifact",
    "BoardAnalogPoolKey",
    "analog_artifact_from_document",
    "board_analog_pool_key",
    "make_board_analog_pool_artifact",
]

BOARD_ANALOG_POOL_ARTIFACT_V1 = "board_analog_pool_artifact.v1.0"

#: Column order of one frozen row. ``row_id`` is the trade id (lineage and
#: the canonical order); the four bucket columns are what the widening
#: ladder matches on; ``ret`` is the realized return, ``None`` where legacy's
#: ``_summarize`` would drop it as non-finite (the row still counts toward
#: ``min_analogs``, exactly as in legacy's ``len(matched)``).
BOARD_ANALOG_COLUMNS = (
    "row_id", "event_date", "exit_date", "mcap_bucket", "dte_band",
    "moneyness_band", "implied_tercile", "ret",
)

BoardAnalogPoolKey = tuple[str, float, "str | None"]


class AnalogArtifactError(ValueError):
    """A board analog artifact is malformed or cannot be verified."""


def _day(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


def _alpha(value: Any) -> float:
    """Legacy keys the pool as ``round(float(alpha), 4)``."""
    return round(float(value), 4)


def board_analog_pool_key(strategy: str, alpha: Any, cutoff: Any) -> BoardAnalogPoolKey:
    return (str(strategy), _alpha(alpha), _day(cutoff))


def _edges(value: Any, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    edges = tuple(float(item) for item in value)
    if len(edges) != 2 or not all(isfinite(item) for item in edges):
        raise AnalogArtifactError(f"{name} must be two finite floats")
    return edges


def _label(value: Any) -> str | None:
    return None if value is None else str(value)


def _row(row: Sequence[Any]) -> tuple:
    if len(row) != len(BOARD_ANALOG_COLUMNS):
        raise AnalogArtifactError(f"analog row must have {len(BOARD_ANALOG_COLUMNS)} columns")
    row_id = str(row[0]).strip()
    if not row_id:
        raise AnalogArtifactError("analog row has an empty row_id")
    ret = row[7]
    if ret is not None:
        ret = float(ret)
        if not isfinite(ret):
            raise AnalogArtifactError("a non-finite return must be frozen as None")
    return (row_id, _day(row[1]), _day(row[2]), _label(row[3]), _label(row[4]),
            _label(row[5]), _label(row[6]), ret)


@dataclass(frozen=True, kw_only=True)
class BoardAnalogPoolArtifact:
    """One ``(strategy, alpha, cutoff)`` causal analog slice, bucketed."""

    schema_version: str = BOARD_ANALOG_POOL_ARTIFACT_V1
    strategy: str
    alpha: float
    cutoff: str | None
    population_edges: tuple[float, float]
    causal_edges: tuple[float, float] | None
    rows: tuple[tuple, ...]
    lineage: Lineage
    content_hash: str

    def __str__(self) -> str:
        return f"{self.schema_version}:{self.content_hash}"

    @property
    def key(self) -> BoardAnalogPoolKey:
        return board_analog_pool_key(self.strategy, self.alpha, self.cutoff)

    @property
    def request_edges(self) -> tuple[float, float]:
        """The edges a request's implied ratio is bucketed on.

        Legacy ``match`` re-buckets the request on the causal edges when the
        slice has them, and otherwise keeps ``buckets_for``'s population-edge
        label (no cutoff, or an empty slice).
        """
        return self.causal_edges if self.causal_edges is not None else self.population_edges

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "strategy": self.strategy,
            "alpha": self.alpha,
            "cutoff": self.cutoff,
            "population_edges": list(self.population_edges),
            "causal_edges": None if self.causal_edges is None else list(self.causal_edges),
            "n": len(self.rows),
            "columns": list(BOARD_ANALOG_COLUMNS),
            "rows": [list(row) for row in self.rows],
            "lineage": self.lineage.document(),
        }


def make_board_analog_pool_artifact(
    *,
    strategy: str,
    alpha: Any,
    cutoff: Any,
    population_edges: Sequence[float],
    causal_edges: Sequence[float] | None,
    rows: Sequence[Sequence[Any]],
    lineage: Lineage,
) -> BoardAnalogPoolArtifact:
    """Freeze already-bucketed rows in their one canonical order (by row_id).

    A row that closed on/after ``cutoff``, or has no exit date under a
    cutoff, is refused, not dropped: dropping is the builder's causal
    filter, and an artifact that receives one anyway was built wrong.
    """
    frozen = tuple(sorted((_row(row) for row in rows), key=lambda row: row[0]))
    ids = [row[0] for row in frozen]
    if len(set(ids)) != len(ids):
        raise AnalogArtifactError("duplicate row_id in the analog pool")
    bound = _day(cutoff)
    if bound is not None and any(row[2] is None or row[2] >= bound for row in frozen):
        raise AnalogArtifactError("analog row closed on/after its own cutoff")
    population = _edges(population_edges, "population_edges")
    if population is None:
        raise AnalogArtifactError("population_edges are required")
    draft = BoardAnalogPoolArtifact(
        strategy=str(strategy), alpha=_alpha(alpha), cutoff=bound,
        population_edges=population, causal_edges=_edges(causal_edges, "causal_edges"),
        rows=frozen, lineage=lineage.canonical(), content_hash="",
    )
    return replace(draft, content_hash=content_hash(draft.payload()))


def analog_artifact_from_document(document: Mapping[str, Any]) -> BoardAnalogPoolArtifact:
    """Rebuild an artifact from its JSON document (hash recomputed, not trusted)."""
    document = untag_nonfinite(dict(document))
    if document.get("schema_version") != BOARD_ANALOG_POOL_ARTIFACT_V1:
        raise AnalogArtifactError(
            f"unsupported analog artifact schema_version: {document.get('schema_version')!r}")
    if tuple(document.get("columns", ())) != BOARD_ANALOG_COLUMNS:
        raise AnalogArtifactError("analog pool columns do not match BOARD_ANALOG_COLUMNS")
    return make_board_analog_pool_artifact(
        strategy=document["strategy"], alpha=document["alpha"],
        cutoff=document.get("cutoff"), population_edges=document["population_edges"],
        causal_edges=document.get("causal_edges"), rows=document["rows"],
        lineage=lineage_from_document(document.get("lineage")),
    )
