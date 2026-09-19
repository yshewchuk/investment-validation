"""Frozen, content-hashed residual pools (P5-4).

Two residual populations reach native scoring, and until now both arrived as
request-supplied rows that the stage re-derived on every request:

* the DRIVER pool -- one forecast model's own held-out ``(prediction,
  residual)`` pairs, decile-bucketed (``native_payoff.bucket_residual_pool``,
  legacy ``registry.bucket_residuals``) and read by the model stage
  (``model_residual_rows`` / ``runup_move_residual_rows``);
* the PAIRED pool -- ``(err_move, err_crush)`` drawn from the SAME historical
  event (legacy ``engine.pnl_sim.ResidualPool`` via
  ``Scorer._residual_pool``), read by the planned-exit simulation. Legacy
  builds it from the Tier-4 forecasts joined against a crush table scoped to
  the LOADED tickers, so its content moves with scorer context -- the
  defect this artifact removes: it is built once, from an explicit universe,
  by ``engine/v2/models/training/residuals.py``, and scoring only reads it.

Both are immutable records carrying the frozen state, their causal key, their
:class:`~engine.v2.models.lineage.Lineage` and a content hash over all of it.
Layer 3: no fitting or bucketing math lives here (that is scoring's
``native_payoff`` at layer 5, called only by the layer-6 builder); these
constructors wrap already-derived arrays, exactly like
``payoff_artifact.make_payoff_line_artifact``.

Causal keys:

* driver pool -- ``(role, model_id, fold)``: the serving fold start as an ISO
  date (a Tier-4 monthly fold), or ``None`` for a full-refit champion's own
  embedded pool. Mirrors legacy ``tier4._pool_before(fold, model, panel)``,
  "a function of (fold, model, panel) and of NOTHING ELSE".
* paired pool -- ``(move_model_id, crush_model_id, cutoff)``: the two
  producers whose errors are paired and the exclusive event-date cutoff the
  pool was frozen at (rows are events dated strictly before it).
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from engine.v2.foundation.canonical import content_hash, untag_nonfinite
from engine.v2.models.lineage import Lineage, lineage_from_document

__all__ = [
    "DRIVER_RESIDUAL_POOL_ARTIFACT_V1",
    "PAIRED_RESIDUAL_POOL_ARTIFACT_V1",
    "PAIRED_COLUMNS",
    "DriverResidualPoolArtifact",
    "DriverResidualPoolKey",
    "PairedResidualPoolArtifact",
    "PairedResidualPoolKey",
    "ResidualArtifactError",
    "driver_residual_pool_key",
    "make_driver_residual_pool_artifact",
    "make_paired_residual_pool_artifact",
    "paired_residual_pool_key",
    "residual_artifact_from_document",
]

DRIVER_RESIDUAL_POOL_ARTIFACT_V1 = "driver_residual_pool_artifact.v1.0"
PAIRED_RESIDUAL_POOL_ARTIFACT_V1 = "paired_residual_pool_artifact.v1.0"

#: Column order of one paired-pool row. ``ticker`` is carried for lineage
#: (which event a row came from) and for the deterministic total order; the
#: simulation reads only the other four.
PAIRED_COLUMNS = ("event_date", "ticker", "pred_abs_move", "err_move", "err_crush")

DriverResidualPoolKey = tuple[str, str, "str | None"]
PairedResidualPoolKey = tuple[str, str, "str | None"]


class ResidualArtifactError(ValueError):
    """A residual artifact is malformed or cannot be verified."""


def _day(value: Any) -> str | None:
    return None if value is None else str(value)[:10]


def driver_residual_pool_key(role: str, model_id: str, fold: Any) -> DriverResidualPoolKey:
    return (str(role), str(model_id), _day(fold))


def paired_residual_pool_key(
    move_model_id: str, crush_model_id: str, cutoff: Any,
) -> PairedResidualPoolKey:
    return (str(move_model_id), str(crush_model_id), _day(cutoff))


def _floats(values: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _buckets_tuple(buckets: Mapping[str, Any] | None):
    if not buckets:
        return None, None
    edges = _floats(buckets["edges"])
    pools = tuple(_floats(pool) for pool in buckets["pools"])
    if len(pools) != len(edges) - 1:
        raise ResidualArtifactError("bucket pools must number len(edges) - 1")
    return edges, pools


@dataclass(frozen=True, kw_only=True)
class DriverResidualPoolArtifact:
    """One driver model's held-out residual pool, flat and decile-bucketed."""

    schema_version: str = DRIVER_RESIDUAL_POOL_ARTIFACT_V1
    role: str
    model_id: str
    fold: str | None
    deciles: int
    min_pool: int
    flat_residuals: tuple[float, ...]
    bucket_edges: tuple[float, ...] | None
    bucket_pools: tuple[tuple[float, ...], ...] | None
    lineage: Lineage
    content_hash: str

    def __str__(self) -> str:
        return f"{self.schema_version}:{self.content_hash}"

    @property
    def key(self) -> DriverResidualPoolKey:
        return driver_residual_pool_key(self.role, self.model_id, self.fold)

    @property
    def buckets(self) -> dict[str, Any] | None:
        """``native_payoff.residual_pool_for``'s bucket mapping, or ``None``."""
        if self.bucket_edges is None:
            return None
        return {"edges": self.bucket_edges, "pools": self.bucket_pools,
                "min_pool": self.min_pool}

    def payload(self) -> dict[str, Any]:
        buckets = None
        if self.bucket_edges is not None:
            buckets = {"edges": list(self.bucket_edges),
                       "pools": [list(pool) for pool in self.bucket_pools]}
        return {
            "schema_version": self.schema_version,
            "role": self.role, "model_id": self.model_id, "fold": self.fold,
            "deciles": int(self.deciles), "min_pool": int(self.min_pool),
            "n": len(self.flat_residuals),
            "flat_residuals": list(self.flat_residuals),
            "buckets": buckets,
            "lineage": self.lineage.document(),
        }


def make_driver_residual_pool_artifact(
    *,
    role: str,
    model_id: str,
    fold: Any,
    flat_residuals: Sequence[float],
    buckets: Mapping[str, Any] | None,
    deciles: int,
    min_pool: int,
    lineage: Lineage,
) -> DriverResidualPoolArtifact:
    """Wrap an already-bucketed driver pool (no math here; see module doc)."""
    edges, pools = _buckets_tuple(buckets)
    draft = DriverResidualPoolArtifact(
        role=str(role), model_id=str(model_id), fold=_day(fold),
        deciles=int(deciles), min_pool=int(min_pool),
        flat_residuals=_floats(flat_residuals), bucket_edges=edges,
        bucket_pools=pools, lineage=lineage.canonical(), content_hash="",
    )
    return _with_hash(draft)


def _paired_row(row: Sequence[Any]) -> tuple[str, str, float, float, float]:
    if len(row) != len(PAIRED_COLUMNS):
        raise ResidualArtifactError(f"paired row must have {len(PAIRED_COLUMNS)} columns")
    frozen = (str(row[0])[:10], str(row[1]), float(row[2]), float(row[3]), float(row[4]))
    # NaN is a MISSING value: legacy ResidualPool drops it (``dropna``), so a
    # builder that lets one through was built wrong. +/-inf is not missing:
    # legacy keeps such a row and simulates it (R4-20 gap 1), so it is frozen.
    if any(value != value for value in frozen[2:]):
        raise ResidualArtifactError("paired pool values must not be NaN")
    return frozen


@dataclass(frozen=True, kw_only=True)
class PairedResidualPoolArtifact:
    """Paired ``(err_move, err_crush)`` rows, date-ordered, frozen at a cutoff."""

    schema_version: str = PAIRED_RESIDUAL_POOL_ARTIFACT_V1
    move_model_id: str
    crush_model_id: str
    cutoff: str | None
    rows: tuple[tuple[str, str, float, float, float], ...]
    lineage: Lineage
    content_hash: str

    def __str__(self) -> str:
        return f"{self.schema_version}:{self.content_hash}"

    @property
    def key(self) -> PairedResidualPoolKey:
        return paired_residual_pool_key(self.move_model_id, self.crush_model_id, self.cutoff)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "move_model_id": self.move_model_id,
            "crush_model_id": self.crush_model_id,
            "cutoff": self.cutoff,
            "n": len(self.rows),
            "columns": list(PAIRED_COLUMNS),
            "rows": [list(row) for row in self.rows],
            "lineage": self.lineage.document(),
        }


def make_paired_residual_pool_artifact(
    *,
    move_model_id: str,
    crush_model_id: str,
    cutoff: Any,
    rows: Sequence[Sequence[Any]],
    lineage: Lineage,
) -> PairedResidualPoolArtifact:
    """Freeze paired rows in their one canonical order.

    The order is the total order ``(event_date, ticker, pred, err_move,
    err_crush)``, so the artifact -- and the simulation's index-based draws
    -- cannot depend on the order a caller happened to assemble rows in.
    A row dated on/after ``cutoff`` is refused, not dropped: dropping is the
    builder's causal filter, and an artifact that receives one anyway was
    built wrong.
    """
    frozen = tuple(sorted(_paired_row(row) for row in rows))
    bound = _day(cutoff)
    if bound is not None and any(row[0] >= bound for row in frozen):
        raise ResidualArtifactError("paired pool row dated on/after its own cutoff")
    draft = PairedResidualPoolArtifact(
        move_model_id=str(move_model_id), crush_model_id=str(crush_model_id),
        cutoff=bound, rows=frozen, lineage=lineage.canonical(), content_hash="",
    )
    return _with_hash(draft)


def _with_hash(draft):
    return replace(draft, content_hash=content_hash(draft.payload()))


def _driver_from_document(document: Mapping[str, Any]) -> DriverResidualPoolArtifact:
    buckets = document.get("buckets")
    if buckets is not None:
        buckets = {"edges": buckets["edges"], "pools": buckets["pools"]}
    return make_driver_residual_pool_artifact(
        role=document["role"], model_id=document["model_id"], fold=document.get("fold"),
        flat_residuals=document["flat_residuals"], buckets=buckets,
        deciles=document["deciles"], min_pool=document["min_pool"],
        lineage=lineage_from_document(document.get("lineage")),
    )


def _paired_from_document(document: Mapping[str, Any]) -> PairedResidualPoolArtifact:
    if tuple(document.get("columns", ())) != PAIRED_COLUMNS:
        raise ResidualArtifactError("paired pool columns do not match PAIRED_COLUMNS")
    return make_paired_residual_pool_artifact(
        move_model_id=document["move_model_id"],
        crush_model_id=document["crush_model_id"],
        cutoff=document.get("cutoff"), rows=document["rows"],
        lineage=lineage_from_document(document.get("lineage")),
    )


def residual_artifact_from_document(
    document: Mapping[str, Any],
) -> DriverResidualPoolArtifact | PairedResidualPoolArtifact:
    """Rebuild an artifact from its JSON document (hash recomputed, not trusted)."""
    document = untag_nonfinite(dict(document))
    schema = document.get("schema_version")
    if schema == DRIVER_RESIDUAL_POOL_ARTIFACT_V1:
        return _driver_from_document(document)
    if schema == PAIRED_RESIDUAL_POOL_ARTIFACT_V1:
        return _paired_from_document(document)
    raise ResidualArtifactError(f"unsupported residual artifact schema_version: {schema!r}")
