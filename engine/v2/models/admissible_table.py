"""The DYN-SV chooser's depth -> ``n_admissible`` map as a versioned artifact (P5-4).

Legacy serves the chooser feature ``n_admissible`` through
``Scorer._N_ADMISSIBLE_BY_DEPTH`` (``engine/score.py``): an 18-point piecewise
map from live chain depth to the training-time conditional median, with
``_N_ADMISSIBLE_MEDIAN`` for a non-finite depth. It is a Python literal, so
no release, registry or completeness check can see it (P5-1 inventory,
"What the guide's list did not anticipate"). This module gives that exact
table an identity -- a table id, a version, its calibration provenance and
a content hash -- that a release pins like any other calibration member,
plus the lookup, bit-for-bit the legacy ``_n_admissible_for``.

The literal is copied, not imported (``engine/v2`` never imports legacy
code). ``tests/test_v2_models_admissible_table.py`` imports the legacy class
and fails if the two ever disagree, and pins
:data:`N_ADMISSIBLE_BY_DEPTH_V1_HASH`, so an edit on either side is a new
version, never a silent change under the same identity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.foundation.canonical import content_hash, untag_nonfinite
from engine.v2.models.lineage import DataDependency, Lineage, lineage_from_document

__all__ = [
    "ADMISSIBLE_DEPTH_TABLE_V1",
    "N_ADMISSIBLE_BY_DEPTH_V1_HASH",
    "N_ADMISSIBLE_TABLE_ID",
    "AdmissibleDepthTable",
    "AdmissibleTableError",
    "admissible_table_from_document",
    "legacy_n_admissible_table",
    "make_admissible_depth_table",
    "n_admissible_for",
]

ADMISSIBLE_DEPTH_TABLE_V1 = "admissible_depth_table.v1.0"
N_ADMISSIBLE_TABLE_ID = "dyn_sv.n_admissible_by_depth"

#: Copy of ``engine.score.Scorer._N_ADMISSIBLE_BY_DEPTH`` (verified equal by
#: the module's test). ``(depth floor, value)``: the value applies to depths
#: strictly below the floor and at or above the previous one.
_LEGACY_BREAKPOINTS: tuple[tuple[float, float], ...] = (
    (6.0, 5.0), (7.0, 11.0), (8.0, 16.0), (9.0, 26.0), (10.0, 31.5),
    (11.0, 54.0), (12.0, 59.5), (13.0, 78.0), (15.0, 127.0),
    (17.0, 159.5), (19.0, 299.0), (23.0, 485.0), (26.0, 718.0),
    (33.0, 1572.5), (42.0, 3158.0), (49.0, 5635.0), (62.0, 6742.0),
    (176.0, 5674.0),
)
#: Copy of ``engine.score._N_ADMISSIBLE_MEDIAN`` -- the non-finite-depth value.
_LEGACY_FALLBACK = 125.0
_LEGACY_PROVENANCE = (
    "engine/score.py Scorer._N_ADMISSIBLE_BY_DEPTH and _N_ADMISSIBLE_MEDIAN: "
    "conditional median of training n_admissible given live chain depth, "
    "calibrated on 2,853 menu events of 2025-2026, measured 2026-09-09"
)
#: What the v1 calibration read: the DYN-SV menu events it was measured on,
#: through the measurement date. A 3B correction to that table dated before
#: 2026-09-10 therefore invalidates this table (and anything built on it).
_LEGACY_LINEAGE = Lineage(data=(
    DataDependency(table="dyn_sv.menu_events", end_exclusive="2026-09-10"),
))

#: The content hash of :func:`legacy_n_admissible_table` -- what a release
#: pins today. Pinned by the test; changing the table means a new version.
N_ADMISSIBLE_BY_DEPTH_V1_HASH = (
    "sha256:a1b4b7909cff6db0296a44acc6825fb83393983bf7291bf7d734af3f47e78f0d"
)


class AdmissibleTableError(ValueError):
    """The table is malformed (non-finite, unordered, or empty)."""


@dataclass(frozen=True, kw_only=True)
class AdmissibleDepthTable:
    schema_version: str = ADMISSIBLE_DEPTH_TABLE_V1
    table_id: str
    version: str
    breakpoints: tuple[tuple[float, float], ...]
    fallback: float
    provenance: str
    lineage: Lineage
    content_hash: str

    def __str__(self) -> str:
        return f"{self.schema_version}:{self.content_hash}"

    @property
    def key(self) -> tuple[str, str]:
        return (self.table_id, self.version)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "table_id": self.table_id,
            "version": self.version,
            "breakpoints": [[floor, value] for floor, value in self.breakpoints],
            "fallback": self.fallback,
            "provenance": self.provenance,
            "lineage": self.lineage.document(),
        }


def _validated(breakpoints: Sequence[Sequence[Any]]) -> tuple[tuple[float, float], ...]:
    points = tuple((float(floor), float(value)) for floor, value in breakpoints)
    if not points:
        raise AdmissibleTableError("table must have at least one breakpoint")
    if not all(isfinite(floor) and isfinite(value) for floor, value in points):
        raise AdmissibleTableError("breakpoints must be finite")
    floors = [floor for floor, _ in points]
    if any(later <= earlier for earlier, later in zip(floors, floors[1:])):
        raise AdmissibleTableError("breakpoint floors must be strictly increasing")
    return points


def make_admissible_depth_table(
    *,
    table_id: str,
    version: str,
    breakpoints: Sequence[Sequence[Any]],
    fallback: float,
    provenance: str,
    lineage: Lineage,
) -> AdmissibleDepthTable:
    if not isfinite(float(fallback)):
        raise AdmissibleTableError("fallback must be finite")
    draft = AdmissibleDepthTable(
        table_id=str(table_id), version=str(version),
        breakpoints=_validated(breakpoints), fallback=float(fallback),
        provenance=str(provenance), lineage=lineage.canonical(), content_hash="",
    )
    return replace(draft, content_hash=content_hash(draft.payload()))


def legacy_n_admissible_table() -> AdmissibleDepthTable:
    """Today's table, version ``v1``, exactly as legacy serves it."""
    return make_admissible_depth_table(
        table_id=N_ADMISSIBLE_TABLE_ID, version="v1",
        breakpoints=_LEGACY_BREAKPOINTS, fallback=_LEGACY_FALLBACK,
        provenance=_LEGACY_PROVENANCE, lineage=_LEGACY_LINEAGE,
    )


def n_admissible_for(table: AdmissibleDepthTable, depth: float) -> float:
    """Legacy ``Scorer._n_admissible_for``, reading the frozen table."""
    depth = float(depth)
    if not isfinite(depth):
        return table.fallback
    for floor, value in table.breakpoints:
        if depth < floor:
            return value
    return table.breakpoints[-1][1]


def admissible_table_from_document(document: Mapping[str, Any]) -> AdmissibleDepthTable:
    document = untag_nonfinite(dict(document))
    if document.get("schema_version") != ADMISSIBLE_DEPTH_TABLE_V1:
        raise AdmissibleTableError(
            f"unsupported admissible table schema_version: {document.get('schema_version')!r}"
        )
    return make_admissible_depth_table(
        table_id=document["table_id"], version=document["version"],
        breakpoints=document["breakpoints"], fallback=document["fallback"],
        provenance=document["provenance"],
        lineage=lineage_from_document(document.get("lineage")),
    )
