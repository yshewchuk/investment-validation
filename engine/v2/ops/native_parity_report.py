"""Per-night native-vs-legacy parity report (spec_ns_c, G2/G4).

This module is the reporting half of the native shadow-serving seam
(``engine.v2.ops.native_shadow_render`` / ``engine.v2.serving.native_shadow_render``):
it compares the legacy rows the projection already carries with the native rows
``build_native_bundle_rows`` built, and writes the differences down.  It is
deliberately NOT a second comparator: every per-dimension verdict comes from
``engine.v2.parity.dimensions.compare_dimension`` -- the same tolerance policy,
the same ``engine.v2.diagnosis`` machinery and the same field groups the Phase
4 corpus comparison uses -- so the nightly report can never drift away from the
checker it reports on.  That module is where the checker's comparator moved
(``spec_ns_c`` part c: production never imports ``checks/``, and "never
duplicate the comparison logic here"); no comparison rule is re-implemented in
this file.

G2 (report, never reconcile): ``compare_native_vs_legacy`` classifies and
returns.  It has no code path that copies a native value into a legacy row (or
the reverse), and a mismatch is a return-value finding, never an exception.

G4 (no hashed payload changes): this module only reads rows and writes a NEW,
separate report artifact at the caller's explicit path.  It never touches
``score.json``'s ``rows``/``ladder``, any ``ScoreRecord`` field, or the corpus
``checks/phase4_real.py`` hashes.

The ``native_parity`` stage is OPTIONAL in ``engine.v2.ops.nightly.GRAPH``: a
parity-report failure degrades the shadow board's receipt, it never blocks it
(G2: a difference is reported, not turned into a hard gate).  In ``"legacy"``
serving mode there are no native rows to compare, so
:func:`native_parity_handler` returns ``{"status": "not_applicable"}`` instead
of the stage being omitted from the graph (a conditionally absent stage is
exactly the DAG-shape drift ``nightly.GRAPH`` exists to avoid).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from engine.v2.ops.errors import fail
from engine.v2.ops.native_shadow_render import native_shadow_serving_mode
from engine.v2.parity.dimensions import (
    ANALOG_FIELDS,
    FINANCIAL_FIELDS,
    FORECAST_FIELDS,
    GATE_FIELDS,
    NEVER_RAN_DIMENSIONS,
    SIMULATION_FIELDS,
    compare_dimension,
)
from engine.v2.parity.tolerance import SCORE_RECORD_V1, TolerancePolicy

__all__ = [
    "PARITY_DIMENSIONS",
    "SCHEMA_VERSION",
    "compare_native_vs_legacy",
    "native_parity_handler",
    "write_parity_report",
]

SCHEMA_VERSION = "native_parity_report.v1.0"

#: The checker's own numeric field groups, reused by name -- see
#: ``checks/phase4_real._compare_numeric_outputs``, whose dimension names and
#: field tuples these are.  A dimension outside this map is refused, never
#: compared against an empty view.
_DIMENSION_FIELDS: dict[str, tuple[str, ...]] = {
    "forecasts": FORECAST_FIELDS,
    "simulation": SIMULATION_FIELDS,
    "financial_diagnostics": FINANCIAL_FIELDS,
    "verdicts": GATE_FIELDS,
    "analogs": ANALOG_FIELDS,
}

#: Every one of the checker's five numeric field groups.  ``forecasts`` and
#: ``financial_diagnostics`` carry the served numbers, so they are compared
#: too; the ``NEVER_RAN_DIMENSIONS`` groups (``analogs``, ``simulation``,
#: ``verdicts``) stay in, including their typed placeholder defaults, rather
#: than being silently excluded from the report.
PARITY_DIMENSIONS: tuple[str, ...] = tuple(sorted(_DIMENSION_FIELDS))
if not set(NEVER_RAN_DIMENSIONS) <= set(PARITY_DIMENSIONS):
    raise RuntimeError("PARITY_DIMENSIONS must cover every never-ran dimension")


def _dimension_fields(dimension: str) -> tuple[str, ...]:
    """The checker's field group for ``dimension``, or a typed refusal."""
    fields = _DIMENSION_FIELDS.get(dimension)
    if fields is None:
        raise fail("INVALID_REQUEST", "unknown native parity dimension",
                   details={"dimension": dimension,
                            "known": sorted(_DIMENSION_FIELDS)})
    return fields


def _dimension_view(row: Mapping[str, Any], dimension: str) -> dict[str, Any]:
    """One row's values for one dimension, read by the checker's field names."""
    return {name: row.get(name) for name in _dimension_fields(dimension)}


def _row_mismatches(key: str, legacy: Mapping[str, Any], native: Mapping[str, Any],
                    dimensions: tuple[str, ...], tolerance_policy: TolerancePolicy,
                    ) -> list[dict[str, Any]]:
    """Every dimension of one shared key that does not agree, with its receipt."""
    mismatches = []
    for dimension in dimensions:
        result = compare_dimension(
            _dimension_view(legacy, dimension), _dimension_view(native, dimension), dimension,
            tolerance_policy=tolerance_policy)
        if not result["agree"]:
            mismatches.append({
                "row_key": key,
                "dimension": dimension,
                "finding_fields": list(result["finding_fields"]),
                "receipt": result["receipt"],
            })
    return mismatches


def _refuse_empty_inputs(legacy_rows: Mapping[str, Any], native_rows: Mapping[str, Any],
                         dimensions: tuple[str, ...]) -> None:
    """Fail closed: an empty comparison must never report ``compared``."""
    for dimension in dimensions:
        _dimension_fields(dimension)
    if not dimensions:
        raise fail("VALIDATION_FAILED", "native parity report has no dimensions")
    if not legacy_rows:
        raise fail("VALIDATION_FAILED", "native parity report has no legacy rows")
    if not native_rows:
        raise fail("VALIDATION_FAILED", "native parity report has no native rows")


def compare_native_vs_legacy(
    legacy_rows: dict[str, dict],
    native_rows: dict[str, dict],
    dimensions: tuple[str, ...],
    *,
    tolerance_policy: TolerancePolicy = SCORE_RECORD_V1,
) -> dict:
    """Classify every row key against the checker's per-dimension comparator.

    Every key present in EITHER side is visited, never only the intersection:
    a key present on only one side is itself a finding (``only_legacy``/
    ``only_native``), never silently skipped.  A shared key is ``compared``,
    and each ``dimensions`` entry that does not agree becomes one entry in
    ``mismatches`` carrying the checker's own finding fields and receipt.
    ``dimensions`` entries outside the checker's numeric field groups are
    refused up front with ``INVALID_REQUEST``.  Empty ``dimensions``, an empty side, or no shared key at all is refused with ``VALIDATION_FAILED``: a report that compared nothing never claims ``"compared"``.

    ``tolerance_policy`` is the ONE config policy every numeric dimension's
    comparison reads (never a per-dimension or per-field override threaded
    in some other way): it defaults to ``engine.v2.parity.tolerance.SCORE_RECORD_V1``,
    which declares zero rules, so the default is exact for every field and
    this parameter's addition changes no existing caller's behavior. A
    caller that has a user-approved, per-field ``TolerancePolicy`` for the
    native-vs-legacy comparison specifically (never for the Phase 4
    checker's own tier-0-style exact check, which does not pass this
    argument) supplies it here; this module invents no field's tolerance
    value itself.

    G2: this function classifies and returns; it has no path that writes a
    native value into a legacy row or the reverse, and a mismatch never raises.
    """
    _refuse_empty_inputs(legacy_rows, native_rows, dimensions)
    compared: list[str] = []
    only_legacy: list[str] = []
    only_native: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(legacy_rows) | set(native_rows)):
        legacy = legacy_rows.get(key)
        native = native_rows.get(key)
        if legacy is None:
            only_native.append(key)
        elif native is None:
            only_legacy.append(key)
        else:
            compared.append(key)
            mismatches.extend(_row_mismatches(key, legacy, native, dimensions, tolerance_policy))
    if not compared:
        raise fail("VALIDATION_FAILED", "native parity report shares no row key",
                   details={"only_legacy": len(only_legacy), "only_native": len(only_native)})
    return {
        "schema_version": SCHEMA_VERSION,
        "tolerance_policy_id": tolerance_policy.policy_id,
        "compared": compared,
        "only_legacy": only_legacy,
        "only_native": only_native,
        "mismatches": mismatches,
    }


def write_parity_report(report: dict, path: Path | str) -> Path:
    """Write ``report`` as deterministic JSON; any filesystem error propagates.

    The caller names the path explicitly (the nightly plan's private root,
    alongside ``run_shadow_nightly``'s own ``receipt_path`` convention) --
    never ``ledger/`` and never the legacy board's output directory.
    """
    path = Path(path)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return path


def native_parity_handler(
    plan: Mapping[str, Any],
    *,
    legacy_rows: dict[str, dict],
    native_rows: dict[str, dict],
    report_path: Path | str,
    dimensions: tuple[str, ...] = PARITY_DIMENSIONS,
    tolerance_policy: TolerancePolicy = SCORE_RECORD_V1,
) -> Callable[[dict], dict]:
    """Build the ``nightly.GRAPH`` handler for the ``native_parity`` stage.

    The plan's explicit ``shadow_serving_scorer`` (G5) decides whether there
    is anything to compare: ``"native"`` compares, writes the report and
    returns ``{"status": "compared"}``; ``"legacy"`` has no native rows and
    returns ``{"status": "not_applicable"}`` without writing.  ``compare``/
    ``write`` failures propagate: ``nightly._run_stage`` already degrades an
    OPTIONAL stage rather than blocking the board.

    ``tolerance_policy`` defaults to ``SCORE_RECORD_V1`` (exact for every
    field, unchanged from before this parameter existed) and is passed
    straight through to :func:`compare_native_vs_legacy` -- see that
    function's docstring for what a caller may plug in here.
    """
    path = Path(report_path)

    def handler(value: dict) -> dict:
        if native_shadow_serving_mode(plan) != "native":
            return {**value, "native_parity": {"status": "not_applicable"}}
        report = compare_native_vs_legacy(
            legacy_rows, native_rows, dimensions, tolerance_policy=tolerance_policy)
        write_parity_report(report, path)
        return {**value, "native_parity": {
            "status": "compared",
            "report_path": str(path),
            "compared": len(report["compared"]),
            "only_legacy": len(report["only_legacy"]),
            "only_native": len(report["only_native"]),
            "mismatches": len(report["mismatches"]),
        }}

    return handler
