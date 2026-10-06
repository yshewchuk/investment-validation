"""Attempt-scoped staging of typed, non-artifact refresh input documents.

An ``incremental_refresh`` worker reads ``incremental_refresh_input.json``
from its own staging directory, but that document is neither a provider
artifact nor a predecessor job's output: it is the admitted job's deployment
identity, which only the executor is positioned to write before the worker
starts. This module owns that one write, per refresh kind, from
``claim.spec.parameters`` -- never the environment and never a legacy path.
Acquired data (raw payloads, revisions, coverage) stays the worker callback's
own output, appended to the same document in place.

S4C Part 3: ``computed_moves_refresh`` gets its own staged document too --
``computed_moves_store._input_document``/``_validate_input_document`` already
read and cross-validate one, including the ``attempt_id``/``fence`` pair its
own commit's fence check needs (``engine.v2.ops.lifecycle.verify_fence``, the
real production signature -- no ``check_lease_time`` parameter to disable the
wall-clock lease-expiry check with). ``forward_calendar_refresh`` gets its own
staged document too, but a much smaller one than ``computed_moves_refresh``'s:
only ``attempt_id``/``fence``, which vary per attempt and so cannot live on
the job's own immutable ``CalendarMovesParameters`` -- every other value
``run_forward_calendar_refresh`` needs already lives there, so restating it
here could only drift, never add anything
(``engine/v2/ops/ARCHITECTURE.md`` "Inputs").
"""
from __future__ import annotations

from pathlib import Path

from engine.v2.foundation import canonical_json, from_document
from engine.v2.ops.incremental_data import RefreshParameters

#: One entry per refresh kind that needs a staged identity document.
REFRESH_INPUT_DOCUMENT_NAMES = {
    "incremental_refresh": "incremental_refresh_input.json",
    "computed_moves_refresh": "computed_moves_refresh_input.json",
    "forward_calendar_refresh": "forward_calendar_refresh_input.json",
}


def stage_refresh_input(claim, staging: Path) -> None:
    """Write the identity document ``claim``'s worker reads, if its kind has one.

    Every value comes from the admitted, hashed ``JobSpec`` parameters;
    ``fetch_root`` is attempt-scoped under this attempt's own ``staging``, so a
    provider-response cache never collides across attempts or leaks a mutable
    legacy root into a reproducible attempt.
    """
    kind = claim.spec.kind
    name = REFRESH_INPUT_DOCUMENT_NAMES.get(kind)
    if name is None:
        return
    if kind == "incremental_refresh":
        document = _daily_document(claim, staging)
    elif kind == "computed_moves_refresh":
        document = _computed_moves_document(claim)
    else:
        document = _forward_calendar_document(claim)
    (staging / name).write_text(canonical_json(document))


def _daily_document(claim, staging: Path) -> dict:
    params = from_document(RefreshParameters, claim.spec.parameters)
    return {
        "catalog_path": params.catalog_path,
        "objects_root": params.objects_root,
        "scope": params.scope,
        "expected_head_generation": params.expected_head_generation,
        "expected_head_snapshot_id": params.expected_head_snapshot_id,
        "table_name": params.table_name,
        "fetch_root": str(staging / "fetch"),
    }


def _computed_moves_document(claim) -> dict:
    """The ``computed_moves_refresh`` identity document, in exactly the shape
    ``computed_moves_store._ALLOWED_DOCUMENT_KEYS`` accepts -- an extra key
    (e.g. ``horizon_days``/``tickers``, which that store never reads) would
    make that store's own ``_validate_document_identity`` refuse the document
    outright as carrying an unknown key.
    """
    from engine.v2.ops.calendar_moves_jobs import CalendarMovesParameters

    params = from_document(CalendarMovesParameters, claim.spec.parameters)
    return {
        "catalog_path": params.catalog_path,
        "objects_root": params.objects_root,
        "scope": params.scope,
        "expected_head_generation": params.expected_head_generation,
        "expected_head_snapshot_id": params.expected_head_snapshot_id,
        "parent_receipt_id": params.parent_receipt_id,
        "as_of": params.as_of,
        "all_scoreable": params.all_scoreable,
        "since": params.since,
        "attempt_id": claim.attempt_id,
        "fence": claim.fence,
    }


def _forward_calendar_document(claim) -> dict:
    """The ``forward_calendar_refresh`` identity document.

    It carries ONLY the two per-attempt values
    ``run_forward_calendar_refresh`` cannot get from the job's own immutable
    ``CalendarMovesParameters`` -- ``attempt_id``/``fence`` vary per attempt
    (a retried attempt gets a new fence), so they must be written fresh at
    claim time, the same as ``_computed_moves_document`` writes them; every
    other value this runner needs is read directly off the decoded
    ``CalendarMovesParameters`` by
    ``incremental_data._load_forward_calendar_refresh_callback``.
    """
    return {"attempt_id": claim.attempt_id, "fence": claim.fence}
