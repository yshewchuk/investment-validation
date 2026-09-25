"""Attempt-scoped staging of typed, non-artifact refresh input documents.

An ``incremental_refresh`` worker reads ``incremental_refresh_input.json``
from its own staging directory, but that document is neither a provider
artifact nor a predecessor job's output: it is the admitted job's deployment
identity, which only the executor is positioned to write before the worker
starts. This module owns that one write, per refresh kind, from
``claim.spec.parameters`` -- never the environment and never a legacy path.
Acquired data (raw payloads, revisions, coverage) stays the worker callback's
own output, appended to the same document in place.

S4C: ``computed_moves_refresh``/``forward_calendar_refresh`` are their own
kinds with their own parameter contract and document names (never a table_name
variant of ``incremental_refresh``); their staged document additionally
carries the horizon/target identity the stores resolve from.
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

    Every value comes from the admitted, hashed ``JobSpec`` parameters.
    """
    kind = claim.spec.kind
    name = REFRESH_INPUT_DOCUMENT_NAMES.get(kind)
    if name is None:
        return
    document = (_daily_document(claim) if kind == "incremental_refresh"
                else _calendar_moves_document(claim))
    (staging / name).write_text(canonical_json(document))


def _daily_document(claim) -> dict:
    params = from_document(RefreshParameters, claim.spec.parameters)
    return {
        "catalog_path": params.catalog_path,
        "objects_root": params.objects_root,
        "scope": params.scope,
        "expected_head_generation": params.expected_head_generation,
        "expected_head_snapshot_id": params.expected_head_snapshot_id,
        "table_name": params.table_name,
    }


def _calendar_moves_document(claim) -> dict:
    from engine.v2.ops.calendar_moves_jobs import CalendarMovesParameters

    params = from_document(CalendarMovesParameters, claim.spec.parameters)
    return {
        "catalog_path": params.catalog_path,
        "objects_root": params.objects_root,
        "scope": params.scope,
        "expected_head_generation": params.expected_head_generation,
        "expected_head_snapshot_id": params.expected_head_snapshot_id,
        "table_name": params.table_name,
        "as_of": params.as_of,
        "horizon_days": params.horizon_days,
        "tickers": list(params.tickers),
        "all_scoreable": params.all_scoreable,
        "since": params.since,
    }
