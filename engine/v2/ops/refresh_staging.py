"""Attempt-scoped staging of typed, non-artifact refresh input documents.

An ``incremental_refresh`` worker reads ``incremental_refresh_input.json``
from its own staging directory, but that document is neither a provider
artifact nor a predecessor job's output: it is the admitted job's deployment
identity, which only the executor is positioned to write before the worker
starts. This module owns that one write, per refresh kind, from
``claim.spec.parameters`` -- never the environment and never a legacy path.
Acquired data (raw payloads, revisions, coverage) stays the worker callback's
own output, appended to the same document in place.
"""
from __future__ import annotations

from pathlib import Path

from engine.v2.foundation import canonical_json, from_document
from engine.v2.ops.incremental_data import RefreshParameters

#: One entry per refresh kind that needs a staged identity document. Slice 4's
#: ``computed_moves_refresh``/``forward_calendar_refresh`` extend this registry
#: with their own document names; no new staging mechanism is needed.
REFRESH_INPUT_DOCUMENT_NAMES = {
    "incremental_refresh": "incremental_refresh_input.json",
}


def stage_refresh_input(claim, staging: Path) -> None:
    """Write the identity document ``claim``'s worker reads, if its kind has one.

    Every value comes from the admitted, hashed ``JobSpec`` parameters;
    ``fetch_root`` is attempt-scoped under this attempt's own ``staging``, so a
    provider-response cache never collides across attempts or leaks a mutable
    legacy root into a reproducible attempt.
    """
    name = REFRESH_INPUT_DOCUMENT_NAMES.get(claim.spec.kind)
    if name is None:
        return
    params = from_document(RefreshParameters, claim.spec.parameters)
    document = {
        "catalog_path": params.catalog_path,
        "objects_root": params.objects_root,
        "scope": params.scope,
        "expected_head_generation": params.expected_head_generation,
        "expected_head_snapshot_id": params.expected_head_snapshot_id,
        "table_name": params.table_name,
        "fetch_root": str(staging / "fetch"),
    }
    (staging / name).write_text(canonical_json(document))
