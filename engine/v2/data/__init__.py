"""Ingestion fetch receipts, normalization, coverage, finality, source revisions

Layer 1 of `system_rearchitecture.md` §4.1. Replaces `data/sources/`, `data/normalize/`, `data/pulls/`, `store.py`, `fetch.py`, `throttle.py`, `finality.py`, `rebuild.py`, `calendar sourcing from calendar.py`.

Rearchitecture phase 2 slice P2-1a wrote the first production code here:
``documents`` performs the strict document decoding for data contracts that
``engine.v2.foundation.typed`` cannot express from annotations alone (hash,
timestamp and date formats; table-contract and query structural rules).
Slice P2-1b adds ``manifests.table_contract_hash`` (a ``TableContract``'s
``definition_hash``) and ``legacy_mapping.build_legacy_mapping`` (the
versioned legacy->v2 ``TableContract`` mapping for the six Tier-2 tables,
the feature panel, and Tier-4 forecasts, phase-2 guide §5.1, §12 D02 — moved
here from ``legacy_adapter.py`` in review round 3, item 1, which kept only
that module's legacy-touching code). Slice P2-2c adds ``manifests``'
fragment/dataset-version/snapshot builders and verifiers (phase-2 guide
§5.2, §11, §12 D03). See ``README.md`` for the public interface.
"""
from __future__ import annotations

from engine.v2.data.documents import decode_document, loads_document
from engine.v2.data.eod_inventory import (
    CURRENT_EOD_INVENTORY,
    EODDataset,
    inventory_by_name,
    inventory_document,
    validate_inventory,
)
from engine.v2.data.event_revisions import (
    EventRevision,
    apply_event_revision,
    event_revision_candidate,
    resolve_event_identity,
)
from engine.v2.data.generic_incremental import (
    GenericTableCandidate,
    build_generic_table_candidate,
    commit_generic_table_candidate,
    load_generic_revisions,
)
from engine.v2.data.legacy_mapping import build_legacy_mapping
from engine.v2.data.manifests import (
    dataset_manifest,
    fragment_record,
    fragment_ref,
    snapshot_ref,
    table_contract_hash,
    verify_dataset_manifest,
    verify_fragment_record,
    verify_snapshot_ref,
)

__all__ = [
    "CURRENT_EOD_INVENTORY",
    "EODDataset",
    "EventRevision",
    "GenericTableCandidate",
    "apply_event_revision",
    "build_legacy_mapping",
    "build_generic_table_candidate",
    "commit_generic_table_candidate",
    "load_generic_revisions",
    "dataset_manifest",
    "decode_document",
    "event_revision_candidate",
    "fragment_record",
    "fragment_ref",
    "inventory_by_name",
    "inventory_document",
    "loads_document",
    "resolve_event_identity",
    "snapshot_ref",
    "table_contract_hash",
    "verify_dataset_manifest",
    "verify_fragment_record",
    "verify_snapshot_ref",
    "validate_inventory",
]
