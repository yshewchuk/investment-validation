"""Ingestion fetch receipts, normalization, coverage, finality, source revisions

Layer 1 of `system_rearchitecture.md` §4.1. Replaces `data/sources/`, `data/normalize/`, `data/pulls/`, `store.py`, `fetch.py`, `throttle.py`, `finality.py`, `rebuild.py`, `calendar sourcing from calendar.py`.

Rearchitecture phase 2 slice P2-1a wrote the first production code here:
``documents`` performs the strict document decoding for data contracts that
``engine.v2.foundation.typed`` cannot express from annotations alone (hash,
timestamp and date formats; table-contract and query structural rules).
Slice P2-1b adds ``manifests.table_contract_hash`` (a ``TableContract``'s
``definition_hash``) and ``legacy_adapter.build_legacy_mapping`` (the
versioned legacy->v2 ``TableContract`` mapping for the six Tier-2 tables,
the feature panel, and Tier-4 forecasts, phase-2 guide §5.1, §12 D02). See
``README.md`` for the public interface.
"""
from __future__ import annotations

from engine.v2.data.documents import decode_document, loads_document
from engine.v2.data.legacy_adapter import build_legacy_mapping
from engine.v2.data.manifests import table_contract_hash

__all__ = ["build_legacy_mapping", "decode_document", "loads_document", "table_contract_hash"]
