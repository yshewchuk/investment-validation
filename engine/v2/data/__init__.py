"""Ingestion fetch receipts, normalization, coverage, finality, source revisions

Layer 1 of `system_rearchitecture.md` §4.1. Replaces `data/sources/`, `data/normalize/`, `data/pulls/`, `store.py`, `fetch.py`, `throttle.py`, `finality.py`, `rebuild.py`, `calendar sourcing from calendar.py`.

Rearchitecture phase 2 slice P2-1a writes the first production code here:
``documents`` performs the strict document decoding for data contracts that
``engine.v2.foundation.typed`` cannot express from annotations alone (hash,
timestamp and date formats; table-contract and query structural rules). See
``README.md`` for the public interface.
"""
from __future__ import annotations

from engine.v2.data.documents import decode_document, loads_document

__all__ = ["decode_document", "loads_document"]
