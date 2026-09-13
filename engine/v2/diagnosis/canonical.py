"""Canonical JSON and content hashing — re-exported from ``engine.v2.foundation``.

Phase 0 implemented RFC 8785 here because diagnosis was the only v2 package
allowed code. Rearchitecture phase 1 §3.2 moved the implementation to
``engine/v2/foundation/canonical.py`` so production packages can hash without
importing diagnosis, which nothing may import.

These are the SAME function objects, not a second canonicalizer: diagnosis may
depend on foundation (it may read every layer), so the corpus, the receipts and
the baseline keep hashing through exactly one implementation.
``tests/test_v2_ops_foundation.py`` asserts the identity and pins the hashes
the phase-0 copy produced before the move.
"""
from __future__ import annotations

from engine.v2.foundation import CONTENT_HASH_PREFIX, canonical_json, content_hash

__all__ = ["canonical_json", "content_hash", "CONTENT_HASH_PREFIX"]
