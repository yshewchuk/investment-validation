"""Deterministic ``TableContract`` identity: ``table_contract_hash``.

Phase-2 guide §5.1: "``definition_hash`` is computed by the registration
function from canonical definition content excluding ``definition_hash``; the
dataclass does not compute it." This module is that one place for the
``TableContract`` case. ``engine/v2/data/legacy_adapter.py`` is its only
caller in this slice (P2-1b); dataset/snapshot manifest construction
(``DatasetManifest``, ``SnapshotRef``) is a later Phase 2 milestone (P2-2/P2-3,
phase-2 guide §4) and does not belong in this module yet.

Layer 1 of ``system_rearchitecture.md`` §4.1: imports only
``engine.v2.contracts`` and ``engine.v2.foundation``, matching
``engine/v2/data/documents.py`` and the phase-2 guide §3.3 rule that the data
package never imports ``engine.v2.ops``.
"""
from __future__ import annotations

from engine.v2.contracts.data import TableContract
from engine.v2.foundation import content_hash, to_document

__all__ = ["table_contract_hash"]


def table_contract_hash(contract: TableContract) -> str:
    """``sha256:...`` over ``contract``'s canonical payload, excluding its own hash.

    Column order, key order, and every other field are meaningful (phase-2
    guide §5.1): this hashes ``to_document(contract)`` verbatim except for the
    ``definition_hash`` field itself, so reordering two columns or changing one
    column's unit changes the result, and two calls over the same content are
    byte-identical.
    """
    payload = to_document(contract)
    del payload["definition_hash"]
    return content_hash(payload)
