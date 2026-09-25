"""Native strategy-derivation document for the operations server (P6-4).

``derivation_document`` is the pure body ``GET /derivation.json`` serves and
``GET /derivation``'s page fetches client-side: the frozen Phase 4 registry's
own ``StrategySpec`` records (``engine.v2.registry.strategies.default_registry``)
projected with ``engine.v2.foundation.to_document``, never a hand-written
subset of fields -- a field added to ``StrategySpec`` appears here without a
second edit. No ``strategy_id`` returns every strategy sorted by id, so the
document is deterministic; an unknown id is an explicit
``UNKNOWN_STRATEGY_ID`` refusal, never an empty success.
"""
from __future__ import annotations

from http import HTTPStatus

from engine.v2.foundation import to_document
from engine.v2.registry.strategies import default_registry

__all__ = ["DERIVATION_SCHEMA_V1", "UNKNOWN_STRATEGY_ID", "derivation_document"]

DERIVATION_SCHEMA_V1 = "strategy_derivation.v1"
UNKNOWN_STRATEGY_ID = "UNKNOWN_STRATEGY_ID"


def derivation_document(strategy_id: str | None) -> tuple[HTTPStatus, dict]:
    """All strategies sorted by id, or one strategy's document, or a refusal."""
    registry = default_registry()
    if strategy_id is None:
        return HTTPStatus.OK, {
            "schema_version": DERIVATION_SCHEMA_V1,
            "status": "available",
            "strategies": [to_document(spec) for spec in
                           sorted(registry.strategies, key=lambda spec: spec.strategy_id)],
        }
    try:
        spec = registry.strategy(strategy_id)
    except KeyError:
        return HTTPStatus.NOT_FOUND, {
            "schema_version": DERIVATION_SCHEMA_V1,
            "status": "refused",
            "reason_code": UNKNOWN_STRATEGY_ID,
        }
    return HTTPStatus.OK, to_document(spec)
