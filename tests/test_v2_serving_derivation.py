"""Phase 6 slice 9: native strategy derivation over the frozen registry (P6-4).

``derivation_document`` is the whole body ``GET /derivation.json`` serves, so
these are the pure-document controls: every strategy sorted by id compared
against ``default_registry()`` itself (never a hardcoded count), one
strategy's document equal to ``to_document`` of its own ``StrategySpec``, and
an unknown id refused with a named reason code rather than falling back to
the full list. HTTP reachability through the preview launcher is covered by
``tests/test_v2_dashboard_preview_derivation.py``.
"""
from __future__ import annotations

from http import HTTPStatus

from engine.v2.foundation import to_document
from engine.v2.registry.strategies import default_registry
from engine.v2.serving.derivation_projection import derivation_document


def test_derivation_document_all_strategies_sorted():
    status, body = derivation_document(None)

    assert status == HTTPStatus.OK
    assert body["schema_version"] == "strategy_derivation.v1"
    ids = [strategy["strategy_id"] for strategy in body["strategies"]]
    expected = sorted(spec.strategy_id for spec in default_registry().strategies)
    assert ids
    assert ids == expected


def test_derivation_document_one_strategy():
    spec = next(s for s in default_registry().strategies if s.strategy_id == "STR-THRU")
    status, document = derivation_document(spec.strategy_id)

    assert status == HTTPStatus.OK
    assert document == to_document(spec)
    assert document["gate_recipe"] == spec.gate_recipe
    assert document["entry_policy"] == spec.entry_policy


def test_derivation_document_unknown_id_refuses():
    status, body = derivation_document("nope-999")

    assert status == HTTPStatus.NOT_FOUND
    assert body["status"] == "refused"
    assert body["reason_code"] == "UNKNOWN_STRATEGY_ID"
    assert "strategies" not in body  # never a silent fallback to the full list
