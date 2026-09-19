"""Read canonical predictions/outcomes from the ops catalog, jsonl-shaped.

Mirrors ``engine.ledger.read_predictions``/``read_outcomes`` exactly, except
the rows come from the catalog ``decisions`` table (``kind='prediction'`` /
``kind='outcome'``) instead of ``ledger/*.jsonl`` files. Every row's
``payload_json`` is the exact legacy row dict — verbatim for a
``ledger_history_import`` row, and the same shape for a live
``decision_commit``/settlement commit (P2 guide §5.5, §12) — so a caller that
feeds these lists into the SAME pure functions legacy uses
(``engine.ledger.canonical_predictions``/``scored_pairs``,
``engine.portfolio.build_book``, all now accepting optional
``predictions=``/``outcomes=`` row overrides — see
:mod:`engine.v2.ledger.legacy_adapter``) gets numbers that agree with the
jsonl ledger's by construction, whichever storage committed the decision.

``sequence`` is the catalog's ``AUTOINCREMENT`` insertion order, which is the
append-only, never-reordered analogue of "file order is chronological" that
``engine.ledger.read_predictions``/``_unresolved``/``scored_pairs`` rely on
for "the last row per id/row_id wins".
"""
from __future__ import annotations

import json
from typing import Any

__all__ = ["read_outcomes", "read_predictions"]


def read_predictions(conn, *, resolve_supersedes: bool = True) -> list[dict[str, Any]]:
    """Every committed prediction decision, or the resolved (non-superseded) view.

    Mirrors ``engine.ledger.read_predictions``: with ``resolve_supersedes``
    (the default) a row a later row supersedes is dropped from the returned
    view — never from the catalog, which is append-only and immutable
    (``engine.v2.ledger.decisions.insert`` refuses a same-``decision_id``
    content change outright).
    """
    rows = [json.loads(row["payload_json"]) for row in conn.execute(
        "SELECT payload_json FROM decisions WHERE kind='prediction' ORDER BY sequence")]
    if not resolve_supersedes:
        return rows
    superseded = {r["supersedes"] for r in rows if r.get("supersedes")}
    return [r for r in rows if r["row_id"] not in superseded]


def read_outcomes(conn) -> list[dict[str, Any]]:
    """Every committed outcome decision, in commit (chronological) order."""
    return [json.loads(row["payload_json"]) for row in conn.execute(
        "SELECT payload_json FROM decisions WHERE kind='outcome' ORDER BY sequence")]
