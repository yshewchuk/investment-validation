"""Native history-analogs document for the operations server (S9H).

``analog_document`` is the pure body ``GET /analogs.json`` serves and
``GET /analogs``'s page fetches client-side: for one ``(release_id, event_id)``
pair it reads the analog row ids the scoring stage already persisted on the
``serving_score_summary`` row (``selected_row_ids`` / ``contributing_row_ids``)
plus the row's own ``n_analogs`` count. No financial value is recomputed here
-- row ids and a count only, matching the "no financial formulas in serving/UI"
constraint by construction. An unknown pair is an explicit
``EVENT_NOT_FOUND`` refusal, never an empty success.
"""
from __future__ import annotations

import json
from http import HTTPStatus

__all__ = ["ANALOG_VIEW_SCHEMA_V1", "EVENT_NOT_FOUND", "analog_document"]

ANALOG_VIEW_SCHEMA_V1 = "history_analogs_view.v1"
EVENT_NOT_FOUND = "EVENT_NOT_FOUND"


def analog_document(conn, store, release_id: str, event_id: str) -> tuple[HTTPStatus, dict]:
    """One event's persisted analog row ids, or a refusal.

    ``store`` is accepted for interface symmetry with the other serving
    readers (``projections.get_score_detail``) and is deliberately not read:
    the row ids and the count live entirely on the summary row, so the full
    ``detail_artifact_id`` object never has to be loaded.
    """
    row = conn.execute(
        "SELECT selected_row_ids, contributing_row_ids, n_analogs "
        "FROM serving_score_summary WHERE release_id = ? AND event_id = ? "
        "ORDER BY score_id LIMIT 1",
        (release_id, event_id)).fetchone()
    if row is None:
        return HTTPStatus.NOT_FOUND, {
            "schema_version": ANALOG_VIEW_SCHEMA_V1,
            "status": "refused",
            "reason_code": EVENT_NOT_FOUND,
        }
    return HTTPStatus.OK, {
        "schema_version": ANALOG_VIEW_SCHEMA_V1,
        "status": "available",
        "release_id": release_id,
        "event_id": event_id,
        "n_analogs": row["n_analogs"],
        "selected_row_ids": json.loads(row["selected_row_ids"] or "[]"),
        "contributing_row_ids": json.loads(row["contributing_row_ids"] or "[]"),
    }
