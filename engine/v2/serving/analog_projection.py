"""Native history-analogs document for the operations server (S9H).

``analog_document`` is the pure body ``GET /analogs.json`` serves and
``GET /analogs``'s page fetches client-side: for one ``(release_id, event_id)``
pair it reads the analog row ids the scoring stage already persisted on the
``serving_score_summary`` rows (``selected_row_ids`` /
``contributing_row_ids``) plus each row's own ``n_analogs`` count. Every
score/strategy of the event is returned, ordered deterministically by
strategy then ``score_id``, with an optional ``strategy`` filter -- a
``LIMIT 1`` here would silently pick one of several legitimate rows. No
financial value is recomputed here -- row ids and a count only, matching the
"no financial formulas in serving/UI" constraint by construction. An unknown
pair (or a filter matching no score) is an explicit ``EVENT_NOT_FOUND``
refusal, never an empty success.
"""
from __future__ import annotations

import json
import sqlite3
from http import HTTPStatus
from pathlib import Path
from typing import Any

__all__ = [
    "ANALOG_VIEW_SCHEMA_V1", "EVENT_NOT_FOUND", "SERVING_INDEX_MISSING",
    "SERVING_INDEX_OUTDATED", "SERVING_INDEX_UNREADABLE", "ServingIndexError",
    "analog_document", "index_is_current", "index_refusal", "open_read_only",
]

ANALOG_VIEW_SCHEMA_V1 = "history_analogs_view.v1"
EVENT_NOT_FOUND = "EVENT_NOT_FOUND"
SERVING_INDEX_MISSING = "SERVING_INDEX_MISSING"
SERVING_INDEX_OUTDATED = "SERVING_INDEX_OUTDATED"
SERVING_INDEX_UNREADABLE = "SERVING_INDEX_UNREADABLE"

#: Migration 3's analog columns on ``serving_score_summary``; a read-only
#: serving index lacking any of them has not been migrated yet.
_ANALOG_COLUMNS = frozenset({"selected_row_ids", "contributing_row_ids", "n_analogs"})


class ServingIndexError(Exception):
    """A read-only serving-index read that cannot proceed; typed ``reason_code``."""

    def __init__(self, reason_code: str) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code


def open_read_only(path: Path | str) -> sqlite3.Connection:
    """Open the serving index read-only: no create, no ``ensure_schema``, no migration.

    ``projections.connect`` is the WRITE path (create + ``ensure_schema`` +
    migrations); a GET must never run it. ``mode=ro`` refuses a missing file
    and makes every statement read-only, so a read cannot add migration rows.
    """
    if not Path(path).is_file():
        raise ServingIndexError(SERVING_INDEX_MISSING)
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ServingIndexError(SERVING_INDEX_UNREADABLE) from exc
    conn.row_factory = sqlite3.Row
    return conn


def index_is_current(conn: sqlite3.Connection) -> bool:
    """True iff migration 3's analog columns are present on the summary table.

    A missing ``serving_score_summary`` (or one predating migration 3) reads
    as outdated, never as "no analogs for this event".
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(serving_score_summary)")}
    return _ANALOG_COLUMNS <= columns


def index_refusal(reason_code: str) -> tuple[HTTPStatus, dict]:
    """The typed 503 body for a serving index a read cannot use."""
    return HTTPStatus.SERVICE_UNAVAILABLE, {
        "schema_version": ANALOG_VIEW_SCHEMA_V1,
        "status": "unavailable",
        "reason_code": reason_code,
    }


def analog_document(conn, store, release_id: str, event_id: str,
                    strategy: str | None = None) -> tuple[HTTPStatus, dict]:
    """Every score/strategy's persisted analog row ids, or a refusal.

    ``store`` is accepted for interface symmetry with the other serving
    readers (``projections.get_score_detail``) and is deliberately not read:
    the row ids and the count live entirely on the summary rows, so the full
    ``detail_artifact_id`` object never has to be loaded. The connection is
    opened read-only by the route (no migration): an index that predates
    migration 3 simply has no rows to read and is refused by the route as
    ``SERVING_INDEX_OUTDATED`` before this is called.
    """
    sql = ("SELECT score_id, strategy, selected_row_ids, contributing_row_ids, n_analogs "
           "FROM serving_score_summary WHERE release_id = ? AND event_id = ?")
    params: list[Any] = [release_id, event_id]
    if strategy is not None:
        sql += " AND strategy = ?"
        params.append(strategy)
    rows = conn.execute(sql + " ORDER BY strategy, score_id", params).fetchall()
    if not rows:
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
        "analogs": [
            {
                "score_id": row["score_id"],
                "strategy": row["strategy"],
                "n_analogs": row["n_analogs"],
                "selected_row_ids": json.loads(row["selected_row_ids"] or "[]"),
                "contributing_row_ids": json.loads(row["contributing_row_ids"] or "[]"),
            }
            for row in rows
        ],
    }
