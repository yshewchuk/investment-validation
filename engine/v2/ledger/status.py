"""Native v2 status: summarize catalog decisions/outcomes the way legacy
``engine/ledger.py::status`` summarizes the jsonl ledger (P6-3
``decisions-status``).

The core counts (``predictions``, ``outcomes``, ``resolved``,
``unresolvable``, ``calibration_due``, ``n_scored``, ``n_at_last_report``,
``settlement_diagnostics``) mirror ``engine.ledger.status()`` field-for-field
and number-for-number when fed the same rows, through the
:mod:`engine.v2.ledger.calibration`/:mod:`engine.v2.ledger.legacy_adapter`
reuse of legacy's own pure ``scored_pairs``/``settlement_summary``.

``duplicates`` and ``pending_settlement`` are v2-native additions beyond
``engine.ledger.status()``'s own shape (which has neither): a lightweight
(ticker, strategy, event_date) restatement count, and a count of predictions
whose event has passed with no resolved outcome yet — never claimed as
byte-identical to legacy's structure-aware ``duplication_report``/
``_unresolved``, only as a status-summary approximation of the same idea.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import calibration, catalog_reader, legacy_adapter

__all__ = ["status"]


def _duplicate_counts(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    per: dict[tuple[str, str, str], int] = {}
    for row in predictions:
        if not row.get("event_date"):
            continue
        key = (str(row.get("ticker")), str(row.get("strategy")), str(row.get("event_date")))
        per[key] = per.get(key, 0) + 1
    return {
        "prediction_rows": len(predictions),
        "distinct_trades": len(per),
        "rows_per_trade_max": max(per.values()) if per else 0,
    }


def _pending_settlement(predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]], *,
                        through=None) -> int:
    through = pd.Timestamp(through).normalize() if through is not None else pd.Timestamp.today().normalize()
    latest_status: dict[str, str] = {}
    for outcome in outcomes:  # catalog sequence order is chronological
        latest_status[outcome["row_id"]] = str(outcome.get("status") or "")
    count = 0
    for row in predictions:
        if latest_status.get(row["row_id"]) == "resolved" or not row.get("event_date"):
            continue
        if pd.Timestamp(row["event_date"]) <= through:
            count += 1
    return count


def status(conn, *, trigger: int = calibration.CALIBRATION_TRIGGER, through=None) -> dict:
    """Counts, duplicates and pending settlement over catalog decisions."""
    predictions = catalog_reader.read_predictions(conn)
    predictions_all = catalog_reader.read_predictions(conn, resolve_supersedes=False)
    outcomes = catalog_reader.read_outcomes(conn)
    resolved = [o for o in outcomes if o.get("status") == "resolved"]
    due, n_now, last = calibration.calibration_due(conn, trigger=trigger)
    pairs = legacy_adapter.scored_pairs(predictions=predictions, outcomes=outcomes)
    return {
        "predictions": len(predictions),
        "prediction_rows_total": len(predictions_all),
        "duplicates": _duplicate_counts(predictions),
        "outcomes": len(outcomes),
        "resolved": len(resolved),
        "unresolvable": len(outcomes) - len(resolved),
        "pending_settlement": _pending_settlement(predictions, outcomes, through=through),
        "calibration_due": due,
        "n_scored": n_now,
        "n_at_last_report": last,
        "settlement_diagnostics": legacy_adapter.settlement_summary(pairs),
        "health": calibration.health_ref(conn),
    }
