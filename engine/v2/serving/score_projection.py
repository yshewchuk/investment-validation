"""Compatibility serialization of canonical score records for Phase 6."""
from __future__ import annotations

from typing import Any

from engine.v2.contracts import ScoreRecord

__all__ = ["legacy_score_projection"]


def legacy_score_projection(record: ScoreRecord) -> dict[str, Any]:
    """Project owned values without recomputing economics in the renderer."""
    return {
        "score_id": record.score_id,
        "request_hash": record.request_hash,
        "status": record.validation_status,
        "flags": list(record.reason_codes),
        "legs": list(record.legs),
        "entry_exit_plan": dict(record.entry_exit_plan),
        "quote_provenance": dict(record.quote_provenance),
        "forecasts": dict(record.forecasts),
        "financial_diagnostics": dict(record.financial_diagnostics),
        "chooser_selection": record.chooser_selection,
    }
