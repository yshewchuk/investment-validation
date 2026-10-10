"""Legacy-nightly readiness for a session (slice 3 of #564).

Contract in ``engine/v2/ops/ARCHITECTURE.md`` ("Nightly legacy readiness"). A leaf: not yet
called by ``nightly_trigger``. It only reads the legacy nightly's JSON run reports; there is
no rebuild step and no rebuild fallback.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from engine.v2.ops.errors import fail

__all__ = ["LegacyReadiness", "check_legacy_report"]

REPORT_NAME = "nightly_{}.json"
CANDIDATE_DAYS = 5  # requested dates D .. D+4


@dataclass(frozen=True)
class LegacyReadiness:
    report_name: str
    requested_as_of: str
    resolved_as_of: str


def _load(path: Path) -> dict | None:
    """The parsed report, ``None`` only when the file does not exist."""
    try:
        doc = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise fail("INTEGRITY_FAILED", "a legacy nightly report is unreadable") from exc
    session = doc.get("resolved_as_of") or doc.get("as_of") if isinstance(doc, dict) else None
    if not isinstance(session, str) or not isinstance(doc.get("steps"), dict):
        raise fail("INTEGRITY_FAILED", "a legacy nightly report has an unexpected shape")
    return doc


def _session(doc: dict) -> str:
    return doc.get("resolved_as_of") or doc["as_of"]


def _require_ready(doc: dict, as_of: str) -> None:
    if doc.get("stopped"):
        raise fail("DEPENDENCY_FAILED", "the legacy nightly stopped before completing",
                   details={"step": str(doc["stopped"])[:80]})
    finality = doc.get("finality")
    if not (isinstance(finality, dict) and finality.get("is_final") is True
            and str(finality.get("date")) == as_of):
        raise fail("SOURCE_NOT_FINAL", "the legacy nightly did not record the session as final")
    tiers = doc["steps"].get("tiers")
    if not isinstance(tiers, dict) or tiers.get("degraded") or "error" in tiers:
        raise fail("DEPENDENCY_FAILED", "the legacy tier step did not complete",
                   details={"step": "tiers"})


def check_legacy_report(reports_dir: Path, as_of: str) -> LegacyReadiness:
    """Verify the legacy nightly for session ``as_of`` completed; raise otherwise.

    File names only locate candidates; the report's own content decides."""
    try:
        day = date.fromisoformat(as_of)
    except (TypeError, ValueError):
        raise fail("INVALID_REQUEST", "as_of must be an ISO date") from None
    if day.isoformat() != as_of:  # compact forms parse but would never equal a report's text
        raise fail("INVALID_REQUEST", "as_of must be a YYYY-MM-DD date")
    matches = []
    for offset in range(CANDIDATE_DAYS):
        requested = (day + timedelta(days=offset)).isoformat()
        path = Path(reports_dir) / REPORT_NAME.format(requested)
        doc = _load(path)
        if doc is not None and _session(doc) == as_of:
            matches.append((path, doc, requested))
    if not matches:
        raise fail("SOURCE_NOT_FOUND", "no legacy nightly report resolves to this session")
    path, doc, requested = matches[-1]  # the latest requested date is the latest run
    _require_ready(doc, as_of)
    return LegacyReadiness(path.name, requested, as_of)