"""Legacy ``ScoreRequest`` document -> canonical V2 ``ScoreRequest``.

A tier-0 pair keeps the LEGACY request in ``payload.request``: it is what
``tools/replay_tier1.py`` rebuilds, what coverage and pinned links hash, and
what the seeded controls read. A strict Phase 4 trace carries the canonical V2
request only in ``payload.input_trace.request``. This module is the one
translation between them, shared by the capture tool (which builds the V2
request) and ``checks/phase4_real.py`` (which verifies that the traced V2
request is the translation of the saved legacy request, so a trace cannot be
re-pointed at a different case).

It reads the saved JSON document, not the legacy dataclass, so the checks can
use it without importing ``engine.score``.
"""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from engine.v2.contracts import ScoreRequest as V2ScoreRequest
from engine.v2.foundation import content_hash, to_document

#: The V2 request fields that are a function of the legacy request (plus the
#: event id and snapshot the V2 request names). Deployment, model artifact and
#: state refs, and the decision clock are capture-environment bindings checked
#: elsewhere (resources, frozen replay), not translations of the legacy request.
LEGACY_BOUND_FIELDS = (
    "strategy_version", "event_revision", "calendar_revision",
    "requested_decision_at", "mode", "fill_model", "geometry_override",
)


class LegacyRequestTranslationError(ValueError):
    pass


def _date(value: Any) -> str | None:
    if value is None:
        return None
    return str(pd.Timestamp(value).date())


def canonical_request_from_legacy(
    legacy: Mapping[str, Any], *, event_id: str, snapshot: str,
) -> V2ScoreRequest:
    """The canonical V2 command for one saved legacy request document."""
    if not isinstance(event_id, str) or not event_id.strip():
        raise LegacyRequestTranslationError("strict trace requires the captured event_id")
    strategy = legacy.get("strategy")
    if not isinstance(strategy, str) or not strategy:
        raise LegacyRequestTranslationError("legacy request has no strategy")
    decision = _date(legacy.get("as_of")
                     if legacy.get("as_of") is not None else legacy.get("chain_as_of"))
    if decision is None:
        raise LegacyRequestTranslationError("strict trace requires a decision date")
    event_date = _date(legacy.get("event_date"))
    if event_date is None:
        raise LegacyRequestTranslationError("strict trace requires an event date")
    event_identity = {
        "event_id": event_id,
        "ticker": legacy.get("ticker"),
        "event_date": event_date,
        "session": legacy.get("session"),
    }
    geometry_override = dict(legacy.get("structure_params") or {})
    if legacy.get("strike") is not None:
        geometry_override["strike"] = float(legacy["strike"])
    if legacy.get("expiry") is not None:
        geometry_override["expiry"] = _date(legacy["expiry"])
    fill = legacy.get("fill")
    alpha = fill.get("alpha") if isinstance(fill, Mapping) else getattr(fill, "alpha", None)
    if alpha is None:
        raise LegacyRequestTranslationError("legacy request has no fill alpha")
    offset = legacy.get("decision_offset")
    return V2ScoreRequest(
        event_id=event_id,
        event_revision="event:" + content_hash(event_identity),
        calendar_revision="calendar:" + content_hash({
            "decision": decision,
            "event": event_identity,
        }),
        strategy_version=strategy,
        deployment_id=f"legacy-capture:{snapshot}",
        decision_clock_id="legacy.decision_offset." + str(offset if offset is not None else 0),
        requested_decision_at=decision,
        snapshot_id=str(snapshot),
        mode="replay",
        fill_model={"policy_id": "legacy.fill_alpha.v1", "alpha": float(alpha)},
        geometry_override=geometry_override or None,
    )


def legacy_binding_mismatches(
    legacy: Mapping[str, Any], native_request: Mapping[str, Any],
) -> list[str]:
    """``LEGACY_BOUND_FIELDS`` where the V2 request is not the translation of
    ``legacy``; empty when it is. An untranslatable legacy request is one
    mismatch named ``legacy_request``."""
    try:
        expected = to_document(canonical_request_from_legacy(
            legacy,
            event_id=native_request.get("event_id"),
            snapshot=native_request.get("snapshot_id"),
        ))
    except (LegacyRequestTranslationError, TypeError, ValueError):
        return ["legacy_request"]
    return [name for name in LEGACY_BOUND_FIELDS
            if expected.get(name) != native_request.get(name)]
