"""P2-C03: the finality-resolved session, shared by every stage that needs it.

v1 (``engine/dashboard/nightly.py:1223-1247``) walks a requested ``as_of``
back to the newest FINAL session (``engine.data.finality.resolve_final_session``)
and uses the resolved date everywhere downstream: scoring, decisions,
settlement, render. v2's job graph keeps ``session`` in every job's
parameters as the REQUESTED date (job identity/idempotency must not depend on
data that has not been read yet); each stage that needs the effective date
instead reads the finality artifact its plan already binds and resolves it
here, the one place that decodes it.

Never trusts a finality document blindly: ``resolve_final_session`` only
ever returns a session at or before the one requested (it raises when none
qualifies), so a bound ``finality.json`` claiming otherwise, or claiming
``is_final`` when it is not literally ``True``, cannot be a genuine product of
that function and is refused rather than acted on.
"""
from __future__ import annotations

from engine.v2.ops.errors import fail

__all__ = ["resolve_effective_session", "walk_back_flag"]


def _date_only(value):
    """The first 10 characters of an ISO date/datetime string.

    Mirrors ``engine.v2.ops.decision_replay._date_only`` — every session
    value in this graph is either already ``YYYY-MM-DD`` or an ISO timestamp
    with that prefix, so lexical comparison of the prefix is chronological
    comparison.
    """
    text = str(value)
    return text[:10] if len(text) >= 10 else text


def resolve_effective_session(finality: dict, requested_session: str) -> str:
    """Return the session ``finality`` resolves to, or refuse.

    Refuses (``VALIDATION_FAILED``) when ``finality`` is malformed, is not
    marked ``is_final``, or names a date strictly AFTER the one requested —
    the one shape :func:`engine.data.finality.resolve_final_session` can
    never produce, so seeing it here means the bound artifact does not speak
    for a real walk-back.
    """
    if not isinstance(finality, dict):
        raise fail("VALIDATION_FAILED", "finality artifact is malformed")
    resolved = finality.get("date")
    if not resolved or finality.get("is_final") is not True:
        raise fail("VALIDATION_FAILED",
                   "finality is not resolved final for the requested session",
                   details={"resolved": resolved, "is_final": finality.get("is_final")})
    if _date_only(resolved) > _date_only(requested_session):
        raise fail("VALIDATION_FAILED",
                   "finality resolved to a session after the one requested",
                   details={"resolved": str(resolved), "requested": str(requested_session)})
    return str(resolved)


def walk_back_flag(requested_session: str, resolved_session: str, finality: dict) -> dict | None:
    """v1's ``as_of_resolved`` flag (``engine/dashboard/nightly.py:1239-1244``),
    or ``None`` when nothing walked back."""
    if _date_only(resolved_session) == _date_only(requested_session):
        return None
    detail = (f"requested {_date_only(requested_session)} resolved to final "
             f"{_date_only(resolved_session)}: {finality.get('detail')}")
    return {"kind": "as_of_resolved", "detail": detail}
