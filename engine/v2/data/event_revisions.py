"""Pure identity rules for earnings-calendar revisions.

An event's date and BMO/AMC session are mutable facts.  They are therefore
never used as the event identity.  A provider-supplied event id, an explicit
supersession, or one unique cluster match is required before a correction can
be applied.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from engine.v2.contracts import RevisionCandidate
from engine.v2.data.errors import fail
from engine.v2.foundation import content_hash

__all__ = [
    "EventRevision",
    "event_revision_candidate",
    "resolve_event_identity",
    "apply_event_revision",
]


@dataclass(frozen=True, kw_only=True)
class EventRevision:
    candidate: RevisionCandidate
    event_id: str
    row: Mapping[str, Any] | None
    deleted: bool = False


def _existing_by_id(existing: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in existing:
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id:
            raise fail("CONTRACT_MISMATCH", "earnings event is missing event_id")
        if event_id in result and dict(result[event_id]) != dict(row):
            raise fail("IDENTITY_CONFLICT", "event_id maps to conflicting rows")
        result[event_id] = row
    return result


def resolve_event_identity(row: Mapping[str, Any],
                           existing: Sequence[Mapping[str, Any]]) -> str:
    """Return the stable id for a revision or refuse an ambiguous mapping.

    Same-ticker cluster matching is intentionally the only inferred mapping.
    Date-only matching is unsafe when a source moves an event by one session.
    """
    ticker = row.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        raise fail("CONTRACT_MISMATCH", "earnings event is missing ticker")
    by_id = _existing_by_id(existing)
    explicit = row.get("event_id")
    supersedes = row.get("supersedes_event_id")
    if explicit is not None or supersedes is not None:
        return _explicit_identity(row, ticker, by_id, explicit, supersedes)
    return _cluster_identity(row, ticker, by_id)


def _explicit_identity(row, ticker, by_id, explicit, supersedes):
    identity = explicit if explicit is not None else supersedes
    if not isinstance(identity, str) or not identity:
        raise fail("CONTRACT_MISMATCH", "earnings event identity is invalid")
    prior = by_id.get(identity)
    if supersedes is not None and prior is None:
        raise fail("IDENTITY_CONFLICT", "event supersession target is unknown")
    if prior is not None and prior.get("ticker") != ticker:
        raise fail("IDENTITY_CONFLICT", "event identity crosses security")
    return identity


def _cluster_identity(row, ticker, by_id):
    cluster = row.get("event_cluster_id")
    if not isinstance(cluster, str) or not cluster:
        raise fail("IDENTITY_CONFLICT", "event revision has no stable identity evidence")
    matches = sorted({
        event_id for event_id, prior in by_id.items()
        if prior.get("ticker") == ticker and prior.get("event_cluster_id") == cluster
    })
    if len(matches) != 1:
        raise fail("IDENTITY_CONFLICT", "event revision mapping is ambiguous")
    return matches[0]


def event_revision_candidate(*, event_id: str, row: Mapping[str, Any] | None,
                             source: str, source_priority: int, finality: str,
                             revision_ordinal: int, received_at: str,
                             deleted: bool = False) -> RevisionCandidate:
    if not event_id or finality not in ("provisional", "final"):
        raise fail("CONTRACT_MISMATCH", "event revision identity is incomplete")
    canonical_row = None if row is None else {**dict(row), "event_id": event_id}
    payload = {"event_id": event_id, "deleted": deleted, "row": canonical_row}
    return RevisionCandidate(
        revision_id="event_rev_" + content_hash(payload).removeprefix("sha256:")[:32],
        logical_key=event_id, source=source, source_priority=source_priority,
        finality=finality, revision_ordinal=revision_ordinal,
        received_at=received_at, content_hash=content_hash(payload))


def apply_event_revision(revision: EventRevision,
                         existing: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Apply a calendar revision while preserving the event id."""
    by_id = _existing_by_id(existing)
    if revision.candidate.logical_key != revision.event_id:
        raise fail("CONTRACT_MISMATCH", "event revision logical key is not event_id")
    if revision.deleted:
        by_id.pop(revision.event_id, None)
    else:
        if revision.row is None:
            raise fail("CONTRACT_MISMATCH", "live event revision has no row")
        row = dict(revision.row)
        if row.get("event_id") not in (None, revision.event_id):
            raise fail("IDENTITY_CONFLICT", "event revision row changes event identity")
        expected = content_hash({"event_id": revision.event_id, "deleted": False,
                                 "row": {**row, "event_id": revision.event_id}})
        if revision.candidate.content_hash != expected:
            raise fail("IDENTITY_CONFLICT", "event revision content hash is invalid")
        row["event_id"] = revision.event_id
        by_id[revision.event_id] = row
    return tuple(by_id[key] for key in sorted(by_id))
