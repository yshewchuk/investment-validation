"""Closed end-of-day quote availability preflight — always refuses, for now.

This is the verifier prerequisite of the quote EOD availability gate, not a
source producer and not a positive admission. It performs only what the
current catalog and artifact store can actually support:

* canonical-form validation of ``session_date``/``decision_at`` with no wall
  clock read and no inferred market-close threshold;
* exact resolution of the pinned ``SnapshotRef`` through
  ``Repository.resolve_full``, requiring the resolved snapshot to equal the
  supplied one field for field (a same-id/different-content handle refuses);
* catalog registration, full-byte verification, and a 1 MiB bound on every
  candidate availability/finality evidence artifact.

None of that proves the source produced or finalized the session. The
registered source completion/finality verifier and its evidence schema do not
exist yet, so every call ends in a fixed ``VALIDATION_FAILED``. A readable
evidence artifact is still unqualified: this module never parses its JSON as
source proof. Full source-byte and exact-domain coverage verification remains
mandatory in a later reviewed producer/admission slice before any positive
admission is possible; this preflight must never be cited as proving it.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import NoReturn

from engine.v2.contracts import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.foundation import ArtifactError, format_timestamp, parse_timestamp
from engine.v2.ops.checkpoints import registered_artifact
from engine.v2.ops.errors import fail

__all__ = ["verify_eod_availability"]

_SUPPORTED_TABLE = "option_chains"
_MAX_EVIDENCE_BYTES = 1 << 20
_UNAVAILABLE = "registered source completion/finality verifier is unavailable"
_TAMPERED = "availability evidence failed full-byte verification"
_SESSION_FORM = "session_date must be a canonical ISO date string"
_DECISION_FORM = "decision_at must be a canonical UTC timestamp"


def verify_eod_availability(conn, store, repository, snapshot, *, table_name,
                            session_date, decision_at) -> NoReturn:
    """Bounded preflight over one pinned snapshot's option_chains evidence.

    Validates the request form, pins the exact resolved snapshot, locates the
    table's actual dataset manifest, and verifies every bound availability and
    finality artifact byte for byte — then unconditionally refuses, because no
    registered source completion/finality verifier exists to interpret that
    evidence. There is no affirmative return, admission object or boolean.
    """
    if table_name != _SUPPORTED_TABLE:
        raise fail("INVALID_REQUEST", "only the option_chains table is supported")
    session_day = _session_day(session_date)
    moment = _decision_instant(decision_at)
    if session_day > moment.date():
        raise fail("INVALID_REQUEST", "session_date is after the decision_at UTC day")
    if not isinstance(snapshot, SnapshotRef):
        raise fail("INVALID_REQUEST", "snapshot must be a SnapshotRef")
    try:
        resolved = repository.resolve_full(snapshot.snapshot_id)
    except DataError as err:
        if err.code != "SNAPSHOT_NOT_FOUND":
            raise
        raise fail("VALIDATION_FAILED",
                   "the pinned snapshot is not registered") from err
    if resolved.snapshot != snapshot:
        raise fail("VALIDATION_FAILED", "the pinned snapshot does not resolve to itself")
    manifest = resolved.table_manifests.get(_SUPPORTED_TABLE)
    if manifest is None:
        raise fail("VALIDATION_FAILED", "the pinned snapshot has no option_chains dataset version")
    candidates = _candidate_ids(manifest.availability_evidence_refs,
                                resolved.snapshot.finality_receipt_refs)
    for artifact_id in candidates:
        _verify_evidence(conn, store, artifact_id)
    raise fail("VALIDATION_FAILED", _UNAVAILABLE)


def _session_day(value: object) -> date:
    """A canonical ``YYYY-MM-DD`` date, or a fixed ``INVALID_REQUEST``."""
    if not isinstance(value, str):
        raise fail("INVALID_REQUEST", _SESSION_FORM)
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        raise fail("INVALID_REQUEST", _SESSION_FORM) from None
    if parsed.isoformat() != value:
        raise fail("INVALID_REQUEST", _SESSION_FORM)
    return parsed


def _decision_instant(value: object) -> datetime:
    """A canonical framework UTC timestamp, or a fixed ``INVALID_REQUEST``."""
    if not isinstance(value, str):
        raise fail("INVALID_REQUEST", _DECISION_FORM)
    try:
        parsed = parse_timestamp(value)
    except ValueError:
        raise fail("INVALID_REQUEST", _DECISION_FORM) from None
    if format_timestamp(parsed) != value:
        raise fail("INVALID_REQUEST", _DECISION_FORM)
    return parsed


def _candidate_ids(availability, finality) -> tuple[str, ...]:
    """Both carriers must be non-empty and internally unique; a ref shared
    between carriers is read once, never counted as two proofs."""
    _check_carrier(availability, "availability")
    _check_carrier(finality, "finality")
    return tuple(dict.fromkeys((*availability, *finality)))


def _check_carrier(refs, carrier: str) -> None:
    if not refs:
        raise fail("VALIDATION_FAILED", f"pinned {carrier} evidence is empty")
    for ref in refs:
        if not isinstance(ref, str) or not ref:
            raise fail("VALIDATION_FAILED", f"pinned {carrier} evidence must name artifacts")
    if len(set(refs)) != len(refs):
        raise fail("VALIDATION_FAILED", f"pinned {carrier} evidence repeats an artifact")


def _verify_evidence(conn, store, artifact_id: str) -> None:
    """Registered metadata, size bound, then full bytes. The metadata lookup
    makes no byte-integrity claim, and oversized candidates are refused before
    any read allocates them. The bytes prove identity only, never availability."""
    ref = registered_artifact(conn, artifact_id)
    if ref.byte_size > _MAX_EVIDENCE_BYTES:
        raise fail("VALIDATION_FAILED", "availability evidence exceeds the 1 MiB limit")
    try:
        store.read_verified(ref)
    except ArtifactError as err:
        if err.code == "INTEGRITY_FAILED":
            raise fail("INTEGRITY_FAILED", _TAMPERED) from err
        raise fail("VALIDATION_FAILED",
                   "availability evidence object is unavailable") from err
