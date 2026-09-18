"""Canonical identity for Phase 4 score requests and records."""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from engine.v2.contracts import ScoreRecord, ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.frozen_record import freeze_record_fields

__all__ = ["canonical_request", "dependency_hash", "request_hash", "score_id"]


def canonical_request(request: ScoreRequest) -> dict[str, Any]:
    """Return the exact numerical request payload, excluding operational time."""
    document = to_document(request)
    document.pop("schema_version", None)
    return document


def request_hash(request: ScoreRequest) -> str:
    return content_hash(canonical_request(request))


def dependency_hash(dependencies: Mapping[str, Any] | None = None) -> str:
    """Hash every declared population/model/residual dependency in order."""
    return content_hash(dict(dependencies or {}))


def score_id(record: ScoreRecord | Mapping[str, Any], *, outcome: Any = None) -> str:
    """Hash immutable score inputs and resolved output identity.

    Operational envelopes and derived hashes are deliberately removed before
    hashing. The result remains replay-identical when it is recomputed later.
    """
    payload = to_document(record) if not isinstance(record, Mapping) else dict(record)
    payload.pop("score_id", None)
    payload.pop("computed_at", None)
    payload.pop("operational_envelope", None)
    payload.pop("payload_hash", None)
    payload.pop("request_hash", None)
    payload.pop("schema_version", None)
    if outcome is not None:
        payload["outcome"] = outcome
    return content_hash(payload)


def with_score_id(record: ScoreRecord, *, outcome: Any = None) -> ScoreRecord:
    """Return a record whose ID is derived from its canonical content.

    ``ScoreRecord`` mapping fields must be recursively immutable before the
    record escapes the scoring package. ``dataclasses.replace`` no longer
    reruns that freeze (the contract module defines shapes only — see
    ``engine/v2/contracts/scoring.py``), so it is applied explicitly here,
    the one place every constructed record passes through.
    """
    request = content_hash(record.canonical_request)
    identity = score_id(record, outcome=outcome)
    updated = replace(record, score_id=identity, request_hash=request,
                       payload_hash=identity)
    return freeze_record_fields(updated)
