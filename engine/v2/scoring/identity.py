"""Canonical identity for Phase 4 score requests and records."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Mapping

from engine.v2.contracts import ScoreRecord, ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.frozen_record import freeze_record_fields

__all__ = ["canonical_request", "dependency_hash", "request_hash", "score_id",
           "bootstrap_seed", "score_request_key"]


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


def score_request_key(context: Mapping[str, Any]) -> str:
    """Rebuild legacy ``ScoreRequest.key()`` from captured request identity.

    ``engine/score.py`` seeds each bootstrap from
    ``sha256(f"{self.snapshot}|{request.key()}")``. The native pipeline must
    draw the identical Monte Carlo path for the same trade, so this reads the
    same request-level facts from a native score context and renders them in
    legacy's exact format. Every required name is checked for PRESENCE, not
    truth: ``None`` is a legitimate legacy state (an open ``as_of`` renders
    ``""``), while an absent name is a capture defect and raises.
    """
    required = (
        "ticker",
        "strategy",
        "requested_as_of",
        "requested_event_date",
        "requested_strike",
        "requested_expiry",
        "fill_alpha",
        "variant",
        "decision_offset",
        "quote_max_age_sessions",
        "chain_as_of",
    )
    for name in required:
        if name not in context:
            raise KeyError(f"score_request_key: missing identity field {name!r}")

    ticker = context["ticker"]
    strategy = context["strategy"]
    as_of = context["requested_as_of"]
    event_date = context["requested_event_date"]
    strike = context["requested_strike"]
    expiry = context["requested_expiry"]
    fill_alpha = context["fill_alpha"]
    variant = context["variant"]
    decision_offset = context["decision_offset"]
    quote_max_age_sessions = context["quote_max_age_sessions"]
    chain_as_of = context["chain_as_of"]
    structure_params = context.get("requested_structure_params")
    parts = [
        str(ticker),
        str(strategy),
        "" if as_of is None else str(as_of),
        "" if event_date is None else str(event_date),
        "" if strike is None else f"{float(strike):.4f}",
        "" if expiry is None else str(expiry),
        f"{float(fill_alpha):.4f}",
        variant or "",
        "" if decision_offset is None else f"d{int(decision_offset):+d}",
        "" if quote_max_age_sessions is None
        else f"q{int(quote_max_age_sessions)}",
        "" if chain_as_of is None else f"c{chain_as_of}",
        "" if not structure_params else ",".join(
            f"{key}={structure_params[key]!r}" for key in sorted(structure_params)
        ),
    ]
    return "|".join(parts)


def bootstrap_seed(snapshot: str, key: str) -> int:
    """Legacy's deterministic bootstrap seed for ``snapshot`` and ``key``."""
    return int.from_bytes(
        hashlib.sha256(f"{snapshot}|{key}".encode()).digest()[:8], "big"
    )
