"""Pure provider-response classification, independent of refresh orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from engine.v2.ops.errors import fail

OutcomeKind = Literal[
    "complete", "empty", "partial", "unsupported", "not_final",
    "credential_invalid", "rate_limited", "transient",
]


@dataclass(frozen=True, kw_only=True)
class AcquisitionOutcome:
    """Redacted acquisition result used for retry and commit admission."""

    request_id: str
    kind: OutcomeKind
    requested_keys: tuple[str, ...]
    returned_keys: tuple[str, ...] = ()
    empty_keys: tuple[str, ...] = ()
    unsupported_keys: tuple[str, ...] = ()
    receipt_ref: str | None = None
    raw_hash: str | None = None
    cache_hit: bool = False
    quota_remaining: int | None = None


def classify_response(status: int, requested_keys: Sequence[str], *,
                      returned_keys: Sequence[str] = (), empty_keys: Sequence[str] = (),
                      unsupported_keys: Sequence[str] = (), final: bool = True,
                      truncated: bool = False, credential_page: bool = False,
                      request_id: str = "", receipt_ref: str | None = None,
                      raw_hash: str | None = None, cache_hit: bool = False,
                      quota_remaining: int | None = None) -> AcquisitionOutcome:
    """Classify one response without treating HTTP 200 as complete coverage."""
    requested = _keys(requested_keys, "requested_keys")
    returned = _subset(returned_keys, requested, "returned_keys")
    empty = _subset(empty_keys, requested, "empty_keys")
    unsupported = _subset(unsupported_keys, requested, "unsupported_keys")
    if quota_remaining is not None and quota_remaining < 0:
        raise fail("INVALID_REQUEST", "negative provider quota observation")
    kind = _response_kind(status, requested, returned, empty, unsupported,
                          final=final, truncated=truncated,
                          credential_page=credential_page)
    return AcquisitionOutcome(
        request_id=request_id, kind=kind, requested_keys=requested,
        returned_keys=returned, empty_keys=empty, unsupported_keys=unsupported,
        receipt_ref=receipt_ref, raw_hash=raw_hash, cache_hit=cache_hit,
        quota_remaining=quota_remaining)


def _response_kind(status, requested, returned, empty, unsupported, *, final, truncated,
                   credential_page) -> OutcomeKind:
    if credential_page or status in (401, 403):
        return "credential_invalid"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "transient"
    if status == 404:
        return "unsupported"
    if not 200 <= status < 300:
        return "transient"
    if not final:
        return "not_final"
    covered = set(returned) | set(empty) | set(unsupported)
    if truncated or covered != set(requested):
        return "partial"
    if requested and set(empty) == set(requested):
        return "empty"
    return "complete"


def _keys(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    keys = tuple(sorted(values))
    if any(not value for value in keys) or len(set(keys)) != len(keys):
        raise fail("INVALID_REQUEST", f"{field_name} must contain unique nonempty keys")
    return keys


def _subset(values, requested, field_name):
    keys = _keys(values, field_name)
    if not set(keys).issubset(requested):
        raise fail("INVALID_REQUEST", f"{field_name} contains an unrequested key")
    return keys
