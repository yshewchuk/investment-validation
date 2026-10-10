"""S4C: the shared provider-receipt cache and failure classification.

Extracted from ``engine.v2.ops.calendar_moves_jobs`` (P6 slice-4c split, Part
0) so both ``forward_calendar_store`` and ``computed_moves_store`` can import
these primitives without depending on the job-kind/worker module, which is a
separate, later PR. Nothing in this module reaches ``engine.v2.ops.nightly``
or job dispatch; it owns only the raw-receipt cache and the provider failure
classification that both stores and the job-kind module (once it lands) share.

Provider accounts: ``yfinance`` (computed_moves, and the forward calendar's
session confirmation) and ``nasdaq`` (the forward calendar's discovery calls)
are unmetered and keyless; their credential tuples are empty, but their budget
rows are still operator-provisioned so the shared scheduler reserves against
them like any other account.
"""
from __future__ import annotations

from typing import Sequence

from engine.v2.ops.errors import fail

NATIVE_COMPUTED_MOVES_ACCOUNT = "yfinance"
#: One budget account backs every yfinance call, whichever job makes it.
NATIVE_YFINANCE_ACCOUNT = NATIVE_COMPUTED_MOVES_ACCOUNT
NATIVE_NASDAQ_ACCOUNT = "nasdaq"

#: S4C failure semantics (spec R1): every provider edge returns one of these
#: kinds. Only ``complete`` may ever be reused from the durable raw cache (R2).
#: ``credential_invalid`` is the provider's own 401/403; ``refused`` is every
#: other unusable answer (unparseable body, non-auth 4xx).
RESPONSE_KINDS = ("complete", "legitimate_empty", "not_final", "transient",
                  "refused", "credential_invalid")
#: The response kinds that are real, parseable payloads: a receipt records its
#: own kind, but ``cached_unit_outcomes`` reuses only ``complete``.
CACHEABLE_RESPONSE_KINDS = ("complete", "legitimate_empty")
#: kind -> the registered failure code that ends the job. ``complete`` and
#: ``legitimate_empty`` are successful unit outcomes; ``transient`` is
#: retryable, ``refused``/``credential_invalid``/``not_final`` are not silently
#: retried. A body that fails to parse is ``SOURCE_INVALID`` (bad source data),
#: never ``CREDENTIAL_INVALID`` -- only the 401/403 kind is that.
_FAILURE_CODE_BY_RESPONSE_KIND = {
    "not_final": "SOURCE_NOT_FINAL",
    "transient": "TRANSIENT_SOURCE",
    "refused": "SOURCE_INVALID",
    "credential_invalid": "CREDENTIAL_INVALID",
}
#: Higher is worse, so a mixed run reports the failure that cannot be retried.
_FAILURE_SEVERITY = {"TRANSIENT_SOURCE": 0, "SOURCE_NOT_FINAL": 1,
                     "SOURCE_INVALID": 2, "CREDENTIAL_INVALID": 3}

__all__ = [
    "NATIVE_COMPUTED_MOVES_ACCOUNT",
    "NATIVE_NASDAQ_ACCOUNT",
    "NATIVE_YFINANCE_ACCOUNT",
    "RESPONSE_KINDS",
    "cached_unit_outcomes",
    "cached_unit_payloads",
    "provider_failure_code",
    "record_unit_receipt",
]


def provider_failure_code(kinds) -> str | None:
    """The worst typed failure code among unit response kinds, or ``None``.

    Spec R3: any unit that ends ``transient`` fails the job with
    ``TRANSIENT_SOURCE`` (retryable); ``refused`` fails it with the
    non-retryable ``SOURCE_INVALID`` and ``credential_invalid`` (a provider
    401/403) with ``CREDENTIAL_INVALID``; a run that is all-``complete``/
    ``legitimate_empty`` has no failure.
    """
    worst = None
    for kind in kinds:
        code = _FAILURE_CODE_BY_RESPONSE_KIND.get(kind)
        if code is not None and (worst is None
                                 or _FAILURE_SEVERITY[code] > _FAILURE_SEVERITY[worst]):
            worst = code
    return worst


# --------------------------------------------------------------------------
# the shared raw-receipt cache
# --------------------------------------------------------------------------


def _unit_request(unit) -> dict:
    return {"request_id": unit.request_id, "table_name": unit.table_name,
            "partition_key": unit.partition_key, "keys": list(unit.expected_keys)}


#: The newest receipt for a given (source, endpoint, request_hash). Two
#: receipts can share ``received_at`` (same-second reruns, or a coarse clock),
#: so the tie is broken by ``rowid`` -- the table's own insertion sequence
#: (data_raw_receipts is append-only, spec R2, so rowid only grows) -- and not
#: left to whatever order the query planner's chosen index happens to return.
#: Extracted to a constant so its tie-break is unit-tested directly, against a
#: bare table, independent of which index (if any) answers the WHERE clause.
_LATEST_RECEIPT_SQL = (
    "SELECT raw_receipt_id, raw_hash, response_kind FROM data_raw_receipts "
    "WHERE source = ? AND endpoint = ? AND request_hash = ? "
    "ORDER BY received_at DESC, rowid DESC LIMIT 1"
)


def record_unit_receipt(conn, store, unit, payload: bytes, *, source: str, endpoint: str,
                        received_at: str, response_kind: str = "complete"):
    """Cache one unit's acquired bytes so a rerun resolves it without a fetch.

    ``cache_raw_receipt`` is the data layer's own idempotent raw cache: the
    receipt identity is a hash of ``(source, endpoint, request, raw)``, so a
    replay of the same bytes returns the existing row and publishes nothing.
    Only a real, parseable payload (spec R2) may be cached at all: a
    ``refused``/``transient``/``not_final`` response is never stored, and the
    recorded kind is what ``cached_unit_outcomes`` later reads back.
    """
    from engine.v2.data.incremental import RawPayload, cache_raw_receipt

    if response_kind not in CACHEABLE_RESPONSE_KINDS:
        raise fail("INVALID_REQUEST",
                   "only a complete or legitimately empty provider response may be cached")
    return cache_raw_receipt(
        conn, store,
        RawPayload(payload=payload, response_kind=response_kind, response_meta={}),
        source=source, endpoint=endpoint, request=_unit_request(unit),
        received_at=received_at)


def cached_unit_outcomes(conn, units: Sequence, *, source: str, endpoint: str) -> dict:
    """``{request_id: AcquisitionOutcome}`` for units with a durable receipt.

    Both the nightly plan builders and the stores themselves call this, so a
    second same-catalog run plans (and acquires) exactly zero provider calls.
    Only a receipt whose recorded ``response_kind`` is ``complete`` is reused
    (spec R2): a ``legitimate_empty`` payload is refetched on the next run, and
    ``not_final``/``transient``/``refused`` receipts were never cached.
    """
    from engine.v2.data.incremental import _jsonable
    from engine.v2.foundation import content_hash
    from engine.v2.ops.provider_response import classify_response

    outcomes = {}
    for unit in units:
        request_hash = content_hash(_jsonable(dict(_unit_request(unit))))
        row = conn.execute(
            _LATEST_RECEIPT_SQL, (source, endpoint, request_hash)).fetchone()
        if row is None or row["response_kind"] != "complete":
            continue
        outcomes[unit.request_id] = classify_response(
            200, unit.expected_keys, returned_keys=unit.expected_keys,
            request_id=unit.request_id, receipt_ref=row["raw_receipt_id"],
            raw_hash=row["raw_hash"], cache_hit=True)
    return outcomes


def cached_unit_payloads(conn, store, plan) -> dict[str, bytes]:
    """Verified bytes of every cache-hit unit in one plan, keyed by request id.

    A same-session retry must rebuild every wanted unit's claims/fragments, not
    only the units it fetches fresh: the plan's cache set holds exactly the
    units whose durable receipt is ``complete`` (spec R2), so re-reading those
    bytes by receipt reconstructs the rows a clean single run would have
    parsed.
    """
    from engine.v2.data.incremental import load_raw_receipt

    payloads: dict[str, bytes] = {}
    for outcome in plan.cached:
        if outcome.receipt_ref is None:
            continue
        payloads[outcome.request_id] = load_raw_receipt(conn, store, outcome.receipt_ref)
    return payloads
