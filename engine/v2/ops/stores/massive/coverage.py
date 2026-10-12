"""Immutable acquisition occurrences and bounded, receipt-pinned interval reuse."""
import json
import re
from datetime import timedelta

from engine.v2.data.incremental import RAW_SCHEMA_REF, load_raw_receipt
from engine.v2.foundation import canonical_json, content_hash, format_timestamp, parse_timestamp
from engine.v2.foundation.artifacts import artifact_reference
from engine.v2.ops import massive_plan
from engine.v2.ops.errors import fail
from engine.v2.ops.stores.massive import journal
from engine.v2.ops.stores.massive.pages import closed_day, validate_page

VERSION = "massive_interval_acquisition.v1"
MAX_OCCURRENCES = 100
MAX_VERIFIED_BYTES = 64 << 20


def validate_generation(generation):
    if (not isinstance(generation, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", generation) is None):
        raise fail("INVALID_REQUEST", "invalid Massive acquisition generation")


def family(request):
    normalized = massive_plan._request_descriptor(request["dataset"], request["ticker"],
                                                  "2000-01-01", "2000-01-01",
                                                  request["adjustment_basis"])
    return content_hash([VERSION, massive_plan.POLICY_VERSION, normalized])


def _columns(document):
    request = document["request"]
    return (content_hash(document), document["job_id"], content_hash(request), family(request),
            document["generation"], request["start"], request["end"],
            document["raw_receipt_id"], canonical_json(document))


def record(conn, document):
    """Caller owns the live-fenced raw-receipt transaction."""
    values = _columns(document)
    existing = conn.execute("SELECT occurrence_id FROM ops_massive_occurrences "
                            "WHERE job_id=? AND request_hash=?", values[1:3]).fetchone()
    if existing is not None:
        if existing[0] != values[0]:
            raise fail("INTEGRITY_FAILED", "Massive acquisition occurrence conflicts")
        return
    conn.execute("INSERT INTO ops_massive_occurrences VALUES (?,?,?,?,?,?,?,?,?)", values)


def _raw(conn, store, document, original, remaining):
    try:
        request = document["request"]
        observed = parse_timestamp(document["observed_at"])
        if (format_timestamp(observed) != document["observed_at"]
                or document["origin"] not in ("raw_reuse", "http_response")):
            raise ValueError
        attempt = conn.execute("SELECT job_id,fence FROM attempts WHERE attempt_id=?",
                               (document["attempt_id"],)).fetchone()
        if attempt is None or tuple(attempt) != (document["job_id"], document["fence"]):
            raise ValueError
        closed_day(original, observed, cached=True)
        raw = conn.execute("SELECT * FROM data_raw_receipts WHERE raw_receipt_id=?",
                           (document["raw_receipt_id"],)).fetchone()
        if (raw is None
                or (raw["source"], raw["endpoint"], raw["request_hash"]) !=
                (massive_plan.SOURCE, request["endpoint"], content_hash(request))
                or document["first_received_at"] != raw["received_at"]):
            raise ValueError
        ref = massive_plan._receipt_identity(raw, request, content_hash(request))
        if raw["raw_hash"] != document["raw_hash"]:
            raise ValueError
        remaining[0] -= ref["byte_size"]
        if remaining[0] < 0:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive coverage verification exceeds byte policy")
        body = load_raw_receipt(conn, store, raw["raw_receipt_id"])
        if ref["object_id"] != artifact_reference(body, RAW_SCHEMA_REF).artifact_id:
            raise ValueError
        continuation = validate_page(body, original, request, cached=True)
        if raw["response_kind"] != ("partial" if continuation is not None else "complete"):
            raise ValueError
        return continuation
    except (ValueError, TypeError, KeyError, RecursionError):
        raise fail("INTEGRITY_FAILED", "Massive acquisition occurrence is malformed") from None


def progress(conn, store, request, generation, job_id, remaining=None):
    remaining = [MAX_VERIFIED_BYTES] if remaining is None else remaining
    return journal.progress(conn, request, generation, job_id,
                            lambda document: _raw(conn, store, document, request, remaining))


def _verified(conn, store, row, remaining):
    try:
        document = json.loads(row["document_json"])
        request = document["request"]
        canonical = massive_plan._request_descriptor(request["dataset"], request["ticker"],
                                                      request["start"], request["end"],
                                                      request["adjustment_basis"])
        if (tuple(row) != _columns(document) or request != canonical
                or document["schema_version"] != VERSION
                or document["policy_version"] != massive_plan.POLICY_VERSION):
            raise ValueError
        if "pages" in document:
            chain = progress(conn, store, request, document["generation"], document["job_id"], remaining)
            fields = ("raw_receipt_id", "raw_hash", "first_received_at", "observed_at", "origin", "attempt_id", "fence")
            if (not document["pages"] or document["pages"] != chain["page_ids"]
                    or chain["next_request"] is not None
                    or any(document[key] != chain["last"][key] for key in fields)):
                raise ValueError
        elif _raw(conn, store, document, request, remaining) is not None:
            raise ValueError
        return document
    except (ValueError, TypeError, KeyError, RecursionError):
        raise fail("INTEGRITY_FAILED", "Massive acquisition completion is malformed") from None


def plan(conn, store, request, generation):
    """Return deterministic clipped receipt bindings and maximal missing intervals."""
    validate_generation(generation)
    first, last = massive_plan._day(request["start"]), massive_plan._day(request["end"])
    if first > last:
        raise fail("INVALID_REQUEST", "Massive coverage start follows end")
    rows = conn.execute(
        "SELECT * FROM ops_massive_occurrences WHERE family=? AND generation=? "
        "AND start_date<=? AND end_date>=? ORDER BY occurrence_id LIMIT ?",
        (family(request), generation, request["end"], request["start"], MAX_OCCURRENCES + 1)).fetchall()
    if len(rows) > MAX_OCCURRENCES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive coverage candidate limit exceeded")
    remaining = [MAX_VERIFIED_BYTES]
    candidates = [(row["occurrence_id"], _verified(conn, store, row, remaining)) for row in rows]
    stop = last + timedelta(days=1)
    boundaries = {first, stop}
    for _, document in candidates:
        boundaries.update((max(first, massive_plan._day(document["request"]["start"])),
                           min(stop, massive_plan._day(document["request"]["end"]) + timedelta(days=1))))
    segments = []
    ordered = sorted(boundaries)
    for start, end in zip(ordered, ordered[1:]):
        eligible = [(doc["observed_at"], occurrence, doc) for occurrence, doc in candidates
                    if doc["request"]["start"] <= start.isoformat() <= doc["request"]["end"]]
        chosen = max(eligible, key=lambda item: item[:2]) if eligible else None
        identity = chosen[1] if chosen else None
        if segments and segments[-1][2] == identity:
            segments[-1] = (segments[-1][0], end, identity)
        else:
            segments.append((start, end, identity))
    documents = dict(candidates)
    bindings, gaps = [], []
    for start, end, identity in segments:
        interval = {"start": start.isoformat(), "end": (end - timedelta(days=1)).isoformat()}
        if identity is None:
            gaps.append(massive_plan._request_descriptor(request["dataset"], request["ticker"],
                                                        **interval, adjustment=request["adjustment_basis"]))
        else:
            bindings.append({**interval, "occurrence_id": identity, "evidence": documents[identity]})
    result = {"schema_version": VERSION, "request": request, "generation": generation,
              "bindings": bindings, "gaps": gaps}
    return {**result, "plan_hash": content_hash(result)}
