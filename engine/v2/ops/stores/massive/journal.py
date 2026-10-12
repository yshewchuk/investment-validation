"""Immutable, bounded page progress; a prefix alone never grants coverage."""
import json
import sqlite3

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops import massive_plan
from engine.v2.ops.errors import fail

VERSION = "massive_acquisition_page.v1"
MAX_PAGES = 100
_TEXT_FIELDS = {"schema_version", "policy_version", "job_id", "generation", "scope_hash",
                "raw_receipt_id", "raw_hash", "first_received_at", "observed_at", "origin",
                "attempt_id"}
_FIELDS = _TEXT_FIELDS | {"page_index", "previous_page_id", "request", "next_request", "fence"}
_MALFORMED = (ValueError, TypeError, KeyError, RecursionError)


def _columns(document):
    if (type(document) is not dict or set(document) != _FIELDS
            or any(type(document[key]) is not str or not document[key] for key in _TEXT_FIELDS)
            or (document["schema_version"], document["policy_version"])
            != (VERSION, massive_plan.POLICY_VERSION)
            or type(document["page_index"]) is not int or document["page_index"] < 0
            or type(document["fence"]) is not int or document["fence"] <= 0
            or type(document["request"]) is not dict
            or type(document["next_request"]) not in (dict, type(None))
            or type(document["previous_page_id"]) not in (str, type(None))
            or document["origin"] not in ("raw_reuse", "http_response")):
        raise ValueError
    return (content_hash(document), document["job_id"], document["scope_hash"],
            document["page_index"], document["raw_receipt_id"], canonical_json(document))


def append(conn, document):
    """Append inside the caller's live-fenced raw-receipt transaction; never commit."""
    try:
        values = _columns(document)
        existing = conn.execute(
            "SELECT * FROM ops_massive_pages WHERE page_id=? OR "
            "(job_id=? AND request_hash=? AND page_index=?)", values[:4]).fetchall()
        if existing:
            if len(existing) != 1 or tuple(existing[0]) != values:
                raise ValueError
        else:
            conn.execute("INSERT INTO ops_massive_pages VALUES (?,?,?,?,?,?)", values)
        return values[0]
    except (*_MALFORMED, sqlite3.IntegrityError):
        raise fail("INTEGRITY_FAILED", "Massive acquisition page conflicts or is malformed") from None


def progress(conn, original_request, generation, job_id, verify):
    """Verify this job's exact gap prefix; verify(document) returns its checked next request.

    The callback rechecks raw receipt/object identity, observations and scoped rows.
    Returned requests are verified actual-request hashes for pre-append cycle checks.
    No raw I/O belongs in append or its caller-owned SQL transaction.
    """
    try:
        scope_hash = content_hash(original_request)
        rows = conn.execute(
            "SELECT * FROM ops_massive_pages WHERE job_id=? AND request_hash=? "
            "ORDER BY page_index LIMIT ?", (job_id, scope_hash, MAX_PAGES + 1)).fetchall()
        if len(rows) > MAX_PAGES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive acquisition page limit exceeded")
        page_ids, requests = [], []
        expected, last = original_request, None
        for index, row in enumerate(rows):
            document = json.loads(row["document_json"])
            values = _columns(document)
            metadata = (document["page_index"], document["job_id"], document["generation"],
                        document["scope_hash"], document["previous_page_id"])
            expected_metadata = (index, job_id, generation, scope_hash,
                                 page_ids[-1] if page_ids else None)
            if tuple(row) != values or metadata != expected_metadata or expected is None:
                raise ValueError
            request_hash = content_hash(document["request"])
            if request_hash != content_hash(expected) or request_hash in requests:
                raise ValueError
            requests.append(request_hash)
            checked_next = verify(document)
            if (canonical_json(document["next_request"]) != canonical_json(checked_next)
                    or (checked_next is not None and content_hash(checked_next) in requests)):
                raise ValueError
            page_ids.append(values[0])
            expected, last = checked_next, document
        return {"page_ids": page_ids, "next_request": expected, "last": last, "requests": requests}
    except _MALFORMED:
        raise fail("INTEGRITY_FAILED", "Massive acquisition page chain is malformed") from None
