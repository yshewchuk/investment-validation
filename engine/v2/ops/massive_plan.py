"""Network-free Massive Basic request planning over the shared raw-receipt cache."""
from __future__ import annotations

import json
import re
from datetime import date, timedelta

from engine.v2.data.incremental import RAW_SCHEMA_REF, load_raw_receipt
from engine.v2.foundation import content_hash
from engine.v2.foundation.artifacts import ArtifactError, artifact_reference
from engine.v2.ops.errors import fail
from engine.v2.ops.providers.massive_policy import MAX_RAW_BYTES

SOURCE = "massive-stocks"
POLICY_VERSION = "massive_stocks_basic.2026-10-10"
DATASETS = ("daily", "splits", "dividends")


def _day(value):
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return parsed
    except (TypeError, ValueError):
        pass
    raise fail("INVALID_REQUEST", "Massive planning requires canonical ISO dates")


def _scope(tickers, start, end, as_of, adjustment, transport, call_cap):
    if transport != "rest":
        raise fail("INVALID_REQUEST", "Massive Basic supports REST planning only",
                   details={"reason": "CAPABILITY_UNAVAILABLE"})
    if adjustment not in ("raw", "split_adjusted"):
        raise fail("INVALID_REQUEST", "an explicit price adjustment is required")
    if type(call_cap) is not int or not 1 <= call_cap <= 100:
        raise fail("INVALID_REQUEST", "per-request HTTP-attempt cap must be 1 to 100")
    if not isinstance(tickers, (list, tuple)) or not 1 <= len(tickers) <= 100:
        raise fail("INVALID_REQUEST", "one to 100 ticker inputs are required")
    if any(not isinstance(t, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", t)
           for t in tickers):
        raise fail("INVALID_REQUEST", "invalid stock ticker syntax")
    first, last, today = _day(start), _day(end), _day(as_of)
    if first > last:
        raise fail("INVALID_REQUEST", "start date must not follow end date")
    if today.year <= 2:
        raise fail("INVALID_REQUEST", "planning date has no two-year history window")
    try:
        earliest = today.replace(year=today.year - 2)
    except ValueError:
        earliest = today.replace(year=today.year - 2, day=28)
    if first < earliest or last >= today:
        raise fail("INVALID_REQUEST", "dates exceed the Basic planning window",
                   details={"reason": "HISTORY_UNAVAILABLE"})
    return sorted(set(tickers)), earliest.isoformat(), (today - timedelta(days=1)).isoformat()


def _request_descriptor(dataset, ticker, start, end, adjustment):
    """Build one credential-free request; plan validates its caller's scope first."""
    if dataset == "daily":
        endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {"adjusted": adjustment == "split_adjusted", "sort": "asc", "limit": 50000}
    elif dataset in ("splits", "dividends"):
        endpoint = f"/stocks/v1/{dataset}"
        field = "execution_date" if dataset == "splits" else "ex_dividend_date"
        params = {"ticker": ticker, field + ".gte": start, field + ".lte": end,
                  "sort": field + ".asc", "limit": 5000}
        adjustment = "corporate_action"
    else:
        raise fail("INVALID_REQUEST", "unsupported Massive dataset",
                   details={"reason": "CAPABILITY_UNAVAILABLE"})
    return {"schema_version": "massive_request.v1", "source": SOURCE,
            "transport": "rest", "endpoint": endpoint, "dataset": dataset,
            "ticker": ticker, "start": start, "end": end,
            "adjustment_basis": adjustment, "params": params}


def _receipt_identity(row, request, request_hash):
    try:
        ref = json.loads(row["artifact_ref_json"])
        expected = "raw_" + content_hash({"source": SOURCE, "endpoint": request["endpoint"],
                                         "request_hash": request_hash,
                                         "raw_hash": row["raw_hash"]}).removeprefix("sha256:")[:32]
        valid = (content_hash(json.loads(row["request_json"])) == request_hash
                 and ref["kind"] == "raw_response"
                 and ref["content_hash"] == row["raw_hash"] and expected == row["raw_receipt_id"])
        size = ref["byte_size"]
    except (ValueError, TypeError, KeyError, RecursionError):
        raise fail("INTEGRITY_FAILED", "Massive raw receipt identity is malformed") from None
    if not valid:
        raise fail("INTEGRITY_FAILED", "Massive raw receipt identity disagrees")
    if type(size) is not int or not 0 <= size <= MAX_RAW_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive raw receipt exceeds planning size policy")
    return ref


def _complete_body(body, request):
    try:
        document = json.loads(body)
        rows = document.get("results", [])
        valid = document.get("status") == "OK" and isinstance(rows, list)
        valid = valid and all(isinstance(row, dict) for row in rows)
        count = document.get("resultsCount", len(rows))
        valid = valid and type(count) is int and count == len(rows)
        if request["dataset"] == "daily":
            valid = (valid and document.get("ticker") == request["ticker"]
                     and document.get("adjusted") is request["params"]["adjusted"])
    except (ValueError, TypeError, AttributeError, RecursionError):
        raise fail("SOURCE_INVALID", "Massive cached response is malformed") from None
    if not valid:
        raise fail("SOURCE_INVALID", "Massive cached response contradicts its request")
    return bool(rows) and document.get("next_url") is None


def _cached(conn, store, request, request_hash):
    if conn is None:
        return None
    row = conn.execute(
        "SELECT * FROM data_raw_receipts WHERE source=? AND endpoint=? AND request_hash=? "
        "ORDER BY received_at DESC, rowid DESC LIMIT 1",
        (SOURCE, request["endpoint"], request_hash)).fetchone()
    if row is None or row["response_kind"] != "complete":
        return None
    ref = _receipt_identity(row, request, request_hash)
    try:
        body = load_raw_receipt(conn, store, row["raw_receipt_id"])
    except ArtifactError as exc:
        if exc.code == "UNSAFE_PATH" and isinstance(exc.__cause__, FileNotFoundError):
            raise ArtifactError("MISSING", "Massive cached raw store is absent") from None
        raise
    if ref["object_id"] != artifact_reference(body, RAW_SCHEMA_REF).artifact_id:
        raise fail("INTEGRITY_FAILED", "Massive raw artifact identity disagrees")
    return row["raw_receipt_id"] if _complete_body(body, request) else None


def plan(conn, store, *, tickers, start, end, as_of, adjustment,
         max_calls_per_request, transport="rest", datasets=DATASETS):
    """Inventory exact raw hits; conn=None validates a request-only plan without I/O."""
    tickers, earliest, latest = _scope(tickers, start, end, as_of, adjustment,
                                      transport, max_calls_per_request)
    if not isinstance(datasets, (tuple, list)) or not datasets or len(datasets) > 3:
        raise fail("INVALID_REQUEST", "one to three supported datasets are required")
    if any(not isinstance(d, str) or d not in DATASETS for d in datasets):
        raise fail("INVALID_REQUEST", "unsupported Massive dataset",
                   details={"reason": "CAPABILITY_UNAVAILABLE"})
    requests = [_request_descriptor(d, t, start, end, adjustment)
                for t in tickers for d in sorted(set(datasets))]
    if conn is not None and conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_raw_receipts'"
    ).fetchone() is None:
        raise fail("VALIDATION_FAILED", "Massive planning requires a raw-receipt catalog")
    units = []
    for request in requests:
        digest = content_hash(request)
        receipt = _cached(conn, store, request, digest)
        units.append({"request": request, "request_hash": digest, "raw_receipt_id": receipt})
    missing = sum(u["raw_receipt_id"] is None for u in units)
    return {"schema_version": "massive_cache_plan.v1", "policy_version": POLICY_VERSION,
            "as_of": as_of, "earliest_date": earliest, "latest_date": latest,
            "requests_per_minute": 5, "plan_only": True, "units": units,
            "minimum_initial_calls": missing,
            "max_calls_per_request": max_calls_per_request,
            "maximum_http_attempts": missing * max_calls_per_request}
