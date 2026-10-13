"""Bounded Massive exact or generation acquisition through the supervisor."""
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass, replace
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

from engine.v2.data.errors import DataError
from engine.v2.data.incremental import RawPayload, cache_raw_receipt, load_raw_receipt
from engine.v2.foundation import (
    ArtifactStore,
    DocumentError,
    SystemClock,
    canonical_json,
    content_hash,
    format_timestamp,
    from_document,
    parse_timestamp,
)
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops import massive_plan, provider_budget
from engine.v2.ops.catalog import connect, transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.stores.massive import coverage, journal
from engine.v2.ops.stores.massive.pages import closed_day, validate_page
from engine.v2.ops.submission import JobKind, RetryPolicy

KIND = "massive_acquisition"
SCHEMA = "massive_acquisition.v1"
HISTORY_KIND = "massive_history_acquisition"
HISTORY_SCHEMA = "massive_history_acquisition.v1"


@dataclass(frozen=True)
class Parameters:
    ticker: str
    dataset: str
    start: str
    end: str
    as_of: str
    adjustment: str
    expected_ids: tuple[str, ...]
    policy_version: str = massive_plan.POLICY_VERSION
    provider_account: str = massive_plan.SOURCE
    provider_calls: int = 1
    generation: str | None = None
    max_http_calls: int = 1


def scope_plan(params, conn=None, store=None):
    if (type(params.max_http_calls) is not int or not 1 <= params.max_http_calls <= 100
            or (params.generation is None and params.max_http_calls != 1)):
        raise fail("INVALID_REQUEST", "invalid Massive generation or cumulative call cap")
    if params.generation is not None:
        coverage.validate_generation(params.generation)
    result = massive_plan.plan(conn, store, tickers=[params.ticker], datasets=[params.dataset],
                               start=params.start, end=params.end, as_of=params.as_of,
                               adjustment=params.adjustment, max_calls_per_request=1)
    if (params.policy_version != result["policy_version"]
            or params.provider_account != massive_plan.SOURCE
            or type(params.provider_calls) is not int or params.provider_calls != 1
            or params.expected_ids != (result["units"][0]["request_hash"],)):
        raise fail("INVALID_REQUEST", "Massive acquisition identity or budget disagrees")
    return result


def parameter_problems(job, params):
    scope_plan(params)
    if (job.provider_budget_ref != massive_plan.SOURCE or job.input_refs or job.dependency_job_ids
            or (job.kind == HISTORY_KIND) != (params.generation is not None)):
        return ("Massive acquisition requires its fixed account and no input dependencies",)
    return ()


def validate_as_of(as_of, clock):
    if as_of > clock.now().astimezone(timezone.utc).date().isoformat():
        raise fail("INVALID_REQUEST", "Massive acquisition date exceeds the current UTC date")


def job_kind():
    return JobKind(name=KIND, worker=KIND, parameters=Parameters,
                   resource_classes=frozenset({"io_fetch"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 1), checkpoint_contract=SCHEMA,
                   namespaces=frozenset({"shadow"}), validate=parameter_problems)


def history_kind():
    return replace(job_kind(), name=HISTORY_KIND, retry=RetryPolicy("bounded", 101, (13,)),
                   checkpoint_contract=HISTORY_SCHEMA)


def _record_fetch_failure(error, response):
    status = {"TRANSIENT_SOURCE": 503, "SOURCE_INVALID": 200,
              "RESOURCE_LIMIT_EXCEEDED": 200}.get(error.code)
    if status is not None:
        response(status)


def _validate_body(body, request):
    if type(body) is not bytes:
        raise fail("SOURCE_INVALID", "Massive response body must be bytes")
    if len(body) > massive_plan.MAX_RAW_BYTES:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive response exceeds byte policy")
    if not massive_plan._complete_body(body, request):
        code = "SOURCE_NOT_FINAL" if json.loads(body).get("next_url") is not None else "SOURCE_EMPTY"
        raise fail(code, "Massive response is not a complete nonempty terminal page")


def _response_body(conn, store, unit, fetch, charge, response, was_charged):
    if unit["raw_receipt_id"] is not None:
        return load_raw_receipt(conn, store, unit["raw_receipt_id"])
    try:
        status, body = fetch(unit["request"], charge)
    except OpsError as error:
        if was_charged():
            _record_fetch_failure(error, response)
        raise
    if not was_charged():
        raise fail("INTEGRITY_FAILED", "Massive transport omitted durable accounting")
    code = response(status)
    if status == 403:
        raise fail("SOURCE_INVALID", "Massive request access is unavailable",
                   details={"reason": "CAPABILITY_UNAVAILABLE"})
    if status != 200:
        raise fail(code or "SOURCE_INVALID", "Massive HTTP request was refused")
    return body


def _acquire(conn, store, params, claim, *, fetch, clock, on_receipt=None,
             page_request=None, previous_requests=()):
    def fence():
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())

    with transaction(conn):
        fence()
    reusable = params.generation in (None, "initial") and page_request is None
    planned = scope_plan(params, conn if reusable else None, store)
    unit = planned["units"][0]
    original = unit["request"]
    if page_request is not None:
        unit = unit | {"request": page_request, "request_hash": content_hash(page_request)}
    receipt_id = unit["raw_receipt_id"]
    cache_hit = receipt_id is not None
    interval = 60 // planned["requests_per_minute"] + 1
    charged = False

    def charge():
        nonlocal charged
        provider_budget.before_request(conn, claim, massive_plan.SOURCE, clock=clock,
                                       minimum_interval_seconds=interval, max_calls_per_job=params.max_http_calls)
        charged = True

    def response(status):
        # A 403 denies this request's capability, not the account's credential.
        return provider_budget.record_response(
            conn, massive_plan.SOURCE, 400 if status == 403 else status, clock=clock,
            fence_check=fence, minimum_interval_seconds=interval)

    if not cache_hit or on_receipt is not None:
        body = _response_body(conn, store, unit, fetch, charge, response, lambda: charged)
        continuation = None
        if on_receipt is not None:
            continuation = validate_page(body, original, unit["request"], cached=cache_hit)
            if continuation is not None and content_hash(continuation) in (*previous_requests, unit["request_hash"]):
                raise fail("SOURCE_INVALID", "Massive continuation repeats a previous request")
        else:
            _validate_body(body, unit["request"])
        def record_receipt(record):
            observed = record.received_at if cache_hit else format_timestamp(clock.now())
            if on_receipt is not None:
                closed_day(unit["request"], parse_timestamp(observed), cached=cache_hit)
                on_receipt(record, observed, cache_hit, continuation)
        record = cache_raw_receipt(
            conn, store, RawPayload(payload=body, response_kind="partial" if continuation else "complete", response_meta={}),
            source=massive_plan.SOURCE, endpoint=unit["request"]["endpoint"], request=unit["request"],
            received_at=format_timestamp(clock.now()), fence_check=fence, record_receipt=record_receipt)
        receipt_id = record.raw_receipt_id
    with transaction(conn):
        fence()
        row = conn.execute("SELECT raw_hash,received_at FROM data_raw_receipts WHERE raw_receipt_id=?",
                           (receipt_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "Massive acquired receipt disappeared")
    return {"schema_version": SCHEMA, "request_hash": unit["request_hash"],
            "raw_receipt_id": receipt_id, "raw_hash": row["raw_hash"],
            "first_received_at": row["received_at"], "observed_at": format_timestamp(clock.now()),
            "attempt_id": claim.attempt_id, "fence": claim.fence, "cache_hit": cache_hit,
            "http_attempts": int(charged), "policy_version": params.policy_version}


def _history(conn, store, params, claim, *, fetch, clock):
    request = scope_plan(params)["units"][0]["request"]
    closed_day(request, clock.now())
    with transaction(conn):
        _, attempt = verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    planned = coverage.plan(conn, store, request, params.generation)
    if planned["gaps"]:
        if planned["candidate_count"] + len(planned["gaps"]) > coverage.MAX_OCCURRENCES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive coverage completion capacity would be exceeded")
        gap = planned["gaps"][0]
        scoped = replace(params, start=gap["start"], end=gap["end"], expected_ids=(content_hash(gap),))
        chain = coverage.progress(conn, store, gap, params.generation, attempt["job_id"])
        if chain["next_request"] is None:
            raise fail("INTEGRITY_FAILED", "Massive terminal journal lacks a completion occurrence")
        if len(chain["page_ids"]) >= journal.MAX_PAGES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive page limit exhausted")
        continuing = False
        def persist(record, observed, reused, following):
            nonlocal continuing
            continuing = following is not None
            evidence = {"schema_version": coverage.VERSION, "policy_version": params.policy_version,
                        "job_id": attempt["job_id"], "generation": params.generation, "request": gap,
                        "raw_receipt_id": record.raw_receipt_id, "raw_hash": record.raw_hash,
                        "first_received_at": record.received_at, "origin": "raw_reuse" if reused else "http_response",
                        "observed_at": observed, "attempt_id": claim.attempt_id, "fence": claim.fence}
            if chain["page_ids"] or continuing:
                page = evidence | {"schema_version": journal.VERSION, "scope_hash": content_hash(gap),
                                   "request": chain["next_request"], "next_request": following,
                                   "page_index": len(chain["page_ids"]),
                                   "previous_page_id": chain["page_ids"][-1] if chain["page_ids"] else None}
                evidence["pages"] = chain["page_ids"] + [journal.append(conn, page)]
            if not continuing:
                coverage.record(conn, evidence)
        _acquire(conn, store, scoped, claim, fetch=fetch, clock=clock, on_receipt=persist,
                 page_request=chain["next_request"] if chain["page_ids"] else None,
                 previous_requests=chain["requests"])
        if continuing:
            raise fail("SOURCE_NOT_FINAL", "Massive acquisition has a pending continuation")
        planned = coverage.plan(conn, store, request, params.generation)
    if planned["gaps"]:
        raise fail("SOURCE_NOT_FINAL", "Massive acquisition has remaining interval gaps")
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    return {"schema_version": HISTORY_SCHEMA, "coverage": planned,
            "attempt_id": claim.attempt_id, "fence": claim.fence}


def run(parameters, root, envelope, *, fetch=None, clock=None):
    try:
        params = from_document(Parameters, parameters)
    except DocumentError:
        raise fail("INVALID_REQUEST", "invalid Massive acquisition parameters") from None
    scope_plan(params)
    clock = clock or SystemClock()
    validate_as_of(params.as_of, clock)
    if not all(envelope.get(k) for k in ("attempt_id", "fence", "catalog_path", "objects_root")):
        raise fail("INVALID_REQUEST", "Massive acquisition requires trusted launch context")
    if fetch is None:
        from engine.v2.ops.providers.massive import fetch
    try:
        with closing(connect(envelope["catalog_path"], must_exist=True)) as conn:
            acquire = _history if params.generation is not None else _acquire
            result = acquire(conn, ArtifactStore(envelope["objects_root"]), params,
                              SimpleNamespace(attempt_id=envelope["attempt_id"], fence=envelope["fence"]),
                              fetch=fetch, clock=clock)
        Path(root, "massive_acquisition.json").write_text(canonical_json(result))
    except (DataError, ArtifactError):
        raise fail("INTEGRITY_FAILED", "Massive raw cache evidence is unusable") from None
    except (OSError, sqlite3.Error):
        raise fail("RESOURCE_UNAVAILABLE", "Massive acquisition storage is unavailable") from None
    schema = HISTORY_SCHEMA if params.generation is not None else SCHEMA
    return {"outputs": [{"name": KIND, "path": "massive_acquisition.json", "schema": schema}],
            "completed_ids": list(params.expected_ids), "no_work": False}
