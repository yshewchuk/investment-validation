"""Explicit one-request Massive acquisition through the existing supervisor."""
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from engine.v2.data.errors import DataError
from engine.v2.data.incremental import RawPayload, cache_raw_receipt
from engine.v2.foundation import (
    ArtifactStore,
    DocumentError,
    SystemClock,
    canonical_json,
    format_timestamp,
    from_document,
)
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops import massive_plan, provider_budget
from engine.v2.ops.catalog import connect, transaction
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.submission import JobKind, RetryPolicy

KIND = "massive_acquisition"
SCHEMA = "massive_acquisition.v1"


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


def scope_plan(params, conn=None, store=None):
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
    if job.provider_budget_ref != massive_plan.SOURCE or job.input_refs or job.dependency_job_ids:
        return ("Massive acquisition requires its fixed account and no input dependencies",)
    return ()


def job_kind():
    return JobKind(name=KIND, worker=KIND, parameters=Parameters,
                   resource_classes=frozenset({"io_fetch"}), effects=("staged",),
                   retry=RetryPolicy("bounded", 1), checkpoint_contract=SCHEMA,
                   namespaces=frozenset({"shadow"}), validate=parameter_problems)


def _record_fetch_failure(error, response):
    status = {"TRANSIENT_SOURCE": 503, "SOURCE_INVALID": 200,
              "RESOURCE_LIMIT_EXCEEDED": 200}.get(error.code)
    if status is not None:
        response(status)


def _acquire(conn, store, params, claim, *, fetch, clock):
    def fence():
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())

    with transaction(conn):
        fence()
    planned = scope_plan(params, conn, store)
    unit = planned["units"][0]
    receipt_id = unit["raw_receipt_id"]
    cache_hit = receipt_id is not None
    interval = 60 // planned["requests_per_minute"] + 1
    charged = False

    def charge():
        nonlocal charged
        provider_budget.before_request(conn, claim, massive_plan.SOURCE, clock=clock,
                                       minimum_interval_seconds=interval)
        charged = True

    def response(status):
        # A 403 denies this request's capability, not the account's credential.
        return provider_budget.record_response(
            conn, massive_plan.SOURCE, 400 if status == 403 else status, clock=clock,
            fence_check=fence, minimum_interval_seconds=interval)

    if not cache_hit:
        try:
            status, body = fetch(unit["request"], charge)
        except OpsError as error:
            if charged:
                _record_fetch_failure(error, response)
            raise
        if not charged:
            raise fail("INTEGRITY_FAILED", "Massive transport omitted durable accounting")
        code = response(status)
        if status == 403:
            raise fail("SOURCE_INVALID", "Massive request access is unavailable",
                       details={"reason": "CAPABILITY_UNAVAILABLE"})
        if status != 200:
            raise fail(code or "SOURCE_INVALID", "Massive HTTP request was refused")
        if type(body) is not bytes or len(body) > massive_plan.MAX_RAW_BYTES:
            raise fail("RESOURCE_LIMIT_EXCEEDED", "Massive response exceeds byte policy")
        if not massive_plan._complete_body(body, unit["request"]):
            code = "SOURCE_NOT_FINAL" if json.loads(body).get("next_url") is not None else "SOURCE_EMPTY"
            raise fail(code, "Massive response is not a complete nonempty terminal page")
        record = cache_raw_receipt(
            conn, store, RawPayload(payload=body, response_kind="complete", response_meta={}),
            source=massive_plan.SOURCE, endpoint=unit["request"]["endpoint"], request=unit["request"],
            received_at=format_timestamp(clock.now()), fence_check=fence)
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


def run(parameters, root, envelope, *, fetch=None, clock=None):
    try:
        params = from_document(Parameters, parameters)
    except DocumentError:
        raise fail("INVALID_REQUEST", "invalid Massive acquisition parameters") from None
    scope_plan(params)
    if not all(envelope.get(k) for k in ("attempt_id", "fence", "catalog_path", "objects_root")):
        raise fail("INVALID_REQUEST", "Massive acquisition requires trusted launch context")
    if fetch is None:
        from engine.v2.ops.providers.massive import fetch
    try:
        with closing(connect(envelope["catalog_path"], must_exist=True)) as conn:
            result = _acquire(conn, ArtifactStore(envelope["objects_root"]), params,
                              SimpleNamespace(attempt_id=envelope["attempt_id"], fence=envelope["fence"]),
                              fetch=fetch, clock=clock or SystemClock())
        Path(root, "massive_acquisition.json").write_text(canonical_json(result))
    except (DataError, ArtifactError):
        raise fail("INTEGRITY_FAILED", "Massive raw cache evidence is unusable") from None
    except (OSError, sqlite3.Error):
        raise fail("RESOURCE_UNAVAILABLE", "Massive acquisition storage is unavailable") from None
    return {"outputs": [{"name": KIND, "path": "massive_acquisition.json", "schema": SCHEMA}],
            "completed_ids": list(params.expected_ids), "no_work": result["cache_hit"]}
