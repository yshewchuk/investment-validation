"""S4C: the two natively-owned calendar/moves refresh job kinds.

``computed_moves_refresh`` and ``forward_calendar_refresh`` are new nightly job
kinds, not ``incremental_refresh`` table variants: each has its own staged
identity document (``engine.v2.ops.refresh_staging``), its own result path and
its own injected provider fetchers (resolved lazily by
``incremental_data._load_*_refresh_callback``). This module owns the shared
job-kind contract -- parameters, validator, ``JobSpec`` builder and the worker
entrypoints -- plus the raw-receipt cache both stores and the nightly plan
builders resolve cache hits through.

Provider accounts: ``yfinance`` (computed_moves, and the forward calendar's
session confirmation) and ``nasdaq`` (the forward calendar's discovery calls)
are unmetered and keyless; their credential tuples are empty, but their budget
rows are still operator-provisioned so the shared scheduler reserves against
them like any other account.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Sequence

from engine.v2.contracts import JobSpec
from engine.v2.foundation import (
    DocumentError,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.incremental_data import REFRESH_RESULT_SCHEMA
from engine.v2.ops.submission import JobKind, RetryPolicy

COMPUTED_MOVES_REFRESH_ACTION = "computed_moves_refresh"
FORWARD_CALENDAR_REFRESH_ACTION = "forward_calendar_refresh"
COMPUTED_MOVES_RESULT_PATH = "computed_moves_refresh_result.json"
FORWARD_CALENDAR_RESULT_PATH = "forward_calendar_refresh_result.json"
#: Both kinds return the shared refresh-evidence document; the checkpoint
#: contract names the same schema.
COMPUTED_MOVES_RESULT_SCHEMA = REFRESH_RESULT_SCHEMA
FORWARD_CALENDAR_RESULT_SCHEMA = REFRESH_RESULT_SCHEMA
NATIVE_COMPUTED_MOVES_ACCOUNT = "yfinance"
#: One budget account backs every yfinance call, whichever job makes it.
NATIVE_YFINANCE_ACCOUNT = NATIVE_COMPUTED_MOVES_ACCOUNT
NATIVE_NASDAQ_ACCOUNT = "nasdaq"
DEFAULT_HORIZON_DAYS = 21
MAX_CALENDAR_MOVES_IDS = 4096
MAX_IDENTITY_LENGTH = 128
MAX_PROVIDER_CALLS = 1_000_000

__all__ = [
    "COMPUTED_MOVES_REFRESH_ACTION",
    "COMPUTED_MOVES_RESULT_PATH",
    "COMPUTED_MOVES_RESULT_SCHEMA",
    "DEFAULT_HORIZON_DAYS",
    "FORWARD_CALENDAR_REFRESH_ACTION",
    "FORWARD_CALENDAR_RESULT_PATH",
    "FORWARD_CALENDAR_RESULT_SCHEMA",
    "NATIVE_COMPUTED_MOVES_ACCOUNT",
    "NATIVE_NASDAQ_ACCOUNT",
    "NATIVE_YFINANCE_ACCOUNT",
    "CalendarMovesParameters",
    "cached_unit_outcomes",
    "calendar_moves_job_spec",
    "calendar_moves_parameter_problems",
    "computed_moves_job_kind",
    "forward_calendar_job_kind",
    "record_unit_receipt",
    "run_computed_moves_worker",
    "run_forward_calendar_worker",
]


@dataclass(frozen=True, kw_only=True)
class CalendarMovesParameters:
    """Strict parameters for one natively-owned calendar/moves refresh job.

    ``expected_ids`` is the worker coverage denominator the supervisor checks
    the result against: one id per computed_moves target ticker, or per wanted
    forward-calendar ticker. The plan-binding fields mirror
    ``incremental_data.RefreshParameters``; ``as_of``/``horizon_days``/
    ``tickers`` are what the forward calendar's staged identity needs.
    """

    expected_ids: tuple[str, ...]
    parent_snapshot_id: str = ""
    refresh_plan_hash: str = ""
    provider_calls: int = 0
    catalog_path: str = ""
    objects_root: str = ""
    scope: str = "shadow"
    expected_head_generation: int = 0
    expected_head_snapshot_id: str | None = None
    provider_account: str | None = None
    input_bindings: dict[str, str] | None = None
    table_name: str = ""
    as_of: str | None = None
    horizon_days: int = DEFAULT_HORIZON_DAYS
    tickers: tuple[str, ...] = ()
    all_scoreable: bool = True
    since: str | None = None


def _is_hash(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(char in "0123456789abcdef" for char in value[7:])


def _is_iso_date(value: object) -> bool:
    from datetime import date

    text = str(value)
    if len(text) != 10:
        return False
    try:
        date.fromisoformat(text)
    except ValueError:
        return False
    return True


def calendar_moves_parameter_problems(job, params: CalendarMovesParameters) -> tuple[str, ...]:
    """Semantic checks layered on the strict dataclass document decoder.

    The plan binding is validated only when supplied, so a caller that pins a
    plan gets all-or-nothing checks while an absent binding never invents a
    failure; ``expected_ids`` and a supplied ``as_of`` are always validated.
    """
    problems = []
    if not params.expected_ids or len(params.expected_ids) > MAX_CALENDAR_MOVES_IDS:
        problems.append("expected_ids must contain 1..4096 request ids")
    elif (len(set(params.expected_ids)) != len(params.expected_ids)
          or any(not item or len(item) > MAX_IDENTITY_LENGTH for item in params.expected_ids)):
        problems.append("expected_ids must be unique bounded nonempty strings")
    if params.as_of is not None and not _is_iso_date(params.as_of):
        problems.append("as_of must be an ISO date")
    if params.parent_snapshot_id or params.refresh_plan_hash or params.provider_calls:
        problems.extend(_plan_binding_problems(params))
    return tuple(problems)


def _plan_binding_problems(params: CalendarMovesParameters) -> list[str]:
    problems = []
    if not params.parent_snapshot_id or len(params.parent_snapshot_id) > MAX_IDENTITY_LENGTH:
        problems.append("parent_snapshot_id must be a bounded nonempty string")
    if not _is_hash(params.refresh_plan_hash):
        problems.append("refresh_plan_hash must be a sha256 content hash")
    if not isinstance(params.provider_calls, int) or not 0 <= params.provider_calls <= MAX_PROVIDER_CALLS:
        problems.append("provider_calls must be between zero and 1000000")
    return problems


def _job_kind(name: str, schema: str) -> JobKind:
    return JobKind(
        name=name, worker=name, parameters=CalendarMovesParameters,
        resource_classes=frozenset({"io_fetch"}), effects=("staged",),
        retry=RetryPolicy("bounded", 3, (5, 65)), checkpoint_contract=schema,
        namespaces=frozenset({"shadow", "smoke"}),
        validate=calendar_moves_parameter_problems)


def computed_moves_job_kind() -> JobKind:
    return _job_kind(COMPUTED_MOVES_REFRESH_ACTION, COMPUTED_MOVES_RESULT_SCHEMA)


def forward_calendar_job_kind() -> JobKind:
    return _job_kind(FORWARD_CALENDAR_REFRESH_ACTION, FORWARD_CALENDAR_RESULT_SCHEMA)


def calendar_moves_job_spec(kind: str, plan, parameters: CalendarMovesParameters, *,
                            implementation_ref: str, environment_ref: str,
                            output_namespace: str, catalog_path: str, objects_root: str,
                            input_bindings=None, input_refs=()) -> JobSpec:
    """The job a native calendar/moves plan becomes (S4C).

    ``expected_ids`` is the caller's own coverage denominator and stays what it
    says; the plan supplies only the provider-budget identity
    (``provider_calls``/``provider_account``/``plan_hash``) and the pinned
    parent head.
    """
    params = replace(
        parameters, parent_snapshot_id=plan.parent_snapshot_id,
        refresh_plan_hash=plan.plan_hash, provider_calls=plan.provider_calls,
        provider_account=plan.provider_account, catalog_path=catalog_path,
        objects_root=objects_root, scope=output_namespace,
        expected_head_generation=plan.expected_head_generation,
        expected_head_snapshot_id=plan.parent_snapshot_id,
        input_bindings=dict(input_bindings) if input_bindings is not None else None)
    return JobSpec(
        kind=kind, implementation_ref=implementation_ref, spec_hash=None,
        environment_ref=environment_ref, parameters=to_document(params),
        input_refs=tuple(input_refs), output_namespace=output_namespace,
        resource_class="io_fetch", provider_budget_ref=plan.provider_account,
        retry_policy_ref="bounded", checkpoint_contract_ref=_schema_for(kind))


def _schema_for(kind: str) -> str:
    if kind == COMPUTED_MOVES_REFRESH_ACTION:
        return COMPUTED_MOVES_RESULT_SCHEMA
    return FORWARD_CALENDAR_RESULT_SCHEMA


# --------------------------------------------------------------------------
# the shared raw-receipt cache
# --------------------------------------------------------------------------


def _unit_request(unit) -> dict:
    return {"request_id": unit.request_id, "table_name": unit.table_name,
            "partition_key": unit.partition_key, "keys": list(unit.expected_keys)}


def record_unit_receipt(conn, store, unit, payload: bytes, *, source: str, endpoint: str,
                        received_at: str, response_kind: str = "complete"):
    """Cache one unit's acquired bytes so a rerun resolves it without a fetch.

    ``cache_raw_receipt`` is the data layer's own idempotent raw cache: the
    receipt identity is a hash of ``(source, endpoint, request, raw)``, so a
    replay of the same bytes returns the existing row and publishes nothing.
    """
    from engine.v2.data.incremental import RawPayload, cache_raw_receipt

    return cache_raw_receipt(
        conn, store,
        RawPayload(payload=payload, response_kind=response_kind, response_meta={}),
        source=source, endpoint=endpoint, request=_unit_request(unit),
        received_at=received_at)


def cached_unit_outcomes(conn, units: Sequence, *, source: str, endpoint: str) -> dict:
    """``{request_id: AcquisitionOutcome}`` for units with a durable receipt.

    Both the nightly plan builders and the stores themselves call this, so a
    second same-catalog run plans (and acquires) exactly zero provider calls.
    """
    from engine.v2.ops.incremental_data import classify_response

    hits: dict[str, tuple[str, str]] = {}
    for row in conn.execute(
            "SELECT request_json, raw_receipt_id, raw_hash FROM data_raw_receipts "
            "WHERE source = ? AND endpoint = ?", (source, endpoint)):
        try:
            request = json.loads(row["request_json"])
        except (TypeError, ValueError):
            continue
        hits[str(request.get("request_id"))] = (row["raw_receipt_id"], row["raw_hash"])
    outcomes = {}
    for unit in units:
        hit = hits.get(unit.request_id)
        if hit is None:
            continue
        outcomes[unit.request_id] = classify_response(
            200, unit.expected_keys, returned_keys=unit.expected_keys,
            request_id=unit.request_id, receipt_ref=hit[0], raw_hash=hit[1],
            cache_hit=True)
    return outcomes


# --------------------------------------------------------------------------
# the worker entrypoints
# --------------------------------------------------------------------------


def run_computed_moves_worker(parameters, root, *, refresh_callback=None) -> dict:
    return _run_calendar_moves_worker(
        parameters, root, kind=COMPUTED_MOVES_REFRESH_ACTION,
        result_path=COMPUTED_MOVES_RESULT_PATH, schema=COMPUTED_MOVES_RESULT_SCHEMA,
        loader=_load_computed_moves_callback, callback=refresh_callback,
        failure_message="computed moves refresh did not produce complete coverage")


def run_forward_calendar_worker(parameters, root, *, refresh_callback=None) -> dict:
    return _run_calendar_moves_worker(
        parameters, root, kind=FORWARD_CALENDAR_REFRESH_ACTION,
        result_path=FORWARD_CALENDAR_RESULT_PATH, schema=FORWARD_CALENDAR_RESULT_SCHEMA,
        loader=_load_forward_calendar_callback, callback=refresh_callback,
        failure_message="forward calendar refresh did not produce complete coverage")


def _load_computed_moves_callback():
    from engine.v2.ops.incremental_data import _load_computed_moves_refresh_callback
    return _load_computed_moves_refresh_callback()


def _load_forward_calendar_callback():
    from engine.v2.ops.incremental_data import _load_forward_calendar_refresh_callback
    return _load_forward_calendar_refresh_callback()


def _decode(parameters) -> CalendarMovesParameters:
    try:
        return from_document(CalendarMovesParameters, dict(parameters))
    except DocumentError as exc:
        raise fail("INVALID_REQUEST", "calendar/moves refresh parameters are malformed",
                   details={"field": "parameters" + exc.path[1:]}) from None


def _run_calendar_moves_worker(parameters, root, *, kind, result_path, schema, loader,
                               callback, failure_message) -> dict:
    from engine.v2.ops import incremental_data

    params = _decode(parameters)
    result = incremental_data.validate_refresh_result_document(
        (callback or loader())(params, root))
    incremental_data._validate_refresh_binding(params, result)
    incremental_data._validate_refresh_coverage(params, result)
    incremental_data._validate_refresh_status(result)
    (root / result_path).write_text(canonical_json(to_document(result)))
    if result.status not in ("complete", "noop"):
        raise fail(incremental_data._failure_for_refresh_status(result.status), failure_message)
    return {
        "outputs": [{"name": kind, "path": result_path, "schema": schema}],
        "completed_ids": list(result.completed_ids),
        "no_work": result.status == "noop",
    }
