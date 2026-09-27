"""S4C Part 3: the two natively-owned calendar/moves refresh job kinds.

``computed_moves_refresh`` and ``forward_calendar_refresh`` are ordinary
``JobKind`` entries (``stages.py::_core_kinds``), dispatched by ``worker.py``
to ``run_computed_moves_worker``/``run_forward_calendar_worker`` below. Each
one decodes the job's own ``CalendarMovesParameters`` document and adapts
Parts 1/2's standalone runners (``computed_moves_store.run_computed_moves_refresh``,
``forward_calendar_store.run_forward_calendar_refresh`` -- neither of which is
itself ``RefreshCallback``-shaped, since ``main``'s ``RefreshParameters`` has
no ``as_of``/``tickers`` field) into a closure that IS. The shared
provider-receipt cache and failure classification these two stores (and this
module) import live in ``engine.v2.ops.unit_receipts`` (P6 slice-4c split,
Part 0); this module re-exports them under their original names for anything
that still imports them from here.

Neither job kind has a ``nightly.py`` ``GRAPH``/``OPTIONAL`` entry yet -- see
``ARCHITECTURE.md`` "Outputs" -- so today each is reachable only through the
general job-submission pipeline, not an ordinary nightly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

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
from engine.v2.ops.unit_receipts import (
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    NATIVE_NASDAQ_ACCOUNT,
    NATIVE_YFINANCE_ACCOUNT,
    RESPONSE_KINDS,
    cached_unit_outcomes,
    cached_unit_payloads,
    provider_failure_code,
    record_unit_receipt,
)

COMPUTED_MOVES_REFRESH_ACTION = "computed_moves_refresh"
FORWARD_CALENDAR_REFRESH_ACTION = "forward_calendar_refresh"
COMPUTED_MOVES_RESULT_PATH = "computed_moves_refresh_result.json"
FORWARD_CALENDAR_RESULT_PATH = "forward_calendar_refresh_result.json"
#: Both kinds return the shared refresh-evidence document; the checkpoint
#: contract names the same schema.
COMPUTED_MOVES_RESULT_SCHEMA = REFRESH_RESULT_SCHEMA
FORWARD_CALENDAR_RESULT_SCHEMA = REFRESH_RESULT_SCHEMA
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
    "RESPONSE_KINDS",
    "CalendarMovesParameters",
    "cached_unit_outcomes",
    "cached_unit_payloads",
    "calendar_moves_job_spec",
    "calendar_moves_parameter_problems",
    "computed_moves_job_kind",
    "forward_calendar_job_kind",
    "provider_failure_code",
    "record_unit_receipt",
    "run_computed_moves_worker",
    "run_forward_calendar_worker",
]


@dataclass(frozen=True, kw_only=True)
class CalendarMovesParameters:
    """Strict parameters for one natively-owned calendar/moves refresh job.

    ``main``'s ``incremental_data.RefreshParameters`` has no
    ``as_of``/``tickers``/``horizon_days`` field, so this job family carries
    its own parameters dataclass with them, plus every field
    ``RefreshParameters`` already has (the plan-binding fields mirror it
    exactly). ``expected_ids`` is the worker coverage denominator the
    supervisor checks the result against: one id per computed_moves target
    ticker, or per wanted forward-calendar ticker.
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
    Neither ``tickers``/``horizon_days``/``scope``/``catalog_path``/
    ``objects_root`` is re-checked here: those are the two stores' own
    responsibility, enforced again -- before any I/O -- by
    ``run_forward_calendar_refresh``'s own argument checks and
    ``computed_moves_store._validate_input_document``, the same layering
    ``incremental_data.RefreshParameters``/``refresh_parameter_problems``
    already has relative to ``run_refresh_worker``.
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
                            input_bindings=None, input_refs=(),
                            dependency_job_ids=()) -> JobSpec:
    """The job a native calendar/moves plan becomes (S4C, wired by Part 4).

    ``expected_ids`` is the caller's own coverage denominator and stays what
    it says; the plan supplies only the provider-budget identity
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
        retry_policy_ref="bounded", checkpoint_contract_ref=_schema_for(kind),
        dependency_job_ids=tuple(dependency_job_ids))


def _schema_for(kind: str) -> str:
    if kind == COMPUTED_MOVES_REFRESH_ACTION:
        return COMPUTED_MOVES_RESULT_SCHEMA
    return FORWARD_CALENDAR_RESULT_SCHEMA


# --------------------------------------------------------------------------
# the worker entrypoints: each adapts its store's standalone runner into the
# RefreshCallback shape incremental_data.run_refresh_worker already uses.
# --------------------------------------------------------------------------


def _decode(parameters) -> CalendarMovesParameters:
    try:
        return from_document(CalendarMovesParameters, dict(parameters))
    except DocumentError as exc:
        raise fail("INVALID_REQUEST", "calendar/moves refresh parameters are malformed",
                   details={"field": "parameters" + exc.path[1:]}) from None


def run_computed_moves_worker(parameters, root, *, refresh_callback=None) -> dict:
    from engine.v2.ops import incremental_data

    params = _decode(parameters)
    callback = refresh_callback or incremental_data._load_computed_moves_refresh_callback(
        params.as_of)
    return _run_calendar_moves_worker(
        params, root, kind=COMPUTED_MOVES_REFRESH_ACTION,
        result_path=COMPUTED_MOVES_RESULT_PATH, schema=COMPUTED_MOVES_RESULT_SCHEMA,
        callback=callback,
        failure_message="computed moves refresh did not produce complete coverage")


def run_forward_calendar_worker(parameters, root, *, refresh_callback=None) -> dict:
    from engine.v2.ops import incremental_data

    params = _decode(parameters)
    callback = refresh_callback or incremental_data._load_forward_calendar_refresh_callback()
    return _run_calendar_moves_worker(
        params, root, kind=FORWARD_CALENDAR_REFRESH_ACTION,
        result_path=FORWARD_CALENDAR_RESULT_PATH, schema=FORWARD_CALENDAR_RESULT_SCHEMA,
        callback=callback,
        failure_message="forward calendar refresh did not produce complete coverage")


def _validate_calendar_moves_coverage(params: CalendarMovesParameters, result) -> None:
    """Same contract as ``incremental_data._validate_refresh_coverage``, but
    set-based rather than ordered-tuple: neither calendar/moves store
    promises to return ``completed_ids`` in the caller's ``expected_ids``
    order (both sort internally on their "complete" paths, and
    ``computed_moves_store``'s nothing-rebuilt noop path orders by its own
    catalog scan) -- an ordered comparison would fail a correctly covered,
    already-committed result on order alone.
    """
    expected = set(params.expected_ids)
    completed = result.completed_ids
    if len(set(completed)) != len(completed):
        raise fail("VALIDATION_FAILED", "calendar/moves refresh coverage differs",
                   details={"field": "completed_ids"})
    if result.status in ("complete", "noop"):
        if set(completed) != expected:
            raise fail("VALIDATION_FAILED", "calendar/moves refresh coverage differs",
                       details={"field": "completed_ids"})
    elif not set(completed).issubset(expected):
        raise fail("VALIDATION_FAILED", "incomplete refresh reported unknown coverage",
                   details={"field": "completed_ids"})


def _run_calendar_moves_worker(params: CalendarMovesParameters, root, *, kind, result_path,
                               schema, callback, failure_message) -> dict:
    """Every argument the two public worker functions above already resolved:
    a decoded ``params`` and a bound ``(parameters, root)``-shaped callback.

    Unlike ``run_daily_market_refresh``, neither
    ``computed_moves_store.run_computed_moves_refresh`` nor
    ``forward_calendar_store.run_forward_calendar_refresh`` writes its own
    result artifact -- both simply return a ``RefreshCallbackResult`` -- so
    this function writes ``root / result_path`` itself, after validating,
    rather than reading one back for an integrity cross-check the way
    ``incremental_data._validate_callback_result`` does for
    ``incremental_refresh``.
    """
    from engine.v2.ops import incremental_data

    result = incremental_data.validate_refresh_result_document(callback(params, root))
    incremental_data._validate_refresh_binding(params, result)
    _validate_calendar_moves_coverage(params, result)
    incremental_data._validate_refresh_status(result)
    Path(root).joinpath(result_path).write_text(canonical_json(
        incremental_data.refresh_result_document(result)))
    if result.status not in ("complete", "noop"):
        raise fail(incremental_data._failure_for_refresh_status(result.status), failure_message)
    return {
        "outputs": [{"name": kind, "path": result_path, "schema": schema}],
        "completed_ids": list(result.completed_ids),
        "no_work": result.status == "noop",
    }
