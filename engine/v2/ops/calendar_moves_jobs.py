"""S4C Part 3: the natively-owned calendar/moves refresh job kind.

``computed_moves_refresh`` is an ordinary ``JobKind`` entry
(``stages.py::_core_kinds``), dispatched by ``worker.py`` to
``run_computed_moves_worker`` below. It decodes the job's own
``CalendarMovesParameters`` document and adapts Part 1's standalone runner
(``computed_moves_store.run_computed_moves_refresh`` -- which is not itself
``RefreshCallback``-shaped, since ``main``'s ``RefreshParameters`` has no
``as_of``/``tickers`` field) into a closure that IS. The shared
provider-receipt cache and failure classification this store (and this
module) imports live in ``engine.v2.ops.unit_receipts`` (P6 slice-4c split,
Part 0); this module re-exports them under their original names for anything
that still imports them from here.

``nightly.py``'s ``GRAPH`` carries a ``"computed_moves_refresh": ("refresh",)``
node and ``OPTIONAL`` includes it (Part 4), but no nightly submission path
ever builds a job for it: it is submitted only by ``supervisor.Service``'s
own tick-loop sidecar, after a native ``"refresh"`` job has already
succeeded -- see ``ARCHITECTURE.md`` "Outputs"/"Failure semantics".
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
COMPUTED_MOVES_RESULT_PATH = "computed_moves_refresh_result.json"
#: This kind returns the shared refresh-evidence document; the checkpoint
#: contract names the same schema `incremental_refresh` uses.
COMPUTED_MOVES_RESULT_SCHEMA = REFRESH_RESULT_SCHEMA

__all__ = [
    "COMPUTED_MOVES_REFRESH_ACTION",
    "COMPUTED_MOVES_RESULT_PATH",
    "COMPUTED_MOVES_RESULT_SCHEMA",
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
    "provider_failure_code",
    "record_unit_receipt",
    "run_computed_moves_worker",
]


@dataclass(frozen=True, kw_only=True)
class CalendarMovesParameters:
    """Strict parameters for one natively-owned calendar/moves refresh job.

    ``main``'s ``incremental_data.RefreshParameters`` has no ``as_of`` field,
    so this job family carries its own parameters dataclass with it, plus
    every field ``RefreshParameters`` already has (the plan-binding fields
    mirror it exactly). ``tickers``/``horizon_days`` were removed once
    ``forward_calendar_refresh``'s job-kind wiring was pulled from this PR
    (they had no remaining reader); ``table_name`` was never read -- see
    ``ARCHITECTURE.md``. ``expected_ids`` is the worker coverage denominator
    the supervisor checks the result against: one id per computed_moves
    target ticker.
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
    as_of: str | None = None
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

    Reuses ``incremental_data``'s own submit-time checks directly (Opus
    review, PR #50 round 3, finding 1): the plan binding
    (``parent_snapshot_id``/``refresh_plan_hash``/``provider_calls``) is now
    ALWAYS validated, never only when a binding field happens to be
    supplied -- an unbound job used to be admitted only to fail inside the
    worker.
    ``as_of`` must be a non-None, real ISO date -- ``computed_moves_store``'s
    own ``_as_of_day`` always refuses ``None`` too, so an admitted job with
    ``as_of=None`` used to be admitted only to fail inside the worker, the same
    pattern already fixed for the plan binding. ``catalog_path``/``objects_root``/``scope`` and
    ``expected_head_generation``/``expected_head_snapshot_id`` are validated
    here too, exactly like ``incremental_data.refresh_parameter_problems``
    validates them for ``incremental_refresh`` -- this really is now "the
    same layering", not merely described as one: `computed_moves_store`'s own
    revalidation of these same fields (`_validate_input_document`) is a
    second, defense-in-depth layer, the same relationship
    `run_refresh_worker` has to `refresh_parameter_problems`, not the only
    place they are ever checked. ``job.provider_budget_ref`` is read (was
    previously ignored) to enforce provider-budget/call-count consistency,
    and ``COMPUTED_MOVES_RESULT_PATH`` may never be bound as this job's own
    input (the worker writes it itself).
    """
    from engine.v2.ops import incremental_data
    problems = (incremental_data._expected_ids_problems(params)
               + incremental_data._plan_binding_problems(params)
               + incremental_data._bounded_nonempty_problems((
                   ("catalog_path", params.catalog_path, 4096),
                   ("objects_root", params.objects_root, 4096),
                   ("scope", params.scope, 128)))
               + incremental_data._head_binding_problems(params)
               + incremental_data._refresh_budget_problems(
                   job, params, result_path=COMPUTED_MOVES_RESULT_PATH))
    if not _is_iso_date(params.as_of):
        problems.append("as_of must be an ISO date")
    return tuple(problems)


def _job_kind(name: str, schema: str) -> JobKind:
    return JobKind(
        name=name, worker=name, parameters=CalendarMovesParameters,
        resource_classes=frozenset({"io_fetch"}), effects=("staged",),
        retry=RetryPolicy("bounded", 3, (5, 65)), checkpoint_contract=schema,
        namespaces=frozenset({"shadow", "smoke"}),
        validate=calendar_moves_parameter_problems)


def computed_moves_job_kind() -> JobKind:
    return _job_kind(COMPUTED_MOVES_REFRESH_ACTION, COMPUTED_MOVES_RESULT_SCHEMA)


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
        retry_policy_ref="bounded", checkpoint_contract_ref=COMPUTED_MOVES_RESULT_SCHEMA,
        dependency_job_ids=tuple(dependency_job_ids))


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


def _validate_calendar_moves_coverage(params: CalendarMovesParameters, result) -> None:
    """Same contract as ``incremental_data._validate_refresh_coverage``, but
    set-based rather than ordered-tuple: ``computed_moves_store`` never
    promises to return ``completed_ids`` in the caller's ``expected_ids``
    order (its "complete"/"noop" paths all report ``tuple(sorted(targets))``,
    alphabetical, not caller order) -- an ordered comparison would fail a
    correctly covered, already-committed result on order alone.
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
    """Every argument the one public worker function above already resolved:
    a decoded ``params`` and a bound ``(parameters, root)``-shaped callback.

    Unlike ``run_daily_market_refresh``,
    ``computed_moves_store.run_computed_moves_refresh`` writes no result
    artifact of its own -- it simply returns a ``RefreshCallbackResult`` --
    so this function writes ``root / result_path`` itself, after validating,
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
