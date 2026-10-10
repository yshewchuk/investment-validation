"""S4C Part 3: the natively-owned calendar/moves refresh job kinds.

``computed_moves_refresh`` and ``forward_calendar_refresh`` are ordinary
``JobKind`` entries (``stages.py::_core_kinds``), each dispatched by
``worker.py`` to its own ``run_*_worker`` below. Each decodes the job's own
``CalendarMovesParameters`` document and adapts its store's standalone runner
into a ``RefreshCallback``-shaped closure: Part 1's
``computed_moves_store.run_computed_moves_refresh`` (which is not itself
``RefreshCallback``-shaped, since ``main``'s ``RefreshParameters`` has no
``as_of``/``tickers`` field), and
``forward_calendar_store.run_forward_calendar_refresh`` (an explicit, fully
keyword-only runner). The latter's closure reads ``attempt_id``/``fence``
from ``forward_calendar_refresh``'s own small staged document
(``refresh_staging.py``), since those vary per attempt and cannot be
pre-bound on the job's immutable ``CalendarMovesParameters``. The shared
provider-receipt cache and failure classification this store (and this
module) imports live in ``engine.v2.ops.unit_receipts`` (P6 slice-4c split,
Part 0); this module re-exports them under their original names for anything
that still imports them from here.

``nightly.py``'s ``GRAPH`` carries a ``"computed_moves_refresh": ("refresh",)``
node and ``OPTIONAL`` includes it (Part 4), but no nightly submission path
ever builds a job for it: it is submitted only by ``supervisor.Service``'s
own tick-loop sidecar, after a native ``"refresh"`` job has already
succeeded -- see ``ARCHITECTURE.md`` "Outputs"/"Failure semantics".
``forward_calendar_refresh`` gets neither a GRAPH/OPTIONAL node nor a
supervisor submitter: that wiring is a separate, later PR.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

from engine.v2.contracts import JobSpec
from engine.v2.foundation import (
    DocumentError,
    SystemClock,
    canonical_json,
    from_document,
    to_document,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.incremental_data import REFRESH_RESULT_SCHEMA, RefreshCallback
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
FORWARD_CALENDAR_REFRESH_ACTION = "forward_calendar_refresh"
FORWARD_CALENDAR_RESULT_PATH = "forward_calendar_refresh_result.json"
#: Both kinds return the shared refresh-evidence document; each checkpoint
#: contract names the same schema `incremental_refresh` uses.
FORWARD_CALENDAR_RESULT_SCHEMA = REFRESH_RESULT_SCHEMA
#: forward_calendar_refresh's own horizon default when a caller's
#: ``CalendarMovesParameters`` document omits ``horizon_days``.
DEFAULT_HORIZON_DAYS = 21

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

    ``main``'s ``incremental_data.RefreshParameters`` has no ``as_of`` field,
    so this job family carries its own parameters dataclass with it, plus
    every field ``RefreshParameters`` already has (the plan-binding fields
    mirror it exactly). ``horizon_days``/``tickers`` exist for
    ``forward_calendar_refresh`` only (``computed_moves_refresh`` never reads
    either); ``table_name`` was never read -- see ``ARCHITECTURE.md``. An
    empty ``tickers`` tuple is meaningful to the standalone runner (it means
    the whole market), but this job kind's own coverage check requires a
    non-empty ``expected_ids``, so a whole-market run cannot be submitted as
    a job today (only a ticker-scoped one can); the standalone runner remains
    the only way to run a whole-market forward-calendar refresh directly.
    ``expected_ids`` is the worker coverage denominator the supervisor checks
    the result against: one id per target ticker.
    """

    expected_ids: tuple[str, ...]
    parent_snapshot_id: str = ""
    parent_receipt_id: str | None = None
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
    horizon_days: int = DEFAULT_HORIZON_DAYS
    tickers: tuple[str, ...] = ()


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


def _result_path_for(job) -> str:
    """The result artifact the job's own worker writes.

    A ``forward_calendar_refresh`` job must refuse a binding of its OWN
    output path; every other kind (and a ``None``/test-double job) falls back
    to the computed-moves path.
    """
    if getattr(job, "kind", None) == FORWARD_CALENDAR_REFRESH_ACTION:
        return FORWARD_CALENDAR_RESULT_PATH
    return COMPUTED_MOVES_RESULT_PATH


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
    and this job's OWN result path (``COMPUTED_MOVES_RESULT_PATH`` for
    ``computed_moves_refresh``, ``FORWARD_CALENDAR_RESULT_PATH`` for
    ``forward_calendar_refresh``) may never be bound as this job's own input
    (the worker writes it itself). The final two checks --
    ``horizon_days`` (a real ``int`` inside ``[1, MAX_HORIZON_DAYS]``) and
    ``tickers`` (a tuple/list of unique bounded non-empty strings that,
    whenever non-empty, must also match ``expected_ids`` as a set, and which
    is additionally refused when EMPTY for a ``forward_calendar_refresh`` job
    specifically -- never for ``computed_moves_refresh``, which never reads
    the field and always leaves it at its empty default) -- apply
    to BOTH job kinds: they are harmless for ``computed_moves_refresh``,
    which never reads either field, since their defaults
    (``DEFAULT_HORIZON_DAYS`` and an empty tuple) both already pass.
    """
    from engine.v2.ops import incremental_data
    from engine.v2.ops.forward_calendar_store import MAX_HORIZON_DAYS

    problems = (incremental_data._expected_ids_problems(params)
               + incremental_data._plan_binding_problems(params)
               + incremental_data._bounded_nonempty_problems((
                   ("catalog_path", params.catalog_path, 4096),
                   ("objects_root", params.objects_root, 4096),
                   ("scope", params.scope, 128)))
               + incremental_data._head_binding_problems(params)
               + incremental_data._refresh_budget_problems(
                   job, params, result_path=_result_path_for(job)))
    if not _is_iso_date(params.as_of):
        problems.append("as_of must be an ISO date")
    if (isinstance(params.horizon_days, bool)
            or not isinstance(params.horizon_days, int)
            or not 1 <= params.horizon_days <= MAX_HORIZON_DAYS):
        problems.append(f"horizon_days must be an int between 1 and {MAX_HORIZON_DAYS}")
    tickers = params.tickers
    if not isinstance(tickers, (tuple, list)):
        problems.append("tickers must be a tuple or list of ticker symbols")
    elif (any(not isinstance(item, str) or not item or len(item) > 128 for item in tickers)
          or len(set(tickers)) != len(tickers)):
        problems.append("tickers must be unique bounded nonempty strings")
    elif not tickers and getattr(job, "kind", None) == FORWARD_CALENDAR_REFRESH_ACTION:
        problems.append("tickers must be a non-empty ticker selection for a forward_calendar_refresh job")
    elif tickers and set(tickers) != set(params.expected_ids):
        problems.append("tickers must match expected_ids for a ticker-scoped forward calendar refresh")
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


def forward_calendar_job_kind() -> JobKind:
    return _job_kind(FORWARD_CALENDAR_REFRESH_ACTION, FORWARD_CALENDAR_RESULT_SCHEMA)


def _schema_for(kind: str) -> str:
    return (COMPUTED_MOVES_RESULT_SCHEMA if kind == COMPUTED_MOVES_REFRESH_ACTION
            else FORWARD_CALENDAR_RESULT_SCHEMA)


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


# --------------------------------------------------------------------------
# the worker entrypoints: each adapts its store's standalone runner into the
# RefreshCallback shape incremental_data.run_refresh_worker already uses.
# --------------------------------------------------------------------------


def _load_computed_moves_refresh_callback(as_of: str | None) -> RefreshCallback:
    """S4C Part 3: resolve the computed_moves callback and its yfinance edge.

    ``as_of`` cannot be pre-bound the way the fetcher is: it varies per job
    dispatch (a session date), never a constant of the deployment the way
    the injected fetcher is. Same lazy shape as ``_load_data_refresh_callback``:
    constructing the fetcher reads nothing and touches no network, and the
    data-owning store is imported only when the worker actually dispatches
    this kind.
    """
    from engine.v2.ops.computed_moves_store import run_computed_moves_refresh
    from engine.v2.ops.providers import yfinance_history_fetcher
    return functools.partial(run_computed_moves_refresh, as_of=as_of,
                             fetcher=yfinance_history_fetcher())


def _load_forward_calendar_refresh_callback() -> RefreshCallback:
    """S4C follow-up: resolve the forward_calendar callback and its two network edges.

    ``forward_calendar_store.run_forward_calendar_refresh`` takes no
    ``(parameters, root)`` pair at all -- it is a standalone runner with an
    explicit, fully keyword-only signature -- so a bare ``functools.partial``
    cannot make it ``RefreshCallback``-shaped the way
    ``_load_data_refresh_callback`` does for ``run_daily_market_refresh``; this
    closure reads every keyword argument off the decoded
    ``CalendarMovesParameters`` except ``attempt_id``/``fence``, which vary per
    attempt (a retried attempt gets a new fence) and so cannot be pre-bound the
    way the fetchers are: they come from ``forward_calendar_refresh``'s own
    small staged document
    (``refresh_staging.REFRESH_INPUT_DOCUMENT_NAMES["forward_calendar_refresh"]``),
    read from ``root`` at call time. Before that file read, or any other I/O,
    ``parameters.expected_head_snapshot_id`` is format-checked with the same
    bounded-1..128-char-or-``None`` one-line shape check
    ``computed_moves_store._validate_document_head`` uses for the same field
    (mirrored, not imported: that name is private, matching this codebase's
    existing convention of mirroring rather than importing another module's
    private validators); constructing the fetchers reads nothing and touches no
    network, the same lazy shape ``_load_computed_moves_refresh_callback``
    already has.
    """
    from engine.v2.ops.forward_calendar_store import (
        NATIVE_NASDAQ_ACCOUNT,
        NATIVE_YFINANCE_ACCOUNT,
        run_forward_calendar_refresh,
    )
    from engine.v2.ops.provider_budget import budgeted_fetcher
    from engine.v2.ops.providers import nasdaq_calendar_fetcher, yfinance_earnings_fetcher
    from engine.v2.ops.stores.refresh_contracts import REFRESH_INPUT_DOCUMENT_NAMES

    nasdaq_fetcher = nasdaq_calendar_fetcher()
    earnings_fetcher = yfinance_earnings_fetcher()
    document_name = REFRESH_INPUT_DOCUMENT_NAMES["forward_calendar_refresh"]

    def _callback(parameters, root):
        head = parameters.expected_head_snapshot_id
        if head is not None and (not isinstance(head, str) or not head or len(head) > 128):
            raise fail("INVALID_REQUEST",
                       "expected_head_snapshot_id must be a bounded nonempty string")
        attempt_id, fence = _staged_forward_calendar_attempt(Path(root), document_name)
        budget = dict(catalog_path=parameters.catalog_path, attempt_id=attempt_id,
                      fence=fence, clock=SystemClock())
        return run_forward_calendar_refresh(
            catalog_path=parameters.catalog_path,
            objects_root=parameters.objects_root,
            parent_snapshot_id=parameters.parent_snapshot_id,
            refresh_plan_hash=parameters.refresh_plan_hash,
            as_of=parameters.as_of,
            tickers=parameters.tickers,
            horizon_days=parameters.horizon_days,
            scope=parameters.scope,
            expected_head_generation=parameters.expected_head_generation,
            expected_head_snapshot_id=parameters.expected_head_snapshot_id,
            attempt_id=attempt_id, fence=fence,
            nasdaq_fetcher=budgeted_fetcher(nasdaq_fetcher, account=NATIVE_NASDAQ_ACCOUNT, **budget),
            earnings_fetcher=budgeted_fetcher(earnings_fetcher, account=NATIVE_YFINANCE_ACCOUNT, **budget))

    return _callback


def _staged_forward_calendar_attempt(root: Path, document_name: str) \
        -> tuple[str | None, int | None]:
    """Read forward_calendar_refresh's tiny staged document (``attempt_id``/
    ``fence`` only -- see ``refresh_staging._forward_calendar_document``) and
    format-check both fields before returning them. A missing or malformed
    document is a typed ``INVALID_REQUEST``, never a bare ``KeyError``/
    ``JSONDecodeError``: this runner is always dispatched through the
    job-submission pipeline once registered, so the document is always
    expected to exist.
    """
    path = root / document_name
    try:
        document = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise fail("INVALID_REQUEST",
                   "forward calendar refresh staged input document is missing or malformed") \
            from exc
    if not isinstance(document, Mapping):
        raise fail("INVALID_REQUEST",
                   "forward calendar refresh staged input document is malformed")
    if "attempt_id" not in document or document["attempt_id"] is None \
            or "fence" not in document or document["fence"] is None:
        raise fail("INVALID_REQUEST",
                   "the staged document must include both attempt_id and fence for a "
                   "job-scheduled forward calendar refresh")
    attempt_id = document["attempt_id"]
    if not isinstance(attempt_id, str) or not attempt_id:
        raise fail("INVALID_REQUEST", "attempt_id must be a non-empty string")
    fence = document["fence"]
    if isinstance(fence, bool) or not isinstance(fence, int) or fence < 1:
        raise fail("INVALID_REQUEST", "fence must be an int of at least 1")
    return attempt_id, fence


def _decode(parameters) -> CalendarMovesParameters:
    try:
        return from_document(CalendarMovesParameters, dict(parameters))
    except DocumentError as exc:
        raise fail("INVALID_REQUEST", "calendar/moves refresh parameters are malformed",
                   details={"field": "parameters" + exc.path[1:]}) from None


def run_computed_moves_worker(parameters, root, *, refresh_callback=None) -> dict:
    params = _decode(parameters)
    callback = refresh_callback or _load_computed_moves_refresh_callback(
        params.as_of)
    return _run_calendar_moves_worker(
        params, root, kind=COMPUTED_MOVES_REFRESH_ACTION,
        result_path=COMPUTED_MOVES_RESULT_PATH, schema=COMPUTED_MOVES_RESULT_SCHEMA,
        callback=callback,
        failure_message="computed moves refresh did not produce complete coverage")


def run_forward_calendar_worker(parameters, root, *, refresh_callback=None) -> dict:
    """Decode the job's parameters, resolve the forward_calendar callback and
    run the shared calendar/moves worker for the forward_calendar kind."""
    params = _decode(parameters)
    callback = refresh_callback or _load_forward_calendar_refresh_callback()
    return _run_calendar_moves_worker(
        params, root, kind=FORWARD_CALENDAR_REFRESH_ACTION,
        result_path=FORWARD_CALENDAR_RESULT_PATH, schema=FORWARD_CALENDAR_RESULT_SCHEMA,
        callback=callback,
        failure_message="forward calendar refresh did not produce complete coverage")


def _validate_calendar_moves_coverage(params: CalendarMovesParameters, result) -> None:
    """Same contract as ``incremental_data._validate_refresh_coverage``, but
    set-based rather than ordered-tuple: BOTH stores share this property --
    ``computed_moves_store`` never promises to return ``completed_ids`` in the
    caller's ``expected_ids`` order (its "complete"/"noop" paths all report
    ``tuple(sorted(targets))``, alphabetical, not caller order), and
    ``forward_calendar_store.run_forward_calendar_refresh`` also always
    reports ``completed_ids`` as the sorted tuple of the unique ticker set on
    every terminal status -- an ordered comparison would fail a correctly
    covered, already-committed result on order alone.
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
    """Every argument the TWO public worker functions above already resolved:
    a decoded ``params`` and a bound ``(parameters, root)``-shaped callback.

    Unlike ``run_daily_market_refresh``, neither
    ``computed_moves_store.run_computed_moves_refresh`` nor
    ``forward_calendar_store.run_forward_calendar_refresh`` writes its own
    result artifact -- each simply returns a ``RefreshCallbackResult`` --
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
