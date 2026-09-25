"""Phase 3B operations integration for incremental EOD refreshes.

This module owns coordination decisions, not normalized data.  It turns a
cache inventory into provider-budget demand, classifies acquisition outcomes,
binds Phase 2 dependency explanations to one immutable snapshot, computes a
conservative invalidation plan, and gates the existing snapshot promotion
primitive.  It deliberately does not write coverage watermarks or manifests:
the data-layer candidate commit must do those atomically.

The refresh itself remains a normal supervised job.  :func:`refresh_job_kind`
and :func:`refresh_job_spec` describe it using the existing ``JobKind`` /
``JobSpec`` contracts, so the existing scheduler owns resource admission,
provider-account exclusivity, quota reservation, retries and process recovery.
There is no second executor or budget ledger here.
"""
from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Protocol, Sequence

from engine.v2.contracts import DataQuery, DependencyPlan, JobSpec, SnapshotRef
from engine.v2.foundation import (
    DocumentError,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.errors import fail
from engine.v2.ops.snapshot_promotion import promote, rollback
from engine.v2.ops.submission import JobKind, RetryPolicy

OutcomeKind = Literal[
    "complete", "empty", "partial", "unsupported", "not_final",
    "credential_invalid", "rate_limited", "transient",
]
RetryAction = Literal["use_cache", "stop", "retry_missing", "retry"]
DependencyMode = Literal["point", "bounded", "suffix", "full", "unknown"]
UnknownPolicy = Literal["invalidate", "refuse"]

REFRESH_RESULT_PATH = "incremental_refresh_result.json"
REFRESH_RESULT_SCHEMA = "incremental_refresh_result.v1.0"
REFRESH_PLAN_SCHEMA = "incremental_refresh_plan.v1.0"
MAX_REFRESH_IDS = 4096
MAX_REFRESH_RESULT_BYTES = 1 << 20


@dataclass(frozen=True, kw_only=True)
class RefreshUnit:
    """One cacheable source request and its explicit coverage denominator."""

    request_id: str
    table_name: str
    partition_key: str
    expected_keys: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class AcquisitionOutcome:
    """Redacted acquisition result used for retry and commit admission."""

    request_id: str
    kind: OutcomeKind
    requested_keys: tuple[str, ...]
    returned_keys: tuple[str, ...] = ()
    empty_keys: tuple[str, ...] = ()
    unsupported_keys: tuple[str, ...] = ()
    receipt_ref: str | None = None
    raw_hash: str | None = None
    cache_hit: bool = False
    quota_remaining: int | None = None


@dataclass(frozen=True, kw_only=True)
class RefreshPlan:
    """Immutable cache-first plan handed to the normal supervisor path."""

    parent_snapshot_id: str
    expected_head_generation: int
    units: tuple[RefreshUnit, ...]
    cached: tuple[AcquisitionOutcome, ...]
    fetch_units: tuple[RefreshUnit, ...]
    provider_account: str | None
    provider_calls: int
    max_attempts: int
    plan_hash: str


@dataclass(frozen=True)
class RefreshParameters:
    """Strict parameters for a supervised incremental refresh job.

    ``catalog_path``/``objects_root``/``scope``/``expected_head_generation``/
    ``expected_head_snapshot_id``/``table_name`` are the deployment identity the
    executor stages into ``incremental_refresh_input.json`` (S4A); they come
    from the admitted, hashed ``JobSpec``, never the environment or a legacy
    path. The acquired-data fields (raw payloads, revisions, coverage) are the
    worker callback's own output.
    """

    expected_ids: tuple[str, ...]
    parent_snapshot_id: str
    refresh_plan_hash: str
    provider_calls: int
    catalog_path: str
    objects_root: str
    scope: str
    expected_head_generation: int
    expected_head_snapshot_id: str | None = None
    table_name: str = "daily_market"
    provider_account: str | None = None
    input_bindings: dict[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class RefreshCallbackResult:
    """Ops-owned evidence returned and staged by the data refresh callback.

    This document carries only committed candidate identity and job binding;
    it deliberately has no dependency on data-layer candidate classes. Bulk
    rows and raw provider payloads stay in the staging directory.
    ``warnings`` records a degradation the job took knowingly (for example a
    calendar fallback), so it is evidence in the result, not only a log line.
    """

    status: Literal[
        "complete", "noop", "partial", "credential_invalid", "rate_limited",
        "not_final", "transient", "failed",
    ]
    completed_ids: tuple[str, ...]
    coverage_advanced: bool
    parent_snapshot_id: str
    refresh_plan_hash: str
    candidate_snapshot_id: str | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = REFRESH_RESULT_SCHEMA


def refresh_result_document(result: RefreshCallbackResult) -> dict:
    """The staged result document; an empty ``warnings`` is omitted.

    ``warnings`` is degradation evidence, not a field every run has. Omitting
    it when empty keeps every result document without a degradation
    byte-identical to what the schema produced before the field existed (and
    any hash over those bytes unchanged); a reader of a document without the
    key gets the field's own empty default.
    """
    document = to_document(result)
    if not result.warnings:
        document.pop("warnings", None)
    return document


class RefreshCallback(Protocol):
    """Structural boundary implemented by the data layer."""

    def __call__(self, parameters: RefreshParameters, root: Path) \
            -> RefreshCallbackResult | Mapping[str, object]:
        ...


@dataclass(frozen=True, kw_only=True)
class CommitAdmission:
    """Coverage evidence that may be passed to a data-layer candidate commit."""

    admitted: bool
    coverage_receipt_refs: tuple[str, ...]
    raw_hashes: tuple[str, ...]
    incomplete_request_ids: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class BoundExplanation:
    """A query dependency plan pinned to the caller-resolved snapshot."""

    consumer_id: str
    table_name: str
    query: DataQuery
    plan: DependencyPlan


@dataclass(frozen=True, kw_only=True)
class DataChange:
    """Ops view of a data-layer changeset entry.

    ``columns_known=False`` and ``dependency_known=False`` are explicit.  An
    absent fact never becomes an empty change.
    """

    change_id: str
    table_name: str
    revision_kind: Literal["append", "correction", "deletion", "schema"]
    columns: tuple[str, ...] = ()
    columns_known: bool = True
    start: str | None = None
    end_exclusive: str | None = None
    keys: tuple[str, ...] = ()
    dependency_known: bool = True
    old_hash: str | None = None
    new_hash: str | None = None


@dataclass(frozen=True, kw_only=True)
class DependencyRule:
    """A downstream recipe rule supplied by its owner."""

    consumer_id: str
    table_names: tuple[str, ...]
    mode: DependencyMode
    columns: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Invalidation:
    consumer_id: str
    change_ids: tuple[str, ...]
    scope: Literal["point", "bounded", "suffix", "full"]
    start: str | None
    end_exclusive: str | None
    reason: str


@dataclass(frozen=True, kw_only=True)
class ImpactPlan:
    snapshot_id: str
    invalidations: tuple[Invalidation, ...]
    unaffected_consumers: tuple[str, ...]
    conservative: bool


@dataclass(frozen=True, kw_only=True)
class CandidatePromotion:
    """The exact compare-and-swap expectations for one refresh candidate."""

    candidate_scope: str
    target_scope: str
    candidate_snapshot_id: str
    expected_target_snapshot_id: str
    expected_target_generation: int
    comparison_receipt_id: str
    commit_admission: CommitAdmission


@dataclass(frozen=True, kw_only=True)
class RecoveryDecision:
    action: Literal["already_promoted", "retry_promotion", "refuse_conflict"]
    snapshot_id: str
    generation: int


def classify_response(status: int, requested_keys: Sequence[str], *,
                      returned_keys: Sequence[str] = (), empty_keys: Sequence[str] = (),
                      unsupported_keys: Sequence[str] = (), final: bool = True,
                      truncated: bool = False, credential_page: bool = False,
                      request_id: str = "", receipt_ref: str | None = None,
                      raw_hash: str | None = None, cache_hit: bool = False,
                      quota_remaining: int | None = None) -> AcquisitionOutcome:
    """Classify one response without treating HTTP 200 as complete coverage."""
    requested = _keys(requested_keys, "requested_keys")
    returned = _subset(returned_keys, requested, "returned_keys")
    empty = _subset(empty_keys, requested, "empty_keys")
    unsupported = _subset(unsupported_keys, requested, "unsupported_keys")
    if quota_remaining is not None and quota_remaining < 0:
        raise fail("INVALID_REQUEST", "negative provider quota observation")
    kind = _response_kind(status, requested, returned, empty, unsupported,
                          final=final, truncated=truncated,
                          credential_page=credential_page)
    return AcquisitionOutcome(
        request_id=request_id, kind=kind, requested_keys=requested,
        returned_keys=returned, empty_keys=empty, unsupported_keys=unsupported,
        receipt_ref=receipt_ref, raw_hash=raw_hash, cache_hit=cache_hit,
        quota_remaining=quota_remaining)


def _response_kind(status, requested, returned, empty, unsupported, *, final, truncated,
                   credential_page) -> OutcomeKind:
    if credential_page or status in (401, 403):
        return "credential_invalid"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "transient"
    if status == 404:
        return "unsupported"
    if not 200 <= status < 300:
        return "transient"
    if not final:
        return "not_final"
    covered = set(returned) | set(empty) | set(unsupported)
    if truncated or covered != set(requested):
        return "partial"
    if requested and set(empty) == set(requested):
        return "empty"
    return "complete"


def retry_action(outcome: AcquisitionOutcome) -> RetryAction:
    """Cache hits stop network use; partial responses retry only missing keys."""
    if outcome.cache_hit and coverage_complete(outcome):
        return "use_cache"
    if outcome.kind in ("complete", "empty", "unsupported", "credential_invalid"):
        return "stop"
    if outcome.kind == "partial":
        return "retry_missing"
    return "retry"


def missing_keys(outcome: AcquisitionOutcome) -> tuple[str, ...]:
    covered = set(outcome.returned_keys) | set(outcome.empty_keys) \
        | set(outcome.unsupported_keys)
    return tuple(key for key in outcome.requested_keys if key not in covered)


def coverage_complete(outcome: AcquisitionOutcome) -> bool:
    if outcome.kind not in ("complete", "empty", "unsupported"):
        return False
    return not missing_keys(outcome)


def plan_refresh(parent_snapshot: SnapshotRef, units: Sequence[RefreshUnit], *,
                 cached_outcomes: Mapping[str, AcquisitionOutcome], provider_account: str | None,
                 expected_head_generation: int, max_attempts: int = 3,
                 calls_per_unit: int = 1) -> RefreshPlan:
    """Plan cache misses and reserve every possible provider attempt up front."""
    if max_attempts < 1:
        raise fail("INVALID_REQUEST", "refresh max_attempts must be positive")
    if not isinstance(expected_head_generation, int) or expected_head_generation < 0:
        raise fail("INVALID_REQUEST", "refresh expected_head_generation must be non-negative")
    ordered = tuple(sorted(units, key=lambda unit: unit.request_id))
    if len({unit.request_id for unit in ordered}) != len(ordered):
        raise fail("INVALID_REQUEST", "refresh request ids must be unique")
    cached, fetch = [], []
    for unit in ordered:
        _validate_unit(unit)
        outcome = cached_outcomes.get(unit.request_id)
        if outcome is not None and _cache_satisfies(unit, outcome):
            cached.append(outcome)
        else:
            fetch.append(unit)
    calls = len(fetch) * max_attempts * calls_per_unit
    if calls and not provider_account:
        raise fail("INVALID_REQUEST", "cache misses require a shared provider account")
    payload = {
        "parent_snapshot_id": parent_snapshot.snapshot_id,
        "expected_head_generation": expected_head_generation,
        "units": [to_document(unit) for unit in ordered],
        "cached_receipts": [outcome.receipt_ref for outcome in cached],
        "provider_account": provider_account, "provider_calls": calls,
        "max_attempts": max_attempts,
    }
    return RefreshPlan(
        parent_snapshot_id=parent_snapshot.snapshot_id,
        expected_head_generation=expected_head_generation, units=ordered,
        cached=tuple(cached), fetch_units=tuple(fetch),
        provider_account=provider_account if calls else None, provider_calls=calls,
        max_attempts=max_attempts, plan_hash=content_hash(payload))


def refresh_job_kind() -> JobKind:
    """Allowlist entry to add to the existing supervisor registry."""
    return JobKind(
        name="incremental_refresh", worker="incremental_refresh",
        parameters=RefreshParameters, resource_classes=frozenset({"io_fetch"}),
        effects=("staged",), retry=RetryPolicy("bounded", 3, (5, 65)),
        checkpoint_contract="incremental_refresh_result.v1.0",
        namespaces=frozenset({"shadow", "smoke"}),
        validate=refresh_parameter_problems)


def _refresh_identity_problems(params: RefreshParameters) -> list[str]:
    problems = []
    if not params.expected_ids or len(params.expected_ids) > MAX_REFRESH_IDS:
        problems.append("expected_ids must contain 1..4096 request ids")
    if (len(set(params.expected_ids)) != len(params.expected_ids)
            or any(not item or len(item) > 128 for item in params.expected_ids)):
        problems.append("expected_ids must be unique bounded nonempty strings")
    if not params.parent_snapshot_id or len(params.parent_snapshot_id) > 128:
        problems.append("parent_snapshot_id must be a bounded nonempty string")
    if not _is_hash(params.refresh_plan_hash):
        problems.append("refresh_plan_hash must be a sha256 content hash")
    if not isinstance(params.provider_calls, int) or not 0 <= params.provider_calls <= 1_000_000:
        problems.append("provider_calls must be between zero and 1000000")
    problems.extend(_refresh_staging_identity_problems(params))
    return problems


def _refresh_staging_identity_problems(params: RefreshParameters) -> list[str]:
    """S4A: the attempt's deployment identity must be reproducible and bounded."""
    problems = []
    bounded = (("catalog_path", params.catalog_path, 4096),
               ("objects_root", params.objects_root, 4096),
               ("scope", params.scope, 128), ("table_name", params.table_name, 128))
    for name, value, limit in bounded:
        if not isinstance(value, str) or not value or len(value) > limit:
            problems.append(f"{name} must be a bounded nonempty string")
    if not isinstance(params.expected_head_generation, int) \
            or params.expected_head_generation < 0:
        problems.append("expected_head_generation must be a non-negative integer")
    head = params.expected_head_snapshot_id
    if head is not None and (not isinstance(head, str) or not head or len(head) > 128):
        problems.append("expected_head_snapshot_id must be a bounded nonempty string")
    return problems


def _refresh_budget_problems(job: JobSpec, params: RefreshParameters) -> list[str]:
    problems = []
    if params.provider_calls and not job.provider_budget_ref:
        problems.append("provider calls require a shared provider budget")
    if not params.provider_calls and job.provider_budget_ref:
        problems.append("a provider budget requires at least one planned call")
    bindings = params.input_bindings or {}
    if REFRESH_RESULT_PATH in bindings:
        problems.append("the refresh output path may not be an input binding")
    return problems


def refresh_parameter_problems(job: JobSpec, params: RefreshParameters) -> tuple[str, ...]:
    """Semantic checks layered on the strict dataclass document decoder."""
    return tuple(_refresh_identity_problems(params) + _refresh_budget_problems(job, params))


def refresh_job_spec(plan: RefreshPlan, *, implementation_ref: str,
                     environment_ref: str, output_namespace: str,
                     catalog_path: str, objects_root: str,
                     input_bindings: Mapping[str, str] | None = None,
                     input_refs: Sequence[str] = ()) -> JobSpec:
    """Build a job whose normal scheduler reserves shared provider budget.

    ``catalog_path``/``objects_root`` are the attempt's deployment identity;
    ``output_namespace`` is also the refresh scope. ``input_refs`` must admit
    every direct artifact named by ``input_bindings`` (e.g. the bound
    ``refresh_plan.json``), exactly like every other kind.
    """
    params = RefreshParameters(
        expected_ids=tuple(unit.request_id for unit in plan.units),
        parent_snapshot_id=plan.parent_snapshot_id,
        refresh_plan_hash=plan.plan_hash, provider_calls=plan.provider_calls,
        catalog_path=catalog_path, objects_root=objects_root,
        scope=output_namespace, expected_head_generation=plan.expected_head_generation,
        expected_head_snapshot_id=plan.parent_snapshot_id,
        provider_account=plan.provider_account,
        input_bindings=dict(input_bindings) if input_bindings is not None else None)
    return JobSpec(
        kind="incremental_refresh", implementation_ref=implementation_ref,
        spec_hash=None, environment_ref=environment_ref,
        parameters=to_document(params), input_refs=tuple(input_refs),
        output_namespace=output_namespace, resource_class="io_fetch",
        provider_budget_ref=plan.provider_account, retry_policy_ref="bounded",
        checkpoint_contract_ref="incremental_refresh_result.v1.0")


def run_refresh_worker(parameters: Mapping[str, object], root: Path, *,
                       refresh_callback: RefreshCallback | None = None) -> dict:
    """Run the data-owned refresh callback behind the fixed worker protocol.

    The default callback is loaded lazily so ops registry construction does
    not import provider adapters. The data layer owns acquisition,
    normalization and candidate staging; ops owns strict parameters, bounded
    result transport and typed retry/failure semantics.
    """
    try:
        params = from_document(RefreshParameters, dict(parameters))
    except DocumentError as exc:
        raise fail("INVALID_REQUEST", "incremental refresh parameters are malformed",
                   details={"field": "parameters" + exc.path[1:]}) from None
    callback = refresh_callback or _load_data_refresh_callback()
    result = validate_refresh_result_document(callback(params, root))
    _validate_callback_result(params, result, root)
    if result.status not in ("complete", "noop"):
        raise fail(_failure_for_refresh_status(result.status),
                   "incremental refresh did not produce complete coverage")
    return {
        "outputs": [{"name": "incremental_refresh", "path": REFRESH_RESULT_PATH,
                     "schema": REFRESH_RESULT_SCHEMA}],
        "completed_ids": list(result.completed_ids),
        "no_work": result.status == "noop",
    }


def _load_data_refresh_callback() -> RefreshCallback:
    """Resolve the public callback at worker runtime, without data candidate types.

    The ops layer injects the native ORATS provider fetcher: the returned
    partial binds the daily_market fetch wrapper (S4A) to
    ``orats_daily_market_fetcher()``, which turns the staged identity document
    and the bound ``refresh_plan.json`` into acquired data and then delegates
    to the unchanged ``run_incremental_refresh`` commit path. Constructing the
    fetcher reads no credentials and touches no network; the key is read only
    when the fetcher is called.
    """
    from engine.v2.data.incremental import run_daily_market_refresh
    from engine.v2.ops.providers import orats_daily_market_fetcher
    return functools.partial(run_daily_market_refresh,
                             fetcher=orats_daily_market_fetcher())


def _load_computed_moves_refresh_callback() -> RefreshCallback:
    """S4C: resolve the computed_moves callback and its yfinance history edge.

    Same lazy shape as ``_load_data_refresh_callback``: constructing the
    fetcher reads nothing and touches no network, and the data-owning store is
    imported only when the worker actually dispatches this kind.
    """
    from engine.v2.ops.computed_moves_store import run_computed_moves_refresh
    from engine.v2.ops.providers import yfinance_history_fetcher
    return functools.partial(run_computed_moves_refresh,
                             fetcher=yfinance_history_fetcher())


def _load_forward_calendar_refresh_callback() -> RefreshCallback:
    """S4C: resolve the forward_calendar callback and its two network edges."""
    from engine.v2.ops.forward_calendar_store import run_forward_calendar_refresh
    from engine.v2.ops.providers import nasdaq_calendar_fetcher, yfinance_earnings_fetcher
    return functools.partial(run_forward_calendar_refresh,
                             nasdaq_fetcher=nasdaq_calendar_fetcher(),
                             earnings_fetcher=yfinance_earnings_fetcher())


def validate_refresh_result_document(value) -> RefreshCallbackResult:
    """Strictly decode the data callback's small, versioned evidence document."""
    document = to_document(value) if isinstance(value, RefreshCallbackResult) else value
    if (not isinstance(document, Mapping)
            or document.get("schema_version") != REFRESH_RESULT_SCHEMA):
        raise fail("VALIDATION_FAILED",
                   "incremental data refresh returned unsupported evidence")
    try:
        result = from_document(RefreshCallbackResult, document)
    except (DocumentError, TypeError):
        raise fail("VALIDATION_FAILED",
                   "incremental data refresh returned malformed evidence") from None
    if not result.parent_snapshot_id or len(result.parent_snapshot_id) > 128:
        raise fail("VALIDATION_FAILED",
                   "incremental data refresh evidence has an invalid parent")
    if not _is_hash(result.refresh_plan_hash):
        raise fail("VALIDATION_FAILED",
                   "incremental data refresh evidence has an invalid plan hash")
    if (result.candidate_snapshot_id is not None
            and (not result.candidate_snapshot_id
                 or len(result.candidate_snapshot_id) > 128)):
        raise fail("VALIDATION_FAILED",
                   "incremental data refresh evidence has an invalid candidate")
    return result


def _validate_callback_result(params: RefreshParameters, result: RefreshCallbackResult,
                              root: Path) -> None:
    staged = _staged_refresh_result(root)
    if to_document(staged) != to_document(result):
        raise fail("INTEGRITY_FAILED",
                   "staged refresh evidence differs from the callback result")
    _validate_refresh_binding(params, result)
    _validate_refresh_coverage(params, result)
    _validate_refresh_status(result)


def _validate_refresh_binding(params, result):
    if (result.parent_snapshot_id, result.refresh_plan_hash) != (
            params.parent_snapshot_id, params.refresh_plan_hash):
        raise fail("STALE_EXPECTATION",
                   "incremental refresh evidence is bound to different inputs")


def _validate_refresh_coverage(params, result):
    expected = params.expected_ids
    if result.status in ("complete", "noop"):
        if result.completed_ids != expected or len(set(result.completed_ids)) != len(expected):
            raise fail("VALIDATION_FAILED", "incremental refresh coverage differs",
                       details={"field": "completed_ids"})
    elif (len(set(result.completed_ids)) != len(result.completed_ids)
          or not set(result.completed_ids).issubset(expected)):
        raise fail("VALIDATION_FAILED", "incomplete refresh reported unknown coverage",
                   details={"field": "completed_ids"})


def _validate_refresh_status(result):
    if result.status not in ("complete", "noop") and result.coverage_advanced:
        raise fail("INTEGRITY_FAILED", "incomplete refresh advanced completed coverage")
    if result.status not in ("complete",) and result.candidate_snapshot_id is not None:
        raise fail("INTEGRITY_FAILED",
                   "non-complete refresh reported a committed candidate")
    if result.status == "noop" and result.coverage_advanced:
        raise fail("INTEGRITY_FAILED", "no-op refresh advanced completed coverage")
    if result.status == "complete":
        if result.candidate_snapshot_id is None:
            raise fail("VALIDATION_FAILED",
                       "complete refresh has no committed candidate snapshot")
        if not result.coverage_advanced:
            raise fail("INTEGRITY_FAILED",
                       "complete refresh did not advance candidate coverage")


def _staged_refresh_result(root: Path) -> RefreshCallbackResult:
    output = root / REFRESH_RESULT_PATH
    if not output.is_file():
        raise fail("VALIDATION_FAILED", "incremental refresh result artifact is missing")
    if output.stat().st_size > MAX_REFRESH_RESULT_BYTES:
        raise fail("VALIDATION_FAILED", "incremental refresh result artifact is too large")
    try:
        document = json.loads(output.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError):
        raise fail("VALIDATION_FAILED",
                   "incremental refresh result artifact is malformed") from None
    return validate_refresh_result_document(document)


def _failure_for_refresh_status(status: str) -> str:
    return {
        "partial": "TRANSIENT_SOURCE",
        "credential_invalid": "CREDENTIAL_INVALID",
        "rate_limited": "RATE_LIMITED",
        "not_final": "SOURCE_NOT_FINAL",
        "transient": "TRANSIENT_SOURCE",
        "failed": "WORKER_FAILED",
    }[status]


def _is_hash(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        return False
    return all(char in "0123456789abcdef" for char in value[7:])


def admit_candidate_commit(plan: RefreshPlan,
                           outcomes: Sequence[AcquisitionOutcome]) -> CommitAdmission:
    """Refuse a candidate commit unless every denominator is complete.

    This is the ops guard that keeps failure/partial outcomes away from the
    data-layer commit that advances completed coverage watermarks.
    """
    combined = {outcome.request_id: outcome for outcome in (*plan.cached, *outcomes)}
    incomplete, receipts, hashes = [], [], []
    for unit in plan.units:
        outcome = combined.get(unit.request_id)
        if outcome is None or not _cache_satisfies(unit, outcome) or not outcome.receipt_ref:
            incomplete.append(unit.request_id)
            continue
        receipts.append(outcome.receipt_ref)
        if outcome.raw_hash is not None:
            hashes.append(outcome.raw_hash)
    return CommitAdmission(
        admitted=not incomplete, coverage_receipt_refs=tuple(receipts) if not incomplete else (),
        raw_hashes=tuple(hashes) if not incomplete else (),
        incomplete_request_ids=tuple(incomplete))


def explain_snapshot_queries(repository, snapshot_ref: SnapshotRef,
                             queries: Sequence[tuple[str, str, DataQuery]]) \
        -> tuple[BoundExplanation, ...]:
    """Explain queries against exactly ``snapshot_ref``, never a moving head."""
    explanations = []
    for consumer_id, table_name, query in queries:
        if query.snapshot_id != snapshot_ref.snapshot_id:
            raise fail("STALE_EXPECTATION", "dependency query is not pinned to the resolved snapshot")
        plan = repository.explain_dependencies(query, table_name=table_name)
        if plan.snapshot_ref.snapshot_id != snapshot_ref.snapshot_id:
            raise fail("STALE_EXPECTATION", "repository explained a different snapshot")
        explanations.append(BoundExplanation(
            consumer_id=consumer_id, table_name=table_name, query=query, plan=plan))
    return tuple(explanations)


def plan_dependency_impacts(snapshot_ref: SnapshotRef, changes: Sequence[DataChange],
                            explanations: Sequence[BoundExplanation],
                            rules: Sequence[DependencyRule], *,
                            unknown_policy: UnknownPolicy = "invalidate") -> ImpactPlan:
    """Translate data changes into complete downstream invalidation.

    Unknown change dependencies or unknown recipe modes become full rebuilds;
    callers may choose a typed refusal instead.  They can never produce an
    empty impact plan.
    """
    if unknown_policy not in ("invalidate", "refuse"):
        raise fail("INVALID_REQUEST", "unknown dependency policy")
    explained = _explanations_by_consumer(snapshot_ref, explanations)
    invalidations, unaffected, conservative = [], [], False
    for rule in sorted(rules, key=lambda item: item.consumer_id):
        relevant = _relevant_changes(rule, changes, explained.get(rule.consumer_id, ()))
        unknown = rule.mode == "unknown" or any(not change.dependency_known for change in relevant)
        if unknown:
            conservative = True
            if unknown_policy == "refuse":
                raise fail("VALIDATION_FAILED", "downstream dependency is unknown",
                           details={"consumer_id": rule.consumer_id})
            relevant = relevant or tuple(changes)
            invalidations.append(_full_invalidation(rule.consumer_id, relevant,
                                                     "unknown dependency"))
        elif relevant:
            invalidations.append(_invalidation_for(rule, relevant))
        else:
            unaffected.append(rule.consumer_id)
    return ImpactPlan(
        snapshot_id=snapshot_ref.snapshot_id, invalidations=tuple(invalidations),
        unaffected_consumers=tuple(unaffected), conservative=conservative)


def promote_refresh_candidate(conn, store, candidate: CandidatePromotion, *, clock,
                              promote_hook: Callable = promote):
    """Promote through the existing CAS only after complete data admission."""
    if not candidate.commit_admission.admitted:
        raise fail("VALIDATION_FAILED", "partial refresh candidate cannot be promoted",
                   details={"incomplete_count": len(
                       candidate.commit_admission.incomplete_request_ids)})
    return promote_hook(
        conn, store, candidate_scope=candidate.candidate_scope,
        target_scope=candidate.target_scope,
        expected_snapshot_id=candidate.expected_target_snapshot_id,
        expected_generation=candidate.expected_target_generation,
        comparison_receipt_id=candidate.comparison_receipt_id, clock=clock)


def recovery_decision(conn, candidate: CandidatePromotion) -> RecoveryDecision:
    """Classify an interrupted promotion without guessing or deleting state."""
    row = conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (candidate.target_scope,)).fetchone()
    if row is None:
        raise fail("SNAPSHOT_NOT_READY", "promotion target has no committed head")
    snapshot_id, generation = row["snapshot_id"], row["generation"]
    if snapshot_id == candidate.candidate_snapshot_id:
        action = "already_promoted"
    elif (snapshot_id, generation) == (
            candidate.expected_target_snapshot_id, candidate.expected_target_generation):
        action = "retry_promotion"
    else:
        action = "refuse_conflict"
    return RecoveryDecision(action=action, snapshot_id=snapshot_id, generation=generation)


def rollback_refresh_candidate(conn, store, *, scope: str, to_snapshot_id: str,
                               expected_snapshot_id: str, expected_generation: int, clock,
                               rollback_hook: Callable = rollback):
    """Recover by the existing immutable-snapshot rollback CAS."""
    return rollback_hook(
        conn, store, scope=scope, to_snapshot_id=to_snapshot_id,
        expected_snapshot_id=expected_snapshot_id,
        expected_generation=expected_generation, clock=clock)


def _keys(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    keys = tuple(sorted(values))
    if any(not value for value in keys) or len(set(keys)) != len(keys):
        raise fail("INVALID_REQUEST", f"{field_name} must contain unique nonempty keys")
    return keys


def _subset(values, requested, field_name):
    keys = _keys(values, field_name)
    if not set(keys).issubset(requested):
        raise fail("INVALID_REQUEST", f"{field_name} contains an unrequested key")
    return keys


def _validate_unit(unit: RefreshUnit) -> None:
    if not unit.request_id or not unit.table_name or not unit.partition_key:
        raise fail("INVALID_REQUEST", "refresh unit identity is incomplete")
    _keys(unit.expected_keys, "expected_keys")


def _cache_satisfies(unit: RefreshUnit, outcome: AcquisitionOutcome) -> bool:
    return (outcome.request_id == unit.request_id
            and outcome.requested_keys == _keys(unit.expected_keys, "expected_keys")
            and coverage_complete(outcome))


def _explanations_by_consumer(snapshot_ref, explanations):
    grouped: dict[str, list[BoundExplanation]] = {}
    for explanation in explanations:
        if explanation.plan.snapshot_ref.snapshot_id != snapshot_ref.snapshot_id:
            raise fail("STALE_EXPECTATION", "impact explanation is bound to another snapshot")
        grouped.setdefault(explanation.consumer_id, []).append(explanation)
    return {key: tuple(value) for key, value in grouped.items()}


def _relevant_changes(rule, changes, explanations):
    explained_tables = {entry.table_name for item in explanations
                        for entry in item.plan.dependencies}
    tables = set(rule.table_names) | explained_tables
    out = []
    for change in changes:
        if change.table_name not in tables:
            continue
        known_columns = set(rule.columns)
        for item in explanations:
            for entry in item.plan.dependencies:
                if entry.table_name == change.table_name:
                    known_columns.update(entry.columns)
        if change.columns_known and known_columns \
                and set(change.columns).isdisjoint(known_columns):
            continue
        out.append(change)
    return tuple(out)


def _invalidation_for(rule, changes):
    if rule.mode == "full":
        return _full_invalidation(rule.consumer_id, changes, "full dependency")
    starts = [change.start for change in changes if change.start is not None]
    ends = [change.end_exclusive for change in changes if change.end_exclusive is not None]
    scope = "suffix" if rule.mode == "suffix" else rule.mode
    end = None if scope in ("suffix", "full") else (max(ends) if ends else None)
    return Invalidation(
        consumer_id=rule.consumer_id,
        change_ids=tuple(change.change_id for change in changes), scope=scope,
        start=min(starts) if starts else None, end_exclusive=end,
        reason="declared dependency")


def _full_invalidation(consumer_id, changes, reason):
    return Invalidation(
        consumer_id=consumer_id,
        change_ids=tuple(change.change_id for change in changes), scope="full",
        start=None, end_exclusive=None, reason=reason)
