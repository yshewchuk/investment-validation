"""Submission identity, the server-owned kind allowlist, and job reads — §5.1.

A submission is a ``SubmitRequest``: a namespace, an idempotency key, a
principal and a ``JobSpec``. Nothing in it names an executable, a module or a
path. ``kind`` selects an entry in a :class:`KindRegistry` the server owns,
which supplies the worker, the permitted parameter schema, the resource
classes, effects, retry and checkpoint policy.

Idempotency is by ``(namespace, idempotency_key)`` plus the canonical digest of
the whole request:

* same key, same digest — the existing job is returned, nothing is inserted;
* same key, different digest — ``IDEMPOTENCY_CONFLICT``, nothing changes.

The job ID is derived from the namespace and key, so two submitters racing on
the same key compute the same ID, and the unique constraint in the schema is
the backstop if the transaction ordering ever were not.

Everything that can be validated without the catalog is validated before the
transaction opens: kind, namespace authority, identifier syntax, bounded
inputs, and parameters decoded strictly against the kind's schema. Graph
submission rejects cycles before inserting anything and inserts every node in
one transaction, so a conflict on one node commits none of them.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine.v2.contracts import (
    FAILURE_CODES,
    JobReceipt,
    JobSpec,
    Problem,
    ProgressEvent,
    QueueReason,
    ResolvedResources,
    SubmitRequest,
)
from engine.v2.foundation import (
    Clock,
    DocumentError,
    content_hash,
    format_timestamp,
    from_document,
    parse_timestamp,
    to_document,
)
from engine.v2.ops.catalog import (
    decode_bounded_text,
    dumps,
    load_json,
    sql_text_budget,
    transaction,
)
from engine.v2.ops.errors import fail, make_problem
from engine.v2.ops.provider_requirements import validate_scalar_provider_shape

__all__ = [
    "JobKind",
    "KindRegistry",
    "NamespacePolicy",
    "RetryPolicy",
    "get_job",
    "MAX_JOB_FAILURE_BYTES",
    "job_id_for",
    "request_digest",
    "submit",
    "submit_graph",
    "validate_receipt_read_policy",
    "validate_request",
]

_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_BLOCKING_PARENT_STATES = frozenset({"failed", "cancelled", "blocked"})
# Collector opt-in ceiling, not a constraint on historical coordinator failures.
MAX_JOB_FAILURE_BYTES = 1024 * 1024


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded attempts and backoff; persisted with the job at submission."""

    name: str
    max_attempts: int
    backoff_seconds: tuple[int, ...] = ()

    def delay_after(self, attempt_number: int) -> int:
        if not self.backoff_seconds:
            return 0
        return self.backoff_seconds[min(attempt_number, len(self.backoff_seconds)) - 1]


@dataclass(frozen=True)
class JobKind:
    """One server-owned allowlist entry (§5.1)."""

    name: str
    worker: str
    parameters: type
    resource_classes: frozenset[str]
    effects: tuple[str, ...]
    retry: RetryPolicy
    checkpoint_contract: str
    namespaces: frozenset[str]
    max_input_refs: int = 64
    validate: Callable[[JobSpec, Any], Sequence[str]] | None = None
    #: Cooperative legacy-store leases this kind needs, as ``(domain, mode)``
    #: pairs with mode ``read`` or ``write`` (§9.2). Claimed atomically with
    #: the attempt; a conflict keeps the job queued without an attempt.
    store_domains: tuple[tuple[str, str], ...] = ()


class KindRegistry:
    """The allowlist. An unknown kind is refused, never dispatched by name."""

    def __init__(self, kinds: Iterable[JobKind]) -> None:
        self._kinds: dict[str, JobKind] = {}
        for kind in kinds:
            if kind.name in self._kinds:
                raise ValueError(f"duplicate job kind {kind.name}")
            self._kinds[kind.name] = kind

    def get(self, name: str) -> JobKind:
        kind = self._kinds.get(name)
        if kind is None:
            raise fail("INVALID_REQUEST", "unsupported job kind", details={"field": "job.kind"})
        return kind

    def names(self) -> list[str]:
        return sorted(self._kinds)


@dataclass(frozen=True)
class NamespacePolicy:
    """Which principals may submit into, or depend on, which namespaces."""

    grants: dict[str, frozenset[str]] = field(default_factory=dict)

    def allows(self, principal: str, namespace: str) -> bool:
        return namespace in self.grants.get(principal, frozenset())


def request_digest(request: SubmitRequest) -> str:
    return content_hash(to_document(request))


def job_id_for(namespace: str, idempotency_key: str) -> str:
    digest = content_hash({"namespace": namespace, "idempotency_key": idempotency_key})
    return "job_" + digest.removeprefix("sha256:")[:32]


# --------------------------------------------------------------------------
# validation, outside any transaction
# --------------------------------------------------------------------------


def _token(field_name: str, value: Any) -> None:
    if not isinstance(value, str) or not _TOKEN.match(value):
        raise fail("INVALID_REQUEST", f"{field_name} is not a plain identifier",
                   details={"field": field_name})


def validate_request(registry: KindRegistry, policy: NamespacePolicy,
                     request: SubmitRequest) -> JobKind:
    """Everything checkable without the catalog. Returns the kind entry."""
    job = request.job
    try:
        from_document(SubmitRequest, to_document(request))
    except DocumentError:
        raise fail("INVALID_REQUEST", "unsupported or malformed submission schema") from None
    validate_scalar_provider_shape(job.parameters)
    for name, value in (("namespace", request.namespace),
                        ("idempotency_key", request.idempotency_key),
                        ("principal", request.principal),
                        ("job.output_namespace", job.output_namespace),
                        ("job.resource_class", job.resource_class)):
        _token(name, value)
    kind = registry.get(job.kind)
    if not policy.allows(request.principal, request.namespace) \
            or request.namespace not in kind.namespaces:
        raise fail("UNAUTHORIZED_NAMESPACE", "this principal may not submit this kind here",
                   details={"namespace": request.namespace, "kind": kind.name})
    if job.output_namespace != request.namespace:
        raise fail("UNAUTHORIZED_NAMESPACE", "output must belong to the submission namespace")
    if job.retry_policy_ref != kind.retry.name or job.checkpoint_contract_ref != kind.checkpoint_contract:
        raise fail("INVALID_REQUEST", "job policies differ from the registered contract")
    _check_job_fields(kind, job)
    return kind


def _check_job_fields(kind: JobKind, job: JobSpec) -> None:
    _reject_secret_text(job)
    if job.resource_class not in kind.resource_classes:
        raise fail("INVALID_REQUEST", "resource class not permitted for this kind",
                   details={"field": "job.resource_class"})
    if len(job.input_refs) > kind.max_input_refs:
        raise fail("INVALID_REQUEST", "too many input references; inputs are bounded",
                   details={"field": "job.input_refs", "limit": kind.max_input_refs})
    refs = (*job.input_refs, *job.dependency_job_ids)
    for index, ref in enumerate(refs):
        _token(f"job.refs[{index}]", ref)
    if len(set(job.dependency_job_ids)) != len(job.dependency_job_ids):
        raise fail("INVALID_REQUEST", "a dependency is listed twice",
                   details={"field": "job.dependency_job_ids"})
    if job.deadline_at is not None:
        try:
            parse_timestamp(job.deadline_at)
        except ValueError:
            raise fail("INVALID_REQUEST", "deadline_at is not a UTC timestamp",
                       details={"field": "job.deadline_at"}) from None
    _check_parameters(kind, job)


def _reject_secret_text(job: JobSpec) -> None:
    """Keep credentials out of canonical requests, catalog rows and receipts."""
    forbidden = ("password", "secret", "token", "api_key", "cookie", "authorization")

    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if any(word in str(key).lower() for word in forbidden):
                    raise fail("INVALID_REQUEST", "secret-bearing submission field refused",
                               details={"field": path})
                walk(item, path + "." + str(key))
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                walk(item, path + "[" + str(index) + "]")
        elif isinstance(value, str):
            lowered = value.lower()
            if any(marker in lowered for marker in (
                    "http://", "https://", "authorization:", "bearer ",
                    "api_key=", "token=", "cookie=")):
                raise fail("INVALID_REQUEST", "secret-bearing submission value refused",
                           details={"field": path})

    walk(to_document(job), "job")


def _check_parameters(kind: JobKind, job: JobSpec) -> None:
    try:
        params = from_document(kind.parameters, job.parameters)
    except DocumentError as exc:
        raise fail("INVALID_REQUEST", f"parameters refused: {exc.code}",
                   details={"field": "job.parameters" + exc.path[1:]}) from None
    problems = list(kind.validate(job, params)) if kind.validate else []
    if problems:
        raise fail("INVALID_REQUEST", "parameters failed the kind's validator",
                   details={"problems": problems})


def _topological(requests: Sequence[SubmitRequest]) -> list[SubmitRequest]:
    """Parents before children; a cycle among the submitted nodes is refused."""
    nodes: dict[str, SubmitRequest] = {}
    for request in requests:
        job_id = job_id_for(request.namespace, request.idempotency_key)
        if job_id in nodes and request_digest(nodes[job_id]) != request_digest(request):
            raise fail("IDEMPOTENCY_CONFLICT", "one submission uses a key twice with "
                       "different payloads", details={"job_id": job_id})
        nodes[job_id] = request
    order: list[SubmitRequest] = []
    marks: dict[str, str] = {}

    def visit(job_id: str, trail: tuple[str, ...]) -> None:
        if marks.get(job_id) == "done":
            return
        if marks.get(job_id) == "active" or job_id in nodes[job_id].job.dependency_job_ids:
            raise fail("INVALID_REQUEST", "dependency cycle",
                       details={"cycle": [*trail, job_id]})
        marks[job_id] = "active"
        for parent in nodes[job_id].job.dependency_job_ids:
            if parent in nodes:
                visit(parent, (*trail, job_id))
        marks[job_id] = "done"
        order.append(nodes[job_id])

    for job_id in nodes:
        visit(job_id, ())
    return order


# --------------------------------------------------------------------------
# insertion
# --------------------------------------------------------------------------


def submit(conn: sqlite3.Connection, registry: KindRegistry, policy: NamespacePolicy,
           request: SubmitRequest, *, clock: Clock) -> JobReceipt:
    return submit_graph(conn, registry, policy, [request], clock=clock)[0]


def submit_graph(conn: sqlite3.Connection, registry: KindRegistry, policy: NamespacePolicy,
                 requests: Sequence[SubmitRequest], *, clock: Clock,
                 receipt_decode_failure_code: str | None = None,
                 max_failure_bytes: int | None = None) -> list[JobReceipt]:
    """Validate every node, reject cycles, then insert all nodes or none."""
    validate_receipt_read_policy(receipt_decode_failure_code, max_failure_bytes)
    kinds = {job_id_for(r.namespace, r.idempotency_key): validate_request(registry, policy, r)
             for r in requests}
    ordered = _topological(requests)
    stamp = format_timestamp(clock.now())
    with transaction(conn):
        for request in ordered:
            job_id = job_id_for(request.namespace, request.idempotency_key)
            _insert_or_match(conn, request, kinds[job_id], policy, stamp)
        return [get_job(conn, job_id_for(r.namespace, r.idempotency_key),
                        receipt_decode_failure_code=receipt_decode_failure_code,
                        max_failure_bytes=max_failure_bytes) for r in requests]


def _insert_or_match(conn: sqlite3.Connection, request: SubmitRequest, kind: JobKind,
                     policy: NamespacePolicy, stamp: str) -> None:
    job_id = job_id_for(request.namespace, request.idempotency_key)
    digest = request_digest(request)
    existing = conn.execute("SELECT request_digest FROM jobs WHERE namespace = ? "
                            "AND idempotency_key = ?",
                            (request.namespace, request.idempotency_key)).fetchone()
    if existing is not None:
        if existing["request_digest"] != digest:
            raise fail("IDEMPOTENCY_CONFLICT", "this idempotency key was already used "
                       "with a different request", details={"job_id": job_id})
        return
    parents = _parents(conn, request, policy)
    blocked = [p["job_id"] for p in parents if p["state"] in _BLOCKING_PARENT_STATES]
    failure = make_problem("DEPENDENCY_FAILED", "an upstream job did not succeed",
                           dependency_refs=blocked) if blocked else None
    job = request.job
    conn.execute(
        "INSERT INTO jobs (job_id, namespace, idempotency_key, request_digest, principal, kind, "
        "spec_hash, spec_json, resource_class, checkpoint_contract_ref, retry_json, state, "
        "priority, deadline_at, max_attempts, created_at, updated_at, failure_json) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (job_id, request.namespace, request.idempotency_key, digest, request.principal,
         job.kind, job.spec_hash, dumps(job), job.resource_class, job.checkpoint_contract_ref,
         dumps(kind.retry), "blocked" if blocked else "queued", job.priority, job.deadline_at,
         kind.retry.max_attempts, stamp, stamp, None if failure is None else dumps(failure)))
    conn.executemany(
        "INSERT INTO job_dependencies (child_job_id, parent_job_id, required_output_contract) "
        "VALUES (?, ?, ?)",
        [(job_id, p["job_id"], p["checkpoint_contract_ref"]) for p in parents])


def _parents(conn: sqlite3.Connection, request: SubmitRequest,
             policy: NamespacePolicy) -> list[sqlite3.Row]:
    parents = []
    for parent_id in request.job.dependency_job_ids:
        row = conn.execute("SELECT job_id, namespace, state, checkpoint_contract_ref "
                           "FROM jobs WHERE job_id = ?", (parent_id,)).fetchone()
        if row is None:
            raise fail("INVALID_REQUEST", "unknown dependency",
                       details={"field": "job.dependency_job_ids"})
        if not policy.allows(request.principal, row["namespace"]):
            raise fail("UNAUTHORIZED_NAMESPACE", "dependency lives in a namespace this "
                       "principal may not use", details={"field": "job.dependency_job_ids"})
        parents.append(row)
    return parents


# --------------------------------------------------------------------------
# reads
# --------------------------------------------------------------------------


def validate_receipt_read_policy(code: str | None, max_failure_bytes: int | None = None) -> None:
    """Admit optional registered failure policy and SQL-safe budget before effects."""
    if code is not None and (not isinstance(code, str) or code not in FAILURE_CODES):
        raise fail("INVALID_REQUEST", "receipt decoder requires a registered failure code")
    if max_failure_bytes is not None and (type(max_failure_bytes) is not int
            or not 0 < max_failure_bytes <= (2**63 - 1) // 2):
        raise fail("INVALID_REQUEST", "receipt failure budget must be a positive SQL-safe integer")


def _receipt_json(cls: type, text: str | None, failure_code: str | None) -> Any:
    try:
        return load_json(cls, text)
    except (ValueError, RecursionError):
        if failure_code is None:
            raise
        raise fail(failure_code, "stored job receipt cannot be decoded") from None


def get_job(conn: sqlite3.Connection, job_id: str, *,
            receipt_decode_failure_code: str | None = None,
            max_failure_bytes: int | None = None) -> JobReceipt:
    """Read a receipt; an explicit failure-text bound admits before decoding."""
    validate_receipt_read_policy(receipt_decode_failure_code, max_failure_bytes)
    encoding, limit = sql_text_budget(conn, max_failure_bytes or MAX_JOB_FAILURE_BYTES)
    row = conn.execute(
        "SELECT job_id, namespace, idempotency_key, request_digest, kind, spec_hash, state, priority, "
        "created_at, fence, attempt_count, active_attempt_id, next_eligible_at, queue_reason_json, "
        "failure_json IS NOT NULL AS has_failure, typeof(failure_json)='text' AS failure_is_text, "
        "CASE WHEN typeof(failure_json)='text' AND (? IS NULL OR length(CAST(failure_json AS BLOB))<=?) "
        "THEN CAST(failure_json AS BLOB) END AS failure_json "
        "FROM jobs WHERE job_id=?", (max_failure_bytes, limit, job_id)).fetchone()
    if row is None:
        raise fail("INVALID_REQUEST", "unknown job", details={"reason": "not_found"})
    failure = None
    if row["has_failure"]:
        if not row["failure_is_text"]:
            raise fail("INTEGRITY_FAILED", "stored job failure requires text")
        try:
            failure = (row["failure_json"].decode(encoding) if max_failure_bytes is None else
                       decode_bounded_text(row["failure_json"], encoding, max_failure_bytes))
        except UnicodeError:
            raise fail("INTEGRITY_FAILED", "stored job failure cannot be decoded") from None
        except ValueError:
            raise fail(receipt_decode_failure_code or "INTEGRITY_FAILED",
                       "stored job failure exceeds its byte bound") from None
    try:
        failure = load_json(Problem, failure)
    except (ValueError, RecursionError):
        raise fail("INTEGRITY_FAILED", "stored job failure cannot be decoded") from None
    resources = None
    if row["active_attempt_id"] is not None:
        attempt = conn.execute("SELECT resources_json FROM attempts WHERE attempt_id = ?",
                               (row["active_attempt_id"],)).fetchone()
        resources = _receipt_json(ResolvedResources, attempt["resources_json"], receipt_decode_failure_code)
    progress = conn.execute("SELECT body_json FROM progress_events WHERE job_id = ? "
                            "ORDER BY recorded_at DESC, sequence DESC LIMIT 1",
                            (job_id,)).fetchone()
    output_rows = conn.execute(
        "SELECT DISTINCT ao.artifact_id FROM attempt_outputs ao JOIN attempts a "
        "ON a.attempt_id = ao.attempt_id WHERE a.job_id = ? ORDER BY ao.artifact_id", (job_id,)
    ).fetchall()
    checkpoint_rows = conn.execute(
        "SELECT DISTINCT c.cache_key FROM checkpoints c JOIN attempts a "
        "ON a.attempt_id = c.producer_attempt_id WHERE a.job_id = ? ORDER BY c.cache_key", (job_id,)
    ).fetchall()
    return JobReceipt(
        job_id=row["job_id"], namespace=row["namespace"],
        idempotency_key=row["idempotency_key"], request_digest=row["request_digest"],
        kind=row["kind"], spec_hash=row["spec_hash"], state=row["state"],
        priority=row["priority"], created_at=row["created_at"], fence=row["fence"],
        attempt_count=row["attempt_count"], active_attempt_id=row["active_attempt_id"],
        next_eligible_at=row["next_eligible_at"],
        queue_reason=_receipt_json(QueueReason, row["queue_reason_json"], receipt_decode_failure_code),
        resolved_resources=resources,
        checkpoint_refs=tuple(row[0] for row in checkpoint_rows),
        output_refs=tuple(row[0] for row in output_rows),
        latest_progress=None if progress is None else _receipt_json(
            ProgressEvent, progress["body_json"], receipt_decode_failure_code),
        failure=failure,
    )
