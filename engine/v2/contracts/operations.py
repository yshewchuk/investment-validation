"""Operational envelopes: failures, progress, resources and capacity.

Phase-1 guide §5.2, §5.3 and §8; component contracts §2.4 and §11.1.

Schemas only. Every type here is an Envelope in the sense of contracts §2.5:
operational metadata that is **excluded from every content hash**. Changing a
thread count, a CPU assignment or a heartbeat must never change the identity of
a score, a checkpoint's economic content or an experiment hypothesis.

No validation, no hashing and no I/O live here; ``engine.v2.foundation.typed``
decodes these shapes strictly, and ``engine.v2.ops`` gives them behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = [
    "CAPACITY_SAMPLE_V1",
    "FAILURE_CODES",
    "PROBLEM_V1",
    "PROGRESS_EVENT_V1",
    "RESOLVED_RESOURCES_V1",
    "RESOURCE_POLICY_V1",
    "CapacitySample",
    "Containment",
    "ExecutorMode",
    "LiveWindow",
    "Problem",
    "ProblemCategory",
    "ProcessIdentity",
    "ProgressEvent",
    "ProgressKind",
    "QueueReason",
    "ResolvedResources",
    "ResourcePolicy",
    "ResourceProfile",
]

PROBLEM_V1 = "problem.v1.0"
PROGRESS_EVENT_V1 = "progress_event.v1.0"
RESOLVED_RESOURCES_V1 = "resolved_resources.v1.0"
RESOURCE_POLICY_V1 = "resource_policy.v1.0"
CAPACITY_SAMPLE_V1 = "capacity_sample.v1.0"

ProblemCategory = Literal["validation", "dependency", "source", "resource", "integrity", "internal"]
#: ``fake`` exists for synthetic tests only; production accepts cgroup or watchdog.
ExecutorMode = Literal["cgroup", "watchdog", "fake"]
#: What the executor can actually promise, declared rather than implied (§9.1).
Containment = Literal["kernel", "best_effort", "none"]
ProgressKind = Literal["heartbeat", "progress", "checkpoint", "error", "final"]

#: §5.3 failure codes -> (contracts §2.4 category, retryable by default). A
#: default only: the retry policy of a job kind decides, and a code absent from
#: this table is refused at construction by the ops layer rather than guessed.
FAILURE_CODES: dict[str, tuple[str, bool]] = {
    "RESOURCE_UNAVAILABLE": ("resource", True),
    "RESOURCE_LIMIT_EXCEEDED": ("resource", False),
    "UNKNOWN_KILL": ("resource", False),
    "TRANSIENT_SOURCE": ("source", True),
    "RATE_LIMITED": ("source", True),
    "CREDENTIAL_INVALID": ("source", False),
    "SOURCE_NOT_FOUND": ("source", False),
    "SOURCE_EMPTY": ("source", False),
    "SOURCE_NOT_FINAL": ("source", True),
    "INPUT_CHANGED": ("dependency", False),
    "CHECKPOINT_INCOMPATIBLE": ("dependency", False),
    "DEPENDENCY_FAILED": ("dependency", False),
    "LEASE_LOST": ("dependency", True),
    "DELIVERY_FAILED": ("dependency", True),
    "BACKUP_FAILED": ("dependency", True),
    "VALIDATION_FAILED": ("validation", False),
    "PUBLICATION_REFUSED": ("validation", False),
    "IDEMPOTENCY_CONFLICT": ("validation", False),
    "INVALID_REQUEST": ("validation", False),
    "UNAUTHORIZED_NAMESPACE": ("validation", False),
    "STALE_EXPECTATION": ("validation", False),
    "INTEGRITY_FAILED": ("integrity", False),
    # P2-C02 (Phase 2 review closeout): pinned Tier-4 serving caches do not
    # cover the planned population of a snapshot-backed scoring/replay
    # launch -- refused before the job starts, never fit on a miss.
    "TIER4_CACHE_MISSING": ("validation", False),
    "CANCELLED": ("internal", False),
    "LAUNCH_FAILED": ("internal", True),
    "WORKER_FAILED": ("internal", True),
}


@dataclass(frozen=True, kw_only=True)
class Problem:
    """The shared failure envelope, contracts §2.4.

    ``message`` is operator-facing and already redacted: no URL, header,
    argument, environment value or raw exception text reaches it (§5.2).
    """

    code: str
    category: ProblemCategory
    retryable: bool
    message: str
    stage: str | None = None
    trace_id: str | None = None
    dependency_refs: tuple[str, ...] = ()
    retry_after_seconds: int | None = None
    diagnostic_ref: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = PROBLEM_V1


@dataclass(frozen=True, kw_only=True)
class QueueReason:
    """Why a queued job was not claimed, with the numbers that decided it (§5.3)."""

    code: str
    needed: dict[str, int] = field(default_factory=dict)
    available: dict[str, int] = field(default_factory=dict)
    reconsider: str = "next_claim_pass"


@dataclass(frozen=True, kw_only=True)
class ProgressEvent:
    """One structured, redacted progress record (§5.2, contracts §11.2).

    Execution metadata, never part of an output hash. ``completed_units`` is
    reported work, never a percentage invented from elapsed time.
    """

    job_id: str
    attempt_id: str
    stage_id: str | None
    sequence: int
    recorded_at: str
    kind: ProgressKind
    elapsed_seconds: float
    message: str
    completed_units: int | None = None
    total_units: int | None = None
    memory_current_bytes: int | None = None
    memory_peak_bytes: int | None = None
    checkpoint_ref: str | None = None
    eta_seconds: float | None = None
    latest_error_code: str | None = None
    schema_version: str = PROGRESS_EVENT_V1


@dataclass(frozen=True, kw_only=True)
class ResourceProfile:
    """One named resource class. Every amount is bytes (§8.1)."""

    name: str
    memory_bytes: int
    cpu_count: int
    scratch_bytes: int
    heavy: bool
    disk_heavy: bool = False
    #: False until a reviewed measurement backs the numbers; an unmeasured
    #: heavy profile runs alone (§8.1).
    measured: bool = False
    #: Conservative completion estimate for live-window admission; None means
    #: unknown, which a window treats as "could overlap".
    estimated_seconds: int | None = None
    thread_count: int | None = None


@dataclass(frozen=True, kw_only=True)
class LiveWindow:
    """A reserved daily window, in UTC, on the listed ISO weekdays (Mon=1)."""

    name: str
    weekdays: tuple[int, ...]
    start_utc: str
    end_utc: str


@dataclass(frozen=True, kw_only=True)
class ResourcePolicy:
    """The versioned operator policy that resolves profiles into reservations."""

    version: str
    base_reserve_bytes: int
    free_margin_bytes: int
    reserved_cpu_count: int
    min_free_disk_bytes: int
    max_heavy_concurrency: int
    max_disk_heavy_concurrency: int
    profiles: tuple[ResourceProfile, ...]
    live_windows: tuple[LiveWindow, ...] = ()
    schema_version: str = RESOURCE_POLICY_V1


@dataclass(frozen=True, kw_only=True)
class CapacitySample:
    """What discovery observed at one instant, outside any transaction (§6.2)."""

    sampled_at: str
    allowed_cpu_ids: tuple[int, ...]
    host_total_bytes: int
    host_available_bytes: int
    container_limit_bytes: int | None
    container_current_bytes: int | None
    swap_total_bytes: int
    swap_free_bytes: int
    disk_free_bytes: int
    executor_mode: ExecutorMode
    containment: Containment
    notes: tuple[str, ...] = ()
    schema_version: str = CAPACITY_SAMPLE_V1


@dataclass(frozen=True, kw_only=True)
class ResolvedResources:
    """contracts §11.1 ResolvedResources — what one attempt was actually given."""

    effective_host_budget_bytes: int
    reserved_memory_bytes: int
    assigned_cpu_ids: tuple[int, ...]
    thread_count: int
    scratch_limit_bytes: int
    executor_mode: ExecutorMode
    containment: Containment
    provider_leases: tuple[str, ...]
    resource_profile_version: str
    schema_version: str = RESOLVED_RESOURCES_V1


@dataclass(frozen=True, kw_only=True)
class ProcessIdentity:
    """A process as the kernel identifies it across PID reuse (§6.3).

    ``pid`` alone names a different process after reuse; ``start_ticks`` (field
    22 of ``/proc/<pid>/stat``) plus the boot ID does not.
    """

    boot_id: str
    pid: int
    start_ticks: int
    process_group: int
    cgroup_path: str | None = None
