"""Central admission arithmetic — phase-1 guide §8.1–§8.2.

Pure functions over one capacity sample and the active reservations reread in
the claim transaction. Both memory tests must pass, in consistent units::

    capacity = min(host_total, finite_container_limit) - base_reserve
    sum(active reservations) + new <= capacity

    headroom = min(host_available, container_remaining) - free_margin
    new + sum(max(0, reserved_i - measured_current_i)) <= headroom

A reservation with no fresh measurement counts as **entirely unconsumed**: an
unobserved worker may still grow to its reservation, so its unused headroom is
still owed. Assuming it already uses everything would silently discard that
debt and admit a job the host cannot hold (§8.1).

A container limit whose current usage is unknown yields no headroom at all
rather than a guess. Admission is conservative coordination; kernel containment
and free margin still matter, because external processes allocate after the
sample is taken (§6.2).

A third test, ``_headroom_ceiling_reason``, refuses rather than queues: a
profile above ``max_possible_headroom_bytes`` (headroom plus everything owed,
i.e. what this sample could ever admit even if every active job released in
full) cannot be explained by transient competition and will not resolve on
its own. ``decide`` returns this as a normal ``QueueReason``; the caller
(``scheduler.claim_next``) is the one that turns it into a terminal job
failure instead of leaving the job queued (added 2026-09-15, after
``legacy_score`` v5's 6 GiB reservation queued attempt 17 indefinitely: the
host's live headroom never reached 6 GiB even fully idle, and nothing was
active to blame it on).

CPUs come from the **allowed affinity**, never ``os.cpu_count()``. The lowest
``reserved_cpu_count`` allowed CPUs stay with the OS, API and supervisor; the
rest are handed out disjointly, lowest free first, which works unchanged on a
non-contiguous affinity mask.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from engine.v2.contracts import (
    CapacitySample,
    QueueReason,
    ResolvedResources,
    ResourcePolicy,
    ResourceProfile,
)

__all__ = [
    "ActiveReservation",
    "Admission",
    "capacity_bytes",
    "decide",
    "headroom_bytes",
    "max_possible_headroom_bytes",
    "owed_unconsumed_bytes",
    "worker_cpu_ids",
    "live_window_reason",
]


@dataclass(frozen=True)
class ActiveReservation:
    """One unreleased reservation, as reread inside the claim transaction."""

    attempt_id: str
    profile: str
    memory_bytes: int
    scratch_bytes: int
    heavy: bool
    disk_heavy: bool
    measured: bool
    cpu_ids: tuple[int, ...]
    #: None when never sampled or when the sample is stale.
    memory_current_bytes: int | None = None


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: QueueReason | None = None
    resources: ResolvedResources | None = None


def capacity_bytes(policy: ResourcePolicy, sample: CapacitySample) -> int:
    ceiling = sample.host_total_bytes
    if sample.container_limit_bytes is not None:
        ceiling = min(ceiling, sample.container_limit_bytes)
    return ceiling - policy.base_reserve_bytes


def headroom_bytes(policy: ResourcePolicy, sample: CapacitySample) -> int:
    available = sample.host_available_bytes
    if sample.container_limit_bytes is not None:
        current = sample.container_current_bytes
        remaining = 0 if current is None else sample.container_limit_bytes - current
        available = min(available, remaining)
    return available - policy.free_margin_bytes


def owed_unconsumed_bytes(active: Sequence[ActiveReservation]) -> int:
    return sum(max(0, r.memory_bytes - (r.memory_current_bytes or 0)) for r in active)


def max_possible_headroom_bytes(policy: ResourcePolicy, sample: CapacitySample,
                                active: Sequence[ActiveReservation]) -> int:
    """The most headroom this sample could ever yield: current headroom plus
    every byte still owed to an active reservation, i.e. as if every active
    job released its whole reservation right now. A profile whose
    ``memory_bytes`` exceeds this can never be admitted from this sample
    forward -- no release manufactures memory the sample does not already
    account for. With ``active`` empty this is exactly ``headroom_bytes``
    (the attempt-17 incident: legacy_score queued with nothing else heavy
    running, so ``owed`` was already zero and the shortfall was structural,
    not transient) (§8.1)."""
    return headroom_bytes(policy, sample) + owed_unconsumed_bytes(active)


def worker_cpu_ids(policy: ResourcePolicy, sample: CapacitySample) -> list[int]:
    return sorted(sample.allowed_cpu_ids)[policy.reserved_cpu_count:]


def live_window_reason(policy: ResourcePolicy, profile: ResourceProfile,
                       now: datetime) -> QueueReason | None:
    """Refuse work that can conservatively overlap a reserved live window."""
    if not policy.live_windows:
        return None
    for window in policy.live_windows:
        try:
            start = time.fromisoformat(window.start_utc)
            end = time.fromisoformat(window.end_utc)
        except ValueError:
            return QueueReason(code="INVALID_LIVE_WINDOW", reconsider="policy_change")
        day = now.date() + timedelta(days=(window.weekdays[0] - now.isoweekday()) % 7)
        begin = datetime.combine(day, start, tzinfo=now.tzinfo)
        finish = datetime.combine(day, end, tzinfo=now.tzinfo)
        if finish <= begin:
            finish += timedelta(days=1)
        estimate = profile.estimated_seconds
        completion = now + timedelta(seconds=estimate or 0)
        overlaps = now < finish and completion > begin
        unknown_heavy = profile.heavy and estimate is None and now < finish
        if overlaps or unknown_heavy:
            return QueueReason(code="LIVE_WINDOW", needed={"completion_seconds": estimate or 0},
                               available={"seconds_until_window": max(
                                   0, int((begin - now).total_seconds()))},
                               reconsider="live_window_end")
    return None


def decide(policy: ResourcePolicy, profile: ResourceProfile, sample: CapacitySample,
           active: Sequence[ActiveReservation]) -> Admission:
    """Admit ``profile`` now, or say exactly why not, with the numbers."""
    reason = (_fits_at_all(policy, profile, sample)
              or _headroom_ceiling_reason(policy, profile, sample, active)
              or _memory_reason(policy, profile, sample, active)
              or _slot_reason(policy, profile, active)
              or _disk_reason(policy, profile, sample, active))
    cpus = _allocate(policy, profile, sample, active)
    if reason is None and cpus is None:
        taken = sum(len(r.cpu_ids) for r in active)
        reason = QueueReason(code="CPU_UNAVAILABLE", needed={"cpus": profile.cpu_count},
                             available={"free_cpus": len(worker_cpu_ids(policy, sample)) - taken},
                             reconsider="reservation_release")
    if reason is not None:
        return Admission(admitted=False, reason=reason)
    assert cpus is not None
    return Admission(admitted=True, resources=ResolvedResources(
        effective_host_budget_bytes=capacity_bytes(policy, sample),
        reserved_memory_bytes=profile.memory_bytes,
        assigned_cpu_ids=cpus,
        thread_count=profile.thread_count or len(cpus),
        scratch_limit_bytes=profile.scratch_bytes,
        executor_mode=sample.executor_mode,
        containment=sample.containment,
        provider_leases=(),
        resource_profile_version=policy.version,
    ))


def _fits_at_all(policy: ResourcePolicy, profile: ResourceProfile,
                 sample: CapacitySample) -> QueueReason | None:
    """A profile larger than the host stays queued with the numbers, never shrunk to fit."""
    capacity = capacity_bytes(policy, sample)
    cpus = len(worker_cpu_ids(policy, sample))
    if profile.memory_bytes <= capacity and profile.cpu_count <= cpus:
        return None
    return QueueReason(code="PROFILE_EXCEEDS_CAPACITY",
                       needed={"memory_bytes": profile.memory_bytes, "cpus": profile.cpu_count},
                       available={"capacity_bytes": capacity, "worker_cpus": cpus},
                       reconsider="capacity_or_profile_change")


def _headroom_ceiling_reason(policy: ResourcePolicy, profile: ResourceProfile,
                             sample: CapacitySample,
                             active: Sequence[ActiveReservation]) -> QueueReason | None:
    """Refuse -- never queue -- a profile bigger than this sample could ever
    admit, even in the best case where every active reservation released in
    full right now. Distinct from ``_memory_reason``'s transient
    ``MEMORY_HEADROOM``: that shortfall can resolve when an active job's real
    usage frees; this one cannot, because nothing currently reserved explains
    it (§8.1, the legacy_score v5 6 GiB incident: attempt 17 queued forever
    with zero active heavy reservations)."""
    ceiling = max_possible_headroom_bytes(policy, sample, active)
    if profile.memory_bytes <= ceiling:
        return None
    return QueueReason(code="PROFILE_EXCEEDS_HEADROOM_CEILING",
                       needed={"memory_bytes": profile.memory_bytes},
                       available={"max_possible_headroom_bytes": ceiling},
                       reconsider="capacity_or_profile_change")


def _memory_reason(policy: ResourcePolicy, profile: ResourceProfile, sample: CapacitySample,
                   active: Sequence[ActiveReservation]) -> QueueReason | None:
    capacity = capacity_bytes(policy, sample)
    reserved = sum(r.memory_bytes for r in active)
    if reserved + profile.memory_bytes > capacity:
        return QueueReason(code="RESERVATION_BUDGET",
                           needed={"memory_bytes": profile.memory_bytes},
                           available={"capacity_bytes": capacity, "reserved_bytes": reserved},
                           reconsider="reservation_release")
    headroom = headroom_bytes(policy, sample)
    owed = owed_unconsumed_bytes(active)
    if profile.memory_bytes + owed > headroom:
        return QueueReason(code="MEMORY_HEADROOM",
                           needed={"memory_bytes": profile.memory_bytes,
                                   "owed_unconsumed_bytes": owed},
                           available={"headroom_bytes": headroom},
                           reconsider="next_capacity_sample")
    return None


def _slot_reason(policy: ResourcePolicy, profile: ResourceProfile,
                 active: Sequence[ActiveReservation]) -> QueueReason | None:
    heavy = [r for r in active if r.heavy]
    if profile.heavy and len(heavy) >= policy.max_heavy_concurrency:
        return QueueReason(code="HEAVY_SLOT", needed={"heavy_slots": 1},
                           available={"heavy_slots": policy.max_heavy_concurrency - len(heavy)},
                           reconsider="reservation_release")
    if profile.heavy and (heavy and (not profile.measured or any(not r.measured for r in heavy))):
        return QueueReason(code="EXCLUSIVE_UNMEASURED", needed={"heavy_slots": 1},
                           available={"heavy_slots": 0}, reconsider="reservation_release")
    disk_heavy = sum(1 for r in active if r.disk_heavy)
    if profile.disk_heavy and disk_heavy >= policy.max_disk_heavy_concurrency:
        return QueueReason(code="DISK_HEAVY_SLOT", needed={"disk_heavy_slots": 1},
                           available={"disk_heavy_slots": 0}, reconsider="reservation_release")
    return None


def _disk_reason(policy: ResourcePolicy, profile: ResourceProfile, sample: CapacitySample,
                 active: Sequence[ActiveReservation]) -> QueueReason | None:
    # Sampled free space already reflects scratch written so far; counting the
    # full outstanding reservations again double-counts on purpose.
    owed = sum(r.scratch_bytes for r in active)
    spare = sample.disk_free_bytes - owed - policy.min_free_disk_bytes
    if profile.scratch_bytes <= spare:
        return None
    return QueueReason(code="DISK_SPACE", needed={"scratch_bytes": profile.scratch_bytes},
                       available={"spare_bytes": max(spare, 0)},
                       reconsider="disk_space_change")


def _allocate(policy: ResourcePolicy, profile: ResourceProfile, sample: CapacitySample,
              active: Sequence[ActiveReservation]) -> tuple[int, ...] | None:
    taken = {cpu for r in active for cpu in r.cpu_ids}
    free = [cpu for cpu in worker_cpu_ids(policy, sample) if cpu not in taken]
    if profile.cpu_count > len(free):
        return None
    return tuple(free[:profile.cpu_count])
