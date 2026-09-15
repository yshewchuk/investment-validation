"""Claiming work: dependency readiness, admission, fence and reservations — §6.2.

One claim, in the order the guide gives:

1. The capacity sample is taken by the caller, **outside** any transaction.
2. A short immediate transaction rereads reservations, job states and
   completed dependencies.
3. Admission is decided. A job that does not fit gets a queued reason with the
   numbers and **no attempt row**: waiting for capacity is not a retry.
4. The fence is incremented, and the attempt, its reservation and its CPUs are
   inserted, all in the same transaction that makes it the active attempt.
5. The caller commits, then launches, then records the launch.

Priority order is ``priority`` descending, then earliest deadline, then
submission time, then job ID, so ties are deterministic. A heavy job that is
blocked by work already running holds the heavy slot against **lower-priority
heavy** jobs, which would otherwise starve it forever by always fitting first.
Light jobs may still pass it when both memory tests allow. A job that can never
fit (``PROFILE_EXCEEDS_CAPACITY``, a profile bigger than the host even without
any container squeeze) holds nothing and stays queued -- unauthored policy
mistakes are rare and inspectable via ``queue_reason``. A profile that exceeds
the live host's own maximum possible headroom (``PROFILE_EXCEEDS_HEADROOM_CEILING``,
resources.py's ``max_possible_headroom_bytes``) is different: it is reachable
by ordinary configuration (the legacy_score v5 6 GiB incident) and an infinite
queue hides it, so this one fails the job outright with a typed
``RESOURCE_PROFILE_UNSATISFIABLE`` problem and blocks its descendants (§8.1,
added 2026-09-15).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.v2.contracts import (
    CapacitySample,
    JobSpec,
    QueueReason,
    ResolvedResources,
    ResourcePolicy,
    ResourceProfile,
)
from engine.v2.foundation import Clock, content_hash, format_timestamp, parse_timestamp
from engine.v2.ops.catalog import dumps, load_json, transaction
from engine.v2.ops.errors import OpsError, fail, make_problem
from engine.v2.ops.lifecycle import block_descendants
from engine.v2.ops.profiles import profile_named
from engine.v2.ops.resources import ActiveReservation, decide, live_window_reason
from engine.v2.ops.store_barrier import acquire_in, domains_of, lease_reason

#: Queue reasons that mean "will never fit this policy", not "wait" -- the
#: job never holds a heavy slot against others, and (unlike a normal
#: QueueReason) claim_next fails it outright instead of leaving it queued.
_UNFITTABLE_CODES = frozenset({"PROFILE_EXCEEDS_CAPACITY", "PROFILE_EXCEEDS_HEADROOM_CEILING"})
#: Of those, the ones that get a terminal job failure (§8.1). Kept separate
#: from ``_UNFITTABLE_CODES`` so ``PROFILE_EXCEEDS_CAPACITY`` (an existing,
#: tested "stays queued" contract -- test_o04_o05) is unchanged; only the new
#: headroom-ceiling refusal added 2026-09-15 gets the fail-fast behaviour.
_FAILS_JOB_CODES = frozenset({"PROFILE_EXCEEDS_HEADROOM_CEILING"})

__all__ = [
    "MEASUREMENT_STALE_SECONDS",
    "Claim",
    "Supervisor",
    "active_reservations",
    "attempt_id_for",
    "claim_next",
]

#: A memory sample older than this is treated as absent, i.e. fully unconsumed.
MEASUREMENT_STALE_SECONDS = 120

_READY = """
SELECT j.* FROM jobs j
WHERE j.state = 'queued'
  AND (j.next_eligible_at IS NULL OR j.next_eligible_at <= :now)
  AND NOT EXISTS (
      SELECT 1 FROM job_dependencies d JOIN jobs p ON p.job_id = d.parent_job_id
      WHERE d.child_job_id = j.job_id AND p.state <> 'succeeded')
ORDER BY j.priority DESC, (j.deadline_at IS NULL), j.deadline_at, j.created_at, j.job_id
"""

_ACTIVE = """
SELECT r.attempt_id, r.profile, r.memory_bytes, r.scratch_bytes, r.heavy, r.disk_heavy,
       r.measured, a.memory_current_bytes, a.memory_sampled_at,
       (SELECT group_concat(c.cpu_id) FROM cpu_assignments c
         WHERE c.attempt_id = r.attempt_id AND c.released_at IS NULL) AS cpus
FROM resource_reservations r JOIN attempts a ON a.attempt_id = r.attempt_id
WHERE r.released_at IS NULL
"""


@dataclass(frozen=True)
class Supervisor:
    """The identity a claim is made under."""

    epoch_id: str
    boot_id: str
    lease_seconds: int = 120


@dataclass(frozen=True)
class Claim:
    job_id: str
    attempt_id: str
    attempt_number: int
    fence: int
    spec: JobSpec
    resources: ResolvedResources
    lease_expires_at: str


def attempt_id_for(job_id: str, attempt_number: int) -> str:
    digest = content_hash({"job_id": job_id, "attempt_number": attempt_number})
    return "att_" + digest.removeprefix("sha256:")[:32]


def active_reservations(conn: sqlite3.Connection, now: datetime) -> list[ActiveReservation]:
    """Every unreleased reservation, with a memory sample only if it is fresh."""
    out = []
    for row in conn.execute(_ACTIVE):
        current = row["memory_current_bytes"]
        sampled = row["memory_sampled_at"]
        if sampled is None or (now - parse_timestamp(sampled)).total_seconds() \
                > MEASUREMENT_STALE_SECONDS:
            current = None
        cpus = tuple(sorted(int(c) for c in (row["cpus"] or "").split(",") if c))
        out.append(ActiveReservation(
            attempt_id=row["attempt_id"], profile=row["profile"],
            memory_bytes=row["memory_bytes"], scratch_bytes=row["scratch_bytes"],
            heavy=bool(row["heavy"]), disk_heavy=bool(row["disk_heavy"]),
            measured=bool(row["measured"]), cpu_ids=cpus, memory_current_bytes=current))
    return out


def claim_next(conn: sqlite3.Connection, *, policy: ResourcePolicy, sample: CapacitySample,
               supervisor: Supervisor, clock: Clock, registry=None) -> Claim | None:
    """Claim the best admissible ready job, or record why each one waits."""
    now = clock.now()
    stamp = format_timestamp(now)
    with transaction(conn):
        conn.execute("UPDATE jobs SET state = 'queued', updated_at = ? "
                     "WHERE state = 'retry_wait' AND next_eligible_at <= ?", (stamp, stamp))
        active = active_reservations(conn, now)
        heavy_held = False
        for row in conn.execute(_READY, {"now": stamp}).fetchall():
            profile, reason = _profile_or_reason(policy, row)
            domains = domains_of(registry, row["kind"], _parameters(row)) if reason is None else ()
            if reason is None:
                reason = lease_reason(conn, domains)
            if reason is None:
                reason = live_window_reason(policy, profile, now)
            if reason is None:
                reason = _provider_reason(conn, row, stamp)
            if reason is None and heavy_held and profile.heavy:
                reason = QueueReason(code="HEAVY_SLOT_HELD", reconsider="higher_priority_heavy_job")
            if reason is None:
                decision = decide(policy, profile, sample, active)
                if decision.admitted:
                    return _create_attempt(conn, row, profile, policy, decision.resources,
                                           supervisor, now, domains)
                reason = decision.reason
                heavy_held |= profile.heavy and reason.code not in _UNFITTABLE_CODES
            if reason.code in _FAILS_JOB_CODES:
                _fail_unfittable(conn, row, reason, now)
                continue
            conn.execute("UPDATE jobs SET queue_reason_json = ? WHERE job_id = ?",
                         (dumps(reason), row["job_id"]))
    return None


def _fail_unfittable(conn: sqlite3.Connection, row: sqlite3.Row, reason: QueueReason,
                     now: datetime) -> None:
    """Terminal failure for a job whose profile can never fit (§8.1): a typed
    ``RESOURCE_PROFILE_UNSATISFIABLE`` problem with the numbers, not an
    infinite ``queued`` wait. Blocks descendants the same way any other
    terminal job failure does."""
    problem = make_problem(
        "RESOURCE_PROFILE_UNSATISFIABLE",
        "the job's resource profile exceeds the host's maximum possible "
        "headroom under the current policy",
        details={"needed": reason.needed, "max_possible": reason.available})
    stamp = format_timestamp(now)
    conn.execute("UPDATE jobs SET state = 'failed', queue_reason_json = NULL, "
                "failure_json = ?, updated_at = ? WHERE job_id = ?",
                (dumps(problem), stamp, row["job_id"]))
    block_descendants(conn, row["job_id"], now)


def _parameters(row):
    return load_json(JobSpec, row["spec_json"]).parameters


def _provider_reason(conn, row, now_stamp):
    spec = load_json(JobSpec, row["spec_json"])
    if not spec.provider_budget_ref:
        return None
    account = conn.execute(
        "SELECT remaining, live_reserve, blocked_code, next_eligible_at "
        "FROM provider_accounts WHERE account = ?",
        (spec.provider_budget_ref,)).fetchone()
    calls = spec.parameters.get("provider_calls", 1)
    if account is None or account["blocked_code"]:
        return QueueReason(code="PROVIDER_UNAVAILABLE", reconsider="operator_action")
    if not isinstance(calls, int) or calls <= 0:
        return QueueReason(code="PROVIDER_UNAVAILABLE", reconsider="specification_change")
    if account["next_eligible_at"] and account["next_eligible_at"] > now_stamp:
        # A 429 backoff recorded by provider_budget.record_response: no attempt
        # row until the account is eligible again, so the backoff is not spent
        # relaunching into the same rate limit. QueueReason.available is
        # int-valued (contracts §5.3), so the stamp travels as epoch seconds.
        eligible_at = int(parse_timestamp(account["next_eligible_at"]).timestamp())
        return QueueReason(code="PROVIDER_BACKOFF", available={"eligible_at": eligible_at},
                           reconsider="provider_backoff_elapsed")
    active = conn.execute(
        "SELECT 1 FROM provider_reservations WHERE account = ? AND released_at IS NULL",
        (spec.provider_budget_ref,)).fetchone()
    if active or calls > account["remaining"] - account["live_reserve"]:
        return QueueReason(code="PROVIDER_BUDGET", needed={"calls": calls},
                           available={"calls": max(0, account["remaining"] -
                                                     account["live_reserve"])},
                           reconsider="provider_release_or_quota")
    return None


def _profile_or_reason(policy: ResourcePolicy,
                       row: sqlite3.Row) -> tuple[ResourceProfile | None, QueueReason | None]:
    try:
        return profile_named(policy, row["resource_class"]), None
    except OpsError:
        return None, QueueReason(code="UNKNOWN_PROFILE", reconsider="policy_change")


def _create_attempt(conn: sqlite3.Connection, row: sqlite3.Row, profile: ResourceProfile,
                    policy: ResourcePolicy, resources: ResolvedResources,
                    supervisor: Supervisor, now: datetime,
                    store_domains: tuple[tuple[str, str], ...] = ()) -> Claim:
    job_id = row["job_id"]
    number, fence = row["attempt_count"] + 1, row["fence"] + 1
    attempt_id = attempt_id_for(job_id, number)
    stamp = format_timestamp(now)
    lease = format_timestamp(now + timedelta(seconds=supervisor.lease_seconds))
    conn.execute(
        "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, supervisor_epoch, "
        "host_boot_id, state, process_state, resources_json, created_at, lease_expires_at) "
        "VALUES (?, ?, ?, ?, ?, ?, 'starting', 'unlaunched', ?, ?, ?)",
        (attempt_id, job_id, number, fence, supervisor.epoch_id, supervisor.boot_id,
         dumps(resources), stamp, lease))
    conn.execute(
        "INSERT INTO resource_reservations (attempt_id, policy_version, profile, memory_bytes, "
        "scratch_bytes, heavy, disk_heavy, measured, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (attempt_id, policy.version, profile.name, profile.memory_bytes, profile.scratch_bytes,
         int(profile.heavy), int(profile.disk_heavy), int(profile.measured), stamp))
    conn.executemany("INSERT INTO cpu_assignments (attempt_id, cpu_id) VALUES (?, ?)",
                     [(attempt_id, cpu) for cpu in resources.assigned_cpu_ids])
    spec = load_json(JobSpec, row["spec_json"])
    if spec.provider_budget_ref:
        _reserve_provider(conn, spec.provider_budget_ref, attempt_id, fence, spec.parameters,
                          stamp)
    updated = conn.execute(
        "UPDATE jobs SET state = 'running', fence = ?, attempt_count = ?, active_attempt_id = ?, "
        "queue_reason_json = NULL, next_eligible_at = NULL, updated_at = ? "
        "WHERE job_id = ? AND state = 'queued' AND fence = ?",
        (fence, number, attempt_id, stamp, job_id, row["fence"])).rowcount
    if updated != 1:
        raise fail("STALE_EXPECTATION", "the job changed while it was being claimed")
    acquire_in(conn, attempt_id, store_domains)
    return Claim(job_id=job_id, attempt_id=attempt_id, attempt_number=number, fence=fence,
                 spec=load_json(JobSpec, row["spec_json"]), resources=resources,
                 lease_expires_at=lease)


def _reserve_provider(conn, account, attempt_id, fence, parameters, now_stamp):
    """Reserve the account lease in the claim transaction, before launch."""
    calls = parameters.get("provider_calls", 1) if isinstance(parameters, dict) else 1
    if not isinstance(calls, int) or calls <= 0 or calls > 1_000_000:
        raise fail("INVALID_REQUEST", "provider call estimate is invalid")
    row = conn.execute(
        "SELECT remaining, live_reserve, blocked_code, next_eligible_at "
        "FROM provider_accounts WHERE account = ?",
        (account,)).fetchone()
    if row is None or row["blocked_code"]:
        raise fail("CREDENTIAL_INVALID", "provider account needs operator action")
    if row["next_eligible_at"] and row["next_eligible_at"] > now_stamp:
        # Backstop: admission should already have queued this on
        # PROVIDER_BACKOFF, but never launch into a live backoff regardless.
        raise fail("RATE_LIMITED", "provider account is in backoff")
    active = conn.execute(
        "SELECT 1 FROM provider_reservations WHERE account = ? AND released_at IS NULL",
        (account,)).fetchone()
    if active or calls > row["remaining"] - row["live_reserve"]:
        raise fail("RESOURCE_UNAVAILABLE", "provider lease or call budget unavailable")
    conn.execute(
        "INSERT INTO provider_reservations(account,attempt_id,fence,reserved_calls) "
        "VALUES (?,?,?,?)", (account, attempt_id, fence, calls))
