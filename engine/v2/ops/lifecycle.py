"""Attempt lifecycle: launch, heartbeat, fenced commit, cancellation, progress — §6.3.

**Every effect is checked inside the transaction that commits it.** A fence
verified before launching a worker, or before calling an uploader, does not
stop that worker from committing after a takeover. So the check —
active attempt, fence, lease, job state — runs in the same immediate transaction
as the checkpoint, decision or pointer write it guards, and ``effects``
callbacks run inside that transaction or not at all.

Two separate facts close an attempt:

* its **logical outcome** (``state``), which moves the job; and
* its **process state**, which releases reservations. Resources are recovered
  only once the worker is ``exited`` or ``verified_dead``. A quarantined or
  unknown tree keeps its reservation, which keeps replacement heavy work out
  (§6.3, §9.1).

Cancellation invalidates the fence **first**, then asks the executor to stop
the worker, and reports ``cancelled`` only after the executor has verified
every owned process is dead. A cancellation naming an attempt that is no longer
the active one conflicts instead of cancelling its replacement.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

from engine.v2.contracts import (
    AttemptReceipt,
    CancellationReceipt,
    Problem,
    ProcessIdentity,
    ProgressEvent,
    ResolvedResources,
)
from engine.v2.foundation import Clock, format_timestamp
from engine.v2.ops.catalog import dumps, load_json, transaction
from engine.v2.ops.errors import fail, make_problem
from engine.v2.ops.submission import RetryPolicy

__all__ = [
    "RELEASABLE_PROCESS_STATES",
    "Outcome",
    "advance_job",
    "attempt_receipts",
    "block_descendants",
    "commit_attempt",
    "complete_cancel",
    "end_attempt",
    "heartbeat",
    "record_launch",
    "record_measurement",
    "record_progress",
    "release_reservations",
    "renew_after_resume",
    "request_cancel",
    "verify_fence",
]

RELEASABLE_PROCESS_STATES = frozenset({"exited", "verified_dead"})
_TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True)
class Outcome:
    """How an attempt ended, as the executor observed it."""

    succeeded: bool
    process_state: str
    exit_code: int | None = None
    failure: Problem | None = None


# --------------------------------------------------------------------------
# the fence
# --------------------------------------------------------------------------


def verify_fence(conn: sqlite3.Connection, attempt_id: str, fence: int,
                 now: datetime) -> tuple[sqlite3.Row, sqlite3.Row]:
    """Inside the caller's transaction: this attempt may still commit. Returns (job, attempt)."""
    attempt = conn.execute("SELECT * FROM attempts WHERE attempt_id = ?",
                           (attempt_id,)).fetchone()
    if attempt is None:
        raise fail("LEASE_LOST", "unknown attempt")
    job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (attempt["job_id"],)).fetchone()
    details = {"attempt_id": attempt_id, "fence": fence, "current_fence": job["fence"]}
    if job["state"] == "cancelling":
        raise fail("CANCELLED", "the job is being cancelled; this fence is void", details=details)
    live = (job["state"] == "running" and job["active_attempt_id"] == attempt_id
            and job["fence"] == fence == attempt["fence"]
            and attempt["state"] in ("starting", "running"))
    if not live:
        raise fail("LEASE_LOST", "the attempt no longer holds the job's fence", details=details)
    if format_timestamp(now) >= attempt["lease_expires_at"]:
        raise fail("LEASE_LOST", "the attempt's lease has expired", details=details)
    return job, attempt


# --------------------------------------------------------------------------
# launch and liveness
# --------------------------------------------------------------------------


def record_launch(conn: sqlite3.Connection, attempt_id: str, fence: int,
                  identity: ProcessIdentity, *, clock: Clock, lease_seconds: int) -> None:
    now = clock.now()
    with transaction(conn):
        verify_fence(conn, attempt_id, fence, now)
        conn.execute(
            "UPDATE attempts SET state = 'running', process_state = 'alive', process_json = ?, "
            "started_at = ?, heartbeat_at = ?, lease_expires_at = ? WHERE attempt_id = ?",
            (dumps(identity), format_timestamp(now), format_timestamp(now),
             format_timestamp(now + timedelta(seconds=lease_seconds)), attempt_id))


def heartbeat(conn: sqlite3.Connection, attempt_id: str, fence: int, *, clock: Clock,
              lease_seconds: int) -> bool:
    """Extend the lease. False, and no write, when the fence is no longer held."""
    now = clock.now()
    with transaction(conn):
        try:
            verify_fence(conn, attempt_id, fence, now)
        except Exception:  # noqa: BLE001 - any refusal means "stop working"
            return False
        conn.execute("UPDATE attempts SET heartbeat_at = ?, lease_expires_at = ? "
                     "WHERE attempt_id = ?",
                     (format_timestamp(now),
                      format_timestamp(now + timedelta(seconds=lease_seconds)), attempt_id))
    return True


def renew_after_resume(conn: sqlite3.Connection, attempt_id: str, fence: int, *, clock: Clock,
                       lease_seconds: int) -> bool:
    """Extend a lease across a wall-clock jump (B2), ignoring lease expiry.

    The same liveness gate as :func:`verify_fence` — job running, this the
    active attempt, the fence and attempt state both still live — except the
    lease-expiry check, since the jump itself may already read as "expired" in
    wall-clock terms. The caller has already verified the recorded identity is
    still alive; this only extends the lease so the very next
    :func:`expire_leases` does not fence a healthy worker. False, with no
    write, when the attempt no longer holds the job's fence.
    """
    now = clock.now()
    with transaction(conn):
        attempt = conn.execute("SELECT * FROM attempts WHERE attempt_id = ?",
                               (attempt_id,)).fetchone()
        if attempt is None:
            return False
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (attempt["job_id"],)).fetchone()
        live = (job["state"] == "running" and job["active_attempt_id"] == attempt_id
                and job["fence"] == fence == attempt["fence"]
                and attempt["state"] in ("starting", "running"))
        if not live:
            return False
        conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                     (format_timestamp(now + timedelta(seconds=lease_seconds)), attempt_id))
        return True


def record_measurement(conn: sqlite3.Connection, attempt_id: str, *, current_bytes: int,
                       peak_bytes: int, clock: Clock) -> None:
    """Execution metadata, recorded even for a fenced-off attempt awaiting reconciliation."""
    with transaction(conn):
        conn.execute(
            "UPDATE attempts SET memory_current_bytes = ?, memory_sampled_at = ?, "
            "memory_peak_bytes = MAX(COALESCE(memory_peak_bytes, 0), ?) WHERE attempt_id = ?",
            (current_bytes, format_timestamp(clock.now()), peak_bytes, attempt_id))


# --------------------------------------------------------------------------
# completion
# --------------------------------------------------------------------------


def commit_attempt(conn: sqlite3.Connection, attempt_id: str, fence: int, outcome: Outcome, *,
                   clock: Clock,
                   effects: Callable[[sqlite3.Connection], None] | None = None) -> str:
    """Close an attempt under its fence; run ``effects`` in the same transaction.

    Returns the job's new state. Any refusal or effect error rolls back
    everything, including the effects.
    """
    if outcome.succeeded == (outcome.failure is not None):
        raise ValueError("a success carries no failure, and a failure must carry one")
    if outcome.process_state not in RELEASABLE_PROCESS_STATES:
        raise fail("STALE_EXPECTATION", "completion requires a reconciled process tree")
    now = clock.now()
    with transaction(conn):
        job, _ = verify_fence(conn, attempt_id, fence, now)
        if effects is not None and outcome.succeeded:
            effects(conn)
        end_attempt(conn, attempt_id, "succeeded" if outcome.succeeded else "failed",
                    outcome, now)
        return advance_job(conn, job, outcome, now)


def end_attempt(conn: sqlite3.Connection, attempt_id: str, state: str, outcome: Outcome,
                now: datetime) -> None:
    stamp = format_timestamp(now)
    conn.execute("UPDATE attempts SET state = ?, process_state = ?, exit_code = ?, "
                 "failure_json = ?, ended_at = ? WHERE attempt_id = ?",
                 (state, outcome.process_state, outcome.exit_code,
                  None if outcome.failure is None else dumps(outcome.failure), stamp,
                  attempt_id))
    if outcome.process_state in RELEASABLE_PROCESS_STATES:
        release_reservations(conn, attempt_id, now, reason=state)


def release_reservations(conn: sqlite3.Connection, attempt_id: str, now: datetime, *,
                         reason: str) -> None:
    stamp = format_timestamp(now)
    conn.execute("UPDATE resource_reservations SET released_at = ?, release_reason = ? "
                 "WHERE attempt_id = ? AND released_at IS NULL", (stamp, reason, attempt_id))
    conn.execute("UPDATE cpu_assignments SET released_at = ? "
                 "WHERE attempt_id = ? AND released_at IS NULL", (stamp, attempt_id))
    conn.execute("UPDATE provider_reservations SET released_at = ? "
                 "WHERE attempt_id = ? AND released_at IS NULL", (stamp, attempt_id))
    conn.execute("UPDATE store_leases SET released_at = ? "
                 "WHERE attempt_id = ? AND released_at IS NULL", (stamp, attempt_id))


def advance_job(conn: sqlite3.Connection, job: sqlite3.Row, outcome: Outcome,
                now: datetime) -> str:
    """Move the job after its attempt ended: succeeded, retry_wait or failed."""
    retry = load_json(RetryPolicy, job["retry_json"])
    eligible = None
    if outcome.succeeded:
        state = "succeeded"
    elif outcome.failure.retryable and job["attempt_count"] < job["max_attempts"]:
        state = "retry_wait"
        eligible = format_timestamp(now + timedelta(
            seconds=retry.delay_after(job["attempt_count"])))
    else:
        state = "failed"
    conn.execute("UPDATE jobs SET state = ?, active_attempt_id = NULL, next_eligible_at = ?, "
                 "failure_json = ?, updated_at = ? WHERE job_id = ?",
                 (state, eligible, None if outcome.failure is None else dumps(outcome.failure),
                  format_timestamp(now), job["job_id"]))
    if state == "failed":
        block_descendants(conn, job["job_id"], now)
    return state


def block_descendants(conn: sqlite3.Connection, job_id: str, now: datetime) -> int:
    """Block every transitive dependent still waiting. Unrelated jobs are untouched."""
    problem = make_problem("DEPENDENCY_FAILED", "an upstream job did not succeed",
                           dependency_refs=(job_id,))
    cursor = conn.execute(
        """WITH RECURSIVE below(id) AS (
               SELECT child_job_id FROM job_dependencies WHERE parent_job_id = ?
               UNION
               SELECT d.child_job_id FROM job_dependencies d JOIN below ON d.parent_job_id = below.id)
           UPDATE jobs SET state = 'blocked', failure_json = ?, next_eligible_at = NULL,
                           updated_at = ?
           WHERE job_id IN (SELECT id FROM below) AND state IN ('queued', 'retry_wait')""",
        (job_id, dumps(problem), format_timestamp(now)))
    return cursor.rowcount


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------


def request_cancel(conn: sqlite3.Connection, job_id: str, expected_attempt_id: str | None, *,
                   clock: Clock) -> CancellationReceipt:
    """Invalidate the fence first. Complete immediately only when nothing is running."""
    now = clock.now()
    stamp = format_timestamp(now)
    with transaction(conn):
        job = _job(conn, job_id)
        receipt = _cancel_precheck(job, expected_attempt_id, stamp)
        if receipt is not None:
            return receipt
        if job["active_attempt_id"] is None:
            conn.execute("UPDATE jobs SET state = 'cancelled', next_eligible_at = NULL, "
                         "updated_at = ? WHERE job_id = ?", (stamp, job_id))
            block_descendants(conn, job_id, now)
            return CancellationReceipt(job_id=job_id, expected_attempt_id=expected_attempt_id,
                                       state="cancelled", fence=job["fence"],
                                       requested_at=stamp, completed_at=stamp)
        conn.execute("UPDATE jobs SET state = 'cancelling', fence = fence + 1, updated_at = ? "
                     "WHERE job_id = ?", (stamp, job_id))
        conn.execute("UPDATE attempts SET state = 'cancelling' WHERE attempt_id = ? "
                     "AND state IN ('starting', 'running')", (job["active_attempt_id"],))
        return CancellationReceipt(job_id=job_id, expected_attempt_id=expected_attempt_id,
                                   state="cancelling", fence=job["fence"] + 1,
                                   requested_at=stamp)


def _cancel_precheck(job: sqlite3.Row, expected: str | None,
                     stamp: str) -> CancellationReceipt | None:
    base = {"job_id": job["job_id"], "expected_attempt_id": expected, "fence": job["fence"],
            "requested_at": stamp}
    if job["state"] in _TERMINAL_JOB_STATES:
        return CancellationReceipt(state="already_terminal", **base)
    if job["active_attempt_id"] != expected:
        return CancellationReceipt(state="conflict", failure=make_problem(
            "STALE_EXPECTATION", "the expected attempt is not the job's active attempt"), **base)
    if job["state"] == "cancelling":
        return CancellationReceipt(state="cancelling", **base)
    return None


def complete_cancel(conn: sqlite3.Connection, job_id: str, *, process_state: str,
                    clock: Clock) -> CancellationReceipt:
    """Finish a cancellation once the executor reports the worker tree's fate."""
    now = clock.now()
    stamp = format_timestamp(now)
    with transaction(conn):
        job = _job(conn, job_id)
        attempt_id = job["active_attempt_id"]
        if job["state"] != "cancelling" or attempt_id is None:
            raise fail("STALE_EXPECTATION", "the job is not being cancelled")
        base = {"job_id": job_id, "expected_attempt_id": attempt_id, "fence": job["fence"],
                "requested_at": stamp}
        if process_state not in RELEASABLE_PROCESS_STATES:
            conn.execute("UPDATE attempts SET process_state = ? WHERE attempt_id = ?",
                         (process_state, attempt_id))
            return CancellationReceipt(state="cancelling", **base)
        failure = make_problem("CANCELLED", "cancelled by request")
        end_attempt(conn, attempt_id, "cancelled",
                    Outcome(succeeded=False, process_state=process_state, failure=failure), now)
        conn.execute("UPDATE jobs SET state = 'cancelled', active_attempt_id = NULL, "
                     "failure_json = ?, updated_at = ? WHERE job_id = ?",
                     (dumps(failure), stamp, job_id))
        block_descendants(conn, job_id, now)
        return CancellationReceipt(state="cancelled", completed_at=stamp, **base)


def _job(conn: sqlite3.Connection, job_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        raise fail("INVALID_REQUEST", "unknown job", details={"reason": "not_found"})
    return row


# --------------------------------------------------------------------------
# progress and reads
# --------------------------------------------------------------------------


def record_progress(conn: sqlite3.Connection, event: ProgressEvent) -> bool:
    """Append one event. An identical duplicate is a no-op; a different one is refused."""
    body = dumps(event)
    with transaction(conn):
        attempt = conn.execute("SELECT job_id FROM attempts WHERE attempt_id = ?",
                               (event.attempt_id,)).fetchone()
        if attempt is None or attempt["job_id"] != event.job_id:
            raise fail("INVALID_REQUEST", "progress names an unknown attempt for this job")
        existing = conn.execute("SELECT body_json FROM progress_events "
                                "WHERE attempt_id = ? AND sequence = ?",
                                (event.attempt_id, event.sequence)).fetchone()
        if existing is not None:
            if existing["body_json"] != body:
                raise fail("INTEGRITY_FAILED", "a different progress event already holds "
                           "this sequence number")
            return False
        conn.execute("INSERT INTO progress_events (attempt_id, sequence, job_id, kind, "
                     "recorded_at, body_json) VALUES (?, ?, ?, ?, ?, ?)",
                     (event.attempt_id, event.sequence, event.job_id, event.kind,
                      event.recorded_at, body))
    return True


def attempt_receipts(conn: sqlite3.Connection, job_id: str) -> list[AttemptReceipt]:
    rows = conn.execute("SELECT * FROM attempts WHERE job_id = ? ORDER BY attempt_number",
                        (job_id,)).fetchall()
    result = []
    for row in rows:
        output_refs = tuple(item[0] for item in conn.execute(
            "SELECT artifact_id FROM attempt_outputs WHERE attempt_id = ? ORDER BY name",
            (row["attempt_id"],)).fetchall())
        checkpoint_refs = tuple(item[0] for item in conn.execute(
            "SELECT cache_key FROM checkpoints WHERE producer_attempt_id = ? ORDER BY cache_key",
            (row["attempt_id"],)).fetchall())
        result.append(AttemptReceipt(
            job_id=row["job_id"], attempt_id=row["attempt_id"],
            attempt_number=row["attempt_number"], fence=row["fence"],
            supervisor_epoch=row["supervisor_epoch"], host_boot_id=row["host_boot_id"],
            state=row["state"], process_state=row["process_state"],
            process_identity=load_json(ProcessIdentity, row["process_json"]),
            started_at=row["started_at"], heartbeat_at=row["heartbeat_at"],
            lease_expires_at=row["lease_expires_at"],
            resolved_resources=load_json(ResolvedResources, row["resources_json"]),
            checkpoint_refs=checkpoint_refs, output_refs=output_refs,
            failure=load_json(Problem, row["failure_json"]),
        ))
    return result
