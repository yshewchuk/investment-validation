"""One supervisor per catalog, and fencing before reconciliation — §6.2–§6.3.

**The held lock is what matters, not the lock file.** A file naming a dead PID
is a record of the last holder, not evidence that anything is alive. The OS
releases an ``flock`` when its holder dies, so a restarted supervisor either
acquires it or genuinely has a live competitor.

After supervisor death, lease expiry or a host resume, recovery runs in this
order and never skips ahead:

1. invalidate the old attempt's fence and **retain** its reservations;
2. let the executor reconcile the actual process tree by boot ID plus process
   start identity — never PID alone, which may now name another job;
3. only when the tree is verified gone, release reservations and let the job
   retry or fail under its policy.

A tree that cannot be proved gone stays ``recovery_pending`` with its
reservation held, which keeps replacement heavy work out until an operator or
a later reconciliation settles it. "Proved gone" is :func:`prove_ownership_gone`
(B1): every recorded identity gone or a zombie, no live session still carrying
the launch pid, and no live process's environ carrying the attempt's staging
marker — the last two catch a ``setsid()`` escaper and a launch that crashed
before its identity was ever recorded, neither of which a process-group walk
alone can see. Ownership is never released on identity count or a guess; an
operator settles the same proof through ``python3 -m engine.v2.ops reconcile``.
"""
from __future__ import annotations

import fcntl
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from engine.v2.contracts import ProcessIdentity
from engine.v2.foundation import Clock, content_hash, format_timestamp
from engine.v2.ops.catalog import load_json, transaction
from engine.v2.ops.errors import fail, make_problem
from engine.v2.ops.executor_watchdog import find_owners, observe
from engine.v2.ops.lifecycle import (
    RELEASABLE_PROCESS_STATES,
    Outcome,
    advance_job,
    block_descendants,
    end_attempt,
)

__all__ = [
    "OwnershipProof",
    "SupervisorLock",
    "begin_epoch",
    "expire_leases",
    "fence_foreign_epochs",
    "prove_ownership_gone",
    "read_boot_id",
    "reconcile_attempt",
]

_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")


def read_boot_id() -> str:
    return _BOOT_ID.read_text().strip()


class SupervisorLock:
    """An exclusive, non-blocking ``flock`` on ``<root>/supervisor.lock``."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self) -> bool:
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None


def begin_epoch(conn: sqlite3.Connection, *, clock: Clock, boot_id: str, pid: int) -> str:
    stamp = format_timestamp(clock.now())
    epoch_id = "sup_" + content_hash({"boot_id": boot_id, "pid": pid,
                                      "started_at": stamp}).removeprefix("sha256:")[:24]
    with transaction(conn):
        conn.execute("INSERT INTO supervisor_epochs (epoch_id, boot_id, pid, started_at) "
                     "VALUES (?, ?, ?, ?)", (epoch_id, boot_id, pid, stamp))
    return epoch_id


def _fence_off(conn: sqlite3.Connection, attempt_id: str, job_id: str, stamp: str) -> None:
    conn.execute("UPDATE attempts SET state = 'recovery_pending' WHERE attempt_id = ?",
                 (attempt_id,))
    conn.execute("UPDATE jobs SET fence = fence + 1, updated_at = ? "
                 "WHERE job_id = ? AND active_attempt_id = ?", (stamp, job_id, attempt_id))


def expire_leases(conn: sqlite3.Connection, *, clock: Clock) -> list[str]:
    """Fence off every live attempt whose lease has run out. Reservations stay held."""
    stamp = format_timestamp(clock.now())
    with transaction(conn):
        rows = conn.execute("SELECT attempt_id, job_id FROM attempts "
                            "WHERE state IN ('starting', 'running', 'cancelling') "
                            "AND lease_expires_at <= ?", (stamp,)).fetchall()
        for row in rows:
            _fence_off(conn, row["attempt_id"], row["job_id"], stamp)
    return [row["attempt_id"] for row in rows]


def fence_foreign_epochs(conn: sqlite3.Connection, *, epoch_id: str, clock: Clock) -> list[str]:
    """On supervisor start: every live attempt of another epoch needs reconciliation."""
    stamp = format_timestamp(clock.now())
    with transaction(conn):
        rows = conn.execute("SELECT attempt_id, job_id FROM attempts "
                            "WHERE state IN ('starting', 'running', 'cancelling') "
                            "AND supervisor_epoch <> ?", (epoch_id,)).fetchall()
        for row in rows:
            _fence_off(conn, row["attempt_id"], row["job_id"], stamp)
    return [row["attempt_id"] for row in rows]


@dataclass(frozen=True)
class OwnershipProof:
    """The B1 ownership proof for one ``recovery_pending`` attempt (§6.3, §9.1).

    ``proven`` only when (a) every recorded identity is gone or a zombie, (b)
    no live session still carries the launch pid at or after its own start,
    and (c) no live process's environ carries this attempt's staging marker.
    A process-group walk alone (``known``/``alive``) proves (a) but cannot see
    a ``setsid()`` escaper or a worker whose launch crashed before its
    identity was recorded — (b) and (c) are what catch those.
    """

    proven: bool
    known: tuple[ProcessIdentity, ...]
    alive: tuple[ProcessIdentity, ...]
    blockers: tuple[tuple[int, int], ...]


def prove_ownership_gone(conn: sqlite3.Connection, attempt_id: str, *,
                         boot_id: str) -> OwnershipProof:
    """Prove or refuse release for one attempt, against one fresh ``/proc`` table.

    Never mutates the catalog: callers persist ``known`` and signal ``alive``
    themselves, then settle through :func:`reconcile_attempt`.
    """
    row = conn.execute("SELECT process_json, host_boot_id FROM attempts WHERE attempt_id = ?",
                       (attempt_id,)).fetchone()
    if row is None:
        raise fail("INVALID_REQUEST", "unknown attempt")
    if row["host_boot_id"] != boot_id:
        # Nothing launched under another boot can still run, and its pid and
        # start ticks mean nothing against this boot's process table: comparing
        # them would only manufacture blockers out of unrelated sessions.
        return OwnershipProof(proven=True, known=(), alive=(), blockers=())
    members = conn.execute("SELECT identity_json FROM process_members WHERE attempt_id = ?",
                           (attempt_id,)).fetchall()
    identities = tuple(load_json(ProcessIdentity, member[0]) for member in members)
    launch = load_json(ProcessIdentity, row["process_json"]) if row["process_json"] else None
    if not identities and launch is not None:
        identities = (launch,)
    known, alive, _ = observe(identities, boot_id)
    blockers = find_owners(boot_id, launch_pid=launch.pid if launch else None,
                           launch_start_ticks=launch.start_ticks if launch else None,
                           marker=attempt_id)
    return OwnershipProof(proven=not alive and not blockers, known=known, alive=alive,
                          blockers=blockers)


def reconcile_attempt(conn: sqlite3.Connection, attempt_id: str, *, process_state: str,
                      clock: Clock) -> str:
    """Resolve a ``recovery_pending`` attempt from the executor's process verdict.

    Returns the attempt's state afterwards: still ``recovery_pending`` unless the
    tree is verified gone.
    """
    now = clock.now()
    with transaction(conn):
        attempt = conn.execute("SELECT * FROM attempts WHERE attempt_id = ?",
                               (attempt_id,)).fetchone()
        if attempt is None or attempt["state"] != "recovery_pending":
            raise fail("STALE_EXPECTATION", "the attempt is not awaiting reconciliation")
        if process_state not in RELEASABLE_PROCESS_STATES:
            conn.execute("UPDATE attempts SET process_state = ? WHERE attempt_id = ?",
                         (process_state, attempt_id))
            return "recovery_pending"
        job = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (attempt["job_id"],)).fetchone()
        return _settle(conn, job, attempt_id, process_state, now)


def _settle(conn: sqlite3.Connection, job: sqlite3.Row, attempt_id: str, process_state: str,
            now) -> str:
    if job["state"] == "cancelling":
        failure = make_problem("CANCELLED", "cancelled; the worker was reconciled after "
                               "losing its lease")
        end_attempt(conn, attempt_id, "cancelled",
                    Outcome(succeeded=False, process_state=process_state, failure=failure), now)
        conn.execute("UPDATE jobs SET state = 'cancelled', active_attempt_id = NULL, "
                     "failure_json = NULL, updated_at = ? WHERE job_id = ?",
                     (format_timestamp(now), job["job_id"]))
        block_descendants(conn, job["job_id"], now)
        return "cancelled"
    failure = make_problem("LEASE_LOST", "the attempt lost its lease; its process tree was "
                           "verified gone before resources were released")
    outcome = Outcome(succeeded=False, process_state=process_state, failure=failure)
    end_attempt(conn, attempt_id, "failed", outcome, now)
    if job["active_attempt_id"] == attempt_id:
        advance_job(conn, job, outcome, now)
    return "failed"
