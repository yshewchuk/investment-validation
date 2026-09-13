"""The ops-owned catalog schema, as numbered migrations — phase-1 guide §6.1.

Invariants live in the schema, not only in Python prechecks:

* ``jobs`` is unique on ``(namespace, idempotency_key)``; a job is ``running``
  or ``cancelling`` exactly when it names an active attempt.
* ``attempts`` are unique on ``(job_id, attempt_number)`` and ``(job_id,
  fence)``, and at most one attempt per job is live at a time. Previous
  attempts are never overwritten; a retry is a new row.
* ``cpu_assignments`` has a partial unique index over unreleased rows, so two
  attempts cannot hold the same CPU even if the admission code were wrong (O03).
* Reservations are released by setting ``released_at``, never by deleting the
  row, so the evidence of what an attempt held survives reconciliation.

Later slices add tables as later migrations — artifacts and checkpoints,
provider reservations, releases and outbox — rather than editing version 1.
"""
from __future__ import annotations

from engine.v2.ops.migrations import Migration
from engine.v2.ops.schema_runtime import STATEMENTS

__all__ = ["MIGRATIONS", "OWNER"]

OWNER = "ops"

_JOB_STATES = ("'queued','running','succeeded','retry_wait','failed',"
               "'cancelling','cancelled','blocked'")
_ATTEMPT_STATES = ("'starting','running','succeeded','failed','cancelling',"
                   "'cancelled','recovery_pending'")
_PROCESS_STATES = "'unlaunched','alive','exited','verified_dead','unknown','quarantined'"

_V1 = (
    """CREATE TABLE supervisor_epochs (
        epoch_id TEXT PRIMARY KEY,
        boot_id TEXT NOT NULL,
        pid INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT
    ) STRICT""",
    f"""CREATE TABLE jobs (
        job_id TEXT PRIMARY KEY,
        namespace TEXT NOT NULL,
        idempotency_key TEXT NOT NULL,
        request_digest TEXT NOT NULL,
        principal TEXT NOT NULL,
        kind TEXT NOT NULL,
        spec_hash TEXT,
        spec_json TEXT NOT NULL,
        resource_class TEXT NOT NULL,
        checkpoint_contract_ref TEXT NOT NULL,
        retry_json TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ({_JOB_STATES})),
        priority INTEGER NOT NULL,
        deadline_at TEXT,
        max_attempts INTEGER NOT NULL CHECK (max_attempts > 0),
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        next_eligible_at TEXT,
        fence INTEGER NOT NULL DEFAULT 0 CHECK (fence >= 0),
        attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
        active_attempt_id TEXT REFERENCES attempts(attempt_id) DEFERRABLE INITIALLY DEFERRED,
        queue_reason_json TEXT,
        failure_json TEXT,
        UNIQUE (namespace, idempotency_key),
        CHECK ((state IN ('running', 'cancelling')) = (active_attempt_id IS NOT NULL))
    ) STRICT""",
    """CREATE TABLE job_dependencies (
        child_job_id TEXT NOT NULL REFERENCES jobs(job_id),
        parent_job_id TEXT NOT NULL REFERENCES jobs(job_id),
        required_output_contract TEXT NOT NULL,
        PRIMARY KEY (child_job_id, parent_job_id),
        CHECK (child_job_id <> parent_job_id)
    ) STRICT""",
    f"""CREATE TABLE attempts (
        attempt_id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL REFERENCES jobs(job_id),
        attempt_number INTEGER NOT NULL CHECK (attempt_number > 0),
        fence INTEGER NOT NULL CHECK (fence > 0),
        supervisor_epoch TEXT NOT NULL REFERENCES supervisor_epochs(epoch_id),
        host_boot_id TEXT NOT NULL,
        state TEXT NOT NULL CHECK (state IN ({_ATTEMPT_STATES})),
        process_state TEXT NOT NULL CHECK (process_state IN ({_PROCESS_STATES})),
        process_json TEXT,
        resources_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        started_at TEXT,
        heartbeat_at TEXT,
        lease_expires_at TEXT NOT NULL,
        ended_at TEXT,
        exit_code INTEGER,
        memory_current_bytes INTEGER,
        memory_peak_bytes INTEGER,
        memory_sampled_at TEXT,
        failure_json TEXT,
        UNIQUE (job_id, attempt_number),
        UNIQUE (job_id, fence)
    ) STRICT""",
    """CREATE UNIQUE INDEX attempts_one_live_per_job ON attempts(job_id)
        WHERE state IN ('starting', 'running', 'cancelling', 'recovery_pending')""",
    """CREATE TABLE resource_reservations (
        attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
        policy_version TEXT NOT NULL,
        profile TEXT NOT NULL,
        memory_bytes INTEGER NOT NULL CHECK (memory_bytes >= 0),
        scratch_bytes INTEGER NOT NULL CHECK (scratch_bytes >= 0),
        heavy INTEGER NOT NULL CHECK (heavy IN (0, 1)),
        disk_heavy INTEGER NOT NULL CHECK (disk_heavy IN (0, 1)),
        measured INTEGER NOT NULL CHECK (measured IN (0, 1)),
        created_at TEXT NOT NULL,
        released_at TEXT,
        release_reason TEXT
    ) STRICT""",
    """CREATE TABLE cpu_assignments (
        attempt_id TEXT NOT NULL REFERENCES resource_reservations(attempt_id),
        cpu_id INTEGER NOT NULL CHECK (cpu_id >= 0),
        released_at TEXT,
        PRIMARY KEY (attempt_id, cpu_id)
    ) STRICT""",
    "CREATE UNIQUE INDEX cpu_assignments_active ON cpu_assignments(cpu_id) "
    "WHERE released_at IS NULL",
    """CREATE TABLE progress_events (
        attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
        sequence INTEGER NOT NULL CHECK (sequence >= 0),
        job_id TEXT NOT NULL REFERENCES jobs(job_id),
        kind TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        body_json TEXT NOT NULL,
        PRIMARY KEY (attempt_id, sequence)
    ) STRICT""",
    "CREATE INDEX jobs_ready ON jobs(state, next_eligible_at, priority, created_at)",
    "CREATE INDEX job_dependencies_parent ON job_dependencies(parent_job_id)",
    "CREATE INDEX attempts_leases ON attempts(state, lease_expires_at)",
    "CREATE INDEX resource_reservations_open ON resource_reservations(released_at)",
    "CREATE INDEX progress_events_job ON progress_events(job_id, recorded_at)",
)

MIGRATIONS = (
    Migration(version=1, name="jobs_attempts_reservations", statements=_V1),
    Migration(version=2, name="runtime_artifacts_effects", statements=STATEMENTS),
    Migration(version=3, name="fenced_outbox_claims", statements=(
        "ALTER TABLE outbox ADD COLUMN claim_token TEXT",
        "ALTER TABLE outbox ADD COLUMN claimed_at TEXT",
        "ALTER TABLE outbox ADD COLUMN claimed_by TEXT",
    )),
    Migration(version=4, name="outbox_claim_leases", statements=(
        "ALTER TABLE outbox ADD COLUMN claim_expires_at TEXT",
    )),
    Migration(version=5, name="store_read_pins", statements=(
        """CREATE TABLE store_read_pins (
            attempt_id TEXT PRIMARY KEY REFERENCES attempts(attempt_id),
            manifest_json TEXT NOT NULL,
            read_set_complete INTEGER NOT NULL CHECK (read_set_complete IN (0, 1))
        ) STRICT""",
    )),
)
