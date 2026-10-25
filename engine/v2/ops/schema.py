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

NATIVE_STAGE_FAILURE_TEXT_BYTES = {
    "occurrence": 64,
    "scope_hash": 128,
    "stage_id": 128,
    "status": 16,
    "error_code": 128,
    "problem_json": 8192,
    "created_at": 64,
    "updated_at": 64,
}

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
    Migration(version=6, name="smoke_runs_hold_no_hypotheses", statements=(
        "DELETE FROM hypotheses WHERE run_id IN "
        "(SELECT run_id FROM experiment_runs WHERE mode='smoke')",
    )),
    Migration(version=7, name="attempt_input_bindings", statements=(
        """CREATE TABLE attempt_input_bindings (
            attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
            name TEXT NOT NULL,
            binding TEXT NOT NULL,
            artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
            content_hash TEXT NOT NULL,
            PRIMARY KEY (attempt_id, name)
        ) STRICT""",
        """CREATE TRIGGER attempt_input_bindings_no_update
        BEFORE UPDATE ON attempt_input_bindings
        BEGIN
            SELECT RAISE(ABORT, 'attempt_input_bindings rows are immutable');
        END""",
        """CREATE TRIGGER attempt_input_bindings_no_delete
        BEFORE DELETE ON attempt_input_bindings
        BEGIN
            SELECT RAISE(ABORT, 'attempt_input_bindings rows are immutable');
        END""",
    )),
    # Phase 3 guide §5.5 item 1 ("make nightly effects generation-aware"):
    # ``watermarks`` was keyed on (pipeline, scope, stage) alone, so a second
    # same-session generation's engineering_gate/ledger_export/backup/
    # publication/delivery receipt collided with an earlier generation's
    # already-completed one (IDEMPOTENCY_CONFLICT: "completed occurrence has
    # another receipt" -- the 2026-09-14 real-run failure). SQLite cannot
    # ALTER a PRIMARY KEY in place, so this rebuilds the table with
    # ``generation`` folded into the key; every existing row is carried
    # forward under ``generation=''`` (the old, still-default, bucket), so
    # every caller that never passes ``generation=`` keeps its exact old
    # behaviour and existing rows keep their exact old identity.
    Migration(version=8, name="generation_aware_watermarks", statements=(
        """CREATE TABLE watermarks_v2 (
            pipeline TEXT NOT NULL, scope TEXT NOT NULL, stage TEXT NOT NULL,
            generation TEXT NOT NULL DEFAULT '',
            occurrence TEXT NOT NULL, receipt_ref TEXT NOT NULL, completed_at TEXT NOT NULL,
            PRIMARY KEY(pipeline, scope, stage, generation)
        ) STRICT""",
        "INSERT INTO watermarks_v2(pipeline,scope,stage,generation,occurrence,receipt_ref,completed_at) "
        "SELECT pipeline,scope,stage,'',occurrence,receipt_ref,completed_at FROM watermarks",
        "DROP TABLE watermarks",
        "ALTER TABLE watermarks_v2 RENAME TO watermarks",
    )),
    # Phase 3 guide §5.5 item 2 ("populate live engineering health from real
    # observations"): a retry or a later generation observing the SAME
    # scheduled night must update that night's own record without ever
    # counting as a second night (health_observations' existing
    # PRIMARY KEY(occurrence, kind) already collapses them onto one row) --
    # but the engineering history window still needs to show a retry count
    # per night. Every existing row starts at 0 (observed exactly once so
    # far, the true count for every row this migration finds).
    Migration(version=9, name="health_observation_retries", statements=(
        "ALTER TABLE health_observations ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0",
    )),
    Migration(version=10, name="final_holdout_authorizations", statements=(
        """CREATE TABLE final_holdout_authorizations (
            experiment_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL CHECK(schema_version='final_holdout_authorization.v1.0'),
            registration_run_id TEXT NOT NULL REFERENCES experiment_runs(run_id),
            request_digest TEXT NOT NULL, request_json TEXT NOT NULL,
            actor TEXT NOT NULL CHECK(length(actor)>0), authorized_at TEXT NOT NULL,
            confirmation TEXT NOT NULL CHECK(length(confirmation)>0)
        ) STRICT""",
        """CREATE TRIGGER final_holdout_authorizations_no_update BEFORE UPDATE ON final_holdout_authorizations
        BEGIN SELECT RAISE(ABORT, 'final holdout authorizations are immutable'); END""",
        """CREATE TRIGGER final_holdout_authorizations_no_delete BEFORE DELETE ON final_holdout_authorizations
        BEGIN SELECT RAISE(ABORT, 'final holdout authorizations are immutable'); END""",
        """CREATE TRIGGER final_holdout_authorizations_no_replace BEFORE INSERT ON final_holdout_authorizations
        WHEN EXISTS (SELECT 1 FROM final_holdout_authorizations WHERE experiment_id=NEW.experiment_id)
        BEGIN SELECT RAISE(ABORT, 'final holdout authorizations are immutable'); END""",
    )),
    Migration(version=11, name="final_holdout_state_events", statements=(
        """CREATE TABLE final_holdout_state_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id TEXT NOT NULL REFERENCES final_holdout_authorizations(experiment_id),
            request_digest TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('read_started','revoked','indeterminate')),
            fence INTEGER NOT NULL CHECK(fence > 0),
            owner_boot_id TEXT CHECK(state <> 'read_started' OR owner_boot_id IS NOT NULL),
            owner_pid INTEGER CHECK(state <> 'read_started' OR owner_pid IS NOT NULL),
            owner_start_ticks INTEGER CHECK(state <> 'read_started' OR owner_start_ticks IS NOT NULL),
            owner_process_group INTEGER CHECK(state <> 'read_started' OR owner_process_group IS NOT NULL),
            actor TEXT CHECK(state NOT IN ('read_started','revoked') OR actor IS NOT NULL),
            occurred_at TEXT NOT NULL,
            confirmation TEXT CHECK(state NOT IN ('read_started','revoked') OR confirmation IS NOT NULL)
        ) STRICT""",
        "CREATE INDEX final_holdout_state_events_experiment ON final_holdout_state_events(experiment_id, event_id)",
        """CREATE TRIGGER final_holdout_state_events_no_update BEFORE UPDATE ON final_holdout_state_events
        BEGIN SELECT RAISE(ABORT, 'final holdout state events are immutable'); END""",
        """CREATE TRIGGER final_holdout_state_events_no_delete BEFORE DELETE ON final_holdout_state_events
        BEGIN SELECT RAISE(ABORT, 'final holdout state events are immutable'); END""",
        """CREATE TRIGGER final_holdout_state_events_no_replace BEFORE INSERT ON final_holdout_state_events
        WHEN EXISTS (SELECT 1 FROM final_holdout_state_events WHERE event_id=NEW.event_id)
        BEGIN SELECT RAISE(ABORT, 'final holdout state events are immutable'); END""",
    )),
    Migration(version=12, name="massive_acquisition_occurrences", statements=(
        """CREATE TABLE ops_massive_occurrences (
            occurrence_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            request_hash TEXT NOT NULL,
            family TEXT NOT NULL,
            generation TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL CHECK(end_date >= start_date),
            raw_receipt_id TEXT NOT NULL REFERENCES data_raw_receipts(raw_receipt_id),
            document_json TEXT NOT NULL,
            UNIQUE(job_id, request_hash)
        ) STRICT""",
        "CREATE INDEX massive_occurrence_coverage ON ops_massive_occurrences(family,generation,start_date,end_date)",
        """CREATE TRIGGER massive_occurrences_no_update BEFORE UPDATE ON ops_massive_occurrences
        BEGIN SELECT RAISE(ABORT, 'Massive occurrences are immutable'); END""",
        """CREATE TRIGGER massive_occurrences_no_delete BEFORE DELETE ON ops_massive_occurrences
        BEGIN SELECT RAISE(ABORT, 'Massive occurrences are immutable'); END""",
        """CREATE TRIGGER massive_occurrences_no_replace BEFORE INSERT ON ops_massive_occurrences
        WHEN EXISTS (SELECT 1 FROM ops_massive_occurrences WHERE occurrence_id=NEW.occurrence_id
                     OR (job_id=NEW.job_id AND request_hash=NEW.request_hash))
        BEGIN SELECT RAISE(ABORT, 'Massive occurrences are immutable'); END""",
        """CREATE TABLE ops_massive_pages (
            page_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            request_hash TEXT NOT NULL,
            page_index INTEGER NOT NULL CHECK(page_index >= 0),
            raw_receipt_id TEXT NOT NULL REFERENCES data_raw_receipts(raw_receipt_id),
            document_json TEXT NOT NULL,
            UNIQUE(job_id, request_hash, page_index)
        ) STRICT""",
        """CREATE TRIGGER massive_pages_no_update BEFORE UPDATE ON ops_massive_pages
        BEGIN SELECT RAISE(ABORT, 'Massive pages are immutable'); END""",
        """CREATE TRIGGER massive_pages_no_delete BEFORE DELETE ON ops_massive_pages
        BEGIN SELECT RAISE(ABORT, 'Massive pages are immutable'); END""",
        """CREATE TRIGGER massive_pages_no_replace BEFORE INSERT ON ops_massive_pages
        WHEN EXISTS (SELECT 1 FROM ops_massive_pages WHERE page_id=NEW.page_id OR
                     (job_id=NEW.job_id AND request_hash=NEW.request_hash AND page_index=NEW.page_index))
        BEGIN SELECT RAISE(ABORT, 'Massive pages are immutable'); END""",
    )),
    Migration(version=13, name="native_gate_sweep_rosters", statements=(
        """CREATE TABLE native_gate_sweeps (
            experiment_id TEXT PRIMARY KEY, document_json TEXT NOT NULL
        ) STRICT""",
        """CREATE TRIGGER native_gate_sweeps_no_update BEFORE UPDATE ON native_gate_sweeps
        BEGIN SELECT RAISE(ABORT, 'native gate rosters are immutable'); END""",
        """CREATE TRIGGER native_gate_sweeps_no_delete BEFORE DELETE ON native_gate_sweeps
        BEGIN SELECT RAISE(ABORT, 'native gate rosters are immutable'); END""",
        """CREATE TRIGGER native_gate_sweeps_no_replace BEFORE INSERT ON native_gate_sweeps
        WHEN EXISTS (SELECT 1 FROM native_gate_sweeps WHERE experiment_id=NEW.experiment_id)
        BEGIN SELECT RAISE(ABORT, 'native gate rosters are immutable'); END""",
    )),
    Migration(version=14, name="native_stage_failures", statements=(
        """CREATE TABLE native_stage_failures (
            occurrence TEXT NOT NULL CHECK (length(CAST(occurrence AS BLOB)) <= {occurrence}),
            scope_hash TEXT NOT NULL CHECK (length(CAST(scope_hash AS BLOB)) <= {scope_hash}),
            stage_id TEXT NOT NULL CHECK (length(CAST(stage_id AS BLOB)) <= {stage_id}),
            status TEXT NOT NULL CHECK (status = 'FAILED')
                CHECK (length(CAST(status AS BLOB)) <= {status}),
            error_code TEXT NOT NULL CHECK (length(CAST(error_code AS BLOB)) <= {error_code}),
            problem_json TEXT NOT NULL CHECK (length(CAST(problem_json AS BLOB)) <= {problem_json}),
            created_at TEXT NOT NULL CHECK (length(CAST(created_at AS BLOB)) <= {created_at}),
            updated_at TEXT NOT NULL CHECK (length(CAST(updated_at AS BLOB)) <= {updated_at}),
            PRIMARY KEY(occurrence, scope_hash, stage_id)
        ) STRICT""".format(**NATIVE_STAGE_FAILURE_TEXT_BYTES),
    )),
    Migration(version=15, name="native_stage_failure_resolution", statements=(
        """ALTER TABLE native_stage_failures ADD COLUMN resolved_at TEXT
            CHECK (resolved_at IS NULL
                   OR length(CAST(resolved_at AS BLOB)) <= {updated_at})""".format(
                       **NATIVE_STAGE_FAILURE_TEXT_BYTES),
    )),
)
