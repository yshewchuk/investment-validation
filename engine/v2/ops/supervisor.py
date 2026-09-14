"""Small polling coordinator; computation lives in bounded fresh subprocesses."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from engine.v2.contracts import CheckpointCandidate, OutputCandidate, ProgressEvent
from engine.v2.foundation import (
    ArtifactStore,
    artifact_reference,
    content_hash,
    format_timestamp,
    to_document,
)
from engine.v2.ops import executor
from engine.v2.ops.checkpoints import (
    artifact,
    cache_identity,
    commit_checkpoint,
    register_artifact,
    reuse,
)
from engine.v2.ops.decision_commit import (
    commit_decisions_in_transaction,
    import_settlement_candidates_in_transaction,
    validated_decision_candidate,
)
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.effects_graph import (
    backup_effect,
    engineering_gate_effect,
    ledger_export_effect,
    publication_effect,
)
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.executor_watchdog import is_alive, signal_owned
from engine.v2.ops.fingerprints import (
    environment_identity,
    snapshot_code,
    worker_source_manifest,
)
from engine.v2.ops.generation_binding import refuse_generation_mismatch
from engine.v2.ops.input_bindings import recorded_bindings, resolve_bindings, resolved_inputs_hash
from engine.v2.ops.legacy_adapter import copy_read_set
from engine.v2.ops.lifecycle import (
    Keepalive,
    Outcome,
    commit_attempt,
    complete_cancel,
    fence_held,
    heartbeat,
    record_measurement,
    record_progress,
    renew_after_resume,
)
from engine.v2.ops.recovery import (
    SupervisorLock,
    begin_epoch,
    expire_leases,
    fence_foreign_epochs,
    prove_ownership_gone,
    read_boot_id,
    reconcile_attempt,
)
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.snapshot_promotion import legacy_rebuild_candidate_effect, snapshot_import_effect
from engine.v2.ops.snapshot_roots import default_materialization_base
from engine.v2.ops.snapshot_stages import (
    cache_inputs,
    confirm_attempt,
    materialize_effect,
    prepare_launch,
)
from engine.v2.ops.stages import BARRIER_ONLY_REASONS, validate_result
from engine.v2.ops.store_barrier import (
    confirm_read_set,
    domains_of,
    pin_read_set,
    read_set_complete,
    verified_write_in,
)

#: Kinds whose coordinator effect does real, non-idempotent catalog/outbox/
#: filesystem work every attempt — the generic checkpoint-reuse shortcut
#: would otherwise skip that work entirely on a cache hit (decision #1).
_COORDINATOR_EFFECT_KINDS = frozenset({
    "legacy_decisions", "legacy_settlement", "legacy_render", "legacy_selfcheck",
    "decision_evidence", "ledger_export", "engineering_gate", "publication", "backup",
    "snapshot_import", "legacy_rebuild_candidate", "legacy_materialize"})

#: The attempt lease every heartbeat, resume renewal and keepalive extends to.
LEASE_SECONDS = 120

#: A heartbeat ``progress_events`` row and memory sample are written at most
#: once per this many seconds per attempt, plus on every observed state change
#: (stopping, exited). Lease renewal still runs on every poll (D).
HEARTBEAT_EVENT_SECONDS = 10.0

#: A commit refused with one of these can never succeed under this fence:
#: nothing more may be written under it, not even the attempt's own failure.
_FENCE_LOST_CODES = frozenset({"LEASE_LOST", "CANCELLED"})


class Service:
    def __init__(self, conn, root, registry, policy, *, clock, code_source, store_root=None,
                 materialization_base=None):
        self.conn, self.root, self.registry, self.policy = conn, Path(root), registry, policy
        self.clock, self.code_source = clock, Path(code_source)
        self.store_root = Path(store_root) if store_root is not None else self.code_source
        self.store = ArtifactStore(self.root)
        #: P2-6 §9.3: private read-only legacy roots, one per materialization request.
        self.materialization_base = (Path(materialization_base) if materialization_base
                                     else default_materialization_base(self.root))
        self.running = {}
        self.launches = {}
        #: attempt_id -> (monotonic time, observed state) of its last heartbeat row.
        self.observed = {}
        self.boot = read_boot_id()
        self.lock = SupervisorLock(self.root / "supervisor.lock")
        self.identity = None
        self.last_wall = clock.now()
        self.last_mono = clock.monotonic()

    def start(self):
        if not self.lock.acquire():
            raise OpsError(make_problem("RESOURCE_UNAVAILABLE", "another supervisor holds this catalog"))
        epoch = begin_epoch(self.conn, clock=self.clock, boot_id=self.boot, pid=os.getpid())
        self.identity = Supervisor(epoch, self.boot)
        fence_foreign_epochs(self.conn, epoch_id=epoch, clock=self.clock)
        self.reconcile()

    def reconcile(self):
        rows = self.conn.execute("SELECT * FROM attempts WHERE state = 'recovery_pending'").fetchall()
        for row in rows:
            proof = prove_ownership_gone(self.conn, row["attempt_id"], boot_id=self.boot)
            signal_owned(proof.alive, self.boot, hard=True)
            if proof.known:
                executor.persist_members(self.conn, row["attempt_id"], proof.known)
            state = "verified_dead" if proof.proven else "quarantined"
            reconcile_attempt(self.conn, row["attempt_id"], process_state=state, clock=self.clock)

    def tick(self):
        self._clock_check()
        expire_leases(self.conn, clock=self.clock)
        self.reconcile()
        for attempt_id, running in list(self.running.items()):
            if self._poll(running):
                os.close(running.result_fd)
                del self.running[attempt_id]
                self.observed.pop(attempt_id, None)
        claim = claim_next(self.conn, policy=self.policy, sample=sample_capacity(self.root, clock=self.clock),
                           supervisor=self.identity, clock=self.clock, registry=self.registry)
        if claim:
            self._launch(claim)
        return bool(self.running or claim)

    def _clock_check(self):
        wall, mono = self.clock.now(), self.clock.monotonic()
        jump = abs((wall - self.last_wall).total_seconds() - (mono - self.last_mono))
        if jump > 60:
            self._resume_after_jump()
        self.last_wall, self.last_mono = wall, mono

    def _resume_after_jump(self):
        """A wall-clock jump (e.g. host suspend) must not fence a live worker (B2).

        Only this supervisor's own tracked attempts are ever touched here: an
        attempt whose recorded identity is still verifiably alive gets its
        lease renewed, ignoring wall-clock expiry, before ``expire_leases``
        runs later this tick. One whose identity is gone is left alone and
        goes through the normal expiry/reconcile path. No other attempt is
        fenced by a jump.
        """
        for attempt_id, running in self.running.items():
            identity = running.identities[0] if running.identities else None
            if identity is not None and is_alive(identity, self.boot):
                renew_after_resume(self.conn, attempt_id, running.claim.fence,
                                   clock=self.clock, lease_seconds=LEASE_SECONDS)

    def _launch(self, claim):
        try:
            manifest = worker_source_manifest(self.code_source)
            if claim.spec.implementation_ref != content_hash(manifest):
                raise OpsError(make_problem("INPUT_CHANGED", "planned worker implementation changed"))
            if claim.spec.environment_ref != content_hash(
                    environment_identity(claim.resources.thread_count)):
                raise OpsError(make_problem("INPUT_CHANGED", "planned worker environment changed"))
            # P2-6 §9.3: a snapshot-backed stage validates its SnapshotRef, request
            # and root instead of pinning or copying mutable legacy data.
            launch = prepare_launch(self.conn, self.store, claim, base=self.materialization_base)
            if launch is None:
                legacy_manifest = self._pin_read_set(claim)
                if legacy_manifest is not None:
                    self._populate_legacy_staging(claim, legacy_manifest)
            if self._cache_allowed(claim) and claim.spec.kind in self.registry.names() \
                    and claim.spec.kind not in _COORDINATOR_EFFECT_KINDS:
                cached = self._reuse_staged_checkpoint(claim, launch)
                if cached:
                    return
            code = self.root / "code" / content_hash(manifest).split(":")[1]
            snapshot_code(self.code_source, code, manifest)
            running = executor.launch(
                self.conn, claim, self.registry.get(claim.spec.kind), self.store, code,
                clock=self.clock, boot_id=self.boot,
                legacy_root=launch.worker_legacy_root if launch else None,
                envelope_extra=launch.envelope_extra if launch else None)
            self.launches[claim.attempt_id] = launch
            self.running[claim.attempt_id] = running
        except Exception as exc:
            problem = exc.problem if isinstance(exc, OpsError) else make_problem(
                "LAUNCH_FAILED", "trusted worker launch failed")
            commit_attempt(self.conn, claim.attempt_id, claim.fence,
                           Outcome(False, "verified_dead", failure=problem), clock=self.clock)

    def _store_domains(self, claim):
        if claim.spec.kind not in self.registry.names():
            return ()
        return domains_of(self.registry, claim.spec.kind, claim.spec.parameters)

    def _cache_allowed(self, claim):
        """An undeclared or incomplete read set disables cross-run cache reuse (§9.2)."""
        if not any(mode == "read" for _, mode in self._store_domains(claim)):
            return True
        return read_set_complete(self.conn, claim.attempt_id) is True

    def _pin_read_set(self, claim):
        if not any(mode == "read" for _, mode in self._store_domains(claim)):
            return None
        bindings = claim.spec.parameters.get("input_bindings") or {}
        manifest_id = str(bindings.get("legacy_manifest.json") or "")
        if not manifest_id or manifest_id.startswith("job_"):
            raise OpsError(make_problem("INPUT_CHANGED", "legacy read set is not declared"))
        ref = artifact(self.conn, self.store, manifest_id)
        manifest = json.loads(self.store.read_verified(ref))
        # P2-C02: a barrier-only kind has no declared read plan and so can
        # never run snapshot-backed -- bind it here, at the one place its
        # legacy read set is pinned, to the same accepted data/model
        # generation snapshot-backed scoring already ran against. Gated on
        # the plan's own snapshot marker (nightly._stage_parameters): a
        # default legacy-mode nightly leaves it empty, so this never fires
        # against "whatever shadow snapshot happens to exist" -- only a
        # barrier job that is itself part of a snapshot-mode plan graph.
        snapshot_id = str(claim.spec.parameters.get("snapshot_generation_id") or "")
        if claim.spec.kind in BARRIER_ONLY_REASONS and snapshot_id:
            scope = str(claim.spec.parameters.get("snapshot_generation_scope") or "")
            refuse_generation_mismatch(self.conn, self.store, scope=scope,
                                       snapshot_id=snapshot_id, barrier_manifest=manifest)
        pin_read_set(self.conn, claim.attempt_id, manifest, self.store_root)
        return manifest

    def _populate_legacy_staging(self, claim, manifest):
        """A1: nothing else copies the declared legacy inputs into staging."""
        refs = manifest.get("file_refs", [])
        total = sum(int(ref["byte_size"]) for ref in refs)
        if total > claim.resources.scratch_limit_bytes:
            raise OpsError(make_problem(
                "RESOURCE_LIMIT_EXCEEDED", "legacy read set exceeds the scratch budget",
                details={"needed_bytes": total,
                         "scratch_limit_bytes": claim.resources.scratch_limit_bytes}))
        staging = self.store.staging_dir(claim.attempt_id)
        copy_read_set(self.store_root, staging / "legacy", [ref["path"] for ref in refs])

    def _reuse_staged_checkpoint(self, claim, launch=None):
        schema = "receipt.v1.0" if claim.spec.kind == "artifact_check" else "legacy_action.v1.0"
        # Resolve without materializing (B1a): a cache hit here never launches
        # the worker, so nothing is staged or recorded for this attempt.
        resolved = resolve_bindings(self.conn, self.store, claim.spec)
        cache_key = cache_identity(
            kind=claim.spec.kind,
            inputs=cache_inputs(resolved_inputs_hash(claim.spec, resolved), launch),
            implementation=claim.spec.implementation_ref,
            parameters=content_hash(claim.spec.parameters),
            environment=claim.spec.environment_ref,
            schema=schema,
            shard="default")
        receipt = reuse(self.conn, self.store, cache_key)
        if receipt is None:
            return False
        names = [row[0] for row in self.conn.execute(
            "SELECT name FROM attempt_outputs WHERE attempt_id = ? ORDER BY name",
            (receipt.producer_attempt_id,)).fetchall()]
        if len(names) != len(receipt.artifact_refs):
            return False
        def effects(conn):
            for index, ref in enumerate(receipt.artifact_refs):
                conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                             (claim.attempt_id, names[index], ref.artifact_id))
        commit_attempt(self.conn, claim.attempt_id, claim.fence,
                       Outcome(True, "verified_dead", 0), clock=self.clock, effects=effects)
        return True

    def _poll(self, running):
        status = executor.poll(self.conn, running, boot_id=self.boot, clock=self.clock)
        claim = running.claim
        row = self.conn.execute("SELECT state FROM jobs WHERE job_id = ?", (claim.job_id,)).fetchone()
        cancelled = row[0] == "cancelling"
        if cancelled or not heartbeat(self.conn, claim.attempt_id, claim.fence,
                                      clock=self.clock, lease_seconds=LEASE_SECONDS):
            running.failure = "CANCELLED" if cancelled else "LEASE_LOST"
            executor.stop(running, boot_id=self.boot, clock=self.clock)
        if self._observation_due(running, status):
            record_measurement(self.conn, claim.attempt_id, current_bytes=status["memory"],
                               peak_bytes=running.peak, clock=self.clock)
            self._progress(running, status)
        if not status["done"]:
            return False
        if cancelled:
            complete_cancel(self.conn, claim.job_id, process_state="verified_dead", clock=self.clock)
        elif running.failure != "LEASE_LOST":
            self._finish(running, status)
        return True

    def _observation_due(self, running, status):
        """D: one heartbeat row per ``HEARTBEAT_EVENT_SECONDS``, or on a state change."""
        attempt_id, now = running.claim.attempt_id, self.clock.monotonic()
        state = (bool(status["done"]), running.failure)
        last = self.observed.get(attempt_id)
        if last is not None and last[1] == state and now - last[0] < HEARTBEAT_EVENT_SECONDS:
            return False
        self.observed[attempt_id] = (now, state)
        return True

    def _progress(self, running, status):
        sequence = self.conn.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM progress_events "
                                     "WHERE attempt_id = ?", (running.claim.attempt_id,)).fetchone()[0]
        message = ("worker exited" if status["done"] else
                   f"worker stopping: {running.failure}" if running.failure else "worker observed")
        event = ProgressEvent(
            job_id=running.claim.job_id, attempt_id=running.claim.attempt_id,
            stage_id=running.claim.spec.kind, sequence=sequence,
            recorded_at=format_timestamp(self.clock.now()), kind="heartbeat",
            elapsed_seconds=self.clock.monotonic() - running.started, message=message,
            memory_current_bytes=status["memory"], memory_peak_bytes=running.peak)
        record_progress(self.conn, event)

    def _finish(self, running, status):
        claim = running.claim
        launch = self.launches.pop(claim.attempt_id, None)
        keepalive = Keepalive(self.conn, claim.attempt_id, claim.fence, clock=self.clock,
                              lease_seconds=LEASE_SECONDS)
        try:
            self._commit_success(running, status, launch, keepalive)
        except Exception as exc:
            problem = exc.problem if isinstance(exc, OpsError) else make_problem(
                "VALIDATION_FAILED", "worker output failed validation")
            self._commit_failure(claim, status, problem)

    def _commit_success(self, running, status, launch, keepalive):
        claim = running.claim
        code = running.failure
        if status["exit_code"] != 0:
            code = code or ("UNKNOWN_KILL" if status["exit_code"] < 0 else "WORKER_FAILED")
        if code:
            raise OpsError(make_problem(code, "worker did not complete its contract"))
        result = json.loads(running.data)
        outputs = validate_result(claim, result)
        confirm_read_set(self.conn, claim.attempt_id, self.store_root)
        keepalive()
        # P2-6 §9.3 item 4: the same verified snapshot binding before any output.
        confirm_attempt(self.conn, self.store, claim, launch)
        running.peak = max(running.peak, int(result.get("self_peak_bytes", 0)))
        record_measurement(self.conn, claim.attempt_id, current_bytes=0,
                           peak_bytes=running.peak, clock=self.clock)
        if self._cache_allowed(claim) and claim.spec.kind in self.registry.names() \
                and claim.spec.kind not in _COORDINATOR_EFFECT_KINDS:
            refs = self._checkpoint_refs(claim, outputs, launch)
        else:
            refs = [(o["name"], self.store.publish_candidate(
                claim.attempt_id, o["path"], schema_ref=o["schema"],
                max_bytes=claim.resources.scratch_limit_bytes)) for o in outputs]
        keepalive()
        effect, extra_refs = self._coordinator_effect(claim, refs, launch, keepalive)
        _refuse_output_name_collisions(refs, extra_refs)
        keepalive()
        def effects(conn):
            for name, ref in (*refs, *extra_refs):
                register_artifact(conn, ref, claim.attempt_id, self.clock)
                conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                             (claim.attempt_id, name, ref.artifact_id))
            if effect is not None:
                effect(conn)
            for domain, mode in self._store_domains(claim):
                if mode == "write":
                    verified_write_in(conn, claim.attempt_id, domain)
        commit_attempt(self.conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                       clock=self.clock, effects=effects)

    def _commit_failure(self, claim, status, problem):
        """Record the failure under the fence — only while the fence is still held (B)."""
        outcome = Outcome(False, "verified_dead", status["exit_code"], problem)
        if fence_held(self.conn, claim.attempt_id, claim.fence, clock=self.clock):
            try:
                commit_attempt(self.conn, claim.attempt_id, claim.fence, outcome, clock=self.clock)
                return
            except OpsError as exc:
                if exc.code not in _FENCE_LOST_CODES:
                    raise
                problem = exc.problem
        self._strand(claim, problem)

    def _strand(self, claim, problem):
        """Hand an attempt whose fence is gone to the existing recovery state machine.

        Nothing is committed under the stale fence. A job being cancelled
        completes its cancellation (the worker has already exited). Anything
        else is fenced off exactly as a lease expiry would do it
        (``recovery.expire_leases``: attempt ``recovery_pending``, job fence
        bumped, reservations held); ``reconcile`` then settles it on a later
        tick, on supervisor restart, or through ``ops reconcile``. An attempt
        already fenced by someone else is already ``recovery_pending``.
        """
        _report_stranded(claim, problem)
        job = self.conn.execute("SELECT state, active_attempt_id FROM jobs WHERE job_id = ?",
                                (claim.job_id,)).fetchone()
        attempt = self.conn.execute("SELECT state FROM attempts WHERE attempt_id = ?",
                                    (claim.attempt_id,)).fetchone()
        if (job["state"] == "cancelling" and job["active_attempt_id"] == claim.attempt_id
                and attempt["state"] == "cancelling"):
            complete_cancel(self.conn, claim.job_id, process_state="verified_dead", clock=self.clock)
            return
        expire_leases(self.conn, clock=self.clock)

    def _checkpoint_refs(self, claim, outputs, launch):
        schema = outputs[0]["schema"]
        # The attempt already staged from these exact resolved bindings
        # (recorded at launch); the checkpoint's identity must match. A
        # snapshot-backed stage also folds in its snapshot manifest hash,
        # materialization request hash and manifest artifact hash (§9.3 item 5).
        inputs_hash = cache_inputs(resolved_inputs_hash(
            claim.spec, recorded_bindings(self.conn, claim.attempt_id)), launch)
        candidate = CheckpointCandidate(
            shard_key="default",
            cache_key=cache_identity(
                kind=claim.spec.kind,
                inputs=inputs_hash,
                implementation=claim.spec.implementation_ref,
                parameters=content_hash(claim.spec.parameters),
                environment=claim.spec.environment_ref,
                schema=schema, shard="default"),
            input_hash=inputs_hash,
            implementation_hash=claim.spec.implementation_ref,
            parameter_hash=content_hash(claim.spec.parameters),
            environment_hash=claim.spec.environment_ref,
            output_schema_ref=schema,
            outputs=tuple(OutputCandidate(name=o["name"], staged_path=o["path"],
                                           schema_ref=o["schema"]) for o in outputs))
        checkpoint = commit_checkpoint(self.conn, self.store, claim, candidate,
                                       clock=self.clock, inputs_hash=inputs_hash)
        return [(str(index), ref) for index, ref in enumerate(checkpoint.artifact_refs)]

    def _coordinator_effect(self, claim, refs, launch=None, keepalive=None):
        """Validate effect candidates before the short fenced commit transaction.

        Returns ``(effect_fn_or_None, extra_refs)``: ``effect_fn`` runs inside
        the caller's own fenced ``commit_attempt`` transaction (or is
        ``None``), and ``extra_refs`` are additional ``(name, ArtifactRef)``
        outputs the coordinator itself produced — beyond the worker's own
        ``refs`` — to register and record as this attempt's outputs (P2-5/
        Task5: ``ledger_export``'s tar, ``engineering_gate``'s rows).
        ``keepalive`` renews the lease between the effect's own long steps.
        """
        keepalive = keepalive or _no_keepalive
        if claim.spec.kind == "legacy_materialize":
            return materialize_effect(self.conn, self.store, claim, refs, launch,
                                      keepalive=keepalive)
        if claim.spec.kind == "legacy_decisions":
            candidate_ref = _named_ref(refs, "legacy_decisions")
            candidates, context = validated_decision_candidate(
                self.conn, self.store, claim, candidate_ref)
            return (lambda conn: commit_decisions_in_transaction(
                conn, claim, candidates, context, clock=self.clock)), ()
        if claim.spec.kind == "legacy_settlement":
            candidate_ref = _named_ref(refs, "legacy_settlement")
            document = json.loads(self.store.read_verified(candidate_ref))
            rows = document.get("rows") if isinstance(document, dict) else None
            if not isinstance(rows, list):
                raise OpsError(make_problem("VALIDATION_FAILED",
                                            "settlement candidate artifact has no rows"))
            return (lambda conn: import_settlement_candidates_in_transaction(
                conn, claim, candidate_ref, rows, clock=self.clock,
                session=document.get("session"))), ()
        if claim.spec.kind == "decision_evidence":
            _verify_decision_evidence(self.conn, self.store, claim, refs, keepalive)
            return None, ()
        if claim.spec.kind == "ledger_export":
            return ledger_export_effect(self.conn, self.store, claim, self.root, self.code_source,
                                        clock=self.clock, keepalive=keepalive)
        if claim.spec.kind == "engineering_gate":
            return engineering_gate_effect(self.conn, self.store, claim, self.code_source,
                                           clock=self.clock)
        if claim.spec.kind == "publication":
            return publication_effect(self.conn, self.store, claim, self.root, self.code_source,
                                      clock=self.clock, keepalive=keepalive)
        if claim.spec.kind == "backup":
            return backup_effect(self.conn, self.store, claim, self.root, clock=self.clock,
                                 keepalive=keepalive)
        if claim.spec.kind == "snapshot_import":
            return snapshot_import_effect(self.conn, self.store, claim, refs, clock=self.clock,
                                          keepalive=keepalive)
        if claim.spec.kind == "legacy_rebuild_candidate":
            return legacy_rebuild_candidate_effect(self.conn, self.store, claim, refs, clock=self.clock)
        return None, ()

    def close(self):
        for running in self.running.values():
            signal_owned(running.identities, self.boot, hard=True)
            running.process.wait(timeout=5)
            os.close(running.result_fd)
        self.running.clear()
        self.lock.release()


def serve(service, *, once=False):
    service.start()
    try:
        while True:
            active = service.tick()
            if once and not active:
                break
            time.sleep(0.1 if once else 1)
    finally:
        service.close()


def _no_keepalive():
    return None


def _report_stranded(claim, problem):
    """One redacted stderr line: stable fields only, never ``details`` (§5.2)."""
    print(json.dumps({"event": "attempt_left_for_recovery", "job_id": claim.job_id,
                      "attempt_id": claim.attempt_id,
                      "problem": {key: to_document(problem)[key]
                                  for key in ("code", "category", "retryable", "message")}}),
          file=sys.stderr, flush=True)


def _refuse_output_name_collisions(refs, extra_refs):
    """A worker output and a coordinator ``extra_refs`` artifact sharing one
    name would both try to claim the same ``(attempt_id, name)`` row in
    ``attempt_outputs`` (its primary key) — refuse cleanly here, before the
    commit transaction, rather than let the second INSERT surface a raw
    ``sqlite3.IntegrityError`` out of ``commit_attempt``.
    """
    names = [name for name, _ in refs] + [name for name, _ in extra_refs]
    if len(names) == len(set(names)):
        return
    duplicates = sorted({name for name in names if names.count(name) > 1})
    raise OpsError(make_problem("VALIDATION_FAILED", "attempt output names collide",
                                details={"duplicates": duplicates}))


def _named_ref(refs, name):
    for candidate_name, ref in refs:
        if candidate_name == name:
            return ref
    raise OpsError(make_problem("VALIDATION_FAILED", "required effect artifact is missing"))


_DECISION_EVIDENCE_BINDINGS = {"score": "score.json", "finality": "finality.json",
                               "replay": "replay.json", "coverage": "finality_coverage.json"}


def _verify_decision_evidence(conn, store, claim, refs, keepalive=_no_keepalive):
    """Re-derive the decision plan/evidence pair from recorded bindings and
    require byte equality with what the worker published (P2-5/B1c).

    Reads only ``attempt_input_bindings`` (never re-queries a ``job_``
    parent's current, possibly-since-changed output) — the same durable-
    record discipline ``decision_commit._resolved_evidence_bindings`` uses.
    Comparing ``content_hash`` values is equivalent to comparing bytes: both
    sides route through the SAME :func:`artifact_reference` identity
    function, so identical bytes always hash identically and differing bytes
    (astronomically) never collide.
    """
    recorded = recorded_bindings(conn, claim.attempt_id)
    missing = [key for key, name in _DECISION_EVIDENCE_BINDINGS.items() if name not in recorded]
    if missing:
        raise OpsError(make_problem("VALIDATION_FAILED", "decision evidence inputs are unavailable",
                                    details={"missing_bindings": missing}))
    docs = {}
    for key, name in _DECISION_EVIDENCE_BINDINGS.items():
        keepalive()
        ref = artifact(conn, store, recorded[name].artifact_id)
        docs[key] = json.loads(store.read_verified(ref))
    keepalive()
    params = claim.spec.parameters
    plan_bytes, evidence_bytes = derive(
        docs["score"], recorded["score.json"], docs["finality"], recorded["finality.json"],
        docs["replay"], docs["coverage"], requested_session=params["session"],
        deployment=params["deployment"], decision_clock=params["decision_clock"],
        # P2-C04: re-derive with the SAME effect scope the worker's own job
        # parameters carry, so the coordinator's byte-for-byte check still
        # agrees on a subset run (never a bare "shadow" default here).
        scope=params.get("effect_scope") or "shadow")
    computed_plan = artifact_reference(plan_bytes, "decision_plan.v1.0")
    computed_evidence = artifact_reference(evidence_bytes, "decision_evidence.v1.0")
    worker_plan = _named_ref(refs, "decision_plan")
    worker_evidence = _named_ref(refs, "decision_evidence")
    if (computed_plan.content_hash != worker_plan.content_hash
            or computed_evidence.content_hash != worker_evidence.content_hash):
        raise OpsError(make_problem("VALIDATION_FAILED",
                                    "decision evidence disagrees with coordinator re-derivation"))
