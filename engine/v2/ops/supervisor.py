"""Small polling coordinator; computation lives in bounded fresh subprocesses."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from engine.v2.contracts import CheckpointCandidate, OutputCandidate, ProcessIdentity, ProgressEvent
from engine.v2.foundation import ArtifactStore, content_hash, format_timestamp
from engine.v2.ops import executor
from engine.v2.ops.catalog import load_json
from engine.v2.ops.checkpoints import (
    artifact,
    cache_identity,
    commit_checkpoint,
    register_artifact,
    reuse,
)
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.errors import OpsError, make_problem
from engine.v2.ops.executor_watchdog import observe, signal_owned
from engine.v2.ops.fingerprints import (
    environment_identity,
    snapshot_code,
    worker_source_manifest,
)
from engine.v2.ops.legacy_adapter import copy_read_set
from engine.v2.ops.lifecycle import (
    Outcome,
    commit_attempt,
    complete_cancel,
    heartbeat,
    record_measurement,
    record_progress,
)
from engine.v2.ops.recovery import (
    SupervisorLock,
    begin_epoch,
    expire_leases,
    fence_foreign_epochs,
    read_boot_id,
    reconcile_attempt,
)
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import validate_result
from engine.v2.ops.store_barrier import (
    confirm_read_set,
    pin_read_set,
    read_set_complete,
    verified_write_in,
)


class Service:
    def __init__(self, conn, root, registry, policy, *, clock, code_source, store_root=None):
        self.conn, self.root, self.registry, self.policy = conn, Path(root), registry, policy
        self.clock, self.code_source = clock, Path(code_source)
        self.store_root = Path(store_root) if store_root is not None else self.code_source
        self.store = ArtifactStore(self.root)
        self.running = {}
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
            members = self.conn.execute("SELECT identity_json FROM process_members WHERE attempt_id = ?",
                                        (row["attempt_id"],)).fetchall()
            identities = tuple(load_json(ProcessIdentity, member[0]) for member in members)
            if not identities and row["process_json"]:
                identities = (load_json(ProcessIdentity, row["process_json"]),)
            if not identities and row["host_boot_id"] == self.boot:
                state = "quarantined"
            else:
                known, alive, _ = observe(identities, self.boot)
                signal_owned(alive, self.boot, hard=True)
                uncertain = (row["host_boot_id"] == self.boot and not alive
                             and len(identities) <= 1)
                state = "quarantined" if alive or uncertain else "verified_dead"
                executor.persist_members(self.conn, row["attempt_id"], known)
            reconcile_attempt(self.conn, row["attempt_id"], process_state=state, clock=self.clock)

    def tick(self):
        self._clock_check()
        expire_leases(self.conn, clock=self.clock)
        self.reconcile()
        for attempt_id, running in list(self.running.items()):
            if self._poll(running):
                os.close(running.result_fd)
                del self.running[attempt_id]
        claim = claim_next(self.conn, policy=self.policy, sample=sample_capacity(self.root, clock=self.clock),
                           supervisor=self.identity, clock=self.clock, registry=self.registry)
        if claim:
            self._launch(claim)
        return bool(self.running or claim)

    def _clock_check(self):
        wall, mono = self.clock.now(), self.clock.monotonic()
        jump = abs((wall - self.last_wall).total_seconds() - (mono - self.last_mono))
        if jump > 60:
            fence_foreign_epochs(self.conn, epoch_id="clock_jump", clock=self.clock)
        self.last_wall, self.last_mono = wall, mono

    def _launch(self, claim):
        try:
            manifest = worker_source_manifest(self.code_source)
            if claim.spec.implementation_ref != content_hash(manifest):
                raise OpsError(make_problem("INPUT_CHANGED", "planned worker implementation changed"))
            if claim.spec.environment_ref != content_hash(
                    environment_identity(claim.resources.thread_count)):
                raise OpsError(make_problem("INPUT_CHANGED", "planned worker environment changed"))
            legacy_manifest = self._pin_read_set(claim)
            if legacy_manifest is not None:
                self._populate_legacy_staging(claim, legacy_manifest)
            if self._cache_allowed(claim) and claim.spec.kind in self.registry.names() \
                    and claim.spec.kind not in {
                    "legacy_decisions", "legacy_render", "legacy_selfcheck"}:
                cached = self._reuse_staged_checkpoint(claim)
                if cached:
                    return
            code = self.root / "code" / content_hash(manifest).split(":")[1]
            snapshot_code(self.code_source, code, manifest)
            running = executor.launch(self.conn, claim, self.registry.get(claim.spec.kind), self.store,
                                      code, clock=self.clock, boot_id=self.boot)
            self.running[claim.attempt_id] = running
        except Exception as exc:
            problem = exc.problem if isinstance(exc, OpsError) else make_problem(
                "LAUNCH_FAILED", "trusted worker launch failed")
            commit_attempt(self.conn, claim.attempt_id, claim.fence,
                           Outcome(False, "verified_dead", failure=problem), clock=self.clock)

    def _store_domains(self, claim):
        if claim.spec.kind not in self.registry.names():
            return ()
        return self.registry.get(claim.spec.kind).store_domains

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

    def _reuse_staged_checkpoint(self, claim):
        schema = "receipt.v1.0" if claim.spec.kind == "artifact_check" else "legacy_action.v1.0"
        cache_key = cache_identity(
            kind=claim.spec.kind,
            inputs=content_hash(list(claim.spec.input_refs)),
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
                                      clock=self.clock, lease_seconds=120):
            running.failure = "CANCELLED" if cancelled else "LEASE_LOST"
            executor.stop(running, boot_id=self.boot, clock=self.clock)
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

    def _progress(self, running, status):
        sequence = self.conn.execute("SELECT COALESCE(MAX(sequence),-1)+1 FROM progress_events "
                                     "WHERE attempt_id = ?", (running.claim.attempt_id,)).fetchone()[0]
        event = ProgressEvent(
            job_id=running.claim.job_id, attempt_id=running.claim.attempt_id,
            stage_id=running.claim.spec.kind, sequence=sequence,
            recorded_at=format_timestamp(self.clock.now()), kind="heartbeat",
            elapsed_seconds=self.clock.monotonic() - running.started, message="worker observed",
            memory_current_bytes=status["memory"], memory_peak_bytes=running.peak)
        record_progress(self.conn, event)

    def _finish(self, running, status):
        claim = running.claim
        code = running.failure
        if status["exit_code"] != 0:
            code = code or ("UNKNOWN_KILL" if status["exit_code"] < 0 else "WORKER_FAILED")
        try:
            if code:
                raise OpsError(make_problem(code, "worker did not complete its contract"))
            result = json.loads(running.data)
            outputs = validate_result(claim, result)
            confirm_read_set(self.conn, claim.attempt_id, self.store_root)
            running.peak = max(running.peak, int(result.get("self_peak_bytes", 0)))
            record_measurement(self.conn, claim.attempt_id, current_bytes=0,
                               peak_bytes=running.peak, clock=self.clock)
            if self._cache_allowed(claim) and claim.spec.kind in self.registry.names() \
                    and claim.spec.kind not in {
                    "legacy_decisions", "legacy_render", "legacy_selfcheck"}:
                schema = outputs[0]["schema"]
                candidate = CheckpointCandidate(
                    shard_key="default",
                    cache_key=cache_identity(
                        kind=claim.spec.kind,
                        inputs=content_hash(list(claim.spec.input_refs)),
                        implementation=claim.spec.implementation_ref,
                        parameters=content_hash(claim.spec.parameters),
                        environment=claim.spec.environment_ref,
                        schema=schema, shard="default"),
                    input_hash=content_hash(list(claim.spec.input_refs)),
                    implementation_hash=claim.spec.implementation_ref,
                    parameter_hash=content_hash(claim.spec.parameters),
                    environment_hash=claim.spec.environment_ref,
                    output_schema_ref=outputs[0]["schema"],
                    outputs=tuple(OutputCandidate(name=o["name"], staged_path=o["path"],
                                                   schema_ref=o["schema"]) for o in outputs))
                checkpoint = commit_checkpoint(self.conn, self.store, claim, candidate,
                                               clock=self.clock)
                refs = [(str(index), ref) for index, ref in enumerate(checkpoint.artifact_refs)]
            else:
                refs = [(o["name"], self.store.publish_candidate(
                    claim.attempt_id, o["path"], schema_ref=o["schema"],
                    max_bytes=claim.resources.scratch_limit_bytes)) for o in outputs]
            def effects(conn):
                for name, ref in refs:
                    register_artifact(conn, ref, claim.attempt_id, self.clock)
                    conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                                 (claim.attempt_id, name, ref.artifact_id))
                for domain, mode in self._store_domains(claim):
                    if mode == "write":
                        verified_write_in(conn, claim.attempt_id, domain)
            commit_attempt(self.conn, claim.attempt_id, claim.fence, Outcome(True, "verified_dead", 0),
                           clock=self.clock, effects=effects)
        except Exception as exc:
            problem = exc.problem if isinstance(exc, OpsError) else make_problem(
                "VALIDATION_FAILED", "worker output failed validation")
            commit_attempt(self.conn, claim.attempt_id, claim.fence,
                           Outcome(False, "verified_dead", status["exit_code"], problem), clock=self.clock)

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
