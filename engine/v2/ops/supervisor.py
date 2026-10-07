"""Small polling coordinator; computation lives in bounded fresh subprocesses."""
from __future__ import annotations

import dataclasses
import json
import os
import re
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
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import (
    artifact,
    cache_identity,
    commit_checkpoint,
    register_artifact,
    reuse,
)
from engine.v2.ops.decision_commit import (
    commit_decisions_in_transaction,
    commit_supersede,
    import_settlement_candidates_in_transaction,
    validated_decision_candidate,
)
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.discovery import sample_capacity
from engine.v2.ops.effects_graph import (
    backup_effect,
    effect_scope,
    engineering_gate_effect,
    experiment_effect,
    ledger_export_effect,
    publication_effect,
    reconcile_publication_status,
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
from engine.v2.ops.legacy_adapter import copy_read_set, overlay_read_set
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
from engine.v2.ops.worker_progress import STEPS_FILENAME, read_new_records

#: Kinds whose coordinator effect does real, non-idempotent catalog/outbox/
#: filesystem work every attempt — the generic checkpoint-reuse shortcut
#: would otherwise skip that work entirely on a cache hit (decision #1).
#: ``experiment``'s durable attempt record is itself idempotent, but it must
#: still be attempted on every attempt (and after a cache hit), so it is
#: listed here too.
_COORDINATOR_EFFECT_KINDS = frozenset({
    "legacy_decisions", "legacy_settlement", "legacy_render", "legacy_selfcheck",
    "decision_evidence", "ledger_export", "engineering_gate", "publication", "backup",
    "snapshot_import", "legacy_rebuild_candidate", "legacy_materialize", "experiment",
    "decisions_supersede"})

#: Snapshot-backed kinds that write into their legacy tree (attempt-19 fix,
#: extended 2026-09-15 for ``legacy_model_evidence``): ``legacy_render``
#: stages the bound model-evidence artifact and ledger generation at their
#: legacy paths (``render_inputs.stage_model_evidence``/
#: ``stage_ledger_generation``); ``legacy_model_evidence``'s own
#: ``build_model_evidence()`` writes ``data/features/model_evidence.json`` at
#: ITS legacy path directly (``engine/dashboard/model_evidence.py``'s own
#: ``path.write_text(...)`` at the end of that function -- legacy parity, not
#: something ``_action_model_evidence`` controls). Neither can mount the
#: shared, read-only materialization root directly like ``legacy_score``/
#: ``legacy_decision_replay``/``legacy_selfcheck`` do. Both get a private
#: writable overlay instead (``legacy_adapter.overlay_read_set``): a fresh
#: ``staging/legacy`` symlinked file-for-file to the SAME verified
#: materialization, never the live tree.
_OVERLAY_KINDS = frozenset({"legacy_render", "legacy_model_evidence"})

#: The attempt lease every heartbeat, resume renewal and keepalive extends to.
LEASE_SECONDS = 120

#: A heartbeat ``progress_events`` row and memory sample are written at most
#: once per this many seconds per attempt, plus on every observed state change
#: (stopping, exited). Lease renewal still runs on every poll (D).
HEARTBEAT_EVENT_SECONDS = 10.0

#: A commit refused with one of these can never succeed under this fence:
#: nothing more may be written under it, not even the attempt's own failure.
_FENCE_LOST_CODES = frozenset({"LEASE_LOST", "CANCELLED"})

#: A pre-fix catalog named checkpoint outputs by their enumeration index
#: ('0', '1', ...) instead of the worker's declared name. Such a name is
#: purely positional and must never be reused or propagated.
_POSITIONAL_OUTPUT_NAME = re.compile(r"^\d+$")

#: Default-message text for an executor-level failure with no dedicated
#: problem builder below (``_failure_problem``) -- a nicer stand-in for the
#: previous blanket "worker did not complete its contract" wherever the cause
#: is already named by the failure code itself.
_FAILURE_MESSAGES = {
    "LEASE_LOST": "the attempt's lease expired before it reported back",
    "CANCELLED": "the attempt was cancelled",
    "UNKNOWN_KILL": "the worker process ended by signal, outside the watchdog's own kill",
    "VALIDATION_FAILED": "the worker's result exceeded the result-pipe size limit",
}


@dataclasses.dataclass
class _StepState:
    """Per-attempt bookkeeping the supervisor keeps between polls to turn a
    worker's step file into heartbeat/step progress rows. Never persisted --
    rebuilt as ``steps.ndjson`` is re-read, and dropped once the attempt ends
    (``Service._finish``)."""

    offset: int = 0
    current_step: str | None = None
    #: max memory observed (any tick) since the step now open last started.
    step_peaks: dict = dataclasses.field(default_factory=dict)
    step_started_at: dict = dataclasses.field(default_factory=dict)
    #: max memory observed (any tick) since the last heartbeat row was written.
    interval_peak: int = 0
    interval_peak_step: str | None = None


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
        #: monotonic time of the last best-effort pass renewing every OTHER
        #: running attempt's lease (issue #106); None until the first pass.
        self._other_leases_renewed_at = None
        self.launches = {}
        #: attempt_id -> (monotonic time, observed state) of its last heartbeat row.
        self.observed = {}
        #: attempt_id -> _StepState; every running attempt's step-file offset,
        #: open step and interval memory peak.
        self._progress_state = {}
        self.boot = read_boot_id()
        self.lock = SupervisorLock(self.root / "supervisor.lock")
        self.identity = None
        self.last_wall = clock.now()
        self.last_mono = clock.monotonic()
        #: (code, message) of the last publication-status-reconciliation
        #: problem actually printed, so a persisting problem prints once
        #: rather than on every ~1s tick (2026-09-14 second review fix).
        self._last_publication_status_problem = None
        #: same dedup, for _reconcile_computed_moves_refresh (S4C Part 4,
        #: revised after Opus BLOCK(3)).
        self._last_computed_moves_problem = None
        #: {"identity": _computed_moves_identity() tuple, "attempts": int,
        #: "not_before": monotonic float} or None -- the backoff memo for
        #: the SAME method (Opus re-gate finding: an unthrottled rebuild
        #: attempt repeated every tick, all day, whenever it did not end in
        #: a submitted job).
        self._computed_moves_memo = None
        #: Cutover PR-7a: one-slot, root-keyed memo of the last release_id
        #: this sidecar has fully verified (success or failure) -- see
        #: ARCHITECTURE.md "Inputs"/"Cutover PR-7a's input sourcing". Checked
        #: EVERY tick, unconditionally, ahead of and independent of
        #: self._native_score_batch_memo's own backoff below. Shape:
        #: {"root": str, "release_id": str, "ok": bool} or None.
        self._native_release_memo = None
        #: dedup for a persisting release-unavailable problem, same pattern
        #: as self._last_computed_moves_problem.
        self._last_native_release_problem = None
        #: Cutover PR-7a: the SAME backoff-schedule memo shape as
        #: self._computed_moves_memo (reusing
        #: _COMPUTED_MOVES_MAX_ATTEMPTS/_COMPUTED_MOVES_BACKOFF_SECONDS,
        #: never a separate schedule), guarding the build step in
        #: submit_native_score_batch_shadow_if_ready. Holds ONLY a real
        #: identity's build-attempt memo -- {"identity": tuple, "attempts":
        #: int, "not_before": float} -- never an identity-lookup failure
        #: (CodeRabbit round 7, real finding: an earlier draft's lookup
        #: failure handler wrote into this SAME slot, silently discarding a
        #: real identity's already-accumulated attempts on every transient
        #: lookup error). See self._native_score_batch_lookup_memo below
        #: for that separate concern.
        self._native_score_batch_memo = None
        #: Cutover PR-7a: a SEPARATE one-slot backoff memo -- {"attempts":
        #: int, "not_before": float} or None -- for a failure INSIDE the
        #: identity lookup itself (a locked database, a malformed
        #: spec_json), never conflated with self._native_score_batch_memo's
        #: real-identity attempt count.
        self._native_score_batch_lookup_memo = None
        self._last_native_score_batch_problem = None
        #: Real-identity attempt memo only, never a lookup failure (CodeRabbit
        #: round 2, same defect class as the _native_score_batch memo pair).
        self._native_parity_memo = None
        #: Lookup-failure backoff memo; see _native_parity_identity_or_none's docstring.
        self._native_parity_lookup_memo = None
        #: One slot keyed by native_score_batch_job_id: a CONFIRMED records/
        #: refusals schema_version mismatch, never a read/decode failure (see
        #: submit_native_parity_if_ready's docstring). Checked BEFORE all
        #: self._native_parity_memo's own machinery: a tick carrying this SAME
        #: job id returns at zero cost; a DIFFERENT id clears it implicitly.
        self._native_parity_schema_mismatch_job_id = None
        self._last_native_parity_problem = None

    def start(self):
        if not self.lock.acquire():
            raise OpsError(make_problem("RESOURCE_UNAVAILABLE", "another supervisor holds this catalog"))
        try:
            epoch = begin_epoch(self.conn, clock=self.clock, boot_id=self.boot, pid=os.getpid())
            self.identity = Supervisor(epoch, self.boot)
            fence_foreign_epochs(self.conn, epoch_id=epoch, clock=self.clock)
            self.reconcile()
        except BaseException:
            self.lock.release()
            raise

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
        self._reconcile_publication_status()
        self._reconcile_computed_moves_refresh()
        self._reconcile_native_score_batch_shadow()
        self._reconcile_native_parity()
        return bool(self.running or claim)

    def _reconcile_publication_status(self):
        """2026-09-14 review fix, item 1: the one place, across the whole
        service loop, that observes every job's state after every possible
        transition this tick (``reconcile()``'s cancel-completions,
        ``_poll``/``_finish``'s failures, ``_launch``'s own launch
        failures) -- so it is also the one place that can reliably notice a
        ``publication`` job newly ``blocked``/``failed``/``cancelled`` and
        keep its scope's status sidecar honest, without threading
        ``store``/``root`` down into ``lifecycle.py``'s kind-agnostic
        transition functions. A failure here (a malformed row, a disk
        error) must never stop job scheduling itself -- this is a reporting
        sidecar, not the pipeline -- so it is caught and reported the same
        redacted way a stranded attempt is (``_report_stranded``), never
        left to crash the tick.

        2026-09-14 second review fix: ``tick()`` runs roughly every second,
        so a PERSISTING problem (a genuinely broken calendar, a wedged
        disk) printed one line per tick forever. Printed only when the
        (code, message) pair differs from the last one this instance
        actually printed -- reset to ``None`` on a clean pass, so a problem
        that resolves and later recurs (even with the identical code/
        message) still prints again.
        """
        try:
            reconcile_publication_status(self.conn, self.store, self.root, clock=self.clock)
            self._last_publication_status_problem = None
        except Exception as exc:
            problem = exc.problem if isinstance(exc, OpsError) else make_problem(
                "VALIDATION_FAILED", "publication status reconciliation failed")
            problem_key = (problem.code, problem.message)
            if problem_key == self._last_publication_status_problem:
                return
            self._last_publication_status_problem = problem_key
            print(json.dumps({"event": "publication_status_reconcile_failed",
                              "problem": {field: to_document(problem)[field]
                                          for field in ("code", "category", "retryable", "message")}}))

    #: Max rebuild attempts per (session, head) identity before giving up
    #: on it entirely (never retried again until a new head or session
    #: changes the identity). Each attempt that reaches
    #: ``submit_computed_moves_refresh_if_ready`` and does not end in a
    #: submitted job re-runs a full pandas scan
    #: (``computed_moves_store.target_tickers_from_snapshot``) and, if
    #: admission rejects the built request, re-hashes the source tree
    #: (``fingerprints.worker_source_manifest``) -- unthrottled, that
    #: repeats every ~1s tick, all day (Opus re-gate finding).
    _COMPUTED_MOVES_MAX_ATTEMPTS = 5
    #: Backoff (seconds) after attempt N fails -- 30s, 2m, 10m, 30m, 1h,
    #: then no more attempts for this identity. Picked to bound the worst
    #: case (an all-day session with nothing to submit) to a handful of
    #: scans rather than tens of thousands of tick-driven ones, while still
    #: noticing a target becoming scoreable within the hour.
    _COMPUTED_MOVES_BACKOFF_SECONDS = (30.0, 120.0, 600.0, 1800.0, 3600.0)

    def _computed_moves_backoff(self, memo, now):
        """Record one spent attempt against ``memo`` and schedule the next
        one via ``_COMPUTED_MOVES_BACKOFF_SECONDS``, indexed by attempt
        number and clamped to the schedule's last entry past
        ``_COMPUTED_MOVES_MAX_ATTEMPTS`` (split out of
        ``_reconcile_computed_moves_refresh`` to stay under the function-line
        budget; same two call sites -- a raised exception and a ``None``
        receipt back -- share this exactly)."""
        memo["attempts"] += 1
        memo["not_before"] = now + self._COMPUTED_MOVES_BACKOFF_SECONDS[
            min(memo["attempts"] - 1, len(self._COMPUTED_MOVES_BACKOFF_SECONDS) - 1)]
        self._computed_moves_memo = memo

    def _report_computed_moves_problem(self, exc):
        """Redacted, deduped report for a ``_reconcile_computed_moves_refresh``
        exception -- the identical dedup-by-(code, message) pattern
        ``_reconcile_publication_status`` already uses for its own
        reporting (split out for the same function-line-budget reason as
        ``_computed_moves_backoff``)."""
        problem = exc.problem if isinstance(exc, OpsError) else make_problem(
            "VALIDATION_FAILED", "computed_moves_refresh reconciliation failed")
        problem_key = (problem.code, problem.message)
        if problem_key == self._last_computed_moves_problem:
            return
        self._last_computed_moves_problem = problem_key
        print(json.dumps({"event": "computed_moves_refresh_reconcile_failed",
                          "problem": {field: to_document(problem)[field]
                                      for field in ("code", "category", "retryable", "message")}}))

    def _report_native_release_problem(self, detail_code, message):
        """Dedup-by-(code, message) report for a release-unavailable
        outcome (MissingReleaseRoot, an unreadable pointer file,
        ModelNotReady, NoCurrentRelease, or "nothing ever promoted") --
        the identical dedup pattern _report_computed_moves_problem uses.
        Takes a plain (detail_code, message) pair rather than an exception
        because deployment.py/release_bindings.py raise plain ValueError
        subclasses, never OpsError, and NoCurrentRelease carries no .code
        attribute at all (Cutover PR-7a). The unregistered deployment-layer
        detail_code is folded into the message text of an always-registered
        VALIDATION_FAILED problem rather than passed to make_problem as its
        code."""
        problem = make_problem("VALIDATION_FAILED", f"{detail_code}: {message}")
        problem_key = (problem.code, problem.message)
        if problem_key == self._last_native_release_problem:
            return
        self._last_native_release_problem = problem_key
        print(json.dumps({"event": "native_score_batch_release_unavailable",
                          "problem": {field: to_document(problem)[field]
                                      for field in ("code", "category", "retryable", "message")}}))

    def _native_release_root_or_none(self):
        """Cutover PR-7a: the cheap release-identity gate -- see
        ARCHITECTURE.md "Inputs"/"Cutover PR-7a's input sourcing". Runs on
        EVERY tick, unconditionally, ahead of and independent of
        self._native_score_batch_memo's own backoff -- a release-unavailable
        outcome here NEVER touches that memo (R2: it must not count as one
        of ITS spent attempts).

        Three calls, gated in two stages: deployment.production_release_root()
        (one os.environ read) and deployment.current_pointer() against this
        release root's own "deployment" subdirectory (one file stat, one
        small JSON decode -- checks/phase5_release.py's own DEPLOYMENT_DIR
        constant is the precedent for this exact literal, since
        release_bindings._DEPLOYMENT_DIR is private to that module) are both
        genuinely cheap and run every call; resolve_production_release_binding()
        (hash-verifies every model file) runs ONLY when the cheap
        current_pointer() read reports a root or release_id this sidecar
        has not already fully verified together (self._native_release_memo
        compares both fields; either one changing invalidates it).

        Returns the verified release root as a plain path string, or None
        when unavailable for any reason."""
        from engine.v2.models import deployment
        from engine.v2.scoring import release_bindings

        try:
            root = deployment.production_release_root()
        except deployment.MissingReleaseRoot as exc:
            self._report_native_release_problem("MISSING_RELEASE_ROOT", str(exc))
            return None
        try:
            pointer = deployment.current_pointer(root / "deployment")
        except (OSError, ValueError):
            self._report_native_release_problem(
                "MODEL_NOT_READY", "the deployment pointer could not be read")
            return None
        if pointer is None:
            self._report_native_release_problem(
                "NO_CURRENT_RELEASE", "no release has ever been promoted at this release root")
            return None
        memo = self._native_release_memo
        if (memo is not None and memo["root"] == str(root)
                and memo["release_id"] == pointer.release_id):
            self._last_native_release_problem = None
            return str(root) if memo["ok"] else None
        try:
            release_bindings.resolve_production_release_binding()
        except (release_bindings.ModelNotReady, release_bindings.NoCurrentRelease) as exc:
            self._native_release_memo = {"root": str(root), "release_id": pointer.release_id,
                                         "ok": False}
            self._report_native_release_problem(getattr(exc, "code", "MODEL_NOT_READY"), str(exc))
            return None
        except Exception:
            self._native_release_memo = {"root": str(root), "release_id": pointer.release_id,
                                         "ok": False}
            self._report_native_release_problem(
                "MODEL_NOT_READY", "release verification failed")
            return None
        self._native_release_memo = {"root": str(root), "release_id": pointer.release_id,
                                     "ok": True}
        self._last_native_release_problem = None
        return str(root)

    def _native_score_batch_backoff(self, memo, now):
        """Cutover PR-7a: identical schedule to _computed_moves_backoff,
        applied to self._native_score_batch_memo instead (see
        ARCHITECTURE.md's R2 account for why this reuses
        _COMPUTED_MOVES_MAX_ATTEMPTS/_COMPUTED_MOVES_BACKOFF_SECONDS rather
        than a separate schedule)."""
        memo["attempts"] += 1
        memo["not_before"] = now + self._COMPUTED_MOVES_BACKOFF_SECONDS[
            min(memo["attempts"] - 1, len(self._COMPUTED_MOVES_BACKOFF_SECONDS) - 1)]
        self._native_score_batch_memo = memo

    def _report_native_score_batch_problem(self, exc):
        """Dedup-by-(code, message) report, identical pattern to
        _report_computed_moves_problem (Cutover PR-7a)."""
        problem = exc.problem if isinstance(exc, OpsError) else make_problem(
            "VALIDATION_FAILED", "native_score_batch shadow reconciliation failed")
        problem_key = (problem.code, problem.message)
        if problem_key == self._last_native_score_batch_problem:
            return
        self._last_native_score_batch_problem = problem_key
        print(json.dumps({"event": "native_score_batch_reconcile_failed",
                          "problem": {field: to_document(problem)[field]
                                      for field in ("code", "category", "retryable", "message")}}))

    def _computed_moves_identity_or_none(self, now):
        """The two CHEAP checks (plain indexed ``SELECT``s, no pandas scan)
        ``_reconcile_computed_moves_refresh`` needs every tick --
        ``nightly._computed_moves_identity`` (the session to key off and
        the current head), and whether a job already exists under that
        session's key -- split out to stay under the function-line budget.

        Both run inside a ``try`` -- CodeRabbit finding on the Opus
        re-gate: before this fix they ran BEFORE any ``try`` in the caller,
        so a raise from either (a locked database, a malformed idempotency
        key making ``_session_from_refresh_key``'s ``split`` choke) would
        escape and crash ``tick()`` itself, contradicting ARCHITECTURE.md's
        "never interrupt the tick or required dispatch" claim -- unlike
        every other failure path here, matching
        ``_reconcile_publication_status``'s own shape, whose entire body is
        inside its ``try``. Such a raise is recorded against a dedicated
        ``identity: None`` memo entry (there is no real identity to key it
        by yet) via the same ``_computed_moves_backoff`` schedule, but
        WITHOUT the ``_COMPUTED_MOVES_MAX_ATTEMPTS`` cap the caller applies
        once a real identity is known: a transient failure here (the lock
        clearing) has no "new identity" signal of its own to reset on, so
        giving up permanently after 5 tries would silently and
        irrecoverably stop this stage for the rest of the process's life;
        instead it keeps retrying forever, just at the schedule's slowest
        (1h) cadence past the 5th attempt.

        Returns the real identity tuple when the caller should proceed, or
        ``None`` when it should return immediately -- covering a caught
        exception (reported and backed off here), "no identity yet"
        (nothing succeeded), and "already submitted" (a job exists under
        that key); the last two clear ``self._computed_moves_memo``
        themselves, since there is nothing to back off from once a job
        exists."""
        from engine.v2.ops.nightly import _computed_moves_identity, _computed_moves_refresh_key
        from engine.v2.ops.submission import job_id_for

        memo = self._computed_moves_memo
        if memo is not None and memo["identity"] is None and now < memo["not_before"]:
            return None
        try:
            identity = _computed_moves_identity(self.conn)
            exists = identity is not None and self.conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?",
                (job_id_for("shadow", _computed_moves_refresh_key(identity[0])),)).fetchone() is not None
        except Exception as exc:
            error_memo = memo if (memo is not None and memo["identity"] is None) else {
                "identity": None, "attempts": 0, "not_before": 0.0}
            self._computed_moves_backoff(error_memo, now)
            self._report_computed_moves_problem(exc)
            return None
        self._last_computed_moves_problem = None
        if identity is None or exists:
            self._computed_moves_memo = None
            return None
        return identity

    def _reconcile_computed_moves_refresh(self):
        """S4C Part 4 (revised after Opus BLOCK(3) on ``dc7f9360``, then
        again after the Opus re-gate on ``e6be41a``, then again after the
        Opus re-gate on ``f531b07``): the ONLY place ``computed_moves_refresh``
        is submitted -- never bundled into ``build_legacy_job_requests``'s
        single ``submit_graph`` call, so it can never race or fail the
        required "refresh" stage (``_check_head_expectation`` rejects
        whichever native job commits its pinned head second).

        ``_computed_moves_identity_or_none`` runs the two CHEAP checks (see
        its own docstring for their failure semantics) and returns either a
        real identity to proceed with, or ``None`` to return immediately.

        Only past that does this method reach the EXPENSIVE path --
        ``nightly.submit_computed_moves_refresh_if_ready``, which (when
        there is no job yet) runs the full
        ``computed_moves_store.target_tickers_from_snapshot`` scan and
        builds+submits a request -- and even then only if
        ``self._computed_moves_memo`` (keyed by THIS identity; a new head
        or a new session's identity always resets ``attempts`` to 0 and
        clears the backoff) has not already spent its
        ``_COMPUTED_MOVES_MAX_ATTEMPTS`` attempts and is not still inside
        its ``not_before`` backoff window. A submitted job clears the memo
        (nothing left to retry); anything else -- ``None`` back (no
        scoreable target) or a raised exception (a resolve/target-selection
        error, admission rejecting the request) -- counts as one spent
        attempt and schedules the next one via
        ``_COMPUTED_MOVES_BACKOFF_SECONDS``. Every exception this method can
        see is caught and reported the same redacted way
        ``_reconcile_publication_status`` reports its own, never left to
        crash the tick or block dispatch of any other job. See
        ``nightly.submit_computed_moves_refresh_if_ready`` and
        ARCHITECTURE.md "Outputs"/"Failure semantics" for the full account.

        Submits under its OWN ``NamespacePolicy``, never ``self.policy``
        (Opus re-gate, confirmed real: ``cli.py``'s ``serve``/
        ``nightly_trigger.py`` construct ``Service`` with ``DEFAULT_POLICY``,
        a ``ResourcePolicy`` -- ``.profiles`` for ``claim_next``, but no
        ``.allows()``, which ``submission.submit`` needs). Every OTHER
        production submission site (``cli.py``'s ``_submit_command``/
        ``_submit_nightly``/HTTP submit/adhoc-rescore/snapshot-submit/
        decisions-supersede) already builds its own ad-hoc
        ``NamespacePolicy`` inline, never from ``Service``'s resource
        policy; this matches that, scoped to just ``"shadow"`` -- the only
        namespace this stage ever targets.
        """
        from engine.v2.ops.nightly import submit_computed_moves_refresh_if_ready
        from engine.v2.ops.snapshot_stages import _catalog_path
        from engine.v2.ops.submission import NamespacePolicy

        now = self.clock.monotonic()
        identity = self._computed_moves_identity_or_none(now)
        if identity is None:
            return
        memo = self._computed_moves_memo
        if memo is None or memo["identity"] != identity:
            memo = {"identity": identity, "attempts": 0, "not_before": 0.0}
        if (memo["attempts"] >= self._COMPUTED_MOVES_MAX_ATTEMPTS
                or now < memo["not_before"]):
            self._computed_moves_memo = memo
            return
        policy = NamespacePolicy({"operator": frozenset({"shadow"})})
        try:
            receipt = submit_computed_moves_refresh_if_ready(
                self.conn, self.registry, policy, self.store,
                catalog_path=_catalog_path(self.conn), objects_root=str(self.root),
                code_source=self.code_source, clock=self.clock)
        except Exception as exc:
            self._computed_moves_backoff(memo, now)
            self._report_computed_moves_problem(exc)
            return
        self._last_computed_moves_problem = None
        if receipt is not None:
            self._computed_moves_memo = None
        else:
            self._computed_moves_backoff(memo, now)

    def _native_score_batch_identity_or_none(self, now):
        """The two CHEAP checks (plain indexed SELECTs, no pandas scan)
        _reconcile_native_score_batch_shadow needs every tick --
        nightly._native_score_batch_identity (the specific succeeded
        "score" job to key off, and whether it pinned a snapshot) and
        whether a job already exists under that identity's key -- split
        out to mirror _computed_moves_identity_or_none's own shape and
        failure semantics exactly (CodeRabbit round 6, real finding: an
        earlier draft ran both checks with no not_before gate at all, so a
        persistently-raising lookup -- a locked database, a malformed
        spec_json -- ran on EVERY tick forever, and its failure was folded
        into whatever real identity's build-attempt memo happened to be
        cached, silently spending that identity's own attempt budget on an
        unrelated lookup failure).

        Returns the real identity tuple when the caller should proceed, or
        None when it should return immediately -- covering a throttled
        prior lookup failure (still inside its own not_before), a caught
        exception (reported and backed off here, in a SEPARATE memo slot
        (self._native_score_batch_lookup_memo), never touching
        self._native_score_batch_memo's own real-identity attempt
        count), "no identity
        yet", and "already submitted" (a job exists under that key); the
        last two clear self._native_score_batch_memo themselves. A
        successful lookup -- whatever it returns -- always resets
        self._native_score_batch_lookup_memo to None, so a later failure
        starts a fresh backoff sequence rather than resuming an old
        one."""
        from engine.v2.ops.nightly import _native_score_batch_identity, _native_score_batch_key
        from engine.v2.ops.submission import job_id_for

        lookup_memo = self._native_score_batch_lookup_memo
        if lookup_memo is not None and now < lookup_memo["not_before"]:
            return None
        try:
            identity = _native_score_batch_identity(self.conn)
            exists = identity is not None and self.conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?",
                (job_id_for("shadow", _native_score_batch_key(identity[0], identity[1])),)
            ).fetchone() is not None
        except Exception as exc:
            lookup_memo = lookup_memo or {"attempts": 0, "not_before": 0.0}
            lookup_memo["attempts"] += 1
            lookup_memo["not_before"] = now + self._COMPUTED_MOVES_BACKOFF_SECONDS[
                min(lookup_memo["attempts"] - 1, len(self._COMPUTED_MOVES_BACKOFF_SECONDS) - 1)]
            self._native_score_batch_lookup_memo = lookup_memo
            self._report_native_score_batch_problem(exc)
            return None
        self._last_native_score_batch_problem = None
        self._native_score_batch_lookup_memo = None
        if identity is None or exists:
            self._native_score_batch_memo = None
            return None
        return identity

    def _reconcile_native_score_batch_shadow(self):
        """Cutover PR-7a slice 5: the ONLY place native_score_batch is ever
        submitted -- called every tick(), right alongside
        _reconcile_computed_moves_refresh, never through build_legacy_job_requests.
        See nightly.submit_native_score_batch_shadow_if_ready and ARCHITECTURE.md
        "Outputs"/"Failure semantics" for the full account.

        R2 (two independent memos): _native_release_root_or_none runs FIRST, every
        tick, and a release-unavailable outcome NEVER touches
        self._native_score_batch_memo's own attempt count;
        _native_score_batch_identity_or_none runs next into its OWN
        self._native_score_batch_lookup_memo, never conflated with a real
        identity's build-attempt count. Only past both does this method reach the
        SAME bounded backoff schedule _reconcile_computed_moves_refresh uses.
        Past the memo gate a snapshot-pinned identity stages its COMPLETE
        events.json/producer_refusals.json pair -- produce, publish, register in
        ONE catalog transaction -- before the builder is reached, so no failure
        path can leave a job referencing a one-sided stage; every exception is
        caught and reported the same redacted way _reconcile_publication_status
        reports its own, never crashing the tick."""
        from engine.v2.data.repository import Repository
        from engine.v2.foundation import canonical_json
        from engine.v2.ops.nightly import submit_native_score_batch_shadow_if_ready
        from engine.v2.ops.nightly_raw_row_producer import build_native_score_batch_events
        from engine.v2.ops.snapshot_stages import _catalog_path
        from engine.v2.ops.submission import NamespacePolicy

        now = self.clock.monotonic()
        release_root = self._native_release_root_or_none()
        if release_root is None:
            return
        identity = self._native_score_batch_identity_or_none(now)
        if identity is None:
            return
        session, _scope_hash, producer_parameters = identity
        memo = self._native_score_batch_memo
        if memo is None or memo["identity"] != identity:
            memo = {"identity": identity, "attempts": 0, "not_before": 0.0}
        if (memo["attempts"] >= self._COMPUTED_MOVES_MAX_ATTEMPTS
                or now < memo["not_before"]):
            self._native_score_batch_memo = memo
            return
        policy = NamespacePolicy({"operator": frozenset({"shadow"})})
        try:
            events_ref = producer_refusals_ref = calendar_revision = snapshot_id = None
            if producer_parameters:
                repository = Repository(self.conn, self.store)
                snapshot = repository.resolve(producer_parameters.snapshot_generation_id)
                events, refusals = build_native_score_batch_events(repository, snapshot,
                    as_of=session, horizon_days=producer_parameters.horizon_days,
                    tickers=producer_parameters.tickers or None)
                earnings = snapshot.table_versions["earnings_events"]
                calendar_revision = earnings.dataset_version_id
                events_ref = self.store.publish_bytes(
                    canonical_json(events).encode(),
                    schema_ref="native_score_batch_events.v1.0")
                producer_refusals_ref = self.store.publish_bytes(
                    canonical_json(refusals).encode(),
                    schema_ref="native_score_batch_producer_refusals.v1.0")
                with transaction(self.conn):
                    register_artifact(self.conn, events_ref, None, self.clock)
                    register_artifact(self.conn, producer_refusals_ref, None, self.clock)
                events_ref, producer_refusals_ref = (events_ref.artifact_id,
                    producer_refusals_ref.artifact_id)
                snapshot_id = snapshot.snapshot_id
            receipt = submit_native_score_batch_shadow_if_ready(
                self.conn, self.registry, policy, self.store, release_root,
                catalog_path=_catalog_path(self.conn), objects_root=str(self.root),
                code_source=self.code_source, clock=self.clock, snapshot_id=snapshot_id,
                calendar_revision=calendar_revision, events_ref=events_ref,
                producer_refusals_ref=producer_refusals_ref)
        except Exception as exc:
            self._native_score_batch_backoff(memo, now)
            self._report_native_score_batch_problem(exc)
            return
        self._last_native_score_batch_problem = None
        if receipt is not None:
            self._native_score_batch_memo = None
        else:
            self._native_score_batch_backoff(memo, now)

    def _native_parity_backoff(self, memo, now):
        """Cutover PR-4 redo slice 2B(b): identical schedule to
        _computed_moves_backoff/_native_score_batch_backoff, applied to
        self._native_parity_memo instead -- its own slot, for the same reason
        _native_score_batch_backoff exists: _computed_moves_backoff hardcodes
        self._computed_moves_memo and would silently write this sidecar's
        attempts into the wrong memo."""
        memo["attempts"] += 1
        memo["not_before"] = now + self._COMPUTED_MOVES_BACKOFF_SECONDS[
            min(memo["attempts"] - 1, len(self._COMPUTED_MOVES_BACKOFF_SECONDS) - 1)]
        self._native_parity_memo = memo

    def _report_native_parity_problem(self, exc):
        """Dedup-by-(code, message) report, identical pattern to
        _report_computed_moves_problem (Cutover PR-4 redo slice 2B(b))."""
        problem = exc.problem if isinstance(exc, OpsError) else make_problem(
            "VALIDATION_FAILED", "native_parity reconciliation failed")
        problem_key = (problem.code, problem.message)
        if problem_key == self._last_native_parity_problem:
            return
        self._last_native_parity_problem = problem_key
        print(json.dumps({"event": "native_parity_reconcile_failed",
                          "problem": {field: to_document(problem)[field]
                                      for field in ("code", "category", "retryable", "message")}}))

    def _native_parity_identity_or_none(self, now):
        """The two CHEAP checks (plain indexed SELECTs, no artifact read)
        _reconcile_native_parity needs every tick --
        nightly._native_parity_identity (the paired succeeded
        native_score_batch/score identity) and whether a job already exists
        under that identity's key -- split out to mirror
        _computed_moves_identity_or_none's own shape and failure semantics
        exactly. See _native_score_batch_identity_or_none's own docstring for
        the shared lookup-failure rationale, argued once there and not
        re-argued here: a raise from either check is reported and backed off
        here, in a SEPARATE memo slot (self._native_parity_lookup_memo), never
        touching self._native_parity_memo's own real-identity attempt count --
        and deliberately WITHOUT the _COMPUTED_MOVES_MAX_ATTEMPTS cap the
        caller applies to a real identity, so a transient catalog lock retries
        forever at the schedule's slowest cadence instead of silently
        disabling this stage for the rest of the process's life.

        Returns the real identity tuple when the caller should proceed, or
        None when it should return immediately -- covering a throttled prior
        lookup failure (still inside its own not_before), a caught exception
        (reported and backed off here), "no identity yet", and "already
        submitted" (a job exists under that key); the last two clear
        self._native_parity_memo themselves. A successful lookup -- whatever
        it returns -- always resets self._native_parity_lookup_memo to None,
        so a later failure starts a fresh backoff sequence rather than
        resuming an old one."""
        from engine.v2.ops.nightly import _native_parity_identity, _native_parity_key
        from engine.v2.ops.submission import job_id_for

        lookup_memo = self._native_parity_lookup_memo
        if lookup_memo is not None and now < lookup_memo["not_before"]:
            return None
        try:
            identity = _native_parity_identity(self.conn)
            exists = identity is not None and self.conn.execute(
                "SELECT 1 FROM jobs WHERE job_id = ?",
                (job_id_for("shadow", _native_parity_key(identity[0], identity[1])),)
            ).fetchone() is not None
        except Exception as exc:
            lookup_memo = lookup_memo or {"attempts": 0, "not_before": 0.0}
            lookup_memo["attempts"] += 1
            lookup_memo["not_before"] = now + self._COMPUTED_MOVES_BACKOFF_SECONDS[
                min(lookup_memo["attempts"] - 1, len(self._COMPUTED_MOVES_BACKOFF_SECONDS) - 1)]
            self._native_parity_lookup_memo = lookup_memo
            self._report_native_parity_problem(exc)
            return None
        self._last_native_parity_problem = None
        self._native_parity_lookup_memo = None
        if identity is None or exists:
            self._native_parity_memo = None
            return None
        return identity

    def _native_parity_attempt_memo(self, identity, native_score_batch_job_id, now):
        """The three gates _reconcile_native_parity checks before it may spend
        an attempt, split out for the function-line budget (the same reason
        _computed_moves_backoff/_native_score_batch_backoff exist). Returns the
        memo to attempt under, or ``None`` when this tick must return without
        touching the expensive path at all:

        1. a parked schema mismatch FIRST -- an identity carrying the same
           native_score_batch_job_id as self._native_parity_schema_mismatch_job_id
           returns at strictly zero cost: no artifact read, no attempt spent,
           and self._native_parity_memo not even read (see
           _reconcile_native_parity's own docstring for why that wait state is
           permanent per job id, and submit_native_parity_if_ready's own
           docstring for what makes a mismatch CONFIRMED);
        2. a new identity resets attempts/backoff (a stale memo for a
           different job never gates this one);
        3. the SAME _COMPUTED_MOVES_MAX_ATTEMPTS cap and not_before window
           _reconcile_computed_moves_refresh applies, reused rather than
           duplicated."""
        if native_score_batch_job_id == self._native_parity_schema_mismatch_job_id:
            return None
        memo = self._native_parity_memo
        if memo is None or memo.get("identity") != identity:
            memo = {"identity": identity, "attempts": 0, "not_before": 0.0}
        if (memo["attempts"] >= self._COMPUTED_MOVES_MAX_ATTEMPTS
                or now < memo["not_before"]):
            self._native_parity_memo = memo
            return None
        return memo

    def _reconcile_native_parity(self):
        """Cutover PR-4 redo slice 2B(b): the ONLY place a ``native_parity``
        job is ever submitted -- called every tick(), right alongside
        _reconcile_computed_moves_refresh and _reconcile_native_score_batch_shadow,
        never through build_legacy_job_requests, so a broken parity submission
        can never abort a required legacy stage. See
        nightly.submit_native_parity_if_ready and ARCHITECTURE.md
        "Outputs"/"Failure semantics" for the full account.

        _native_parity_identity_or_none runs the two CHEAP checks (see its own
        docstring for their failure semantics) and returns either a real
        identity to proceed with, or None to return immediately.

        Past that, _native_parity_attempt_memo checks the ONE short-circuit
        ahead of all memo/backoff machinery: an identity whose
        native_score_batch_job_id equals
        self._native_parity_schema_mismatch_job_id returns None immediately, at
        strictly zero cost -- no artifact file read, no attempt spent,
        self._native_parity_memo not even looked at.
        submit_native_parity_if_ready's OWN docstring accounts for what makes a
        mismatch CONFIRMED (a stale records.json/refusals.json schema_version
        tag, never a read or decode error) and why that is a permanent wait
        state for that job id -- its artifacts can never change schema
        underneath it -- and a DIFFERENT job id clears it implicitly, with no
        separate reset step anywhere.

        Only past both does this method reach the EXPENSIVE path --
        submit_native_parity_if_ready -- and only if self._native_parity_memo
        (keyed by THIS identity) still has attempts/backoff room. A submitted
        job clears the memo; a None back or a retryable OpsError schedules the
        next backoff via _COMPUTED_MOVES_BACKOFF_SECONDS, EXCEPT a confirmed
        non-retryable OpsError (problem.retryable is False -- e.g. a malformed
        committed artifact that can never change), which spends the FULL
        attempt budget immediately instead of walking a schedule toward a cap
        it can never avoid. Every exception is caught and reported the same
        redacted way _reconcile_publication_status reports its own, never
        crashing the tick."""
        from engine.v2.ops.nightly import submit_native_parity_if_ready
        from engine.v2.ops.snapshot_stages import _catalog_path
        from engine.v2.ops.submission import NamespacePolicy

        now = self.clock.monotonic()
        identity = self._native_parity_identity_or_none(now)
        if identity is None:
            return
        _as_of, _scope_hash, _score_job_id, native_score_batch_job_id = identity
        memo = self._native_parity_attempt_memo(identity, native_score_batch_job_id, now)
        if memo is None:
            return
        policy = NamespacePolicy({"operator": frozenset({"shadow"})})
        try:
            receipt = submit_native_parity_if_ready(
                self.conn, self.registry, policy, self.store,
                catalog_path=_catalog_path(self.conn), objects_root=str(self.root),
                code_source=self.code_source, clock=self.clock)
        except OpsError as exc:
            if exc.problem.details.get("reason") == "schema_mismatch":
                self._native_parity_schema_mismatch_job_id = exc.problem.details.get(
                    "native_score_batch_job_id", native_score_batch_job_id)
                self._last_native_parity_problem = None
                return
            if not exc.problem.retryable:
                memo["attempts"] = self._COMPUTED_MOVES_MAX_ATTEMPTS
                self._native_parity_memo = memo
                self._report_native_parity_problem(exc)
                return
            self._native_parity_backoff(memo, now)
            self._report_native_parity_problem(exc)
            return
        except Exception as exc:
            self._native_parity_backoff(memo, now)
            self._report_native_parity_problem(exc)
            return
        self._last_native_parity_problem = None
        if receipt is not None:
            self._native_parity_memo = None
        else:
            self._native_parity_backoff(memo, now)

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
            overlay = self._stage_legacy_inputs(claim, launch)
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
                legacy_root=(None if overlay else
                            (launch.worker_legacy_root if launch else None)),
                envelope_extra=launch.envelope_extra if launch else None)
            self.launches[claim.attempt_id] = launch
            self.running[claim.attempt_id] = running
        except Exception as exc:
            problem = exc.problem if isinstance(exc, OpsError) else make_problem(
                "LAUNCH_FAILED", "trusted worker launch failed")
            # Same fence-aware recording as a failed completion (``_finish`` ->
            # ``_commit_failure``): when the lease expires while this launch's
            # own pre-work runs (manifest hashing, the code snapshot, the read
            # pin), a raw ``commit_attempt`` re-raised LEASE_LOST out of
            # ``tick()`` and crashed the supervisor loop. The attempt is handed
            # to recovery instead — reservations and the store lease held
            # until reconciliation, every lease-expiry check unchanged; only
            # WHERE the refusal is recorded changes.
            self._commit_failure(claim, {"exit_code": None}, problem)

    def _stage_legacy_inputs(self, claim, launch):
        """Populate this attempt's legacy inputs before the worker launches.
        Returns whether a private snapshot overlay root was built (attempt-19
        fix, ``_OVERLAY_KINDS``) -- the caller must not pass ``launch.
        worker_legacy_root`` to ``executor.launch`` when this is true.

        ``launch.mode == "finality_check"`` (last read-set gap fix,
        2026-09-15) still stages the ordinary barrier read set -- unlike
        ``"snapshot"``, ``legacy_finality`` never mounts the materialization
        as its own legacy root; that mode only means a verified materialization
        root was ALSO resolved, for ``_action_finality``'s own content-level
        cross-check (passed through ``launch.envelope_extra``, never here)."""
        if launch is None or launch.mode == "finality_check":
            legacy_manifest = self._pin_read_set(claim)
            if legacy_manifest is not None:
                self._populate_legacy_staging(claim, legacy_manifest)
            return False
        overlay = launch.mode == "snapshot" and claim.spec.kind in _OVERLAY_KINDS
        if overlay:
            self._build_snapshot_overlay(claim, launch)
        return overlay

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
            # External review #5 (2026-09-14): the data snapshot id alone
            # does not identify a generation -- a later reference-only
            # reimport can commit a new receipt against the same snapshot id
            # with different pinned model/reference files. The plan's own
            # ``pin_snapshot_inputs`` call stamped the EXACT receipt id it
            # resolved (the only place "latest" is allowed); a job planned
            # before this field existed has none, and must be re-planned
            # rather than silently falling back to "whatever is newest now".
            receipt_id = str(claim.spec.parameters.get("snapshot_generation_receipt_id") or "")
            if not receipt_id:
                raise OpsError(make_problem(
                    "INPUT_CHANGED",
                    "snapshot-mode job has no pinned generation receipt (planned before this "
                    "fix) -- re-plan the job",
                    details={"reason": "generation_not_pinned"}))
            refuse_generation_mismatch(self.conn, self.store, receipt_id=receipt_id,
                                       barrier_manifest=manifest)
        pin_read_set(self.conn, claim.attempt_id, manifest, self.store_root,
                     keepalive=self._claim_keepalive(claim))
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
        copy_read_set(self.store_root, staging / "legacy", [ref["path"] for ref in refs],
                      keepalive=self._claim_keepalive(claim))

    def _build_snapshot_overlay(self, claim, launch):
        """Attempt-19 fix: an ``_OVERLAY_KINDS`` attempt's private legacy
        tree is symlinked from the SAME verified materialization root
        ``legacy_score`` used (``launch.root``), never a live-tree copy.
        Runs before ``executor.launch`` so ``staging/legacy`` exists (and is
        fully populated) before the worker starts; idempotent against a
        relaunch of the same attempt, since ``staging_dir`` never changes
        for a given ``attempt_id`` and a leftover overlay from an earlier,
        interrupted launch is reused rather than rebuilt.
        """
        staging = self.store.staging_dir(claim.attempt_id)
        destination = staging / "legacy"
        if not destination.exists():
            overlay_read_set(launch.root, destination)

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
        # Join by artifact identity, not by sorted-index position: the
        # producer's attempt_outputs rows are not guaranteed to sort into
        # the same order the worker declared/published them in (B: any kind
        # whose output order isn't alphabetical would otherwise have names
        # silently swapped onto the wrong artifacts). A producer whose own
        # names are purely positional (pre-fix catalogs, e.g. '0'/'1') or
        # whose output set doesn't exactly match this receipt is refused
        # outright rather than reused, so positional names never propagate.
        rows = self.conn.execute(
            "SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id = ?",
            (receipt.producer_attempt_id,)).fetchall()
        by_artifact = {artifact_id: name for name, artifact_id in rows}
        receipt_ids = [ref.artifact_id for ref in receipt.artifact_refs]
        if (len(by_artifact) != len(rows)
                or set(by_artifact) != set(receipt_ids)
                or any(_POSITIONAL_OUTPUT_NAME.match(name) for name in by_artifact.values())):
            return False
        def effects(conn):
            for ref in receipt.artifact_refs:
                conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                             (claim.attempt_id, by_artifact[ref.artifact_id], ref.artifact_id))
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
        self._track_steps(running, status)
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

    def _track_steps(self, running, status):
        """Every tick (~1s, matching the watchdog's own sampling): pull new
        step boundaries off the worker's private ``steps.ndjson`` and fold
        this tick's memory sample into whichever step is currently open, so a
        ramp under the 10s heartbeat gate still lands on the right step."""
        attempt_id = running.claim.attempt_id
        state = self._progress_state.setdefault(attempt_id, _StepState())
        path = self.store.staging_dir(attempt_id) / "diagnostics" / STEPS_FILENAME
        records, state.offset = read_new_records(path, state.offset)
        for record in records:
            self._apply_step_record(running, state, record)
        if state.current_step is not None:
            state.step_peaks[state.current_step] = max(
                state.step_peaks.get(state.current_step, 0), status["memory"])
        if status["memory"] >= state.interval_peak:
            state.interval_peak = status["memory"]
            state.interval_peak_step = state.current_step

    def _apply_step_record(self, running, state, record):
        name, event = record.get("step"), record.get("event")
        if not isinstance(name, str) or event not in ("start", "end"):
            return
        if event == "start":
            state.current_step = name
            state.step_peaks.setdefault(name, 0)
            state.step_started_at[name] = record.get("elapsed_seconds")
            self._emit_progress(running, kind="progress", step=name, message="step started",
                               elapsed_seconds=record.get("elapsed_seconds"),
                               memory_current_bytes=record.get("rss_bytes"))
            return
        started = state.step_started_at.pop(name, None)
        elapsed = record.get("elapsed_seconds")
        duration = elapsed - started if started is not None and elapsed is not None else None
        peak = state.step_peaks.pop(name, 0)
        if state.current_step == name:
            state.current_step = None
        self._emit_progress(running, kind="progress", step=name, message="step complete",
                           elapsed_seconds=elapsed, memory_current_bytes=record.get("rss_bytes"),
                           memory_peak_bytes=peak, duration_seconds=duration,
                           units=record.get("units"))

    def _emit_progress(self, running, *, kind, message, elapsed_seconds, step=None,
                       memory_current_bytes=None, memory_peak_bytes=None,
                       duration_seconds=None, units=None):
        claim = running.claim
        sequence = self._next_sequence(claim.attempt_id)
        event = ProgressEvent(
            job_id=claim.job_id, attempt_id=claim.attempt_id, stage_id=claim.spec.kind,
            sequence=sequence, recorded_at=format_timestamp(self.clock.now()), kind=kind,
            elapsed_seconds=float(elapsed_seconds or 0.0), message=message, step=step,
            memory_current_bytes=memory_current_bytes, memory_peak_bytes=memory_peak_bytes,
            step_duration_seconds=duration_seconds, step_units=units)
        record_progress(self.conn, event)

    def _next_sequence(self, attempt_id):
        return self.conn.execute(
            "SELECT COALESCE(MAX(sequence),-1)+1 FROM progress_events WHERE attempt_id = ?",
            (attempt_id,)).fetchone()[0]

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
        """D: one heartbeat row per ``HEARTBEAT_EVENT_SECONDS``. Its
        ``memory_peak_bytes``/``step`` are the peak *since the previous
        heartbeat* and whichever step was active when that peak was
        sampled -- the cumulative all-time peak the 10s row used to carry is
        still recorded every tick, unthrottled, on ``attempts.memory_peak_bytes``
        (``record_measurement``, called right before this in ``_poll``)."""
        state = self._progress_state.get(running.claim.attempt_id)
        peak = status["memory"] if state is None else max(state.interval_peak, status["memory"])
        peak_step = None if state is None else state.interval_peak_step
        message = ("worker exited" if status["done"] else
                   f"worker stopping: {running.failure}" if running.failure else "worker observed")
        self._emit_progress(running, kind="heartbeat", message=message,
                           elapsed_seconds=self.clock.monotonic() - running.started,
                           memory_current_bytes=status["memory"], memory_peak_bytes=peak,
                           step=peak_step)
        if state is not None:
            state.interval_peak, state.interval_peak_step = status["memory"], state.current_step

    def _renew_other_leases(self, exclude_attempt_id=None):
        """issue #106: best-effort lease renewal for every OTHER running attempt.

        One long single-attempt stretch -- a coordinator effect inside
        ``_finish``, or ``_launch``'s read-set hashing/copying before the new
        attempt is even in ``self.running`` -- used to starve every other live
        attempt of its ``_poll`` heartbeat, so the next tick's ``expire_leases``
        fenced a perfectly healthy sibling. This is the renewal point those
        stretches call. Throttled (ONE shared timer) to at most once per
        ``LEASE_SECONDS / 4`` monotonic seconds; a sibling's renewal failure is
        swallowed -- that attempt's own ``_poll``/``expire_leases`` path owns
        reporting it, it is not this pass's failure to raise. The shared timer
        advances only after an ALL-SUCCEEDED pass (round-3 fix), so a failed
        renewal is retried on the very next call rather than waiting out the
        full throttle interval while the failing sibling nears its expiry.
        """
        now = self.clock.monotonic()
        if (self._other_leases_renewed_at is not None
                and now - self._other_leases_renewed_at < LEASE_SECONDS / 4):
            return
        all_ok = True
        for attempt_id, running in list(self.running.items()):
            if attempt_id == exclude_attempt_id:
                continue
            try:
                heartbeat(self.conn, attempt_id, running.claim.fence,
                          clock=self.clock, lease_seconds=LEASE_SECONDS)
            except Exception:
                all_ok = False
        if all_ok:
            self._other_leases_renewed_at = now

    def _claim_keepalive(self, claim):
        """A callable that renews THIS attempt's own lease, then every other one.

        Used where the attempt is not (or no longer) renewed by ``_poll``: the
        pre-launch read-set pin/copy in ``_launch`` (the claimed attempt is not
        in ``self.running`` yet, so a staging stretch longer than
        ``LEASE_SECONDS`` otherwise expires its own lease before
        ``record_launch``) and ``_finish``'s coordinator work. The own renewal
        is throttled to once per ``LEASE_SECONDS / 4`` by ``Keepalive`` and
        raises ``LEASE_LOST`` when ``heartbeat`` is refused (a void fence is
        never extended); sibling renewal failures are swallowed.
        """
        own_keepalive = Keepalive(self.conn, claim.attempt_id, claim.fence, clock=self.clock,
                                  lease_seconds=LEASE_SECONDS)

        def keepalive():
            own_keepalive()
            self._renew_other_leases(exclude_attempt_id=claim.attempt_id)

        return keepalive

    def _finish(self, running, status):
        claim = running.claim
        launch = self.launches.pop(claim.attempt_id, None)
        keepalive = self._claim_keepalive(claim)

        try:
            self._commit_success(running, status, launch, keepalive)
        except Exception as exc:
            problem = (exc.problem if isinstance(exc, OpsError)
                       else self._completion_problem(claim, exc))
            self._commit_failure(claim, status, problem)
        finally:
            self._progress_state.pop(claim.attempt_id, None)

    def _completion_problem(self, claim, exc):
        """Keep coordinator failures diagnosable without exposing exception text.

        Reuse the worker catch-all redaction: class and code location only.
        In particular, a ValueError message can contain credentials or prices.
        Keep the existing nonretryable classification for completion failures.
        """
        from engine.v2.ops.worker import _generic_problem, _write_failure_details

        details = dict(_generic_problem(exc).details)
        causes, seen = [], {id(exc)}
        cause = exc
        while len(causes) < 8:
            cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
            if cause is None or id(cause) in seen:
                break
            seen.add(id(cause))
            causes.append(dict(_generic_problem(cause).details))
        if causes:
            details["causes"] = causes
        problem = make_problem(
            "VALIDATION_FAILED", f"coordinator completion raised {type(exc).__name__}")
        try:
            _write_failure_details(self.store.staging_dir(claim.attempt_id), details)
            return self._publish_failure_details(claim, problem)
        except (OSError, OpsError):
            # A failed diagnostics write must not strand an otherwise recordable
            # failure (e.g. a full staging filesystem or scratch limit).
            return problem

    def _commit_success(self, running, status, launch, keepalive):
        claim = running.claim
        code = running.failure
        # The worker's own reported failure ("failure" in its result makes it
        # exit non-zero, worker.py ``main``) is the only case with a result
        # payload worth trusting: any executor-level ``running.failure``
        # (LEASE_LOST, RESOURCE_LIMIT_EXCEEDED, an over-cap VALIDATION_FAILED)
        # or a signal kill means the data is not the worker's own report.
        worker_reported = code is None and status["exit_code"] > 0
        if status["exit_code"] != 0:
            code = code or ("UNKNOWN_KILL" if status["exit_code"] < 0 else "WORKER_FAILED")
        if code:
            problem = self._failure_problem(claim, running, code, worker_reported=worker_reported)
            raise OpsError(problem)
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

    def _failure_problem(self, claim, running, code, *, worker_reported):
        """A ``Problem`` for an attempt that will not succeed, with the best
        detail available for its class (Do §3): a resource kill gets the
        peak/limit/step/elapsed that explain it; anything the worker itself
        typed is recovered via ``_worker_typed_problem``; everything else
        falls back to a code-specific message, never the old blanket
        "worker did not complete its contract" where a better one is known.
        """
        if code == "RESOURCE_LIMIT_EXCEEDED":
            state = self._progress_state.get(claim.attempt_id)
            started = running.stop_at if running.stop_at is not None else self.clock.monotonic()
            return make_problem(
                code, "worker exceeded its reserved memory",
                details={"peak_bytes": running.peak,
                         "limit_bytes": claim.resources.reserved_memory_bytes,
                         "step": state.current_step if state is not None else None,
                         "elapsed_s": round(started - running.started, 1)})
        problem = make_problem(code, _FAILURE_MESSAGES.get(code, "worker did not complete its contract"))
        if worker_reported:
            problem = self._worker_typed_problem(claim, running) or problem
        return problem

    def _worker_typed_problem(self, claim, running):
        """Recover the ``Problem`` a worker's caught ``OpsError`` carried,
        instead of the generic retryable WORKER_FAILED (real nightly attempt
        9: a deterministic ``VALIDATION_FAILED`` was flattened and wasted a
        retry). ``None`` — the generic problem stays — for anything not a
        well-formed, registered failure code: an unknown code, a missing or
        mistyped field, or result bytes that will not even parse (including a
        result the executor's 1 MiB cap truncated). A malformed or lying
        worker must never crash the tick.

        ``details`` never reaches here — worker.py already routed it to a
        private ``staging/diagnostics/failure_details.json``, which this
        publishes as a verified artifact and references, never inlines
        (§5.2: ``details`` must stay out of ``failure_json``).
        """
        try:
            result = json.loads(bytes(running.data))
            doc = result.get("problem") if isinstance(result, dict) else None
            if not isinstance(doc, dict):
                return None
            code, message = doc.get("code"), doc.get("message")
            category, retryable = doc.get("category"), doc.get("retryable")
            if not (isinstance(code, str) and isinstance(message, str)
                    and isinstance(category, str) and isinstance(retryable, bool)):
                return None
            problem = make_problem(code, message)
        except (ValueError, TypeError, AttributeError):
            return None
        return self._publish_failure_details(claim, problem)

    def _publish_failure_details(self, claim, problem):
        details_path = (self.store.staging_dir(claim.attempt_id)
                        / "diagnostics" / "failure_details.json")
        if details_path.is_file():
            ref = self.store.publish_candidate(
                claim.attempt_id, "diagnostics/failure_details.json",
                schema_ref="failure_diagnostic.v1.0", max_bytes=claim.resources.scratch_limit_bytes)
            with transaction(self.conn):
                register_artifact(self.conn, ref, claim.attempt_id, self.clock)
            problem = dataclasses.replace(problem, diagnostic_ref=ref.artifact_id)
        return problem

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
        # commit_checkpoint publishes candidate.outputs in order (checkpoints.py
        # builds `refs` with one store.publish_candidate() per candidate.output,
        # in sequence) and only ever substitutes a PRIOR receipt after asserting
        # that receipt's artifact_refs are tuple-equal -- order included -- to
        # the freshly published refs. So checkpoint.artifact_refs is always in
        # candidate.outputs order, which is `outputs` order, on every path. The
        # count check below is the refusal if that ever stops holding, rather
        # than silently pairing outputs with the wrong artifacts.
        if len(checkpoint.artifact_refs) != len(outputs):
            raise OpsError(make_problem(
                "CHECKPOINT_INCOMPATIBLE",
                "checkpoint artifact count does not match the worker's declared outputs",
                details={"declared_outputs": len(outputs),
                         "committed_artifacts": len(checkpoint.artifact_refs)}))
        return [(o["name"], ref) for o, ref in zip(outputs, checkpoint.artifact_refs)]

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

            def _commit(conn):
                commit_decisions_in_transaction(conn, claim, candidates, context, clock=self.clock)
                # guide §5.5 item 1: surface any divergence this commit
                # recorded (a later generation's differing content for an
                # already-decided occurrence) on the job's own output, so an
                # operator sees it without reading the ledger directly. Only
                # written when at least one divergence exists, so the normal
                # (non-diverging) commit's outputs are unchanged.
                count = conn.execute(
                    "SELECT COUNT(*) FROM decision_divergences WHERE scope=? AND occurrence=?",
                    (context["scope"], context["session"])).fetchone()[0]
                if count:
                    document = {"schema_version": "decision_commit_receipt.v1.0",
                               "scope": context["scope"], "session": context["session"],
                               "divergence_count": count}
                    ref = self.store.publish_bytes(
                        json.dumps(document, sort_keys=True).encode(),
                        schema_ref="decision_commit_receipt.v1.0")
                    register_artifact(conn, ref, claim.attempt_id, self.clock)
                    conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                                (claim.attempt_id, "decision_commit_receipt", ref.artifact_id))

            return _commit, ()
        if claim.spec.kind == "decisions_supersede":
            return commit_supersede(self.conn, claim, clock=self.clock)
        if claim.spec.kind == "legacy_settlement":
            return self._settlement_effect(claim, refs)
        if claim.spec.kind == "decision_evidence":
            _verify_decision_evidence(self.conn, self.store, claim, refs, keepalive)
            return None, ()
        if claim.spec.kind == "ledger_export":
            return ledger_export_effect(self.conn, self.store, claim, self.root, self.code_source,
                                        clock=self.clock, keepalive=keepalive)
        if claim.spec.kind == "engineering_gate":
            return engineering_gate_effect(self.conn, self.store, claim, self.code_source,
                                           clock=self.clock, keepalive=keepalive)
        if claim.spec.kind == "publication":
            return publication_effect(self.conn, self.store, claim, self.root, self.code_source,
                                      clock=self.clock, store_root=self.store_root,
                                      keepalive=keepalive)
        if claim.spec.kind == "backup":
            return backup_effect(self.conn, self.store, claim, self.root, clock=self.clock,
                                 keepalive=keepalive)
        if claim.spec.kind == "snapshot_import":
            return snapshot_import_effect(self.conn, self.store, claim, refs, clock=self.clock,
                                          keepalive=keepalive)
        if claim.spec.kind == "experiment":
            return experiment_effect(self.conn, self.store, claim, refs, clock=self.clock,
                                     code_source=self.code_source, store_root=self.store_root)
        if claim.spec.kind == "legacy_rebuild_candidate":
            return legacy_rebuild_candidate_effect(self.conn, self.store, claim, refs, clock=self.clock)
        return None, ()

    def _settlement_effect(self, claim, refs):
        """``legacy_settlement``'s own commit closure, split out of
        ``_coordinator_effect`` (function-length budget). A settlement line
        that conflicts with its recorded prediction's contract is recorded
        as a divergence and dropped rather than failing the stage (guide
        §5.5 item 1 applied to settlement -- see
        ``engine.v2.ops.decision_commit.import_settlement_candidates_in_transaction``).
        """
        candidate_ref = _named_ref(refs, "legacy_settlement")
        document = json.loads(self.store.read_verified(candidate_ref))
        rows = document.get("rows") if isinstance(document, dict) else None
        if not isinstance(rows, list):
            raise OpsError(make_problem("VALIDATION_FAILED",
                                        "settlement candidate artifact has no rows"))
        session = document.get("session")
        occurrence = session or claim.spec.parameters["session"]

        def _commit(conn):
            # Surface any divergence, and which proof kind admitted each
            # resolved line, on the job's own output -- row_ids and counts
            # only, never payloads -- so an operator sees both without
            # reading the ledger directly. ``proof_counts`` distinguishes a
            # grandfathered resolved line admitted on v2's own exit-date
            # proof ("v2_finality_session") from one admitted on legacy's
            # own ``exit_finality`` ("legacy_exit_finality") -- see
            # ``decision_commit._validate_settlement_state``. ``skip_counts``
            # (task brief rule 5) surfaces the same-session rerun dedupe:
            # ``already_resolved`` (a row_id with an already-committed
            # ``resolved`` outcome, any session -- rule 1) and
            # ``already_observed_this_session`` (rule 2). Only written when
            # there is something to report (a divergence, a resolved
            # admission, or a dedupe skip), so a purely-unresolvable
            # first-time settlement's outputs are unchanged.
            diverged_row_ids = []
            proof_counts = {}
            skip_counts = {}

            def _record_proof(row_id, proof):
                proof_counts[proof] = proof_counts.get(proof, 0) + 1

            def _record_skip(row_id, reason):
                skip_counts[reason] = skip_counts.get(reason, 0) + 1

            import_settlement_candidates_in_transaction(
                conn, claim, candidate_ref, rows, clock=self.clock, session=session,
                on_divergence=diverged_row_ids.append, on_admitted=_record_proof,
                on_skip=_record_skip)
            resolved_admitted = (proof_counts.get("legacy_exit_finality", 0)
                                + proof_counts.get("v2_finality_session", 0))
            if diverged_row_ids or resolved_admitted or skip_counts:
                row_ids = sorted(str(row_id) for row_id in diverged_row_ids)
                document_out = {"schema_version": "settlement_commit_receipt.v1.0",
                                "scope": effect_scope(claim), "session": occurrence,
                                "settlement_divergences": len(row_ids), "row_ids": row_ids,
                                "proof_counts": proof_counts,
                                "already_resolved": skip_counts.get("already_resolved", 0),
                                "already_observed_this_session":
                                    skip_counts.get("already_observed_this_session", 0)}
                ref = self.store.publish_bytes(
                    json.dumps(document_out, sort_keys=True).encode(),
                    schema_ref="settlement_commit_receipt.v1.0")
                register_artifact(conn, ref, claim.attempt_id, self.clock)
                conn.execute("INSERT INTO attempt_outputs VALUES (?,?,?)",
                            (claim.attempt_id, "settlement_commit_receipt", ref.artifact_id))

        return _commit, ()

    def close(self):
        for running in self.running.values():
            signal_owned(running.identities, self.boot, hard=True)
            running.process.wait(timeout=5)
            os.close(running.result_fd)
        self.running.clear()
        self.lock.release()


def serve(service, *, once=False, until=None, deadline_at=None):
    """Tick until stopped: ``once`` for a single idle pass, ``until`` when the
    caller owns a completion predicate, or ``deadline_at`` (an absolute,
    ``service.clock``-comparable datetime) as a hard wall-clock stop checked
    every tick alongside ``until`` -- never before ``service.start()``, so the
    recovery pass a start performs always runs. Returns ``"deadline_exceeded"``
    if the deadline fired before ``until``/``once`` did, else ``"until"`` (an
    ``until`` predicate fired), ``"once_idle"`` (an idle ``once`` pass), or
    ``None`` (the bare forever-loop shape ``once=False, until=None,
    deadline_at=None`` never returns by construction, unchanged from before
    this change).
    """
    service.start()
    try:
        while True:
            active = service.tick()
            if once and not active:
                return "once_idle"
            if until is not None and until():
                return "until"
            if deadline_at is not None and service.clock.now() >= deadline_at:
                return "deadline_exceeded"
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
