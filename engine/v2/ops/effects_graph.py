"""Coordinator-side work for the export, engineering-gate, publication and
backup outbox effects (P2-5/Task5, guide §9.4 item 3).

Each of these four job kinds has a trivial pure worker (``worker.py``'s
``_dispatch_effect_receipt``) that never touches the catalog. All real
catalog, outbox, watermark and filesystem work happens here, called from
``supervisor.Service._coordinator_effect`` — the same fenced finish path
``decision_commit``/``decision_evidence`` already use.

``checks/*`` is verification/dev-tooling, never importable from
``engine/v2/**`` (``checks/import_layers.py``'s "production packages cannot
depend on verification or test code"), and only ``engine.v2.ops.executor``
and ``engine.v2.ops.legacy_adapter`` may call ``subprocess`` at all
(``check_runtime_edges``'s "undeclared-process-edge"). The engineering gate,
the security scan and the export read-back all cross a process boundary, so
this module never calls ``subprocess`` itself — it delegates to the three
audited functions ``legacy_adapter`` exposes for exactly this.
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from engine.v2.contracts import EngineeringNight, JobSpec, OperationsStatus
from engine.v2.foundation import content_hash, ensure_directory, format_timestamp, to_document
from engine.v2.ledger.export import export_generation
from engine.v2.ops.backup import prepare_backup, run_backup
from engine.v2.ops.catalog import dumps, load_json, transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.health import (
    engineering_history,
    engineering_streak_from_history,
    health,
    record_check,
    trailing_occurrences,
)
from engine.v2.ops.health import write_health as _write_status_document
from engine.v2.ops.input_bindings import recorded_bindings
from engine.v2.ops.legacy_adapter import run_engineering_gate as _run_engineering_gate_subprocess
from engine.v2.ops.legacy_adapter import run_security_scan
from engine.v2.ops.legacy_adapter import verify_export_generation as _verify_generation
from engine.v2.ops.outbox import fail_effect, watermark
from engine.v2.ops.publication import current as release_current
from engine.v2.ops.publication import publish_local, stage_release
from engine.v2.ops.session_resolution import resolve_effective_session

__all__ = ["EXPORT_PURPOSES", "backup_effect", "effect_scope", "engineering_gate_effect",
           "experiment_effect", "ledger_export_effect", "publication_effect",
           "reconcile_publication_status", "write_publication_terminal_status"]

#: P2-5/D20: only these two purposes are legacy-compatible rows; a
#: ``research_reconstruction`` row must never reach a legacy-compatible export.
EXPORT_PURPOSES = ("legacy_import", "shadow")


def effect_scope(claim):
    """The watermark/export/release scope this attempt was planned under."""
    return claim.spec.parameters.get("effect_scope") or claim.spec.output_namespace


def _watermark_row(conn, scope, stage):
    return conn.execute(
        "SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' "
        "AND scope=? AND stage=?", (scope, stage)).fetchone()


# --------------------------------------------------------------------------
# ledger_export
# --------------------------------------------------------------------------


def _catalog_counts(conn, purposes):
    placeholders = ",".join("?" for _ in purposes)
    rows = conn.execute(
        "SELECT kind, COUNT(*) FROM decisions WHERE purpose IN (" + placeholders + ") "
        "GROUP BY kind", tuple(purposes)).fetchall()
    counts = {"predictions": 0, "outcomes": 0}
    for kind, count in rows:
        counts["predictions" if kind == "prediction" else "outcomes"] = count
    return counts


def _tar_bytes(generation_dir):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for child in sorted(Path(generation_dir).iterdir()):
            archive.add(child, arcname=child.name)
    return buffer.getvalue()


def _mark_export_delivered(conn, logical_key, receipt):
    conn.execute("UPDATE outbox SET state='delivered', attempts=attempts+1, receipt_json=? "
                 "WHERE kind='export' AND logical_key=? AND state IN ('pending','running')",
                 (dumps(receipt), logical_key))


def _no_keepalive():
    return None


def ledger_export_effect(conn, store, claim, ops_root, repo_root, *, clock,
                         keepalive=_no_keepalive, fault=None):
    """Build, verify and publish one export generation; deliver its outbox effects.

    Settlement is read from its own watermark, never as a scheduler
    dependency (nightly.py's ``parent_map``): a failed or not-yet-run
    settlement leaves that watermark absent, and export still proceeds,
    recording settlement as absent in its receipt.

    ``fault`` (P2-C06, test-only) is threaded straight into
    ``export_generation`` so a test can interrupt the write of one export
    file among several; the old ``CURRENT`` stays readable and a retry
    (a fresh call with no ``fault``) resumes and completes without an
    ``export generation differs from catalog`` conflict.
    """
    scope = effect_scope(claim)
    # guide §5.5 item 1: this job's own pinned plan identity, so a genuinely
    # new same-session generation records its OWN "export" watermark receipt
    # instead of colliding with an earlier generation's.
    generation = _generation_ref(claim)
    # P2-C03: the job's own ``session`` param is always the REQUESTED date;
    # the decisions watermark's occurrence is decision_commit's own
    # finality-RESOLVED date (context["session"], written verbatim by
    # commit_decisions_in_transaction) — export refers to that resolved
    # session, never the requested one, which legitimately differs on a
    # walk-back night.
    requested_session = claim.spec.parameters["session"]
    decisions_wm = _watermark_row(conn, scope, "decisions")
    if decisions_wm is None:
        raise fail("VALIDATION_FAILED", "no committed decisions for this session",
                   details={"scope": scope, "session": requested_session})
    session = decisions_wm["occurrence"]
    release_key = decisions_wm["receipt_ref"]
    settlement_wm = _watermark_row(conn, scope, "settlement")
    settlement_present = settlement_wm is not None and settlement_wm["occurrence"] == session

    root = Path(ops_root) / "exports" / scope
    # The full ledger can grow independently of this decision receipt (for
    # example through history imports or settlement). Its immutable export
    # identity therefore follows the exported bytes, not the outbox key.
    generation_dir = export_generation(conn, root, purposes=EXPORT_PURPOSES,
                                       fault=fault)
    keepalive()
    catalog_counts = _catalog_counts(conn, EXPORT_PURPOSES)
    verified_counts = _verify_generation(generation_dir, repo_root)
    keepalive()
    if verified_counts != catalog_counts:
        raise fail("INTEGRITY_FAILED", "exported generation disagrees with the catalog",
                   details={"catalog": catalog_counts, "verified": verified_counts})

    tar_ref = store.publish_bytes(_tar_bytes(generation_dir), schema_ref="ledger_generation.v1.0")
    keepalive()
    receipt = {"schema_version": "ledger_export_receipt.v1.0", "scope": scope, "session": session,
               "requested_session": requested_session,
               "generation": generation_dir.name, "counts": verified_counts,
               "settlement": {"present": settlement_present,
                              "release_key": settlement_wm["receipt_ref"] if settlement_present
                              else None}}

    def effect(inner_conn):
        _mark_export_delivered(inner_conn, release_key, receipt)
        if settlement_present:
            _mark_export_delivered(inner_conn, settlement_wm["receipt_ref"], receipt)
        watermark(inner_conn, "nightly", scope, "export", session, tar_ref.content_hash, clock=clock,
                 generation=generation)

    return effect, (("ledger_export", tar_ref),)


# --------------------------------------------------------------------------
# engineering_gate
# --------------------------------------------------------------------------


def _run_engineering_gate(repo_root):
    raw = _run_engineering_gate_subprocess(repo_root)
    # Coverage is a commit-time ratchet, not a nightly check (decision #3):
    # excluded from both the recorded rows and this stage's own "ok".
    rows = {**raw.get("structural", {}),
            **{k: v for k, v in raw.get("engineering", {}).items() if k != "coverage"}}
    ok = bool(rows) and all(isinstance(row, dict) and row.get("ok") for row in rows.values())
    return {"schema_version": "engineering_gate.v1.0", "ok": ok, "rows": rows,
            "code_hash": raw.get("code_hash")}


def engineering_gate_effect(conn, store, claim, repo_root, *, clock):
    scope = effect_scope(claim)
    session = claim.spec.parameters["session"]
    # guide §5.5 item 1: the real 2026-09-14 failure -- an earlier generation
    # (a different code_hash) had already completed engineering_gate for this
    # (scope, session), so this run's own gate document, differing only in
    # code_hash, collided with it. Scoping the receipt per generation is what
    # fixes that; a genuine retry of the SAME generation stays idempotent.
    generation = _generation_ref(claim)
    document = _run_engineering_gate(repo_root)
    ref = store.publish_bytes(json.dumps(document, sort_keys=True, default=str).encode(),
                              schema_ref="engineering_gate.v1.0")

    def effect(inner_conn):
        watermark(inner_conn, "nightly", scope, "engineering_gate", session, ref.content_hash,
                 clock=clock, generation=generation)
        # guide §5.5 item 2: one durable engineering OBSERVATION per
        # scheduled occurrence (scope, session) -- independent of the
        # per-generation watermark above. ``record_check``'s own
        # PRIMARY KEY(occurrence, kind) collapses a retry or a later same-
        # night generation onto the SAME row (bumping only its retry
        # counter), so the engineering history this feeds
        # (``health.engineering_history``) counts nights, never retries or
        # generations; a night nobody ever observed simply has no row and
        # reads back as unknown.
        record_check(inner_conn, session, "engineering", document["ok"],
                     {"schema_version": "engineering_observation.v1.0", "scope": scope,
                      "generation": generation, "code_hash": document.get("code_hash"),
                      "receipt_ref": ref.content_hash, "recorded_at": format_timestamp(clock.now())})

    return effect, (("engineering_gate", ref),)


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------


def _gate_dict(store, document):
    ref = store.publish_bytes(json.dumps(document, sort_keys=True, default=str).encode(),
                              schema_ref=document["kind"] + "_gate.v1.0")
    return {"ok": document["status"] == "passed", "receipt_ref": ref.content_hash,
            "input_hash": document["input_hash"], "receipt_artifact": to_document(ref)}


def _decision_gate(conn, store, scope, session, binding_hash):
    # Passed only when the ("nightly", scope, "decisions") watermark shows
    # occurrence == this release's own session — the exact session decision
    # commit's watermark(...) call wrote, whether or not it carried any
    # candidates: a no-entry night still commits (zero decisions, export and
    # release_intent still enqueued) and still advances this watermark, so it
    # still passes. An earlier (or absent) watermark means this release does
    # not speak for its own session's decisions and must be refused.
    row = _watermark_row(conn, scope, "decisions")
    passed = row is not None and row["occurrence"] == session
    document = {"schema_version": "decision_gate.v1.0", "kind": "decision",
                "status": "passed" if passed else "failed", "input_hash": binding_hash,
                "release_key": row["receipt_ref"] if row else None}
    return _gate_dict(store, document)


def _projection_gate(conn, store, bindings, binding_hash):
    row = bindings.get("selfcheck.json")
    passed, detail = False, "selfcheck receipt missing"
    if row is not None:
        ref = artifact(conn, store, row.artifact_id)
        payload = json.loads(store.read_verified(ref))
        passed = payload.get("ok") is True
        detail = None if passed else "selfcheck ok is not true"
    document = {"schema_version": "projection_gate.v1.0", "kind": "projection",
                "status": "passed" if passed else "failed", "input_hash": binding_hash,
                "detail": detail}
    return _gate_dict(store, document)


def _engineering_receipt_gate(conn, store, bindings, binding_hash):
    row = bindings.get("engineering_gate.json")
    passed = False
    if row is not None:
        ref = artifact(conn, store, row.artifact_id)
        payload = json.loads(store.read_verified(ref))
        passed = payload.get("ok") is True
    document = {"schema_version": "engineering_receipt_gate.v1.0", "kind": "engineering",
                "status": "passed" if passed else "failed", "input_hash": binding_hash}
    return _gate_dict(store, document)


def _security_gate(store, files, binding_hash, repo_root, store_root):
    bundle_ref = files.get("bundle.tar")
    scan = (run_security_scan(store.verify(bundle_ref), repo_root, store_root)
            if bundle_ref else {"ok": False})
    document = {"schema_version": "security_gate.v1.0", "kind": "security",
                "status": "passed" if scan.get("ok") else "failed", "input_hash": binding_hash,
                "secrets_loaded": scan.get("secrets_loaded", 0),
                "violations": scan.get("violations", [])}
    return _gate_dict(store, document)


def _generation_ref(claim):
    """The pinned plan identity this publication job's own parameters carry
    (guide §5.5 item 1; ``nightly.py``'s ``_legacy_params``, action ==
    "publication" mirrors what ``decision_evidence`` already pins).

    Empty for a claim built without those fields -- pre-existing lower-level
    tests construct a minimal ``LegacyParameters`` claim directly and never
    set them, so they keep getting the release id's original
    ``(scope, session)``-only form. Every REAL nightly plan
    (``build_nightly_plan``/``plans.py::nightly_plan``) sets both
    unconditionally, so a real publication job always takes the generation-
    aware branch below.
    """
    params = claim.spec.parameters
    deployment = params.get("deployment") or ""
    decision_clock = params.get("decision_clock") or ""
    input_bindings = params.get("input_bindings") or {}
    manifest = input_bindings.get("legacy_manifest.json") or ""
    if not (deployment or decision_clock or manifest):
        return ""
    return content_hash({"deployment": deployment, "decision_clock": decision_clock,
                         "manifest": manifest, "input_bindings": input_bindings})


def _bind_release_intent(conn, scope, session, release_id, *, clock):
    """Mark the ``release_intent`` outbox row this release is bound to as
    delivered (P2-C06 decision 2): "bound to a release", never "published" —
    that stays with ``publish_local``'s own pointer acknowledgement.

    The matching row is found the same way ``_decision_gate`` verified
    eligibility: the ``decisions`` watermark's own ``receipt_ref`` IS the
    ``release_key`` ``commit_decisions_in_transaction`` enqueued both
    ``export`` and ``release_intent`` under. Idempotent by construction (the
    ``state IN (...)`` guard is a no-op once already delivered), so a retried
    ``publication_effect`` call after a failed ``publish_local`` re-uses the
    same binding without conflict.
    """
    row = _watermark_row(conn, scope, "decisions")
    if row is None or row["occurrence"] != session:
        return
    receipt = {"release_id": release_id, "bound_at": format_timestamp(clock.now())}
    with transaction(conn):
        conn.execute(
            "UPDATE outbox SET state='delivered', attempts=attempts+1, receipt_json=? "
            "WHERE kind='release_intent' AND logical_key=? AND state IN ('pending','running')",
            (dumps(receipt), row["receipt_ref"]))


def _publication_files(conn, store, bindings):
    """The release's own file set: the bundle, plus an optional bound
    ``projection_binding.json`` (guide §5.4/P3-1c) -- the chosen operator
    entry point for binding a Phase-3 projection candidate to this fenced
    publisher. This is the smaller of the guide's two named options ("a
    coordinator step in tools/v2_dashboard_project.py, or a publication-
    effect input"): one more optional named ``input_bindings`` entry, the
    same mechanism ``bundle.tar``/``finality.json``/``selfcheck.json``/
    ``engineering_gate.json`` already use -- no new job kind or DAG wiring.
    `tools/v2_dashboard_project.py` (the one place allowed to compose
    serving and ops) is what actually BUILDS this document, via
    ``engine.v2.serving.projections.projection_binding``, and emits it
    alongside its existing release output so an operator/submission script
    can register it and bind it here the same way it already binds the
    render bundle. Its presence is the only branch: an ordinary bundle-only
    publication (no projection candidate yet, e.g. still on the P3-0
    compatibility preview) behaves exactly as before. Adding this file
    changes the caller's ``binding_hash``, so a candidate with a bound
    projection always gets freshly bound gates -- never a receipt computed
    for an earlier, projection-less binding."""
    bundle_row = bindings.get("bundle.tar")
    if bundle_row is None:
        raise fail("VALIDATION_FAILED", "publication has no projection bundle bound")
    files = {"bundle.tar": artifact(conn, store, bundle_row.artifact_id)}
    binding_row = bindings.get("projection_binding.json")
    if binding_row is not None:
        files["projection_binding.json"] = artifact(conn, store, binding_row.artifact_id)
    return files


# --------------------------------------------------------------------------
# operations status -- guide §5.5 items 2-3: the versioned status sidecar
# publication_effect writes on every attempt, success or failure.
# --------------------------------------------------------------------------


def _bundle_flags(store, bundle_ref):
    """Conflicts and degraded-model-evidence flags, read back out of the
    ALREADY-RENDERED bundle's own ``data/flags.json`` (P2-C08,
    ``engine.v2.ops.render_inputs.render_flags``) -- never recomputed here.
    A bundle with no such member (a synthetic test fixture, or a bundle
    format that predates flags.json) simply carries neither list -- not an
    error, since ``bundle.tar``'s only REQUIRED content is the board itself.
    """
    try:
        with tarfile.open(store.verify(bundle_ref)) as archive:
            member = archive.extractfile("bundle/data/flags.json")
            flags_doc = json.loads(member.read()) if member is not None else {}
    except KeyError:
        flags_doc = {}
    flags = flags_doc.get("flags") or []
    conflicts = tuple(f for f in flags if isinstance(f, dict) and f.get("kind") == "calendar_date_conflict")
    degraded = tuple(f for f in flags if isinstance(f, dict) and f.get("kind") == "model_evidence_stale")
    return conflicts, degraded


def _selfcheck_document(conn, store, bindings):
    """This publication's own bound ``selfcheck.json`` (``legacy_selfcheck``,
    which already validated the just-rendered bundle) verbatim -- carried,
    never reconstructed. An explicit unknown state when none is bound,
    mirroring ``render_inputs.unknown_selfcheck_report`` (never a bare
    ``None`` a reader could mistake for "checked and fine")."""
    row = bindings.get("selfcheck.json")
    if row is None:
        return {"ok": None, "known": False, "detail": "no selfcheck bound to this publication"}
    ref = artifact(conn, store, row.artifact_id)
    return json.loads(store.read_verified(ref))


def _write_operations_status(conn, store, target, *, scope, requested_session, resolved_session,
                             bindings, bundle_ref, clock, attempted_release_id, failed_update, failure):
    """Write the guide §5.5 items 2-3 status sidecar to
    ``<target>/operations_status.json`` -- a plain mutable file directly
    under the fenced publisher's own scope root (a sibling of ``CURRENT``),
    never inside a specific release's own immutable file set. That placement
    is what lets a FAILED update still record its own failure reason here
    while the old release stays current and untouched: an immutable, per-
    release file could only ever describe the release that carries it, and a
    generation that never reaches ``stage_release``/``publish_local``
    success never gets one.

    Called twice from ``publication_effect`` -- once on success, once from
    the ``except`` branch on failure -- so every attempt, not only a
    published one, leaves a fresh, accurate document.
    """
    occurrences = trailing_occurrences(resolved_session)
    history = tuple(EngineeringNight(**row) for row in engineering_history(conn, occurrences))
    conflicts, degraded = _bundle_flags(store, bundle_ref) if bundle_ref is not None else ((), ())
    selfcheck_doc = _selfcheck_document(conn, store, bindings)
    snapshot = health(conn, clock=clock)
    current_release_id = release_current(target)
    # A judgement call (guide §5.5 item 2's own report should note it): a
    # served release that is NOT the one this latest attempt just tried to
    # publish is, by definition, stale relative to that attempt -- on
    # success the two always match (the attempt IS what became current); on
    # a failed update they differ because the old release was kept.
    stale = current_release_id != attempted_release_id
    failed_update_reason = f"{failure.code}: {failure.problem.message}" if failure is not None else None
    withheld_release = snapshot.get("withheld_release")
    withheld_reason = None
    if withheld_release is not None:
        withheld_reason = (f"occurrence {withheld_release['occurrence']} release "
                           f"{withheld_release['release_id']} was staged but not eligible")
    document = OperationsStatus(
        scope=scope, release_id=current_release_id, attempted_release_id=attempted_release_id,
        generated_at=format_timestamp(clock.now()),
        requested_session=requested_session, resolved_session=resolved_session,
        engineering_history=history, engineering_streak=engineering_streak_from_history(history),
        conflicts=conflicts, degraded_model_evidence=degraded, selfcheck=selfcheck_doc,
        stale=stale, stale_reason=(failed_update_reason or "the current release was not updated "
                                   "by the latest attempt") if stale else None,
        withheld=withheld_release is not None, withheld_reason=withheld_reason,
        failed_update=bool(failed_update), failed_update_reason=failed_update_reason)
    ensure_directory(target)
    # ``health.write_health``'s atomic temp-write/fsync/rename is generic --
    # reused verbatim rather than duplicated for this differently-schemad
    # sidecar (imported here as ``_write_status_document``).
    _write_status_document(target / "operations_status.json", to_document(document))


def _terminal_job_failure_reason(conn, job_row):
    """Name the kind and failure code of the job that actually caused
    ``job_row``'s own terminal state -- 2026-09-14 review fix, item 1.

    ``job_row`` itself, for a publication job that was ``block_descendants``-
    ed rather than run, carries a ``DEPENDENCY_FAILED`` ``Problem`` whose
    ``dependency_refs`` names the ROOT job of the cascade (``lifecycle.
    block_descendants``'s recursive CTE stamps every transitive descendant
    with the SAME root id, never the immediate parent) -- so one lookup of
    that job's own row gives "the first failed job's kind and failure
    code" the guide asks for, whether the cascade came from a failure
    (``advance_job``) or a cancellation (``request_cancel``/
    ``complete_cancel``/``recovery.expire_leases``, all of which use the
    identical ``DEPENDENCY_FAILED`` framing). A publication job that instead
    failed or was cancelled DIRECTLY (no upstream root to look up) names
    itself.
    """
    failure = json.loads(job_row["failure_json"]) if job_row["failure_json"] else None
    if failure and failure.get("code") == "DEPENDENCY_FAILED" and failure.get("dependency_refs"):
        upstream = conn.execute("SELECT kind, failure_json FROM jobs WHERE job_id = ?",
                                (failure["dependency_refs"][0],)).fetchone()
        if upstream is not None:
            upstream_failure = (json.loads(upstream["failure_json"])
                               if upstream["failure_json"] else None)
            code = upstream_failure.get("code") if upstream_failure else "CANCELLED"
            message = (upstream_failure.get("message") if upstream_failure
                      else "cancelled by request")
            return f"{upstream['kind']} {code}: {message}"
    if failure:
        return f"{job_row['kind']} {failure.get('code')}: {failure.get('message')}"
    return f"{job_row['kind']} cancelled: cancelled by request"


def write_publication_terminal_status(conn, store, ops_root, job_row, *, clock):
    """The guide §5.5 items 2-3 status sidecar's OTHER writer -- 2026-09-14
    review fix, item 1: a nightly run whose PUBLICATION job reaches a
    terminal non-success state (blocked, failed or cancelled) WITHOUT
    ``publication_effect`` ever running at all. That is the REAL 2026-09-14
    failure shape: an upstream dependency (``engineering_gate``,
    ``legacy_finality``) fails or the run is cancelled, ``block_descendants``
    marks the publication job ``blocked`` before it is ever claimed, and
    ``publication_effect`` -- the ONLY place ``_write_operations_status``
    was previously called from -- never runs, leaving the sidecar
    describing the OLD release as fine. Never duplicates
    ``_write_operations_status``'s own ``OperationsStatus`` construction or
    file write; both funnel through the SAME ``OperationsStatus`` dataclass
    and ``_write_status_document``.

    Called from ``supervisor.Service``'s own per-tick reconciliation
    (``_reconcile_publication_status``) -- the one place, across the whole
    service loop, that already observes every job's state after each
    ``tick()`` -- rather than threading ``store``/``ops_root`` down through
    ``lifecycle.py``'s job-transition functions themselves (a foundational,
    kind-agnostic layer no other effect reaches into either). This function
    itself always (re)writes unconditionally, exactly like
    ``_write_operations_status``; ``_reconcile_publication_status`` is what
    keeps calling it idempotent and self-healing, by only calling it while
    this job's own ``updated_at`` is NEWER than what the sidecar currently
    reports (``generated_at``) -- so a later successful publish's own,
    fresher document is never clobbered by a stale blocked/cancelled row
    that is still sitting in the table.
    """
    spec = load_json(JobSpec, job_row["spec_json"])
    scope = spec.parameters.get("effect_scope") or spec.output_namespace
    requested_session = spec.parameters.get("session")
    if not scope or not requested_session:
        return
    target = Path(ops_root) / "releases" / scope
    try:
        history = tuple(EngineeringNight(**row) for row in
                        engineering_history(conn, trailing_occurrences(requested_session)))
    except OpsError:
        # guide §5.5 item 3: a calendar that cannot be resolved refuses
        # rather than silently falling back -- an unresolved window here
        # still must not block reporting the (already-known) failure/
        # cancellation reason, so the history is explicitly empty (renders
        # as no observed nights, never a fabricated pass) rather than this
        # whole write being skipped.
        history = ()
    reason = _terminal_job_failure_reason(conn, job_row)
    document = OperationsStatus(
        scope=scope, release_id=release_current(target), attempted_release_id=None,
        generated_at=format_timestamp(clock.now()),
        requested_session=requested_session, resolved_session=requested_session,
        engineering_history=history, engineering_streak=engineering_streak_from_history(history),
        conflicts=(), degraded_model_evidence=(),
        selfcheck={"ok": None, "known": False, "detail": "publication did not run: " + reason},
        stale=True, stale_reason=reason,
        withheld=False, withheld_reason=None,
        failed_update=True, failed_update_reason=reason)
    ensure_directory(target)
    _write_status_document(target / "operations_status.json", to_document(document))


def _sidecar_generated_at(target):
    """The ``generated_at`` a scope's ``operations_status.json`` currently
    reports, or ``None`` for a missing/unreadable file -- the one thing
    ``reconcile_publication_status`` compares a job's own ``updated_at``
    against to decide whether it is describing something NEWER than what
    is already on disk."""
    path = Path(target) / "operations_status.json"
    if path.is_symlink() or not path.is_file():
        return None
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return document.get("generated_at") if isinstance(document, dict) else None


#: Latest 'publication' job PER SCOPE, in one grouped query rather than a
#: scan plus per-row reads (2026-09-14 second review fix) -- ``scope`` is
#: computed the SAME way ``effect_scope(claim)`` does (parameters'
#: ``effect_scope`` if present and non-empty, else ``output_namespace``;
#: ``NULLIF(..., '')`` folds the "present but empty" default
#: ``nightly.py``'s own ``_legacy_params`` can pass into the same NULL
#: COALESCE falls through on), and the window function keeps ONLY rank 1
#: (latest ``updated_at``, ``job_id`` breaking a tie) per scope -- every
#: OTHER historical row, terminal or not, is never even materialized into
#: Python. ``ANY`` state is selected (not filtered to blocked/failed/
#: cancelled) so the caller can tell "latest is a genuine failure" apart
#: from "latest already succeeded, or is still in flight" without a second
#: query.
_LATEST_PUBLICATION_PER_SCOPE_SQL = """
    WITH scoped AS (
        SELECT *,
               COALESCE(NULLIF(json_extract(spec_json, '$.parameters.effect_scope'), ''),
                         json_extract(spec_json, '$.output_namespace')) AS scope
        FROM jobs WHERE kind = 'publication'
    )
    SELECT * FROM (
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY scope ORDER BY updated_at DESC, job_id DESC) AS rn
        FROM scoped
    ) WHERE rn = 1
"""


def reconcile_publication_status(conn, store, ops_root, *, clock):
    """2026-09-14 review fix, item 1 (plus its own second-pass fix): call
    once per ``supervisor.Service.tick()`` (``Service.
    _reconcile_publication_status``). For EACH scope, looks only at that
    scope's OWN latest ``publication`` job (any state, most recent
    ``updated_at``) and, only when that latest job is itself blocked,
    failed or cancelled AND its scope's sidecar is not already newer, calls
    :func:`write_publication_terminal_status` -- covering the failure shape
    ``publication_effect``'s own sidecar write can never see, because
    ``publication_effect`` never runs for a job that was blocked or
    cancelled before its own attempt even started (the real 2026-09-14
    failure: an upstream ``engineering_gate``/``legacy_finality`` failure,
    or a cancelled run).

    Second-pass fix, found reviewing against ``/root/phase2-shadow-ops``
    (several historical blocked publications, no sidecar yet, several
    attempts): the FIRST pass scanned every blocked/failed/cancelled
    ``publication`` row UNORDERED and wrote whichever one SQLite returned
    first, relying only on ``generated_at > updated_at`` to skip later
    ones. On a real root that row is not necessarily the LATEST failure --
    once ANY one of them got written, its own fresh ``generated_at`` was
    already newer than every OTHER row's (older) ``updated_at``, so the
    actual most recent failure was skipped forever, and a scope with only
    an OLDER blocked job but a NEWER already-succeeded publication (a
    scenario the first pass never even considered) got a false failure
    sidecar. Both are fixed by only ever looking at each scope's single
    latest job at all: a stale sidecar can never outrank a newer failure
    because there is no other blocked/failed/cancelled row left to compare
    against, and a newer success is never shadowed by an older failure
    because the older row is not even in the result set.

    The ``generated_at`` STRICT-greater-than skip (never ``>=``) still
    matters ONLY for that ONE latest job: a rewrite whose own
    ``generated_at`` lands in the SAME instant as the job's own
    ``updated_at`` (a frozen test clock; two events in one real wall-clock
    tick) must still go through, since equal timestamps carry no
    information about which happened first; once a LATER successful
    publish writes a sidecar with a ``generated_at`` strictly past this
    job's own ``updated_at``, the same still-blocked row (now no longer
    even the scope's latest, since the newer success outranks it) is
    skipped on every later tick either way.
    """
    for row in conn.execute(_LATEST_PUBLICATION_PER_SCOPE_SQL).fetchall():
        scope = row["scope"]
        if not scope or row["state"] not in ("blocked", "failed", "cancelled"):
            continue
        target = Path(ops_root) / "releases" / scope
        existing = _sidecar_generated_at(target)
        if existing is not None and existing > row["updated_at"]:
            continue
        write_publication_terminal_status(conn, store, ops_root, row, clock=clock)


def _stage_and_publish(conn, store, claim, release_id, session, files, gates, *, expected_current,
                       target, scope, generation_ref, fault, keepalive, clock, status_kwargs):
    """Stage then publish one release, recording the guide §5.5 items 2-3
    status sidecar either way -- a failed update's own reason on an
    ``OpsError``, or the newly current release's state on success -- and
    always propagating the original exception (if any) unchanged."""
    try:
        staged = stage_release(conn, store, release_id, session, files, expected_current=expected_current,
                               gates=gates, clock=clock, claim=claim)
        if staged["eligible"]:
            _bind_release_intent(conn, scope, session, release_id, clock=clock)
        keepalive()
        # guide §5.5 item 1: the SAME generation identity used to form
        # ``release_id`` also scopes the "publication"/"delivery" watermark
        # rows ``publish_local`` writes, so a second generation's publish
        # never collides with the first's already-acknowledged receipt.
        publish_local(conn, claim, store, target, release_id, scope=scope, clock=clock,
                      generation=generation_ref, fault=fault)
    except OpsError as exc:
        _write_operations_status(conn, store, target, failed_update=True, failure=exc, **status_kwargs)
        raise
    _write_operations_status(conn, store, target, failed_update=False, failure=None, **status_kwargs)


def publication_effect(conn, store, claim, ops_root, repo_root, *, clock, store_root=None,
                       keepalive=_no_keepalive, fault=None):
    """Build the four gate receipts, stage the release, then publish it.

    ``stage_release``/``publish_local`` manage their own short transactions
    (§catalog.py forbids nesting), so this whole function runs outside the
    supervisor's own commit_attempt transaction — everything it commits is
    already durable by the time it returns, so the effect closure returned
    to the caller is a no-op.

    P2-C06: once ``stage_release`` produces an ELIGIBLE release (all four
    gates bound, including the decision gate), the release-intent lifecycle
    completes here — ``_bind_release_intent`` — before ``publish_local`` is
    even attempted, so a release that is staged but never successfully
    published still leaves ``release_intent`` bound (never re-pending) while
    ``publication``/``delivery`` stay exactly where a failed attempt left
    them. ``fault`` (test-only) is threaded straight into ``publish_local``.

    ``store_root`` is the checkout the security gate loads the real
    ``.env`` from (``Service.store_root``) -- distinct from ``repo_root``,
    which is only the CODE checkout the security scan subprocess runs from
    and may be a frozen/snapshot worktree with no ``.env`` at all. Defaults
    to ``repo_root`` for a caller with one combined checkout; a caller with
    the two apart (production) always passes it.
    """
    scope = effect_scope(claim)
    bindings = recorded_bindings(conn, claim.attempt_id)
    # P2-C03: the bound finality document is the independent anchor for
    # "which session does this release speak for" — never the decisions
    # watermark's own occurrence alone, so a stale watermark for the wrong
    # session still refuses ``_decision_gate`` below, exactly as before,
    # while a genuine walk-back (resolved != requested) still passes.
    finality_row = bindings.get("finality.json")
    if finality_row is None:
        raise fail("VALIDATION_FAILED", "publication has no finality bound")
    finality_doc = json.loads(store.read_verified(artifact(conn, store, finality_row.artifact_id)))
    session = resolve_effective_session(finality_doc, claim.spec.parameters["session"])
    files = _publication_files(conn, store, bindings)
    target = Path(ops_root) / "releases" / scope
    # guide §5.5 item 1: fold the pinned plan identity into the release id so
    # a genuinely new same-session generation (changed implementation,
    # manifest or decision_clock) gets a DISTINCT release -- never colliding
    # with, overwriting, or ``IDEMPOTENCY_CONFLICT``-ing against an older
    # generation's already-staged release -- while an identical-plan retry
    # (unchanged generation ref) reproduces the exact same release id.
    generation_ref = _generation_ref(claim)
    release_parts = [scope, session, generation_ref] if generation_ref else [scope, session]
    release_id = "rel" + content_hash(release_parts).split(":")[1][:24]
    expected_current = release_current(target)
    binding_hash = content_hash({"release_id": release_id, "occurrence": session, "files": files})
    builders = (
        ("decision", lambda: _decision_gate(conn, store, scope, session, binding_hash)),
        ("projection", lambda: _projection_gate(conn, store, bindings, binding_hash)),
        ("security", lambda: _security_gate(store, files, binding_hash, repo_root,
                                            store_root if store_root is not None else repo_root)),
        ("engineering", lambda: _engineering_receipt_gate(conn, store, bindings, binding_hash)),
    )
    gates = {}
    for name, build in builders:
        keepalive()
        gates[name] = build()
    keepalive()
    # guide §5.5 items 2-3: staging/publishing may still fail (a foreseeable
    # gate/fence/idempotency refusal) -- ``_stage_and_publish`` records the
    # status sidecar either way, a failed-update reason on refusal or the
    # newly current release's state on success, and re-raises unchanged.
    status_kwargs = dict(scope=scope, requested_session=claim.spec.parameters["session"],
                         resolved_session=session, bindings=bindings, bundle_ref=files.get("bundle.tar"),
                         clock=clock, attempted_release_id=release_id)
    _stage_and_publish(conn, store, claim, release_id, session, files, gates,
                       expected_current=expected_current, target=target, scope=scope,
                       generation_ref=generation_ref, fault=fault, keepalive=keepalive, clock=clock,
                       status_kwargs=status_kwargs)
    return None, ()


# --------------------------------------------------------------------------
# backup
# --------------------------------------------------------------------------


def backup_effect(conn, store, claim, ops_root, *, clock, fault=None, keepalive=_no_keepalive):
    """Prepare then run one backup; a failure releases its outbox effect back
    to pending (rather than waiting out its lease) so an immediate retry can
    claim it — the job's own retry policy is what makes this "retryable"."""
    scope = effect_scope(claim)
    # guide §5.5 item 1: this job's own pinned plan identity, so a genuinely
    # new same-session generation records its OWN "backup" watermark receipt.
    generation = _generation_ref(claim)
    # P2-C03: key/watermark the backup by the same resolved session export
    # and decisions use, falling back to the requested one before any
    # decisions watermark exists.
    decisions_wm = _watermark_row(conn, scope, "decisions")
    session = decisions_wm["occurrence"] if decisions_wm is not None else claim.spec.parameters["session"]
    # guide §5.5 item 1: the backup EFFECT's own enqueue key (not only its
    # watermark) must be generation-scoped too -- unlike engineering_gate/
    # ledger_export, ``prepare_backup`` is ALREADY idempotent by content
    # (P2-C07: an existing row for this key is reused untouched), so a
    # second generation reusing generation 1's key would not conflict, but
    # ``run_backup``'s own ``claim(...)`` would find nothing pending once
    # generation 1's backup already delivered and refuse with
    # STALE_EXPECTATION -- generation 2 would never get its own receipt at
    # all. Folding ``generation`` in gives it its own key, effect row and
    # on-disk snapshot, exactly like the other three effects.
    key = "bkp" + content_hash([scope, session, generation]).split(":")[1][:24]
    owner = claim.attempt_id
    prepare_backup(conn, key, {}, clock=clock)
    target = Path(ops_root) / "backups" / scope
    try:
        manifest = run_backup(conn, key=key, owner=owner, target=target, clock=clock, store=store,
                              fault=fault, keepalive=keepalive)
    except Exception:
        row = conn.execute(
            "SELECT effect_id, claim_token FROM outbox WHERE kind='backup' AND logical_key=? "
            "AND claimed_by=?", (key, owner)).fetchone()
        if row is not None:
            fail_effect(conn, row["effect_id"], {"error": "backup_failed"}, owner=owner,
                       claim_token=row["claim_token"], clock=clock)
        raise
    # "its own" watermark (D20): a retry after failure never advances any
    # OTHER stage's watermark, only this one, and only on success.
    with transaction(conn):
        watermark(conn, "nightly", scope, "backup", session, content_hash(manifest), clock=clock,
                 generation=generation)
    return None, ()


# --------------------------------------------------------------------------
# experiment
# --------------------------------------------------------------------------


def _named_ref(refs, name):
    for candidate_name, ref in refs:
        if candidate_name == name:
            return ref
    raise fail("VALIDATION_FAILED", "required effect artifact is missing")


def experiment_effect(conn, store, claim, refs, *, clock):
    """P6 slice 10/11: durably record one experiment attempt after its
    worker receipt validates. Mirrors backup_effect's shape (a coordinator
    effect with no watermark/outbox of its own — an experiment run has no
    generation/session scope to key on).

    ``spec.json`` is re-read from the attempt's OWN durably recorded binding
    (``attempt_input_bindings``, written once at launch by
    ``executor._materialize_inputs``): the exact bytes the worker ran, never
    a re-resolution of the job's live parameters and never a second
    ``resolve_and_record`` call (those rows are immutable and already exist).

    A ``no_ledger=False`` attempt registers a ``primary`` hypothesis and, on
    the one call that actually created it (``created``), appends the durable
    "ran" row itself. The worker subprocess never writes the ledger, so a
    killed-and-retried attempt can never double-append; the idempotent
    registration is what makes this effect exactly-once across retries.
    """
    from engine.v2.ops.experiments import experiment_spec_from_document, register_hypothesis

    receipt = json.loads(store.read_verified(_named_ref(refs, "experiment_receipt")))
    binding = recorded_bindings(conn, claim.attempt_id).get("spec.json")
    if binding is None:
        raise fail("VALIDATION_FAILED", "experiment specification is not bound")
    document = json.loads(store.read_verified(artifact(conn, store, binding.artifact_id)))
    spec = experiment_spec_from_document(document)
    no_ledger = claim.spec.parameters.get("no_ledger", True)
    mode = "smoke" if no_ledger else "primary"
    _run_id, created = register_hypothesis(conn, spec, receipt["input_hash"], mode=mode,
                                           run_id=claim.attempt_id)
    if not no_ledger and created:
        _append_ledger_row(store, spec, receipt)
    return None, ()


def _append_ledger_row(store, spec, receipt):
    """Append the "ran" row for one primary experiment to
    ``<store.root>/experiments/LEDGER.csv``.

    ``experiments.lib`` is NOT a legacy-adapter dependency (it is a plain
    data-format helper package outside ``engine.*``, so
    ``checks/import_layers.py`` records no edge and
    ``checks/legacy_adapters.json`` stays at 75/75); its ``ledger_append``
    also carries the append-only prefix check this writer would otherwise
    have to duplicate. The path is always the operations store root, never the
    live checkout's own ledger, so a test root writes only inside itself.

    The ``spec_hash`` column is the LEGACY spec identity of the registered
    runner's ``spec.yaml`` (its file hash, exactly what ``runner_manifest``
    reports) — never ``spec.spec_hash``, the v2 document hash, which lives in
    a different identity space. A synthetic runner has no legacy spec, so the
    column is empty, matching ``experiments.lib.record_evaluation``'s shape
    for a row with no headline metrics.
    """
    from datetime import datetime, timezone

    from engine.v2.ops.experiments import RUNNER_INVENTORY
    from engine.v2.ops.fingerprints import file_hash
    from experiments.lib import LEDGER_COLUMNS, ledger_append

    root = Path(store.root)
    entry = RUNNER_INVENTORY.get(spec.runner)
    spec_hash_value = ""
    if entry is not None:
        spec_path = root / entry["spec_source"]
        if spec_path.is_file() and not spec_path.is_symlink():
            spec_hash_value = file_hash(spec_path)
    runner_result = (receipt.get("evidence") or {}).get("runner_result") or {}
    if not isinstance(runner_result, dict):
        runner_result = {}
    headline = runner_result.get("headline")
    headline = headline if isinstance(headline, dict) else runner_result
    row = {"id": spec.experiment_id,
           "spec_hash": spec_hash_value,
           "date": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
           "stage": "ran",
           "oos_mean_mid": headline.get("mean", headline.get("oos_mean_mid", "")),
           "sharpe_trade": headline.get("sharpe_trade", ""),
           "promoted": "False"}
    ledger_append([{name: row[name] for name in LEDGER_COLUMNS}],
                  path=root / "experiments" / "LEDGER.csv")
