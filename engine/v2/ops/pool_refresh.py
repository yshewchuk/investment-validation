"""Pool-nightly refresh sidecar (#192 slice 4).

Advances the three ``phase5_state_stage`` pool members (``paired_residual_pool``,
``board_analog_matcher``, ``trailing_pnl_cutoff``) one cutoff per night: the session
of the latest succeeded legacy ``score`` job. Contract:
``engine/v2/models/ARCHITECTURE.md`` section 2.2.1. Called from ``Service.tick()``;
never a GRAPH node; the Service sidecar catches and reports any failure. Reads
durable state only (no memo): each phase key is checked against the ``jobs`` table
every call, and phase N+1 is submitted only once every phase-N job has succeeded.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from engine.v2.foundation import ArtifactError, content_hash, from_document
from engine.v2.models import deployment
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state
from engine.v2.ops import nightly, state_stage, training
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import worker_source_manifest
from engine.v2.ops.plans import request_from_plan
from engine.v2.ops.submission import job_id_for, submit

PHASES = ("train-paired", "train-board", "train-trailing", "stage", "promote")
TRAINING_PHASES = PHASES[:3]

FILL_ALPHA = 0.5  # engine.fills.MID.alpha, the shadow fill alpha native_score_batch scores with

#: Largest ``spec_json`` a score job may carry to still be considered (a bound on
#: the per-tick read; anything larger is ignored rather than parsed).
MAX_SPEC_BYTES = 256 * 1024

#: Newest ``nightly:`` job rows the inner identity query visits per tick, before
#: the residual kind/state/spec-size filters run. With about ten job rows per
#: night this is roughly 25 nights; a score job older than this window is never
#: found.
MAX_SCAN_ROWS = 256

#: Most succeeded score-job rows the outer identity query keeps per tick. Score
#: keys are ``nightly:<ISO session>:<scope>:score``, so descending key order
#: lists the newest session first and a bounded read of the latest rows is
#: enough.
MAX_SCORE_ROWS = 32

#: Identity selection query. The inner query visits at most ``MAX_SCAN_ROWS``
#: newest ``nightly:`` keys (a range SEARCH on the (namespace, idempotency_key)
#: index), so the residual filters never walk unbounded history; the outer query
#: re-applies the key order and keeps at most ``MAX_SCORE_ROWS`` rows.
IDENTITY_SQL = (
    "SELECT idempotency_key, spec_json, created_at FROM ("
    "SELECT idempotency_key, kind, state, spec_json, created_at FROM jobs "
    "WHERE namespace = 'shadow' "
    "AND idempotency_key >= 'nightly:' AND idempotency_key < 'nightly;' "
    "ORDER BY idempotency_key DESC LIMIT ?) "
    "WHERE kind = ? AND state = 'succeeded' AND length(spec_json) <= ? "
    "ORDER BY idempotency_key DESC LIMIT ?"
)

#: Largest ``DEPLOYED`` pointer file that may be read as a regular file.
MAX_POINTER_BYTES = 64 * 1024


def manifest_tag(manifest_id):
    return hashlib.sha256(manifest_id.encode()).hexdigest()[:8]


def phase_key(session, phase, prior, tag):
    return f"nightly:{session}:pool:{phase}:{tag}:{prior}"


def pool_refresh_identity(conn):
    """``(session, manifest_id)`` from the latest succeeded legacy ``score`` job,
    or ``None``. Mirrors ``nightly._native_score_batch_identity``: max by
    (session, created_at, scope_hash) over parseable keys with a usable
    ``created_at``; ``manifest_id`` is that job's ``legacy_manifest.json`` binding.
    Returns ``None`` when no score job exists or its binding is absent/not a string."""
    # Bounded per-tick read: IDENTITY_SQL's inner query visits at most
    # MAX_SCAN_ROWS newest ``nightly:`` keys (the ``idempotency_key`` range pins
    # the scan to the ``nightly:`` key space -- ``:`` is followed by ``;`` in
    # ASCII -- so the primary-key index is used as a range, never a full scan of
    # every ``shadow`` key), then the outer query filters and keeps the newest
    # MAX_SCORE_ROWS. The Python max-selection below re-ranks those rows by
    # created_at.
    rows = conn.execute(
        IDENTITY_SQL,
        (MAX_SCAN_ROWS, nightly.LEGACY_SCORE_KIND, MAX_SPEC_BYTES,
         MAX_SCORE_ROWS)).fetchall()
    candidates = []
    for row in rows:
        parsed = nightly.session_scope_from_score_key(row["idempotency_key"])
        if parsed is None:
            continue
        created_at = row["created_at"]
        if not created_at:
            continue
        candidates.append((parsed[0], created_at, parsed[1], row["spec_json"]))
    if not candidates:
        return None
    session, _, _scope, spec_json = max(candidates, key=lambda item: item[:3])
    try:
        document = json.loads(spec_json)
    except ValueError:
        return None
    if not isinstance(document, Mapping):
        return None
    parameters = document.get("parameters") or {}
    manifest_id = (parameters.get("input_bindings") or {}).get("legacy_manifest.json")
    if not isinstance(manifest_id, str) or not manifest_id:
        return None
    return session, manifest_id


def _exists(conn, key):
    return conn.execute(
        "SELECT 1 FROM jobs WHERE job_id = ?",
        (job_id_for("shadow", key),)).fetchone() is not None


def _state(conn, key):
    row = conn.execute(
        "SELECT state FROM jobs WHERE job_id = ?",
        (job_id_for("shadow", key),)).fetchone()
    return None if row is None else row["state"]


def _training_plans(session, manifest_id, phases):
    builders = {
        "train-paired": lambda: training.training_plan(
            mode="state", state="paired_residual_pool", cutoffs=(session,),
            manifest_ref=manifest_id),
        "train-board": lambda: training.training_plan(
            mode="board_analog", alpha=FILL_ALPHA, cutoffs=(session,),
            manifest_ref=manifest_id),
        "train-trailing": lambda: training.training_plan(
            mode="trailing_cutoff", cutoffs=(session,), manifest_ref=manifest_id),
    }
    return {phase: builders[phase]() for phase in phases}


def _implementation_ref(code_source):
    return content_hash(worker_source_manifest(code_source))


def _read_pointer(deployment_root):
    """Read the ``DEPLOYED`` pointer once, or ``None`` when it is absent.

    Reads the file exactly once: a symlink, FIFO, directory or oversize
    regular file is refused before any content read, so a planted pointer
    cannot steer or exhaust the per-tick read, and bytes that are not a
    decodable ``PointerState`` document are refused too. A missing pointer
    is the only benign outcome -- it means nothing has ever been promoted."""
    try:
        data = read_bounded_state(Path(deployment_root) / "DEPLOYED",
                                  MAX_POINTER_BYTES, no_follow=True)
    except FileNotFoundError:
        return None
    except (BoundedStateReadError, OSError):
        raise fail("VALIDATION_FAILED", "deployment pointer is not a bounded regular file")
    try:
        return from_document(deployment.PointerState, json.loads(data.decode("utf-8")))
    except (ValueError, TypeError, ArtifactError):
        raise fail("VALIDATION_FAILED", "deployment pointer could not be decoded")


def submit_pool_nightly_refresh_if_ready(conn, registry, policy, store, release_root, *,
                                         code_source, clock):
    """Submit at most the ready phase of the pool-nightly cycle; returns this call's
    JobReceipts (empty = waiting/idle). ``release_root`` is the store's parent; the
    deployment dir lives under ``<release_root>/deployment``."""
    if conn is None:
        return []
    identity = pool_refresh_identity(conn)
    if identity is None:
        return []
    session, manifest_id = identity
    tag = manifest_tag(manifest_id)
    pointer = _read_pointer(Path(release_root) / "deployment")
    if pointer is None:
        return []
    prior = pointer.release_id
    release_id = f"pools-{session}-{tag}"
    if prior == release_id:
        return []

    missing = [phase for phase in TRAINING_PHASES
               if not _exists(conn, phase_key(session, phase, prior, tag))]
    if missing:
        built = _training_plans(session, manifest_id, missing)
        receipts = []
        impl = _implementation_ref(code_source)
        for phase in missing:
            plan = built[phase]
            plan["implementation_ref"] = impl
            request = request_from_plan(plan, phase_key(session, phase, prior, tag))
            receipts.append(submit(conn, registry, policy, request, clock=clock))
        return receipts
    if any(_state(conn, phase_key(session, phase, prior, tag)) != "succeeded"
           for phase in TRAINING_PHASES):
        return []

    stage_key = phase_key(session, "stage", prior, tag)
    if not _exists(conn, stage_key):
        state_refs = _state_refs(conn, session, prior, tag, code_source)
        plan = state_stage.state_stage_plan(
            release_root=str(release_root), prior_release_id=prior,
            release_id=release_id, cutoff=session, state_refs=state_refs,
            source_root=code_source)
        return [submit(conn, registry, policy,
                       request_from_plan(plan, stage_key), clock=clock)]
    if _state(conn, stage_key) != "succeeded":
        return []

    promote_key = phase_key(session, "promote", prior, tag)
    if _exists(conn, promote_key):
        return []
    plan = training.promote_plan(
        release_root=str(Path(release_root) / "deployment"),
        release_id=release_id, expected_previous_release_id=prior)
    plan["implementation_ref"] = _implementation_ref(code_source)
    return [submit(conn, registry, policy,
                   request_from_plan(plan, promote_key), clock=clock)]


def _state_refs(conn, session, prior, tag, code_source):
    """Latest succeeded attempt's plain ``training/*.json`` outputs across the three
    training jobs, keyed by file remainder (``paired.json`` etc.). A duplicate name
    or an empty set is ``VALIDATION_FAILED``; summaries (``*.summary.json``) and
    non-``.json`` names are excluded."""
    state_refs = {}
    for phase in TRAINING_PHASES:
        key = phase_key(session, phase, prior, tag)
        job_id = job_id_for("shadow", key)
        attempt = conn.execute(
            "SELECT attempt_id FROM attempts WHERE job_id = ? AND state = 'succeeded' "
            "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
        if attempt is None:
            raise fail("VALIDATION_FAILED", "a training job has no succeeded attempt",
                       details={"job_id": job_id})
        for name, _artifact in conn.execute(
                "SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id = ?",
                (attempt["attempt_id"],)).fetchall():
            if not name.startswith("training/") or not name.endswith(".json"):
                continue
            if name.endswith(".summary.json"):
                continue
            remainder = name[len("training/"):]
            if remainder in state_refs:
                raise fail("VALIDATION_FAILED", "a state file name appears twice",
                           details={"name": remainder})
            state_refs[remainder] = f"{job_id}#{name}"
    if not state_refs:
        raise fail("VALIDATION_FAILED", "no training state files to stage")
    return state_refs