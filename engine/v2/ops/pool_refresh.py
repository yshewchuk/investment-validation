"""Pool-nightly refresh sidecar (#192 slice 4; #1102 slice C bootstrap).

Advances the three ``phase5_state_stage`` pool members (``paired_residual_pool``,
``board_analog_matcher``, ``trailing_pnl_cutoff``) one cutoff per night: the session
of the latest succeeded legacy ``finality`` or ``features`` job. When the pointer's
release classifies ``NONE``/``NO_POOL_CATALOG`` it runs the bootstrap cycle instead
(the 12 training phases, a ``mode: "bootstrap"`` stage and an empty-store or
expected-incumbent guarded promote); a ``DAMAGED`` prior submits nothing and reports
``VALIDATION_FAILED``. Contract: ``engine/v2/models/ARCHITECTURE.md`` sections 2.2.1
and 2.2.2. Called from ``Service.tick()``; never a GRAPH node; the Service sidecar
catches and reports any failure. Reads durable state only (no memo): each phase key
is checked against the ``jobs`` table every call, and phase N+1 is submitted only
once every phase-N job has succeeded.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

from engine.v2.foundation import ArtifactError, content_hash, from_document
from engine.v2.models import deployment
from engine.v2.models.frozen_state_io import BoundedStateReadError, read_bounded_state
from engine.v2.ops import state_stage, training
from engine.v2.ops.errors import fail
from engine.v2.ops.fingerprints import worker_source_manifest
from engine.v2.ops.plans import request_from_plan
from engine.v2.ops.submission import job_id_for, submit

PHASES = ("train-paired", "train-board", "train-trailing", "stage", "promote")
TRAINING_PHASES = PHASES[:3]

#: The bootstrap cycle's training phases (#1102 slice C): the three carried pools,
#: the three driver residual pools, the chooser analog pool and the six calibration
#: recipes; then ``stage`` and ``promote``.
BOOTSTRAP_TRAINING_PHASES = (
    "train-paired", "train-board", "train-trailing",
    "train-driver-size", "train-driver-implied_t1", "train-driver-runup_move",
    "train-chooser",
    "train-line-thru", "train-map-thru",
    "train-surface-runup", "train-line-runup", "train-map-runup",
)
BOOTSTRAP_PHASES = BOOTSTRAP_TRAINING_PHASES + ("stage", "promote")

#: The three calibration phases keyed at the STR-RUNUP entry cutoff.
_RUNUP_PHASES = frozenset({"train-surface-runup", "train-line-runup",
                           "train-map-runup"})

#: The STR-RUNUP calibration key cutoff: the session's date this many US trading
#: days earlier (legacy's evidence cutoff is the entry date, 14 trading days before
#: the decision date; ``min(entry, session)`` as in ``tools/phase5_calibration_keys.py``).
RUNUP_ENTRY_TRADING_DAYS = 14

#: Calendar window guaranteed to contain ``RUNUP_ENTRY_TRADING_DAYS`` trading days
#: (about four weeks plus every possible holiday in the window).
RUNUP_ENTRY_WINDOW_DAYS = 70

#: Bootstrap assembly inputs: environment variable name -> ``bootstrap`` key, read
#: at submit time (never at import time).
_BOOTSTRAP_ENV = (
    ("BOOTSTRAP_TIER4_DIR", "tier4_dir"),
    ("BOOTSTRAP_PANEL_CATALOG", "panel_catalog"),
    ("BOOTSTRAP_PANEL_SNAPSHOT_ID", "panel_snapshot_id"),
    ("BOOTSTRAP_PANEL_ARTIFACT_ROOT", "panel_artifact_root"),
)

#: The prior classifier, imported lazily by ``_classify_prior`` from the
#: ``tools.phase5_state_stage_job`` re-export (``engine/v2`` may not import
#: ``checks`` at all, so the classification crosses through the tools layer);
#: ``None`` until then, and tests may patch this attribute to a stub.
classify_prior = None

FILL_ALPHA = 0.5  # engine.fills.MID.alpha, the shadow fill alpha native_score_batch scores with

#: Largest ``spec_json`` a finality or features job may carry to still be
#: considered (a bound on the per-tick read; anything larger is ignored rather
#: than parsed).
MAX_SPEC_BYTES = 256 * 1024

#: Newest ``nightly:`` job rows the inner identity query visits per tick, before
#: the residual kind/state/spec-size filters run. With about ten job rows per
#: night this is roughly 25 nights; a finality or features job older than this
#: window is never found.
MAX_SCAN_ROWS = 256

#: Most succeeded finality/features job rows the outer identity query keeps per
#: tick. Their keys are ``nightly:<ISO session>:<scope>:<stage>``, so descending
#: key order lists the newest session first and a bounded read of the latest
#: rows is enough.
MAX_SCORE_ROWS = 32

#: The legacy nightly stages that keep the plan's ``legacy_manifest.json``
#: binding in snapshot mode (``score`` does not), so their succeeded jobs carry
#: the session's scoring manifest.
_FINALITY_KIND = "legacy_finality"
_FEATURES_KIND = "legacy_features"
_STAGE_NAMES = ("finality", "features")

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
    "WHERE kind IN (?, ?) AND state = 'succeeded' AND length(spec_json) <= ? "
    "ORDER BY idempotency_key DESC LIMIT ?"
)

#: Largest ``DEPLOYED`` pointer file that may be read as a regular file.
MAX_POINTER_BYTES = 64 * 1024


def manifest_tag(manifest_id):
    return hashlib.sha256(manifest_id.encode()).hexdigest()[:8]


def phase_key(session, phase, prior, tag):
    return f"nightly:{session}:pool:{phase}:{tag}:{prior}"


def _log(message):
    """One stderr line (stdout stays for machine-readable results)."""
    print(f"[pool] {message}", file=sys.stderr)


def _classify_prior(release_root, prior_id):
    """Classify the pointer's release with the stage job's own classifier
    (``checks.phase5_release_assembly.classify_prior`` via the sanctioned
    ``tools.phase5_state_stage_job`` re-export; ``engine/v2`` may not import
    ``checks`` directly) using the stage job's argument shape: the first arg is the store root (this
    sidecar's ``release_root``) and a missing pointer passes ``None`` (which the
    classifier reports as ``NONE``). The import is lazy."""
    fn = classify_prior
    if fn is None:
        from tools.phase5_state_stage_job import classify_prior as fn
    return fn(Path(release_root), prior_id)


def runup_entry_date(session):
    """The session's date minus ``RUNUP_ENTRY_TRADING_DAYS`` US trading days,
    counted strictly before the session (so a session that is not a trading day
    still keys the entry of its latest prior trading day, bounded by
    ``min(entry, session)`` at the caller). Trading days are
    ``engine.v2.data.computed_moves.projected_trading_days`` weekdays minus
    ``us_market_holidays`` (the verbatim v2 twin of ``engine.calendar``'s helpers;
    ``engine/v2`` may not import the legacy module)."""
    from engine.v2.data.computed_moves import projected_trading_days

    end = date.fromisoformat(session) - timedelta(days=1)
    days = projected_trading_days(end - timedelta(days=RUNUP_ENTRY_WINDOW_DAYS), end)
    if len(days) < RUNUP_ENTRY_TRADING_DAYS:
        raise fail("VALIDATION_FAILED", "could not resolve the STR-RUNUP entry date",
                   details={"session": session})
    return str(days[-RUNUP_ENTRY_TRADING_DAYS].date())


def _session_scope_from_stage_key(idempotency_key):
    """Recover ``(session, scope_hash)`` from a legacy ``finality`` or ``features``
    job's own idempotency key (``"nightly:<session>:<scope_hash>:<stage>"``).

    Mirrors ``nightly._session_scope_from_score_key``: ``partition``/``rpartition``
    rather than a fixed ``split(":")`` count, because ``scope_hash`` itself is a
    content-hash string (``"sha256:<hex>"``) that already contains its own colon.
    Only the ``"nightly"`` prefix, the session field and the trailing stage suffix
    are structural; ``stage`` must be exactly ``"finality"`` or ``"features"``.
    Returns ``None`` for anything that does not match this exact shape."""
    prefix, sep1, rest = idempotency_key.partition(":")
    if prefix != "nightly" or not sep1:
        return None
    session, sep2, remainder = rest.partition(":")
    if not sep2 or not session:
        return None
    scope_hash, sep3, stage = remainder.rpartition(":")
    if not sep3 or stage not in _STAGE_NAMES or not scope_hash:
        return None
    return session, scope_hash


def _manifest_id_from_spec(document):
    """``document``'s ``legacy_manifest.json`` binding, or ``None`` when either
    nested level is absent or is not a mapping (a non-mapping ``parameters`` or
    ``input_bindings`` is refused rather than probed with ``.get``)."""
    parameters = document.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        return None
    bindings = parameters.get("input_bindings") or {}
    if not isinstance(bindings, Mapping):
        return None
    manifest_id = bindings.get("legacy_manifest.json")
    if not isinstance(manifest_id, str) or not manifest_id:
        return None
    return manifest_id


def pool_refresh_identity(conn):
    """``(session, manifest_id)`` from the latest succeeded legacy ``finality``
    or ``features`` job, or ``None``. Max by (session, created_at, scope_hash)
    over parseable keys with a usable ``created_at``; ``manifest_id`` is that
    job's ``legacy_manifest.json`` binding. Returns ``None`` when no such job
    exists or its binding is absent/not a string."""
    # Bounded per-tick read: IDENTITY_SQL's inner query visits at most
    # MAX_SCAN_ROWS newest ``nightly:`` keys (the ``idempotency_key`` range pins
    # the scan to the ``nightly:`` key space -- ``:`` is followed by ``;`` in
    # ASCII -- so the primary-key index is used as a range, never a full scan of
    # every ``shadow`` key), then the outer query filters and keeps the newest
    # MAX_SCORE_ROWS. The Python max-selection below re-ranks those rows by
    # created_at.
    rows = conn.execute(
        IDENTITY_SQL,
        (MAX_SCAN_ROWS, _FINALITY_KIND, _FEATURES_KIND, MAX_SPEC_BYTES,
         MAX_SCORE_ROWS)).fetchall()
    candidates = []
    for row in rows:
        parsed = _session_scope_from_stage_key(row["idempotency_key"])
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
    manifest_id = _manifest_id_from_spec(document)
    if manifest_id is None:
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
    if any(phase in _RUNUP_PHASES for phase in phases):
        runup_cutoff = min(runup_entry_date(session), session)
    else:
        runup_cutoff = session  # unused: no STR-RUNUP phase was requested
    builders = {
        "train-paired": lambda: training.training_plan(
            mode="state", state="paired_residual_pool", cutoffs=(session,),
            manifest_ref=manifest_id),
        "train-board": lambda: training.training_plan(
            mode="board_analog", alpha=FILL_ALPHA, cutoffs=(session,),
            manifest_ref=manifest_id),
        "train-trailing": lambda: training.training_plan(
            mode="trailing_cutoff", cutoffs=(session,), manifest_ref=manifest_id),
        "train-driver-size": lambda: training.training_plan(
            mode="state", state="driver_residual_pool:size", manifest_ref=manifest_id),
        "train-driver-implied_t1": lambda: training.training_plan(
            mode="state", state="driver_residual_pool:implied_t1",
            manifest_ref=manifest_id),
        "train-driver-runup_move": lambda: training.training_plan(
            mode="state", state="driver_residual_pool:runup_move",
            manifest_ref=manifest_id),
        "train-chooser": lambda: training.training_plan(
            mode="state", state="chooser_analog_pool", cutoffs=(session,),
            manifest_ref=manifest_id),
        "train-line-thru": lambda: training.training_plan(
            mode="recipe", recipe="payoff_line:STR-THRU:calibration",
            alpha=FILL_ALPHA, cutoffs=(session,), manifest_ref=manifest_id),
        "train-map-thru": lambda: training.training_plan(
            mode="recipe", recipe="recalibration_map:STR-THRU:calibration",
            alpha=FILL_ALPHA, cutoffs=(session,), manifest_ref=manifest_id),
        "train-surface-runup": lambda: training.training_plan(
            mode="recipe", recipe="payoff_surface:STR-RUNUP:calibration",
            alpha=FILL_ALPHA, cutoffs=(runup_cutoff,), manifest_ref=manifest_id),
        "train-line-runup": lambda: training.training_plan(
            mode="recipe", recipe="payoff_line:STR-RUNUP:calibration",
            alpha=FILL_ALPHA, cutoffs=(runup_cutoff,), manifest_ref=manifest_id),
        "train-map-runup": lambda: training.training_plan(
            mode="recipe", recipe="recalibration_map:STR-RUNUP:calibration",
            alpha=FILL_ALPHA, cutoffs=(runup_cutoff,), manifest_ref=manifest_id),
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


#: Phase states that end the cycle's wait for operator recovery (#154).
_STALLED_STATES = frozenset({"failed", "cancelled", "blocked"})
_TERMINAL_STATES = _STALLED_STATES | {"succeeded"}

#: Bounded lookup for a promote attempt whose key embeds a ``prior`` that is no
#: longer readable from the pointer (the pointer already names the candidate).
#: An explicit binary key range rather than ``LIKE`` (case-insensitive in
#: SQLite, so it cannot use the (namespace, idempotency_key) index as a range):
#: exactly the keys prefixed ``nightly:<session>:pool:promote:<tag>:`` are
#: covered, since ``:`` is followed by ``;`` in ASCII.
PROMOTE_LOOKUP_SQL = (
    "SELECT state FROM jobs WHERE namespace = 'shadow' "
    "AND idempotency_key >= ? AND idempotency_key < ? "
    "ORDER BY idempotency_key DESC LIMIT 8"
)


def _promote_states_with_unknown_prior(conn, session, tag):
    rows = conn.execute(
        PROMOTE_LOOKUP_SQL,
        (f"nightly:{session}:pool:promote:{tag}:",
         f"nightly:{session}:pool:promote:{tag};")).fetchall()
    return [row["state"] for row in rows]


def _pointer_named_candidate_states(conn, session, tag):
    """Classify the promote attempts when the pointer already names the candidate
    release: ``"stalled"`` when an attempt is failed/cancelled/blocked,
    ``"active"`` while one has not reached a terminal state, else ``"clear"``."""
    states = _promote_states_with_unknown_prior(conn, session, tag)
    if any(state in _STALLED_STATES for state in states):
        return "stalled"
    if any(state not in _TERMINAL_STATES for state in states):
        return "active"
    return "clear"


def _cycle_job_states(conn, session, prior, tag, release_id, phases=PHASES):
    """Classify this cycle's phase jobs (the carry five by default, the bootstrap
    fourteen): ``"stalled"`` when one is failed/cancelled/blocked, ``"active"``
    while an existing phase job has not succeeded, else ``"clear"``."""
    if prior == release_id:
        return _pointer_named_candidate_states(conn, session, tag)
    states = [_state(conn, phase_key(session, phase, prior, tag)) for phase in phases]
    if any(state in _STALLED_STATES for state in states):
        return "stalled"
    if any(state is not None and state != "succeeded" for state in states):
        return "active"
    return "clear"


def pool_cycle_pending(conn, release_root):
    """Is the latest session's pool cycle still pending?

    ``True`` when a candidate release for the identity session is not yet
    deployed and no phase job of this cycle has failed or stalled; also
    ``True`` while an existing phase job (including the promote attempt) has
    not succeeded, even if the pointer already names the candidate release:
    the promote must be drained before the serve closes the worker. A missing
    pointer or a ``NO_POOL_CATALOG`` prior is checked against the bootstrap
    cycle's key set (prior token ``none`` for a missing pointer), so the serve
    stays up while a bootstrap cycle is active and not stalled; a missing
    pointer with no bootstrap job running stays ``False`` (nothing has started).
    A ``DAMAGED`` prior, an unreadable pointer, no connection, no identity or
    a classification failure is ``False``. Reads only the same durable state as
    the sidecar, never raises and never writes."""
    if conn is None:
        return False
    identity = pool_refresh_identity(conn)
    if identity is None:
        return False
    session, manifest_id = identity
    tag = manifest_tag(manifest_id)
    release_id = f"pools-{session}-{tag}"
    try:
        from tools.phase5_state_stage_job import PRIOR_DAMAGED, PRIOR_NO_POOL_CATALOG, PRIOR_NONE
        pointer = _read_pointer(Path(release_root) / "deployment")
        if pointer is not None and pointer.release_id == release_id:
            # The cycle is done except for draining its promote attempt; the
            # candidate is never the prior to classify.
            return _pointer_named_candidate_states(conn, session, tag) == "active"
        prior = _classify_prior(Path(release_root),
                                None if pointer is None else pointer.release_id)
    except Exception:
        return False
    if pointer is None:
        if prior.kind != PRIOR_NONE:
            return False
        return _cycle_job_states(conn, session, "none", tag, release_id,
                                 phases=BOOTSTRAP_PHASES) == "active"
    if prior.kind == PRIOR_DAMAGED:
        return False
    prior_id = pointer.release_id
    if prior.kind in (PRIOR_NONE, PRIOR_NO_POOL_CATALOG):
        job_states = _cycle_job_states(conn, session, prior_id, tag, release_id,
                                       phases=BOOTSTRAP_PHASES)
    else:
        job_states = _cycle_job_states(conn, session, prior_id, tag, release_id)
    if job_states == "stalled":
        return False
    if job_states == "active":
        return True
    return prior_id != release_id


def submit_pool_nightly_refresh_if_ready(conn, registry, policy, store, release_root, *,
                                         code_source, clock):
    """Submit at most the ready phase of the pool-nightly cycle; returns this call's
    JobReceipts (empty = waiting/idle). ``release_root`` is the store's parent; the
    deployment dir lives under ``<release_root>/deployment``. The pointer's release
    is classified every tick (§2.2.2): ``USABLE`` runs the carry cycle, ``NONE``
    (a missing pointer) and ``NO_POOL_CATALOG`` run the bootstrap cycle, and
    ``DAMAGED`` submits nothing and raises for the sidecar's deduped report."""
    if conn is None:
        return []
    identity = pool_refresh_identity(conn)
    if identity is None:
        return []
    session, manifest_id = identity
    tag = manifest_tag(manifest_id)
    pointer = _read_pointer(Path(release_root) / "deployment")
    prior_id = None if pointer is None else pointer.release_id
    release_id = f"pools-{session}-{tag}"
    if prior_id == release_id:
        return []
    from tools.phase5_state_stage_job import PRIOR_DAMAGED, PRIOR_NO_POOL_CATALOG, PRIOR_NONE
    classified = _classify_prior(Path(release_root), prior_id)
    if classified.kind == PRIOR_DAMAGED:
        _log_prior(session, tag, classified)
        raise fail("VALIDATION_FAILED", "prior release could not be read",
                   details={"code": "PRIOR_MISMATCH"})
    if classified.kind in (PRIOR_NONE, PRIOR_NO_POOL_CATALOG):
        return _submit_bootstrap(conn, registry, policy, store, release_root, session,
                                 manifest_id, tag, prior_id, classified,
                                 code_source=code_source, clock=clock)
    if prior_id is None:
        # Unreachable with the real classifier (a missing pointer is NONE); keeps
        # a stubbed classifier from keying the carry cycle on a "None" prior token.
        return []
    return _submit_carry(conn, registry, policy, release_root, session, manifest_id,
                         tag, prior_id, release_id, classified,
                         code_source=code_source, clock=clock)


def _submit_carry(conn, registry, policy, release_root, session, manifest_id, tag,
                  prior, release_id, prior_class, *, code_source, clock):
    """The existing carry cycle (§2.2.1), keyed by the usable prior release id."""
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
        _log_prior(session, tag, prior_class)
        _log(f"{session} {tag} carry submitted {len(missing)}/"
             f"{len(TRAINING_PHASES)} training jobs")
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
            manifest_ref=manifest_id, source_root=code_source)
        _log_prior(session, tag, prior_class)
        _log(f"{session} {tag} carry submitted 1 stage job")
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
    _log_prior(session, tag, prior_class)
    _log(f"{session} {tag} carry submitted 1 promote job")
    return [submit(conn, registry, policy,
                   request_from_plan(plan, promote_key), clock=clock)]


def _log_prior(session, tag, prior):
    """The cycle's prior class and reason, logged once on every tick that acts on
    it (a purely waiting tick stays silent)."""
    _log(f"{session} {tag} prior class {prior.kind}: {prior.reason}")


def _bootstrap_inputs():
    """The four bootstrap assembly inputs, read from the environment at submit
    time; any unset or blank value refuses ``VALIDATION_FAILED`` before any
    training job is submitted (a cycle that cannot stage must not spend training)."""
    inputs = {}
    for env_name, key in _BOOTSTRAP_ENV:
        value = (os.environ.get(env_name) or "").strip()
        if not value:
            raise fail("VALIDATION_FAILED",
                       "bootstrap panel/tier4 inputs are not configured")
        inputs[key] = value
    return inputs


def _submit_bootstrap(conn, registry, policy, store, release_root, session, manifest_id,
                      tag, prior_id, prior, *, code_source, clock):
    """The bootstrap cycle (#1102 slice C): the twelve training phases (all
    missing ones in one tick, like the carry cycle), then one ``mode: "bootstrap"``
    stage, then the guarded promote -- each phase only once every earlier job has
    succeeded."""
    bootstrap = _bootstrap_inputs()
    prior_token = "none" if prior_id is None else prior_id
    release_id = f"pools-{session}-{tag}"

    missing = [phase for phase in BOOTSTRAP_TRAINING_PHASES
               if not _exists(conn, phase_key(session, phase, prior_token, tag))]
    if missing:
        built = _training_plans(session, manifest_id, missing)
        receipts = []
        impl = _implementation_ref(code_source)
        for phase in missing:
            plan = built[phase]
            plan["implementation_ref"] = impl
            request = request_from_plan(plan, phase_key(session, phase, prior_token, tag))
            receipts.append(submit(conn, registry, policy, request, clock=clock))
        _log_prior(session, tag, prior)
        _log(f"{session} {tag} bootstrap submitted {len(missing)}/"
             f"{len(BOOTSTRAP_TRAINING_PHASES)} training jobs")
        return receipts
    if any(_state(conn, phase_key(session, phase, prior_token, tag)) != "succeeded"
           for phase in BOOTSTRAP_TRAINING_PHASES):
        return []

    stage_key = phase_key(session, "stage", prior_token, tag)
    if not _exists(conn, stage_key):
        state_refs = _state_refs(conn, session, prior_token, tag, code_source,
                                 phases=BOOTSTRAP_TRAINING_PHASES)
        plan = state_stage.state_stage_plan(
            release_root=str(release_root), prior_release_id=prior_id,
            release_id=release_id, cutoff=session, state_refs=state_refs,
            manifest_ref=manifest_id, source_root=code_source,
            mode="bootstrap", bootstrap=bootstrap)
        _log_prior(session, tag, prior)
        _log(f"{session} {tag} bootstrap submitted 1 stage job")
        return [submit(conn, registry, policy,
                       request_from_plan(plan, stage_key), clock=clock)]
    if _state(conn, stage_key) != "succeeded":
        return []

    promote_key = phase_key(session, "promote", prior_token, tag)
    if _exists(conn, promote_key):
        return []
    promote_kwargs = {"release_root": str(Path(release_root) / "deployment"),
                      "release_id": release_id}
    if prior_id is None:
        promote_kwargs["require_empty_store"] = True
    else:
        promote_kwargs["expected_previous_release_id"] = prior_id
    plan = training.promote_plan(**promote_kwargs)
    plan["implementation_ref"] = _implementation_ref(code_source)
    _log_prior(session, tag, prior)
    _log(f"{session} {tag} bootstrap submitted 1 promote job")
    return [submit(conn, registry, policy,
                   request_from_plan(plan, promote_key), clock=clock)]


def _state_refs(conn, session, prior, tag, code_source, phases=TRAINING_PHASES):
    """Latest succeeded attempt's plain ``training/*.json`` outputs across the given
    training phases (the carry cycle's three by default, the bootstrap cycle's
    twelve), keyed by file remainder (``paired.json`` etc.). A duplicate name
    or an empty set is ``VALIDATION_FAILED``; summaries (``*.summary.json``) and
    non-``.json`` names are excluded."""
    state_refs = {}
    for phase in phases:
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