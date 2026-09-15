"""D15 loading: committed ``score.json`` documents, target consistency, and
the snapshot binding a snapshot-mode ``legacy_score`` job recorded at launch.

The actual field comparison lives at ``checks/rearchitecture_phase2_parity.py``,
which may import ``engine.v2.diagnosis`` — a declared *sink* layer (7.5) that
no other v2 package may import (``checks/layer_map.py``, enforced by
``checks/import_layers.py`` inside the Phase 1 gate's "imports" row): a
comparator must never become a dependency of the thing it compares. This
module (layer 7.0, ``engine.v2.ops``) therefore never re-derives field
comparison or tolerance logic — it only loads and returns plain rows plus
bindings, so the one real comparator (``compare_records``, under its one
declared tolerance policy) is the only place that can drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from engine.v2.contracts import JobSpec
from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.documents import decode_document
from engine.v2.ops.catalog import load_json
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.errors import fail
from engine.v2.ops.input_bindings import recorded_bindings

__all__ = ["ScoreParityInputs", "load_score_parity_inputs"]

_OUTPUT_NAME = "legacy_score"


@dataclass(frozen=True)
class ScoreParityInputs:
    """Everything a D15 comparator needs: two row sets keyed by ``row_id``,
    plus the snapshot job's bound ``SnapshotRef`` (``None`` when it recorded
    no ``snapshot_ref.json`` binding at launch)."""

    legacy_job_id: str
    snapshot_job_id: str
    legacy_rows: dict[str, dict]
    snapshot_rows: dict[str, dict]
    snapshot_ref: SnapshotRef | None


def _succeeded_attempt(conn, job_id: str) -> str:
    row = conn.execute(
        "SELECT attempt_id FROM attempts WHERE job_id=? AND state='succeeded' "
        "ORDER BY attempt_number DESC LIMIT 1", (job_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "job has no succeeded attempt", details={"job_id": job_id})
    return row[0]


def _committed_score(conn, store, job_id: str) -> tuple[dict, str]:
    attempt_id = _succeeded_attempt(conn, job_id)
    row = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=? AND name=?",
                       (attempt_id, _OUTPUT_NAME)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "job did not commit a score artifact", details={"job_id": job_id})
    ref = artifact(conn, store, row[0])
    return json.loads(store.read_verified(ref)), attempt_id


def _job_parameters(conn, job_id: str) -> dict:
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if row is None:
        raise fail("INPUT_CHANGED", "unknown job", details={"job_id": job_id})
    return load_json(JobSpec, row[0]).parameters


def _snapshot_binding(conn, store, attempt_id: str) -> SnapshotRef | None:
    """The ``snapshot_ref.json`` bound at launch, if this attempt is snapshot-backed."""
    row = recorded_bindings(conn, attempt_id).get("snapshot_ref.json")
    if row is None:
        return None
    ref = artifact(conn, store, row.artifact_id)
    return decode_document(SnapshotRef, json.loads(store.read_verified(ref)))


def _rows_by_id(doc: dict) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for row in doc.get("rows", []):
        row_id = row.get("row_id")
        if not row_id or row_id in rows:
            raise fail("VALIDATION_FAILED", "score document has a missing or duplicate row_id")
        rows[row_id] = row
    return rows


def _verify_distinct_jobs(legacy_job_id: str, snapshot_job_id: str) -> None:
    """Refuse the reviewer's self-parity repro: passing the SAME job as both
    the legacy and the snapshot side trivially agrees with itself (identical
    rows, identical key sets) over a nonzero population, certifying nothing.
    """
    if legacy_job_id == snapshot_job_id:
        raise fail("INPUT_CHANGED",
                   "legacy and snapshot score jobs must be two different jobs",
                   details={"reason": "SAME_JOB", "job_id": legacy_job_id})


def _verify_execution_modes(legacy_params: dict, snapshot_params: dict) -> None:
    """The legacy side must actually be the barrier path (``input_mode``
    absent or ``"legacy"``) and the snapshot side must actually be the
    adapted path (``input_mode == "snapshot"``) -- a D15 receipt is
    legacy-vs-adapter parity by definition, so two jobs run in the same mode
    (both legacy, or both snapshot) are not a legacy/snapshot comparison at
    all, regardless of what job ids were passed for which argument.
    """
    legacy_mode = legacy_params.get("input_mode") or "legacy"
    if legacy_mode != "legacy":
        raise fail("INPUT_CHANGED",
                   "legacy_job_id did not run in legacy input mode",
                   details={"reason": "LEGACY_MODE_MISMATCH", "input_mode": legacy_mode})
    snapshot_mode = snapshot_params.get("input_mode") or "legacy"
    if snapshot_mode != "snapshot":
        raise fail("INPUT_CHANGED",
                   "snapshot_job_id did not run in snapshot input mode",
                   details={"reason": "SNAPSHOT_MODE_MISMATCH", "input_mode": snapshot_mode})


def _verify_same_target(legacy_doc, snapshot_doc, legacy_params, snapshot_params) -> None:
    legacy_target = (legacy_doc.get("session"), tuple(sorted(legacy_doc.get("tickers") or ())),
                     legacy_params.get("horizon_days", 35), legacy_params.get("effect_scope") or "")
    snapshot_target = (snapshot_doc.get("session"), tuple(sorted(snapshot_doc.get("tickers") or ())),
                       snapshot_params.get("horizon_days", 35),
                       snapshot_params.get("effect_scope") or "")
    if legacy_target != snapshot_target:
        raise fail("INPUT_CHANGED",
                   "legacy and snapshot score jobs do not target the same resolved "
                   "session, tickers, horizon and effect scope",
                   details={"legacy": list(legacy_target), "snapshot": list(snapshot_target)})


def load_score_parity_inputs(conn, store, *, legacy_job_id: str,
                             snapshot_job_id: str) -> ScoreParityInputs:
    """Load both jobs' committed ``score.json``, refuse a target mismatch,
    and resolve the snapshot job's launch-time snapshot binding.

    ``legacy_job_id`` must have actually run ``--input-mode legacy`` (the
    barrier path) and ``snapshot_job_id`` must have actually run
    ``--input-mode snapshot`` (the adapted path) -- both are verified against
    each job's own recorded ``JobSpec.parameters["input_mode"]``, not merely
    assumed from which keyword argument the caller used. The two job ids must
    also differ, and both must target the same resolved session, tickers,
    horizon and effect scope. Raises (``INPUT_CHANGED``/``VALIDATION_FAILED``)
    rather than returning inputs when either job has no committed score, the
    two are the same job, either ran in the wrong execution mode, or the two
    disagree on target.
    """
    _verify_distinct_jobs(legacy_job_id, snapshot_job_id)
    legacy_params = _job_parameters(conn, legacy_job_id)
    snapshot_params = _job_parameters(conn, snapshot_job_id)
    _verify_execution_modes(legacy_params, snapshot_params)
    legacy_doc, _ = _committed_score(conn, store, legacy_job_id)
    snapshot_doc, snapshot_attempt_id = _committed_score(conn, store, snapshot_job_id)
    _verify_same_target(legacy_doc, snapshot_doc, legacy_params, snapshot_params)
    snapshot_ref = _snapshot_binding(conn, store, snapshot_attempt_id)
    return ScoreParityInputs(
        legacy_job_id=legacy_job_id, snapshot_job_id=snapshot_job_id,
        legacy_rows=_rows_by_id(legacy_doc), snapshot_rows=_rows_by_id(snapshot_doc),
        snapshot_ref=snapshot_ref)
