"""One-shot catalog data migration: backfill ``decisions.generation_ref`` for
``kind="outcome"`` rows committed before 2026-09-15.

Why this exists: ``decision_commit._match_same_session`` (before 2026-09-15)
matched a ``generation_ref IS NULL`` row to an incoming settlement line by
content signature (payload minus the wall-clock fields) REGARDLESS of
session -- so on a LATER session, an unresolvable re-observation whose
content happened to equal an older NULL row was wrongly skipped as
``already_observed_this_session``, silently dropping a real re-observation
(legacy normally re-observes unresolvable rows daily). The fix
(``decision_commit.py``, same date) removes that fallback: same-session
matching now uses ONLY a recorded ``generation_ref``. This module recovers
that session, from durable catalog provenance alone, for as many pre-fix
rows as the catalog can prove it for -- every row this cannot derive a
session for is left ``generation_ref IS NULL``, exactly as before (it can
still make a prediction terminal via ``already_resolved`` if it is itself
resolved; it just never matches a same-session rerun).

Two sources, matching the two things that ever write a ``kind="outcome"``
decision:

* **A nightly ``legacy_settlement`` attempt**
  (``engine.v2.ops.decision_commit.import_settlement_candidates_in_transaction``,
  called from ``engine.v2.ops.supervisor.Service._settlement_effect``,
  supervisor.py:670-706). That call site reads ``session =
  document.get("session")`` off the attempt's own ``legacy_settlement``
  candidate artifact (falling back to the job's requested session,
  ``claim.spec.parameters["session"]``, supervisor.py:674, when the
  candidate predates the ``session`` field) and passes it to
  ``import_settlement_candidates_in_transaction(..., session=session)``,
  which commits every admitted line via ``import_lines(conn,
  candidate_ref.content_hash, source_lines, kind="outcome", ...,
  generation_ref=effective_session)`` (decision_commit.py:382-383). So a
  freshly-committed row's ``decision_imports.source_hash`` IS its batch's
  candidate ``content_hash``, and that same candidate document's own
  ``session`` field (or, absent that, its producing job's requested
  session) is the value a fixed-forward commit would have stamped as
  ``generation_ref``. This module rebuilds that mapping from
  ``attempt_outputs`` (name='legacy_settlement') -> ``artifacts`` ->
  the candidate document, read back through the store, and applies it to
  every pre-fix row whose ``decision_imports.source_hash`` matches.
* **``ops ledger import-history``**
  (``engine.v2.ops.ledger_history_import.import_history``). That importer
  calls ``import_lines(..., on_conflict="diverge", provenance_label=path.name)``
  WITHOUT ``generation_ref``, and ``provenance_label`` is never written to a
  durable, queryable column -- it is only folded into a divergence row's
  human-readable ``reason`` text, and only for a line that diverges. So the
  legacy ``outcomes/<date>.jsonl`` file a given row came from cannot be
  recovered from the catalog alone. Per the task brief, this module instead
  uses the UTC date of the row's own ``resolved_at``: legacy's nightly
  writes ``outcomes/<date>.jsonl`` with ``resolved_at`` stamped on that same
  wall-clock date (``engine/ledger.py::score_outcomes``), so the two
  ordinarily agree, and ``resolved_at`` is always present on any row that
  reached ``decisions`` at all (``decisions._import_decision_id`` requires
  it for ``kind="outcome"``).

Idempotent by construction (only ``generation_ref IS NULL`` rows are ever
touched) and additionally guarded by a one-shot marker in the shared
``schema_versions`` bookkeeping table (``engine.v2.ops.migrations``'s
docstring), under a DELIBERATELY SEPARATE owner from the "ops" DDL sequence
in ``engine/v2/ops/schema.py`` (at v9): this is a data migration, not a
schema change, and the DDL ``migrate()``/``require_current`` machinery only
ever executes literal SQL statements -- it cannot read artifact-store
documents, which this backfill needs. A distinct owner keeps this out of
``schema.py``'s ``MIGRATIONS``/``require_current`` checks entirely, so it can
never be flagged "schema is newer than this code supports" on a later
``ops bootstrap``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from engine.v2.contracts import ArtifactRef
from engine.v2.foundation import ArtifactError, content_hash, format_timestamp
from engine.v2.ops.catalog import transaction

__all__ = ["backfill_outcome_sessions"]

_OWNER = "outcome_session_backfill"
_VERSION = 1
_NAME = "backfill_outcome_generation_ref_from_provenance"

_ARTIFACT_READ_ERRORS = (ArtifactError, TypeError, ValueError, KeyError, json.JSONDecodeError)


def _marker_checksum():
    return content_hash({"owner": _OWNER, "version": _VERSION, "name": _NAME})


def _already_applied(conn):
    return conn.execute("SELECT 1 FROM schema_versions WHERE owner=? AND version=?",
                        (_OWNER, _VERSION)).fetchone() is not None


def _job_requested_session(conn, attempt_id):
    """The requested ``session`` job parameter for the job that ran
    ``attempt_id`` -- ``_settlement_effect``'s own fallback
    (supervisor.py:674) when a candidate document predates the ``session``
    field."""
    row = conn.execute(
        "SELECT jobs.spec_json FROM attempts JOIN jobs ON jobs.job_id = attempts.job_id "
        "WHERE attempts.attempt_id = ?", (attempt_id,)).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0]).get("parameters", {}).get("session")
    except _ARTIFACT_READ_ERRORS:
        return None


def _nightly_session_map(conn, store):
    """``{candidate content_hash: settlement session}`` for every nightly
    ``legacy_settlement`` attempt this catalog still holds the candidate
    artifact for. An artifact this cannot read or parse is skipped -- its
    rows fall through to the ``resolved_at``-derived path, or stay NULL --
    never raised: one unreadable artifact must not fail the whole backfill.
    """
    mapping: dict[str, str] = {}
    query = ("SELECT attempt_outputs.attempt_id, artifacts.ref_json FROM attempt_outputs "
             "JOIN artifacts ON artifacts.artifact_id = attempt_outputs.artifact_id "
             "WHERE attempt_outputs.name = 'legacy_settlement'")
    for attempt_id, ref_json in conn.execute(query):
        try:
            ref = ArtifactRef(**json.loads(ref_json))
            document = json.loads(store.read_verified(ref))
        except _ARTIFACT_READ_ERRORS:
            continue
        session = document.get("session") if isinstance(document, dict) else None
        if not session:
            session = _job_requested_session(conn, attempt_id)
        if session:
            mapping[ref.content_hash] = str(session)
    return mapping


def _resolved_at_session(payload):
    observed = payload.get("resolved_at") or payload.get("settled_at")
    if not observed:
        return None
    try:
        stamp = datetime.fromisoformat(str(observed).replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).date().isoformat()


def backfill_outcome_sessions(conn, store, *, clock):
    """Derive and write ``generation_ref`` for every ``kind='outcome'``
    decision that still has ``generation_ref IS NULL``.

    Returns a JSON-safe summary: ``{"already_applied": bool,
    "nightly_settlement": int, "import_history_resolved_at": int,
    "undetermined": int}`` -- the three counts partition every row this call
    considered (a repeat call after a completed backfill returns zero counts
    with ``already_applied: True`` and touches nothing).
    """
    if _already_applied(conn):
        return {"already_applied": True, "nightly_settlement": 0,
                "import_history_resolved_at": 0, "undetermined": 0}
    nightly_map = _nightly_session_map(conn, store)
    counts = {"nightly_settlement": 0, "import_history_resolved_at": 0, "undetermined": 0}
    with transaction(conn):
        query = ("SELECT decisions.decision_id, decisions.payload_json, decision_imports.source_hash "
                 "FROM decisions JOIN decision_imports "
                 "ON decision_imports.decision_id = decisions.decision_id "
                 "WHERE decisions.kind = 'outcome' AND decisions.generation_ref IS NULL")
        by_decision: dict = {}
        for decision_id, payload_json, source_hash in conn.execute(query):
            entry = by_decision.setdefault(decision_id, {"payload_json": payload_json, "hashes": set()})
            entry["hashes"].add(source_hash)
        for decision_id, info in by_decision.items():
            session = next((nightly_map[h] for h in info["hashes"] if h in nightly_map), None)
            source = "nightly_settlement" if session is not None else None
            if session is None:
                session = _resolved_at_session(json.loads(info["payload_json"]))
                source = "import_history_resolved_at" if session is not None else "undetermined"
            counts[source] += 1
            if session is not None:
                conn.execute("UPDATE decisions SET generation_ref=? WHERE decision_id=?",
                            (session, decision_id))
        conn.execute(
            "INSERT INTO schema_versions (owner, version, name, checksum, applied_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (_OWNER, _VERSION, _NAME, _marker_checksum(), format_timestamp(clock.now())))
    return {"already_applied": False, **counts}
