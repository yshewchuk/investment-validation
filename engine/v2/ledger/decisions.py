"""Narrow append-only authority. The caller owns the fence and transaction.

Legacy payloads are retained intact. Historical imports are never represented
as newly validated decisions. The table is owned here, not by the scheduler.
"""
from __future__ import annotations

import json

from engine.v2.foundation import canonical_json, content_hash

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS decisions (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, logical_key TEXT NOT NULL UNIQUE,
        decision_id TEXT NOT NULL UNIQUE, payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL,
        purpose TEXT NOT NULL, kind TEXT NOT NULL, validation_json TEXT NOT NULL,
        supersedes TEXT REFERENCES decisions(decision_id), created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TABLE IF NOT EXISTS decision_imports (
        source_hash TEXT NOT NULL, line_number INTEGER NOT NULL,
        decision_id TEXT NOT NULL REFERENCES decisions(decision_id), original_bytes BLOB NOT NULL,
        PRIMARY KEY(source_hash, line_number)
    ) STRICT""",
    """CREATE TABLE IF NOT EXISTS decision_authority (
        singleton INTEGER PRIMARY KEY CHECK(singleton=1), owner TEXT NOT NULL,
        generation INTEGER NOT NULL, switched_at TEXT NOT NULL
    ) STRICT""",
)

#: Phase 3 guide §5.5 item 1: the first committed record for a scheduled
#: decision occurrence stays authoritative forever; a later generation's
#: differing content is never a silent duplicate and never a rewrite -- it is
#: durable, append-only evidence. ``generation_ref`` on ``decisions`` is the
#: identity of the generation that FIRST committed that row (empty/NULL for
#: rows written before this migration, and for legacy imports, which have no
#: generation). True superseding decisions (an operator-approved re-decision)
#: remain ledger-phase scope; this table only records that a conflict was
#: seen and refused, never a decision about which content should win.
SCHEMA_V2 = (
    "ALTER TABLE decisions ADD COLUMN generation_ref TEXT",
    """CREATE TABLE IF NOT EXISTS decision_divergences (
        divergence_id TEXT PRIMARY KEY,
        decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
        scope TEXT NOT NULL, occurrence TEXT NOT NULL,
        existing_generation_ref TEXT, attempted_generation_ref TEXT NOT NULL,
        existing_payload_hash TEXT NOT NULL, attempted_payload_hash TEXT NOT NULL,
        reason TEXT NOT NULL, created_at TEXT NOT NULL
    ) STRICT""",
    """CREATE TRIGGER decision_divergences_no_update
    BEFORE UPDATE ON decision_divergences
    BEGIN
        SELECT RAISE(ABORT, 'decision_divergences rows are immutable');
    END""",
    """CREATE TRIGGER decision_divergences_no_delete
    BEFORE DELETE ON decision_divergences
    BEGIN
        SELECT RAISE(ABORT, 'decision_divergences rows are immutable');
    END""",
)


class DecisionConflict(ValueError):
    pass


def install(conn):
    for statement in SCHEMA:
        conn.execute(statement)


def set_authority(conn, expected_owner, owner, stamp):
    if not conn.in_transaction:
        raise ValueError("authority change needs the caller transaction")
    row = conn.execute("SELECT owner,generation FROM decision_authority WHERE singleton=1").fetchone()
    current = row[0] if row else None
    if current != expected_owner:
        raise DecisionConflict("writer authority changed")
    if row:
        conn.execute("UPDATE decision_authority SET owner=?,generation=generation+1,switched_at=?",
                     (owner, stamp))
    else:
        conn.execute("INSERT INTO decision_authority VALUES (1,?,1,?)", (owner, stamp))


def insert(conn, *, logical_key, decision_id, payload, purpose, kind, validations, created_at,
           supersedes=None, owner="catalog", generation_ref=None):
    if not conn.in_transaction:
        raise ValueError("decision insertion requires the caller transaction")
    authority = conn.execute("SELECT owner FROM decision_authority WHERE singleton=1").fetchone()
    if authority is None or authority[0] != owner:
        raise DecisionConflict("this process does not hold decision authority")
    digest = content_hash(payload)
    old = conn.execute("SELECT * FROM decisions WHERE logical_key=? OR decision_id=?",
                       (logical_key, decision_id)).fetchall()
    if old:
        existing = old[0] if len(old) == 1 else None
        if (existing is None or existing["logical_key"] != logical_key
                or existing["decision_id"] != decision_id
                or existing["payload_hash"] != digest
                or existing["purpose"] != purpose or existing["kind"] != kind
                or existing["supersedes"] != supersedes):
            raise DecisionConflict("IDEMPOTENCY_CONFLICT")
        return dict(existing)
    if supersedes and not payload.get("supersede_reason"):
        raise DecisionConflict("supersession requires a reason")
    conn.execute(
        "INSERT INTO decisions(logical_key,decision_id,payload_hash,payload_json,purpose,kind,"
        "validation_json,supersedes,created_at,generation_ref) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (logical_key, decision_id, digest, canonical_json(payload), purpose, kind,
         canonical_json(validations), supersedes, created_at, generation_ref))
    return dict(conn.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone())


def record_divergence(conn, *, decision_id, scope, occurrence, existing_generation_ref,
                       attempted_generation_ref, existing_payload_hash, attempted_payload_hash,
                       reason, created_at):
    """Append-only evidence that a later generation tried to commit different
    content for an already-decided scheduled occurrence (Phase 3 guide §5.5
    item 1). The first committed decision (``decision_id``) is never
    rewritten; this is a durable divergence record, not a decision. Keyed so
    an identical retry of the same divergent attempt (same decision, same
    attempting generation, same content) is a no-op, matching the identical-
    retry-stays-idempotent contract every other effect in this task follows.
    """
    if not conn.in_transaction:
        raise ValueError("divergence recording requires the caller transaction")
    divergence_id = "div_" + content_hash(
        [decision_id, attempted_generation_ref, attempted_payload_hash]).split(":")[1][:32]
    existing = conn.execute("SELECT * FROM decision_divergences WHERE divergence_id=?",
                            (divergence_id,)).fetchone()
    if existing:
        return dict(existing)
    conn.execute(
        "INSERT INTO decision_divergences(divergence_id,decision_id,scope,occurrence,"
        "existing_generation_ref,attempted_generation_ref,existing_payload_hash,"
        "attempted_payload_hash,reason,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (divergence_id, decision_id, scope, occurrence, existing_generation_ref,
         attempted_generation_ref, existing_payload_hash, attempted_payload_hash, reason,
         created_at))
    return dict(conn.execute("SELECT * FROM decision_divergences WHERE divergence_id=?",
                             (divergence_id,)).fetchone())


def rows(conn, *, kind=None, through=None):
    query = "SELECT * FROM decisions WHERE (? IS NULL OR kind=?) AND (? IS NULL OR sequence<=?) ORDER BY sequence"
    return [dict(row) for row in conn.execute(query, (kind, kind, through, through))]


def _row_conflicts(existing_row, prior_bytes, original, payload):
    """Whether ``payload`` (as ``original`` bytes) disagrees with whatever is
    already on file under this line's ``decision_id`` -- either a prior
    ``decision_imports`` entry with different bytes, or a committed
    ``decisions`` row with different canonical content."""
    if prior_bytes and any(bytes(item[0]) != original for item in prior_bytes):
        return True
    return bool(existing_row) and existing_row["payload_json"] != canonical_json(payload)


def _record_legacy_divergence(conn, *, decision_id, row_id, existing_row, original, payload,
                              source_hash, number, created_at, provenance_label):
    """The user's standing 2026-09-14 decision applied to the legacy ledger
    itself: the FIRST-imported occurrence of a ``decision_id`` stays
    authoritative; a later, differing occurrence is durable evidence
    (``decision_divergences``), never a block and never an overwrite. Its
    original bytes are still kept in ``decision_imports`` provenance,
    referencing the FIRST occurrence's ``decision_id`` -- that table has no
    per-``decision_id`` uniqueness constraint (only ``(source_hash,
    line_number)``), so this holds without conflict.
    """
    record_divergence(
        conn, decision_id=decision_id, scope="legacy_import", occurrence=row_id,
        existing_generation_ref=existing_row["generation_ref"],
        attempted_generation_ref=source_hash + ":" + str(number),
        existing_payload_hash=existing_row["payload_hash"],
        attempted_payload_hash=content_hash(payload),
        reason="legacy_duplicate_row_id: " + (provenance_label or source_hash) + " line "
               + str(number) + " differs from the first-imported occurrence for " + decision_id,
        created_at=created_at)
    conn.execute("INSERT INTO decision_imports VALUES (?,?,?,?)",
                 (source_hash, number, existing_row["decision_id"], original))
    return dict(existing_row)


def import_lines(conn, source_hash, lines, *, kind, created_at, on_conflict="raise",
                 provenance_label=None):
    """Import exact legacy JSONL bytes; duplicate bytes collapse.

    ``on_conflict`` governs a decision_id that already carries DIFFERENT
    content than the line being imported (a mismatched ``decisions.
    payload_json``, or mismatched ``decision_imports.original_bytes``
    recorded under an earlier line):

    - ``"raise"`` (default -- every pre-existing caller, e.g.
      ``decision_commit.import_settlement_candidates_in_transaction``):
      refuses typed as a :class:`DecisionConflict`, unchanged from before
      ``on_conflict`` existed.
    - ``"diverge"`` (used only by the legacy history bootstrap importer,
      :mod:`engine.v2.ops.ledger_history_import`): keeps the FIRST-imported
      content authoritative and records the later, differing occurrence as a
      durable ``decision_divergences`` row instead of raising -- see
      :func:`_record_legacy_divergence`.

    A byte-identical RE-import of the exact same ``(source_hash,
    line_number)`` always stays a plain no-op in both modes, and a changed
    byte at an ALREADY-imported ``(source_hash, line_number)`` -- a
    provenance conflict -- always still refuses in both modes: the ``prior``
    check below runs first, before ``on_conflict`` is even consulted.
    """
    if on_conflict not in ("raise", "diverge"):
        raise ValueError("on_conflict must be 'raise' or 'diverge'")
    receipts = []
    for number, raw in enumerate(lines, 1):
        original = raw if isinstance(raw, bytes) else raw.encode("utf-8")
        payload = json.loads(original)
        row_id = payload.get("row_id")
        if not row_id:
            raise DecisionConflict("legacy row has no row_id")
        decision_id = _import_decision_id(kind, row_id, payload)
        prior = conn.execute("SELECT decision_id,original_bytes FROM decision_imports "
                             "WHERE source_hash=? AND line_number=?", (source_hash, number)).fetchone()
        if prior:
            if bytes(prior[1]) != original:
                raise DecisionConflict("import provenance conflict")
            receipts.append(dict(conn.execute("SELECT * FROM decisions WHERE decision_id=?",
                                               (prior[0],)).fetchone()))
            continue
        existing_row = conn.execute("SELECT * FROM decisions WHERE decision_id=?",
                                    (decision_id,)).fetchone()
        prior_bytes = conn.execute("SELECT original_bytes FROM decision_imports WHERE decision_id=?",
                                   (decision_id,)).fetchall()
        if _row_conflicts(existing_row, prior_bytes, original, payload):
            if on_conflict == "raise":
                raise DecisionConflict("conflicting legacy duplicate; reconciliation required")
            receipts.append(_record_legacy_divergence(
                conn, decision_id=decision_id, row_id=row_id, existing_row=existing_row,
                original=original, payload=payload, source_hash=source_hash, number=number,
                created_at=created_at, provenance_label=provenance_label))
            continue
        receipt = insert(conn, logical_key=decision_id, decision_id=decision_id, payload=payload,
                         purpose="legacy_import", kind=kind, validations=[], created_at=created_at,
                         supersedes=kind + ":" + payload["supersedes"] if payload.get("supersedes") else None)
        conn.execute("INSERT INTO decision_imports VALUES (?,?,?,?)",
                     (source_hash, number, receipt["decision_id"], original))
        receipts.append(receipt)
    return receipts


def _import_decision_id(kind, row_id, payload):
    """Keep repeated outcome observations while collapsing copied bytes.

    A prediction has one immutable identity.  Outcomes are observations of
    that prediction over time: an early ``unresolvable`` row may legitimately
    be followed by a later ``resolved`` row.  The observation clock and state
    identify that append; changed content under the same observation remains
    a conflict in :func:`insert`.
    """
    if kind != "outcome":
        return kind + ":" + row_id
    observed = payload.get("resolved_at") or payload.get("settled_at")
    if not observed or payload.get("status") not in ("resolved", "unresolvable"):
        raise DecisionConflict("legacy outcome has no observation identity")
    suffix = content_hash([row_id, observed]).split(":")[1][:24]
    return "outcome:" + row_id + ":" + suffix
