"""Narrow append-only authority. The caller owns the fence and transaction.

Legacy payloads are retained intact. Historical imports are never represented
as newly validated decisions. The table is owned here, not by the scheduler.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

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


def outcome_generation_ref(payload):
    """The session an outcome ``payload`` alone proves, for a caller with no
    other durable provenance to derive ``generation_ref`` from: the UTC date
    of its own ``resolved_at`` (or ``settled_at``, for an older shape).

    Legacy's nightly writes ``outcomes/<date>.jsonl`` with ``resolved_at``
    stamped on that same wall-clock date (``engine/ledger.py::
    score_outcomes``), so this ordinarily agrees with the session a fixed-
    forward commit would have stamped. Returns ``None`` when the payload
    carries no parseable timestamp -- the caller must leave
    ``generation_ref`` unset rather than guess.

    Shared by :mod:`engine.v2.ops.session_backfill` (recovering the session
    for pre-fix rows already in the catalog) and :func:`import_lines` below
    (stamping it on freshly-imported ``kind="outcome"`` rows going forward)
    so the rule is defined exactly once.
    """
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


def import_lines(conn, source_hash, lines, *, kind, created_at, on_conflict="raise",
                 provenance_label=None, generation_ref=None, derive_generation_ref=False):
    """Import exact legacy JSONL bytes; duplicate bytes collapse.

    ``generation_ref``, when given, is stamped on every FRESHLY inserted
    decision (never on the ``prior``/exact-bytes shortcut, a no-op). Used by
    ``decision_commit.import_settlement_candidates_in_transaction`` to record
    the settlement session an outcome observation was committed for.

    ``derive_generation_ref`` (2026-09-15, default ``False`` so every
    pre-existing caller -- and every test that imports a row bare to
    simulate a pre-fix ``generation_ref IS NULL`` row -- keeps getting NULL
    back unchanged): when ``True`` and ``generation_ref`` is ``None``, each
    freshly-inserted ``kind="outcome"`` row instead gets a PER-ROW value
    from :func:`outcome_generation_ref`. Set only by
    ``engine.v2.ops.ledger_history_import``, the one caller with no other
    durable provenance to stamp a session from. A row with no derivable
    timestamp still lands NULL. Ignored for ``kind="prediction"`` and
    whenever ``generation_ref`` is given explicitly.

    ``on_conflict`` governs a decision_id that already carries DIFFERENT
    content than the line being imported: ``"raise"`` (default -- every
    pre-existing caller) refuses typed as a :class:`DecisionConflict`.
    ``"diverge"`` (only :mod:`engine.v2.ops.ledger_history_import`) keeps
    the FIRST-imported content authoritative and records the later,
    differing occurrence as a ``decision_divergences`` row instead --
    see :func:`_record_legacy_divergence`.

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
        row_generation_ref = generation_ref
        if row_generation_ref is None and derive_generation_ref and kind == "outcome":
            row_generation_ref = outcome_generation_ref(payload)
        receipt = insert(conn, logical_key=decision_id, decision_id=decision_id, payload=payload,
                         purpose="legacy_import", kind=kind, validations=[], created_at=created_at,
                         supersedes=kind + ":" + payload["supersedes"] if payload.get("supersedes") else None,
                         generation_ref=row_generation_ref)
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
