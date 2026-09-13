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
           supersedes=None, owner="catalog"):
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
        "validation_json,supersedes,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (logical_key, decision_id, digest, canonical_json(payload), purpose, kind,
         canonical_json(validations), supersedes, created_at))
    return dict(conn.execute("SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone())


def rows(conn, *, kind=None, through=None):
    query = "SELECT * FROM decisions WHERE (? IS NULL OR kind=?) AND (? IS NULL OR sequence<=?) ORDER BY sequence"
    return [dict(row) for row in conn.execute(query, (kind, kind, through, through))]


def import_lines(conn, source_hash, lines, *, kind, created_at):
    """Import exact legacy JSONL bytes; duplicate bytes collapse, conflicts abort."""
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
        same = conn.execute("SELECT decision_id,payload_json FROM decisions WHERE decision_id=?",
                            (decision_id,)).fetchone()
        prior_bytes = conn.execute("SELECT original_bytes FROM decision_imports WHERE decision_id=?",
                                   (decision_id,)).fetchall()
        if prior_bytes and any(bytes(item[0]) != original for item in prior_bytes):
            raise DecisionConflict("conflicting legacy duplicate; reconciliation required")
        if same and same[1] != canonical_json(payload):
            raise DecisionConflict("conflicting legacy duplicate; reconciliation required")
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
