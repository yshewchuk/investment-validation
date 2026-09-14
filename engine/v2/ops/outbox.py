"""Idempotent effects and independent completed watermarks."""
from __future__ import annotations

import uuid
from datetime import timedelta

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ops.catalog import dumps, transaction
from engine.v2.ops.errors import fail


def enqueue(conn, kind, logical_key, payload):
    if not conn.in_transaction:
        raise ValueError("outbox effects require the commit transaction")
    digest = content_hash(payload)
    old = conn.execute("SELECT * FROM outbox WHERE kind=? AND logical_key=?", (kind, logical_key)).fetchone()
    if old:
        if old["payload_hash"] != digest:
            raise fail("IDEMPOTENCY_CONFLICT", "outbox payload changed under an existing key")
        return old["effect_id"]
    effect_id = "eff_" + content_hash([kind, logical_key]).split(":")[1][:32]
    conn.execute("INSERT INTO outbox(effect_id,kind,logical_key,payload_hash,payload_json) VALUES (?,?,?,?,?)",
                 (effect_id, kind, logical_key, digest, dumps(payload)))
    return effect_id


def _watermark_row(conn, pipeline, scope, stage, generation):
    return conn.execute(
        "SELECT occurrence,receipt_ref FROM watermarks WHERE pipeline=? AND scope=? AND stage=? "
        "AND generation=?", (pipeline, scope, stage, generation)).fetchone()


def watermark_would_conflict(conn, pipeline, scope, stage, occurrence, receipt_ref, *,
                             generation=""):
    """Read-only precheck: would ``watermark(...)`` below refuse this exact
    write with ``IDEMPOTENCY_CONFLICT``? Never writes.

    Guide §5.4/§5.5 item 1: a caller with an irreversible side effect before
    its own watermark call (``publication.publish_local``'s ``CURRENT``
    pointer swap) must know about a foreseeable refusal BEFORE that side
    effect runs, not after -- the refusal itself must never leave that side
    effect half-done. Does not need an open transaction; it is a plain read.
    """
    row = _watermark_row(conn, pipeline, scope, stage, generation)
    return bool(row) and row["occurrence"] == occurrence and row["receipt_ref"] != receipt_ref


def watermark(conn, pipeline, scope, stage, occurrence, receipt_ref, *, clock, generation=""):
    """Record one stage's completed occurrence; idempotent, monotone, and --
    since guide §5.5 item 1 -- scoped per ``generation`` (default ``""``, the
    original single-bucket behaviour every pre-existing caller keeps).

    A caller that never passes ``generation`` sees EXACTLY the old contract:
    one completed receipt per (pipeline, scope, stage), a same-occurrence
    different-receipt write refused, an older occurrence silently ignored.
    A caller that passes a real ``generation`` (engineering_gate,
    ledger_export, backup, publication/delivery -- see
    ``effects_graph._generation_ref``) gets its OWN row: a genuinely new
    same-session generation never collides with, and never rewrites, an
    earlier generation's already-completed receipt for the same occurrence.
    """
    if not conn.in_transaction:
        raise ValueError("watermark advancement requires the effect transaction")
    row = _watermark_row(conn, pipeline, scope, stage, generation)
    if row and row["occurrence"] > occurrence:
        return
    if row and row["occurrence"] == occurrence and row["receipt_ref"] != receipt_ref:
        raise fail("IDEMPOTENCY_CONFLICT", "completed occurrence has another receipt")
    conn.execute(
        "INSERT INTO watermarks(pipeline,scope,stage,generation,occurrence,receipt_ref,completed_at) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(pipeline,scope,stage,generation) "
        "DO UPDATE SET occurrence=excluded.occurrence,receipt_ref=excluded.receipt_ref,"
        "completed_at=excluded.completed_at",
         (pipeline, scope, stage, generation, occurrence, receipt_ref, format_timestamp(clock.now())))


def claim(conn, kind, *, owner, clock, logical_key=None, lease_seconds=300):
    """Claim one pending effect; attempts are durable and retries do not duplicate rows."""
    token = uuid.uuid4().hex
    with transaction(conn):
        predicate = "kind=? AND (state='pending' OR (state='running' AND claim_expires_at<=?))"
        args = [kind, format_timestamp(clock.now())]
        if logical_key is not None:
            predicate += " AND logical_key=?"
            args.append(logical_key)
        row = conn.execute("SELECT * FROM outbox WHERE " + predicate + " ORDER BY effect_id LIMIT 1", args).fetchone()
        if row is None:
            return None
        expiry = format_timestamp(clock.now() + timedelta(seconds=lease_seconds))
        conn.execute("UPDATE outbox SET state='running',attempts=attempts+1,claim_token=?,claimed_by=?,"
                     "claim_expires_at=? WHERE effect_id=? AND (state='pending' OR "
                     "(state='running' AND claim_expires_at<=?))",
                     (token, owner, expiry, row["effect_id"], format_timestamp(clock.now())))
        claimed = conn.execute("SELECT * FROM outbox WHERE effect_id=? AND claim_token=?",
                               (row["effect_id"], token)).fetchone()
        return dict(claimed) if claimed else None


def complete(conn, effect_id, receipt, *, owner, claim_token, clock):
    with transaction(conn):
        row = conn.execute("SELECT state FROM outbox WHERE effect_id=? AND claimed_by=? AND claim_token=?",
                           (effect_id, owner, claim_token)).fetchone()
        if row is None or row[0] != "running":
            raise fail("STALE_EXPECTATION", "effect is not owned by this retry")
        conn.execute("UPDATE outbox SET state='delivered',receipt_json=?,claim_expires_at=NULL "
                     "WHERE effect_id=?", (dumps(receipt), effect_id))


def fail_effect(conn, effect_id, receipt, *, owner, claim_token, clock):
    with transaction(conn):
        row = conn.execute("SELECT state FROM outbox WHERE effect_id=? AND claimed_by=? AND claim_token=?",
                           (effect_id, owner, claim_token)).fetchone()
        if row is None or row[0] != "running":
            raise fail("STALE_EXPECTATION", "effect claim is no longer current")
        conn.execute("UPDATE outbox SET state='pending',receipt_json=?,claim_token=NULL,claimed_by=NULL,"
                     "claim_expires_at=NULL "
                     "WHERE effect_id=?", (dumps(receipt), effect_id))
