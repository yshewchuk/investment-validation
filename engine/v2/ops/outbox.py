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


def watermark(conn, pipeline, scope, stage, occurrence, receipt_ref, *, clock):
    if not conn.in_transaction:
        raise ValueError("watermark advancement requires the effect transaction")
    row = conn.execute("SELECT occurrence,receipt_ref FROM watermarks WHERE pipeline=? AND scope=? AND stage=?",
                       (pipeline, scope, stage)).fetchone()
    if row and row["occurrence"] > occurrence:
        return
    if row and row["occurrence"] == occurrence and row["receipt_ref"] != receipt_ref:
        raise fail("IDEMPOTENCY_CONFLICT", "completed occurrence has another receipt")
    conn.execute(
        "INSERT INTO watermarks VALUES (?,?,?,?,?,?) ON CONFLICT(pipeline,scope,stage) "
        "DO UPDATE SET occurrence=excluded.occurrence,receipt_ref=excluded.receipt_ref,"
        "completed_at=excluded.completed_at",
         (pipeline, scope, stage, occurrence, receipt_ref, format_timestamp(clock.now())))


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
