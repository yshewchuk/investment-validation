"""Validate outside, then fence and insert decisions plus outbox atomically."""
from __future__ import annotations

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ledger.decisions import DecisionConflict, insert
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import fail
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.outbox import enqueue, watermark

REQUIRED = frozenset({"causality", "coverage", "finality", "selection", "replay"})


def validate_candidates(candidates, context):
    findings = []
    if context["purpose"] not in ("shadow", "research_reconstruction", "production"):
        findings.append({"field": "purpose", "reason": "unknown"})
    checks = context.get("validations", {})
    for kind in sorted(REQUIRED):
        receipt = checks.get(kind, {})
        if receipt.get("ok") is not True or receipt.get("input_hash") != context["input_hash"]:
            findings.append({"field": "validations." + kind, "reason": "missing_failed_or_wrong_input"})
    for index, row in enumerate(candidates):
        prefix = f"candidates[{index}]"
        if row.get("as_of") != context["session"]:
            findings.append({"field": prefix + ".as_of", "reason": "not_entry_session"})
        score = row.get("score", {})
        if score.get("is_ladder") or score.get("ladder"):
            findings.append({"field": prefix + ".score", "reason": "ladder_not_official"})
        if row.get("snapshot_hash") != context["input_hash"]:
            findings.append({"field": prefix + ".snapshot_hash", "reason": "input_mismatch"})
        if not row.get("row_id") or not row.get("event_id"):
            findings.append({"field": prefix + ".identity", "reason": "missing"})
    if findings:
        raise fail("VALIDATION_FAILED", "candidate decisions failed validation",
                   details={"stage": "decision_validation", "findings": findings})
    return content_hash({"rows": candidates, "context": context})


def commit_decisions(conn, claim, candidates, context, validated_hash, *, clock):
    if validated_hash != validate_candidates(candidates, context):
        raise fail("INPUT_CHANGED", "validated candidate content changed")
    with transaction(conn):
        verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
        if context["input_hash"] not in claim.spec.input_refs:
            raise fail("INPUT_CHANGED", "decision inputs were not admitted")
        if context["purpose"] == "production":
            raise fail("VALIDATION_FAILED", "production authority is shadow-only in Phase 1")
        if context.get("scope") != claim.spec.output_namespace:
            raise fail("VALIDATION_FAILED", "scope differs from the pinned job plan")
        receipts = []
        for row in candidates:
            key = content_hash([context["purpose"], row["event_id"], row["strategy"],
                                context["deployment"], context["clock"], context["session"]])
            try:
                receipts.append(insert(
                    conn, logical_key=key, decision_id="prediction:" + row["row_id"],
                    payload=row, purpose=context["purpose"], kind="prediction",
                    validations=context["validations"], created_at=format_timestamp(clock.now())))
            except DecisionConflict:
                raise fail("IDEMPOTENCY_CONFLICT", "decision content conflicts with its logical identity") from None
        release_key = content_hash([context["scope"], context["session"], validated_hash])
        enqueue(conn, "export", release_key, {"validation": validated_hash})
        enqueue(conn, "release_intent", release_key, {"validation": validated_hash})
        watermark(conn, "nightly", context["scope"], "decisions", context["session"], release_key, clock=clock)
    return receipts
