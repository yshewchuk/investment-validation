"""Fenced catalog commit helpers for shadow decision and settlement candidates."""
from __future__ import annotations

import base64
import json

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ledger.decisions import DecisionConflict, import_lines, insert
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.errors import fail
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.outbox import enqueue, watermark


def _document(conn, store, artifact_id, name):
    ref = artifact(conn, store, artifact_id)
    try:
        return json.loads(store.read_verified(ref)), {
            "artifact_id": ref.artifact_id, "content_hash": ref.content_hash,
        }
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise fail("VALIDATION_FAILED", name + " artifact is not a JSON document") from None


def validated_decision_candidate(conn, store, claim, candidate_ref):
    """Load only admitted immutable inputs and return strict validated context."""
    bindings = claim.spec.parameters.get("input_bindings") or {}
    names = {
        "score": "score.json", "finality": "finality.json",
        "plan": "decision_plan.json", "evidence": "decision_evidence.json",
    }
    missing = [name for name, binding in names.items()
               if not bindings.get(binding) or str(bindings[binding]).startswith("job_")]
    if missing:
        raise fail("VALIDATION_FAILED", "decision evidence inputs are unavailable",
                   details={"missing_bindings": missing})
    score, score_ref = _document(conn, store, bindings[names["score"]], "score")
    finality, finality_ref = _document(conn, store, bindings[names["finality"]], "finality")
    plan, plan_ref = _document(conn, store, bindings[names["plan"]], "decision plan")
    evidence, evidence_ref = _document(conn, store, bindings[names["evidence"]], "decision evidence")
    candidate = json.loads(store.read_verified(candidate_ref))
    if not isinstance(candidate, dict) or not isinstance(candidate.get("rows"), list):
        raise fail("VALIDATION_FAILED", "decision candidate artifact has no rows")
    admitted = set(claim.spec.input_refs)
    if not {score_ref["artifact_id"], finality_ref["artifact_id"], plan_ref["artifact_id"],
            evidence_ref["artifact_id"]}.issubset(admitted):
        raise fail("INPUT_CHANGED", "decision validation inputs were not admitted")
    context = validate(candidate["rows"], score=score, finality=finality, plan=plan,
                       evidence=evidence, bindings={
                           "score": score_ref, "finality": finality_ref, "plan": plan_ref,
                           "evidence": evidence_ref,
                       })
    context["candidate_artifact_id"] = candidate_ref.artifact_id
    context["candidate_content_hash"] = candidate_ref.content_hash
    context["candidate_rows_hash"] = content_hash(candidate["rows"])
    context["bindings"] = {
        "score": score_ref, "finality": finality_ref, "plan": plan_ref,
        "evidence": evidence_ref,
    }
    return candidate["rows"], context


def validate_candidates(candidates, context):
    """Compatibility test seam for the same pure validator, never caller booleans."""
    strict = context.get("candidate_validation")
    if not isinstance(strict, dict):
        raise fail("VALIDATION_FAILED", "candidate validation requires bound evidence")
    return validate(candidates, score=strict["score"], finality=strict["finality"],
                    plan=strict["plan"], evidence=strict["evidence"],
                    bindings=strict["bindings"])


def commit_decisions_in_transaction(conn, claim, candidates, context, *, clock):
    """Insert decisions plus export/release intent under an already-open attempt transaction."""
    if not conn.in_transaction:
        raise ValueError("decision commit requires the attempt transaction")
    verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    if context["purpose"] != "shadow" or context["scope"] != claim.spec.output_namespace:
        raise fail("VALIDATION_FAILED", "only pinned shadow authority is enabled")
    if context.get("session") != claim.spec.parameters.get("session"):
        raise fail("INPUT_CHANGED", "validated session differs from the admitted job")
    if context.get("candidate_rows_hash") != content_hash(candidates):
        raise fail("INPUT_CHANGED", "candidate rows changed after validation")
    bindings = context.get("bindings")
    input_bindings = claim.spec.parameters.get("input_bindings") or {}
    names = {"score": "score.json", "finality": "finality.json",
             "plan": "decision_plan.json", "evidence": "decision_evidence.json"}
    if not isinstance(bindings, dict):
        raise fail("INPUT_CHANGED", "validated input bindings are missing")
    admitted = set(claim.spec.input_refs)
    for key, name in names.items():
        ref = bindings.get(key) if isinstance(bindings.get(key), dict) else {}
        if ref.get("artifact_id") != input_bindings.get(name) or ref.get("artifact_id") not in admitted:
            raise fail("INPUT_CHANGED", "validated inputs differ from the admitted job",
                       details={"binding": name})
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
    validated_hash = content_hash({
        "candidate": context.get("candidate_content_hash", content_hash(candidates)),
        "plan": context.get("plan_artifact_id"), "evidence": context.get("evidence_artifact_id"),
    })
    release_key = content_hash([context["scope"], context["session"], validated_hash])
    enqueue(conn, "export", release_key, {"validation": validated_hash})
    enqueue(conn, "release_intent", release_key, {"validation": validated_hash})
    watermark(conn, "nightly", context["scope"], "decisions", context["session"], release_key, clock=clock)
    return receipts


def commit_decisions(conn, claim, candidates, context, validated_context, *, clock):
    """Compatibility wrapper; production callers use the in-transaction form."""
    if validated_context != validate_candidates(candidates, context):
        raise fail("INPUT_CHANGED", "validated candidate content changed")
    with transaction(conn):
        return commit_decisions_in_transaction(conn, claim, candidates, validated_context, clock=clock)


def import_settlement_candidates_in_transaction(conn, claim, candidate_ref, rows, *, clock):
    """Import only rows captured from the isolated legacy append, under the active fence."""
    if not conn.in_transaction:
        raise ValueError("settlement import requires the attempt transaction")
    verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    source_lines = [_settlement_line(conn, item) for item in rows]
    try:
        receipts = import_lines(conn, candidate_ref.content_hash, source_lines, kind="outcome",
                                created_at=format_timestamp(clock.now()))
    except DecisionConflict:
        raise fail("IDEMPOTENCY_CONFLICT", "settlement observation conflicts with history") from None
    release_key = content_hash(["settlement", claim.spec.output_namespace,
                                candidate_ref.content_hash])
    enqueue(conn, "export", release_key, {"settlement_candidate": candidate_ref.content_hash})
    watermark(conn, "nightly", claim.spec.output_namespace, "settlement",
              claim.spec.parameters["session"], release_key, clock=clock)
    return receipts


def _settlement_line(conn, item):
    if not isinstance(item, dict):
        raise fail("VALIDATION_FAILED", "settlement candidate is malformed")
    try:
        original = base64.b64decode(item["original_b64"], validate=True)
        payload = json.loads(original)
    except (KeyError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        raise fail("VALIDATION_FAILED", "settlement candidate has invalid original bytes") from None
    if payload != item.get("row"):
        raise fail("VALIDATION_FAILED", "settlement payload differs from captured bytes")
    prediction = conn.execute("SELECT payload_json FROM decisions WHERE decision_id=?",
                              ("prediction:" + str(payload.get("row_id")),)).fetchone()
    if prediction is None:
        raise fail("VALIDATION_FAILED", "settlement names no committed prediction")
    _validate_settlement(payload, json.loads(prediction[0]))
    return original


def _validate_settlement(payload, recorded):
    row_id = payload.get("row_id")
    for field in ("ticker", "strategy", "event_date", "settlement"):
        if payload.get(field) != recorded.get(field):
            raise fail("VALIDATION_FAILED", "settlement does not match recorded contract",
                       details={"field": field, "row_id": row_id})
    if payload.get("status") not in ("resolved", "unresolvable") or not payload.get("resolved_at"):
        raise fail("VALIDATION_FAILED", "settlement has no valid observation state")
    if payload["status"] == "resolved" and (
            not payload.get("settlement_source") or not payload.get("exit_source")
            or not isinstance(payload.get("exit_finality"), dict)
            or payload["exit_finality"].get("is_final") is not True):
        raise fail("VALIDATION_FAILED", "resolved settlement lacks recorded exit evidence")
