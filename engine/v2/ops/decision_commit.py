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
from engine.v2.ops.input_bindings import recorded_bindings
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


def _resolved_evidence_bindings(conn, claim, names):
    """The recorded launch-time resolution for each evidence binding, verified.

    Reads only ``attempt_input_bindings`` — never re-queries a ``job_``
    binding's parent — so a coordinator validating this attempt always uses
    exactly what was staged, even if the parent's current output has since
    moved on.
    """
    input_bindings = claim.spec.parameters.get("input_bindings") or {}
    missing = [key for key, binding in names.items() if not input_bindings.get(binding)]
    if missing:
        raise fail("VALIDATION_FAILED", "decision evidence inputs are unavailable",
                   details={"missing_bindings": missing})
    recorded = recorded_bindings(conn, claim.attempt_id)
    resolved = {}
    for key, binding_name in names.items():
        row = recorded.get(binding_name)
        if row is None or row.binding != str(input_bindings[binding_name]):
            raise fail("INPUT_CHANGED", "decision validation input was not resolved at launch",
                       details={"binding": binding_name})
        resolved[key] = row
    return resolved


def validated_decision_candidate(conn, store, claim, candidate_ref):
    """Load only admitted immutable inputs and return strict validated context."""
    names = {
        "score": "score.json", "finality": "finality.json",
        "plan": "decision_plan.json", "evidence": "decision_evidence.json",
    }
    resolved = _resolved_evidence_bindings(conn, claim, names)
    score, score_ref = _document(conn, store, resolved["score"].artifact_id, "score")
    finality, finality_ref = _document(conn, store, resolved["finality"].artifact_id, "finality")
    plan, plan_ref = _document(conn, store, resolved["plan"].artifact_id, "decision plan")
    evidence, evidence_ref = _document(conn, store, resolved["evidence"].artifact_id, "decision evidence")
    candidate = json.loads(store.read_verified(candidate_ref))
    if not isinstance(candidate, dict) or not isinstance(candidate.get("rows"), list):
        raise fail("VALIDATION_FAILED", "decision candidate artifact has no rows")
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


def _verify_committed_bindings(conn, claim, bindings):
    """Refuse unless every validated binding still matches the durable record.

    Re-reads ``attempt_input_bindings`` rather than trusting ``context``
    alone: this is the coordinator's own check that what it is about to
    commit still matches what launch recorded, never a re-query of a
    ``job_`` parent's current (possibly since-changed) state.
    """
    if not isinstance(bindings, dict):
        raise fail("INPUT_CHANGED", "validated input bindings are missing")
    input_bindings = claim.spec.parameters.get("input_bindings") or {}
    names = {"score": "score.json", "finality": "finality.json",
             "plan": "decision_plan.json", "evidence": "decision_evidence.json"}
    recorded = recorded_bindings(conn, claim.attempt_id)
    admitted = set(claim.spec.input_refs) | {row.artifact_id for row in recorded.values()}
    for key, name in names.items():
        ref = bindings.get(key) if isinstance(bindings.get(key), dict) else {}
        row = recorded.get(name)
        if (row is None or row.binding != str(input_bindings.get(name))
                or ref.get("artifact_id") != row.artifact_id
                or ref.get("content_hash") != row.content_hash
                or ref.get("artifact_id") not in admitted):
            raise fail("INPUT_CHANGED", "validated inputs differ from the admitted job",
                       details={"binding": name})


def commit_decisions_in_transaction(conn, claim, candidates, context, *, clock):
    """Insert decisions plus export/release intent under an already-open attempt transaction."""
    if not conn.in_transaction:
        raise ValueError("decision commit requires the attempt transaction")
    verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    if context["purpose"] != "shadow" or context["scope"] != claim.spec.output_namespace:
        raise fail("VALIDATION_FAILED", "only pinned shadow authority is enabled")
    # P2-C03: the job's own ``session`` parameter is always the REQUESTED
    # date (walk-back never changes job identity) — compare it against the
    # plan's ``requested_session``, never against the resolved ``session``,
    # which legitimately differs from it on a walk-back night. The resolved
    # session is independently cross-checked against the recorded finality
    # binding's own date, never merely trusted from ``context``.
    if context.get("requested_session") != claim.spec.parameters.get("session"):
        raise fail("INPUT_CHANGED", "validated session differs from the admitted job")
    if context.get("session") != context.get("finality_date"):
        raise fail("INPUT_CHANGED", "validated session differs from the recorded finality binding")
    if context.get("candidate_rows_hash") != content_hash(candidates):
        raise fail("INPUT_CHANGED", "candidate rows changed after validation")
    _verify_committed_bindings(conn, claim, context.get("bindings"))
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


def import_settlement_candidates_in_transaction(conn, claim, candidate_ref, rows, *, clock,
                                                 session=None):
    """Import only rows captured from the isolated legacy append, under the active fence.

    ``session`` is the finality-resolved date the settlement worker actually
    scored ``through`` (P2-C03) — the caller reads it off the bound
    ``settlement.json`` document (``_action_settlement``'s own ``session``
    field). It defaults to the job's REQUESTED ``session`` parameter for a
    caller that predates that field (no walk-back, same value either way).
    """
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
              session or claim.spec.parameters["session"], release_key, clock=clock)
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
