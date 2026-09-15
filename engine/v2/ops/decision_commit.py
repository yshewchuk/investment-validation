"""Fenced catalog commit helpers for shadow decision and settlement candidates."""
from __future__ import annotations

import base64
import json

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ledger.decisions import DecisionConflict, import_lines, insert, record_divergence
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.effects_graph import effect_scope as _job_effect_scope
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


def _decision_generation_ref(context):
    """The generation identity of the plan that produced this validated
    commit (guide §5.5 item 1) -- ``context["deployment"]``/``context["clock"]``
    are the decision plan's own pinned ``deployment``/``decision_clock``
    (``decision_validation.validate``), the same two fields
    ``effects_graph._generation_ref`` folds into every other generation-aware
    receipt. A fresh ``ops plan nightly`` always re-pins ``decision_clock``,
    so two distinct generations for the same session always compute a
    different ref here, while a true retry of the same saved plan reproduces
    the identical one.
    """
    return content_hash({"deployment": context.get("deployment") or "",
                         "decision_clock": context.get("clock") or ""})


def _commit_row_or_diverge(conn, context, row, generation_ref, *, clock):
    """Insert one candidate row's decision; on a same-``decision_id`` content
    conflict, record a divergence instead of failing (guide §5.5 item 1).

    Returns the inserted receipt, or ``None`` when this row diverged from an
    already-committed decision (nothing new was written). An ambiguous
    conflict -- no single existing row shares this ``decision_id`` -- is a
    real identity bug, not a cross-generation divergence, and still fails.
    """
    key = content_hash([context["purpose"], row["event_id"], row["strategy"],
                        context["deployment"], context["clock"], context["session"]])
    decision_id = "prediction:" + row["row_id"]
    try:
        return insert(conn, logical_key=key, decision_id=decision_id, payload=row,
                      purpose=context["purpose"], kind="prediction",
                      validations=context["validations"], created_at=format_timestamp(clock.now()),
                      generation_ref=generation_ref)
    except DecisionConflict:
        existing = conn.execute(
            "SELECT decision_id, generation_ref, payload_hash FROM decisions WHERE decision_id=?",
            (decision_id,)).fetchone()
        if existing is None:
            raise fail("IDEMPOTENCY_CONFLICT",
                      "decision content conflicts with its logical identity") from None
        record_divergence(
            conn, decision_id=decision_id, scope=context["scope"], occurrence=context["session"],
            existing_generation_ref=existing["generation_ref"], attempted_generation_ref=generation_ref,
            existing_payload_hash=existing["payload_hash"], attempted_payload_hash=content_hash(row),
            reason="a later generation committed different content for an already-decided "
                   "scheduled occurrence; the first committed prediction stays authoritative "
                   "(guide §5.5 item 1)",
            created_at=format_timestamp(clock.now()))
        return None


def _advance_decisions_watermark(conn, context, candidates, *, clock):
    """Enqueue export/release_intent and advance the "decisions" watermark --
    unless an earlier generation already completed this exact (scope,
    session) with different content, in which case none of the three runs
    (guide §5.5 item 1): a second generation's own release_key almost always
    differs from the first's even when no individual row diverged (it is
    derived from THIS generation's own plan/evidence artifacts), and
    advancing anything here would either IDEMPOTENCY_CONFLICT the watermark
    or point export at content this generation never actually committed.
    Leaving the first generation's watermark in place is what makes it
    authoritative.
    """
    validated_hash = content_hash({
        "candidate": context.get("candidate_content_hash", content_hash(candidates)),
        "plan": context.get("plan_artifact_id"), "evidence": context.get("evidence_artifact_id"),
    })
    release_key = content_hash([context["scope"], context["session"], validated_hash])
    prior = conn.execute(
        "SELECT occurrence, receipt_ref FROM watermarks WHERE pipeline='nightly' AND scope=? "
        "AND stage='decisions' AND generation=''", (context["scope"],)).fetchone()
    superseded = (prior is not None and prior["occurrence"] == context["session"]
                 and prior["receipt_ref"] != release_key)
    if superseded:
        return
    enqueue(conn, "export", release_key, {"validation": validated_hash})
    enqueue(conn, "release_intent", release_key, {"validation": validated_hash})
    watermark(conn, "nightly", context["scope"], "decisions", context["session"], release_key,
             clock=clock)


def commit_decisions_in_transaction(conn, claim, candidates, context, *, clock):
    """Insert decisions plus export/release intent under an already-open attempt transaction."""
    if not conn.in_transaction:
        raise ValueError("decision commit requires the attempt transaction")
    verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    # P2-C04: the validated scope must match this JOB's own effect scope
    # (``parameters["effect_scope"]``, set once by ``nightly.build_legacy_job_requests``
    # for every stage of one graph) — never the output_namespace authority
    # alone, which is always "shadow" whether this is a full or subset run.
    # A claim with no ``effect_scope`` parameter (every job built before this
    # field existed) falls back to ``output_namespace``, the old behaviour.
    if context["purpose"] != "shadow" or context["scope"] != _job_effect_scope(claim):
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
    # guide §5.5 item 1: the first committed record for a scheduled decision
    # occurrence stays authoritative forever; a later generation's differing
    # content for the same row never rewrites it (see the two helpers above).
    generation_ref = _decision_generation_ref(context)
    receipts = [receipt for receipt in
               (_commit_row_or_diverge(conn, context, row, generation_ref, clock=clock)
                for row in candidates) if receipt is not None]
    _advance_decisions_watermark(conn, context, candidates, clock=clock)
    return receipts


def commit_decisions(conn, claim, candidates, context, validated_context, *, clock):
    """Compatibility wrapper; production callers use the in-transaction form."""
    if validated_context != validate_candidates(candidates, context):
        raise fail("INPUT_CHANGED", "validated candidate content changed")
    with transaction(conn):
        return commit_decisions_in_transaction(conn, claim, candidates, validated_context, clock=clock)


#: Fixed, literal scope for settlement-line divergences -- deliberately NOT
#: the job's effect scope (``_job_effect_scope``, e.g. "shadow"), matching
#: ``ledger_history_import._record_legacy_divergence``'s own "legacy_import"
#: convention: a scope names the KIND of legacy-duplicate evidence, and
#: ``occurrence`` (below, the row_id) names the specific decision it is
#: about -- not the nightly session.
_SETTLEMENT_DIVERGENCE_SCOPE = "legacy_settlement"


def import_settlement_candidates_in_transaction(conn, claim, candidate_ref, rows, *, clock,
                                                 session=None, on_divergence=None):
    """Import only rows captured from the isolated legacy append, under the active fence.

    ``session`` is the finality-resolved date the settlement worker actually
    scored ``through`` (P2-C03) — the caller reads it off the bound
    ``settlement.json`` document (``_action_settlement``'s own ``session``
    field). It defaults to the job's REQUESTED ``session`` parameter for a
    caller that predates that field (no walk-back, same value either way).

    A settlement line that CONFLICTS with its recorded prediction on a
    contract field (``ticker``/``strategy``/``event_date``/``settlement``) is
    the standing legacy-duplicate case (2026-09-15 nightly attempt 10, the
    DLNG row: the prediction was first-imported with the event's original
    date, and the legacy ledger's later, differing outcome line names the
    date it moved to after an AMC->BMO shift). The same user decision that
    governs ``ops ledger import-history`` (guide §5.5 item 1 / the
    ``legacy_import`` scope) applies here: the first committed content stays
    authoritative, the conflicting line is recorded as a durable
    ``decision_divergences`` row (scope ``legacy_settlement``) and dropped,
    and the stage COMMITS the rest instead of refusing outright. A line
    naming no committed prediction at all is a different failure (a missing
    contract, not a duplicate) and still refuses -- see ``_settlement_line``.

    ``on_divergence``, when given, is called with each diverging line's
    ``row_id`` as it is recorded -- the caller's hook for surfacing a
    per-job divergence count/row_id list on the job's own output (see
    ``engine.v2.ops.supervisor``'s ``legacy_settlement`` effect) without this
    function's return value (the committed ``receipts``, unchanged) having
    to carry it.
    """
    if not conn.in_transaction:
        raise ValueError("settlement import requires the attempt transaction")
    verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    source_lines = []
    for item in rows:
        line = _settlement_line(conn, item, clock=clock, on_divergence=on_divergence)
        if line is not None:
            source_lines.append(line)
    try:
        receipts = import_lines(conn, candidate_ref.content_hash, source_lines, kind="outcome",
                                created_at=format_timestamp(clock.now()))
    except DecisionConflict:
        raise fail("IDEMPOTENCY_CONFLICT", "settlement observation conflicts with history") from None
    # P2-C04: settlement uses the same effect scope decision commit and
    # export use, never the bare output_namespace — a subset settlement run
    # must never advance the global watermark either.
    scope = _job_effect_scope(claim)
    release_key = content_hash(["settlement", scope, candidate_ref.content_hash])
    enqueue(conn, "export", release_key, {"settlement_candidate": candidate_ref.content_hash})
    watermark(conn, "nightly", scope, "settlement",
              session or claim.spec.parameters["session"], release_key, clock=clock)
    return receipts


def _settlement_line(conn, item, *, clock, on_divergence=None):
    """Validate one captured settlement line; return its original bytes to
    import, or ``None`` when it diverged from the recorded prediction (a
    divergence row was recorded instead, and the line must not be
    imported).
    """
    if not isinstance(item, dict):
        raise fail("VALIDATION_FAILED", "settlement candidate is malformed")
    try:
        original = base64.b64decode(item["original_b64"], validate=True)
        payload = json.loads(original)
    except (KeyError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        raise fail("VALIDATION_FAILED", "settlement candidate has invalid original bytes") from None
    if payload != item.get("row"):
        raise fail("VALIDATION_FAILED", "settlement payload differs from captured bytes")
    row_id = payload.get("row_id")
    decision_id = "prediction:" + str(row_id)
    prediction = conn.execute(
        "SELECT payload_json, generation_ref, payload_hash FROM decisions WHERE decision_id=?",
        (decision_id,)).fetchone()
    if prediction is None:
        # Not a duplicate: a missing contract. Always a hard refusal.
        raise fail("VALIDATION_FAILED", "settlement names no committed prediction")
    recorded = json.loads(prediction["payload_json"])
    field = _contract_mismatch_field(payload, recorded)
    if field is not None:
        _record_settlement_divergence(conn, decision_id=decision_id, row_id=row_id, field=field,
                                      payload=payload, prediction=prediction, clock=clock)
        if on_divergence is not None:
            on_divergence(row_id)
        return None
    _validate_settlement_state(payload)
    return original


def _contract_mismatch_field(payload, recorded):
    for field in ("ticker", "strategy", "event_date", "settlement"):
        if payload.get(field) != recorded.get(field):
            return field
    return None


def _record_settlement_divergence(conn, *, decision_id, row_id, field, payload, prediction, clock):
    """Durable evidence that a settlement line disagreed with the recorded
    prediction's contract (guide §5.5 item 1 applied to settlement). Keyed so
    an identical retry or replay of the SAME divergent line never duplicates
    the row: ``attempted_generation_ref`` is the settlement's own observation
    identity (``resolved_at``/``settled_at``, the same field
    ``decisions._import_decision_id`` already uses to distinguish repeated
    outcome observations for one row) rather than anything session- or
    wall-clock-derived, so ``record_divergence``'s content-keyed
    ``divergence_id`` is reproduced exactly on a retry.
    """
    attempted_hash = content_hash(payload)
    generation_ref = str(payload.get("resolved_at") or payload.get("settled_at") or attempted_hash)
    record_divergence(
        conn, decision_id=decision_id, scope=_SETTLEMENT_DIVERGENCE_SCOPE, occurrence=str(row_id),
        existing_generation_ref=prediction["generation_ref"],
        attempted_generation_ref=generation_ref,
        existing_payload_hash=prediction["payload_hash"],
        attempted_payload_hash=attempted_hash,
        reason="settlement_contract_mismatch: field " + field + " differs from the recorded "
               "prediction; the first committed prediction stays authoritative (guide §5.5 item 1)",
        created_at=format_timestamp(clock.now()))


def _validate_settlement_state(payload):
    if payload.get("status") not in ("resolved", "unresolvable") or not payload.get("resolved_at"):
        raise fail("VALIDATION_FAILED", "settlement has no valid observation state")
    if payload["status"] == "resolved" and (
            not payload.get("settlement_source") or not payload.get("exit_source")
            or not isinstance(payload.get("exit_finality"), dict)
            or payload["exit_finality"].get("is_final") is not True):
        raise fail("VALIDATION_FAILED", "resolved settlement lacks recorded exit evidence")
