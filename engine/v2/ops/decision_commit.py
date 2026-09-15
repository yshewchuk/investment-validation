"""Fenced catalog commit helpers for shadow decision and settlement candidates."""
from __future__ import annotations

import base64
import json
from datetime import datetime

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ledger.decisions import (
    DecisionConflict,
    _import_decision_id,
    import_lines,
    insert,
    record_divergence,
)
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.effects_graph import effect_scope as _job_effect_scope
from engine.v2.ops.errors import fail
from engine.v2.ops.input_bindings import recorded_bindings
from engine.v2.ops.lifecycle import verify_fence
from engine.v2.ops.outbox import enqueue, watermark, watermark_would_conflict


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


#: Wall-clock fields ``engine.ledger.score_outcomes`` stamps from
#: ``datetime.now()`` on every row (``resolved_at``, and
#: ``calendar_checked_at`` -- both the same value; see
#: ``engine/ledger.py::score_outcomes``). Nothing else in a settlement row is
#: wall-clock derived (verified against the real attempt-13 candidate:
#: ``exit_finality`` carries no timestamp of its own). Stripping these two
#: gives two rerun lines for the identical underlying determination the same
#: signature even though their wall clocks differ.
_WALL_CLOCK_OUTCOME_FIELDS = ("resolved_at", "calendar_checked_at")


def _content_signature(payload):
    return content_hash({k: v for k, v in payload.items() if k not in _WALL_CLOCK_OUTCOME_FIELDS})


def _existing_outcomes_by_row(conn):
    """Every already-committed outcome decision, grouped by the prediction
    row_id it observes.

    ``generation_ref`` carries the settlement session that committed the row
    (task brief rule 2 -- stamped by ``import_lines``' new ``generation_ref``
    parameter below). Rows committed before this migration (the real
    attempt-13 candidate's 626 admitted lines: verified ``generation_ref IS
    NULL`` on every one) have no durable session recorded there; those are
    matched by content instead -- see ``_match_same_session``.
    """
    index: dict = {}
    for row in conn.execute("SELECT payload_json, generation_ref FROM decisions WHERE kind='outcome'"):
        payload = json.loads(row["payload_json"])
        row_id = payload.get("row_id")
        if not row_id:
            continue
        index.setdefault(row_id, []).append({
            "status": payload.get("status"), "generation_ref": row["generation_ref"],
            "signature": _content_signature(payload),
        })
    return index


def _match_same_session(existing, session, signature):
    """The already-committed observation (if any) that counts as THIS same
    settlement session -- by recorded ``generation_ref`` when the committing
    run stamped one, else (legacy pre-migration rows) by content signature
    against the incoming line, which is the only session evidence those rows
    carry (task brief real-check requirement: derived from committed data
    only)."""
    for entry in existing:
        if entry["generation_ref"] is not None:
            if entry["generation_ref"] == session:
                return entry
        elif entry["signature"] == signature:
            return entry
    return None


def _already_this_observation(conn, row_id, payload):
    """Whether THIS exact (row_id, resolved_at) outcome observation --
    ``decisions._import_decision_id``'s own identity -- is already on file.

    Checked before the rules below run so a byte-identical retry (same
    candidate bytes, same wall clock: every pre-existing idempotency test)
    still falls through unchanged to ``import_lines``' own exact-bytes
    shortcut, instead of being intercepted here as a dedupe skip -- that
    shortcut is what makes a byte-identical retry return the SAME receipts,
    which the new dedupe below (built for a rerun with a NEW wall clock and
    therefore a NEW decision_id) would not reproduce.
    """
    try:
        decision_id = _import_decision_id("outcome", row_id, payload)
    except DecisionConflict:
        return False
    return conn.execute("SELECT 1 FROM decisions WHERE decision_id=?", (decision_id,)).fetchone() is not None


def import_settlement_candidates_in_transaction(conn, claim, candidate_ref, rows, *, clock,
                                                 session=None, on_divergence=None, on_admitted=None,
                                                 on_skip=None):
    """Import only rows captured from the isolated legacy append, under the active fence.

    ``session`` is the finality-resolved date the settlement worker actually
    scored ``through`` (P2-C03) — read off the bound ``settlement.json``
    document, defaulting to the job's REQUESTED ``session`` parameter for a
    caller that predates that field. Also v2's own finality proof for a
    grandfathered resolved line (``_validate_settlement_state``), and stamped
    as ``generation_ref`` on every freshly-committed outcome decision (task
    brief rule 2) -- what a later rerun's same-session dedupe checks against.

    A CONTRACT-mismatched line (``ticker``/``strategy``/``event_date``/
    ``settlement`` differs from the recorded prediction -- the DLNG shape,
    real nightly attempt 10) records a durable ``decision_divergences`` row
    (scope ``legacy_settlement``, guide §5.5 item 1) and is dropped; the
    stage still COMMITS the rest. A line naming no committed prediction at
    all is a different failure (missing contract) and still hard-refuses --
    see ``_settlement_line``.

    Once a line clears the contract check, ``_settlement_dedupe_skip``
    applies task brief rules 1/2/4 -- terminal-resolved, same-session
    dedupe, and same-session status-change divergence -- the fix for a
    same-session rerun's NEW wall-clock ``resolved_at`` otherwise minting a
    brand-new ``decisions.decision_id`` and committing a duplicate
    observation (``decisions._import_decision_id``).

    ``on_divergence(row_id)`` fires for every diverging/status-changed line;
    ``on_admitted(row_id, proof_kind)`` for every line that passes validation
    (``proof_kind``: ``"unresolvable"``, ``"legacy_exit_finality"``,
    ``"v2_finality_session"`` -- see ``_validate_settlement_state``);
    ``on_skip(row_id, reason)`` for every dedupe drop (``reason``:
    ``"already_resolved"``, ``"already_observed_this_session"``). None alter
    the committed ``receipts`` return value.
    """
    if not conn.in_transaction:
        raise ValueError("settlement import requires the attempt transaction")
    verify_fence(conn, claim.attempt_id, claim.fence, clock.now())
    effective_session = session or claim.spec.parameters["session"]
    existing_index = _existing_outcomes_by_row(conn)
    source_lines = []
    for item in rows:
        line = _settlement_line(conn, item, session=effective_session, clock=clock,
                                existing_index=existing_index, on_divergence=on_divergence,
                                on_admitted=on_admitted, on_skip=on_skip)
        if line is not None:
            source_lines.append(line)
    try:
        receipts = import_lines(conn, candidate_ref.content_hash, source_lines, kind="outcome",
                                created_at=format_timestamp(clock.now()), generation_ref=effective_session)
    except DecisionConflict:
        raise fail("IDEMPOTENCY_CONFLICT", "settlement observation conflicts with history") from None
    # P2-C04: settlement uses the same effect scope decision commit and
    # export use, never the bare output_namespace — a subset settlement run
    # must never advance the global watermark either.
    scope = _job_effect_scope(claim)
    release_key = content_hash(["settlement", scope, candidate_ref.content_hash])
    # 2026-09-15 (task brief): a same-session rerun that commits NOTHING new
    # (every line dropped by the rules 1/2/4 dedupe above) must not touch the
    # outbox/watermark either. ``release_key`` is derived from
    # ``candidate_ref.content_hash``, which changes on every rerun --
    # ``engine.ledger.score_outcomes`` stamps a new wall-clock ``resolved_at``
    # even when nothing about the determination changed -- so without this
    # guard a bare no-op rerun would still collide with the FIRST successful
    # pass's already-recorded receipt at ``watermark()``'s own idempotency
    # check (a same-occurrence, different-receipt write is refused by
    # design) and fail the whole attempt instead of being the no-op it is.
    if not source_lines and watermark_would_conflict(
            conn, "nightly", scope, "settlement", effective_session, release_key):
        return receipts
    enqueue(conn, "export", release_key, {"settlement_candidate": candidate_ref.content_hash})
    watermark(conn, "nightly", scope, "settlement", effective_session, release_key, clock=clock)
    return receipts


def _settlement_line(conn, item, *, session, clock, existing_index, on_divergence=None,
                     on_admitted=None, on_skip=None):
    """Validate one captured settlement line; return its original bytes to
    import, or ``None`` when it diverged from the recorded prediction, or was
    dropped by the same-session/terminal-resolved dedupe (a divergence row,
    or nothing at all, was recorded instead; either way the line must not be
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
                                      payload=payload, prediction=prediction, session=session, clock=clock)
        if on_divergence is not None:
            on_divergence(row_id)
        return None

    if _settlement_dedupe_skip(conn, decision_id=decision_id, row_id=row_id, payload=payload,
                               prediction=prediction, existing_index=existing_index, session=session,
                               clock=clock, on_divergence=on_divergence, on_skip=on_skip):
        return None

    proof = _validate_settlement_state(payload, recorded=recorded, session=session)
    if on_admitted is not None:
        on_admitted(row_id, proof)
    return original


def _settlement_dedupe_skip(conn, *, decision_id, row_id, payload, prediction, existing_index, session,
                            clock, on_divergence=None, on_skip=None):
    """Task brief rules 1/2/4 -- terminal-resolved, same-session dedupe, and
    same-session status-change divergence. Split out of ``_settlement_line``
    (function-length/complexity budget). Returns ``True`` when the line must
    NOT be imported (a skip was counted, or a divergence recorded); ``False``
    when it should proceed to ``_validate_settlement_state``.

    Skipped entirely when THIS exact observation is already on file
    (``_already_this_observation``) -- see that function's docstring: a
    byte-identical retry must fall through to ``import_lines``' own
    exact-bytes shortcut unchanged, never be intercepted here.
    """
    if _already_this_observation(conn, row_id, payload):
        return False
    existing = existing_index.get(row_id, ())
    same_session = _match_same_session(existing, session, _content_signature(payload))
    if same_session is not None:
        if same_session["status"] == payload.get("status"):
            if on_skip is not None:
                on_skip(row_id, "already_observed_this_session")
            return True
        _record_status_change_divergence(
            conn, decision_id=decision_id, row_id=row_id, payload=payload, prediction=prediction,
            session=session, clock=clock, previous_status=same_session["status"])
        if on_divergence is not None:
            on_divergence(row_id)
        return True
    if any(entry["status"] == "resolved" for entry in existing):
        if on_skip is not None:
            on_skip(row_id, "already_resolved")
        return True
    return False


def _contract_mismatch_field(payload, recorded):
    for field in ("ticker", "strategy", "event_date", "settlement"):
        if payload.get(field) != recorded.get(field):
            return field
    return None


def _divergence_already_recorded(conn, *, decision_id, occurrence, marker):
    """Whether a ``legacy_settlement`` divergence naming ``marker`` already
    exists for this occurrence -- checked by the stable, human-authored
    ``reason`` text rather than by recomputing ``record_divergence``'s
    content-keyed ``divergence_id``, because the 3 real divergences already
    committed by attempt 13 were keyed off wall-clock ``resolved_at`` (the
    pre-fix scheme) and a freshly-computed session-keyed id would never
    match them -- this check is what keeps a rerun from re-recording those."""
    row = conn.execute(
        "SELECT 1 FROM decision_divergences WHERE decision_id=? AND scope=? AND occurrence=? "
        "AND reason LIKE ?",
        (decision_id, _SETTLEMENT_DIVERGENCE_SCOPE, occurrence, "%" + marker + "%")).fetchone()
    return row is not None


def _record_settlement_divergence(conn, *, decision_id, row_id, field, payload, prediction, session,
                                  clock):
    """Durable evidence that a settlement line disagreed with the recorded
    prediction's contract (guide §5.5 item 1 applied to settlement).

    Idempotent per (row_id, field) -- task brief rule 3, "at most once per
    (row_id, field, session)" applied as its stricter, simpler upper bound:
    a contract mismatch is a property of the recorded prediction versus a
    legacy-ledger fact, not something that legitimately varies by session, so
    recording it once is enough evidence forever. ``attempted_generation_ref``
    is now the settlement ``session`` (never wall-clock ``resolved_at``, the
    2026-09-15 finding that made the OLD keying re-diverge on every rerun),
    but the ``_divergence_already_recorded`` guard above is what actually
    keeps a rerun from adding a new row, since it also covers rows recorded
    under the old wall-clock keying before this fix.
    """
    marker = "field " + field + " differs"
    if _divergence_already_recorded(conn, decision_id=decision_id, occurrence=str(row_id), marker=marker):
        return
    attempted_hash = content_hash({"field": field, "value": payload.get(field), "session": session})
    record_divergence(
        conn, decision_id=decision_id, scope=_SETTLEMENT_DIVERGENCE_SCOPE, occurrence=str(row_id),
        existing_generation_ref=prediction["generation_ref"],
        attempted_generation_ref=str(session),
        existing_payload_hash=prediction["payload_hash"],
        attempted_payload_hash=attempted_hash,
        reason="settlement_contract_mismatch: field " + field + " differs from the recorded "
               "prediction; the first committed prediction stays authoritative (guide §5.5 item 1)",
        created_at=format_timestamp(clock.now()))


def _record_status_change_divergence(conn, *, decision_id, row_id, payload, prediction, session, clock,
                                     previous_status):
    """Task brief rule 4: a rerun for the SAME settlement session that
    produces a DIFFERENT status than what was already committed for that
    session (unresolvable -> resolved, or the reverse) commits nothing and
    records exactly ONE divergence instead -- idempotent per (row_id,
    session) the same way as ``_record_settlement_divergence`` above (a
    fixed reason marker, checked before recording)."""
    if _divergence_already_recorded(conn, decision_id=decision_id, occurrence=str(row_id),
                                    marker="status_change_this_session"):
        return
    new_status = payload.get("status")
    attempted_hash = content_hash({"field": "status", "from": previous_status, "to": new_status,
                                   "session": session})
    record_divergence(
        conn, decision_id=decision_id, scope=_SETTLEMENT_DIVERGENCE_SCOPE, occurrence=str(row_id),
        existing_generation_ref=prediction["generation_ref"],
        attempted_generation_ref=str(session),
        existing_payload_hash=prediction["payload_hash"],
        attempted_payload_hash=attempted_hash,
        reason="settlement_status_change_this_session: recorded status " + str(previous_status)
               + " differs from " + str(new_status) + " for the same settlement session "
               + str(session) + "; the first committed observation for this session stays "
               "authoritative",
        created_at=format_timestamp(clock.now()))


def _stamp_date(value):
    """A bare ``date`` for a date-or-datetime string, or ``None`` -- used
    only to compare a recorded exit date against a settlement session, never
    to reconstruct a timestamp."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def _stamp_date(value):
    """A bare ``date`` for a date-or-datetime string, or ``None`` -- used
    only to compare a recorded exit date against a settlement session, never
    to reconstruct a timestamp."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()
    except (TypeError, ValueError):
        return None


def _validate_settlement_state(payload, *, recorded, session):
    """Refuse a settlement line without a valid observation state, or a
    ``resolved`` one without exit evidence; return the proof kind that
    admitted it (``"unresolvable"``, ``"legacy_exit_finality"`` or
    ``"v2_finality_session"``).

    ``engine.ledger.score_outcomes`` computes ``exit_finality`` (legacy's own
    proof, a ``finality_fn`` result) ONLY for a row whose RECORDED prediction
    has ``schema_version >= SCHEMA_VERSION`` and a recorded exit date;
    "Legacy rows predate this proof and keep their historical retry
    behavior" (``engine/ledger.py`` ``score_outcomes``) -- every such
    grandfathered row settles with ``exit_finality: None`` by construction,
    never because evidence is missing. Refusing those on ``exit_finality``
    (real shadow nightly attempt 12, job ``legacy_settlement``, attempt
    ``att_99875e127ec1b3a446372e31e1ea4521``: all 207 resolved rows,
    ``schema_version`` 2 < ``SCHEMA_VERSION`` 3) is a v2-side gap, not a
    legacy defect -- legacy's own finality proof was never computed for
    these rows in the first place.

    v2 supplies its OWN proof for exactly that grandfathered case instead of
    weakening the check: the RECORDED prediction's exit date (never the
    settlement candidate's own ``schema_version``, which ``score_outcomes``
    always stamps as the CURRENT ``SCHEMA_VERSION`` regardless of the source
    row's vintage) must fall on or before ``session`` -- the settlement's own
    finality-resolved session, the same date ``import_settlement_candidates_in_transaction``
    already receives off the bound ``settlement.json`` document. A session
    this settlement was scored ``through`` cannot itself be non-final, so an
    exit on or before it is exactly as final as legacy's own
    ``finality_fn`` proof would have found -- just derived from data v2
    already holds instead of a value legacy never computed. A current-schema
    row (``schema_version >= SCHEMA_VERSION``) keeps the unweakened original
    check: legacy's own ``exit_finality.is_final is True`` is still
    required, exactly as before.
    """
    if payload.get("status") not in ("resolved", "unresolvable") or not payload.get("resolved_at"):
        raise fail("VALIDATION_FAILED", "settlement has no valid observation state")
    if payload["status"] != "resolved":
        return "unresolvable"
    if not payload.get("settlement_source") or not payload.get("exit_source"):
        raise fail("VALIDATION_FAILED", "resolved settlement lacks recorded exit evidence")
    from engine.v2.ops.legacy_adapter import legacy_ledger_schema_version

    if int((recorded or {}).get("schema_version") or 0) >= legacy_ledger_schema_version():
        return _require_legacy_exit_finality(payload)
    return _require_v2_finality_session(recorded, session)


def _require_legacy_exit_finality(payload):
    """Current-schema proof, unchanged: legacy's own ``exit_finality`` must
    say the exit session is final."""
    finality = payload.get("exit_finality")
    if not isinstance(finality, dict) or finality.get("is_final") is not True:
        raise fail("VALIDATION_FAILED", "resolved settlement lacks recorded exit evidence")
    return "legacy_exit_finality"


def _require_v2_finality_session(recorded, session):
    """Grandfathered proof: the RECORDED prediction's own exit date, at or
    before the settlement's finality-resolved ``session``."""
    structure = (recorded or {}).get("structure") or {}
    score = (recorded or {}).get("score") or {}
    exit_date = _stamp_date(structure.get("exit_date") or score.get("exit_date"))
    session_date = _stamp_date(session)
    if exit_date is None or session_date is None or exit_date > session_date:
        raise fail("VALIDATION_FAILED", "resolved settlement lacks recorded exit evidence")
    return "v2_finality_session"
