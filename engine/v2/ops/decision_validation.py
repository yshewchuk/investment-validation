"""Pure validation of copy-only legacy decision candidates and evidence."""
from __future__ import annotations

from datetime import datetime, timezone

from engine.v2.foundation import canonical_json, content_hash
from engine.v2.ops.errors import fail

REQUIRED = frozenset({"causality", "coverage", "finality", "selection", "replay"})


def population_key(row):
    return "|".join(str(row.get(key, "")) for key in ("ticker", "strategy", "event_date"))


def _same(left, right):
    return canonical_json(left) == canonical_json(right)


def _stamp(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _add(findings, field, reason):
    findings.append({"field": field, "reason": reason})


def _validate_plan(plan, findings):
    """Return ``plan.expected_population`` as a list, or ``None`` if the plan
    does not even carry a well-typed one.

    An empty list is a legitimate population (a no-entry night, review fix
    #2) — only an ABSENT or wrongly-typed ``expected_population`` is
    "missing"; every other required plan field still may not be empty.
    """
    if plan.get("schema_version") != "decision_plan.v1.0":
        _add(findings, "plan.schema_version", "unsupported")
    for field in ("session", "deployment", "decision_clock"):
        if not plan.get(field):
            _add(findings, "plan." + field, "missing")
    population = plan.get("expected_population")
    if not isinstance(population, list) or not all(isinstance(item, str) for item in population):
        _add(findings, "plan.expected_population", "missing")
        population = None
    expected = list(population or ())
    if len(expected) != len(set(expected)):
        _add(findings, "plan.expected_population", "duplicate")
    if _stamp(plan.get("decision_clock")) is None:
        _add(findings, "plan.decision_clock", "invalid")
    return expected


def _validate_finality(finality, plan, findings):
    if finality.get("date") != plan.get("session") or finality.get("is_final") is not True:
        _add(findings, "finality", "not_final_for_session")
    try:
        covered = int(finality.get("covered") or 0)
    except (TypeError, ValueError):
        covered = 0
    if finality.get("market_wide") is not True or covered <= 0:
        _add(findings, "finality", "insufficient_coverage")
    for field in ("daily_share", "chain_share"):
        try:
            if float(finality.get(field)) < 0.80 or float(finality.get(field)) > 1:
                _add(findings, "finality." + field, "outside_finality_floor")
        except (TypeError, ValueError):
            _add(findings, "finality." + field, "invalid")


def _validate_receipt_bindings(receipts, bindings, plan, findings):
    required = {
        "score_artifact_id": bindings["score"]["artifact_id"],
        "score_content_hash": bindings["score"]["content_hash"],
        "finality_artifact_id": bindings["finality"]["artifact_id"],
        "finality_content_hash": bindings["finality"]["content_hash"],
        "plan_artifact_id": bindings["plan"]["artifact_id"],
        "plan_content_hash": bindings["plan"]["content_hash"],
        "session": plan["session"], "deployment": plan["deployment"],
        "decision_clock": plan["decision_clock"], "expected_population": plan["expected_population"],
    }
    for kind in sorted(REQUIRED):
        receipt = receipts.get(kind)
        if not isinstance(receipt, dict):
            _add(findings, "evidence." + kind, "missing")
            continue
        if receipt.get("schema_version") != "decision_receipt.v1.0" or receipt.get("kind") != kind:
            _add(findings, "evidence." + kind, "unsupported_receipt")
        for field, expected in required.items():
            if not _same(receipt.get(field), expected):
                _add(findings, "evidence." + kind + "." + field, "binding_mismatch")


def _validate_no_eligible_rows(score, plan, findings):
    """Review fix #2: an empty ``expected_population`` is trusted only after
    independently recomputing the decision-eligible population from the
    BOUND score document and finding it genuinely empty too — never merely
    because the plan and the evidence receipt happen to agree with each
    other on an unverified claim.
    """
    from engine.v2.ops.decision_replay import decision_population
    try:
        population = decision_population(score, plan.get("session"))
    except Exception:
        _add(findings, "plan.expected_population", "population_recompute_failed")
        return
    if population:
        _add(findings, "plan.expected_population", "eligible_rows_exist")


def _validate_rows(candidates, score, finality, plan, expected, findings):
    score_rows = score.get("rows") if isinstance(score, dict) else None
    if not isinstance(score_rows, list) or not all(isinstance(row, dict) for row in score_rows):
        _add(findings, "score.rows", "missing_or_malformed")
        score_rows = []
    source = {population_key(row): row for row in score_rows}
    actual = [population_key(row) for row in candidates]
    if len(source) != len(score_rows) or set(actual) != set(expected) or len(actual) != len(set(actual)):
        _add(findings, "candidates.population", "expected_population_mismatch")
    if not expected:
        _validate_no_eligible_rows(score, plan, findings)
    for index, row in enumerate(candidates):
        _validate_candidate(row, index, source.get(population_key(row)), finality, plan, findings)
    return source, actual


def _validate_candidate(row, index, source_row, finality, plan, findings):
    prefix = "candidates[" + str(index) + "]"
    if not row.get("row_id") or not row.get("event_id"):
        _add(findings, prefix + ".identity", "missing")
    clocks = (row.get("as_of") == plan.get("session"),
              row.get("written_at") == plan.get("decision_clock"),
              row.get("decision_ts") == plan.get("decision_clock"))
    if not all(clocks):
        _add(findings, prefix + ".entry_or_clock", "not_pinned")
    if source_row is None or not _same(row.get("score"), source_row):
        _add(findings, prefix + ".score", "source_mismatch")
        return
    entry = source_row.get("entry_date")
    if entry not in (None, "", plan.get("session")) or source_row.get("as_of") != plan.get("session"):
        _add(findings, prefix + ".entry_date", "ineligible")
    if not row.get("snapshot_hash") or row.get("snapshot_hash") != source_row.get("snapshot_hash"):
        _add(findings, prefix + ".snapshot_hash", "source_mismatch")
    score = row.get("score", {})
    if row.get("finality") != finality or score.get("is_ladder") or score.get("ladder"):
        _add(findings, prefix + ".finality_or_ladder", "not_eligible")


def _validate_evidence(receipts, source, actual, expected, finality, findings):
    _validate_causality(receipts.get("causality", {}), source, findings)
    if not _same(receipts.get("coverage", {}).get("observed_population"), expected):
        _add(findings, "evidence.coverage.observed_population", "mismatch")
    _validate_finality_receipt(receipts.get("finality", {}), source, finality, findings)
    if not _same(receipts.get("selection", {}).get("eligible_candidate_keys"), actual):
        _add(findings, "evidence.selection.eligible_candidate_keys", "mismatch")
    _validate_replay(receipts.get("replay", {}), source, expected, findings)


def _validate_causality(causal, source, findings):
    for key, row in source.items():
        cutoff, actual_cutoff = _stamp(causal.get("observed_cutoffs", {}).get(key)), _stamp(row.get("evidence_cutoff"))
        row_as_of = _stamp(row.get("as_of"))
        if (cutoff is None or actual_cutoff is None or row_as_of is None
                or cutoff != actual_cutoff or cutoff > row_as_of):
            _add(findings, "evidence.causality." + key, "unbound_or_late_cutoff")


def _validate_finality_receipt(receipt, source, finality, findings):
    covered = receipt.get("covered_tickers")
    if (receipt.get("observed_finality_hash") != content_hash(finality)
            or not isinstance(covered, list) or not all(isinstance(item, str) for item in covered)):
        _add(findings, "evidence.finality", "unbound")
    if not set(row.get("ticker") for row in source.values()).issubset(set(covered or ())):
        _add(findings, "evidence.finality.covered_tickers", "missing_candidate")


def _validate_replay(replay, source, expected, findings):
    source_rows = [source[key] for key in expected if key in source]
    if not _same(replay.get("source_rows"), source_rows) or not _same(replay.get("replayed_rows"), source_rows) or replay.get("source_rows_hash") != content_hash(source_rows) or replay.get("replayed_rows_hash") != content_hash(source_rows) or replay.get("findings") != []:
        _add(findings, "evidence.replay", "no_verified_agreement")


def validate(candidates, *, score, finality, plan, evidence, bindings):
    """Return a bound commit context or fail without catalog mutation."""
    findings = []
    if not isinstance(plan, dict):
        plan = {}
        _add(findings, "plan", "malformed")
    if not isinstance(finality, dict):
        finality = {}
        _add(findings, "finality", "malformed")
    if not isinstance(score, dict):
        score = {}
        _add(findings, "score", "malformed")
    if not isinstance(candidates, list) or not all(isinstance(row, dict) for row in candidates):
        candidates = []
        _add(findings, "candidates", "malformed")
    expected = _validate_plan(plan, findings)
    _validate_finality(finality, plan, findings)
    receipts = evidence.get("receipts") if isinstance(evidence, dict) else {}
    if not isinstance(evidence, dict) or evidence.get("schema_version") != "decision_evidence.v1.0":
        _add(findings, "evidence.schema_version", "unsupported")
    if not isinstance(receipts, dict):
        receipts = {}
        _add(findings, "evidence.receipts", "missing_or_malformed")
    _validate_receipt_bindings(receipts, bindings, plan, findings)
    source, actual = _validate_rows(candidates, score, finality, plan, expected, findings)
    _validate_evidence(receipts, source, actual, expected, finality, findings)
    if findings:
        raise fail("VALIDATION_FAILED", "candidate decisions failed validation",
                   details={"stage": "decision_validation", "findings": findings})
    return {"purpose": "shadow", "scope": "shadow", "session": plan["session"],
            "deployment": plan["deployment"], "clock": plan["decision_clock"],
            "input_hash": bindings["score"]["content_hash"], "validations": receipts,
            "candidate_rows_hash": content_hash(candidates), "bindings": bindings,
            "plan_artifact_id": bindings["plan"]["artifact_id"],
            "evidence_artifact_id": bindings["evidence"]["artifact_id"]}
