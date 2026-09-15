"""Shadow nightly attempt 18 (job_1499e0fe80be77652b9de9aaeeff241f,
``legacy_decisions``) failed ``VALIDATION_FAILED`` with exactly one finding,
``evidence.selection.eligible_candidate_keys: mismatch``, on a run with a
non-empty candidate set (18 keys) that agreed on content but not order.

Root cause: ``decision_evidence.derive`` builds every population-derived
receipt field from ``decision_replay.decision_population``, which returns
rows **sorted by population key** (see its docstring). That sorted order is
what ``coverage.observed_population``, ``replay.source_rows`` and
``selection.eligible_candidate_keys`` all carry. But the legacy adapter's
``decisions.json`` (``_action_decisions``, via ``engine.ledger.
build_prediction_rows``) emits candidates in score/panel order, not sorted
by population key. ``decision_validation._validate_rows`` built ``actual``
straight from that candidate order and compared it, via ``_same`` (an
order-sensitive ``canonical_json`` equality), against the receipt's sorted
list -- refusing a real run whose candidate SET was correct.

The fix sorts ``actual`` by population key before that one comparison
(``decision_validation.py::_validate_evidence``), matching the canonical
order every other population-derived receipt already uses. Every other
comparison in this module (population set/duplicate checks, replay,
coverage) was already order-independent or already built from the same
sorted source; this was the one latent spot.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.foundation import artifact_reference, canonical_json, content_hash
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.errors import OpsError

SESSION = "2026-09-11"
CLOCK = SESSION + "T21:00:00+00:00"


def _finality(date=SESSION):
    return {"date": date, "is_final": True, "market_wide": True,
            "daily_share": 1.0, "chain_share": 1.0, "covered": 2}


def _score_row(ticker):
    return {"ticker": ticker, "event_id": "event-" + ticker.lower(),
            "event_date": SESSION, "as_of": SESSION, "entry_date": SESSION,
            "evidence_cutoff": SESSION, "strategy": "TWIN-P", "strike": 100.0,
            "expiry": "2026-10-16", "session": "AMC",
            "snapshot_hash": "sha256:" + ticker.lower() * 16, "strike_offset": None,
            "is_ladder": False}


def _candidate(row, finality):
    return {"row_id": row["ticker"] + "-row", "event_id": row["event_id"],
            "ticker": row["ticker"], "event_date": row["event_date"],
            "strategy": row["strategy"], "as_of": row["as_of"],
            "written_at": CLOCK, "decision_ts": CLOCK,
            "snapshot_hash": row["snapshot_hash"], "finality": finality,
            "score": row}


def _bound():
    """Build plan/evidence with the REAL ``decision_evidence.derive`` over
    two rows whose population-key sort order (AAA before ZZZ) differs from
    their score-document order (ZZZ first) -- mirroring the real
    ``decisions.json`` (panel order) vs receipt (sorted) mismatch.
    """
    row_zzz, row_aaa = _score_row("ZZZ"), _score_row("AAA")
    score_doc = {"rows": [row_zzz, row_aaa]}
    finality = _finality()
    score_bytes = json.dumps(score_doc, sort_keys=True).encode("utf-8")
    finality_bytes = json.dumps(finality, sort_keys=True).encode("utf-8")
    score_ref = artifact_reference(score_bytes, "legacy_action.v1.0")
    finality_ref = artifact_reference(finality_bytes, "legacy_action.v1.0")
    # decision_population sorts by population key: AAA|TWIN-P|SESSION < ZZZ|...
    sorted_rows = [row_aaa, row_zzz]
    keys = ["AAA|TWIN-P|" + SESSION, "ZZZ|TWIN-P|" + SESSION]
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "requested_session": SESSION, "population": keys,
                  "source_rows": sorted_rows, "replayed_rows": sorted_rows,
                  "source_rows_hash": content_hash(sorted_rows),
                  "replayed_rows_hash": content_hash(sorted_rows), "findings": []}
    coverage_doc = {"schema_version": "finality_coverage.v1.0", "date": SESSION,
                    "covered_tickers": ["ZZZ", "AAA"]}
    plan_bytes, evidence_bytes = derive(
        score_doc, score_ref, finality, finality_ref, replay_doc, coverage_doc,
        requested_session=SESSION, deployment="shadow:test-impl", decision_clock=CLOCK)
    plan = json.loads(plan_bytes)
    evidence = json.loads(evidence_bytes)
    assert plan["expected_population"] == keys  # receipt/plan carry SORTED order
    plan_ref = artifact_reference(plan_bytes, "decision_plan.v1.0")
    evidence_ref = artifact_reference(evidence_bytes, "decision_evidence.v1.0")
    bindings = {"score": {"artifact_id": score_ref.artifact_id,
                          "content_hash": score_ref.content_hash},
                "finality": {"artifact_id": finality_ref.artifact_id,
                            "content_hash": finality_ref.content_hash},
                "plan": {"artifact_id": plan_ref.artifact_id,
                        "content_hash": plan_ref.content_hash},
                "evidence": {"artifact_id": evidence_ref.artifact_id,
                            "content_hash": evidence_ref.content_hash}}
    # candidates in SCORE-DOC (non-sorted) order: ZZZ before AAA.
    candidates = [_candidate(row_zzz, finality), _candidate(row_aaa, finality)]
    return score_doc, finality, plan, evidence, bindings, candidates


def test_candidates_in_non_sorted_order_validate_against_sorted_receipt():
    """Real-shaped end-to-end check (>=2 candidates, non-sorted order) --
    this is exactly the attempt-18 shape. Must not raise.
    """
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    result = validate(candidates, score=score_doc, finality=finality, plan=plan,
                      evidence=evidence, bindings=bindings)
    assert result["session"] == SESSION


def test_duplicate_candidate_still_refuses():
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    with pytest.raises(OpsError) as err:
        validate(candidates + [candidates[0]], score=score_doc, finality=finality,
                 plan=plan, evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    fields = {finding["field"] for finding in err.value.problem.details["findings"]}
    assert "candidates.population" in fields


def test_missing_candidate_still_refuses():
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    with pytest.raises(OpsError) as err:
        validate(candidates[:1], score=score_doc, finality=finality, plan=plan,
                 evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    fields = {finding["field"] for finding in err.value.problem.details["findings"]}
    assert "candidates.population" in fields


def test_extra_candidate_still_refuses():
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    extra_row = _score_row("BBB")
    extra = _candidate(extra_row, finality)
    with pytest.raises(OpsError) as err:
        validate(candidates + [extra], score=score_doc, finality=finality, plan=plan,
                 evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    fields = {finding["field"] for finding in err.value.problem.details["findings"]}
    assert "candidates.population" in fields


def test_selection_receipt_content_mismatch_still_refuses():
    """The order fix must not become a content-blind check: a receipt whose
    key SET is wrong (not just its order) still refuses.
    """
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    tampered = json.loads(json.dumps(evidence))
    tampered["receipts"]["selection"]["eligible_candidate_keys"] = ["AAA|TWIN-P|" + SESSION]
    with pytest.raises(OpsError) as err:
        validate(candidates, score=score_doc, finality=finality, plan=plan,
                 evidence=tampered, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    fields = {finding["field"] for finding in err.value.problem.details["findings"]}
    assert "evidence.selection.eligible_candidate_keys" in fields
