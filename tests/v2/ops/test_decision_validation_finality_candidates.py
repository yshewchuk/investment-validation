"""Finality coverage is required of the CANDIDATE tickers only.

``_validate_finality_receipt`` used to demand that every ticker in the bound
score document (``source``) appear in ``covered_tickers`` -- but a score
document can carry rows that were scored with no window at all (as_of/
entry_date/evidence_cutoff all None) and were therefore never decision
candidates for the session (mirroring real shadow nightly attempt 14). Such
a row can legitimately lack a final session, so demanding its coverage
refused real runs. Coverage is now required of the candidate tickers only
(the tickers of the validated ``actual`` population); the ``unbound``
receipt-shape check is unchanged.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.foundation import artifact_reference, content_hash
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.decision_validation import _validate_finality_receipt, validate
from engine.v2.ops.errors import OpsError
from tests.test_v2_ops_decision_validation_order import (
    CLOCK, SESSION, _bound, _candidate, _finality, _score_row)

FINALITY = {"date": SESSION, "is_final": True, "market_wide": True,
            "daily_share": 1.0, "chain_share": 1.0, "covered": 2}
_UNSET = object()


def _receipt(covered=_UNSET):
    receipt = {"observed_finality_hash": content_hash(FINALITY)}
    if covered is not _UNSET:
        receipt["covered_tickers"] = covered
    return receipt


def _bound_with_extra_row(covered=("ZZZ", "AAA")):
    """``_bound`` with a THIRD score row NNN whose ``as_of``/``entry_date``/
    ``evidence_cutoff`` are all None -- scored, but never a decision
    candidate (``decision_population`` excludes it), so the plan/evidence
    and the candidate list carry only AAA and ZZZ.
    """
    row_zzz, row_aaa = _score_row("ZZZ"), _score_row("AAA")
    row_nnn = _score_row("NNN")
    row_nnn["as_of"] = row_nnn["entry_date"] = row_nnn["evidence_cutoff"] = None
    score_doc = {"rows": [row_zzz, row_aaa, row_nnn]}
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
                    "covered_tickers": list(covered)}
    plan_bytes, evidence_bytes = derive(
        score_doc, score_ref, finality, finality_ref, replay_doc, coverage_doc,
        requested_session=SESSION, deployment="shadow:test-impl", decision_clock=CLOCK)
    plan = json.loads(plan_bytes)
    evidence = json.loads(evidence_bytes)
    assert plan["expected_population"] == keys  # NNN stays out of the population
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
    candidates = [_candidate(row_zzz, finality), _candidate(row_aaa, finality)]
    return score_doc, finality, plan, evidence, bindings, candidates


def test_uncovered_non_candidate_is_tolerated():
    findings = []
    _validate_finality_receipt(_receipt(["AAA"]), {"AAA"}, FINALITY, findings)
    assert findings == []


def test_uncovered_candidate_refuses():
    findings = []
    _validate_finality_receipt(_receipt(["AAA"]), {"AAA", "BBB"}, FINALITY, findings)
    assert findings == [{"field": "evidence.finality.covered_tickers",
                         "reason": "missing_candidate"}]


def test_empty_candidates_with_well_formed_covered_passes():
    findings = []
    _validate_finality_receipt(_receipt([]), set(), FINALITY, findings)
    assert findings == []
    _validate_finality_receipt(_receipt(["AAA"]), set(), FINALITY, findings)
    assert findings == []


@pytest.mark.parametrize("covered", [_UNSET, None, "AAA", [1]])
def test_missing_or_malformed_covered_refuses_with_candidates(covered):
    findings = []
    _validate_finality_receipt(_receipt(covered), {"AAA"}, FINALITY, findings)
    assert {"field": "evidence.finality", "reason": "unbound"} in findings
    assert {"field": "evidence.finality.covered_tickers",
            "reason": "missing_candidate"} in findings


def test_missing_covered_with_empty_candidates_is_not_vacuous():
    findings = []
    _validate_finality_receipt(_receipt(), set(), FINALITY, findings)
    assert findings == [{"field": "evidence.finality", "reason": "unbound"}]


def test_full_route_uncovered_non_candidate_score_row_passes():
    """A scored row that is not a decision candidate (NNN) may stay out of
    ``covered_tickers``; only the two real candidates need final coverage.
    """
    score_doc, finality, plan, evidence, bindings, candidates = _bound_with_extra_row()
    result = validate(candidates, score=score_doc, finality=finality, plan=plan,
                      evidence=evidence, bindings=bindings)
    assert result["session"] == SESSION


def test_full_route_uncovered_candidate_refuses():
    score_doc, finality, plan, evidence, bindings, candidates = _bound_with_extra_row(
        covered=("AAA",))
    with pytest.raises(OpsError) as err:
        validate(candidates, score=score_doc, finality=finality, plan=plan,
                 evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    findings = err.value.problem.details["findings"]
    assert {"field": "evidence.finality.covered_tickers",
            "reason": "missing_candidate"} in findings


def test_is_final_and_share_findings_unchanged():
    """The retargeting must not weaken ``_validate_finality``'s own checks."""
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    with pytest.raises(OpsError) as err:
        validate(candidates, score=score_doc, finality={**finality, "is_final": False},
                 plan=plan, evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    assert any(finding["field"].startswith("finality")
               for finding in err.value.problem.details["findings"])
    score_doc, finality, plan, evidence, bindings, candidates = _bound()
    with pytest.raises(OpsError) as err:
        validate(candidates, score=score_doc,
                 finality={**finality, "daily_share": float("nan")}, plan=plan,
                 evidence=evidence, bindings=bindings)
    assert err.value.code == "VALIDATION_FAILED"
    assert {"field": "finality.daily_share", "reason": "invalid"} in \
        err.value.problem.details["findings"]
