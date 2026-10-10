"""Carried-set filtering regression, against the REAL producer and the REAL validator.

The plan comes from ``workflows.commands._plan_command`` over the real snapshot fixture (only the
pin step is stubbed, by the imported ``_plan`` helper); the plan and evidence
documents are the real ``decision_evidence.derive`` outputs and every refusal is
the real ``decision_validation.validate`` raising — no finality result is stubbed.
"""
from __future__ import annotations

import json

import pytest

from engine.v2.foundation import artifact_reference, content_hash
from engine.v2.foundation.score_population import population_key
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.errors import OpsError
from tests.v2.ops.test_generated_population import _AS_OF, _keys, _plan, env  # noqa: F401  (env is a fixture)

SESSION = _AS_OF            # "2026-12-20"
CLOCK = SESSION + "T21:00:00+00:00"


def _row(key):
    ticker, strategy, event_date = key.split("|")
    return {"ticker": ticker, "event_id": f"event-{ticker}-{strategy}-{event_date}",
            "event_date": event_date, "as_of": SESSION, "entry_date": SESSION,
            "evidence_cutoff": SESSION, "strategy": strategy, "strike": 100.0,
            "expiry": "2027-02-19", "session": "AMC",
            "snapshot_hash": "sha256:" + ticker.lower() * 16,
            "strike_offset": None, "is_ladder": False}


def _candidate(row, finality):
    return {"row_id": row["event_id"] + "-row", "event_id": row["event_id"],
            "ticker": row["ticker"], "event_date": row["event_date"],
            "strategy": row["strategy"], "as_of": row["as_of"], "written_at": CLOCK,
            "decision_ts": CLOCK, "snapshot_hash": row["snapshot_hash"],
            "finality": finality, "score": row}


def _bound(keys, covered):
    rows = [_row(k) for k in keys]
    score_doc = {"rows": rows}
    finality = {"date": SESSION, "is_final": True, "market_wide": True,
                "daily_share": 1.0, "chain_share": 1.0, "covered": len(covered)}
    score_ref = artifact_reference(json.dumps(score_doc, sort_keys=True).encode("utf-8"),
                                   "legacy_action.v1.0")
    finality_ref = artifact_reference(json.dumps(finality, sort_keys=True).encode("utf-8"),
                                      "legacy_action.v1.0")
    sorted_rows = sorted(rows, key=population_key)
    population = [population_key(r) for r in sorted_rows]
    replay_doc = {"schema_version": "decision_replay.v1.0", "session": SESSION,
                  "requested_session": SESSION, "population": population,
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
    candidates = [_candidate(r, finality) for r in rows]
    kwargs = dict(score=score_doc, finality=finality, plan=plan, evidence=evidence,
                  bindings=bindings)
    return candidates, kwargs


def test_uncarried_ticker_is_absent_from_candidates_and_cannot_fail_validation(env, monkeypatch):
    plan, _ = _plan(env, monkeypatch)
    keys = plan["expected_population"]
    assert not any(key.startswith("EEE|") for key in keys)
    assert plan["candidate_exclusions"] == [{
        "ticker": "EEE", "reason_code": "UNCARRIED_TICKER",
        "missing_tables": ["daily_market", "option_chains"],
    }]

    candidates, kwargs = _bound(keys, ["AAA", "BBB"])
    assert {c["ticker"] for c in candidates} == {"AAA", "BBB"}
    result = validate(candidates, **kwargs)
    assert result["session"] == SESSION


def test_the_same_uncarried_ticker_unfiltered_would_refuse(env, monkeypatch):
    plan, _ = _plan(env, monkeypatch)
    keys = [*plan["expected_population"], *_keys("EEE", "2027-01-03")]

    candidates, kwargs = _bound(keys, ["AAA", "BBB"])
    with pytest.raises(OpsError) as err:
        validate(candidates, **kwargs)
    assert err.value.code == "VALIDATION_FAILED"
    assert {"field": "evidence.finality.covered_tickers", "reason": "missing_candidate"} \
        in err.value.problem.details["findings"]


def test_carried_ticker_without_final_coverage_still_refuses(env, monkeypatch):
    plan, _ = _plan(env, monkeypatch)
    keys = plan["expected_population"]

    candidates, kwargs = _bound(keys, ["AAA"])  # BBB is carried but not final
    with pytest.raises(OpsError) as err:
        validate(candidates, **kwargs)
    assert err.value.code == "VALIDATION_FAILED"
    assert {"field": "evidence.finality.covered_tickers", "reason": "missing_candidate"} \
        in err.value.problem.details["findings"]
