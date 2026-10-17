"""Finite receipt reads preserve historical coordinator failures without rewriting them."""
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import Problem
from engine.v2.foundation import artifact_reference, canonical_json
from engine.v2.ops import submission
from engine.v2.ops.catalog import dumps
from engine.v2.ops.decision_evidence import derive
from engine.v2.ops.decision_validation import validate
from engine.v2.ops.errors import OpsError
from engine.v2.ops.migrations import migrate
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor
from engine.v2.ops.schema import MIGRATIONS, OWNER
from tests.ops_support import FakeClock
from tests.v2.ops.native_bounds_support import BoundedSqlRows
from tools.v2_ops_fixtures import POLICY, REGISTRY, enqueue_claim, request

CAP = 1024 * 1024
DEFAULT_CAP = 8 * CAP


def coordinator_inputs():
    """A broad board with missing cutoffs; no arbitrary oversized Problem.

    Score boards cover events times strategies, including rows whose entry
    day is later than this session. Evidence derivation accepts those rows;
    the decision coordinator checks every non-null as_of's causality even
    when today's candidate population is empty. Artifact contents never
    travel through the worker's 1 MiB result pipe.
    """
    session = "2026-09-12"
    strategies = ("CAL-P", "STR-THRU", "STR-RUNUP", "CND-P", "TWIN-P5", "CND-PS",
                  "BFLY-P5", "RAMP7")
    score = {"rows": [
        {"ticker": f"T{i:04}", "strategy": strategy, "event_date": "2026-09-18",
         "as_of": "2026-09-15", "entry_date": "2026-09-15", "evidence_cutoff": None}
        for i in range(1500) for strategy in strategies
    ]}
    finality = {"date": session, "is_final": True, "market_wide": True,
                "covered": 1500, "daily_share": 1.0, "chain_share": 1.0}
    score_ref = artifact_reference(canonical_json(score).encode(), "legacy_action.v1.0")
    finality_ref = artifact_reference(canonical_json(finality).encode(), "legacy_action.v1.0")
    plan_bytes, evidence_bytes = derive(
        score, score_ref, finality, finality_ref, {"source_rows": [], "replayed_rows": []},
        {"schema_version": "finality_coverage.v1.0", "date": session, "covered_tickers": []},
        requested_session=session, deployment="test", decision_clock=session + "T20:00:00Z")
    refs = dict(score=score_ref, finality=finality_ref,
                plan=artifact_reference(plan_bytes, "decision_plan.v1.0"),
                evidence=artifact_reference(evidence_bytes, "decision_evidence.v1.0"))
    return dict(score=score, finality=finality, plan=json.loads(plan_bytes),
                evidence=json.loads(evidence_bytes), bindings={
                    key: {"artifact_id": ref.artifact_id, "content_hash": ref.content_hash}
                    for key, ref in refs.items()})


def coordinator_failure():
    """Produce the actual 12,000-finding Problem with no workers or market data."""
    with pytest.raises(OpsError) as captured:
        validate([], **coordinator_inputs())
    problem = captured.value.problem
    assert problem.code == "VALIDATION_FAILED" and not problem.retryable
    assert len(problem.details["findings"]) == 12000
    assert {item["reason"] for item in problem.details["findings"]} == {
        "unbound_or_late_cutoff"}
    assert CAP < len(dumps(problem).encode()) < DEFAULT_CAP
    return problem


@contextmanager
def historical_failure(problem, *, encoding="UTF-8"):
    """Seed rows an older writer could commit; never bypass the new writer in production."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(f"PRAGMA encoding='{encoding}'")
    try:
        clock = FakeClock()
        migrate(conn, OWNER, MIGRATIONS, clock=clock)
        epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
        claim = enqueue_claim(conn, clock, Supervisor(epoch, "boot"))
        payload = dumps(problem)
        conn.execute("UPDATE jobs SET state='failed', active_attempt_id=NULL, failure_json=?",
                     (payload,))
        conn.execute("UPDATE attempts SET state='failed', process_state='verified_dead', "
                     "failure_json=?", (payload,))
        for table in ("jobs", "attempts"):
            row = conn.execute(f"SELECT state, failure_json FROM {table}").fetchone()
            assert tuple(row) == ("failed", payload)
        yield SimpleNamespace(conn=conn, clock=clock, job_id=claim.job_id,
                              problem=problem, payload=payload)
    finally:
        conn.close()


@pytest.fixture(scope="module")
def large_failed_job():
    with historical_failure(coordinator_failure()) as case:
        yield case


@pytest.fixture(scope="module")
def over_default_problem():
    return replace(coordinator_failure(), message="historical diagnostic " + "x" * DEFAULT_CAP)


def read_receipt(case, conn, reader, **options):
    if reader == "get-job":
        return submission.get_job(conn, case.job_id, **options)
    return submission.submit_graph(conn, REGISTRY, POLICY, [request()],
                                   clock=case.clock, **options)[0]


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("options", [{}, {"max_failure_bytes": None}], ids=["omitted", "none"])
def test_general_receipts_preserve_large_coordinator_failure(large_failed_job, reader, options):
    case = large_failed_job
    assert submission.DEFAULT_JOB_FAILURE_BYTES == DEFAULT_CAP
    receipt = read_receipt(case, case.conn, reader, **options)
    assert receipt.state == "failed" and receipt.failure == case.problem
    assert not case.conn.in_transaction


def test_identical_submission_preserves_large_coordinator_failure(large_failed_job):
    case = large_failed_job
    receipt = submission.submit(case.conn, REGISTRY, POLICY, request(), clock=case.clock)
    assert receipt.job_id == case.job_id and receipt.failure == case.problem
    assert case.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert case.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
    assert not case.conn.in_transaction


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
def test_explicit_failure_bound_refuses_before_transfer_and_decode(
        large_failed_job, monkeypatch, reader):
    case = large_failed_job
    bounded = BoundedSqlRows(case.conn, max_bytes=CAP, column="failure_json",
                             sql_fragment="end as failure_json from jobs")
    original_load = submission.load_json

    def guarded_load(cls, value):
        if cls is Problem:
            pytest.fail("oversized coordinator failure reached JSON decoding")
        return original_load(cls, value)

    monkeypatch.setattr(submission, "load_json", guarded_load)
    with pytest.raises(OpsError) as captured:
        read_receipt(case, bounded, reader, max_failure_bytes=CAP,
                     receipt_decode_failure_code="EXPERIMENT_IDENTITY_CONFLICT")
    assert captured.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert not captured.value.problem.retryable
    assert captured.value.problem.message == "stored job failure exceeds its byte bound"
    assert bounded.sizes == [None]
    assert not case.conn.in_transaction
    for table in ("jobs", "attempts"):
        assert case.conn.execute(f"SELECT failure_json FROM {table}").fetchone()[0] == case.payload


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("options", [{}, {"max_failure_bytes": None}], ids=["omitted", "none"])
@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le"])
def test_default_limit_refuses_before_transfer_and_finite_budget_recovers(
        over_default_problem, monkeypatch, reader, options, encoding):
    with historical_failure(over_default_problem, encoding=encoding) as case:
        bounded = BoundedSqlRows(case.conn, max_bytes=DEFAULT_CAP * (2 if encoding == "UTF-16le" else 1),
                                 column="failure_json", sql_fragment="end as failure_json from jobs")
        original_load = submission.load_json

        def guarded_load(cls, value):
            if cls is Problem:
                pytest.fail("failure above the default limit reached JSON decoding")
            return original_load(cls, value)

        with monkeypatch.context() as patch:
            patch.setattr(submission, "load_json", guarded_load)
            with pytest.raises(OpsError) as captured:
                read_receipt(case, bounded, reader, **options)
        assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
        assert captured.value.problem.retryable is False
        assert captured.value.problem.message == "stored job failure exceeds its byte bound"
        assert bounded.sizes == [None]
        assert not case.conn.in_transaction

        # An explicit finite recovery allowance reads the original envelope.
        budget = len(case.payload.encode())
        recovered = read_receipt(case, case.conn, reader, max_failure_bytes=budget)
        assert recovered.failure == case.problem
        for table in ("jobs", "attempts"):
            assert case.conn.execute(f"SELECT failure_json FROM {table}").fetchone()[0] == case.payload
        assert case.conn.execute("SELECT COUNT(*) FROM artifacts").fetchone()[0] == 0
        assert not case.conn.in_transaction


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
def test_default_limit_keeps_explicit_receipt_refusal_policy(over_default_problem, reader):
    with historical_failure(over_default_problem) as case:
        with pytest.raises(OpsError) as captured:
            read_receipt(case, case.conn, reader,
                         receipt_decode_failure_code="EXPERIMENT_IDENTITY_CONFLICT")
        assert captured.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
        assert captured.value.problem.message == "stored job failure exceeds its byte bound"
        assert captured.value.problem.retryable is False
        assert not case.conn.in_transaction


def test_plain_submit_replay_obeys_default_limit(over_default_problem):
    with historical_failure(over_default_problem) as case:
        with pytest.raises(OpsError) as captured:
            submission.submit(case.conn, REGISTRY, POLICY, request(), clock=case.clock)
        assert captured.value.code == "RESOURCE_LIMIT_EXCEEDED"
        assert case.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert case.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1
        assert not case.conn.in_transaction
