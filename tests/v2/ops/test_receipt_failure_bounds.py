"""Collection opts into bounds without hiding legitimate coordinator failures."""
import json
import sqlite3
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
from engine.v2.ops.supervisor import Service
from tests.ops_support import FakeClock
from tests.v2.ops.native_bounds_support import BoundedSqlRows
from tools.v2_ops_fixtures import POLICY, REGISTRY, enqueue_claim, request

CAP = 1024 * 1024


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


@pytest.fixture(scope="module")
def large_failed_job():
    """Run the real typed-coordinator handler and lifecycle on an in-memory DB."""
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        clock = FakeClock()
        migrate(conn, OWNER, MIGRATIONS, clock=clock)
        epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
        claim = enqueue_claim(conn, clock, Supervisor(epoch, "boot"))
        inputs, problems = coordinator_inputs(), []

        def coordinator_validation(*unused):
            try:
                validate([], **inputs)
            except OpsError as exc:
                problems.append(exc.problem)
                raise

        # Bypass service startup/subprocesses only. The failure is produced
        # by the coordinator's real validator, then _finish commits it through
        # _commit_failure -> commit_attempt -> end_attempt / advance_job.
        service = Service.__new__(Service)
        service.conn, service.clock = conn, clock
        service.launches, service._progress_state = {}, {}
        service._claim_keepalive_cache = None
        service._commit_success = coordinator_validation
        service._finish(SimpleNamespace(claim=claim), {"exit_code": 0})
        assert len(problems) == 1
        problem = problems[0]
        assert problem.code == "VALIDATION_FAILED" and not problem.retryable
        assert len(problem.details["findings"]) == 12000
        assert {item["reason"] for item in problem.details["findings"]} == {
            "unbound_or_late_cutoff"}
        payload = dumps(problem)
        assert len(payload.encode()) > CAP
        for table in ("jobs", "attempts"):
            row = conn.execute(f"SELECT state, failure_json FROM {table}").fetchone()
            assert tuple(row) == ("failed", payload)
        yield SimpleNamespace(conn=conn, clock=clock, job_id=claim.job_id,
                              problem=problem, payload=payload)
    finally:
        conn.close()


def read_receipt(case, conn, reader, **options):
    if reader == "get-job":
        return submission.get_job(conn, case.job_id, **options)
    return submission.submit_graph(conn, REGISTRY, POLICY, [request()],
                                   clock=case.clock, **options)[0]


@pytest.mark.parametrize("reader", ["get-job", "submit-graph-replay"])
@pytest.mark.parametrize("options", [{}, {"max_failure_bytes": None}], ids=["omitted", "none"])
def test_general_receipts_preserve_large_coordinator_failure(large_failed_job, reader, options):
    case = large_failed_job
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
    assert bounded.sizes == [None]
    assert not case.conn.in_transaction
    for table in ("jobs", "attempts"):
        assert case.conn.execute(f"SELECT failure_json FROM {table}").fetchone()[0] == case.payload
