"""The registered calendar kind refuses unsupported coverage at real submission."""
from __future__ import annotations

import json

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import SystemClock, to_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.calendar_moves_jobs import CalendarMovesParameters, forward_calendar_job_kind
from engine.v2.ops.errors import OpsError
from engine.v2.ops.submission import KindRegistry, NamespacePolicy, submit


def _request(tickers, expected_ids):
    kind = forward_calendar_job_kind()
    parameters = CalendarMovesParameters(
        tickers=tickers, expected_ids=expected_ids, as_of="2026-01-05",
        parent_snapshot_id="synthetic-parent", refresh_plan_hash="sha256:" + "a" * 64,
        catalog_path="catalog.sqlite", objects_root="objects")
    job = JobSpec(
        kind=kind.name, implementation_ref="synthetic-code", spec_hash=None,
        environment_ref="synthetic-environment", parameters=to_document(parameters),
        output_namespace="shadow", resource_class="io_fetch",
        retry_policy_ref=kind.retry.name, checkpoint_contract_ref=kind.checkpoint_contract)
    return SubmitRequest(namespace="shadow", idempotency_key="calendar-admission",
                         principal="operator", job=job)


@pytest.fixture
def admission(tmp_path):
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    registry = KindRegistry([forward_calendar_job_kind()])
    policy = NamespacePolicy({"operator": frozenset({"shadow"})})
    yield conn, registry, policy, clock
    conn.close()


@pytest.mark.parametrize(("tickers", "expected_ids"), [
    ((), ("ALPHA",)),
    ((), ()),
    (("ALPHA",), ()),
    (("ALPHA",), ("BETA",)),
    (("ALPHA", "ALPHA"), ("ALPHA",)),
    (("ALPHA",), ("ALPHA", "ALPHA")),
])
def test_unsupported_selection_refuses_before_submission_transaction(
        admission, tickers, expected_ids):
    conn, registry, policy, clock = admission
    statements = []
    conn.set_trace_callback(statements.append)
    before = conn.total_changes
    with pytest.raises(OpsError) as exc:
        submit(conn, registry, policy, _request(tickers, expected_ids), clock=clock)
    assert exc.value.problem.code == "INVALID_REQUEST"
    assert conn.total_changes == before
    assert statements == []  # Rejection precedes even the catalog transaction.
    assert not conn.in_transaction
    assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_reordered_equal_ticker_coverage_queues_without_execution(admission):
    conn, registry, policy, clock = admission
    request = _request(("BETA", "ALPHA"), ("ALPHA", "BETA"))
    receipt = submit(conn, registry, policy, request, clock=clock)
    row = conn.execute("SELECT state, spec_json FROM jobs WHERE job_id = ?",
                       (receipt.job_id,)).fetchone()
    assert row["state"] == "queued"
    assert json.loads(row["spec_json"])["parameters"] == request.job.parameters
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM data_raw_receipts").fetchone()[0] == 0
