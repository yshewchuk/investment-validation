"""Focused Phase 3B ops/dependency integration tests.

No network, market data or normalization implementation is used here.  The
tests prove the ops boundary: existing supervisor admission, provider-budget
sharing, conservative dependency planning and existing snapshot CAS hooks.
"""
from __future__ import annotations

import sqlite3

import pytest

from engine.v2.contracts import (
    DataQuery,
    DependencyPlan,
    SnapshotRef,
    SubmitRequest,
    TableContractRef,
)
from engine.v2.ops import incremental_data, provider_budget
from engine.v2.ops.errors import OpsError
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.submission import KindRegistry, NamespacePolicy, submit
from tests.ops_support import catalog, sample


def _snapshot(snapshot_id="snap-parent"):
    return SnapshotRef(
        snapshot_id=snapshot_id, manifest_hash="sha256:" + "a" * 64,
        parent_snapshot_id=None, table_versions={}, calendar_version="cal-v1",
        source_priority_version="priority-v1", finality_receipt_refs=(),
        knowledge_mode_by_table={})


def _unit(request_id, key):
    return incremental_data.RefreshUnit(
        request_id=request_id, table_name="daily_market",
        partition_key="2026-09-15", expected_keys=(key,))


def _complete(request_id, key, *, cache_hit=False):
    return incremental_data.classify_response(
        200, (key,), returned_keys=(key,), request_id=request_id,
        receipt_ref="receipt-" + request_id, raw_hash="sha256:" + "b" * 64,
        cache_hit=cache_hit)


def test_response_classification_distinguishes_partial_empty_auth_and_retry():
    partial = incremental_data.classify_response(
        200, ("AAPL", "MSFT"), returned_keys=("AAPL",), request_id="r1",
        receipt_ref="receipt-r1")
    assert partial.kind == "partial"
    assert incremental_data.missing_keys(partial) == ("MSFT",)
    assert incremental_data.retry_action(partial) == "retry_missing"
    assert not incremental_data.coverage_complete(partial)

    empty = incremental_data.classify_response(
        200, ("contract",), empty_keys=("contract",), request_id="r2",
        receipt_ref="receipt-r2")
    assert empty.kind == "empty"
    assert incremental_data.coverage_complete(empty)
    assert incremental_data.retry_action(empty) == "stop"

    auth = incremental_data.classify_response(
        200, ("AAPL",), credential_page=True, request_id="r3")
    assert auth.kind == "credential_invalid"
    assert incremental_data.retry_action(auth) == "stop"


def test_cache_first_plan_uses_existing_supervisor_and_shared_provider_budget(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    provider_budget.configure_account(
        conn, "orats-account", "generation-1", remaining=20, live_reserve=5)
    cached = _complete("cached", "AAPL", cache_hit=True)
    plan = incremental_data.plan_refresh(
        _snapshot(), (_unit("cached", "AAPL"), _unit("fetch", "MSFT")),
        cached_outcomes={"cached": cached}, provider_account="orats-account",
        max_attempts=3, expected_head_generation=1)
    assert [unit.request_id for unit in plan.fetch_units] == ["fetch"]
    assert plan.provider_calls == 3

    spec = incremental_data.refresh_job_spec(
        plan, implementation_ref="code", environment_ref="env",
        output_namespace="shadow", catalog_path="catalog.sqlite",
        objects_root="objects")
    request = SubmitRequest(
        namespace="shadow", idempotency_key="refresh-2026-09-15",
        principal="operator", job=spec)
    registry = KindRegistry([incremental_data.refresh_job_kind()])
    policy = NamespacePolicy({"operator": frozenset({"shadow"})})
    receipt = submit(conn, registry, policy, request, clock=clock)
    claim = claim_next(
        conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor,
        clock=clock, registry=registry)
    assert claim is not None and claim.job_id == receipt.job_id
    reservation = conn.execute(
        "SELECT reserved_calls, used_calls FROM provider_reservations "
        "WHERE account = ? AND released_at IS NULL", ("orats-account",)).fetchone()
    assert tuple(reservation) == (3, 0)


def test_plan_refresh_reserves_calls_per_unit_for_multi_call_providers():
    plan = incremental_data.plan_refresh(
        _snapshot(), (_unit("fetch", "MSFT"),), cached_outcomes={},
        provider_account="orats-daily-market", max_attempts=3,
        expected_head_generation=1, calls_per_unit=2)
    assert plan.provider_calls == 6


def test_failed_or_partial_outcomes_cannot_admit_watermark_advancing_commit():
    unit = _unit("r1", "AAPL")
    plan = incremental_data.plan_refresh(
        _snapshot(), (unit,), cached_outcomes={}, provider_account="polygon",
        max_attempts=2, expected_head_generation=1)
    partial = incremental_data.classify_response(
        200, ("AAPL",), returned_keys=(), request_id="r1",
        receipt_ref="receipt-r1")
    admission = incremental_data.admit_candidate_commit(plan, (partial,))
    assert not admission.admitted
    assert admission.coverage_receipt_refs == ()
    assert admission.raw_hashes == ()
    assert admission.incomplete_request_ids == ("r1",)

    complete = _complete("r1", "AAPL")
    admission = incremental_data.admit_candidate_commit(plan, (complete,))
    assert admission.admitted
    assert admission.coverage_receipt_refs == ("receipt-r1",)


class _ExplainingRepository:
    def __init__(self, snapshot_ref):
        self.snapshot_ref = snapshot_ref
        self.calls = []

    def explain_dependencies(self, query, *, table_name):
        self.calls.append((query.snapshot_id, table_name))
        return DependencyPlan(
            request_hash="sha256:" + "c" * 64,
            snapshot_ref=self.snapshot_ref, dependencies=())


def _query(snapshot_id):
    return DataQuery(
        snapshot_id=snapshot_id,
        table_contract_ref=TableContractRef(
            contract_id="daily_market.v1", definition_hash="sha256:" + "d" * 64),
        columns=("ticker", "date", "close"), key_filter=(),
        order_by=("ticker", "date"), max_batch_rows=100,
        max_result_rows=1000)


def test_dependency_explanations_are_bound_to_one_resolved_snapshot():
    snapshot = _snapshot()
    repository = _ExplainingRepository(snapshot)
    explanations = incremental_data.explain_snapshot_queries(
        repository, snapshot, (("features", "daily_market", _query(snapshot.snapshot_id)),))
    assert explanations[0].plan.snapshot_ref.snapshot_id == snapshot.snapshot_id
    assert repository.calls == [(snapshot.snapshot_id, "daily_market")]

    with pytest.raises(OpsError) as caught:
        incremental_data.explain_snapshot_queries(
            repository, snapshot, (("features", "daily_market", _query("new-head")),))
    assert caught.value.code == "STALE_EXPECTATION"


def test_dependency_impact_is_suffix_aware_and_unknown_is_full_or_refused():
    snapshot = _snapshot()
    correction = incremental_data.DataChange(
        change_id="chg-correction", table_name="daily_market",
        revision_kind="correction", columns=("close",),
        start="2025-01-02", end_exclusive="2025-01-03")
    unknown = incremental_data.DataChange(
        change_id="chg-unknown", table_name="earnings_events",
        revision_kind="correction", columns_known=False,
        dependency_known=False, start="2024-10-01")
    rules = (
        incremental_data.DependencyRule(
            consumer_id="rolling_features", table_names=("daily_market",),
            mode="suffix", columns=("close",)),
        incremental_data.DependencyRule(
            consumer_id="model_folds", table_names=("earnings_events",),
            mode="unknown"),
    )
    plan = incremental_data.plan_dependency_impacts(
        snapshot, (correction, unknown), (), rules)
    assert [(item.consumer_id, item.scope, item.start) for item in plan.invalidations] == [
        ("model_folds", "full", None),
        ("rolling_features", "suffix", "2025-01-02"),
    ]
    assert plan.conservative

    with pytest.raises(OpsError) as caught:
        incremental_data.plan_dependency_impacts(
            snapshot, (unknown,), (), (rules[1],), unknown_policy="refuse")
    assert caught.value.code == "VALIDATION_FAILED"


def _admission(admitted=True):
    return incremental_data.CommitAdmission(
        admitted=admitted,
        coverage_receipt_refs=("coverage-r1",) if admitted else (),
        raw_hashes=("sha256:" + "e" * 64,) if admitted else (),
        incomplete_request_ids=() if admitted else ("r1",))


def _candidate(admitted=True):
    return incremental_data.CandidatePromotion(
        candidate_scope="candidate", target_scope="shadow",
        candidate_snapshot_id="snap-next",
        expected_target_snapshot_id="snap-parent",
        expected_target_generation=7, comparison_receipt_id="comparison-r1",
        commit_admission=_admission(admitted))


def test_promotion_hook_refuses_partial_and_passes_exact_cas_expectation():
    calls = []

    def hook(conn, store, **kwargs):
        calls.append(kwargs)
        return "promoted"

    with pytest.raises(OpsError) as caught:
        incremental_data.promote_refresh_candidate(
            object(), object(), _candidate(False), clock=object(), promote_hook=hook)
    assert caught.value.code == "VALIDATION_FAILED"
    assert calls == []

    result = incremental_data.promote_refresh_candidate(
        object(), object(), _candidate(), clock="clock", promote_hook=hook)
    assert result == "promoted"
    assert calls == [{
        "candidate_scope": "candidate", "target_scope": "shadow",
        "expected_snapshot_id": "snap-parent", "expected_generation": 7,
        "comparison_receipt_id": "comparison-r1", "clock": "clock",
    }]


def test_recovery_hook_detects_done_retry_and_conflict_without_mutation():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE data_snapshot_heads (scope TEXT PRIMARY KEY, "
        "snapshot_id TEXT NOT NULL, generation INTEGER NOT NULL)")
    conn.execute(
        "INSERT INTO data_snapshot_heads VALUES (?, ?, ?)",
        ("shadow", "snap-parent", 7))
    candidate = _candidate()
    assert incremental_data.recovery_decision(conn, candidate).action == "retry_promotion"

    conn.execute(
        "UPDATE data_snapshot_heads SET snapshot_id = ?, generation = ? WHERE scope = ?",
        ("snap-next", 8, "shadow"))
    assert incremental_data.recovery_decision(conn, candidate).action == "already_promoted"

    conn.execute(
        "UPDATE data_snapshot_heads SET snapshot_id = ?, generation = ? WHERE scope = ?",
        ("snap-other", 8, "shadow"))
    decision = incremental_data.recovery_decision(conn, candidate)
    assert decision.action == "refuse_conflict"
    assert (decision.snapshot_id, decision.generation) == ("snap-other", 8)
