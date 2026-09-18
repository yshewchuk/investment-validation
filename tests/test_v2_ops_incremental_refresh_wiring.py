"""R3B-3: the nightly DAG must actually submit ``incremental_refresh``.

Before this, ``incremental_refresh`` was registered as a stage and had a
worker (``engine/v2/ops/stages.py``, ``engine/v2/ops/worker.py``), but
nothing built a job for it: ``NIGHTLY_GRAPH``/``GRAPH``'s ``"refresh"`` node
always fell through ``nightly._action_for`` to ``_legacy_action``, and
``_DAG_STAGES`` (the stages a real nightly plan actually submits) never even
named "refresh". These tests prove the native path now exists, is opt-in
(explicit ``refresh_mode``), is recorded on the run's own plan document, and
leaves the default/legacy DAG byte-identical to before.
"""
from __future__ import annotations

import pytest

from engine.v2.contracts import SnapshotRef, SubmitRequest
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import (
    RefreshUnit,
    classify_response,
    plan_refresh,
)
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.plans import nightly_plan
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit_graph
from tests.ops_support import FakeClock

#: The exact kind sequence the pre-existing legacy-only test
#: (test_v2_ops_legacy_workflows.py::test_allowlisted_legacy_dag_submits_with_real_dependencies)
#: pins for a default nightly plan. Repeated here as the regression control:
#: native refresh mode must never change this list.
_LEGACY_KINDS = [
    "legacy_finality", "legacy_score", "legacy_decision_replay", "decision_evidence",
    "legacy_decisions", "legacy_settlement", "legacy_model_evidence", "ledger_export",
    "engineering_gate", "legacy_render", "legacy_selfcheck", "publication", "backup"]


def _snapshot(snapshot_id="snap-parent"):
    return SnapshotRef(
        snapshot_id=snapshot_id, manifest_hash="sha256:" + "a" * 64,
        parent_snapshot_id=None, table_versions={}, calendar_version="cal-v1",
        source_priority_version="priority-v1", finality_receipt_refs=(),
        knowledge_mode_by_table={})


def _cache_satisfied_refresh_plan():
    """A fully valid, zero-network RefreshPlan: one unit, already cached."""
    unit = RefreshUnit(request_id="eod-2026-09-18-AAPL", table_name="daily_market",
                       partition_key="2026-09-18", expected_keys=("AAPL",))
    outcome = classify_response(
        200, ("AAPL",), returned_keys=("AAPL",), request_id="eod-2026-09-18-AAPL",
        receipt_ref="receipt-eod-2026-09-18-AAPL", raw_hash="sha256:" + "b" * 64,
        cache_hit=True)
    return plan_refresh(_snapshot(), (unit,), cached_outcomes={"eod-2026-09-18-AAPL": outcome},
                        provider_account=None)


def test_legacy_mode_dag_is_unchanged():
    """R3B-3 must not touch the route production still uses."""
    plan = build_nightly_plan("/root/investing-plan", "2026-09-18")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    assert [request.job.kind for request in requests] == _LEGACY_KINDS
    assert "incremental_refresh" not in {r.job.kind for r in requests}


def test_native_refresh_mode_submits_the_real_native_stage(tmp_path):
    """The submitted action must be ``incremental_refresh`` -- never a
    ``legacy_refresh`` adapter action (which is not even a registered kind)."""
    refresh_plan = _cache_satisfied_refresh_plan()
    plan = build_nightly_plan("/root/investing-plan", "2026-09-18")
    requests = build_legacy_job_requests(
        plan, tickers=("FAKE",), year_start=2025, year_end=2026,
        refresh_mode="native", refresh_plan=refresh_plan)

    kinds = [request.job.kind for request in requests]
    assert kinds[0] == "incremental_refresh"
    assert kinds[1:] == _LEGACY_KINDS
    assert "legacy_refresh" not in kinds  # never a registered kind; proves it is not the adapter

    refresh_request = requests[0]
    assert refresh_request.job.resource_class == "io_fetch"
    assert refresh_request.job.parameters["parent_snapshot_id"] == "snap-parent"
    assert refresh_request.job.parameters["expected_ids"] == ["eod-2026-09-18-AAPL"]
    assert refresh_request.job.parameters["provider_calls"] == 0

    # Real admission proof: the production registry (stages.registry()) accepts
    # it, exactly like every other stage's job in this same DAG.
    conn = open_catalog(tmp_path / "ops.sqlite", clock=FakeClock())
    try:
        policy = NamespacePolicy({"operator": frozenset({"shadow"})})
        receipts = submit_graph(conn, registry(), policy, requests, clock=FakeClock())
        assert len(receipts) == len(requests)
        row = conn.execute(
            "SELECT kind FROM jobs WHERE job_id = ?", (receipts[0].job_id,)).fetchone()
        assert row[0] == "incremental_refresh"
    finally:
        conn.close()


def test_native_refresh_mode_needs_a_pinned_plan():
    plan = build_nightly_plan("/root/investing-plan", "2026-09-18")
    with pytest.raises(OpsError, match="native refresh mode needs a pinned refresh plan"):
        build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026,
                                  refresh_mode="native")


def test_bad_refresh_mode_is_refused():
    plan = build_nightly_plan("/root/investing-plan", "2026-09-18")
    with pytest.raises(OpsError, match="refresh_mode must be legacy or native"):
        build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026,
                                  refresh_mode="bogus")


def test_refresh_mode_is_explicit_and_inspectable_on_the_saved_plan():
    """A reader of a completed run's plan document -- not just job presence
    in the catalog -- must be able to tell which route executed."""
    legacy = nightly_plan("/root/investing-plan", "2026-09-18",
                          manifest_ref="artifact:fake", expected_population=("FAKE|x|2026-09-18",))
    assert "refresh_mode" not in legacy  # absent means legacy, mirroring input_mode's own convention

    refresh_plan = _cache_satisfied_refresh_plan()
    native = nightly_plan("/root/investing-plan", "2026-09-18",
                          manifest_ref="artifact:fake", expected_population=("FAKE|x|2026-09-18",),
                          refresh_mode="native", refresh_plan=refresh_plan)
    assert native["refresh_mode"] == "native"
    assert native["refresh_plan"]["parent_snapshot_id"] == "snap-parent"
    assert native["refresh_plan"]["plan_hash"] == refresh_plan.plan_hash
    # opting in changes the pinned plan_hash -- a native and a legacy plan for
    # the identical session/universe are never confused with each other.
    assert native["plan_hash"] != legacy["plan_hash"]


def test_native_refresh_mode_needs_a_pinned_plan_at_plan_time():
    with pytest.raises(OpsError, match="native refresh mode needs a pinned refresh plan"):
        nightly_plan("/root/investing-plan", "2026-09-18", manifest_ref="artifact:fake",
                    expected_population=("FAKE|x|2026-09-18",), refresh_mode="native")


def test_end_to_end_plan_document_drives_which_action_is_submitted(tmp_path):
    """The same plan document a completed run's operator would read back
    (`plan["refresh_mode"]`) is exactly what decides the submitted action --
    not a second, separately-tracked flag."""
    refresh_plan = _cache_satisfied_refresh_plan()
    plan = nightly_plan("/root/investing-plan", "2026-09-18", manifest_ref="artifact:fake",
                        expected_population=("FAKE|x|2026-09-18",),
                        refresh_mode="native", refresh_plan=refresh_plan)
    requests = build_legacy_job_requests(
        plan, tickers=(), year_start=plan["year_start"], year_end=plan["year_end"],
        input_refs=(plan["input_manifest_ref"],),
        expected_population=tuple(plan["expected_population"]),
        refresh_mode=plan.get("refresh_mode", "legacy"), refresh_plan=plan.get("refresh_plan"))
    assert requests[0].job.kind == "incremental_refresh"
