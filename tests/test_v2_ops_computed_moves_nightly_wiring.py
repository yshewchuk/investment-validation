"""S4C Part 4: nightly.py's own computed_moves_refresh wiring -- the plan
builder, the native/legacy stage-sequence split, and the submitted job's
shape. ``run_computed_moves_refresh`` itself (the worker) is already covered
end to end by ``tests/test_v2_ops_computed_moves_store.py``; these tests
prove only nightly.py's own new code, never re-testing the worker or the
store's own target-selection logic (monkeypatched here to a fixed list).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from engine.v2.foundation import ArtifactStore
from engine.v2.ops import computed_moves_store
from engine.v2.ops.calendar_moves_jobs import COMPUTED_MOVES_RESULT_SCHEMA
from engine.v2.ops.nightly import (
    COMPUTED_MOVES_REFRESH_ACTION,
    NATIVE_COMPUTED_MOVES_ACCOUNT,
    NATIVE_REFRESH_ACTION,
    _NATIVE_ACTION_STAGES,
    _build_native_computed_moves_plan,
    _stage_sequence,
    build_legacy_job_requests,
    build_nightly_plan,
)
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit_graph
from tests.data_scan_support import commit_tables, contract_for
from tests.ops_support import catalog

ROOT = Path(__file__).resolve().parents[1]
_EVENTS = contract_for("earnings_events")
_DAILY = contract_for("daily_market")


def _commit_parent(conn, clock, store, *, tables=("earnings_events", "daily_market")):
    contracts = {name: (_EVENTS if name == "earnings_events" else _DAILY) for name in tables}
    commit_tables(conn, clock, {name: [] for name in tables}, contracts, store=store)
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = 'shadow'").fetchone()


# --------------------------------------------------------------------------
# _stage_sequence: the native/legacy split
# --------------------------------------------------------------------------


def test_native_action_stages_maps_each_stage_to_its_own_kind():
    assert _NATIVE_ACTION_STAGES == {
        "refresh": NATIVE_REFRESH_ACTION,
        COMPUTED_MOVES_REFRESH_ACTION: COMPUTED_MOVES_REFRESH_ACTION,
    }


def test_stage_sequence_native_mode_prepends_both_native_stages():
    plan = build_nightly_plan(ROOT, "2026-09-18")
    stages = _stage_sequence(plan, False, None, "native")
    assert stages[:2] == ("refresh", COMPUTED_MOVES_REFRESH_ACTION)


def test_stage_sequence_legacy_prerequisite_walk_filters_computed_moves_refresh():
    """computed_moves_refresh is a real GRAPH node, so a prerequisite-inclusive
    plan["order"] names it -- but legacy mode has no kind to build for it, so
    _stage_sequence must filter it back out. "refresh" itself stays (the
    pre-existing legacy include_prerequisites behaviour, unchanged)."""
    plan = build_nightly_plan(ROOT, "2026-09-18")
    assert COMPUTED_MOVES_REFRESH_ACTION in plan["order"]
    stages = _stage_sequence(plan, True, None, "legacy")
    assert COMPUTED_MOVES_REFRESH_ACTION not in stages
    assert "refresh" in stages


def test_legacy_mode_default_dag_never_submits_computed_moves_refresh():
    plan = build_nightly_plan(ROOT, "2026-09-18")
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    kinds = [request.job.kind for request in requests]
    assert COMPUTED_MOVES_REFRESH_ACTION not in kinds
    assert "legacy_computed_moves_refresh" not in kinds


# --------------------------------------------------------------------------
# _build_native_computed_moves_plan: graceful degradation to None
# --------------------------------------------------------------------------


def test_no_open_catalog_returns_none():
    plan = build_nightly_plan(ROOT, "2026-09-18")
    assert _build_native_computed_moves_plan(
        plan, ("AAPL",), catalog_path=None, objects_root=None,
        conn=None, store=None, clock=None) is None


def test_no_shadow_head_returns_none(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    plan = build_nightly_plan(ROOT, "2026-09-18")
    assert _build_native_computed_moves_plan(
        plan, ("AAPL",), catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock) is None


def test_head_missing_daily_market_returns_none(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store, tables=("earnings_events",))
    plan = build_nightly_plan(ROOT, "2026-09-18")
    assert _build_native_computed_moves_plan(
        plan, ("AAPL",), catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock) is None


def test_empty_target_list_returns_none(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: ([], {}))
    plan = build_nightly_plan(ROOT, "2026-09-18")
    assert _build_native_computed_moves_plan(
        plan, ("AAPL",), catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock) is None


def test_real_targets_build_a_refresh_plan_with_matching_expected_ids(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    head = _commit_parent(conn, clock, store)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL", "MSFT"], {}))
    plan = build_nightly_plan(ROOT, "2026-09-18")
    built = _build_native_computed_moves_plan(
        plan, ("AAPL", "MSFT"), catalog_path=None, objects_root=None,
        conn=conn, store=store, clock=clock)
    assert built is not None
    refresh_plan, expected_ids = built
    assert expected_ids == ("AAPL", "MSFT")
    assert refresh_plan.parent_snapshot_id == head["snapshot_id"]
    assert refresh_plan.provider_account == NATIVE_COMPUTED_MOVES_ACCOUNT


# --------------------------------------------------------------------------
# end to end through build_legacy_job_requests + real admission
# --------------------------------------------------------------------------


def test_native_mode_submits_computed_moves_refresh_alongside_refresh(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: (["AAPL"], {}))
    plan = build_nightly_plan(ROOT, "2026-09-18")
    requests = build_legacy_job_requests(
        plan, tickers=("AAPL",), year_start=2025, year_end=2026,
        refresh_mode="native", catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), conn=conn, store=store, clock=clock)

    kinds = [request.job.kind for request in requests]
    assert kinds[0] == NATIVE_REFRESH_ACTION
    assert kinds[1] == COMPUTED_MOVES_REFRESH_ACTION

    cm_request = requests[1]
    assert cm_request.job.parameters["expected_ids"] == ["AAPL"]
    assert cm_request.job.provider_budget_ref == NATIVE_COMPUTED_MOVES_ACCOUNT
    assert cm_request.job.resource_class == "io_fetch"
    assert cm_request.job.checkpoint_contract_ref == COMPUTED_MOVES_RESULT_SCHEMA

    policy = NamespacePolicy({"operator": frozenset({"shadow"})})
    receipts = submit_graph(conn, registry(), policy, requests, clock=clock)
    assert len(receipts) == len(requests)
    row = conn.execute("SELECT kind FROM jobs WHERE job_id = ?",
                       (receipts[1].job_id,)).fetchone()
    assert row[0] == COMPUTED_MOVES_REFRESH_ACTION


def test_native_mode_omits_computed_moves_refresh_when_there_is_no_scoreable_target(
        tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, clock, store)
    monkeypatch.setattr(computed_moves_store, "target_tickers_from_snapshot",
                        lambda *a, **k: ([], {}))
    plan = build_nightly_plan(ROOT, "2026-09-18")
    requests = build_legacy_job_requests(
        plan, tickers=("AAPL",), year_start=2025, year_end=2026,
        refresh_mode="native", catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path), conn=conn, store=store, clock=clock)

    kinds = [request.job.kind for request in requests]
    assert kinds[0] == NATIVE_REFRESH_ACTION
    assert COMPUTED_MOVES_REFRESH_ACTION not in kinds
