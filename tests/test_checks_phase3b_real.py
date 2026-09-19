"""Focused tests for measured Phase 3B real-run acceptance controls."""
from __future__ import annotations

import pytest

from engine.v2.contracts import SnapshotRef, TableContract
from engine.v2.data import build_legacy_mapping
from engine.v2.foundation import ArtifactStore, content_hash, from_document
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.finality import SessionFinality

from checks import phase3b_real


def _snapshot():
    return SnapshotRef(
        snapshot_id="phase3b-test-parent",
        manifest_hash=content_hash({"phase3b": "test-parent"}),
        parent_snapshot_id=None, table_versions={},
        calendar_version="frozen-calendar.v1",
        source_priority_version="frozen-source-priority.v1",
        finality_receipt_refs=(), knowledge_mode_by_table={})


@pytest.mark.needs_data  # reads the real data/ root (gitignored, absent in CI and worktrees)
def test_table_run_reopens_persisted_clean_rebuild_and_downstream_scan(tmp_path):
    clock = phase3b_real._AcceptanceClock()
    conn = open_catalog(tmp_path / "catalog.sqlite3", clock=clock)
    store = ArtifactStore(tmp_path / "objects")
    mapping = build_legacy_mapping()
    contract = from_document(
        TableContract, mapping["tables"]["securities"])
    source = phase3b_real._sources()["securities"]
    parent, _, base_rows = phase3b_real._base_snapshot(
        conn, store, (contract,), {"securities": source}, clock)

    _, results, _ = phase3b_real._table_runs(
        conn, store, (contract,), parent, base_rows, clock)

    result = results["securities"]
    assert result["persisted_rows_equal"] is True
    assert result["downstream_results_equal"] is True
    assert result["rebuild_equal"] is True
    assert result["clean_rebuild_rows"] == result["rows_after"]
    assert result["clean_rebuild_artifact_hash"].startswith("sha256:")


def test_empty_response_and_shared_quota_controls_execute_real_behavior(tmp_path):
    conn = open_catalog(
        tmp_path / "catalog.sqlite3", clock=phase3b_real._AcceptanceClock())

    metrics = phase3b_real._acceptance_acquisition_controls(conn, _snapshot())

    assert metrics["empty_kind"] == "empty"
    assert metrics["omitted_empty_kind"] == "partial"
    assert metrics["empty_commit_admitted"] is True
    assert metrics["omitted_empty_commit_admitted"] is False
    assert metrics["empty_distinguished"] is True
    assert metrics["quota_first_claimed"] is True
    assert metrics["quota_second_claimed"] is False
    assert metrics["quota_reservation"] == [1, 0]
    assert "PROVIDER_BUDGET" in metrics["quota_queue_codes"]
    assert metrics["quota_shared"] is True


def test_finality_control_omits_override_and_accepts_joint_coverage(monkeypatch):
    calls = []

    def correct(value, tickers, *, frames=None, **kwargs):
        calls.append(kwargs)
        daily = set(frames["daily_market"]["ticker"])
        chains = set(frames["option_chains"]["ticker"])
        wanted = set(tickers)
        joint = wanted & daily & chains
        matched = len(wanted) == 1 and len(joint) == 1
        return SessionFinality(
            date=str(value), market_wide=True,
            daily_share=1.0 if matched else 0.5,
            chain_share=1.0 if matched else 0.5,
            is_final=matched, detail="final" if matched else "split coverage",
            tickers=len(wanted), covered=len(joint))

    monkeypatch.setattr(phase3b_real.v2_finality, "session_finality", correct)
    metrics = phase3b_real._finality_metrics("2026-09-17")

    assert calls == [{}, {}]
    assert metrics["split_coverage_refused"] is True
    assert metrics["native_behavior_valid"] is True


def test_finality_control_fails_closed_on_split_ticker_false_positive(monkeypatch):
    def faulty(value, tickers, *, frames=None, **kwargs):
        return SessionFinality(
            date=str(value), market_wide=True, daily_share=1.0,
            chain_share=1.0, is_final=True, detail="final",
            tickers=len(tuple(tickers)), covered=1)

    monkeypatch.setattr(phase3b_real.v2_finality, "session_finality", faulty)
    metrics = phase3b_real._finality_metrics("2026-09-17")

    assert metrics["split_is_final"] is True
    assert metrics["split_covered"] == 1
    assert metrics["split_coverage_refused"] is False
    assert metrics["native_behavior_valid"] is False
