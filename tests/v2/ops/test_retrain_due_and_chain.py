"""#805 slice 2: due-month decision, durable records and the retrain/pool-sidecar chain."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.pool_refresh import submit_pool_nightly_refresh_if_ready
from engine.v2.ops.retrain_due import (
    chain_blocks_pool_sidecar,
    due_decision,
    read_recorded,
    write_superseded,
)
from engine.v2.ops.submission import job_id_for
from engine.v2.ops.worker import _dispatch_retrain_step


def _conn():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE jobs (namespace TEXT, idempotency_key TEXT, state TEXT)")
    return conn

def test_due_decision():
    recorded = frozenset()

    decision = due_decision("2026-11-02", None, recorded)
    assert decision["state"] == "due"
    assert decision["month"] == "2026-10"
    assert decision["m_end"] == "2026-10-31"
    assert decision["superseded"] == []

    decision = due_decision("2026-10-31", None, recorded)
    assert decision["month"] == "2026-09"

    decision = due_decision("2026-11-02", None, frozenset({"2026-10"}))
    assert decision["state"] == "not_due"
    assert decision["reason"] == "month_already_recorded"

    decision = due_decision("2026-01-05", None, recorded)
    assert decision["month"] == "2025-12"
    assert decision["m_end"] == "2025-12-31"

    decision = due_decision("2026-11-02", "2026-10-01", recorded)
    assert decision["month"] == "2026-10"
    assert decision["superseded"] == []

    decision = due_decision("2026-11-02", "2026-08-01", recorded)
    assert decision["month"] == "2026-10"
    assert decision["superseded"] == ["2026-08", "2026-09"]

    decision = due_decision("2026-11-02", "2026-11-01", recorded)
    assert decision["state"] == "not_due"
    assert decision["reason"] == "base_retrain_cutoff_current"

    decision = due_decision("2024-03-01", None, recorded)
    assert decision["m_end"] == "2024-02-29"

    with pytest.raises(ValueError):
        due_decision("2026-13-01", None, recorded)


def test_read_recorded_and_chain(tmp_path):
    conn = _conn()
    conn.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-promote:2026-11-02:2026-10:rel1', 'succeeded')")
    conn.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-promote:2026-09-30:2026-09:rel1', 'failed')")
    conn.commit()

    recorded = read_recorded(conn, tmp_path)
    assert "2026-10" in recorded
    assert "2026-09" not in recorded

    marker_dir = tmp_path / "retrain_superseded"
    marker_dir.mkdir(parents=True)
    (marker_dir / "2026-08.json").write_text('{"month":"2026-08"}')

    recorded = read_recorded(conn, tmp_path)
    assert "2026-08" in recorded

    ops_root = tmp_path / "ops_root"
    write_superseded(ops_root, ["2026-08"], "2026-10", "2026-11-02")
    first_content = (ops_root / "retrain_superseded" / "2026-08.json").read_text()
    write_superseded(ops_root, ["2026-08"], "2026-11", "2026-11-02")
    assert (ops_root / "retrain_superseded" / "2026-08.json").read_text() == first_content

    conn2 = _conn()
    conn2.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-restage:2026-11-02', 'succeeded')")
    conn2.commit()
    assert chain_blocks_pool_sidecar(conn2, "2026-11-02") is False

    conn2.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-restage:2026-11-02', 'running')")
    conn2.commit()
    assert chain_blocks_pool_sidecar(conn2, "2026-11-02") is True

    conn3 = _conn()
    conn3.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-restage:2026-11-03', 'failed')")
    conn3.commit()
    assert chain_blocks_pool_sidecar(conn3, "2026-11-02") is False


def _plan(session="2026-09-18"):
    root = Path(__file__).resolve().parents[3]
    return build_nightly_plan(str(root), session)


def _requests(plan):
    return build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)


def _retrain_requests(requests):
    return [r for r in requests if r.job.kind == "retrain_step"]


def _score_request(requests):
    for r in requests:
        if r.job.kind == "legacy_score":
            return r
    raise AssertionError("no legacy_score request")


def test_plan_without_retrain_has_no_retrain_step():
    plan = _plan()
    before = _requests(plan)
    assert _retrain_requests(before) == []
    after = _requests(plan)
    assert [r.job.kind for r in before] == [r.job.kind for r in after]


def test_not_due_plan_has_restage_only():
    plan = _plan()
    plan["retrain"] = {"state": "not_due", "reason": "month_already_recorded", "base": "rel1"}
    requests = _requests(plan)
    retrain = _retrain_requests(requests)
    assert len(retrain) == 1
    assert retrain[0].job.parameters["step"] == "restage"
    assert retrain[0].idempotency_key == "retrain-restage:2026-09-18"
    score = _score_request(requests)
    assert job_id_for("shadow", "retrain-restage:2026-09-18") in score.job.dependency_job_ids
    assert requests.index(retrain[0]) < requests.index(score)


def test_due_plan_has_six_retrain_steps():
    plan = _plan()
    plan["retrain"] = {
        "state": "due", "month": "2026-10", "m_end": "2026-10-31",
        "superseded": [], "base": "rel1",
    }
    requests = _requests(plan)
    retrain = _retrain_requests(requests)
    assert len(retrain) == 6
    expected_keys = [
        "retrain-restage:2026-09-18",
        "retrain-train:2026-09-18:2026-10:rel1",
        "retrain-stage:2026-09-18:2026-10:rel1",
        "retrain-accept:2026-09-18:2026-10:rel1",
        "retrain-check:2026-09-18:2026-10:rel1",
        "retrain-promote:2026-09-18:2026-10:rel1",
    ]
    for request, key in zip(retrain, expected_keys):
        assert request.idempotency_key == key

    assert retrain[0].job.dependency_job_ids == ()
    for previous, current in zip(retrain, retrain[1:]):
        assert job_id_for("shadow", previous.idempotency_key) in current.job.dependency_job_ids

    score = _score_request(requests)
    last_key = "retrain-promote:2026-09-18:2026-10:rel1"
    assert job_id_for("shadow", last_key) in score.job.dependency_job_ids


def test_worker_restage_stub(tmp_path):
    result = _dispatch_retrain_step(
        {"step": "restage", "session": "s", "month": "", "expected_ids": ["k"]},
        tmp_path,
    )
    assert result["outputs"][0]["name"] == "retrain_step_receipt"
    document = json.loads((tmp_path / "receipt.json").read_text())
    assert document["restage"] == "unchecked"


@pytest.mark.parametrize("step", ["train", "stage", "accept", "check", "promote"])
def test_worker_chain_steps_raise(step, tmp_path):
    with pytest.raises(OpsError) as excinfo:
        _dispatch_retrain_step(
            {"step": step, "session": "s", "month": "m", "expected_ids": ["k"]},
            tmp_path,
        )
    assert excinfo.value.problem.code == "VALIDATION_FAILED"
    details = excinfo.value.problem.details
    assert details["code"] == "RETRAIN_UNAVAILABLE"
    assert details["step"] == step


def test_sidecar_gate(monkeypatch, tmp_path):
    import engine.v2.ops.pool_refresh as pool_refresh

    monkeypatch.setattr(pool_refresh, "pool_refresh_identity", lambda conn: ("2026-11-02", "manifest"))

    conn = _conn()
    conn.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-restage:2026-11-02', 'failed')")
    conn.commit()

    assert submit_pool_nightly_refresh_if_ready(
        conn, None, None, None, tmp_path, code_source=None, clock=None
    ) == []

    conn2 = _conn()
    conn2.execute("INSERT INTO jobs VALUES ('shadow', 'retrain-restage:2026-11-02', 'succeeded')")
    conn2.commit()

    with pytest.raises(Exception):
        submit_pool_nightly_refresh_if_ready(
            conn2, None, None, None, tmp_path, code_source=None, clock=None
        )


def test_decide_reads_a_real_catalog_and_writes_markers(tmp_path):
    from engine.v2.foundation import SystemClock
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.retrain_due import decide

    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    decision = decide(conn, tmp_path, "2026-11-02", "rel1", base_retrain_cutoff="2026-08-01")
    assert decision["state"] == "due"
    assert decision["month"] == "2026-10"
    assert decision["superseded"] == ["2026-08", "2026-09"]
    assert decision["base"] == "rel1"
    assert (tmp_path / "retrain_superseded" / "2026-08.json").exists()
    assert (tmp_path / "retrain_superseded" / "2026-09.json").exists()

    second = decide(conn, tmp_path, "2026-11-02", "rel1", base_retrain_cutoff="2026-08-01")
    assert second["month"] == "2026-10"
    assert second["superseded"] == []
    assert chain_blocks_pool_sidecar(conn, "2026-11-02") is False


def test_chain_submits_through_the_real_registry_and_gates_the_sidecar(tmp_path):
    from engine.v2.foundation import SystemClock
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.stages import registry
    from engine.v2.ops.submission import NamespacePolicy, submit_graph

    conn = open_catalog(tmp_path / "catalog.sqlite", clock=SystemClock())
    plan = _plan()
    plan["retrain"] = {
        "state": "due", "month": "2026-10", "m_end": "2026-10-31",
        "superseded": [], "base": "rel1",
    }
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    policy = NamespacePolicy({"operator": frozenset({"shadow"})})
    submit_graph(conn, registry(), policy, requests, clock=SystemClock())
    assert chain_blocks_pool_sidecar(conn, "2026-09-18") is True
    assert read_recorded(conn, tmp_path) == frozenset()
