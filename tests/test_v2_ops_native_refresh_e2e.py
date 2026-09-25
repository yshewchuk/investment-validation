"""S4A end-to-end: a native daily_market refresh completes one real attempt.

The attempt's identity document is staged by the executor
(``engine.v2.ops.refresh_staging.stage_refresh_input``), the worker subprocess
is the real ``engine.v2.ops.worker`` (its argv swapped for a stub -- the seam
``test_v2_ops_worker_typed_failures.py`` uses) with a fake fetcher bound in
place of a live provider, and the commit runs through the unchanged
``run_incremental_refresh``. No network and no market data are used.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from engine.v2.contracts import SubmitRequest
from engine.v2.data import incremental as data_incremental
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.manifests import dataset_manifest, snapshot_ref
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactStore,
    canonical_json,
    content_hash,
    to_document,
)
from engine.v2.ops import executor
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.incremental_data import (
    REFRESH_PLAN_SCHEMA,
    RefreshParameters,
    RefreshUnit,
    plan_refresh,
    refresh_job_spec,
)
from engine.v2.ops.profiles import profile_named
from engine.v2.ops.provider_budget import configure_account
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, run_until
from tests.test_v2_data_manifests import (
    _DAILY_MARKET_CONTRACT,
    _DAILY_MARKET_REF,
    _daily_market_rows,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})
SCOPE = "shadow"
REQUEST_ID = "eod-2026-01-02-AAA"
ACCOUNT = "s4a-account"


def _commit_parent(conn, store, clock):
    receipt_ref = content_hash({"s4a": "parent-receipt"})
    manifest = dataset_manifest(
        _DAILY_MARKET_REF, (), knowledge_mode="reconstructed",
        coverage_receipt_refs=(receipt_ref,), availability_evidence_refs=())
    snapshot = snapshot_ref(
        {"daily_market": manifest}, calendar_version="cal.v1",
        source_priority_version="prio.v1", finality_receipt_refs=(receipt_ref,))
    commit_snapshot(
        conn, scope=SCOPE, request_hash=content_hash({"s4a": "base"}),
        contracts=(_DAILY_MARKET_CONTRACT,), objects=(), records=(), manifests=(manifest,),
        snapshot=snapshot, expected_head_snapshot_id=None, expected_head_generation=0,
        receipt_id="s4a-base", attempt_id="s4a-base", fence=1,
        fence_check=lambda _conn: None, clock=clock, store=store)


def _head(conn):
    return conn.execute(
        "SELECT snapshot_id, generation FROM data_snapshot_heads WHERE scope = ?",
        (SCOPE,)).fetchone()


def _install_fake_fetcher(monkeypatch, row):
    """Bind a canned response as the worker's provider edge (real subprocess)."""
    stub = (
        "import functools, json, sys\n"
        "import engine.v2.data.incremental as incremental\n"
        "from engine.v2.ops import worker\n"
        "ROW = json.loads(%r)\n"
        "def fake_fetcher(unit):\n"
        "    row = dict(ROW)\n"
        "    return (json.dumps(row).encode(), \"complete\", {\"status\": 200}, [row])\n"
        "incremental.run_daily_market_refresh = functools.partial(\n"
        "    incremental.run_daily_market_refresh, fetcher=fake_fetcher)\n"
        "raise SystemExit(worker.main())\n"
    ) % json.dumps(row)
    real = subprocess.Popen

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", stub]
        return real(args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)


def _submit_refresh(conn, store, clock, tmp_path, head):
    parent = Repository(conn).resolve(head["snapshot_id"])
    unit = RefreshUnit(request_id=REQUEST_ID, table_name="daily_market",
                       partition_key="2026", expected_keys=("AAA",))
    configure_account(conn, ACCOUNT, "generation-1", remaining=5, live_reserve=1)
    refresh_plan = plan_refresh(
        parent, (unit,), cached_outcomes={}, provider_account=ACCOUNT,
        max_attempts=1, expected_head_generation=head["generation"])
    plan_ref = store.publish_bytes(
        canonical_json(to_document(refresh_plan)).encode(), schema_ref=REFRESH_PLAN_SCHEMA)
    with transaction(conn):
        register_artifact(conn, plan_ref, None, clock)
    profile = profile_named(TEST_POLICY, "io_fetch")
    job = refresh_job_spec(
        refresh_plan, implementation_ref=content_hash(worker_source_manifest(ROOT)),
        environment_ref=content_hash(
            environment_identity(profile.thread_count or profile.cpu_count)),
        output_namespace=SCOPE, catalog_path=str(tmp_path / "ops.sqlite"),
        objects_root=str(tmp_path),
        input_bindings={"refresh_plan.json": plan_ref.artifact_id},
        input_refs=(plan_ref.artifact_id,))
    return submit(conn, registry(), POLICY, SubmitRequest(
        namespace=SCOPE, idempotency_key="s4a-refresh", principal="operator", job=job),
        clock=clock)


def test_native_daily_market_refresh_commits_one_attempt_end_to_end(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)
    assert (head["snapshot_id"], head["generation"]) != (None, None)

    row = json.loads(json.dumps(_daily_market_rows(2026)[0], default=str))
    _install_fake_fetcher(monkeypatch, row)
    receipt = _submit_refresh(conn, store, clock, tmp_path, head)

    service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT)
    service.start()
    try:
        state = run_until(service, conn, receipt.job_id, timeout=120)
        if state != "succeeded":
            attempt = conn.execute(
                "SELECT attempt_id FROM attempts WHERE job_id = ? "
                "ORDER BY attempt_number DESC LIMIT 1", (receipt.job_id,)).fetchone()
            diagnostics = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics"
            stderr = (diagnostics / "worker.stderr").read_text()
            failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                                   (receipt.job_id,)).fetchone()[0]
            raise AssertionError(f"refresh attempt did not succeed: {failure}\n{stderr}")
    finally:
        service.close()

    new_head = _head(conn)
    assert new_head["generation"] == head["generation"] + 1
    assert new_head["snapshot_id"] != head["snapshot_id"]
    revisions = conn.execute(
        "SELECT ticker, session_date FROM data_daily_market_revisions").fetchall()
    assert [(item["ticker"], item["session_date"]) for item in revisions] == [
        ("AAA", "2026-01-02")]


def test_refresh_without_staged_identity_fails_closed_without_catalog_change(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    _commit_parent(conn, store, clock)
    head = _head(conn)

    empty_root = tmp_path / "attempt-without-staging"
    empty_root.mkdir()
    result = data_incremental.run_incremental_refresh(
        RefreshParameters(
            expected_ids=(REQUEST_ID,), parent_snapshot_id=head["snapshot_id"],
            refresh_plan_hash="sha256:" + "b" * 64, provider_calls=0,
            catalog_path=str(tmp_path / "ops.sqlite"), objects_root=str(tmp_path),
            scope=SCOPE, expected_head_generation=head["generation"],
            expected_head_snapshot_id=head["snapshot_id"]),
        empty_root)

    assert result["status"] == "failed"
    assert result["coverage_advanced"] is False
    assert result["candidate_snapshot_id"] is None
    unchanged = _head(conn)
    assert (unchanged["snapshot_id"], unchanged["generation"]) == (
        head["snapshot_id"], head["generation"])
    assert conn.execute("SELECT COUNT(*) FROM data_daily_market_revisions").fetchone()[0] == 0
