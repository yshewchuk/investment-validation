"""The claimed attempt's own lease survives pre-launch read-set staging.

``Service._launch`` pins and copies the declared read set before the attempt is
in ``Service.running``, so ``_poll`` cannot renew it. Staging that outlives
``LEASE_SECONDS`` used to expire the attempt's own lease and ``record_launch``
then refused with ``LEASE_LOST`` (observed on a 1.9 GB / 1,322-file nightly
``legacy_finality`` read set). These tests run the real pin, copy, heartbeat,
fence and worker launch on a synthetic production root; only the clock is
fake: each keepalive call first advances it, so staging spans several leases
without sleeping.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import engine.v2.ops.supervisor as supervisor_module
from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
from engine.v2.foundation import ArtifactStore, content_hash, to_document
from engine.v2.ops import executor
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.fingerprints import environment_identity, file_hash, worker_source_manifest
from engine.v2.ops.recovery import fence_foreign_epochs
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import LEASE_SECONDS, Service
from tests.ops_support import POLICY, TEST_POLICY, FakeClock, request, run_until, sample
from tests.test_v2_ops_store_barrier import BOUND_REGISTRY

pytestmark = pytest.mark.xdist_group("serial")

REPO = Path(__file__).resolve().parents[3]
FILES = 6
TICK_SECONDS = 45  # per keepalive call: under one lease, but the calls add up to several


@pytest.fixture(autouse=True)
def _fixed_capacity(monkeypatch):
    monkeypatch.setattr(supervisor_module, "sample_capacity",
                        lambda root, *, clock: sample(clock))


def _slow(real, clock, hook=None, seconds=TICK_SECONDS):
    """Wrap a staging function so every keepalive call first spends ``seconds``."""
    def wrapper(*args, keepalive=None, **kwargs):
        def ticking():
            clock.advance(seconds)
            if hook is not None:
                hook()
            keepalive()
        return real(*args, keepalive=ticking, **kwargs)
    return wrapper


def _setup(tmp_path, monkeypatch, hook=None, seconds=TICK_SECONDS):
    prod = tmp_path / "prod"
    (prod / "data").mkdir(parents=True)
    refs = []
    for index in range(FILES):
        path = prod / "data" / f"f{index}.bin"
        path.write_bytes(b"pinned-%d" % index)
        refs.append(LegacyFileRef(path=f"data/f{index}.bin", content_hash=file_hash(path),
                                  byte_size=path.stat().st_size))
    manifest = to_document(LegacyInputManifest(
        manifest_id="manifest-1", file_refs=tuple(refs), table_contract_refs=(),
        registry_and_model_refs=(), calendar_ref=None, selected_session="2026-09-12",
        finality_receipt_refs=(), knowledge_mode_by_table={}, availability_evidence_refs=(),
        read_set_complete=True, capture_implementation_ref="capture.v1"))
    clock = FakeClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    ref = ArtifactStore(tmp_path).publish_bytes(
        json.dumps(manifest, sort_keys=True).encode(),
        schema_ref="legacy_input_manifest.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    job = submit(conn, BOUND_REGISTRY, POLICY, request(
        "staged", kind="bounded_reader", checkpoint_contract_ref="receipt.v1.0",
        parameters={"expected_ids": ["a"],
                    "input_bindings": {"legacy_manifest.json": ref.artifact_id}},
        input_refs=(ref.artifact_id,),
        implementation_ref=content_hash(worker_source_manifest(REPO)),
        environment_ref=content_hash(environment_identity(1))), clock=clock)
    monkeypatch.setattr(supervisor_module, "pin_read_set",
                        _slow(supervisor_module.pin_read_set, clock, hook, seconds))
    monkeypatch.setattr(supervisor_module, "copy_read_set",
                        _slow(supervisor_module.copy_read_set, clock, hook, seconds))
    service = Service(conn, tmp_path, BOUND_REGISTRY, TEST_POLICY, clock=clock,
                      code_source=REPO, store_root=prod)
    service.start()
    return conn, clock, job, service


def _attempt(conn, job):
    return conn.execute("SELECT * FROM attempts WHERE job_id = ?", (job.job_id,)).fetchone()


def test_staging_longer_than_the_lease_renews_the_claimed_attempts_own_lease(tmp_path,
                                                                             monkeypatch):
    conn, clock, job, service = _setup(tmp_path, monkeypatch)
    start = clock.now()
    try:
        assert service.tick() is True
        attempt = _attempt(conn, job)
        assert (clock.now() - start).total_seconds() > 3 * LEASE_SECONDS
        assert attempt["started_at"] is not None  # record_launch found a live lease
        assert attempt["state"] in ("running", "succeeded")
        assert attempt["failure_json"] is None
        run_until(service, conn, job.job_id, timeout=90)
    finally:
        service.close()
        conn.close()


def test_lost_fence_during_staging_still_refuses_with_lease_lost(tmp_path, monkeypatch):
    calls = {"n": 0}
    leases = {}

    def lose_fence():
        calls["n"] += 1
        if calls["n"] == 3:
            fence_foreign_epochs(conn, epoch_id="sup_competitor", clock=clock)
            leases["after_loss"] = _attempt(conn, job)["lease_expires_at"]

    conn, clock, job, service = _setup(tmp_path, monkeypatch, hook=lose_fence)
    launched = []
    real_launch = executor.launch
    monkeypatch.setattr(executor, "launch",
                        lambda *a, **k: launched.append(1) or real_launch(*a, **k))
    try:
        service.tick()
        attempt = _attempt(conn, job)
        assert launched == []  # the child was never launched
        assert attempt["started_at"] is None
        assert attempt["attempt_id"] not in service.running
        assert attempt["state"] == "recovery_pending"
        # a heartbeat never extends a lease whose fence is gone
        assert attempt["lease_expires_at"] == leases["after_loss"]
        assert calls["n"] == 3  # staging stopped at the first refused renewal
    finally:
        service.close()
        conn.close()


def test_pin_and_copy_share_one_throttle_across_their_boundary(tmp_path, monkeypatch):
    import engine.v2.ops.lifecycle as lifecycle_module

    # Staging spans far less than LEASE_SECONDS / 4, so the claimed attempt's
    # own lease is renewed exactly once: a second keepalive built for the copy
    # phase would heartbeat again the moment the pin phase ends.
    conn, clock, job, service = _setup(tmp_path, monkeypatch, seconds=0.1)
    beats = []
    real_heartbeat = lifecycle_module.heartbeat

    def counting(conn_, attempt_id, *args, **kwargs):
        beats.append(attempt_id)
        return real_heartbeat(conn_, attempt_id, *args, **kwargs)

    monkeypatch.setattr(lifecycle_module, "heartbeat", counting)
    real_stage = service._stage_legacy_inputs
    staged = {}

    def stage(claim, launch):
        before = len(beats)
        try:
            return real_stage(claim, launch)
        finally:
            staged["beats"] = len(beats) - before

    monkeypatch.setattr(service, "_stage_legacy_inputs", stage)
    try:
        assert service.tick() is True
        assert staged["beats"] == 1
        run_until(service, conn, job.job_id, timeout=90)
    finally:
        service.close()
        conn.close()
