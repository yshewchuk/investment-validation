"""B1/B2 recovery-ownership fault tests: real short-lived child processes.

B1 (permanent quarantine): a process-group/ppid walk alone cannot prove a
process tree gone — it misses a ``setsid()`` escaper and a launch that
crashed before its identity was recorded. ``prove_ownership_gone`` adds a
session check and an environ-marker check for exactly those two cases, and
only then may a reservation be released.

B2 (clock jump): a wall-clock jump must renew a still-alive worker's lease
rather than fence it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.foundation import SystemClock
from engine.v2.ops import cli
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.executor_watchdog import process_info, signal_owned
from engine.v2.ops.lifecycle import Outcome, commit_attempt, record_launch
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.recovery import (
    SupervisorLock,
    begin_epoch,
    expire_leases,
    fence_foreign_epochs,
    read_boot_id,
)
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import POLICY, TEST_POLICY, catalog, request, sample

CODE_SOURCE = Path(__file__).resolve().parents[1]

# Real short-lived child processes with process-group/session signaling and a
# fixed-deadline poll for one to die (see tests/conftest.py's grouping rule).
pytestmark = pytest.mark.xdist_group("serial")


def _child(source, *, env=None, new_session=True):
    return subprocess.Popen([sys.executable, "-u", "-c", source], start_new_session=new_session,
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _gone(process):
    for _ in range(50):
        if process.poll() is not None:
            return True
        time.sleep(0.01)
    return process.poll() is not None


def _wait_for(path):
    for _ in range(100):
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def _fenced_claim(conn, clock, boot, *, key="one", kind="artifact_check"):
    """A claimed attempt belonging to a now-foreign epoch, ready to launch."""
    epoch = begin_epoch(conn, clock=clock, boot_id=boot, pid=1)
    supervisor = Supervisor(epoch, boot)
    submit(conn, registry(), POLICY, request(key, kind=kind, checkpoint_contract_ref="receipt.v1.0",
           parameters={"expected_ids": []}, implementation_ref="code", environment_ref="env"),
           clock=clock)
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                      supervisor=supervisor, clock=clock)


def _service(conn, clock, boot, root):
    service = Service(conn, root, registry(), TEST_POLICY, clock=clock, code_source=CODE_SOURCE)
    service.boot = boot
    return service


def _reservation_released(conn, attempt_id):
    return conn.execute("SELECT released_at FROM resource_reservations WHERE attempt_id = ?",
                        (attempt_id,)).fetchone()[0]


def _process_state(conn, attempt_id):
    return conn.execute("SELECT process_state FROM attempts WHERE attempt_id = ?",
                        (attempt_id,)).fetchone()[0]


# --------------------------------------------------------------------------
# 1. a single-process worker that exited
# --------------------------------------------------------------------------


def test_single_process_exit_settles_verified_dead(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    boot = read_boot_id()
    claim = _fenced_claim(conn, clock, boot)
    child = _child("import time; time.sleep(0.05)")
    try:
        identity = process_info(child.pid, boot)[0]
        record_launch(conn, claim.attempt_id, claim.fence, identity, clock=clock, lease_seconds=120)
        assert _gone(child)
        service = _service(conn, clock, boot, tmp_path)
        service.start()
        for _ in range(4):
            service.reconcile()
        assert _process_state(conn, claim.attempt_id) == "verified_dead"
        assert _reservation_released(conn, claim.attempt_id) is not None
        job_state = conn.execute("SELECT state FROM jobs WHERE job_id = ?",
                                 (claim.job_id,)).fetchone()[0]
        assert job_state != "running"
        service.close()
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


# --------------------------------------------------------------------------
# 2. a setsid() escaper, unreachable by a process-group/ppid walk
# --------------------------------------------------------------------------


def test_setsid_escaper_stays_quarantined_until_killed(tmp_path):
    """The recorded identity dies; a separate, still-live process calling
    ``os.setsid()`` carries the attempt's env marker. A walk from the dead
    identity can never reach it (it is not even a descendant of it) — only
    the environ-marker check (c) can, and must, still block on it.
    """
    conn, clock, _ = catalog(tmp_path)
    boot = read_boot_id()
    claim = _fenced_claim(conn, clock, boot)
    dead = _child("import time; time.sleep(0.05)")
    ready = tmp_path / "ready"
    escaper_env = dict(os.environ, OPS_TEST_MARKER=claim.attempt_id)
    escaper = _child(
        "import os, pathlib, time; os.setsid(); "
        f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)",
        env=escaper_env, new_session=False)
    try:
        identity = process_info(dead.pid, boot)[0]
        record_launch(conn, claim.attempt_id, claim.fence, identity, clock=clock, lease_seconds=120)
        assert _gone(dead)
        assert _wait_for(ready)
        service = _service(conn, clock, boot, tmp_path)
        service.start()
        assert _process_state(conn, claim.attempt_id) == "quarantined"
        assert _reservation_released(conn, claim.attempt_id) is None
        escaper.kill()
        escaper.wait()
        service.reconcile()
        assert _process_state(conn, claim.attempt_id) == "verified_dead"
        assert _reservation_released(conn, claim.attempt_id) is not None
        service.close()
    finally:
        if escaper.poll() is None:
            escaper.kill()
        escaper.wait()
        if dead.poll() is None:
            dead.kill()
        dead.wait()


# --------------------------------------------------------------------------
# 3. no recorded identity at all (crash between claim and launch record)
# --------------------------------------------------------------------------


def test_no_identity_with_live_marker_stays_quarantined_until_gone(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    boot = read_boot_id()
    claim = _fenced_claim(conn, clock, boot)
    ready = tmp_path / "ready"
    marker_env = dict(os.environ, OPS_TEST_MARKER=claim.attempt_id)
    orphan = _child(
        "import pathlib, time; "
        f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)",
        env=marker_env)
    try:
        assert _wait_for(ready)
        # No record_launch: process_json and process_members stay empty.
        service = _service(conn, clock, boot, tmp_path)
        service.start()
        assert _process_state(conn, claim.attempt_id) == "quarantined"
        assert _reservation_released(conn, claim.attempt_id) is None
        orphan.kill()
        orphan.wait()
        service.reconcile()
        assert _process_state(conn, claim.attempt_id) == "verified_dead"
        assert _reservation_released(conn, claim.attempt_id) is not None
        service.close()
    finally:
        if orphan.poll() is None:
            orphan.kill()
        orphan.wait()


# --------------------------------------------------------------------------
# 4. a pid-reuse imposter is never signalled
# --------------------------------------------------------------------------


def test_pid_reuse_imposter_is_never_signalled(tmp_path):
    child = _child("import time; time.sleep(5)")
    try:
        real = process_info(child.pid, "boot")[0]
        imposter = replace(real, start_ticks=real.start_ticks + 1)
        signal_owned((imposter,), "boot", hard=True)
        time.sleep(0.1)
        assert child.poll() is None
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


# --------------------------------------------------------------------------
# 5. a wall-clock jump must renew, not fence, a live worker
# --------------------------------------------------------------------------


def test_clock_jump_renews_live_worker_and_can_still_commit(tmp_path):
    conn, clock, _ = catalog(tmp_path)
    boot = read_boot_id()
    claim = _fenced_claim(conn, clock, boot)
    child = _child("import time; time.sleep(30)")
    try:
        identity = process_info(child.pid, boot)[0]
        record_launch(conn, claim.attempt_id, claim.fence, identity, clock=clock, lease_seconds=120)
        before = conn.execute("SELECT lease_expires_at FROM attempts WHERE attempt_id = ?",
                              (claim.attempt_id,)).fetchone()[0]
        service = _service(conn, clock, boot, tmp_path)
        service.running[claim.attempt_id] = SimpleNamespace(claim=claim, identities=(identity,))
        clock.value += timedelta(seconds=600)  # wall jumps; monotonic does not advance
        service._clock_check()
        row = conn.execute("SELECT state, lease_expires_at FROM attempts WHERE attempt_id = ?",
                           (claim.attempt_id,)).fetchone()
        assert row["state"] == "running"
        assert row["lease_expires_at"] > before
        assert claim.attempt_id in service.running
        assert child.poll() is None
        expire_leases(conn, clock=clock)
        assert conn.execute("SELECT state FROM attempts WHERE attempt_id = ?",
                            (claim.attempt_id,)).fetchone()[0] == "running"
        state = commit_attempt(conn, claim.attempt_id, claim.fence,
                               Outcome(True, "verified_dead", 0), clock=clock)
        assert state == "succeeded"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


# --------------------------------------------------------------------------
# 6. the operator CLI command
# --------------------------------------------------------------------------


def test_cli_reconcile(tmp_path, capsys):
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()

    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    boot = read_boot_id()

    dead_claim = _fenced_claim(conn, clock, boot, key="dead")
    dead_child = _child("import time; time.sleep(0.05)")
    identity = process_info(dead_child.pid, boot)[0]
    record_launch(conn, dead_claim.attempt_id, dead_claim.fence, identity, clock=clock,
                  lease_seconds=120)
    assert _gone(dead_child)

    live_claim = _fenced_claim(conn, clock, boot, key="live")
    ready = tmp_path / "ready"
    marker_env = dict(os.environ, OPS_TEST_MARKER=live_claim.attempt_id)
    live_marker = _child(
        "import pathlib, time; "
        f"pathlib.Path({str(ready)!r}).write_text('ready'); time.sleep(30)",
        env=marker_env)
    assert _wait_for(ready)

    # Fence both into recovery_pending together, as a restarted supervisor would.
    fence_foreign_epochs(conn, epoch_id="not_either_epoch", clock=clock)
    conn.close()

    lock = None
    try:
        # a running supervisor already holds the lock: refused outright.
        lock = SupervisorLock(root / "supervisor.lock")
        assert lock.acquire()
        rc = cli.main(["--root", str(root), "reconcile", dead_claim.job_id,
                       "--expected-attempt", dead_claim.attempt_id])
        doc = json.loads(capsys.readouterr().out)
        assert rc == 2
        assert doc["code"] == "RESOURCE_UNAVAILABLE"
        lock.release()

        # a stale --expected-attempt conflicts.
        rc = cli.main(["--root", str(root), "reconcile", dead_claim.job_id,
                       "--expected-attempt", live_claim.attempt_id])
        doc = json.loads(capsys.readouterr().out)
        assert rc == 2
        assert doc["code"] == "STALE_EXPECTATION"

        # a provably dead attempt settles.
        rc = cli.main(["--root", str(root), "reconcile", dead_claim.job_id,
                       "--expected-attempt", dead_claim.attempt_id])
        doc = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert doc["settled"] is True
        verify = open_catalog(root / "catalog.sqlite", clock=clock)
        assert _process_state(verify, dead_claim.attempt_id) == "verified_dead"
        assert _reservation_released(verify, dead_claim.attempt_id) is not None
        verify.close()

        # a blocked attempt lists its blockers as pid + start_ticks only.
        rc = cli.main(["--root", str(root), "reconcile", live_claim.job_id,
                       "--expected-attempt", live_claim.attempt_id])
        doc = json.loads(capsys.readouterr().out)
        assert rc == 2
        assert doc["code"] == "RESOURCE_UNAVAILABLE"
        blocking = doc["details"]["blocking"]
        assert blocking
        for entry in blocking:
            assert set(entry) == {"pid", "start_ticks"}
    finally:
        if live_marker.poll() is None:
            live_marker.kill()
        live_marker.wait()
        if lock is not None and lock.held:
            lock.release()
