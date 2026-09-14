"""Coordinator work after the worker exits (fix after heavy-run stage 9a).

A: a coordinator effect keeps its lease only through explicit ``keepalive``
calls. B: a lost lease inside ``Service._finish`` never escapes ``serve`` and
is settled by the recovery state machine. D: heartbeat rows are throttled.
E: ``ops cancel`` on a never-started job needs no expected attempt.

Real SQLite, a real ``Service`` and a real ``artifact_check`` worker subprocess
under ``TEST_POLICY``; only the coordinator effect is synthetic, and the fake
clock advances inside it to stand in for minutes of publish/verify work.
"""
import json
from datetime import timedelta
from pathlib import Path

import pytest

from engine.v2.foundation import content_hash, format_timestamp
from engine.v2.ops import cli, executor
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import OpsError
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.lifecycle import request_cancel
from engine.v2.ops.recovery import begin_epoch, fence_foreign_epochs
from engine.v2.ops.scheduler import Supervisor
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import HEARTBEAT_EVENT_SECONDS, LEASE_SECONDS, Service, serve
from tests.ops_support import (
    POLICY,
    REGISTRY,
    TEST_POLICY,
    FakeClock,
    catalog,
    enqueue_claim,
    request,
)

#: Real-process recovery: reconcile's ownership proof scans this host's live
#: process table, so it runs in the serial xdist group (see tests/conftest.py).
pytestmark = pytest.mark.xdist_group("serial")

ROOT = Path(__file__).resolve().parents[1]
STEPS, STEP_SECONDS = 8, 50


@pytest.fixture(scope="module")
def implementation_ref():
    return content_hash(worker_source_manifest(ROOT))


def _service(tmp_path, clock, implementation_ref):
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    spec = request(kind="artifact_check", checkpoint_contract_ref="receipt.v1.0",
                   parameters={"expected_ids": ["a", "b"]}, implementation_ref=implementation_ref,
                   environment_ref=content_hash(environment_identity(1)))
    job = submit(conn, registry(), POLICY, spec, clock=clock)
    service = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT)
    return conn, job, service


def _long_effect(monkeypatch, clock, *, keepalive_calls, hook=None):
    """A first-attempt coordinator effect spanning STEPS * STEP_SECONDS of clock time."""
    seen = {"committed": False, "raised": None}

    def effect(self, claim, refs, launch=None, keepalive=None):
        if claim.attempt_number > 1:
            return None, ()
        try:
            for step in range(STEPS):
                clock.advance(STEP_SECONDS)
                if hook is not None:
                    hook(self, claim, step)
                if keepalive_calls:
                    keepalive()
        except OpsError as exc:
            seen["raised"] = exc.code
            raise

        def commit(conn):
            seen["committed"] = True
        return commit, ()

    monkeypatch.setattr(Service, "_coordinator_effect", effect)
    return seen


def _row(conn, sql, *args):
    return conn.execute(sql, args).fetchone()


def test_keepalive_carries_an_effect_past_three_leases(tmp_path, monkeypatch, implementation_ref):
    clock = FakeClock()
    conn, job, service = _service(tmp_path, clock, implementation_ref)
    seen = _long_effect(monkeypatch, clock, keepalive_calls=True)
    serve(service, once=True)
    assert STEPS * STEP_SECONDS > 3 * LEASE_SECONDS
    assert seen == {"committed": True, "raised": None}
    assert _row(conn, "SELECT state FROM jobs WHERE job_id = ?", job.job_id)[0] == "succeeded"


def test_lost_lease_survives_serve_and_recovery_settles_it(tmp_path, monkeypatch,
                                                           implementation_ref):
    clock = FakeClock()
    conn, job, service = _service(tmp_path, clock, implementation_ref)
    seen = _long_effect(monkeypatch, clock, keepalive_calls=False)
    serve(service, once=True)  # must return normally: no LEASE_LOST escapes
    assert seen["committed"] is False
    attempt = _row(conn, "SELECT * FROM attempts WHERE job_id = ?", job.job_id)
    assert attempt["state"] == "recovery_pending"
    job_row = _row(conn, "SELECT * FROM jobs WHERE job_id = ?", job.job_id)
    assert job_row["state"] == "running" and job_row["fence"] == attempt["fence"] + 1
    assert _row(conn, "SELECT COUNT(*) FROM attempt_outputs")[0] == 0
    assert _row(conn, "SELECT COUNT(*) FROM resource_reservations WHERE released_at IS NULL")[0] == 1

    restarted = Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT)
    restarted.start()  # a supervisor restart reconciles every recovery_pending attempt
    restarted.close()
    attempt = _row(conn, "SELECT * FROM attempts WHERE job_id = ?", job.job_id)
    assert (attempt["state"], attempt["process_state"]) == ("failed", "verified_dead")
    assert json.loads(attempt["failure_json"])["code"] == "LEASE_LOST"
    assert _row(conn, "SELECT state FROM jobs WHERE job_id = ?", job.job_id)[0] == "retry_wait"
    assert _row(conn, "SELECT COUNT(*) FROM resource_reservations WHERE released_at IS NULL")[0] == 0

    clock.advance(60)  # past the retry delay: the stranded job completes on its next attempt
    serve(Service(conn, tmp_path, registry(), TEST_POLICY, clock=clock, code_source=ROOT),
          once=True)
    assert _row(conn, "SELECT state FROM jobs WHERE job_id = ?", job.job_id)[0] == "succeeded"


def test_competing_reclaim_mid_effect_stops_work_and_commits_nothing(tmp_path, monkeypatch,
                                                                    implementation_ref):
    clock = FakeClock()
    conn, job, service = _service(tmp_path, clock, implementation_ref)

    def reclaim(svc, claim, step):
        if step == 3:
            fence_foreign_epochs(svc.conn, epoch_id="sup_competitor", clock=clock)

    seen = _long_effect(monkeypatch, clock, keepalive_calls=True, hook=reclaim)
    serve(service, once=True)
    assert seen == {"committed": False, "raised": "LEASE_LOST"}
    attempt = _row(conn, "SELECT * FROM attempts WHERE job_id = ?", job.job_id)
    assert attempt["state"] == "recovery_pending"
    assert _row(conn, "SELECT fence FROM jobs WHERE job_id = ?", job.job_id)[0] == attempt["fence"] + 1
    assert _row(conn, "SELECT COUNT(*) FROM attempt_outputs")[0] == 0


def test_cancel_during_effect_completes_the_cancellation(tmp_path, monkeypatch,
                                                         implementation_ref):
    clock = FakeClock()
    conn, job, service = _service(tmp_path, clock, implementation_ref)

    def cancel(svc, claim, step):
        if step == 2:
            request_cancel(svc.conn, claim.job_id, claim.attempt_id, clock=clock)

    seen = _long_effect(monkeypatch, clock, keepalive_calls=True, hook=cancel)
    serve(service, once=True)
    assert seen["committed"] is False
    assert _row(conn, "SELECT state FROM jobs WHERE job_id = ?", job.job_id)[0] == "cancelled"
    attempt = _row(conn, "SELECT * FROM attempts WHERE job_id = ?", job.job_id)
    assert (attempt["state"], attempt["process_state"]) == ("cancelled", "verified_dead")


# --------------------------------------------------------------------------
# D: heartbeat rows
# --------------------------------------------------------------------------


def test_heartbeat_rows_are_throttled_but_state_changes_are_recorded(tmp_path, monkeypatch):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    service = Service(conn, tmp_path, REGISTRY, TEST_POLICY, clock=clock, code_source=ROOT)
    running = executor.Running(claim=claim, process=None, result_fd=-1, identities=(),
                               started=clock.monotonic())
    status = {"done": False, "memory": 4096, "exit_code": None}
    monkeypatch.setattr(executor, "poll", lambda *args, **kwargs: dict(status))
    finished = []
    monkeypatch.setattr(Service, "_finish", lambda self, run, observed: finished.append(observed))

    def heartbeats():
        return _row(conn, "SELECT COUNT(*) FROM progress_events WHERE attempt_id = ? "
                    "AND kind = 'heartbeat'", claim.attempt_id)[0]

    for _ in range(600):  # 60 s of polling every 0.1 s
        clock.advance(0.1)
        assert service._poll(running) is False
    assert HEARTBEAT_EVENT_SECONDS == 10.0
    assert 1 <= heartbeats() <= 7
    # lease renewal is unthrottled: renewed on the very last poll
    expected_lease = format_timestamp(clock.now() + timedelta(seconds=LEASE_SECONDS))
    assert _row(conn, "SELECT lease_expires_at FROM attempts WHERE attempt_id = ?",
                claim.attempt_id)[0] == expected_lease

    before = heartbeats()
    running.failure = "RESOURCE_LIMIT_EXCEEDED"
    clock.advance(0.1)
    service._poll(running)
    assert heartbeats() == before + 1
    status.update(done=True, exit_code=-9)
    clock.advance(0.1)
    assert service._poll(running) is True
    assert heartbeats() == before + 2 and len(finished) == 1
    messages = [json.loads(row[0])["message"] for row in conn.execute(
        "SELECT body_json FROM progress_events WHERE attempt_id = ? ORDER BY sequence",
        (claim.attempt_id,)).fetchall()]
    assert messages[-2:] == ["worker stopping: RESOURCE_LIMIT_EXCEEDED", "worker exited"]


# --------------------------------------------------------------------------
# E: cancel
# --------------------------------------------------------------------------


def _cli_catalog(tmp_path):
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    clock = FakeClock()
    return root, open_catalog(root / "catalog.sqlite", clock=clock), clock


def _cancel(root, capsys, job_id, *extra):
    capsys.readouterr()
    assert cli.main(["--root", str(root), "cancel", job_id, *extra]) == 0
    return json.loads(capsys.readouterr().out)


def test_cancel_never_started_job_without_expected_attempt(tmp_path, capsys):
    root, conn, clock = _cli_catalog(tmp_path)
    job = submit(conn, REGISTRY, POLICY, request("never-started"), clock=clock)
    assert _row(conn, "SELECT active_attempt_id FROM jobs WHERE job_id = ?", job.job_id)[0] is None
    assert _cancel(root, capsys, job.job_id)["state"] == "cancelled"
    assert _row(conn, "SELECT state FROM jobs WHERE job_id = ?", job.job_id)[0] == "cancelled"


def test_cancel_with_stale_or_omitted_expectation_still_refuses(tmp_path, capsys):
    root, conn, clock = _cli_catalog(tmp_path)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    claim = enqueue_claim(conn, clock, Supervisor(epoch, "boot"))
    for extra in ((), ("--expected-attempt", "att_stale")):
        receipt = _cancel(root, capsys, claim.job_id, *extra)
        assert receipt["state"] == "conflict"
        assert receipt["failure"]["code"] == "STALE_EXPECTATION"
    assert _row(conn, "SELECT state FROM jobs WHERE job_id = ?", claim.job_id)[0] == "running"
    receipt = _cancel(root, capsys, claim.job_id, "--expected-attempt", claim.attempt_id)
    assert receipt["state"] == "cancelling"
