"""#192 slice 4: the scheduled serve holds for a pending pool cycle.

``nightly_trigger._drive_jobs_to_terminal`` gains a ``hold_for_pool`` keyword:
when true, the ``until`` predicate stays false while
``pool_refresh.pool_cycle_pending`` reports the latest session's cycle is still
pending, so the scheduled serve keeps ticking (bounded by the serve deadline)
until the pool release is deployed or the cycle stalls. The scheduled serve
also waits for its eligible native score batch and releases on a recorded
refusal or exhausted submission attempts. The snapshot-import serve never
passes it. All synthetic -- a fake ``serve`` captures the ``until`` callable
and a fake catalog connection answers the terminal-state reads.
"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from engine.v2.ops import nightly_trigger, supervisor

ET = ZoneInfo("America/New_York")


class _FakeClock:
    def now(self):
        return datetime(2026, 9, 26, 8, 0, tzinfo=ET)

    def monotonic(self):
        return 0.0


class _FakeConn:
    """The minimal ``jobs`` table ``_job_states`` reads."""

    def __init__(self, states):
        self._rows = [{"state": state} for state in states]

    def execute(self, sql, params=()):
        return self

    def fetchall(self):
        return self._rows


class _DummyConn:
    def close(self):
        return None


def _capture_until(monkeypatch, outcome="deadline_exceeded"):
    captured = {}

    def fake_serve(service, *, until=None, deadline_at=None):
        captured["until"] = until
        captured["deadline_at"] = deadline_at
        return outcome

    monkeypatch.setattr(supervisor, "serve", fake_serve)
    return captured


def test_hold_for_pool_keeps_until_false_while_a_cycle_is_pending(monkeypatch):
    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_pool_pending", lambda conn: True)
    conn = _FakeConn(("succeeded", "succeeded"))

    outcome = nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("job_a", "job_b"), None, hold_for_pool=True)

    assert outcome == "timed_out"
    assert captured["until"]() is False


def test_hold_for_pool_lets_until_finish_once_the_cycle_is_done(monkeypatch):
    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_native_score_batch_pending",
                        lambda conn, job_ids, service: False)
    monkeypatch.setattr(nightly_trigger, "_pool_pending", lambda conn: False)
    conn = _FakeConn(("succeeded", "succeeded"))

    outcome = nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("job_a", "job_b"), None, hold_for_pool=True)

    assert outcome == "timed_out"
    assert captured["until"]() is True


def test_hold_for_pool_passes_the_conn_and_resolved_release_root(monkeypatch, tmp_path):
    from engine.v2.models import deployment
    from engine.v2.ops import pool_refresh

    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_native_score_batch_pending",
                        lambda conn, job_ids, service: False)
    calls = []
    pending = {"value": True}

    def fake_pending(conn, release_root):
        calls.append((conn, release_root))
        return pending["value"]

    monkeypatch.setattr(deployment, "production_release_root", lambda: tmp_path)
    monkeypatch.setattr(pool_refresh, "pool_cycle_pending", fake_pending)
    conn = _FakeConn(("succeeded", "succeeded"))

    outcome = nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("job_a", "job_b"), None, hold_for_pool=True)

    assert outcome == "timed_out"
    assert captured["until"]() is False
    assert calls == [(conn, tmp_path)]
    pending["value"] = False
    assert captured["until"]() is True
    assert calls == [(conn, tmp_path), (conn, tmp_path)]


def test_pool_pending_exception_does_not_hold(monkeypatch):
    from engine.v2.models import deployment
    from engine.v2.ops import pool_refresh

    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_native_score_batch_pending",
                        lambda conn, job_ids, service: False)

    def broken(conn, release_root):
        raise RuntimeError("boom")

    monkeypatch.setattr(deployment, "production_release_root", lambda: "/release")
    monkeypatch.setattr(pool_refresh, "pool_cycle_pending", broken)
    conn = _FakeConn(("succeeded", "succeeded"))

    nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("job_a", "job_b"), None, hold_for_pool=True)

    assert captured["until"]() is True


def test_default_until_ignores_the_pool(monkeypatch):
    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_pool_pending",
                        lambda conn: (_ for _ in ()).throw(
                            AssertionError("the pool must not be consulted")))
    conn = _FakeConn(("succeeded", "succeeded"))

    outcome = nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("job_a", "job_b"), None)

    assert outcome == "timed_out"
    assert captured["until"]() is True


def test_last_job_still_running_keeps_until_false(monkeypatch):
    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_pool_pending", lambda conn: False)
    conn = _FakeConn(("succeeded", "running"))

    nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("job_a", "job_b"), None, hold_for_pool=True)

    assert captured["until"]() is False


def test_pool_pending_is_false_without_a_release_root(monkeypatch):
    from engine.v2.models import deployment

    def _missing():
        raise deployment.MissingReleaseRoot()

    monkeypatch.setattr(deployment, "production_release_root", _missing)
    assert nightly_trigger._pool_pending(object()) is False


def test_default_serve_holds_for_the_pool(tmp_path, monkeypatch):
    from engine.v2.ops import bootstrap, supervisor as sup
    from engine.v2.ops.workflows import commands

    monkeypatch.setattr(bootstrap, "open_catalog", lambda *a, **k: _DummyConn())
    monkeypatch.setattr(commands, "_submit_command",
                        lambda args, root, conn, clock, host_policy: {"jobs": [{"job_id": "job_1"}]})
    monkeypatch.setattr(sup, "Service", lambda *a, **k: object())
    captured = {}

    def fake_drive(service, conn, job_ids, deadline_at, *, hold_for_pool=False):
        captured["hold_for_pool"] = hold_for_pool
        return "completed"

    monkeypatch.setattr(nightly_trigger, "_drive_jobs_to_terminal", fake_drive)

    assert nightly_trigger._default_serve(tmp_path, "plan_hold", _FakeClock()) == "completed"
    assert captured["hold_for_pool"] is True


class _NativeBatchConn:
    def __init__(self):
        self.batch_state = "queued"
        self.failure_recorded = False
        self.sql = ""
        self.params = ()

    def execute(self, sql, params=()):
        self.sql, self.params = sql, params
        return self

    def fetchall(self):
        if self.sql.startswith("SELECT state FROM jobs WHERE job_id IN"):
            return [{"state": "succeeded"}]
        return []

    def fetchone(self):
        if "kind = 'score'" in self.sql:
            if ("score-job" in self.params
                    and self.params[-1] == "nightly:2026-09-26:sha256:scope:score"):
                return {"1": 1}
            return None
        if "FROM native_stage_failures" in self.sql:
            return {"1": 1} if self.failure_recorded else None
        if self.sql.startswith("SELECT state FROM jobs WHERE job_id = ?"):
            if self.params == ("native-job",) and self.batch_state is not None:
                return {"state": self.batch_state}
        return None


def test_hold_for_pool_waits_for_its_native_score_batch(monkeypatch):
    from engine.v2.ops import nightly, submission

    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_pool_pending", lambda conn: False)
    monkeypatch.setattr(
        nightly, "native_score_batch_identity",
        lambda conn: ("2026-09-26", "sha256:scope", object()))
    monkeypatch.setattr(
        nightly, "native_score_batch_key",
        lambda session, scope: "nightly:2026-09-26:sha256:scope:native_score_batch")
    monkeypatch.setattr(submission, "job_id_for", lambda namespace, key: "native-job")
    conn = _NativeBatchConn()

    outcome = nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("score-job",), None, hold_for_pool=True)

    assert outcome == "timed_out"
    assert captured["until"]() is False
    conn.batch_state = "failed"
    assert captured["until"]() is True
    conn.batch_state = "succeeded"
    assert captured["until"]() is True


class _RetryableNativeWaitService:
    def native_score_batch_submission_exhausted(self, session, scope_hash):
        return False


class _ExhaustedNativeWaitService:
    def native_score_batch_submission_exhausted(self, session, scope_hash):
        return True


def _native_wait_setup(monkeypatch):
    from engine.v2.ops import nightly, submission

    captured = _capture_until(monkeypatch)
    monkeypatch.setattr(nightly_trigger, "_pool_pending", lambda _conn: False)
    monkeypatch.setattr(
        nightly, "native_score_batch_identity",
        lambda _conn: ("2026-09-26", "sha256:scope", object()))
    monkeypatch.setattr(nightly, "native_score_batch_key", lambda *_: "batch-key")
    monkeypatch.setattr(submission, "job_id_for", lambda *_: "native-job")
    return captured


def test_hold_for_pool_releases_when_release_failure_is_recorded(monkeypatch):
    conn = _NativeBatchConn()
    conn.batch_state = None
    conn.failure_recorded = True
    captured = _native_wait_setup(monkeypatch)

    nightly_trigger._drive_jobs_to_terminal(
        None, conn, ("score-job",), None, hold_for_pool=True)

    assert captured["until"]() is True


def test_hold_for_pool_releases_when_submit_attempts_are_exhausted(monkeypatch):
    conn = _NativeBatchConn()
    conn.batch_state = None
    captured = _native_wait_setup(monkeypatch)

    nightly_trigger._drive_jobs_to_terminal(
        _ExhaustedNativeWaitService(), conn, ("score-job",), None,
        hold_for_pool=True)

    assert captured["until"]() is True


def test_hold_for_pool_keeps_waiting_while_submit_can_retry(monkeypatch):
    conn = _NativeBatchConn()
    conn.batch_state = None
    captured = _native_wait_setup(monkeypatch)

    nightly_trigger._drive_jobs_to_terminal(
        _RetryableNativeWaitService(), conn, ("score-job",), None,
        hold_for_pool=True)

    assert captured["until"]() is False
