"""Issue #1239: the serve loop's leases survive an off-loop renewal path.

A single long effect on the supervisor's one thread used to starve every other
live attempt of its heartbeat, so the next ``expire_leases`` fenced a healthy
sibling. Two independent guards close that gap, both exercised here on a real
catalog with ONLY a fake clock (nothing sleeps a lease period):

* the dedicated ``LeaseRenewer`` -- its own catalog connection, fenced
  :func:`heartbeat` calls, this epoch's live ``starting``/``running`` attempts
  only, and it never revives a void fence, a cancellation or an expired lease;
* the widened ``_clock_check`` -- a monotonic tick pause that outlasted a lease
  renews the verified-live tracked attempts before the tick's expiry sweep.

``Service.start``/``close`` wiring (the renewer starts after identity
initialization and stops-and-joins before the process lock releases) is checked
with a spy so the ordering is deterministic.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import types
from datetime import timedelta

import pytest

import engine.v2.ops.supervisor as supervisor_module
from engine.v2.foundation import format_timestamp
from engine.v2.ops import recovery
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.errors import make_problem
from engine.v2.ops.recovery import begin_epoch, expire_leases
from engine.v2.ops.scheduler import Supervisor
from engine.v2.ops.supervisor import LEASE_SECONDS, LeaseRenewer, Service
from tests.ops_support import TEST_POLICY, FakeClock, catalog
from tools.v2_ops_fixtures import enqueue_claim

pytestmark = pytest.mark.xdist_group("serial")

EPOCH_BOOT = "boot"
DB_NAME = "ops.sqlite"


def _stamp(clock, offset_seconds):
    """``lease_expires_at``-compatible string via the real formatter."""
    return format_timestamp(clock.now() + timedelta(seconds=offset_seconds))


def _lease(conn, attempt_id):
    return conn.execute("SELECT lease_expires_at, heartbeat_at FROM attempts "
                        "WHERE attempt_id = ?", (attempt_id,)).fetchone()


def _log_payloads(stream):
    payloads = []
    for line in stream.text().splitlines():
        if "{" in line:
            try:
                payloads.append(json.loads(line[line.index("{"):]))
            except ValueError:
                continue
    return payloads


class _CapturingStream:
    """A stderr stand-in that collects the renewer's bounded progress lines."""

    def __init__(self):
        self._lines = []
        self._lock = threading.Lock()

    def write(self, text):
        with self._lock:
            self._lines.append(text)

    def flush(self):
        return None

    def text(self):
        with self._lock:
            return "".join(self._lines)


class _Running:
    """The minimum tracked-worker shape ``_resume_after_jump`` reads: the first
    identity and the claim's fence."""

    def __init__(self, attempt_id, fence, marker):
        self.claim = types.SimpleNamespace(attempt_id=attempt_id, fence=fence)
        self.identities = (marker,)


# ---------------------------------------------------------------------------
# LeaseRenewer.renew_pass semantics (driven directly on its own connection)
# ---------------------------------------------------------------------------


def test_renew_pass_extends_this_epoch_live_attempts_on_its_own_connection(tmp_path):
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="live")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                 (_stamp(clock, 1), claim.attempt_id))
    before = _lease(conn, claim.attempt_id)

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                               lease_seconds=LEASE_SECONDS)
        renewed, total = renewer.renew_pass(own)
        after = _lease(conn, claim.attempt_id)
    finally:
        own.close()
        conn.close()

    assert (renewed, total) == (1, 1)
    assert after["lease_expires_at"] > before["lease_expires_at"]
    assert after["heartbeat_at"] is not None


def test_renew_pass_never_revives_void_fence_cancellation_or_expired_lease(tmp_path):
    conn, clock, sup = catalog(tmp_path)
    # (a) a current-epoch attempt whose lease has ALREADY expired.
    expired = enqueue_claim(conn, clock, sup, key="expired")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                 (_stamp(clock, -1), expired.attempt_id))
    # (b) an attempt whose job is cancelling (its fence is void for renewal).
    cancelling = enqueue_claim(conn, clock, sup, key="cancelling")
    conn.execute("UPDATE jobs SET state = 'cancelling' WHERE job_id = ?", (cancelling.job_id,))
    # (c) a live attempt under a DIFFERENT supervisor epoch.
    foreign = enqueue_claim(conn, clock,
                            Supervisor(begin_epoch(conn, clock=clock, boot_id=EPOCH_BOOT, pid=2),
                                       EPOCH_BOOT), key="foreign")
    before = {row.attempt_id: _lease(conn, row.attempt_id)["lease_expires_at"]
              for row in (expired, cancelling, foreign)}

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                               lease_seconds=LEASE_SECONDS)
        renewed, total = renewer.renew_pass(own)
        after = {aid: _lease(conn, aid)["lease_expires_at"] for aid in before}
    finally:
        own.close()
        conn.close()

    # Only the live same-epoch unexpired attempt is selected; the cancelling and
    # foreign-epoch rows are not queried at all, and the expired row IS queried
    # but its fenced heartbeat refuses -- a dead lease is never revived.
    assert (renewed, total) == (0, 1)
    for attempt_id, snapshot in before.items():
        assert after[attempt_id] == snapshot


def test_renew_pass_selects_only_the_current_fence_running_attempt(tmp_path):
    """A ``running`` attempt whose JOB fence was bumped is never handed to heartbeat.

    ``_LIVE_EPOCH_ATTEMPT_FILTER`` joins ``jobs.fence = attempts.fence``, so a job whose
    fence advanced past its own attempt row (recovery fenced it off) is excluded at
    SELECT -- not merely refused inside the fenced heartbeat. Two live ``running``
    claims under this epoch, both leases captured, one second of clock drift, then a
    pass on the renewer's own connection renews exactly the current-fence row and
    leaves the mismatched-fence lease untouched."""
    conn, clock, sup = catalog(tmp_path)
    valid = enqueue_claim(conn, clock, sup, key="valid")
    stale = enqueue_claim(conn, clock, sup, key="stale")
    conn.execute("UPDATE attempts SET state = 'running' WHERE attempt_id IN (?, ?)",
                 (valid.attempt_id, stale.attempt_id))
    # Void ONLY the stale job's fence; leave its attempt fence current so the two
    # diverge. The valid job/attempt pair stays matched.
    conn.execute("UPDATE jobs SET fence = fence + 1 WHERE job_id = ?", (stale.job_id,))

    before = {aid: _lease(conn, aid)["lease_expires_at"]
              for aid in (valid.attempt_id, stale.attempt_id)}
    clock.advance(1)

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                               lease_seconds=LEASE_SECONDS)
        renewed, total = renewer.renew_pass(own)
        after = {aid: _lease(conn, aid)["lease_expires_at"] for aid in before}
    finally:
        own.close()
        conn.close()

    # Only the active same-fence running row is even selected; the mismatched row is
    # excluded by the join, so a single renewal is committed.
    assert (renewed, total) == (1, 1)
    assert after[valid.attempt_id] > before[valid.attempt_id]
    assert after[stale.attempt_id] == before[stale.attempt_id]


def test_lease_loss_does_not_refund_attempt_count(tmp_path):
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="count")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                 (_stamp(clock, -1), claim.attempt_id))
    fenced = expire_leases(conn, clock=clock)
    job = conn.execute("SELECT attempt_count, state FROM jobs WHERE job_id = ?",
                       (claim.job_id,)).fetchone()
    conn.close()
    assert claim.attempt_id in fenced
    assert job["attempt_count"] == 1  # claiming incremented it; lease loss never refunds
    assert job["state"] == "running"


def test_renew_pass_query_failure_is_contained_and_retryable(tmp_path):
    conn, clock, sup = catalog(tmp_path)
    enqueue_claim(conn, clock, sup, key="queryfail")
    stream = _CapturingStream()

    class _FailingConnection:
        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS, stream=stream)
    assert renewer.renew_pass(_FailingConnection()) == (0, 0)
    assert renewer.passes == 1
    assert renewer.errors == 1
    logged = stream.text()
    assert "lease query failed; retried next interval" in logged
    payload = _log_payloads(stream)[-1]
    assert payload["error"] == "OperationalError"

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewed, total = renewer.renew_pass(own)
    finally:
        own.close()
    assert (renewed, total) == (1, 1)
    assert renewer.passes == 2
    conn.close()


def test_renew_pass_continues_after_one_heartbeat_failure(tmp_path, monkeypatch):
    conn, clock, sup = catalog(tmp_path)
    healthy = enqueue_claim(conn, clock, sup, key="healthy")
    failing = enqueue_claim(conn, clock, sup, key="failing")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id IN (?, ?)",
                 (_stamp(clock, 1), healthy.attempt_id, failing.attempt_id))
    healthy_before = _lease(conn, healthy.attempt_id)
    failing_before = _lease(conn, failing.attempt_id)

    real_heartbeat = supervisor_module.heartbeat

    def selective_heartbeat(conn, attempt_id, fence, *, clock, lease_seconds):
        if attempt_id == failing.attempt_id:
            raise sqlite3.OperationalError("locked for " + attempt_id)
        return real_heartbeat(conn, attempt_id, fence, clock=clock, lease_seconds=lease_seconds)

    monkeypatch.setattr(supervisor_module, "heartbeat", selective_heartbeat)
    stream = _CapturingStream()

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                               lease_seconds=LEASE_SECONDS, stream=stream)
        renewed, total = renewer.renew_pass(own)
    finally:
        own.close()

    healthy_after = _lease(conn, healthy.attempt_id)
    failing_after = _lease(conn, failing.attempt_id)
    assert (renewed, total) == (1, 2)
    assert renewer.errors == 1
    assert healthy_after["lease_expires_at"] > healthy_before["lease_expires_at"]
    assert failing_after["lease_expires_at"] == failing_before["lease_expires_at"]

    logged = stream.text()
    assert "renewal errors this pass" in logged
    summary = next(p for p in _log_payloads(stream) if "failed_attempts" in p)
    assert summary["failed"] == 1
    assert summary["failed_attempts"] == [
        {"attempt_id": failing.attempt_id, "error": "OperationalError"}]
    conn.close()


def test_renew_pass_logs_bounded_progress_with_injected_clock(tmp_path, monkeypatch):
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="progress")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                 (_stamp(clock, 100), claim.attempt_id))

    real_heartbeat = supervisor_module.heartbeat

    def advancing_heartbeat(conn, attempt_id, fence, *, clock, lease_seconds):
        clock.advance(61)
        return real_heartbeat(conn, attempt_id, fence, clock=clock, lease_seconds=lease_seconds)

    monkeypatch.setattr(supervisor_module, "heartbeat", advancing_heartbeat)
    stream = _CapturingStream()

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                               lease_seconds=LEASE_SECONDS, stream=stream)
        renewed, total = renewer.renew_pass(own)
    finally:
        own.close()

    assert (renewed, total) == (1, 1)
    assert renewer.passes == 1

    logged = stream.text()
    progress = [p for p in _log_payloads(stream) if "done" in p]
    assert len(progress) == 1
    payload = progress[0]
    assert payload["pass_number"] == 1
    assert payload["done"] == 1
    assert payload["total"] == 1
    assert "rate_per_second" in payload
    assert "eta_seconds" in payload
    assert payload["renewed"] == 1
    assert "errors" in payload
    assert "elapsed_seconds" in payload
    assert "rss_kb" in payload
    assert "pass complete" in logged
    conn.close()


def test_renew_stop_between_rows_cuts_the_pass_short(tmp_path, monkeypatch):
    """``_stop`` set mid-pass ends the pass after the in-flight heartbeat.

    A fake connection hands back two dict rows; the patched heartbeat counts its
    calls, stops the renewer after the first success and renews. The pass must
    still return the FULL selected total with only the one committed renewal --
    ``(1, 2)`` -- so a stop never trims ``total`` and never renews the rest."""
    clock = FakeClock()
    stream = _CapturingStream()
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id="epoch",
                           lease_seconds=LEASE_SECONDS, stream=stream)
    rows = [{"attempt_id": "a1", "fence": 1}, {"attempt_id": "a2", "fence": 1}]
    called = []

    def counting_heartbeat(conn, attempt_id, fence, *, clock, lease_seconds):
        called.append(attempt_id)
        if len(called) == 1:
            renewer._stop.set()
        return True

    class _TwoRowConnection:
        def execute(self, *args, **kwargs):
            return self

        def fetchone(self):
            return (len(rows),)

        def fetchall(self):
            return list(rows)

    monkeypatch.setattr(supervisor_module, "heartbeat", counting_heartbeat)

    assert renewer.renew_pass(_TwoRowConnection()) == (1, 2)
    assert called == ["a1"]
    assert "pass cut short by stop" in stream.text()


def test_renew_pass_keeps_an_exclusion_recorded_while_the_query_runs(tmp_path, monkeypatch):
    """Verified CodeRabbit race: an exclusion landing mid-``fetchall`` survives.

    ``live_ids`` comes from a query snapshot that cannot know about an
    ``exclude_attempt`` call recorded while the query executed, so the old
    post-query ``intersection_update(live_ids)`` pruned that fresh exclusion
    whenever the queried rows omitted its attempt -- the NEXT pass then handed
    the just-excluded attempt to a heartbeat. The prune is now a difference
    against the exclusions snapshotted under the lock BEFORE the query. A fake
    connection's ``fetchall`` records the exclusion at exactly that moment and
    returns a live row omitting ``"late"``; the pass must renew the live row
    and keep ``"late"`` excluded."""
    clock = FakeClock()
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id="epoch",
                           lease_seconds=LEASE_SECONDS, stream=_CapturingStream())

    class _ExcludingDuringFetchConnection:
        def execute(self, *args, **kwargs):
            return self

        def fetchone(self):
            return (1,)

        def fetchall(self):
            renewer.exclude_attempt("late")
            return [{"attempt_id": "a1", "fence": 1}]

    renewed_ids = []

    def recording_heartbeat(conn, attempt_id, fence, *, clock, lease_seconds):
        renewed_ids.append(attempt_id)
        return True

    monkeypatch.setattr(supervisor_module, "heartbeat", recording_heartbeat)

    assert renewer.renew_pass(_ExcludingDuringFetchConnection()) == (1, 1)
    assert renewed_ids == ["a1"]
    assert "late" in renewer._excluded_attempts


def test_renew_pass_pages_a_257_row_live_set_in_bounded_fetches(tmp_path, monkeypatch):
    """A 257-row live set streams in keyset pages, never materialized whole.

    The paged query carries ``a.attempt_id > ?`` plus ``ORDER BY a.attempt_id``
    and ``LIMIT ?``, and every issued limit and fetched page is bounded by
    ``_RENEWAL_PAGE_ROWS`` (128). A fake connection applies the keyset filter
    and cap itself; a recording heartbeat proves ALL 257 attempts are renewed,
    exactly once each, in attempt-id order, with a short final page ending the
    pass.
    """
    clock = FakeClock()
    stream = _CapturingStream()
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id="epoch",
                           lease_seconds=LEASE_SECONDS, stream=stream)
    rows = [{"attempt_id": f"att{i:05d}", "fence": 1} for i in range(1, 258)]
    issued = []

    class _PagedConnection:
        def execute(self, sql, params=(), **kwargs):
            self.sql, self.params = sql, params
            return self

        def fetchone(self):
            assert self.sql.startswith("SELECT COUNT(")
            return (len(rows),)

        def fetchall(self):
            assert "a.attempt_id > ?" in self.sql and "ORDER BY a.attempt_id" in self.sql
            assert "LIMIT ?" in self.sql
            last_id, limit = self.params[1], self.params[2]
            page = [row for row in rows if row["attempt_id"] > last_id][:limit]
            issued.append((limit, len(page)))
            return page

    renewed_ids = []

    def recording_heartbeat(conn, attempt_id, fence, *, clock, lease_seconds):
        renewed_ids.append(attempt_id)
        return True

    monkeypatch.setattr(supervisor_module, "heartbeat", recording_heartbeat)

    assert renewer.renew_pass(_PagedConnection()) == (257, 257)
    assert supervisor_module._RENEWAL_PAGE_ROWS == 128
    assert renewed_ids == [row["attempt_id"] for row in rows]
    assert issued == [(128, 128), (128, 128), (128, 1)]
    assert all(limit <= 128 and page_rows <= 128 for limit, page_rows in issued)
    complete = next(p for p in _log_payloads(stream) if "skipped" in p)
    assert complete["attempts"] == 257 and complete["renewed"] == 257


# ---------------------------------------------------------------------------
# LeaseRenewer thread lifecycle: a real thread on its own connection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("interval_seconds", [0, -1, float("nan"), float("inf"), float("-inf")])
def test_non_positive_interval_is_rejected(tmp_path, interval_seconds):
    """A zero, negative or non-finite override would spin the wait loop or leave
    it unbounded; the constructor refuses all of them with the exact message."""
    with pytest.raises(ValueError,
                       match=r"^interval_seconds must be a finite positive number$"):
        LeaseRenewer(tmp_path / DB_NAME, clock=FakeClock(), epoch_id="epoch",
                     lease_seconds=LEASE_SECONDS, interval_seconds=interval_seconds)


def test_stop_after_failed_thread_start_does_not_join(tmp_path, monkeypatch):
    """A Thread whose start() raised has no ident; stop() must not join it."""
    conn, clock, sup = catalog(tmp_path)
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS)
    error = RuntimeError("thread start failed")

    def failing_start(self):
        raise error

    monkeypatch.setattr(threading.Thread, "start", failing_start)
    with pytest.raises(RuntimeError) as raised:
        renewer.start()
    conn.close()

    assert raised.value is error
    assert renewer._thread is not None
    renewer.stop()
    assert renewer._thread is None


def test_thread_renews_then_stops_and_joins(tmp_path):
    """Issue #1239 deterministic reproduction: a blocked main-thread tick does not
    starve the independent renewal thread.

    A simulated tick (the serve loop's one thread) blocks on an Event inside one
    long effect -- it renews nothing. With a short ``interval_seconds`` the
    LeaseRenewer runs on its OWN thread and connection; the fake clock advances
    to one second before the first renewal's deadline, and the renewer lands a
    later fenced renewal while the tick is still stuck. The renewed lease then
    survives the old deadline passing, and ``stop()`` joins the thread to
    completion before returning. Nothing sleeps a lease period, so the test
    stays fast: the extra pass is one short real interval.
    """
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="thread")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                 (_stamp(clock, 1), claim.attempt_id))
    stream = _CapturingStream()
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS, interval_seconds=0.02, stream=stream)

    tick_blocked = threading.Event()   # the simulated tick has entered its long effect
    release_tick = threading.Event()   # set only in finally to let the tick finish

    def blocked_tick():
        tick_blocked.set()
        release_tick.wait(timeout=10.0)  # no heartbeat path: the tick is simply stuck

    tick = threading.Thread(target=blocked_tick, name="simulated-tick", daemon=True)
    try:
        tick.start()
        assert tick_blocked.wait(timeout=5.0), "simulated tick never entered its effect"

        renewer.start()
        assert renewer.first_pass.wait(timeout=5.0), "renewal thread never ran a pass"
        after_first = _lease(conn, claim.attempt_id)
        assert after_first["heartbeat_at"] is not None
        deadline_before = after_first["lease_expires_at"]
        renewed_before = renewer.renewed

        # The stall: advance to one second before the first renewal's deadline.
        # The next committed renewal then has to be an off-thread pass at this
        # new time, extending the lease from near expiry.
        clock.advance(LEASE_SECONDS - 1)
        assert deadline_before == _stamp(clock, 1)

        # Deterministic: the renewer's own thread must land another COMMITTED
        # renewal while the tick stays blocked -- ``renewed`` is counted after the
        # fenced heartbeat commits, so waiting on it (not ``passes``, which bumps
        # before the write) reads a durable lease. Poll a real deadline; the short
        # interval is <1s, so this finishes fast.
        deadline = time.monotonic() + 5.0
        while ((renewer.renewed <= renewed_before
                or _lease(conn, claim.attempt_id)["lease_expires_at"] <= deadline_before)
               and time.monotonic() < deadline):
            time.sleep(0.005)
        assert renewer.renewed > renewed_before, \
            "renewal thread did not renew during the blocked tick"

        during = _lease(conn, claim.attempt_id)
        assert during["heartbeat_at"] > after_first["heartbeat_at"]
        assert during["lease_expires_at"] > deadline_before
        assert tick.is_alive() and not release_tick.is_set()  # tick REMAINED blocked

        # The old deadline has now passed; only the off-thread renewal keeps the
        # lease live, so the tick's own expiry sweep must fence nothing.
        clock.advance(2)
        assert deadline_before <= _stamp(clock, 0)
        assert during["lease_expires_at"] > _stamp(clock, 0)
        assert claim.attempt_id not in expire_leases(conn, clock=clock)
    finally:
        release_tick.set()
        tick.join(timeout=5.0)
        thread = renewer._thread
        renewer.stop()
    conn.close()

    assert not tick.is_alive()        # simulated tick released and joined
    assert thread is not None and not thread.is_alive()
    assert renewer._thread is None    # stop() joined to completion, never left running
    assert renewer.renewed >= 2       # this epoch's live attempt renewed each pass
    logged = stream.text()
    assert "lease_renewal" in logged and "thread starting" in logged and "thread stopped" in logged
    pass_complete = next(payload for payload in _log_payloads(stream) if "skipped" in payload)
    assert {"pass_number", "attempts", "renewed", "skipped", "errors",
            "elapsed_seconds", "rss_kb"} <= set(pass_complete)


def test_run_retries_catalog_open_then_renews(tmp_path):
    """#1239 reliability: a failed catalog OPEN is retried, not fatal.

    The renewal thread opens its OWN connection, so a transient failure there is
    a catalog error like any other and must be retried at the next bounded
    interval rather than exiting the thread forever (which would strand this
    epoch's live leases to the expiry sweep). A flaky ``connect`` fails twice
    then hands back a real catalog; the thread must log exactly one bounded
    failure line per failed interval, ask for ``must_exist=True`` (never create a
    missing catalog), reach the renewal loop and renew once connected."""
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="reconnect")
    conn.execute("UPDATE attempts SET lease_expires_at = ? WHERE attempt_id = ?",
                 (_stamp(clock, 1), claim.attempt_id))
    stream = _CapturingStream()
    must_exist_seen = []

    def flaky_connect(db_path, *, must_exist):
        must_exist_seen.append(must_exist)
        if len(must_exist_seen) < 3:
            raise sqlite3.OperationalError("catalog is busy")
        return open_catalog(tmp_path / DB_NAME, clock=clock)

    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS, interval_seconds=0.02,
                           connect=flaky_connect, stream=stream)
    try:
        renewer.start()
        # first_pass fires only after a connection opened AND a renewal pass ran,
        # so by the time it is set the two failed opens have already been counted.
        assert renewer.first_pass.wait(timeout=5.0), "renewal never began after the open retries"
        assert renewer.passes >= 1                 # it entered the renewal loop, not a one-shot exit
        assert renewer.errors == 2                 # the two failed opens, counted not swallowed
        deadline = time.monotonic() + 5.0
        while renewer.renewed < 1 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert renewer.renewed >= 1                # the live attempt is renewed once connected
    finally:
        thread = renewer._thread
        renewer.stop()
    conn.close()

    assert thread is not None and not thread.is_alive()
    assert renewer._thread is None                 # stop() joined the retrying thread to completion
    assert renewer.passes >= 1
    assert all(seen is True for seen in must_exist_seen)  # never opened a catalog that could create one
    logged = stream.text()
    # exactly ONE bounded failure line per failed interval, then it keeps going
    assert logged.count("could not open catalog connection; retried next interval") == 2
    assert "thread running" in logged              # eventually connected and entered the loop


def test_stop_while_awaiting_catalog_reconnect_exits_and_joins(tmp_path):
    """Stop during the connect-retry wait must exit and join, running no renewal.

    ``connect`` always fails, so the thread loops on open->log->wait; ``stop()``
    called while it waits between retries must unblock that wait, let ``_run``
    return through its ``finally`` (no connection to close) and be joined by
    ``stop()`` -- never leaving a retrying thread behind the process lock."""
    conn, clock, sup = catalog(tmp_path)
    enqueue_claim(conn, clock, sup, key="never")
    stream = _CapturingStream()
    attempted = threading.Event()

    def always_fails(db_path, *, must_exist):
        attempted.set()
        raise sqlite3.OperationalError("catalog locked")

    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS, interval_seconds=0.05,
                           connect=always_fails, stream=stream)
    renewer.start()
    assert attempted.wait(timeout=5.0), "renewal thread never attempted the catalog open"
    thread = renewer._thread
    renewer.stop()
    conn.close()

    assert thread is not None and not thread.is_alive()
    assert renewer._thread is None                 # joined to completion, not left retrying
    assert renewer.passes == 0                     # a catalog was never opened, so no renewal ran
    assert not renewer.first_pass.is_set()         # first pass only fires once renewing
    assert "could not open catalog connection; retried next interval" in stream.text()


# ---------------------------------------------------------------------------
# Service wiring: renewer starts after identity, stops-and-joins before release
# ---------------------------------------------------------------------------


def test_start_spawns_renewer_after_identity_and_close_stops_it_before_lock(tmp_path, monkeypatch):
    clock = FakeClock()
    conn = open_catalog(tmp_path / DB_NAME, clock=clock)
    service = Service(conn, tmp_path, _EmptyRegistry(), TEST_POLICY, clock=clock,
                      code_source=tmp_path, store_root=tmp_path)
    expected_db_path = service._catalog_path()  # the exact path start() must hand over
    events = []

    def spy_renewer(db_path, *, clock, epoch_id, lease_seconds, **kw):
        events.append(("renewer.spawn", epoch_id, service.identity is not None, lease_seconds,
                       db_path, clock))
        return _SpyRenewer(events, epoch_id)

    original_release = service.lock.release
    monkeypatch.setattr(supervisor_module, "LeaseRenewer", spy_renewer)
    monkeypatch.setattr(service.lock, "release",
                        lambda: (events.append(("lock.release",)), original_release()))
    try:
        service.start()
        assert service._renewer is not None and service._renewer.started
        service.close()
    finally:
        conn.close()

    names = [e[0] for e in events]
    assert names == ["renewer.spawn", "renewer.start", "renewer.stop", "lock.release"]
    spawn = events[0]
    assert spawn[1] == service.identity.epoch_id  # bound to THIS epoch
    assert spawn[2] is True                        # spawned only after identity init
    assert spawn[3] == LEASE_SECONDS               # production cadence and lease are pinned
    assert spawn[4] == expected_db_path            # the exact path from service._catalog_path()
    assert spawn[5] is clock                       # the exact clock object is forwarded
    default_renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock,
                                   epoch_id=spawn[1], lease_seconds=LEASE_SECONDS)
    assert default_renewer.interval == LEASE_SECONDS / 4


class _EmptyRegistry:
    def names(self):
        return ()

    def get(self, kind):  # pragma: no cover - never reached by these tests
        raise AssertionError("registry unused here")


class _SpyRenewer:
    def __init__(self, events, epoch_id):
        self.events = events
        self.epoch_id = epoch_id
        self.started = False

    def start(self):
        self.started = True
        self.events.append(("renewer.start", self.epoch_id))

    def stop(self):
        self.events.append(("renewer.stop", self.epoch_id))


# ---------------------------------------------------------------------------
# _clock_check: a long monotonic tick pause renews the verified-live workers
# ---------------------------------------------------------------------------


def test_clock_check_renews_live_workers_before_expiry_sweep_but_not_dead(tmp_path, monkeypatch):
    clock = FakeClock()
    conn = open_catalog(tmp_path / DB_NAME, clock=clock)
    sup = Supervisor(begin_epoch(conn, clock=clock, boot_id=EPOCH_BOOT, pid=os.getpid()),
                     EPOCH_BOOT)
    live = enqueue_claim(conn, clock, sup, key="live")
    dead = enqueue_claim(conn, clock, sup, key="dead")

    service = Service(conn, tmp_path, _EmptyRegistry(), TEST_POLICY, clock=clock,
                      code_source=tmp_path, store_root=tmp_path)
    service.identity = sup
    service.running = {
        live.attempt_id: _Running(live.attempt_id, live.fence, "LIVE"),
        dead.attempt_id: _Running(dead.attempt_id, dead.fence, "DEAD"),
    }
    monkeypatch.setattr(supervisor_module, "is_alive", lambda identity, boot: identity == "LIVE")

    # No wall-clock jump (both clocks advance together, so the old jump>60 guard
    # stays False): ONLY the monotonic-pause detection can fire here.
    clock.advance(LEASE_SECONDS + 30)
    service._clock_check()
    fenced = expire_leases(conn, clock=clock)
    conn.close()

    assert dead.attempt_id in fenced       # a gone worker follows the fail-closed sweep
    assert live.attempt_id not in fenced   # a proven-live worker was renewed first


def test_clock_check_exactly_lease_seconds_monotonic_pause_does_not_resume(tmp_path, monkeypatch):
    """The monotonic-pause trigger is STRICTLY greater than one lease period.

    A tick pause of exactly ``LEASE_SECONDS`` (wall and monotonic advancing
    together, so the wall/monotonic discrepancy stays 0) is the boundary BELOW
    the trigger: ``_resume_after_jump`` must not be reached."""
    conn, clock, sup = catalog(tmp_path)
    service = Service(conn, tmp_path, _EmptyRegistry(), TEST_POLICY, clock=clock,
                      code_source=tmp_path, store_root=tmp_path)
    service.identity = sup
    resumed = []
    monkeypatch.setattr(service, "_resume_after_jump", lambda: resumed.append(True))

    clock.advance(LEASE_SECONDS)
    service._clock_check()

    assert resumed == []
    conn.close()


def test_resume_after_jump_emits_one_summary_and_contains_renewal_error(tmp_path, monkeypatch,
                                                                       capsys):
    """One ``lease_resume`` line covers all four outcomes; a raise is one error.

    Four tracked attempts: renewed, refused, one whose renewal RAISES, and one
    whose identity is gone. The raise must be contained per row (counted, never
    propagating), the not-live row must not be renewed at all, and the pass must
    emit exactly one stderr JSON summary -- never one line per attempt."""
    conn, clock, sup = catalog(tmp_path)
    service = Service(conn, tmp_path, _EmptyRegistry(), TEST_POLICY, clock=clock,
                      code_source=tmp_path, store_root=tmp_path)
    service.identity = sup
    service.running = {
        "a-ok": _Running("a-ok", 1, "LIVE_OK"),
        "a-refused": _Running("a-refused", 1, "LIVE_REFUSED"),
        "a-error": _Running("a-error", 1, "LIVE_ERROR"),
        "a-dead": _Running("a-dead", 1, "DEAD"),
    }
    monkeypatch.setattr(supervisor_module, "is_alive",
                        lambda identity, boot: identity != "DEAD")

    def renewing(conn, attempt_id, fence, *, clock, lease_seconds):
        if attempt_id == "a-error":
            raise sqlite3.OperationalError("database is locked")
        return attempt_id == "a-ok"

    monkeypatch.setattr(supervisor_module, "renew_after_resume", renewing)
    service._resume_after_jump()  # returns normally: the raise is contained

    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["event"] == "lease_resume"
    assert (payload["tracked"], payload["renewed"], payload["refused"],
            payload["not_live"], payload["errors"]) == (4, 1, 1, 1, 1)
    assert "rss_kb" in payload
    conn.close()


# ---------------------------------------------------------------------------
# Service._strand: exclusion prevents renewal, expiry holds reservations
# ---------------------------------------------------------------------------


def test_strand_excludes_attempt_from_renewal_and_holds_reservation_until_reconcile(tmp_path, monkeypatch):
    """A fresh live claim passed through ``Service._strand`` must not be renewed;
    expiry fences/bumps the job fence and holds the reservation, while only
    reconcile proving process gone releases it."""
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="strand")

    initial_lease = _lease(conn, claim.attempt_id)["lease_expires_at"]
    initial_fence = conn.execute("SELECT fence FROM jobs WHERE job_id = ?",
                                 (claim.job_id,)).fetchone()["fence"]

    service = Service(conn, tmp_path, _EmptyRegistry(), TEST_POLICY, clock=clock,
                      code_source=tmp_path, store_root=tmp_path)
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS)
    service._renewer = renewer

    service._strand(claim, make_problem("INPUT_CHANGED", "test strand"))

    # Move the fake clock forward before the pass: an erroneous renewal of the
    # stranded attempt would then write a DIFFERENT lease_expires_at, so the
    # unchanged-lease assertion below can only hold if the pass skipped it.
    clock.advance(1)

    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        renewer.renew_pass(own)
    finally:
        own.close()

    assert _lease(conn, claim.attempt_id)["lease_expires_at"] == initial_lease

    clock.advance(LEASE_SECONDS + 1)
    fenced = expire_leases(conn, clock=clock)
    assert claim.attempt_id in fenced

    attempt = conn.execute("SELECT state FROM attempts WHERE attempt_id = ?",
                           (claim.attempt_id,)).fetchone()
    assert attempt["state"] == "recovery_pending"
    job = conn.execute("SELECT fence FROM jobs WHERE job_id = ?",
                       (claim.job_id,)).fetchone()
    assert job["fence"] > initial_fence

    reservation = conn.execute("SELECT released_at FROM resource_reservations "
                               "WHERE attempt_id = ?", (claim.attempt_id,)).fetchone()
    assert reservation is not None
    assert reservation["released_at"] is None

    monkeypatch.setattr(supervisor_module, "prove_ownership_gone",
                        lambda *args, **kwargs: recovery.OwnershipProof(True, (), (), ()))
    service.identity = sup
    service.boot = "boot"
    service.reconcile()

    released = conn.execute("SELECT released_at FROM resource_reservations "
                            "WHERE attempt_id = ?", (claim.attempt_id,)).fetchone()
    assert released["released_at"] is not None

    conn.close()


def test_renew_pass_retires_an_exclusion_once_the_attempt_leaves_the_live_set(tmp_path):
    """A stranded exclusion is retired only after recovery fences its attempt.

    While the excluded attempt is still a live, current-fence row it appears in
    ``_LIVE_EPOCH_ATTEMPT_PAGE``, so the exhausted-pass prune KEEPS the exclusion and the
    pass still skips the heartbeat. Once ``expire_leases`` moves it to
    ``recovery_pending`` and bumps the job fence it leaves the live rows, and the
    next pass drops the now-unusable id instead of holding it forever."""
    conn, clock, sup = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, sup, key="retire")

    service = Service(conn, tmp_path, _EmptyRegistry(), TEST_POLICY, clock=clock,
                      code_source=tmp_path, store_root=tmp_path)
    renewer = LeaseRenewer(tmp_path / DB_NAME, clock=clock, epoch_id=sup.epoch_id,
                           lease_seconds=LEASE_SECONDS)
    service._renewer = renewer
    service._strand(claim, make_problem("INPUT_CHANGED", "test strand retire"))
    assert claim.attempt_id in renewer._excluded_attempts

    # Still eligible (live row): the exclusion survives the prune and is honored.
    clock.advance(1)
    initial_lease = _lease(conn, claim.attempt_id)["lease_expires_at"]
    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        assert renewer.renew_pass(own) == (0, 1)
    finally:
        own.close()
    assert claim.attempt_id in renewer._excluded_attempts
    assert _lease(conn, claim.attempt_id)["lease_expires_at"] == initial_lease

    # No longer eligible (fenced): the next pass prunes the id.
    clock.advance(LEASE_SECONDS + 1)
    assert claim.attempt_id in expire_leases(conn, clock=clock)
    own = open_catalog(tmp_path / DB_NAME, clock=clock)
    try:
        assert renewer.renew_pass(own) == (0, 0)
    finally:
        own.close()
        conn.close()

    assert claim.attempt_id not in renewer._excluded_attempts
