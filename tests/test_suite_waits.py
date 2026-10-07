"""The suite's own guards against sleeping forever (see ``tests/README.md``,
"Running the whole suite on this box").

* ``tests.ops_support.AdmissionWatch`` / ``run_until``: a real-``Service`` test
  whose job the host cannot admit fails with ``RESOURCE WAIT`` and the queue
  reason's numbers, instead of polling until its deadline.
* ``tests/conftest.py``'s per-test alarm: a test that runs past its budget
  fails with its node id, and the run continues.

Each guard gets a negative control: waits that are NOT environmental (a
reason from the test's own catalog, a shortage that clears) must not fail.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from engine.v2.contracts import ProcessIdentity, ProgressEvent, ResolvedResources
from engine.v2.ops import executor_watchdog
from engine.v2.ops.catalog import dumps
from engine.v2.ops.lifecycle import record_progress
from engine.v2.ops.recovery import read_boot_id
from engine.v2.ops.submission import submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import (
    POLICY,
    REGISTRY,
    TEST_POLICY,
    AdmissionWatch,
    catalog,
    request,
    run_until,
    RunUntilTimeout,
)

REPO = Path(__file__).resolve().parents[1]
GIB = 1 << 30


def _queued_job(tmp_path, reason: dict | None, **changes):
    conn, clock, _ = catalog(tmp_path)
    job_id = submit(conn, REGISTRY, POLICY, request("wait", **changes), clock=clock).job_id
    conn.execute("UPDATE jobs SET queue_reason_json=? WHERE job_id=?",
                 (None if reason is None else json.dumps(reason), job_id))
    return conn, job_id


def test_profile_exceeds_capacity_fails_on_the_first_check(tmp_path):
    conn, job_id = _queued_job(tmp_path, {
        "code": "PROFILE_EXCEEDS_CAPACITY", "needed": {"memory_bytes": 256 << 20, "cpus": 5},
        "available": {"capacity_bytes": 6 * GIB, "worker_cpus": 3}},
        resource_class="legacy_score")
    with pytest.raises(pytest.fail.Exception) as excinfo:
        AdmissionWatch(conn, job_id, wait_seconds=3600).check()
    message = str(excinfo.value)
    assert message.startswith("RESOURCE WAIT")
    assert "profile legacy_score" in message and "needs 5 worker CPUs" in message
    assert "at least 6 CPUs" in message


def test_run_until_fails_at_its_explicit_budget_on_an_own_catalog_heavy_slot_wait(tmp_path, monkeypatch):
    """Regression (corpus timeout): a ``HEAVY_SLOT`` wait comes from the
    test's own catalog, so it is never environmental -- ``run_until`` must
    fall through to the caller's explicit deadline and raise
    ``RunUntilTimeout`` at exactly that budget, not fail ``RESOURCE WAIT``
    or overrun. A controlled fake clock (``monotonic()`` reads a mutable
    timestamp, ``sleep`` advances it) pins the budget down; the deadline
    diagnostics are the separate
    ``test_run_until_fails_at_its_deadline_with_catalog_diagnostics``'s job."""
    conn, job_id = _queued_job(tmp_path, {"code": "HEAVY_SLOT"})

    class _NeverAdmits:
        def tick(self):
            pass

    clock = {"t": 0.0}

    class _FakeTime:
        def monotonic(self):
            return clock["t"]

        def sleep(self, seconds):
            clock["t"] += seconds

    monkeypatch.setattr("tests.ops_support.time", _FakeTime())
    timeout = 0.2
    with pytest.raises(RunUntilTimeout):
        run_until(_NeverAdmits(), conn, job_id, timeout=timeout, poll=0.01)
    # the fake clock lands on the caller's deadline: no retry, no extension
    assert clock["t"] == pytest.approx(timeout)
    assert clock["t"] <= timeout


def test_run_until_deadline_wins_when_host_wait_window_expires(tmp_path, monkeypatch):
    """The admission window clamped to the caller's deadline, on its exact
    edge: ``MEMORY_HEADROOM`` first sampled at fake t=0.25 with a 0.75 s
    window would expire at t=1.0 -- the caller's deadline too -- so the
    watch must go silent there and let ``RunUntilTimeout`` speak, its
    diagnostics carrying the queue reason and no ``RESOURCE WAIT``. The fake
    clock (``monotonic()`` reads mutable time, ``sleep`` advances it) makes
    the edge deterministic: no real wait."""
    conn, job_id = _queued_job(tmp_path, {
        "code": "MEMORY_HEADROOM",
        "needed": {"memory_bytes": 256 << 20, "owed_unconsumed_bytes": 0},
        "available": {"headroom_bytes": 100 << 20}})

    clock = {"t": 0.0}

    class _FakeTime:
        def monotonic(self):
            return clock["t"]

        def sleep(self, seconds):
            clock["t"] += seconds

    class _AdvancingTwice:
        def __init__(self):
            self._advances = [0.25, 0.5]

        def tick(self):
            if self._advances:
                clock["t"] += self._advances.pop(0)

    monkeypatch.setattr("tests.ops_support.time", _FakeTime())
    monkeypatch.setattr("tests.ops_support.ADMISSION_WAIT_SECONDS", 0.75)
    with pytest.raises(RunUntilTimeout) as excinfo:
        run_until(_AdvancingTwice(), conn, job_id, timeout=1.0, poll=0.25)
    message = str(excinfo.value)
    assert "MEMORY_HEADROOM" in message
    assert "RESOURCE WAIT" not in message
    assert clock["t"] == 1.0


def test_memory_headroom_waits_out_the_window_then_fails_with_the_numbers(tmp_path):
    conn, job_id = _queued_job(tmp_path, {
        "code": "MEMORY_HEADROOM",
        "needed": {"memory_bytes": 256 << 20, "owed_unconsumed_bytes": 0},
        "available": {"headroom_bytes": 100 << 20}})
    watch = AdmissionWatch(conn, job_id, wait_seconds=0.3)
    watch.check()  # a transient shortage still just waits
    time.sleep(0.35)
    with pytest.raises(pytest.fail.Exception) as excinfo:
        watch.check()
    message = str(excinfo.value)
    assert "MEMORY_HEADROOM" in message and "0.75 GiB free" in message  # 0.25 + 0.5 margin


def test_memory_shortage_that_clears_resets_the_window(tmp_path):
    conn, job_id = _queued_job(tmp_path, {"code": "MEMORY_HEADROOM", "needed": {},
                                          "available": {}})
    watch = AdmissionWatch(conn, job_id, wait_seconds=0.3)
    watch.check()
    conn.execute("UPDATE jobs SET queue_reason_json=NULL WHERE job_id=?", (job_id,))
    time.sleep(0.35)
    watch.check()  # admitted meanwhile (no reason): the window resets
    conn.execute("UPDATE jobs SET queue_reason_json=? WHERE job_id=?",
                 (json.dumps({"code": "MEMORY_HEADROOM"}), job_id))
    watch.check()  # a fresh window, not 0.35 s old


@pytest.mark.parametrize("code", ["HEAVY_SLOT", "HEAVY_SLOT_HELD", "STORE_LEASE_HELD"])
def test_waits_on_the_tests_own_catalog_never_fail_even_at_the_deadline(tmp_path, code):
    conn, job_id = _queued_job(tmp_path, {"code": code})
    AdmissionWatch(conn, job_id, wait_seconds=0).check(final=True)


def test_run_until_fails_fast_when_cpu_affinity_cannot_hold_the_profile(tmp_path):
    """The real mechanism behind the corpus-parity stall: a real ``Service``
    sampling a narrowed CPU affinity (as ``taskset -c 0-3`` / ``bounded_run
    --cores 4`` leave it) can never admit a 5-CPU profile. The job never
    launches, so the ``tiny`` worker needs no implementation."""
    if not hasattr(os, "sched_setaffinity"):
        pytest.skip("no CPU affinity control on this platform")
    original = os.sched_getaffinity(0)
    conn, clock, _ = catalog(tmp_path)
    job_id = submit(conn, REGISTRY, POLICY, request("cpu", resource_class="legacy_score"),
                    clock=clock).job_id
    service = Service(conn, tmp_path, REGISTRY, TEST_POLICY, clock=clock, code_source=REPO)
    started = time.monotonic()
    try:
        os.sched_setaffinity(0, set(sorted(original)[:2]))
        service.start()
        with pytest.raises(pytest.fail.Exception) as excinfo:
            run_until(service, conn, job_id, timeout=60)
    finally:
        os.sched_setaffinity(0, original)
        service.close()
    assert time.monotonic() - started < 10
    assert "PROFILE_EXCEEDS_CAPACITY" in str(excinfo.value)


def test_run_until_fails_at_its_deadline_with_catalog_diagnostics(tmp_path, monkeypatch):
    """A job that simply never completes on a non-environmental reason must
    fail with an AssertionError quoting the ops query helpers (get_job,
    attempt_receipts), not be returned to the caller as ``assert 'queued'
    == 'succeeded'`` (the #363/#358/#345 shape). A tiny deadline against a
    never-ticking fake service: no real wait. The second half proves a
    failing helper only ``n/a``s its own field: every required label stays
    present and the other diagnostics still print their real values. The
    third half proves a missing-attribute receipt cannot collapse several
    fields to blank: a bare ``get_job`` object and a malformed
    ``attempt_receipts`` item still print every required label, each value
    field ``n/a`` on its own, while the job id and state classification
    remain. The fourth half proves an unprintable REQUIRED value
    (``attempt_count`` whose ``__str__`` raises) fails only its own field: the
    other diagnostics keep their real values, with no whole-message
    ``diagnostics failed`` fallback."""
    conn, job_id = _queued_job(tmp_path, {"code": "DEPENDENCY_PENDING"})

    class _NeverAdmits:
        def tick(self):
            pass

    started = time.monotonic()
    with pytest.raises(AssertionError) as excinfo:
        run_until(_NeverAdmits(), conn, job_id, timeout=0.2)
    assert time.monotonic() - started < 10
    message = str(excinfo.value)
    assert job_id in message
    assert "'queued'" in message and "never admitted" in message
    assert "DEPENDENCY_PENDING" in message
    assert "attempt count: 0" in message
    # absent fields: each names itself, and the two heartbeats never blur
    assert "attempt lease heartbeat: unavailable (no attempts recorded)" in message
    assert "latest progress event: unavailable (no progress event recorded)" in message
    assert ("latest non-heartbeat progress event: unavailable "
            "(no non-heartbeat progress event recorded)") in message
    assert "process family liveness: 0 live / 0 tracked (diagnostic only)" in message
    assert "log tail: unavailable (no progress recorded)" in message

    def _broken_attempt_receipts(*args, **kwargs):
        raise RuntimeError("simulated helper failure")

    monkeypatch.setattr("tests.ops_support.attempt_receipts", _broken_attempt_receipts)
    with pytest.raises(AssertionError) as excinfo:
        run_until(_NeverAdmits(), conn, job_id, timeout=0.2)
    message = str(excinfo.value)
    assert "attempt count: 0" in message            # other diagnostics survive
    assert "DEPENDENCY_PENDING" in message
    assert "log tail: unavailable (no progress recorded)" in message
    assert job_id in message
    assert "'queued'" in message and "never admitted" in message
    # a failing attempt_receipts n/a's ONLY the lease-heartbeat field: process
    # family liveness is rendered independently and keeps its real value
    assert ("attempt lease heartbeat: unavailable (attempt_receipts failed: RuntimeError)"
            in message)
    assert "process family liveness: 0 live / 0 tracked (diagnostic only)" in message

    class _Blank:  # a receipt that carries none of the attributes the helper reads
        pass

    monkeypatch.setattr("tests.ops_support.get_job", lambda *a, **k: _Blank())
    monkeypatch.setattr("tests.ops_support.attempt_receipts", lambda *a, **k: [_Blank()])
    with pytest.raises(AssertionError) as excinfo:
        run_until(_NeverAdmits(), conn, job_id, timeout=0.2)
    message = str(excinfo.value)
    # a bare get_job + a malformed attempt item blank only their OWN field; every
    # required label is still printed, and the job id / state classification --
    # formatted before the helpers -- never vanish.
    assert job_id in message
    assert "'queued'" in message and "never admitted" in message
    for label in ("attempt count: unavailable (attempt count failed: AttributeError)",
                  "queue/admission reason: unavailable "
                  "(queue/admission reason failed: AttributeError)",
                  "attempt lease heartbeat: unavailable "
                  "(attempt lease heartbeat failed: AttributeError)",
                  "latest progress event: unavailable "
                  "(latest progress event failed: AttributeError)",
                  "latest non-heartbeat progress event: unavailable "
                  "(no non-heartbeat progress event recorded)",
                  "log tail: unavailable (log tail failed: AttributeError)"):
        assert label in message, message
    assert "process family liveness: 0 live / 0 tracked (diagnostic only)" in message

    class _RaisingStr:  # a REQUIRED value whose __str__ raises
        def __str__(self):
            raise ValueError("simulated __str__ failure")

    class _Reason:
        code = "DEPENDENCY_PENDING"
        needed = {"dep": "job-x"}
        available = None

    class _Progress:
        kind = "tick"
        recorded_at = "2026-09-12T00:00:00+00:00"
        message = "still queued"

    class _Attempt:
        attempt_number = 1
        state = "queued"
        heartbeat_at = "2026-09-12T00:00:05+00:00"
        lease_expires_at = "2026-09-12T00:00:35+00:00"

    class _Receipt:  # unprintable attempt_count; every other required value present
        attempt_count = _RaisingStr()
        queue_reason = _Reason()
        latest_progress = _Progress()

    monkeypatch.setattr("tests.ops_support.get_job", lambda *a, **k: _Receipt())
    monkeypatch.setattr("tests.ops_support.attempt_receipts", lambda *a, **k: [_Attempt()])
    with pytest.raises(AssertionError) as excinfo:
        run_until(_NeverAdmits(), conn, job_id, timeout=0.2)
    message = str(excinfo.value)
    # the guarded str() in field() confines the __str__ failure to attempt count:
    # no whole-message fallback, every other required label keeps its real
    # value, and the job id / state classification never vanish.
    assert job_id in message
    assert "'queued'" in message and "never admitted" in message
    assert "diagnostics failed" not in message
    assert "attempt count: unavailable (attempt count failed: ValueError)" in message
    assert "queue/admission reason: DEPENDENCY_PENDING" in message
    assert "log tail: tick at" in message
    assert ("attempt lease heartbeat: attempt 1 (queued): heartbeat at "
            "2026-09-12T00:00:05+00:00, lease expires at 2026-09-12T00:00:35+00:00") in message
    assert "latest progress event: tick at 2026-09-12T00:00:00+00:00" in message
    assert "process family liveness: 0 live / 0 tracked (diagnostic only)" in message

    # The final-poll race CodeRabbit flagged on #377: a job that completes
    # DURING the last sleep -- i.e. after the deadline already elapsed -- is a
    # success, not a timeout. Isolated fake clock and job-state reader, so there
    # is no real wait: the reader reports ``queued`` at the start and in the loop
    # body, then flips to ``succeeded`` only once the fake sleep has advanced the
    # clock past the tiny deadline, and ``run_until`` returns it.
    clock = {"t": 0.0}
    fake_deadline = 0.2

    class _FakeTime:
        def monotonic(self):
            return clock["t"]

        def sleep(self, seconds):
            clock["t"] += fake_deadline * 2  # one step clears the deadline

    def _final_poll_state(conn_, job_id_):
        return "succeeded" if clock["t"] >= fake_deadline else "queued"

    monkeypatch.setattr("tests.ops_support.time", _FakeTime())
    monkeypatch.setattr("tests.ops_support.job_state", _final_poll_state)
    assert run_until(_NeverAdmits(), conn, job_id, timeout=fake_deadline) == "succeeded"


def _controlled_stat_line(pid: int, *, start_ticks: int) -> str:
    """A syntactically real ``/proc/<pid>/stat`` row for the controlled process
    table: live state ``S``, this pid, this recorded start time, and a process
    group / session that match the ``ProcessIdentity`` the test writes (111).
    Local (not imported from another test module) so this suite adds no
    cross-test import edge; the field positions mirror what
    ``executor_watchdog.process_info`` reads after ``rfind(")")``."""
    fields = ["S", "1", "111", "111", "0", "-1", "0", "0", "0", "0", "0",
              "0", "0", "0", "0", "0", "0", "1", "0", str(start_ticks), "0", "10"]
    return f"{pid} (fake) " + " ".join(fields)


def _install_controlled_process_table(monkeypatch, tmp_path, pid: int, start_ticks: int):
    """Point the seam ``diagnostics.process_family_liveness`` walks
    (``executor_watchdog.process_table``, the default ``table=`` argument of
    the existing ``observe`` ownership proof) at a synthetic ``/proc`` tree
    holding exactly one row: this pid, this recorded start time, live (state
    ``S``). Liveness is then proved from a fixture instead of the host -- a
    positive ``live`` count without an unrelated process, and a dead family
    without a pid the kernel could never hand out -- and no production
    identity mechanism is added for the test's benefit."""
    proc = tmp_path / "proc"
    entry = proc / str(pid)
    entry.mkdir(parents=True)
    (entry / "stat").write_text(_controlled_stat_line(pid, start_ticks=start_ticks))
    real = executor_watchdog.process_table
    monkeypatch.setattr(executor_watchdog, "process_table",
                        lambda boot_id: real(boot_id, proc=proc))


def test_run_until_deadline_reports_the_live_tracked_family_and_stderr_tail(tmp_path):
    """Regression: the ``run_until`` deadline diagnostics must describe the
    tracked process family from the real ``/proc`` tree and surface the
    worker's stderr, not only catalog rows. A real long-lived parent Python
    subprocess owns one long-lived CPU-spinning child; the parent pid is
    registered for the job with its real /proc identity (start ticks and
    process group), and a ``worker.stderr`` sits at the attempt staging path
    beside the temporary catalog. A never-ticking fake service keeps the job
    nonterminal until ``run_until`` times out, and the timeout message must
    then carry the stderr tail, the tracked pid, a CPU tick delta, the
    process state, the wait channel and the child pid. The stderr tail
    surfaces the traceback frame and exception line while the frame's
    absolute path, the source line and the secret text stay redacted, and a
     recorded start ticks that no longer matches the live process reports the
     pid identity unavailable with no cpu delta for that pid. The same live
     pid is also recorded under an older superseded attempt whose start ticks
     differ by one, so the message must render two distinct tracked-identity
     entries for that pid -- the stale one unavailable and never signaled,
     the live one with the traceback signal sent. Only the fake
    service's tick is stubbed; the stderr tail, the tracked-process
    diagnostics, /proc and ``run_until`` itself stay real. The parent and
    its child are terminated and reaped in ``finally`` even when an
    assertion fails, the child killed by pid independently of the parent's
    exit timing."""
    conn, job_id = _queued_job(tmp_path, {"code": "DEPENDENCY_PENDING"})
    boot_id = read_boot_id()
    epoch = conn.execute("SELECT epoch_id FROM supervisor_epochs").fetchone()[0]
    parent = None
    child_pid = None
    try:
        parent = subprocess.Popen(
            [sys.executable, "-c",
             "import signal, subprocess, sys\n"
             "child = subprocess.Popen([sys.executable, '-c', 'while True: pass'])\n"
             "signal.signal(signal.SIGTERM,\n"
             "              lambda *_: (child.kill(), child.wait(), sys.exit(1)))\n"
             "print(child.pid, flush=True)\n"
             "child.wait()\n"],
            stdout=subprocess.PIPE, text=True, start_new_session=True)
        child_pid = int(parent.stdout.readline())
        live_stat = Path(f"/proc/{parent.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        identity = ProcessIdentity(boot_id=boot_id, pid=parent.pid,
                                   start_ticks=int(live_stat[19]),
                                   process_group=int(live_stat[2]))
        stale_identity = ProcessIdentity(boot_id=boot_id, pid=identity.pid,
                                         start_ticks=identity.start_ticks + 1,
                                         process_group=identity.process_group)
        resources = ResolvedResources(
            effective_host_budget_bytes=1 << 30, reserved_memory_bytes=1 << 29,
            assigned_cpu_ids=(0,), thread_count=1, scratch_limit_bytes=1 << 28,
            executor_mode="fake", containment="none", provider_leases=(),
            resource_profile_version="test")
        conn.execute(
            "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, "
            "supervisor_epoch, host_boot_id, state, process_state, process_json, "
            "resources_json, created_at, heartbeat_at, lease_expires_at) "
            "VALUES (?, ?, 1, 1, ?, ?, 'failed', 'exited', ?, ?, ?, ?, ?)",
            ("att_stale", job_id, epoch, boot_id, dumps(stale_identity),
             dumps(resources), "2026-09-12T00:00:00+00:00",
             "2026-09-12T00:00:05+00:00", "2026-09-12T00:00:35+00:00"))
        conn.execute("INSERT INTO process_members (attempt_id, pid, start_ticks, "
                     "identity_json) VALUES (?, ?, ?, ?)",
                     ("att_stale", stale_identity.pid, stale_identity.start_ticks,
                      dumps(stale_identity)))
        conn.execute(
            "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, "
            "supervisor_epoch, host_boot_id, state, process_state, process_json, "
            "resources_json, created_at, heartbeat_at, lease_expires_at) "
            "VALUES (?, ?, 2, 2, ?, ?, 'failed', 'exited', ?, ?, ?, ?, ?)",
            ("att_live", job_id, epoch, boot_id, dumps(identity), dumps(resources),
             "2026-09-12T00:00:00+00:00", "2026-09-12T00:00:05+00:00",
             "2026-09-12T00:00:35+00:00"))
        conn.execute("INSERT INTO process_members (attempt_id, pid, start_ticks, "
                     "identity_json) VALUES (?, ?, ?, ?)",
                     ("att_live", identity.pid, identity.start_ticks, dumps(identity)))
        conn.execute("UPDATE jobs SET attempt_count = 2 WHERE job_id = ?", (job_id,))
        stderr_path = (tmp_path / "attempts" / "att_live" / "staging" /
                       "diagnostics" / "worker.stderr")
        stderr_path.parent.mkdir(parents=True)
        fake_secret = "fake-credential-7f3d9a2b"
        fake_path = "/opt/fake-secret-stage/worker_impl.py"
        fake_source_line = f'    return load_credential("{fake_secret}")'
        stderr_path.write_text(
            "Traceback (most recent call last):\n"
            f'  File "{fake_path}", line 9, in ramp\n'
            f"{fake_source_line}\n"
            f"ValueError: credential {fake_secret} rejected by stage ramp\n")

        class _NeverTicks:
            def tick(self):
                pass

        diagnostic_started = time.monotonic()
        with pytest.raises(AssertionError) as excinfo:
            run_until(_NeverTicks(), conn, job_id, timeout=0.2)
        diagnostic_elapsed = time.monotonic() - diagnostic_started
        message = str(excinfo.value)
        assert diagnostic_elapsed >= 1.0
        # stderr privacy: the traceback frame and the exception line surface
        # with the path and message redacted; the absolute path, the source
        # line and the secret text do not
        assert 'File "<path>"' in message
        assert "ValueError: <message redacted>" in message
        assert fake_path not in message
        assert fake_source_line not in message
        assert fake_secret not in message
        assert str(parent.pid) in message
        assert str(child_pid) in message
        assert "cpu " in message and " ticks" in message
        state = re.search(r"\bstate ([A-Z]+)\b", message)
        assert state is not None and state.group(1) in ("S", "D"), message
        assert "wchan " in message
        assert "children " in message
        assert "'queued'" in message and "never admitted" in message

        # reused-pid identities: the same live pid recorded under the older
        # superseded attempt and the matching live attempt renders two distinct
        # tracked entries -- the stale start-ticks identity stays unavailable
        # and is never signaled, the matching live identity carries the
        # traceback signal, so the worker was signaled only through the
        # identity that matches the live process
        tracked = [part for part in message.split("; ")
                   if part.startswith("tracked process ")]
        assert len(tracked) == 2, message
        stale_entries = [part for part in tracked
                         if "unavailable (stat unreadable or start time mismatch)" in part]
        live_entries = [part for part in tracked if "traceback signal sent" in part]
        assert len(stale_entries) == 1 and len(live_entries) == 1, message
        assert f"(pid {identity.pid})" in stale_entries[0]
        assert f"(pid {identity.pid})" in live_entries[0]
        parent.wait(timeout=10)
        assert parent.returncode == -signal.SIGABRT, str(parent.returncode)

        # pid identity: a recorded start_ticks that no longer matches the live
        # process reports the pid identity unavailable, with no cpu delta for
        # that pid
        conn.execute(
            "UPDATE process_members SET start_ticks=? WHERE attempt_id=? AND pid=?",
            (identity.start_ticks + 1, "att_live", identity.pid))
        with pytest.raises(AssertionError) as excinfo:
            run_until(_NeverTicks(), conn, job_id, timeout=0.2)
        message2 = str(excinfo.value)
        tracked2 = [part for part in message2.split("; ")
                    if part.startswith("tracked process diagnostics:")]
        assert len(tracked2) == 1
        assert "unavailable" in tracked2[0]
        assert "cpu " not in message2 and " ticks" not in message2

        # pid identity: a recorded host boot id that differs from the live
        # boot reports the pid identity unavailable with no cpu delta
        conn.execute("UPDATE attempts SET host_boot_id=? WHERE attempt_id IN (?, ?)",
                     ("stale-test-boot-id", "att_live", "att_stale"))
        with pytest.raises(AssertionError) as excinfo:
            run_until(_NeverTicks(), conn, job_id, timeout=0.2)
        message3 = str(excinfo.value)
        tracked3 = [part for part in message3.split("; ")
                    if part.startswith("tracked process diagnostics:")]
        assert len(tracked3) == 1
        assert "unavailable" in tracked3[0]
        assert "cpu " not in message3 and " ticks" not in message3
    finally:
        if parent is not None:
            if parent.poll() is None:
                parent.terminate()
            try:
                parent.wait(timeout=10)
            except subprocess.TimeoutExpired:
                parent.kill()
                try:
                    parent.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if parent.stdout is not None:
                parent.stdout.close()
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_run_until_deadline_signals_a_stuck_tracked_child_and_dumps_its_faulthandler_tail(tmp_path):
    """Regression: the deadline diagnostics must not only track a live
    family but actually request and surface a real all-thread traceback from
    a stalled tracked Python child. The child enables ``faulthandler``, then
    blocks forever on a ``threading.Lock`` held by another thread (a futex
    wait, state ``S``) with its stderr redirected to the attempt's real
    staging ``diagnostics/worker.stderr``, created before launch. Its live
    pid is registered with the real boot id, start ticks and process group,
    the matching attempts and process_members rows are inserted, and a
    never-ticking fake service keeps the job nonterminal until ``run_until``
    times out. The message must then report ``traceback signal sent`` and a
    worker stderr tail carrying the sanitized faulthandler dump -- a
    ``Current thread`` heading and a ``File "<path>", line N`` frame with the
    temporary source path absent -- and the child must actually be
    terminated by that diagnostic SIGABRT, reaped in ``finally`` even when an
    assertion fails. Only the fake service's tick is inert; /proc, the
    stderr read, the signal path and ``run_until`` itself stay real."""
    conn, job_id = _queued_job(tmp_path, {"code": "DEPENDENCY_PENDING"})
    boot_id = read_boot_id()
    epoch = conn.execute("SELECT epoch_id FROM supervisor_epochs").fetchone()[0]
    attempt_id = "att_fh"
    child_path = tmp_path / "fh_child_worker.py"
    child_path.write_text(
        "import faulthandler, threading\n"
        "faulthandler.enable()\n"
        "lock = threading.Lock()\n"
        "holder_ready = threading.Event()\n"
        "\n"
        "def _hold():\n"
        "    lock.acquire()\n"
        "    holder_ready.set()\n"
        "    threading.Event().wait()\n"
        "\n"
        "threading.Thread(target=_hold, daemon=True).start()\n"
        "holder_ready.wait()\n"
        "print('ready', flush=True)\n"
        "lock.acquire()\n")
    stderr_path = (tmp_path / "attempts" / attempt_id / "staging" /
                   "diagnostics" / "worker.stderr")
    stderr_path.parent.mkdir(parents=True)
    proc = None
    try:
        with open(stderr_path, "ab") as stderr_fh:
            proc = subprocess.Popen([sys.executable, str(child_path)],
                                    stdout=subprocess.PIPE, stderr=stderr_fh,
                                    text=True, start_new_session=True)
        ready_line = proc.stdout.readline()
        assert ready_line.strip() == "ready", (
            f"child never blocked on its held lock: stdout {ready_line!r}")
        live_stat = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(") ", 1)[1].split()
        identity = ProcessIdentity(boot_id=boot_id, pid=proc.pid,
                                   start_ticks=int(live_stat[19]),
                                   process_group=int(live_stat[2]))
        resources = ResolvedResources(
            effective_host_budget_bytes=1 << 30, reserved_memory_bytes=1 << 29,
            assigned_cpu_ids=(0,), thread_count=1, scratch_limit_bytes=1 << 28,
            executor_mode="fake", containment="none", provider_leases=(),
            resource_profile_version="test")
        conn.execute(
            "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, "
            "supervisor_epoch, host_boot_id, state, process_state, process_json, "
            "resources_json, created_at, heartbeat_at, lease_expires_at) "
            "VALUES (?, ?, 1, 1, ?, ?, 'failed', 'exited', ?, ?, ?, ?, ?)",
            (attempt_id, job_id, epoch, boot_id, dumps(identity), dumps(resources),
             "2026-09-12T00:00:00+00:00", "2026-09-12T00:00:05+00:00",
             "2026-09-12T00:00:35+00:00"))
        conn.execute("INSERT INTO process_members (attempt_id, pid, start_ticks, "
                     "identity_json) VALUES (?, ?, ?, ?)",
                     (attempt_id, identity.pid, identity.start_ticks, dumps(identity)))
        conn.execute("UPDATE jobs SET attempt_count = 1 WHERE job_id = ?", (job_id,))

        class _NeverTicks:
            def tick(self):
                pass

        deadline_started = time.monotonic()
        with pytest.raises(AssertionError) as excinfo:
            run_until(_NeverTicks(), conn, job_id, timeout=0.2)
        deadline_elapsed = time.monotonic() - deadline_started
        message = str(excinfo.value)
        assert deadline_elapsed < 4.0
        tracked = [part for part in message.split("; ")
                   if part.startswith("tracked process diagnostics:")]
        assert len(tracked) == 1
        assert "traceback signal sent" in tracked[0]
        tails = [part for part in message.split("; ")
                 if part.startswith("worker stderr tail:")]
        assert len(tails) == 1
        assert "Current thread <thread id redacted> (most recent call first):" in tails[0]
        assert "Thread <thread id redacted> (most recent call first):" in tails[0]
        assert re.search(r'File "<path>", line \d+', tails[0])
        assert str(child_path) not in message
        proc.wait(timeout=10)
        assert proc.returncode == -signal.SIGABRT, (
            f"child not terminated by the diagnostic SIGABRT: {proc.returncode}")
    finally:
        if proc is not None:
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if proc.stdout is not None:
                proc.stdout.close()


def test_run_until_deadline_distinguishes_the_two_heartbeats(tmp_path, monkeypatch):
    """The fenced lease stamp (``attempts.heartbeat_at``), a throttled
    supervisor ``heartbeat`` observation event, the latest meaningful
    non-heartbeat step and the diagnostic-only process-family counts each
    print as their own labelled field, so a stalled worker's two different
    heartbeats never read as one -- and the family count exposes no cgroup
    path or command line. The family counts are read off a controlled process
    table (see the helper above), so ``live`` is a positive number the test
    actually proves rather than an artifact of a pid no kernel could assign."""
    conn, job_id = _queued_job(tmp_path, {"code": "DEPENDENCY_PENDING",
                                          "needed": {"dep": 1}})
    boot_id = read_boot_id()
    epoch = conn.execute("SELECT epoch_id FROM supervisor_epochs").fetchone()[0]
    pid, start_ticks = 424242, 4242
    _install_controlled_process_table(monkeypatch, tmp_path, pid, start_ticks)
    identity = ProcessIdentity(boot_id=boot_id, pid=pid, start_ticks=start_ticks,
                               process_group=111)
    resources = ResolvedResources(
        effective_host_budget_bytes=1 << 30, reserved_memory_bytes=1 << 29,
        assigned_cpu_ids=(0,), thread_count=1, scratch_limit_bytes=1 << 28,
        executor_mode="fake", containment="none", provider_leases=(),
        resource_profile_version="test")
    conn.execute(
        "INSERT INTO attempts (attempt_id, job_id, attempt_number, fence, "
        "supervisor_epoch, host_boot_id, state, process_state, process_json, "
        "resources_json, created_at, heartbeat_at, lease_expires_at) "
        "VALUES (?, ?, 1, 1, ?, ?, 'failed', 'exited', ?, ?, ?, ?, ?)",
        ("att_never", job_id, epoch, boot_id, dumps(identity), dumps(resources),
         "2026-09-12T00:00:00+00:00", "2026-09-12T00:00:05+00:00",
         "2026-09-12T00:00:35+00:00"))
    conn.execute("INSERT INTO process_members (attempt_id, pid, start_ticks, "
                 "identity_json) VALUES (?, ?, ?, ?)",
                 ("att_never", identity.pid, identity.start_ticks, dumps(identity)))
    conn.execute("UPDATE jobs SET attempt_count = 1 WHERE job_id = ?", (job_id,))
    record_progress(conn, ProgressEvent(
        job_id=job_id, attempt_id="att_never", stage_id=None, sequence=1,
        recorded_at="2026-09-12T00:00:08+00:00", kind="progress",
        elapsed_seconds=8.0, message="step complete", step="ramp",
        step_duration_seconds=3.0, step_units=10))
    record_progress(conn, ProgressEvent(
        job_id=job_id, attempt_id="att_never", stage_id=None, sequence=2,
        recorded_at="2026-09-12T00:00:12+00:00", kind="heartbeat",
        elapsed_seconds=12.0, message="still running", step="ramp"))

    class _NeverTicks:
        def tick(self):
            pass

    with pytest.raises(AssertionError) as excinfo:
        run_until(_NeverTicks(), conn, job_id, timeout=0.2)
    message = str(excinfo.value)
    assert job_id in message
    assert "'queued'" in message and "never admitted" in message
    assert ("attempt lease heartbeat: attempt 1 (failed): heartbeat at "
            "2026-09-12T00:00:05+00:00, lease expires at "
            "2026-09-12T00:00:35+00:00") in message
    assert ("latest progress event: heartbeat at 2026-09-12T00:00:12+00:00 "
            "(progress/observation event, not a lease signal)") in message
    assert ("latest non-heartbeat progress event: progress at "
            "2026-09-12T00:00:08+00:00 (step ramp): step complete") in message
    assert "process family liveness: 1 live / 1 tracked (diagnostic only)" in message
    assert "attempt count: 1" in message
    assert "queue/admission reason: DEPENDENCY_PENDING" in message
    assert "log tail: heartbeat at 2026-09-12T00:00:12+00:00: still running" in message
    assert "cgroup" not in message
    assert "command line" not in message
    # The aggregate family field remains limited to counts, while the new
    # tracked-process diagnostic names the pid whose /proc data it reports.
    family_fields = [part for part in message.split("; ")
                     if part.startswith("process family liveness:")]
    assert family_fields == ["process family liveness: 1 live / 1 tracked "
                             "(diagnostic only)"]
    counts = family_fields[0].split(": ", 1)[1].split(" (", 1)[0]
    live_text, tracked_text = counts.split(" live / ")
    assert int(live_text) > 0 and int(tracked_text.split()[0]) > 0
    tracked_fields = [part for part in message.split("; ")
                      if part.startswith("tracked process diagnostics:")]
    assert len(tracked_fields) == 1 and f"(pid {identity.pid})" in tracked_fields[0]
    assert boot_id not in message


@pytest.mark.parametrize("workers", [["-p", "no:xdist"], ["-n", "2"]], ids=["serial", "xdist"])
def test_per_test_timeout_fails_the_test_by_name_and_the_run_continues(tmp_path, workers):
    """Serial and under xdist: the alarm must fire in a worker too, since
    that is where the suite is actually run."""
    case = tmp_path / "test_sleeper.py"
    case.write_text(
        "import time\n\n"
        "def test_sleeps():\n    time.sleep(30)\n\n"
        "def test_after():\n    assert True\n")
    result = subprocess.run(
        # no:gremlins: pytest-gremlins (requirements-dev.txt, installed in every CI
        # env that runs this suite) implements xdist's pytest_configure_node hook;
        # with xdist disabled (the "serial" case's -p no:xdist) pluggy's plugin
        # validation then raises PluginValidationError and pytest exits via
        # INTERNALERROR instead of the per-test-timeout failure this test checks for
        # (same fact tools/gremlin_pilot.py documents about not disabling xdist
        # while gremlins is loaded). This test never wants gremlins active either
        # way, so disable it unconditionally.
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:gremlins", *workers,
         "-p", "tests.conftest", "--test-timeout", "1", str(case)],
        cwd=REPO, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(REPO)})
    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "TIMEOUT: " in out and "test_sleeper.py::test_sleeps" in out, out
    assert "1 failed, 1 passed" in out, out
