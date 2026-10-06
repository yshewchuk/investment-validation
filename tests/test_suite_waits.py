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


def test_run_until_fails_at_its_explicit_budget_on_an_own_catalog_heavy_slot_wait(tmp_path):
    """Regression (corpus timeout): a ``HEAVY_SLOT`` wait comes from the
    test's own catalog, so it is never environmental -- ``run_until`` must
    fall through to the caller's explicit deadline and raise
    ``RunUntilTimeout`` inside that budget, carrying the queue reason and
    the attempt/lease diagnostics, not fail ``RESOURCE WAIT`` or overrun."""
    conn, job_id = _queued_job(tmp_path, {"code": "HEAVY_SLOT"})

    class _NeverAdmits:
        def tick(self):
            pass

    timeout = 0.2
    started = time.monotonic()
    with pytest.raises(RunUntilTimeout) as excinfo:
        run_until(_NeverAdmits(), conn, job_id, timeout=timeout, poll=0.01)
    # budget + 0.1 s for the deadline-edge diagnostics only: no retry, no extension
    assert time.monotonic() - started <= timeout + 0.1
    message = str(excinfo.value)
    assert "state 'queued'" in message
    assert "HEAVY_SLOT" in message
    assert "attempt count:" in message
    assert "attempt lease heartbeat:" in message


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
    # the documented family output is only the live/tracked counts: that one
    # rendered field carries nothing beyond them -- no pid or boot identity --
    # and the controlled live row above makes BOTH counts nonzero, so a
    # "0 live" that only proves an impossible pid can never pass here
    family_fields = [part for part in message.split("; ")
                     if part.startswith("process family liveness:")]
    assert family_fields == ["process family liveness: 1 live / 1 tracked "
                             "(diagnostic only)"]
    counts = family_fields[0].split(": ", 1)[1].split(" (", 1)[0]
    live_text, tracked_text = counts.split(" live / ")
    assert int(live_text) > 0 and int(tracked_text.split()[0]) > 0
    assert str(identity.pid) not in message and boot_id not in message


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
