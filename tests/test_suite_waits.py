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
    assert "last worker heartbeat: n/a" in message  # absent field: placeholder
    assert "log tail: n/a" in message               # absent field: placeholder

    def _broken_attempt_receipts(*args, **kwargs):
        raise RuntimeError("simulated helper failure")

    monkeypatch.setattr("tests.ops_support.attempt_receipts", _broken_attempt_receipts)
    with pytest.raises(AssertionError) as excinfo:
        run_until(_NeverAdmits(), conn, job_id, timeout=0.2)
    message = str(excinfo.value)
    assert "last worker heartbeat: n/a" in message  # failed helper: placeholder
    assert "attempt count: 0" in message            # other diagnostics survive
    assert "DEPENDENCY_PENDING" in message
    assert "log tail: n/a" in message
    assert job_id in message
    assert "'queued'" in message and "never admitted" in message

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
    for label in ("attempt count: n/a", "queue/admission reason: n/a",
                  "last worker heartbeat: n/a", "log tail: n/a"):
        assert label in message, message

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
    assert "attempt count: n/a (attempt count unavailable: ValueError)" in message
    assert "queue/admission reason: DEPENDENCY_PENDING" in message
    assert "log tail: tick at" in message
    assert "last worker heartbeat: attempt 1 (queued)" in message

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
