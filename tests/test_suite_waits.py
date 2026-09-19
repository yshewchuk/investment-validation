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
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *workers,
         "-p", "tests.conftest", "--test-timeout", "1", str(case)],
        cwd=REPO, capture_output=True, text=True, timeout=60,
        env={**os.environ, "PYTHONPATH": str(REPO)})
    out = result.stdout + result.stderr
    assert result.returncode == 1, out
    assert "TIMEOUT: " in out and "test_sleeper.py::test_sleeps" in out, out
    assert "1 failed, 1 passed" in out, out
