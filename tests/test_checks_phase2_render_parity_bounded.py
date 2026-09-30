"""D19: the bounded-subprocess mechanism the render-parity comparator uses.

A tiny RSS cap crossed by a trivial child must be reported as a CLEAR
failure (``tools/bounded_run.py``'s own exit 137, with the per-process
memory breakdown on stdout) rather than a hang or a bare timeout -- the
watchdog polls and kills the tree itself. No render bundle, no catalog, no
legacy import anywhere in this file.
"""
from __future__ import annotations

import os
import subprocess

import checks.rearchitecture_phase2_render_parity as parity
from checks.rearchitecture_phase2_render_parity import run_bounded


def test_run_bounded_passes_a_resource_wait_below_its_own_timeout(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["timeout"] = command, kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    parity.run_bounded(["/usr/bin/python3", "-c", "pass"], max_rss_gb=1.0,
                       cwd=tmp_path, env=dict(os.environ), timeout=1800)
    argv = seen["command"]
    assert "--max-wait-s" in argv
    assert float(argv[argv.index("--max-wait-s") + 1]) < seen["timeout"]


def test_run_bounded_keeps_the_resource_wait_below_a_short_timeout(tmp_path, monkeypatch):
    # A 60 s floor would make --max-wait-s equal to (not below) a 60 s
    # timeout; the fix floors at 0 instead so the invariant holds even here.
    seen = {}

    def fake_run(command, **kwargs):
        seen["command"], seen["timeout"] = command, kwargs["timeout"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(parity.subprocess, "run", fake_run)
    parity.run_bounded(["/usr/bin/python3", "-c", "pass"], max_rss_gb=1.0,
                       cwd=tmp_path, env=dict(os.environ), timeout=60)
    argv = seen["command"]
    assert "--max-wait-s" in argv
    wait = float(argv[argv.index("--max-wait-s") + 1])
    assert wait < seen["timeout"]
    assert wait == 0.0


def test_exceeding_a_tiny_rss_cap_is_reported_as_a_clean_137_not_a_hang(tmp_path):
    # The child HOLDS the allocation: a watchdog that samples every poll_s
    # cannot see a spike that is allocated and freed between two samples, and
    # on a fast runner the bare allocation exits before the first sample (seen
    # in CI 2026-09-20: "0.00G pss ... exited 0"). Sleeping keeps the breach
    # observable for as long as the watchdog needs to notice it, which is what
    # this test is about -- not how quickly Python can allocate a list.
    env = dict(os.environ, BOUNDED_RUN_STATE_DIR=str(tmp_path / "bounded_state"))
    proc = run_bounded(
        ["/usr/bin/python3", "-c",
         "import time; x = [0] * (200 * 1024 * 1024); time.sleep(30)"],
        max_rss_gb=0.05, cwd=tmp_path, env=env, poll_s=0.25, timeout=60)
    assert proc.returncode == 137
    assert "CAP BREACH" in proc.stdout
    assert "killed at the memory cap" in proc.stdout


def test_a_command_under_the_cap_exits_cleanly(tmp_path):
    env = dict(os.environ, BOUNDED_RUN_STATE_DIR=str(tmp_path / "bounded_state"))
    proc = run_bounded(["/usr/bin/python3", "-c", "print('ok')"], max_rss_gb=1.0,
                       cwd=tmp_path, env=env, poll_s=1, timeout=60)
    assert proc.returncode == 0
    assert "ok" in proc.stdout


def test_isolation_survives_a_fully_occupied_unrelated_pool(tmp_path, monkeypatch):
    """#187: without the BOUNDED_RUN_STATE_DIR override the two tests above
    now use, a launch here would share whatever pool BOUNDED_RUN_STATE_DIR
    resolves to with every other concurrent bounded_run.py caller. Point the
    AMBIENT BOUNDED_RUN_STATE_DIR at a stand-in "unrelated" pool (never the
    real default -- that could race actually-concurrent unrelated tests),
    hold every one of its slots, and confirm both halves: (a) a launch that
    still inherits that ambient pool is starved (exit 75 -- proving the
    contention this test sets up is real, not a no-op), while (b) a launch
    given its own private dir, exactly as the two tests above now do, is
    admitted immediately regardless."""
    import fcntl

    unrelated_pool = tmp_path / "unrelated_pool"
    unrelated_pool.mkdir()
    monkeypatch.setenv("BOUNDED_RUN_STATE_DIR", str(unrelated_pool))
    monkeypatch.setenv("BOUNDED_RUN_SLOTS", "3")
    held = []
    for index in range(3):
        fd = os.open(unrelated_pool / f"slot-{index}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.append(fd)
    try:
        # BOUNDED_RUN_NESTED=1 leaks in from the bounded_run.py wrapper that
        # launched this very pytest session (oc_check runs tests under it);
        # a nested child SKIPS the slot wait, so the flag must be cleared for
        # the contention below to be real rather than bypassed.
        starved_env = dict(os.environ, BOUNDED_RUN_NESTED="0")
        starved = run_bounded(["/usr/bin/python3", "-c", "print('ok')"], max_rss_gb=1.0,
                              cwd=tmp_path, env=starved_env, poll_s=1, timeout=60)
        assert starved.returncode == 75

        isolated_env = dict(os.environ, BOUNDED_RUN_NESTED="0",
                            BOUNDED_RUN_STATE_DIR=str(tmp_path / "bounded_state"))
        proc = run_bounded(["/usr/bin/python3", "-c", "print('ok')"], max_rss_gb=1.0,
                           cwd=tmp_path, env=isolated_env, poll_s=1, timeout=60)
        assert proc.returncode == 0
        assert "ok" in proc.stdout
    finally:
        for fd in held:
            os.close(fd)
