"""D19: the bounded-subprocess mechanism the render-parity comparator uses.

A tiny RSS cap crossed by a trivial child must be reported as a CLEAR
failure (``tools/bounded_run.py``'s own exit 137, with the per-process
memory breakdown on stdout) rather than a hang or a bare timeout -- the
watchdog polls and kills the tree itself. No render bundle, no catalog, no
legacy import anywhere in this file.
"""
from __future__ import annotations

import os

from checks.rearchitecture_phase2_render_parity import run_bounded


def test_exceeding_a_tiny_rss_cap_is_reported_as_a_clean_137_not_a_hang(tmp_path):
    # The child HOLDS the allocation: a watchdog that samples every poll_s
    # cannot see a spike that is allocated and freed between two samples, and
    # on a fast runner the bare allocation exits before the first sample (seen
    # in CI 2026-09-20: "0.00G pss ... exited 0"). Sleeping keeps the breach
    # observable for as long as the watchdog needs to notice it, which is what
    # this test is about -- not how quickly Python can allocate a list.
    proc = run_bounded(
        ["/usr/bin/python3", "-c",
         "import time; x = [0] * (200 * 1024 * 1024); time.sleep(30)"],
        max_rss_gb=0.05, cwd=tmp_path, env=dict(os.environ), poll_s=0.25, timeout=60)
    assert proc.returncode == 137
    assert "CAP BREACH" in proc.stdout
    assert "killed at the memory cap" in proc.stdout


def test_a_command_under_the_cap_exits_cleanly(tmp_path):
    proc = run_bounded(["/usr/bin/python3", "-c", "print('ok')"], max_rss_gb=1.0,
                       cwd=tmp_path, env=dict(os.environ), poll_s=1, timeout=60)
    assert proc.returncode == 0
    assert "ok" in proc.stdout
