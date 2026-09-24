"""An attempt launched under a previous boot is dead by definition.

Its recorded pid and start ticks belong to another boot's process table. The
session check must not match them against this boot's live sessions, or a
reboot would quarantine the attempt behind an unrelated process forever.
"""
from __future__ import annotations

import subprocess
import sys

import pytest

from engine.v2.contracts import ProcessIdentity
from engine.v2.ops.executor_watchdog import process_info
from engine.v2.ops.lifecycle import record_launch
from engine.v2.ops.recovery import prove_ownership_gone, read_boot_id
from tests.ops_support import catalog, enqueue_claim

# Real boot identity / process-identity machinery, grouped with its sibling
# test_v2_ops_recovery_ownership.py (see tests/conftest.py's grouping rule).
pytestmark = pytest.mark.xdist_group("serial")


def _live_session_leader():
    """Popen a child leading its own session; the caller kills/waits in finally.

    Not os.getsid(0): a wrapper that launches the test process as its own
    session leader (tools/bounded_run.py start_new_session, mutmut's
    in-process stats run) names the running process itself, which find_owners
    excludes as our own ancestor, and a wrapper whose leader already exited
    names a dead pid. Either way the "live session" would be vacuous.
    """
    return subprocess.Popen([sys.executable, "-u", "-c", "import time; time.sleep(30)"],
                            start_new_session=True, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _session_of(leader):
    session = process_info(leader.pid, "boot")[4]  # stat session field
    assert session == leader.pid  # leads its own session, alive
    return session


def test_previous_boot_attempt_is_proven_gone_despite_matching_live_session(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)  # supervisor boot id is "boot"
    boot = read_boot_id()
    assert boot != "boot"  # else the previous-boot guard would be skipped
    claim = enqueue_claim(conn, clock, supervisor)
    leader = _live_session_leader()
    try:
        live_session = _session_of(leader)  # a session that IS alive in THIS boot
        record_launch(conn, claim.attempt_id, claim.fence,
                      ProcessIdentity(boot_id="boot", pid=live_session, start_ticks=0,
                                      process_group=live_session),
                      clock=clock, lease_seconds=120)

        proof = prove_ownership_gone(conn, claim.attempt_id, boot_id=boot)

        # The recorded owner belongs to another boot, so it is proven gone
        # despite its pid naming this boot's live session: without the
        # boot guard, find_owners' session check would hit this very child.
        assert proof.proven is True
        assert proof.blockers == () and proof.alive == ()
        assert leader.poll() is None  # the live session held throughout the proof
        assert process_info(live_session, boot)[4] == live_session
    finally:
        leader.kill()
        leader.wait()


def test_same_boot_matching_live_session_still_blocks(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    leader = _live_session_leader()
    try:
        live_session = _session_of(leader)
        record_launch(conn, claim.attempt_id, claim.fence,
                      ProcessIdentity(boot_id="boot", pid=live_session, start_ticks=0,
                                      process_group=live_session),
                      clock=clock, lease_seconds=120)

        proof = prove_ownership_gone(conn, claim.attempt_id, boot_id="boot")

        assert proof.proven is False
        assert proof.blockers
        assert live_session in {pid for pid, _ in proof.blockers}
        # start_ticks=0 matches no real process: only check (b), the live
        # session carrying the launch pid, can block here.
        assert proof.alive == ()
    finally:
        leader.kill()
        leader.wait()
