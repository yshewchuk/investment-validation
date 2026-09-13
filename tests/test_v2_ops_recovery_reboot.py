"""An attempt launched under a previous boot is dead by definition.

Its recorded pid and start ticks belong to another boot's process table. The
session check must not match them against this boot's live sessions, or a
reboot would quarantine the attempt behind an unrelated process forever.
"""
from __future__ import annotations

import os

from engine.v2.contracts import ProcessIdentity
from engine.v2.ops.lifecycle import record_launch
from engine.v2.ops.recovery import prove_ownership_gone, read_boot_id
from tests.ops_support import catalog, enqueue_claim


def test_previous_boot_attempt_is_proven_gone_despite_matching_live_session(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)  # supervisor boot id is "boot"
    claim = enqueue_claim(conn, clock, supervisor)
    live_session = os.getsid(0)  # a session that is alive in THIS boot
    record_launch(conn, claim.attempt_id, claim.fence,
                  ProcessIdentity(boot_id="boot", pid=live_session, start_ticks=0,
                                  process_group=live_session),
                  clock=clock, lease_seconds=120)

    proof = prove_ownership_gone(conn, claim.attempt_id, boot_id=read_boot_id())

    assert proof.proven is True
    assert proof.blockers == () and proof.alive == ()


def test_same_boot_matching_live_session_still_blocks(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    live_session = os.getsid(0)
    record_launch(conn, claim.attempt_id, claim.fence,
                  ProcessIdentity(boot_id="boot", pid=live_session, start_ticks=0,
                                  process_group=live_session),
                  clock=clock, lease_seconds=120)

    proof = prove_ownership_gone(conn, claim.attempt_id, boot_id="boot")

    assert proof.proven is False
    assert proof.blockers
