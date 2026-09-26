"""tools/v2_ops_fixtures.py: enqueue_claim works with no tests/ import."""
from __future__ import annotations

from datetime import datetime, timezone

from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor
from tools.v2_ops_fixtures import enqueue_claim


class _Clock:
    def now(self):
        return datetime(2026, 9, 12, tzinfo=timezone.utc)


def test_enqueue_claim_returns_a_claim_with_no_tests_package_import(tmp_path):
    clock = _Clock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    epoch_id = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    claim = enqueue_claim(conn, clock, Supervisor(epoch_id, "boot"), key="one")
    assert claim is not None
