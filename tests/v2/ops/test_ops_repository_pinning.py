"""Ops Repository pinning: materialize's worker opens a pinned Repository, and a pinned Repository reads a table's membership once across repeated scans.

Part 2 of #1018: the ops sites listed in ``engine/v2/ops/ARCHITECTURE.md``
already pass ``pin_snapshots=True``; these tests pin that contract at the two
seams a regression would silently reintroduce — the materialization worker's
constructor kwargs, and the membership memo a repeated scan depends on.
"""
from __future__ import annotations

from engine.v2.data.repository import Repository
from engine.v2.ops.snapshot_roots import materialization_root
from tests.v2.data.test_repository_pinned_snapshots import (
    _build,
    _count_membership_reads,
    _scan_and_bound,
)
from tests.v2.ops.test_v2_ops_snapshot_stages import Case, _envelope


def test_write_root_opens_a_pinned_repository(tmp_path, monkeypatch):
    """``materialization_worker._write_root`` constructs its ``Repository`` with ``pin_snapshots=True``.

    Uses the same ``Case`` fixture the neighbouring ``materialization_worker``
    tests exercise, so the assertion runs against the real catalog/store/request
    path rather than a stub.
    """
    from engine.v2.ops import materialization_worker as worker

    case = Case(tmp_path)
    try:
        calls: list[dict] = []
        original_init = Repository.__init__

        def recording_init(self, conn, store=None, **kwargs):
            calls.append(kwargs)
            original_init(self, conn, store, **kwargs)

        monkeypatch.setattr(Repository, "__init__", recording_init)

        envelope = _envelope(case, "att_pin_check")
        dest = materialization_root(case.base, case.request.request_hash)
        worker._write_root(case.request, envelope["materialization"], dest, "att_pin_check")

        assert calls, "materialization_worker._write_root constructed no Repository"
        assert any(kwargs.get("pin_snapshots") is True for kwargs in calls), calls
    finally:
        case.conn.close()


def test_pinned_repository_reads_membership_once_across_two_scans(tmp_path, monkeypatch):
    """A pinned Repository scans the same table twice but runs the membership query once.

    Resolve is warmed outside the counted window so the assertion isolates the
    per-table fragment membership fetch (``Repository._records`` via
    ``_table_records``), which the pin memoises by immutable catalog identity.
    """
    conn, store, ref, snapshot = _build(tmp_path)
    try:
        repository = Repository(conn, store, pin_snapshots=True)
        repository.resolve(snapshot.snapshot_id)
        calls = _count_membership_reads(monkeypatch)
        first = _scan_and_bound(repository, snapshot, ref)
        second = _scan_and_bound(repository, snapshot, ref)
        assert first == second == (1, ["B"])
        assert len(calls) == 1, calls
    finally:
        conn.close()
