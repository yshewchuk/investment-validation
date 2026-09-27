"""Direct tests for two ``computed_moves_store.py`` helpers with no other
coverage yet: the store's full ``RefreshCallback`` path is covered starting
Parts 3/4 (see ``engine/v2/ops/ARCHITECTURE.md``'s "not yet wired" note)."""
from __future__ import annotations

import pytest

from engine.v2.ops.catalog import transaction
from engine.v2.ops.computed_moves_store import _capture_id_for, _fence_check_for
from engine.v2.ops.errors import OpsError
from engine.v2.ops.incremental_data import RefreshUnit
from tests.ops_support import catalog, enqueue_claim


def _unit(ticker: str, day: str) -> RefreshUnit:
    return RefreshUnit(request_id=f"computed_moves:{ticker}:{day}",
                       table_name="computed_moves", partition_key=ticker,
                       expected_keys=(ticker,))


def test_capture_id_is_stable_for_the_same_unit_and_differs_for_a_different_one():
    """Same (ticker, as_of) -> same capture_id, called twice -- independent of
    wall-clock time. This is the bug the ``request_id``-derived id fixes: the
    old code folded in the run's own ``created_at``, so a same-day retry never
    matched the already-logged capture and ``_insert_captures`` would have
    double-logged it."""
    unit = _unit("ABCD", "2026-09-24")
    first = _capture_id_for(unit)
    second = _capture_id_for(unit)
    assert first == second
    assert first.startswith("capture_")
    other_ticker = _capture_id_for(_unit("WXYZ", "2026-09-24"))
    other_day = _capture_id_for(_unit("ABCD", "2026-09-25"))
    assert other_ticker != first
    assert other_day != first


def test_fence_check_for_matches_the_real_verify_fence_and_keeps_the_lease_check(tmp_path):
    """``_fence_check_for`` must build a callable ``verify_fence`` accepts with
    its REAL signature (``conn, attempt_id, fence, now`` -- no
    ``check_lease_time`` keyword; ``engine.v2.ops.lifecycle.verify_fence`` has
    never had one). It must also still enforce the production wall-clock
    lease-expiry check -- never a skipped check, no matter who calls this
    store."""
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    check = _fence_check_for(claim.attempt_id, claim.fence, clock)

    with transaction(conn):
        job, attempt = check(conn)
        assert job["fence"] == claim.fence
        assert attempt["fence"] == claim.fence

    clock.advance(10 ** 6)  # long past any lease_expires_at
    with transaction(conn):
        with pytest.raises(OpsError) as err:
            check(conn)
        assert err.value.code == "LEASE_LOST"


def test_fence_check_for_with_no_staged_attempt_is_a_noop():
    """No staged attempt (e.g. a manual/ad-hoc invocation with nothing to
    fence against): the returned callable does nothing and returns ``None``,
    matching ``_commit_generation``'s previous inline ternary."""
    check = _fence_check_for(None, None, clock=None)
    assert check(object()) is None
