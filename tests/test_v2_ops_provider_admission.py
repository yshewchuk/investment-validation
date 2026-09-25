"""D4: a provider account's recorded 429 backoff blocks admission (§6.2)."""
from dataclasses import dataclass

import pytest

from engine.v2.foundation import format_timestamp
from engine.v2.ops import provider_budget
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError
from engine.v2.ops.lifecycle import complete_cancel, request_cancel
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import _reserve_provider, claim_next
from engine.v2.ops.submission import (
    JobKind,
    KindRegistry,
    RetryPolicy,
    get_job,
    job_id_for,
    submit,
)
from tests.ops_support import POLICY, catalog, enqueue_claim, request, sample


@dataclass(frozen=True)
class ProviderParameters:
    provider_calls: int = 1


REGISTRY = KindRegistry([JobKind(
    name="tiny_provider", worker="tiny", parameters=ProviderParameters,
    resource_classes=frozenset({"delivery"}), effects=("staged",),
    retry=RetryPolicy("bounded", 3, (1, 2)), checkpoint_contract="rows.v1.0",
    namespaces=frozenset({"shadow"}))])


def test_provider_backoff_blocks_admission_until_it_elapses(tmp_path):
    conn, clock, supervisor = catalog(tmp_path)
    provider_budget.configure_account(conn, "acct-1", "gen-1", remaining=100, live_reserve=0)
    provider_budget.record_response(conn, "acct-1", 429, clock=clock)
    submit(conn, REGISTRY, POLICY,
          request(kind="tiny_provider", parameters={"provider_calls": 3},
                  provider_budget_ref="acct-1"), clock=clock)

    assert claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                      supervisor=supervisor, clock=clock) is None
    job = get_job(conn, job_id_for("shadow", "one"))
    assert job.queue_reason.code == "PROVIDER_BACKOFF"
    assert job.attempt_count == 0
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0

    clock.advance(66)
    claim = claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock),
                       supervisor=supervisor, clock=clock)
    assert claim is not None
    assert claim.job_id == job_id_for("shadow", "one")


def test_reserve_blocks_through_backoff_and_admits_at_the_eligible_instant(tmp_path):
    """The stamp record_response(429) writes IS the admission instant: a
    reservation taken exactly there succeeds, and the durable lease it leaves
    behind carries the claim's own fence and call estimate."""
    conn, clock, supervisor = catalog(tmp_path)
    provider_budget.configure_account(conn, "acct-1", "gen-1", remaining=10, live_reserve=0)
    provider_budget.record_response(conn, "acct-1", 429, clock=clock)
    claim = enqueue_claim(conn, clock, supervisor)

    clock.advance(64)  # still inside the 65 s source backoff
    with pytest.raises(OpsError) as err:
        provider_budget.reserve(conn, claim, "acct-1", 1, clock=clock)
    assert err.value.code == "RATE_LIMITED"
    assert conn.execute("SELECT COUNT(*) FROM provider_reservations").fetchone()[0] == 0

    clock.advance(1)  # the account is eligible at the recorded stamp, not after
    provider_budget.reserve(conn, claim, "acct-1", 1, clock=clock)
    row = conn.execute("SELECT * FROM provider_reservations").fetchone()
    assert row["account"] == "acct-1"
    assert row["attempt_id"] == claim.attempt_id
    assert row["fence"] == claim.fence
    assert row["reserved_calls"] == 1
    assert row["released_at"] is None


def test_reserve_refuses_a_fence_invalidated_by_cancellation(tmp_path):
    """Cancellation voids the fence before the worker is even asked to stop
    (§6.3): while the job is only cancelling the fence is already void
    (CANCELLED), and once the attempt is closed the claim is superseded
    outright (LEASE_LOST) -- either way the reservation never reaches the
    account and the ledger stays untouched."""
    conn, clock, supervisor = catalog(tmp_path)
    provider_budget.configure_account(conn, "acct-1", "gen-1", remaining=10, live_reserve=0)
    claim = enqueue_claim(conn, clock, supervisor)
    request_cancel(conn, claim.job_id, claim.attempt_id, clock=clock)

    with pytest.raises(OpsError) as err:
        provider_budget.reserve(conn, claim, "acct-1", 2, clock=clock)
    assert err.value.code == "CANCELLED"
    assert conn.execute("SELECT COUNT(*) FROM provider_reservations").fetchone()[0] == 0

    complete_cancel(conn, claim.job_id, process_state="verified_dead", clock=clock)
    with pytest.raises(OpsError) as err:
        provider_budget.reserve(conn, claim, "acct-1", 2, clock=clock)
    assert err.value.code == "LEASE_LOST"
    assert conn.execute("SELECT COUNT(*) FROM provider_reservations").fetchone()[0] == 0
    account = conn.execute("SELECT remaining, uncertain FROM provider_accounts "
                           "WHERE account = 'acct-1'").fetchone()
    assert account[:] == (10, 0)


def test_reserve_holds_one_live_lease_and_only_unreserved_calls(tmp_path):
    """One attempt at a time may hold an account, an estimate that counts no
    planned retry is refused before anything is written, an account missing
    or flagged for operator action never leases, and live_reserve stays
    spendable headroom -- the boundary that fits exactly is the one that
    fits, and the lease frees only when the prior attempt is reconciled
    dead."""
    conn, clock, supervisor = catalog(tmp_path)
    claim_a = enqueue_claim(conn, clock, supervisor, "one")
    claim_b = enqueue_claim(conn, clock, supervisor, "two")
    provider_budget.configure_account(conn, "acct-lease", "gen-1", remaining=5, live_reserve=2)

    with pytest.raises(OpsError) as err:
        provider_budget.reserve(conn, claim_a, "acct-lease", 0, clock=clock)
    assert err.value.code == "INVALID_REQUEST"

    provider_budget.reserve(conn, claim_a, "acct-lease", 3, clock=clock)
    with pytest.raises(OpsError) as err:
        provider_budget.reserve(conn, claim_b, "acct-lease", 1, clock=clock)
    assert err.value.code == "RESOURCE_UNAVAILABLE"
    live = conn.execute("SELECT COUNT(*) FROM provider_reservations "
                        "WHERE released_at IS NULL").fetchone()[0]
    assert live == 1

    provider_budget.configure_account(conn, "acct-blocked", "gen-1", remaining=10, live_reserve=0)
    provider_budget.record_response(conn, "acct-blocked", 401, clock=clock)
    for account in ("acct-blocked", "acct-missing"):
        with pytest.raises(OpsError) as err:
            provider_budget.reserve(conn, claim_b, account, 1, clock=clock)
        assert err.value.code == "CREDENTIAL_INVALID"

    provider_budget.configure_account(conn, "acct-tight", "gen-1", remaining=2, live_reserve=2)
    with pytest.raises(OpsError) as err:
        provider_budget.reserve(conn, claim_b, "acct-tight", 1, clock=clock)
    assert err.value.code == "RESOURCE_UNAVAILABLE"

    request_cancel(conn, claim_a.job_id, claim_a.attempt_id, clock=clock)
    complete_cancel(conn, claim_a.job_id, process_state="verified_dead", clock=clock)
    provider_budget.reserve(conn, claim_b, "acct-lease", 1, clock=clock)
    rows = {row["attempt_id"]: row["released_at"] for row in conn.execute(
        "SELECT attempt_id, released_at FROM provider_reservations")}
    assert rows[claim_a.attempt_id] is not None
    assert rows[claim_b.attempt_id] is None


def test_claim_reservation_refuses_exhausted_account_and_admits_when_budget_fits(tmp_path):
    """The production reservation path: scheduler._reserve_provider, run inside
    the claim transaction from _create_attempt (engine/v2/ops/scheduler.py
    ~326) -- provider_budget.reserve is not reached in production. An exhausted
    account (an estimate above remaining minus live_reserve) is refused with
    RESOURCE_UNAVAILABLE and leaves no reservation behind; the identical call
    against an account whose budget covers the estimate is admitted (negative
    control) and holds the live lease for the claim's own fence."""
    conn, clock, supervisor = catalog(tmp_path)
    claim = enqueue_claim(conn, clock, supervisor)
    stamp = format_timestamp(clock.now())

    provider_budget.configure_account(conn, "acct-exhausted", "gen-1", remaining=1, live_reserve=0)
    with pytest.raises(OpsError) as err:
        with transaction(conn):
            _reserve_provider(conn, "acct-exhausted", claim.attempt_id, claim.fence,
                              {"provider_calls": 2}, stamp)
    assert err.value.code == "RESOURCE_UNAVAILABLE"
    assert conn.execute("SELECT COUNT(*) FROM provider_reservations").fetchone()[0] == 0

    provider_budget.configure_account(conn, "acct-open", "gen-1", remaining=2, live_reserve=0)
    with transaction(conn):
        _reserve_provider(conn, "acct-open", claim.attempt_id, claim.fence,
                          {"provider_calls": 2}, stamp)
    row = conn.execute("SELECT * FROM provider_reservations WHERE account = 'acct-open'").fetchone()
    assert row["attempt_id"] == claim.attempt_id
    assert row["fence"] == claim.fence
    assert row["reserved_calls"] == 2
    assert row["released_at"] is None
