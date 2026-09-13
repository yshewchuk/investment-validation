"""D4: a provider account's recorded 429 backoff blocks admission (§6.2)."""
from dataclasses import dataclass

from engine.v2.ops import provider_budget
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.submission import (
    JobKind,
    KindRegistry,
    RetryPolicy,
    get_job,
    job_id_for,
    submit,
)
from tests.ops_support import POLICY, catalog, request, sample


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
