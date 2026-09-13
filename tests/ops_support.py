"""Small operations fixtures. No market data, numerical imports or network."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from engine.v2.contracts import CapacitySample, JobSpec, SubmitRequest
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.profiles import DEFAULT_POLICY, MIB
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.submission import JobKind, KindRegistry, NamespacePolicy, RetryPolicy, submit


class FakeClock:
    def __init__(self):
        self.value = datetime(2026, 9, 12, tzinfo=timezone.utc)
        self.elapsed = 0.0

    def now(self):
        return self.value

    def monotonic(self):
        return self.elapsed

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)
        self.elapsed += seconds


@dataclass(frozen=True)
class Parameters:
    value: int = 1


REGISTRY = KindRegistry([JobKind(
    name="tiny", worker="tiny", parameters=Parameters,
    resource_classes=frozenset({"delivery", "legacy_score", "legacy_rebuild"}),
    effects=("staged",), retry=RetryPolicy("bounded", 3, (1, 2)),
    checkpoint_contract="rows.v1.0", namespaces=frozenset({"shadow"}))])
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})

#: A small-memory mirror of DEFAULT_POLICY for any test that runs a job
#: through a real ``Service.tick()``. Unlike ``claim_next(..., sample=sample(clock))``
#: above (a fixed fake ``CapacitySample``), ``Service.tick()`` samples the
#: host's *actual* free memory (``discovery.sample_capacity``). DEFAULT_POLICY
#: reserves up to 11 GiB // 2 for its heavy profiles (profiles.py), which on a
#: shared host makes admission race whatever else happens to be using memory
#: at that moment -- a job stays "queued" past a test's fixed poll deadline
#: whenever headroom is briefly short, with nothing wrong in the code under
#: test. Every field is copied from DEFAULT_POLICY except each profile's
#: memory_bytes, capped small: cpu_count/thread_count stay identical, so an
#: environment_ref built from DEFAULT_POLICY (production code and most tests
#: still do, since it does not depend on memory_bytes) still matches what
#: launch resolves under this policy.
_TEST_PROFILE_MEMORY_BYTES = 256 * MIB
TEST_POLICY = replace(DEFAULT_POLICY, profiles=tuple(
    replace(p, memory_bytes=min(p.memory_bytes, _TEST_PROFILE_MEMORY_BYTES))
    for p in DEFAULT_POLICY.profiles))


def request(key="one", **changes):
    fields = dict(kind="tiny", implementation_ref="code", spec_hash=None, environment_ref="env",
                  parameters={"value": 1}, output_namespace="shadow", resource_class="delivery",
                  retry_policy_ref="bounded", checkpoint_contract_ref="rows.v1.0")
    fields.update(changes)
    return SubmitRequest(namespace="shadow", idempotency_key=key, principal="operator", job=JobSpec(**fields))


def sample(clock):
    return CapacitySample(sampled_at=clock.now().strftime("%Y-%m-%dT%H:%M:%S.000000Z"),
                          allowed_cpu_ids=(1, 3, 5, 7, 9, 11), host_total_bytes=8 << 30,
                          host_available_bytes=7 << 30, container_limit_bytes=None,
                          container_current_bytes=None, swap_total_bytes=0, swap_free_bytes=0,
                          disk_free_bytes=100 << 30, executor_mode="fake", containment="none")


def catalog(tmp_path):
    clock = FakeClock()
    conn = open_catalog(tmp_path / "ops.sqlite", clock=clock)
    epoch = begin_epoch(conn, clock=clock, boot_id="boot", pid=1)
    return conn, clock, Supervisor(epoch, "boot")


def enqueue_claim(conn, clock, supervisor, key="one", **changes):
    submit(conn, REGISTRY, POLICY, request(key, **changes), clock=clock)
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor, clock=clock)
