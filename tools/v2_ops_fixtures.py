"""Tiny synthetic job-kind fixtures, usable from outside ``tests/``.

``enqueue_claim`` and the ``REGISTRY``/``POLICY``/``request``/``sample``
pieces it needs were moved here out of ``tests/ops_support.py`` so that
``checks/`` and ``tools/`` drills can submit and claim one synthetic job
without importing the ``tests`` package at runtime. ``tests/ops_support.py``
re-exports every name here unchanged, so its existing callers see no
difference.
"""
from dataclasses import dataclass

from engine.v2.contracts import CapacitySample, JobSpec, SubmitRequest
from engine.v2.ops.profiles import DEFAULT_POLICY
from engine.v2.ops.scheduler import claim_next
from engine.v2.ops.submission import JobKind, KindRegistry, NamespacePolicy, RetryPolicy, submit


@dataclass(frozen=True)
class Parameters:
    value: int = 1


REGISTRY = KindRegistry([JobKind(
    name="tiny", worker="tiny", parameters=Parameters,
    resource_classes=frozenset({"delivery", "legacy_score", "legacy_rebuild"}),
    effects=("staged",), retry=RetryPolicy("bounded", 3, (1, 2)),
    checkpoint_contract="rows.v1.0", namespaces=frozenset({"shadow"}))])
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


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


def enqueue_claim(conn, clock, supervisor, key="one", **changes):
    submit(conn, REGISTRY, POLICY, request(key, **changes), clock=clock)
    return claim_next(conn, policy=DEFAULT_POLICY, sample=sample(clock), supervisor=supervisor, clock=clock)
