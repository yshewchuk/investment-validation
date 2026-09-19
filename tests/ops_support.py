"""Small operations fixtures. No market data, numerical imports or network."""
import json
import os
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from engine.v2.contracts import CapacitySample, JobSpec, SubmitRequest
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.profiles import DEFAULT_POLICY, MIB
from engine.v2.ops.recovery import begin_epoch
from engine.v2.ops.scheduler import Supervisor, claim_next
from engine.v2.ops.submission import JobKind, KindRegistry, NamespacePolicy, RetryPolicy, submit

GIB = 1 << 30


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


# -- bounded admission waits ---------------------------------------------------
#
# A test that drives a real ``Service.tick()`` loop depends on the HOST:
# ``tick()`` samples this process's real CPU affinity, MemAvailable and free
# disk (``discovery.sample_capacity``), and a job the sample cannot admit just
# stays ``queued`` with a ``queue_reason_json``. Before these helpers each test
# polled with its own deadline (18 s to 1800 s) and then returned "queued", so a
# starved job read as ``assert 'queued' == 'succeeded'`` at best. In the 1800 s
# corpus-parity loop it read as a sleeping xdist worker that never reported a
# test name. ``run_until`` and ``AdmissionWatch`` fail the test instead, quoting
# the queue reason's own numbers, as soon as the wait is known to be
# environmental:
#
# * ``PROFILE_EXCEEDS_CAPACITY`` fails on the first sample. It compares the
#   profile against host TOTAL memory and this process's CPU affinity, and
#   neither changes while the test runs, so waiting can never help. Measured
#   2026-09-19: under ``taskset -c 0-3`` (what ``bounded_run --cores 4`` does)
#   4 allowed CPUs leave 3 worker CPUs, the 5-CPU ``legacy_score`` profile
#   never fits, and a corpus-parity test that takes 10 s sat queued with the
#   worker asleep.
# * The other host-dependent reasons (``HOST_RESOURCE_REASONS``) fail once the
#   job has sat continuously queued on them for ``ADMISSION_WAIT_SECONDS``
#   (env ``OPS_TEST_ADMISSION_WAIT_S``, default 60), or at the caller's own
#   deadline if that comes first. A shortage that clears inside the window
#   (another process briefly holding memory) still just waits, exactly as the
#   production scheduler does.
#
# Queue reasons that come from the test's OWN catalog (a dependency, a held
# heavy slot, a store lease) are never treated as environmental: they fall
# through to the caller's deadline and assertions unchanged.
#
# These are failures, not skips, on purpose: the check did not run, and a
# skip is easy to read past in a summary that is otherwise green. Every message
# starts with ``RESOURCE WAIT`` so it can be told apart from a code failure.

TERMINAL_STATES = ("succeeded", "failed", "blocked", "cancelled")

#: Queue reasons (``engine.v2.ops.resources.decide``) whose outcome depends on
#: the host sample rather than on other jobs in the test's own catalog.
HOST_RESOURCE_REASONS = frozenset({
    "PROFILE_EXCEEDS_CAPACITY", "MEMORY_HEADROOM", "RESERVATION_BUDGET",
    "CPU_UNAVAILABLE", "DISK_SPACE"})

ADMISSION_WAIT_SECONDS = float(os.environ.get("OPS_TEST_ADMISSION_WAIT_S", "60"))


def _gib(value) -> str:
    return f"{value / GIB:.2f} GiB" if isinstance(value, (int, float)) else "?"


def _mem_available_bytes() -> int | None:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def admission_message(conn, job_id, reason: dict, waited: float, *, policy=TEST_POLICY) -> str:
    """What the job waits for, what it needs, what the host has, and what to
    do about it."""
    row = conn.execute("SELECT spec_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    spec = json.loads(row[0]) if row and row[0] else {}
    kind, profile = spec.get("kind", "?"), spec.get("resource_class", "?")
    code = reason.get("code", "?")
    needed, available = reason.get("needed") or {}, reason.get("available") or {}
    head = (f"RESOURCE WAIT (environment, not a code failure): job {job_id} "
            f"(kind {kind}, profile {profile}) stayed queued {waited:.0f}s on {code}. ")
    margin = policy.free_margin_bytes
    if code == "MEMORY_HEADROOM":
        want = needed.get("memory_bytes", 0) + needed.get("owed_unconsumed_bytes", 0)
        body = (f"Waiting for memory: needs {_gib(want)} of headroom and has "
                f"{_gib(available.get('headroom_bytes'))} (headroom = MemAvailable - "
                f"{_gib(margin)} free margin), so it needs {_gib(want + margin)} free; "
                f"MemAvailable is {_gib(_mem_available_bytes())} now. Free memory "
                f"(other agents, heavy jobs, fewer xdist workers) and rerun this file.")
    elif code == "PROFILE_EXCEEDS_CAPACITY":
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
        body = (f"The profile needs {needed.get('cpus')} worker CPUs and "
                f"{_gib(needed.get('memory_bytes'))}. This process's CPU affinity has "
                f"{len(cpus)} CPUs, {available.get('worker_cpus')} after the "
                f"{policy.reserved_cpu_count} reserved, and memory capacity is "
                f"{_gib(available.get('capacity_bytes'))}. That cannot change during "
                f"the run: give pytest at least "
                f"{(needed.get('cpus') or 0) + policy.reserved_cpu_count} CPUs in its "
                f"affinity (no taskset, bounded_run --cores or --cpu-set narrower "
                f"than that).")
    elif code == "CPU_UNAVAILABLE":
        body = (f"Waiting for CPUs: needs {needed.get('cpus')} free worker CPUs, "
                f"{available.get('free_cpus')} free in this process's affinity.")
    elif code == "DISK_SPACE":
        body = (f"Waiting for disk: needs {_gib(needed.get('scratch_bytes'))} of scratch, "
                f"{_gib(available.get('spare_bytes'))} spare above the "
                f"{_gib(policy.min_free_disk_bytes)} floor.")
    elif code == "RESERVATION_BUDGET":
        body = (f"Waiting for reservation budget: needs {_gib(needed.get('memory_bytes'))}, "
                f"{_gib(available.get('reserved_bytes'))} of "
                f"{_gib(available.get('capacity_bytes'))} already reserved.")
    else:
        body = f"needed={needed} available={available}."
    return head + body


class AdmissionWatch:
    """Fail the test once one job's host-resource wait is known to be
    hopeless (see the block comment above). Call :meth:`check` once per poll,
    and with ``final=True`` at the caller's own deadline."""

    def __init__(self, conn, job_id, *, policy=TEST_POLICY, wait_seconds=None):
        self.conn, self.job_id, self.policy = conn, job_id, policy
        self.wait_seconds = ADMISSION_WAIT_SECONDS if wait_seconds is None else wait_seconds
        self.since = None

    def check(self, *, final=False) -> None:
        import pytest

        row = self.conn.execute("SELECT state, queue_reason_json FROM jobs WHERE job_id=?",
                                (self.job_id,)).fetchone()
        reason = json.loads(row[1]) if row and row[0] == "queued" and row[1] else None
        if reason is None or reason.get("code") not in HOST_RESOURCE_REASONS:
            self.since = None
            return
        now = time.monotonic()
        self.since = now if self.since is None else self.since
        waited = now - self.since
        if reason["code"] == "PROFILE_EXCEEDS_CAPACITY" or final or waited >= self.wait_seconds:
            pytest.fail(admission_message(self.conn, self.job_id, reason, waited,
                                          policy=self.policy), pytrace=False)


def job_state(conn, job_id) -> str:
    return conn.execute("SELECT state FROM jobs WHERE job_id=?", (job_id,)).fetchone()[0]


def run_until(service, conn, job_id, *, timeout, states=TERMINAL_STATES, poll=0.05) -> str:
    """Tick ``service`` until ``job_id`` reaches one of ``states`` or ``timeout``
    elapses, and return its state. This is the loop every real-Service test
    used to carry, plus :class:`AdmissionWatch`: a job the host cannot admit
    fails the test with its queue reason instead of sleeping to the deadline."""
    watch = AdmissionWatch(conn, job_id, policy=getattr(service, "policy", TEST_POLICY))
    deadline = time.monotonic() + timeout
    state = job_state(conn, job_id)
    while time.monotonic() < deadline:
        service.tick()
        state = job_state(conn, job_id)
        if state in states:
            return state
        watch.check()
        time.sleep(poll)
    watch.check(final=True)
    return state
