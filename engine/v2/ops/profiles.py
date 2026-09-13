"""The versioned resource policy for this host — phase-1 guide §8.1.

Callers pick a named profile; this policy turns it into bytes, CPUs and
scratch. Every amount is **bytes**; GiB below is 2**30, never 10**9.

The numbers are starting evidence, not measurements. The historical scorer
(~3 GiB resident) and tier rebuild (~5.5 GiB peak) come from AGENTS.md's
2026-09-11 notes, with margin added where a peak step was observed to exceed
the average. Every heavy profile is ``measured=False``, which makes it
exclusive of other heavy work until a reviewed measurement changes it.
Profiles change **between runs, in a new policy version**, never automatically
after one cheap cache-hit run, and never by lowering a reservation until a
stage fits on paper (§8.1).
"""
from __future__ import annotations

from engine.v2.contracts import ResourcePolicy, ResourceProfile
from engine.v2.ops.errors import fail

__all__ = ["DEFAULT_POLICY", "GIB", "MIB", "POLICY_VERSION", "policy_problems", "profile_named"]

GIB = 1 << 30
MIB = 1 << 20

POLICY_VERSION = "ops_resources.2026-09-13.v2"

DEFAULT_POLICY = ResourcePolicy(
    version=POLICY_VERSION,
    # OS, the API process and the supervisor: capacity no worker may reserve.
    base_reserve_bytes=1 * GIB,
    # Buffer against allocation spikes the watchdog cannot see between polls.
    free_margin_bytes=512 * MIB,
    reserved_cpu_count=1,
    min_free_disk_bytes=5 * GIB,
    max_heavy_concurrency=1,
    max_disk_heavy_concurrency=1,
    profiles=(
        ResourceProfile(name="io_fetch", memory_bytes=512 * MIB, cpu_count=1,
                        scratch_bytes=2 * GIB, heavy=False),
        ResourceProfile(name="delivery", memory_bytes=256 * MIB, cpu_count=1,
                        scratch_bytes=1 * GIB, heavy=False),
        ResourceProfile(name="projection", memory_bytes=2 * GIB, cpu_count=2,
                        scratch_bytes=2 * GIB, heavy=False),
        # The serialized selfcheck builds a bounded scorer of its own. Raised
        # to 11/2 GiB with legacy_score below: the adapted legacy scoring path
        # (validation shares that code) peaked at 4.15 GiB tree RSS on the
        # 2026-09-13 38-request canary, above the prior 4 GiB reservation.
        ResourceProfile(name="validation", memory_bytes=11 * GIB // 2, cpu_count=4,
                        scratch_bytes=1 * GIB, heavy=True),
        # Measured 2026-09-13: 4.15 GiB tree RSS peak on a 38-request canary,
        # above the prior 4 GiB reservation — the watchdog was killing it.
        ResourceProfile(name="legacy_score", memory_bytes=11 * GIB // 2, cpu_count=5,
                        scratch_bytes=2 * GIB, heavy=True),
        ResourceProfile(name="model_evidence", memory_bytes=4 * GIB, cpu_count=4,
                        scratch_bytes=1 * GIB, heavy=True),
        ResourceProfile(name="legacy_rebuild", memory_bytes=11 * GIB // 2, cpu_count=5,
                        scratch_bytes=20 * GIB, heavy=True, disk_heavy=True),
        ResourceProfile(name="experiment_heavy", memory_bytes=11 * GIB // 2, cpu_count=5,
                        scratch_bytes=10 * GIB, heavy=True, disk_heavy=True),
    ),
)


def profile_named(policy: ResourcePolicy, name: str) -> ResourceProfile:
    for profile in policy.profiles:
        if profile.name == name:
            return profile
    raise fail("INVALID_REQUEST", "the resource policy has no such profile",
               details={"profile": name, "policy": policy.version})


def policy_problems(policy: ResourcePolicy) -> list[str]:
    """Structural problems with a policy, before it is used to admit anything."""
    problems: list[str] = []
    names = [p.name for p in policy.profiles]
    if len(set(names)) != len(names):
        problems.append("duplicate profile names")
    if policy.max_heavy_concurrency < 1 or policy.max_disk_heavy_concurrency < 1:
        problems.append("concurrency limits must be at least 1")
    if min(policy.base_reserve_bytes, policy.free_margin_bytes, policy.reserved_cpu_count,
           policy.min_free_disk_bytes) < 0:
        problems.append("reserves and margins cannot be negative")
    for profile in policy.profiles:
        if profile.memory_bytes <= 0 or profile.cpu_count <= 0 or profile.scratch_bytes < 0:
            problems.append(f"profile {profile.name} has a non-positive allocation")
    return problems
