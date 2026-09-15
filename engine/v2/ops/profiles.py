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

v4 (2026-09-14) right-sizes ``scratch_bytes`` for every profile a kind that
goes through ``supervisor.Service._pin_read_set`` /
``_populate_legacy_staging`` can carry (``stages.registry()``'s
``legacy_finality``/``legacy_score``/``legacy_decisions``/``legacy_settlement``/
``legacy_model_evidence``/``legacy_render``/``legacy_selfcheck``/
``legacy_score_requests``/``legacy_decision_replay``/``snapshot_import``).
Basis: the real 2026-09-14 4-ticker snapshot-mode nightly staged a
``legacy_finality`` read set of ``needed_bytes=1796916876`` (~1.67 GiB)
against the ``validation`` profile's old 1 GiB scratch limit and was refused
at claim time (``RESOURCE_LIMIT_EXCEEDED``). 4 GiB is ~2.4x that measured
read set -- headroom for a larger watchlist, not a value chosen to clear one
run by the smallest margin; disk free on this host is ~900 GiB, so nothing
here is disk-constrained. ``legacy_rebuild`` (20 GiB) and ``materialize``
(20 GiB) already clear this basis and are unchanged.

v5 (2026-09-15) raises ``legacy_score``'s ``memory_bytes`` from 5 GiB to
6 GiB -- an interim safety margin, not a substitute for the real fix (see
``engine/data/features/tier4.py::serving_model``'s cache-hit branch, same
commit: a serving-cache HIT now serves its pool from the joblib file instead
of unconditionally re-deriving it via ``_pool_before`` -> ``model.prepare
(panel)`` -> a full ``daily_market`` re-read; that fix only pays off once
Tier 4 rebuilds and repins cache files in the new format -- every cache file
pinned as of this commit still lacks the embedded pool and falls back to the
exact prior behaviour, so THIS corpus still needs the extra headroom below
until Tier 4 is rebuilt). Basis: real attempt-16 (6-ticker shadow nightly,
201-ticker context, `/root/phase2-shadow-ops`) --
``legacy_score`` attempt ``att_2e0745184ae4c4eb17aeb782a0d398ac`` measured
``memory_peak_bytes=5447757824`` (~5.07 GiB), above the old 5 GiB (
5368709120) reservation; ``legacy_decision_replay`` attempt
``att_86920be0153fb81004abef9ae927d0f9`` was killed at 93s,
``memory_peak_bytes=5396807680`` and rising. A clean reproduction against
the same materialized inputs, via the real adapter action path
(``engine.v2.ops.legacy_adapter._action_score`` /
``_action_decision_replay``, current code, no profile/code change), peaked
at self-reported RSS 4748 MiB for score and 5126 MiB for replay --
``legacy_decision_replay`` exercises MORE distinct Tier-4 producers than
``legacy_score`` for the identical board (the DYN-SV chooser's menu touches
``size_v1_4``/``iv_crush_v1_gbm`` in addition to the two producers
``legacy_score`` itself needed), which is why replay's peak is higher and
why it was the one actually killed. 6 GiB (6442450944 bytes) gives ~0.93 GiB
margin over the measured 5.07 GiB score peak and ~1.26 GiB over the
reproduced 5.13 GiB (5126 MiB) replay peak, and fits inside the effective
host budget of 7089033216 bytes (~6.60 GiB) with ``max_heavy_concurrency=1``
(no other heavy profile is ever admitted alongside it) -- margin to the host
budget itself is ~0.60 GiB. This is not full headroom for an arbitrarily
larger board: a wider watchlist that touches a THIRD Tier-4 producer or a
second fold would need re-measurement, per this module's own "never lower a
reservation until a stage fits on paper" rule (§8.1) applied in reverse --
raise again with numbers, do not guess ahead of one.

v6 (2026-09-15) lowers ``legacy_score`` from 6 GiB back to 5.25 GiB
(``5*GIB + GIB//4`` = 5637144576 bytes): v5's 6 GiB was never actually
admittable on this host and queued attempt 17 forever
(``queue_reason MEMORY_HEADROOM``, ``needed.memory_bytes=6442450944``,
``available.headroom_bytes=5846372352``, nothing else heavy running). v5's
docstring compared 6 GiB to ``capacity_bytes`` (host_total - base_reserve,
7089033216 bytes, ~6.60 GiB) -- the wrong ceiling. Admission's binding test
is ``headroom_bytes`` (live ``host_available_bytes`` - ``free_margin_bytes``),
which never reaches ``capacity_bytes`` in practice because base baseline
processes (this box's own agents/shells) always hold some of the difference
between ``base_reserve_bytes`` (1 GiB) and ``free_margin_bytes`` (512 MiB)
plus more. Two real samples on this host (both via
``engine.v2.ops.discovery.sample_capacity`` -> ``resources.headroom_bytes``,
2026-09-15): the attempt-17 queuing moment gave 5846372352 bytes
(~5.44 GiB, ~5.85 GB); a fresh idle read minutes later gave 6205313024 bytes
(~5.78 GiB). The conservative (lower) of the two, 5846372352 bytes, is this
module's stated ceiling -- a live figure, not a policy constant, so it can
be lower on a busier day.

5.25 GiB (5637144576 bytes) sits between the two real measured bases:
~180.6 MiB (189386752 bytes) above ``legacy_score``'s own attempt-16
watchdog tree-RSS peak of 5447757824 bytes (~5.07 GiB, the same measurement
the executor's own kill check uses -- ``executor.py::poll`` fails the
attempt with ``RESOURCE_LIMIT_EXCEEDED`` once tree RSS exceeds
``reserved_memory_bytes``, so this margin **is** the kill margin, not a
separate number), and ~199.5 MiB (209227776 bytes) below the conservative
ceiling. Both margins are real and positive but thin (prior profiles in
this file carry ~0.85-2.4x); there is no room to widen either side without
either shrinking the other or touching ``free_margin_bytes``/
``base_reserve_bytes`` (not done here -- no new evidence that either is
wrong for this host, and the rule above is "don't lower the safety reserve
without evidence").

This number is NOT equally justified for the three actions sharing this
profile. ``legacy_score_requests`` (warm-cache peak 4.44 GiB, measured) and
``legacy_score`` (5.07 GiB watchdog peak, measured to completion) both clear
5.25 GiB with real margin. ``legacy_decision_replay`` does not have a
completed watchdog (tree-RSS) peak at all: its only tree-RSS data point is
attempt ``att_86920be0153fb81004abef9ae927d0f9``, which the watchdog KILLED
at 5396807680 bytes (~5.03 GiB) while still rising, under the OLD 5 GiB cap
-- a lower bound, not a peak. The "5126 MiB replay peak" cited in v5's
docstring is a same-process self-report from a clean reproduction via the
adapter action path directly (not through ``bounded_run``/the watchdog), so
it is not on the same tree-RSS basis as the 5.07 GiB score figure and
cannot be trusted as replay's true ceiling. Judgement call (mine, not
independently re-verified): 5.25 GiB is the best-evidenced number available
for all three actions today, chosen over other splits (e.g. carving
``legacy_decision_replay`` a separate, larger profile) because there is no
real completed watchdog measurement to size a larger number against --
inventing one would not be measurement, it would be a guess ahead of
evidence, which this file's own rule forbids. The residual risk is real:
if replay's true tree-RSS peak exceeds ~5.25 GiB, it will hit
``RESOURCE_LIMIT_EXCEEDED`` again, the same failure mode v5 was chasing.
The needed follow-up is a real, isolated ``legacy_decision_replay`` run
under ``bounded_run.py --max-rss-gb`` set well above 5.25 GiB (per AGENTS.md
"Running jobs on this box", with no other heavy job active) to get a
completed tree-RSS peak, not a further paper resize.
"""
from __future__ import annotations

from engine.v2.contracts import ResourcePolicy, ResourceProfile
from engine.v2.ops.errors import fail

__all__ = ["DEFAULT_POLICY", "GIB", "MIB", "POLICY_VERSION", "policy_problems", "profile_named"]

GIB = 1 << 30
MIB = 1 << 20

POLICY_VERSION = "ops_resources.2026-09-15.v6"

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
        # legacy_render (store_domains read) stages the legacy read set
        # through the same barrier as validation/legacy_score below --
        # scratch bumped to the v4 basis (1.67 GiB measured + headroom).
        ResourceProfile(name="projection", memory_bytes=2 * GIB, cpu_count=2,
                        scratch_bytes=4 * GIB, heavy=False),
        # The serialized selfcheck builds a bounded scorer of its own.
        # Lowered 2026-09-14 from 11/2 GiB to 5 GiB (right-sizing pass, ops
        # memory reservation review): the adapted legacy scoring path
        # (validation shares that code) peaked at 4.15 GiB tree RSS on the
        # 2026-09-13 38-request canary. 5 GiB is 0.85 GiB of margin over that
        # measured peak; memory.high (90%, executor_cgroup.py) becomes
        # 4.5 GiB, still above the peak. Admission needs
        # available - 512 MiB margin >= reservation, and measured headroom on
        # this 7.8 GiB host is 4.15-5.43 GiB while other agents run tests, so
        # 11/2 GiB (5.5 GiB) could never be admitted concurrently with other
        # work; 5 GiB can. Scratch bumped 2026-09-14 (v4) from 1 GiB: this
        # profile carries legacy_finality/legacy_decisions/legacy_selfcheck,
        # whose barrier-path read set measured 1.67 GiB on a real 4-ticker
        # snapshot nightly (see module docstring) -- 4 GiB is that plus
        # headroom, never a value chosen to make one run pass on paper.
        ResourceProfile(name="validation", memory_bytes=5 * GIB, cpu_count=4,
                        scratch_bytes=4 * GIB, heavy=True),
        # Lowered 2026-09-14 from 11/2 GiB to 5 GiB for the same reason as
        # validation above: measured peak is 4.15 GiB tree RSS (2026-09-13
        # 38-request canary), so 5 GiB keeps 0.85 GiB of margin while fitting
        # the host's measured 4.15-5.43 GiB headroom under contention.
        # Scratch bumped 2026-09-14 (v4) from 2 GiB to the same 1.67 GiB +
        # headroom basis as validation above -- this profile also carries
        # legacy_score/legacy_score_requests/legacy_decision_replay's own
        # barrier-path read set outside snapshot mode.
        # Raised 2026-09-15 (v5) from 5 GiB to 6 GiB: real attempt-16
        # measured legacy_score at 5.07 GiB and legacy_decision_replay was
        # killed rising through 5.03 GiB. 6 GiB was never admittable here --
        # it exceeded the host's live headroom ceiling (~5.44-5.78 GiB
        # measured, not the larger host-total-based capacity v5 compared
        # against) and queued attempt 17 forever. Lowered 2026-09-15 (v6) to
        # 5.25 GiB (5*GIB + GIB//4): ~180 MiB above the measured 5.07 GiB
        # watchdog peak (also the kill margin -- executor.py's cap equals
        # this reservation) and ~200 MiB below the conservative measured
        # ceiling. See the module docstring's v6 entry for both real
        # capacity samples, why the margins are this thin, and the
        # unresolved risk for legacy_decision_replay specifically (no
        # completed watchdog peak exists for it).
        ResourceProfile(name="legacy_score", memory_bytes=5 * GIB + GIB // 4, cpu_count=5,
                        scratch_bytes=4 * GIB, heavy=True),
        # legacy_model_evidence (store_domains read) stages the legacy read
        # set through the same barrier -- scratch bumped 2026-09-14 (v4) from
        # 1 GiB to the 1.67 GiB + headroom basis above.
        ResourceProfile(name="model_evidence", memory_bytes=4 * GIB, cpu_count=4,
                        scratch_bytes=4 * GIB, heavy=True),
        ResourceProfile(name="legacy_rebuild", memory_bytes=11 * GIB // 2, cpu_count=5,
                        scratch_bytes=20 * GIB, heavy=True, disk_heavy=True),
        # Covers the legacy_materialize worker (engine/v2/data/legacy_materialization.py
        # ::materialize_tree): whole-table curated tables and pinned reference
        # files are byte copies from the ArtifactStore (streamed in 1 MiB
        # chunks — _copy_verified_object), and every other table (only
        # option_chains is evidence_scoped) is a bounded Arrow scan written
        # batch-by-batch via repository.scan(), capped at
        # maximum_batch_rows=50,000 rows per batch (legacy_annotations.json)
        # and never materializing a whole table or year partition in memory
        # at once. There is no legacy scorer or panel model load on this
        # path. Measurement basis: the full synthetic
        # tests/test_v2_data_legacy_materialization.py suite (41 tests,
        # exercising every scan/copy path against small fixtures) peaked at
        # 293 MiB RSS for the whole pytest process under
        # ``/usr/bin/time -v`` on 2026-09-14; static reading of every scan
        # path above confirms real data cannot exceed a small multiple of
        # one 50,000-row batch. 2 GiB leaves wide margin over both. Routed
        # here from legacy_rebuild (previously 5.5 GiB, shared with the much
        # heavier tier rebuild/legacy_settlement work) so a materialize job
        # can be admitted without reserving tier-rebuild-sized memory it
        # never uses.
        ResourceProfile(name="materialize", memory_bytes=2 * GIB, cpu_count=2,
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
