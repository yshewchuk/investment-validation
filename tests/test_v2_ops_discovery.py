"""#341: admission reads each constrained cgroup's working set, not raw usage.

Synthetic temporary cgroup trees only — never live cgroups, never workers.
"""
from dataclasses import replace

import pytest

from engine.v2.contracts import ResourcePolicy
from engine.v2.foundation import SystemClock
from engine.v2.ops import discovery
from engine.v2.ops.resources import headroom_bytes

MiB = 1 << 20
GiB = 1 << 30


def _cgroup(root, name="cg", limit=1000, current=None, stat=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "memory.max").write_text("max" if limit is None else str(limit))
    if current is not None:
        (directory / "memory.current").write_text(str(current))
    if stat is not None:
        (directory / "memory.stat").write_text(stat)
    return directory


def _stat(**values):
    return "".join(f"{key} {value}\n" for key, value in values.items())


def _policy(free_margin=100):
    return ResourcePolicy(version="test", base_reserve_bytes=0, free_margin_bytes=free_margin,
                          reserved_cpu_count=1, min_free_disk_bytes=0,
                          max_heavy_concurrency=1, max_disk_heavy_concurrency=1, profiles=())


def _proc(tmp_path, available_bytes=4 * GiB):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text(
        f"MemTotal: {8 * GiB // 1024} kB\n"
        f"MemAvailable: {available_bytes // 1024} kB\n"
        "SwapTotal: 0 kB\nSwapFree: 0 kB\n")
    return proc


def _sample(tmp_path, monkeypatch, cg, available_bytes=4 * GiB):
    monkeypatch.setattr(discovery, "cgroup_directory", lambda proc: cg)
    return discovery.sample_capacity(tmp_path, clock=SystemClock(),
                                     proc=_proc(tmp_path, available_bytes=available_bytes))


# -- memory_limits: working set ------------------------------------------------


def test_inactive_file_is_the_fallback_when_shmem_missing(tmp_path):
    cg = _cgroup(tmp_path, current=800,
                 stat=_stat(anon=50, file=500, inactive_file=300, active_file=200, inactive_anon=100))
    assert discovery.memory_limits(cg) == (1000, 500)


def test_only_the_exact_inactive_file_key_is_read(tmp_path):
    cg = _cgroup(tmp_path, current=800,
                 stat="file 100\nactive_file 200\ninactive_anon 400\n"
                      "not_inactive_file 450\ninactive_file 300\n")
    assert discovery.memory_limits(cg) == (1000, 500)


def test_zero_inactive_file_keeps_usage_unchanged(tmp_path):
    cg = _cgroup(tmp_path, current=800, stat=_stat(inactive_file=0))
    assert discovery.memory_limits(cg) == (1000, 800)


def test_inactive_file_above_current_clamps_usage_to_zero(tmp_path):
    cg = _cgroup(tmp_path, current=400, stat=_stat(inactive_file=900))
    assert discovery.memory_limits(cg) == (1000, 0)


def test_missing_memory_stat_falls_back_to_raw_current(tmp_path):
    cg = _cgroup(tmp_path, current=800)
    assert discovery.memory_limits(cg) == (1000, 800)


def test_absent_inactive_file_falls_back_to_raw_current(tmp_path):
    cg = _cgroup(tmp_path, current=800, stat=_stat(anon=400, file=300, active_file=300))
    assert discovery.memory_limits(cg) == (1000, 800)


@pytest.mark.parametrize("stat", ["inactive_file abc\n", "inactive_file -5\n",
                                  "inactive_file\n", "inactive_file 1 2\n",
                                  "inactive_file 3_0\n"])
def test_malformed_inactive_file_falls_back_to_raw_current(tmp_path, stat):
    cg = _cgroup(tmp_path, current=800, stat=stat)
    assert discovery.memory_limits(cg) == (1000, 800)


@pytest.mark.parametrize("stat", ["inactive_file +300\n", "inactive_file ٣٠٠\n"])
def test_int_parsable_but_not_ascii_decimal_falls_back_to_raw_current(tmp_path, stat):
    # Negative controls for the ASCII-decimal guard: int() accepts a sign,
    # underscores, and non-ASCII decimal digits, but cgroup memory.stat is
    # ASCII decimal, so these must keep the raw memory.current.
    cg = _cgroup(tmp_path, current=800, stat=stat)
    assert discovery.memory_limits(cg) == (1000, 800)


def test_unreadable_memory_stat_falls_back_to_raw_current(tmp_path):
    cg = _cgroup(tmp_path, current=800)
    (cg / "memory.stat").mkdir()
    assert discovery.memory_limits(cg) == (1000, 800)


def test_undecodable_memory_stat_falls_back_to_raw_current(tmp_path):
    cg = _cgroup(tmp_path, current=800)
    (cg / "memory.stat").write_bytes(b"inactive_file 300\n\xff\xfe\xfa\n")
    assert discovery.memory_limits(cg) == (1000, 800)


def test_hierarchical_remaining_uses_each_parents_working_set(tmp_path):
    parent = _cgroup(tmp_path, limit=GiB, current=900 * MiB, stat=_stat(inactive_file=400 * MiB))
    child = _cgroup(parent, name="child", limit=512 * MiB, current=300 * MiB,
                    stat=_stat(inactive_file=0))
    # Raw, the parent caps remaining at GiB - 900 MiB = 124 MiB (usage 388 MiB);
    # with the parent's 500 MiB working set the child's 212 MiB remaining binds.
    assert discovery.memory_limits(child) == (512 * MiB, 300 * MiB)


def test_unlimited_memory_max_still_reports_no_container(tmp_path):
    cg = _cgroup(tmp_path, limit=None, current=800, stat=_stat(inactive_file=300))
    assert discovery.memory_limits(cg) == (None, None)


def test_unknown_memory_current_still_reports_full_limit(tmp_path):
    cg = _cgroup(tmp_path, limit=1000, current=None, stat=_stat(inactive_file=300))
    assert discovery.memory_limits(cg) == (1000, 1000)


# -- wired through sample_capacity into resources.headroom_bytes ----------------


def test_headroom_rises_by_inactive_file_fallback_only(tmp_path, monkeypatch):
    cg = _cgroup(tmp_path, current=800,
                 stat=_stat(file=500, active_file=200, inactive_anon=100, inactive_file=300))
    sample = _sample(tmp_path, monkeypatch, cg)
    assert (sample.container_limit_bytes, sample.container_current_bytes) == (1000, 500)
    policy = _policy()
    adjusted = headroom_bytes(policy, sample)
    raw = headroom_bytes(policy, replace(sample, container_current_bytes=800))
    assert adjusted == 400
    assert raw == 100
    assert adjusted - raw == 300


def test_no_finite_limit_keeps_headroom_host_only(tmp_path, monkeypatch):
    cg = _cgroup(tmp_path, limit=None, current=800, stat=_stat(inactive_file=300))
    sample = _sample(tmp_path, monkeypatch, cg)
    assert (sample.container_limit_bytes, sample.container_current_bytes) == (None, None)
    assert headroom_bytes(_policy(), sample) == 4 * GiB - 100


def test_unknown_current_still_yields_no_container_headroom(tmp_path, monkeypatch):
    cg = _cgroup(tmp_path, limit=1000, current=None, stat=_stat(inactive_file=300))
    sample = _sample(tmp_path, monkeypatch, cg)
    assert sample.container_current_bytes == 1000
    assert headroom_bytes(_policy(), sample) == -100


# -- #347: file - shmem is the working set; inactive_file is only the fallback --


def test_active_file_cache_is_reclaimable_not_just_inactive_file(tmp_path):
    # Dominant active cache: file=500 is mostly active_file, inactive_file=50.
    # #343 subtracted inactive_file only (used 750/headroom 150); #347 subtracts
    # file - shmem = 500, so used is 300.
    cg = _cgroup(tmp_path, current=800,
                 stat=_stat(anon=250, file=500, shmem=0, inactive_file=50, active_file=450))
    assert discovery.memory_limits(cg) == (1000, 300)


def test_headroom_counts_active_file_cache_as_reclaimable(tmp_path, monkeypatch):
    cg = _cgroup(tmp_path, current=800,
                 stat=_stat(file=500, shmem=0, inactive_file=50, active_file=450))
    sample = _sample(tmp_path, monkeypatch, cg)
    assert (sample.container_limit_bytes, sample.container_current_bytes) == (1000, 300)
    assert headroom_bytes(_policy(free_margin=100), sample) == 600


def test_shmem_stays_counted_as_used(tmp_path):
    # Same fixture with shmem=200: only file - shmem = 300 is reclaimable, so
    # used climbs from 300 to 500 -- the exact 200 that tmpfs holds. Subtracting
    # every file byte (ignoring shmem) would wrongly report 300.
    tmpfs = _cgroup(tmp_path, name="tmpfs", current=800,
                    stat=_stat(file=500, shmem=200, inactive_file=450, active_file=50))
    reclaimable = _cgroup(tmp_path, name="cache", current=800, stat=_stat(file=500, shmem=0))
    assert discovery.memory_limits(tmpfs) == (1000, 500)
    assert discovery.memory_limits(reclaimable) == (1000, 300)


def test_headroom_leaves_shmem_counted_as_used(tmp_path, monkeypatch):
    cg = _cgroup(tmp_path, current=800, stat=_stat(file=500, shmem=200, inactive_file=450))
    sample = _sample(tmp_path, monkeypatch, cg)
    assert sample.container_current_bytes == 500
    assert headroom_bytes(_policy(free_margin=100), sample) == 400


def test_zero_reclaimable_pair_beats_inactive_file(tmp_path):
    # file == shmem (nothing reclaimable) and file == shmem == 0 are both VALID
    # zero subtractions that must take precedence over the inactive_file fallback.
    equal = _cgroup(tmp_path, name="equal", current=800,
                    stat=_stat(file=500, shmem=500, inactive_file=450))
    zeros = _cgroup(tmp_path, name="zeros", current=800,
                    stat=_stat(file=0, shmem=0, inactive_file=300))
    assert discovery.memory_limits(equal) == (1000, 800)
    assert discovery.memory_limits(zeros) == (1000, 800)


def test_reclaimable_above_current_clamps_usage_to_zero(tmp_path):
    cg = _cgroup(tmp_path, current=400, stat=_stat(file=900, shmem=0))
    assert discovery.memory_limits(cg) == (1000, 0)


def test_host_available_caps_headroom_when_container_remaining_exceeds_it(tmp_path, monkeypatch):
    # file-shmem=500 -> used 300 -> container remaining 9700, but only 8 KiB is
    # host-available, so the host binds (a container-only reading would be 9600).
    cg = _cgroup(tmp_path, limit=10_000, current=800, stat=_stat(file=500, shmem=0))
    sample = _sample(tmp_path, monkeypatch, cg, available_bytes=8 * 1024)
    assert sample.container_current_bytes == 300
    assert headroom_bytes(_policy(free_margin=100), sample) == 8 * 1024 - 100


def test_child_limits_remaining_using_its_own_file_and_shmem(tmp_path):
    # Parent's dominant active cache (file-shmem=800 MiB) makes parent almost
    # empty, so it does NOT cap; the child subtracts its OWN file-shmem (80 MiB).
    # #343 (inactive_file only) would cap parent at 124 MiB and report 388 MiB.
    parent = _cgroup(tmp_path, limit=GiB, current=900 * MiB,
                     stat=_stat(file=800 * MiB, shmem=0, inactive_file=0))
    child = _cgroup(parent, name="child", limit=512 * MiB, current=300 * MiB,
                    stat=_stat(file=100 * MiB, shmem=20 * MiB))
    assert discovery.memory_limits(child) == (512 * MiB, 220 * MiB)


def test_parent_limits_remaining_using_its_own_file_and_shmem(tmp_path):
    # Child reports a valid file==shmem==0 pair (zero reclaimable, NOT the large
    # inactive_file), so the parent's own file-shmem working set binds instead.
    parent = _cgroup(tmp_path, limit=GiB, current=900 * MiB,
                     stat=_stat(file=100 * MiB, shmem=0))
    child = _cgroup(parent, name="child", limit=512 * MiB, current=100 * MiB,
                    stat=_stat(file=0, shmem=0, inactive_file=400 * MiB))
    assert discovery.memory_limits(child) == (512 * MiB, 288 * MiB)


def test_hierarchical_missing_child_current_still_reports_full_ceiling(tmp_path):
    parent = _cgroup(tmp_path, limit=GiB, current=600 * MiB,
                     stat=_stat(file=100 * MiB, shmem=0))
    child = _cgroup(parent, name="child", limit=512 * MiB, current=None,
                    stat=_stat(file=400 * MiB, shmem=0))
    assert discovery.memory_limits(child) == (512 * MiB, 512 * MiB)


def test_hierarchical_unlimited_child_leaves_parent_in_charge(tmp_path):
    parent = _cgroup(tmp_path, limit=GiB, current=600 * MiB,
                     stat=_stat(file=200 * MiB, shmem=0))
    child = _cgroup(parent, name="child", limit=None, current=10 * MiB,
                    stat=_stat(file=9 * MiB, shmem=0))
    assert discovery.memory_limits(child) == (GiB, 400 * MiB)


@pytest.mark.parametrize("bad", [
    "file 500\n",                # shmem missing
    "shmem 100\n",               # file missing
    "file 500\nshmem abc\n",     # shmem non-numeric
    "file -5\nshmem 0\n",        # negative file
    "file +5\nshmem 0\n",        # signed file
    "file 3_0\nshmem 0\n",       # underscore separator
    "file ٣٠٠\nshmem 0\n",       # non-ASCII decimal digits
    "file 500 500\nshmem 0\n",   # malformed field count for file
    "file 500\nshmem 900\n",     # shmem > file
])
def test_bad_file_or_shmem_falls_back_to_inactive_file(tmp_path, bad):
    cg = _cgroup(tmp_path, current=800, stat=bad + "inactive_file 300\n")
    assert discovery.memory_limits(cg) == (1000, 500)


@pytest.mark.parametrize("bad", [
    "file 500\n",                       # shmem missing, no inactive_file
    "file 500\nshmem 900\n",            # shmem > file, no inactive_file
    "file abc\nshmem 0\ninactive_file x\n",  # file invalid, inactive_file invalid
])
def test_bad_file_or_shmem_without_inactive_file_keeps_raw_current(tmp_path, bad):
    cg = _cgroup(tmp_path, current=800, stat=bad)
    assert discovery.memory_limits(cg) == (1000, 800)


def test_valid_file_and_shmem_beat_a_malformed_inactive_file(tmp_path):
    cg = _cgroup(tmp_path, current=800, stat="file 500\nshmem 100\ninactive_file abc\n")
    assert discovery.memory_limits(cg) == (1000, 400)


def test_valid_file_and_shmem_beat_a_missing_inactive_file(tmp_path):
    cg = _cgroup(tmp_path, current=800, stat=_stat(file=500, shmem=100))
    assert discovery.memory_limits(cg) == (1000, 400)


def test_similarly_named_stat_keys_are_ignored(tmp_path):
    # Only the exact `file`/`shmem` keys count; file_mapped, shmem_huge and the
    # active/inactive variants must not leak into the subtraction.
    cg = _cgroup(tmp_path, current=800,
                 stat=_stat(file_mapped=9000, anon=6000, shmem_huge=7000,
                            active_file=4000, inactive_anon=5000)
                 + "file 500\nshmem 100\n")
    assert discovery.memory_limits(cg) == (1000, 400)
