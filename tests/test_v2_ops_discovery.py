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


def _sample(tmp_path, monkeypatch, cg):
    monkeypatch.setattr(discovery, "cgroup_directory", lambda proc: cg)
    return discovery.sample_capacity(tmp_path, clock=SystemClock(), proc=_proc(tmp_path))


# -- memory_limits: working set ------------------------------------------------


def test_inactive_file_yields_working_set_usage(tmp_path):
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
                                  "inactive_file\n", "inactive_file 1 2\n"])
def test_malformed_inactive_file_falls_back_to_raw_current(tmp_path, stat):
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


def test_headroom_rises_by_inactive_file_only(tmp_path, monkeypatch):
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
