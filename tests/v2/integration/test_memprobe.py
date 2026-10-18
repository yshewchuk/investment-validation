"""engine.memprobe: opt-in memory diagnostics are silent when off and report caches when on.

# packages: foundation
"""
from __future__ import annotations

import gc
import re
import sys
import time
import types

import pytest


# --------------------------------------------------------------------------
# disabled by default
# --------------------------------------------------------------------------


def test_disabled_by_default_snapshot_and_tick_print_nothing(monkeypatch, capsys):
    import engine.memprobe as memprobe

    def boom():
        raise AssertionError("gc.get_objects must not be walked when disabled")

    monkeypatch.setattr(gc, "get_objects", boom)
    monkeypatch.delenv("INVESTING_PLAN_MEMPROFILE", raising=False)
    memprobe.snapshot("x")
    memprobe.tick("x", 250)
    out = capsys.readouterr().out
    assert out == ""
    assert not memprobe.enabled()


# --------------------------------------------------------------------------
# enabled: output shape
# --------------------------------------------------------------------------


def test_enabled_snapshot_prints_rss_line_and_watch_and_dataframe(monkeypatch, capsys):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TOP", "1000000")

    class Holder:
        def __init__(self):
            self.cache = {i: i for i in range(5000)}

    holder = Holder()
    memprobe.watch("test.cache", holder, "cache")
    df = pd.DataFrame(
        {
            "a": range(2000),
            "b": range(2000),
            "c": range(2000),
        }
    )
    assert df.shape == (2000, 3)
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "[mem] x rss=" in out
    assert "cache test.cache: 5000 entries" in out
    assert re.search(r"cache test\.cache: 5000 entries ~[0-9.]+M", out)
    assert "DataFrame" in out and "2000 x 3" in out
    assert re.search(r"obj [0-9.]+M DataFrame len=2000 x 3", out)
    assert df is not None


def test_owner_lifetime_watch_line_present_then_absent(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TOP", "1000000")

    class Owner:
        def __init__(self):
            self.bucket = list(range(2000))

    o = Owner()
    memprobe.watch("owner.alive", o, "bucket")
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "cache owner.alive: 2000 entries" in out

    del o
    gc.collect()
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "owner.alive" not in out


def test_two_live_owners_same_name_second_suffixed_and_deletion_keeps_one(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TOP", "1000000")

    class Owner:
        def __init__(self):
            self.bucket = list(range(1500))

    o1 = Owner()
    o2 = Owner()
    memprobe.watch("owner.dup", o1, "bucket")
    memprobe.watch("owner.dup", o2, "bucket")
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "cache owner.dup: 1500 entries" in out
    assert "cache owner.dup#2: 1500 entries" in out

    del o2
    gc.collect()
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "cache owner.dup: 1500 entries" in out
    assert "owner.dup#2" not in out


# --------------------------------------------------------------------------
# tick cadence
# --------------------------------------------------------------------------


def test_tick_fires_only_on_multiples_of_every(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_EVERY", "3")
    for i in range(1, 8):
        memprobe.tick("t", i)
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("  [mem] t")]
    labels = {re.search(r"\[mem\] (t \d+)", line).group(1) for line in lines}
    assert labels == {"t 3", "t 6"}
    assert len(lines) == 2


# --------------------------------------------------------------------------
# trim flag
# --------------------------------------------------------------------------


def test_trim_flag_calls_gc_then_malloc_trim_once_per_snapshot(monkeypatch, capsys):
    import engine.memprobe as memprobe

    calls: list[str] = []
    real_gc = gc.collect
    real_trim = memprobe._malloc_trim

    def gc_wrapper():
        calls.append("gc")
        return real_gc()

    def trim_wrapper():
        calls.append("trim")
        return real_trim()

    monkeypatch.setattr(gc, "collect", gc_wrapper)
    monkeypatch.setattr(memprobe, "_malloc_trim", trim_wrapper)
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TRIM", "1")
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert calls == ["gc", "trim"]
    assert re.search(r"trimmed from [0-9.]+G", out)


def test_trim_flag_off_never_calls_malloc_trim(monkeypatch, capsys):
    import engine.memprobe as memprobe

    calls: list[int] = []
    monkeypatch.setattr(memprobe, "_malloc_trim", lambda: calls.append(1))
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.delenv("INVESTING_PLAN_MEMPROFILE_TRIM", raising=False)
    memprobe.snapshot("x")
    capsys.readouterr()
    assert calls == []


# --------------------------------------------------------------------------
# estimate_size
# --------------------------------------------------------------------------


def test_estimate_size_dict_of_tuples_within_half_to_double(monkeypatch):
    import engine.memprobe as memprobe

    data = {(i, i + 1): (i, i + 1) for i in range(10000)}

    def exact(o):
        if isinstance(o, dict):
            return sys.getsizeof(o) + sum(exact(k) + exact(v) for k, v in o.items())
        if isinstance(o, (tuple, list)):
            return sys.getsizeof(o) + sum(exact(x) for x in o)
        if isinstance(o, set):
            return sys.getsizeof(o) + sum(exact(x) for x in o)
        return sys.getsizeof(o)

    exact_total = exact(data)
    estimate = memprobe.estimate_size(data)
    assert 0.5 * exact_total <= estimate <= 2 * exact_total


def test_estimate_size_never_raises_on_len_that_raises():
    import engine.memprobe as memprobe

    class Bad:
        def __len__(self):
            raise ValueError("nope")

    assert isinstance(memprobe.estimate_size(Bad()), int)


def test_estimate_size_set_200k_under_one_second():
    import engine.memprobe as memprobe

    big = {(i, i + 1) for i in range(200000)}
    started = time.perf_counter()
    estimate = memprobe.estimate_size(big)
    elapsed = time.perf_counter() - started
    assert estimate > 0
    assert elapsed < 1.0


def test_estimate_size_series_of_1000_strings_exceeds_150k():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import engine.memprobe as memprobe

    s = pd.Series(["x" * 200 for _ in range(1000)])
    assert memprobe.estimate_size(s) > 150_000


def test_estimate_size_nested_dict_of_lists_extrapolates():
    import engine.memprobe as memprobe

    data = {i: list(range(1000)) for i in range(50)}
    estimate = memprobe.estimate_size(data)
    assert estimate >= 0.5 * (50 * 1000 * 8)


# --------------------------------------------------------------------------
# failure swallowing
# --------------------------------------------------------------------------


def test_snapshot_swallows_top_objects_failure(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")

    def boom(n):
        raise ValueError("boom")

    monkeypatch.setattr(memprobe, "top_objects", boom)
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "probe failed: ValueError" in out
    assert "boom" in out


def test_snapshot_propagates_memory_error(monkeypatch):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")

    def boom(n):
        raise MemoryError()

    monkeypatch.setattr(memprobe, "top_objects", boom)
    with pytest.raises(MemoryError):
        memprobe.snapshot("x")


# --------------------------------------------------------------------------
# NightlyReport.mark hook
# --------------------------------------------------------------------------


def test_nightly_mark_returns_elapsed_matching_timeline_flag_off(monkeypatch, capsys):
    from engine.dashboard.nightly import NightlyReport

    monkeypatch.delenv("INVESTING_PLAN_MEMPROFILE", raising=False)
    r = NightlyReport(as_of="2026-01-01")
    r.start(0.0)
    time.sleep(0.05)
    ret = r.mark("settle", started=0.0)
    assert ret == pytest.approx(r.timeline[0]["elapsed_s"], abs=0.06)
    out = capsys.readouterr().out
    assert out == ""
    assert r.timeline[0]["step"] == "settle"


def test_nightly_mark_on_prints_mem_phase_line(monkeypatch, capsys):
    from engine.dashboard.nightly import NightlyReport

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    r = NightlyReport(as_of="2026-01-01")
    r.start(0.0)
    time.sleep(0.05)
    ret = r.mark("settle", started=0.0)
    assert ret == pytest.approx(r.timeline[0]["elapsed_s"], abs=0.06)
    out = capsys.readouterr().out
    assert "[mem] phase settle" in out
    assert r.timeline[0]["step"] == "settle"