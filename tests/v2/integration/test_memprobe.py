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
            self.cache = {i: "x" * 100 for i in range(5000)}

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
    assert re.search(r"cache test\.cache: 5000 entries ~(?!0\.0M)[0-9.]+M", out)
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


def test_estimate_size_set_of_uniform_tuples_exact():
    import engine.memprobe as memprobe

    s = {(i + 10**6, i + 2 * 10**6) for i in range(20000)}
    n = 20000
    t0 = next(iter(s))
    expected = sys.getsizeof(s) + n * (
        sys.getsizeof(t0) + sys.getsizeof(t0[0]) + sys.getsizeof(t0[1])
    )
    estimate = memprobe.estimate_size(s)
    assert 0.9 * expected <= estimate <= 1.1 * expected


def test_estimate_size_list_of_uniform_int_lists_exact():
    import engine.memprobe as memprobe

    data = [[i + 10**6 for i in range(100)] for _ in range(1000)]
    inner = data[0]
    expected = sys.getsizeof(data) + 1000 * (
        sys.getsizeof(inner) + 100 * sys.getsizeof(inner[0])
    )
    estimate = memprobe.estimate_size(data)
    assert 0.9 * expected <= estimate <= 1.1 * expected


def test_estimate_size_dict_of_uniform_int_to_tuple_exact():
    import engine.memprobe as memprobe

    data = {i + 10**6: (i + 10**6, i + 2 * 10**6) for i in range(5000)}
    n = 5000
    k0, v0 = next(iter(data.items()))
    expected = sys.getsizeof(data) + n * (
        sys.getsizeof(k0) + sys.getsizeof(v0) + 2 * sys.getsizeof(v0[0])
    )
    estimate = memprobe.estimate_size(data)
    assert 0.9 * expected <= estimate <= 1.1 * expected


def test_estimate_size_never_raises_on_len_that_raises():
    import engine.memprobe as memprobe

    class Bad(list):
        def __len__(self):
            raise ValueError

    assert memprobe.estimate_size(Bad([1, 2, 3])) == 0

    class Boom(list):
        def __len__(self):
            raise MemoryError

    with pytest.raises(MemoryError):
        memprobe.estimate_size(Boom([1]))


def test_estimate_size_set_200k_under_one_second():
    import engine.memprobe as memprobe

    big = {(i, i + 1) for i in range(200000)}
    started = time.perf_counter()
    estimate = memprobe.estimate_size(big)
    elapsed = time.perf_counter() - started
    assert estimate > 0
    assert estimate > 200_000 * 100
    assert elapsed < 1.0


def test_estimate_size_series_of_uniform_strings_within_bounds():
    pd = pytest.importorskip("pandas")
    pytest.importorskip("numpy")
    import engine.memprobe as memprobe

    s = pd.Series([f"{i:0200d}" for i in range(1000)])
    s0 = s.iloc[0]
    # estimator adds pointer-array bytes twice (memory_usage + getsizeof each)
    expected = 1000 * (sys.getsizeof(s0) + 8) + sys.getsizeof(s.index)
    expected *= 1.8
    estimate = memprobe.estimate_size(s)
    assert 0.9 * expected <= estimate <= 1.1 * expected


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


def test_nightly_mark_elapsed_and_timeline_flag_on_off(monkeypatch, capsys):
    from engine.dashboard.nightly import NightlyReport

    times = iter([100.5, 102.0])

    def _fake_time():
        return next(times)

    monkeypatch.setattr("engine.dashboard.nightly.time.time", _fake_time)
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    r = NightlyReport(as_of="2026-01-01")
    r.start(100.0)
    ret_a = r.mark("a", started=100.0)
    ret_b = r.mark("b", started=100.0)
    assert ret_a == 0.5
    assert ret_b == 1.5
    assert [round(e["elapsed_s"], 1) for e in r.timeline] == [0.5, 1.5]
    assert [round(e["at_s"], 1) for e in r.timeline] == [0.5, 2.0]
    out = capsys.readouterr().out
    assert "[mem] phase a" in out
    assert "phase b" in out

    times = iter([100.5, 102.0])
    monkeypatch.delenv("INVESTING_PLAN_MEMPROFILE", raising=False)
    r = NightlyReport(as_of="2026-01-01")
    r.start(100.0)
    ret_a = r.mark("a", started=100.0)
    ret_b = r.mark("b", started=100.0)
    assert ret_a == 0.5
    assert ret_b == 1.5
    assert [round(e["elapsed_s"], 1) for e in r.timeline] == [0.5, 1.5]
    assert [round(e["at_s"], 1) for e in r.timeline] == [0.5, 2.0]
    out = capsys.readouterr().out
    assert out == ""


def test_watch_nested_float_dicts_reports_nonzero_size(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TOP", "1000000")

    class Holder:
        def __init__(self):
            self.cache = {i: {"a": 1.0, "b": 2.0} for i in range(3000)}

    holder = Holder()
    memprobe.watch("test.nested", holder, "cache")
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert re.search(r"cache test\.nested: 3000 entries ~(?!0\.0M)[0-9.]+M", out)


def test_disabled_path_never_walks_gc_objects(monkeypatch):
    import engine.memprobe as memprobe

    count = 0

    def counting():
        nonlocal count
        count += 1
        return []

    monkeypatch.setattr(gc, "get_objects", counting)
    monkeypatch.delenv("INVESTING_PLAN_MEMPROFILE", raising=False)
    memprobe.snapshot("x")
    memprobe.tick("x", 250)
    assert count == 0