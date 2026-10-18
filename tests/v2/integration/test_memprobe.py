"""engine.memprobe: opt-in memory diagnostics are silent when off and report caches when on.

# packages: foundation
"""
from __future__ import annotations

import gc
import re
import sys
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
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TOP", "12")

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
    # keep df alive until after snapshot
    assert df is not None


def test_watch_does_not_keep_owner_alive(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")

    class Owner:
        pass

    o = Owner()
    memprobe.watch("owner.gone", o, "missing")
    del o
    gc.collect()
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "owner.gone" not in out


def test_module_string_watch_resolves_via_sys_modules(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    mod = types.ModuleType("throwaway_mod")
    mod.bucket = ["a", "b", "c"]
    sys.modules["throwaway_mod"] = mod
    try:
        memprobe.watch("throwaway.bucket", "throwaway_mod", "bucket")
        memprobe.snapshot("x")
        out = capsys.readouterr().out
        assert "cache throwaway.bucket: 3 entries" in out
    finally:
        del sys.modules["throwaway_mod"]


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
    assert out.count("[mem] t") == 2


# --------------------------------------------------------------------------
# trim flag
# --------------------------------------------------------------------------


def test_trim_flag_calls_malloc_trim_once_per_snapshot(monkeypatch, capsys):
    import engine.memprobe as memprobe

    calls: list[int] = []
    monkeypatch.setattr(memprobe, "_malloc_trim", lambda: calls.append(1))
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE_TRIM", "1")
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert len(calls) == 1
    assert "trimmed from" in out


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


def test_estimate_size_set_200k_is_sampled_not_copied():
    import engine.memprobe as memprobe

    def exact(o):
        if isinstance(o, (set, frozenset)):
            return sys.getsizeof(o) + sum(exact(x) for x in o)
        if isinstance(o, (tuple, list)):
            return sys.getsizeof(o) + sum(exact(x) for x in o)
        return sys.getsizeof(o)

    small = {(i, i + 1) for i in range(20000)}
    exact_small = exact(small)
    big = {(i, i + 1) for i in range(200000)}
    estimate = memprobe.estimate_size(big)
    scaled = exact_small * (len(big) / len(small))
    assert 0.5 * scaled <= estimate <= 2 * scaled


# --------------------------------------------------------------------------
# failure swallowing
# --------------------------------------------------------------------------


def test_snapshot_swallows_top_objects_failure(monkeypatch, capsys):
    import engine.memprobe as memprobe

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    monkeypatch.setattr(memprobe, "top_objects", lambda n: (_ for _ in ()).throw(RuntimeError("x")))
    memprobe.snapshot("x")
    out = capsys.readouterr().out
    assert "probe failed" in out


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


def test_nightly_mark_off_prints_nothing_but_records_timeline(monkeypatch, capsys):
    from engine.dashboard.nightly import NightlyReport

    monkeypatch.delenv("INVESTING_PLAN_MEMPROFILE", raising=False)
    r = NightlyReport(as_of="2026-01-01")
    r.start(0.0)
    r.mark("settle", started=0.0)
    out = capsys.readouterr().out
    assert out == ""
    assert len(r.timeline) == 1
    assert r.timeline[0]["step"] == "settle"


def test_nightly_mark_on_prints_mem_phase_line(monkeypatch, capsys):
    from engine.dashboard.nightly import NightlyReport

    monkeypatch.setenv("INVESTING_PLAN_MEMPROFILE", "1")
    r = NightlyReport(as_of="2026-01-01")
    r.start(0.0)
    r.mark("settle", started=0.0)
    out = capsys.readouterr().out
    assert "[mem] phase settle" in out
    assert len(r.timeline) == 1