"""Stage-boundary memory logging is bounded, quiet on failure, and never raises."""
import sys

import pytest

from experiments import search_memory_log as log


def test_deep_size_counts_nested_structures_and_handles_cycles():
    small = [1, 2]
    large = {"a": [list(range(50)) for _ in range(20)], "b": {"c": (1, 2, {3})}}
    small_size, _ = log.deep_size(small)
    large_size, large_truncated = log.deep_size(large)
    assert large_size > small_size
    assert large_truncated is False
    cycle = []
    cycle.append(cycle)
    size, truncated = log.deep_size(cycle)
    assert size > 0 and truncated is False
    _, truncated = log.deep_size(large, cap=3)
    assert truncated is True


def test_log_memory_prints_one_line_and_omits_none(capsys):
    log.log_memory("probe", alpha=[1, 2, 3], beta=None, gamma={"k": "v"})
    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    assert "[search memory] probe:" in lines[0]
    assert "VmRSS" in lines[0] and "peak" in lines[0] and "alpha" in lines[0]
    assert "beta" not in lines[0]


def test_deep_size_wide_list_truncates_to_lower_bound():
    wide = list(range(10))
    full_size, full_truncated = log.deep_size(wide)
    size, truncated = log.deep_size(wide, cap=3)
    assert full_truncated is False
    assert truncated is True
    assert size <= full_size


def test_log_memory_propagates_memory_error(monkeypatch):
    def exhausted():
        raise MemoryError("synthetic RSS watchdog trip")
    monkeypatch.setattr(log, "rss_bytes", exhausted)
    with pytest.raises(MemoryError):
        log.log_memory("watchdog", holder=[1])


def test_log_memory_never_raises(capsys, monkeypatch):
    def unreadable():
        raise RuntimeError("synthetic unreadable /proc")
    monkeypatch.setattr(log, "rss_bytes", unreadable)
    log.log_memory("broken", holder=[1])
    assert capsys.readouterr().err == ""


def test_panel_memo_holder_absent(monkeypatch):
    monkeypatch.delitem(sys.modules, "experiments.size_search_binding", raising=False)
    assert log.panel_memo_holder() is None


def test_panel_memo_holder_present():
    import experiments.size_search_binding as binding
    assert log.panel_memo_holder() is binding._PANEL_MEMO
