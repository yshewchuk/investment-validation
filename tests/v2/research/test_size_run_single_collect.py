"""SIZE run collects outer results once; stage timers reach stderr."""
import pytest

from experiments import size_search, size_search_report, v2_magnitude_search_run
from experiments.size_search_report import collect
from experiments.v2_magnitude_search_run import run_search
from tests.v2.research.test_size_search_integration import case, completed_case  # noqa: F401

pytestmark = pytest.mark.xdist_group("size_search_integration")


def counting_collects(monkeypatch):
    calls = []
    original = size_search_report.collect_outer_results

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    for module in (v2_magnitude_search_run, size_search_report):
        monkeypatch.setattr(module, "collect_outer_results", counted)
    return calls


def test_run_search_collect_results_false_returns_none_without_collecting(completed_case, monkeypatch):
    calls = counting_collects(monkeypatch)
    result = run_search(*completed_case.context, limits=completed_case.request["limits"]["child"],
                        collect_results=False)
    assert result is None
    assert calls == []


def test_run_collects_outer_results_exactly_once(completed_case, monkeypatch):
    calls = counting_collects(monkeypatch)
    real_read = size_search.read_record

    def fake_read(conn, store, run_id, slot):
        return None if "terminal:" in slot else real_read(conn, store, run_id, slot)

    seen = []

    def fake_run_search(*context, **kwargs):
        seen.append(kwargs)
        return None

    monkeypatch.setattr(size_search, "read_record", fake_read)
    monkeypatch.setattr(size_search, "run_search", fake_run_search)
    monkeypatch.setattr(size_search, "remaining", lambda *args: 1)
    monkeypatch.setattr(size_search.signal, "alarm", lambda seconds=0: 0)
    result = size_search._run(completed_case.context)
    assert seen == [{"limits": completed_case.request["limits"]["child"], "collect_results": False}]
    assert len(calls) == 1
    assert result == collect(*completed_case.context)


def test_collect_logs_stage_timings(completed_case, capsys):
    collect(*completed_case.context)
    error = capsys.readouterr().err
    assert "[stage] collect start" in error
    assert "[stage] collect done in" in error
