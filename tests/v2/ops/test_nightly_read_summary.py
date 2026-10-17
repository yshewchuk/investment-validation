"""Tests for #754 slice 2: ReadTracker, worker_progress, supervisor, producer wiring."""
from __future__ import annotations

import builtins
import io
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from engine.v2.ops import nightly_read_summary, supervisor, worker_progress
from engine.v2.ops.nightly_read_summary import ReadTracker
from engine.v2.ops.nightly_raw_row_producer import worker_progress as nrp_progress


def _counters(snapshots):
    class _C:
        def __init__(self):
            self._i = 0
            self._snapshots = snapshots

        def snapshot(self):
            val = self._snapshots[self._i]
            self._i = min(self._i + 1, len(self._snapshots) - 1)
            return val
    return _C()


def test_delta_and_truncation():
    c = _counters([
        {"a": {"rows_decoded": 10, "bytes_read": 100}},
        {"a": {"rows_decoded": 15, "bytes_read": 150},
         "b": {"rows_decoded": 5, "bytes_read": 50},
         "c": {"rows_decoded": 3, "bytes_read": 30},
         "d": {"rows_decoded": 4, "bytes_read": 40}},
        {"a": {"rows_decoded": 15, "bytes_read": 150},
         "b": {"rows_decoded": 5, "bytes_read": 50},
         "c": {"rows_decoded": 3, "bytes_read": 30},
         "d": {"rows_decoded": 4, "bytes_read": 40}},
    ])
    t = ReadTracker(c)
    d1 = t.delta()
    assert d1["a"] == {"rows_decoded": 5, "bytes_read": 50}
    d2 = t.delta()
    assert d2 == {}


def test_delta_top3_truncation_exact_dict():
    c = _counters([
        {"t": {"rows_decoded": 0, "bytes_read": 0}},
        {"a": {"rows_decoded": 9, "bytes_read": 90},
         "b": {"rows_decoded": 7, "bytes_read": 70},
         "c": {"rows_decoded": 5, "bytes_read": 50},
         "d": {"rows_decoded": 3, "bytes_read": 30}},
    ])
    t = ReadTracker(c)
    assert t.delta() == {
        "a": {"rows_decoded": 9, "bytes_read": 90},
        "b": {"rows_decoded": 7, "bytes_read": 70},
        "c": {"rows_decoded": 5, "bytes_read": 50},
    }


def test_record_keeps_10_slowest():
    c = _counters([{}])
    t = ReadTracker(c)
    for i in range(12):
        t.record(f"s{i}", float(i), {"t": {"rows_decoded": i}})
    assert len(t._steps) == 10
    assert t._steps[0]["seconds"] == 11.0
    assert t._steps[-1]["seconds"] == 2.0
    assert [(e["step"], e["seconds"], e["reads"]) for e in t._steps] == [
        (f"s{i}", float(i), {"t": {"rows_decoded": i}}) for i in range(11, 1, -1)]


def test_summary_amplification_and_none_when_zero():
    c = _counters([{
        "t": {"rows_decoded": 10, "rows_returned": 2, "bytes_read": 100,
              "scans": 1, "fragments_opened": 1, "seconds": 0.1},
        "z": {"rows_decoded": 5, "rows_returned": 0, "bytes_read": 50,
              "scans": 1, "fragments_opened": 1, "seconds": 0.2},
    }])
    t = ReadTracker(c)
    assert t.summary()["tables"] == {
        "t": {"scans": 1, "fragments_opened": 1, "rows_decoded": 10,
              "rows_returned": 2, "bytes_read": 100, "seconds": 0.1,
              "amplification": 5.0},
        "z": {"scans": 1, "fragments_opened": 1, "rows_decoded": 5,
              "rows_returned": 0, "bytes_read": 50, "seconds": 0.2,
              "amplification": None},
    }


def test_summary_complete_payload():
    c = _counters([{
        "t": {"rows_decoded": 10, "rows_returned": 2, "bytes_read": 100,
              "scans": 1, "fragments_opened": 1, "seconds": 0.1},
    }])
    t = ReadTracker(c)
    t.record("a", 1.5, {"t": {"rows_decoded": 1}})
    s = t.summary()
    assert s["tables"] == {
        "t": {"scans": 1, "fragments_opened": 1, "rows_decoded": 10,
              "rows_returned": 2, "bytes_read": 100, "seconds": 0.1,
              "amplification": 5.0},
    }
    assert s["slowest_steps"] == [
        {"step": "a", "seconds": 1.5, "reads": {"t": {"rows_decoded": 1}}}]
    assert isinstance(s["wall_seconds"], float) and s["wall_seconds"] >= 0
    assert isinstance(s["cpu_seconds"], float) and s["cpu_seconds"] >= 0
    assert s["unavailable"] == []


def test_process_read_bytes_parses_rchar(monkeypatch):
    payload = io.StringIO("rchar: 4096\nwchar: 1\n")
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):
        if path == "/proc/self/io":
            return io.StringIO("rchar: 4096\nwchar: 1\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    c = _counters([{}])
    assert ReadTracker(c).summary()["process_read_bytes"] == 4096


def test_counters_that_raise():
    class _Boom:
        def snapshot(self):
            raise RuntimeError("boom")
    t = ReadTracker(_Boom())
    assert t.delta() == {}
    s = t.summary()
    assert s["tables"] is None
    assert "tables" in s["unavailable"]
    assert isinstance(s["wall_seconds"], float)
    assert isinstance(s["cpu_seconds"], float)
    t.record("s", 1.0, {})
    assert s["slowest_steps"] == []
    assert t.summary()["slowest_steps"] == [
        {"step": "s", "seconds": 1.0, "reads": {}}]
    t2 = ReadTracker(None)
    assert t2.delta() == {}
    assert "unavailable" in t2.summary()


def test_worker_progress_records_reads(tmp_path):
    worker_progress.configure(tmp_path)
    try:
        worker_progress.step_end("a", reads={"t": {"rows_decoded": 1}})
        worker_progress.step_end("b", units=1)
    finally:
        worker_progress.reset()
    lines = (tmp_path / "diagnostics" / "steps.ndjson").read_text().splitlines()
    recs = [json.loads(line) for line in lines]
    assert recs[0]["reads"] == {"t": {"rows_decoded": 1}}
    assert "reads" not in recs[1]


def test_supervisor_apply_step_record():
    from engine.v2.ops.supervisor import Service

    calls = []
    stub = SimpleNamespace(_emit_progress=lambda running, **kw: calls.append(kw))
    state = SimpleNamespace(current_step=None, step_peaks={}, step_started_at={})
    Service._apply_step_record(stub, None, state,
                               {"event": "end", "step": "x", "reads": {"t": {"rows_decoded": 5}}})
    assert calls[-1]["step_reads"] == {"t": {"rows_decoded": 5}}
    calls.clear()
    Service._apply_step_record(stub, None, state, {"event": "end", "step": "y", "reads": "x"})
    assert calls[-1]["step_reads"] is None


def test_emit_progress_persists_step_reads(monkeypatch):
    captured = []
    monkeypatch.setattr(supervisor, "record_progress",
                        lambda conn, event: captured.append(event))
    stub = SimpleNamespace(
        clock=SimpleNamespace(now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc)),
        _next_sequence=lambda attempt_id: 0, conn=None)
    running = SimpleNamespace(
        claim=SimpleNamespace(job_id="j", attempt_id="a",
                              spec=SimpleNamespace(kind="k")))
    supervisor.Service._emit_progress(
        stub, running, kind="progress", message="m", elapsed_seconds=1.0,
        step="s", step_reads={"t": 1})
    assert captured and captured[-1].step_reads == {"t": 1}


def _repo_events(tmp_path, monkeypatch, **build_kwargs):
    from tests.v2.ops import test_nightly_raw_row_producer as producer_tests
    return producer_tests._snapshot(tmp_path, [
        producer_tests._event_row("AAA", producer_tests._MIDNIGHT, "BMO",
                                  event_id="aaa-midnight"),
        producer_tests._event_row("BBB", producer_tests._MIDNIGHT, "BMO",
                                  event_id="bbb-midnight"),
    ])


def _rig(tmp_path, monkeypatch):
    from tests.v2.ops import test_nightly_raw_row_producer as producer_tests

    repository, snapshot = _repo_events(tmp_path, monkeypatch)
    rig = producer_tests._ReaderRig(monkeypatch)
    return repository, snapshot, rig


def test_producer_records_step_reads_and_summary(tmp_path, monkeypatch):
    repository, snapshot, rig = _rig(tmp_path, monkeypatch)
    from engine.v2.ops import nightly_raw_row_producer as nrp

    records = []
    real_start, real_end = nrp.worker_progress.step_start, nrp.worker_progress.step_end

    def start(name):
        records.append(("start", name, None, None))

    def end(name, *, units=None, reads=None):
        records.append(("end", name, units, reads))

    monkeypatch.setattr(nrp.worker_progress, "step_start", start)
    monkeypatch.setattr(nrp.worker_progress, "step_end", end)
    events, _ = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=producer_as_of(), horizon_days=producer_horizon())

    compose_ends = [r for r in records
                    if r[0] == "end" and r[1].startswith("compose_event_")]
    assert compose_ends
    for _event, name, units, reads in compose_ends:
        assert units == 1
        assert isinstance(reads, dict)
    summary_ends = [r for r in records if r[0] == "end" and r[1] == "read_summary"]
    assert len(summary_ends) == 1
    assert isinstance(summary_ends[0][3], dict)
    assert "tables" in summary_ends[0][3]
    assert summary_ends[0][3]["wall_seconds"] is not None


def producer_as_of():
    from tests.v2.ops import test_nightly_raw_row_producer as producer_tests
    return producer_tests._AS_OF


def producer_horizon():
    from tests.v2.ops import test_nightly_raw_row_producer as producer_tests
    return producer_tests._HORIZON_DAYS


def test_producer_summary_recorded_on_compose_exception(tmp_path, monkeypatch):
    repository, snapshot, rig = _rig(tmp_path, monkeypatch)
    from engine.v2.ops import nightly_raw_row_producer as nrp

    records = []
    calls = {"n": 0}
    real = nrp._compose_event

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("compose failed")
        return real(*args, **kwargs)

    monkeypatch.setattr(nrp, "_compose_event", boom)
    monkeypatch.setattr(nrp.worker_progress, "step_start",
                        lambda name: records.append(("start", name)))
    monkeypatch.setattr(nrp.worker_progress, "step_end",
                        lambda name, *, units=None, reads=None:
                        records.append(("end", name, units, reads)))

    with pytest.raises(RuntimeError, match="compose failed"):
        nrp.build_native_score_batch_events(
            repository, snapshot, as_of=producer_as_of(),
            horizon_days=producer_horizon())
    assert any(r[0] == "end" and r[1] == "read_summary" for r in records)


def test_producer_unaffected_by_read_counters_failure(tmp_path, monkeypatch):
    repository, snapshot, rig = _rig(tmp_path, monkeypatch)
    from engine.v2.ops import nightly_raw_row_producer as nrp

    normal = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=producer_as_of(), horizon_days=producer_horizon())

    class Boom:
        def snapshot(self):
            raise RuntimeError("no counters")

        def add(self, *args, **kwargs):
            return None
    monkeypatch.setattr(repository, "read_counters", Boom(), raising=False)
    degraded = nrp.build_native_score_batch_events(
        repository, snapshot, as_of=producer_as_of(), horizon_days=producer_horizon())
    assert degraded == normal