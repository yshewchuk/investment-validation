"""Tests for #754 slice 2: ReadTracker, worker_progress, supervisor, producer wiring."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engine.v2.ops import nightly_read_summary, worker_progress
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


def test_delta_top3_truncation():
    c = _counters([
        {"t": {"rows_decoded": 0, "bytes_read": 0}},
        {"a": {"rows_decoded": 9, "bytes_read": 90},
         "b": {"rows_decoded": 7, "bytes_read": 70},
         "c": {"rows_decoded": 5, "bytes_read": 50},
         "d": {"rows_decoded": 3, "bytes_read": 30}},
    ])
    t = ReadTracker(c)
    d = t.delta()
    assert list(d) == ["a", "b", "c"]


def test_record_keeps_10_slowest():
    c = _counters([{}])
    t = ReadTracker(c)
    for i in range(12):
        t.record(f"s{i}", float(i), {})
    assert len(t._steps) == 10
    assert t._steps[0]["seconds"] == 11.0
    assert t._steps[-1]["seconds"] == 2.0


def test_summary_amplification_and_none_when_zero():
    c = _counters([{
        "t": {"rows_decoded": 10, "rows_returned": 2, "bytes_read": 100,
              "scans": 1, "fragments_opened": 1, "seconds": 0.1},
        "z": {"rows_decoded": 5, "rows_returned": 0, "bytes_read": 50,
              "scans": 1, "fragments_opened": 1, "seconds": 0.2},
    }])
    t = ReadTracker(c)
    s = t.summary()
    assert s["tables"]["t"]["amplification"] == 5.0
    assert s["tables"]["z"]["amplification"] is None


def test_counters_that_raise():
    class _Boom:
        def snapshot(self):
            raise RuntimeError("boom")
    t = ReadTracker(_Boom())
    assert t.delta() == {}
    assert "unavailable" in t.summary()
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