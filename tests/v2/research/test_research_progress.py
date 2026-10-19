"""Progress logging (#953): bounded cadence, stderr only; pinned build is output-identical and cheaper."""
import pandas as pd
import pytest

from engine.v2.data.repository import Repository
from engine.v2.research import _progress
from tests.v2.research.magnitude_trial_support import (
    HISTORY_REQUEST,
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
)
from experiments import magnitude_history_store as entry


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_progress_cadence_is_bounded_by_interval_not_items():
    clock, lines = Clock(), []
    progress = _progress.Progress("x", 1000, min_interval=30, clock=clock, emit=lines.append)
    for _ in range(1000):
        clock.now += 1.0
        progress.tick()
    assert len(lines) == 33  # one per 30 s of the 1000 s pass, never one per item
    for number, line in enumerate(lines, start=1):
        done = number * 30
        assert line.startswith(f"  [x] {done:,}/1,000, {done}s elapsed, 1.00/s, ETA {1000 - done}s, peak RSS ")


def test_progress_is_silent_for_short_passes_and_honours_min_items():
    clock, lines = Clock(), []
    short = _progress.Progress("x", 5, min_interval=30, clock=clock, emit=lines.append)
    for _ in range(5):
        clock.now += 1.0
        short.tick()
    assert lines == []
    gated = _progress.Progress("y", 100, min_interval=1, min_items=10, clock=clock, emit=lines.append)
    for _ in range(25):
        clock.now += 5.0
        gated.tick()
    assert len(lines) == 2 and lines[0].startswith("  [y] 10/100")


def test_stage_logs_start_done_and_failure_and_reraises():
    clock, lines = Clock(), []
    with _progress.stage("a", clock=clock, emit=lines.append):
        clock.now += 3
    assert len(lines) == 2 and lines[0].startswith("  [a] start, peak RSS ")
    assert lines[1].startswith("  [a] done in 3s, peak RSS ")
    lines.clear()
    with pytest.raises(RuntimeError, match="^boom$"):
        with _progress.stage("b", clock=clock, emit=lines.append):
            raise RuntimeError("boom")
    assert lines[0].startswith("  [b] start") and lines[1].startswith("  [b] failed after 0s")


def test_quiet_stage_prints_only_when_slow():
    clock, lines = Clock(), []
    with _progress.stage("q", quiet_under=5.0, clock=clock, emit=lines.append):
        clock.now += 4
    assert lines == []
    with _progress.stage("q", quiet_under=5.0, clock=clock, emit=lines.append):
        clock.now += 6
    assert len(lines) == 1 and lines[0].startswith("  [q] done in 6s")


def test_emit_failure_never_breaks_a_pass(monkeypatch):
    attempts = []

    def broken(*args, **kwargs):
        attempts.append(args)
        raise OSError("closed")
    monkeypatch.setattr("builtins.print", broken)
    with _progress.stage("c"):
        result = 7
    assert result == 7 and len(attempts) == 2  # start and done were both attempted


def _source(tmp_path):
    days = [d.date().isoformat() for d in pd.date_range("2022-06-02", "2023-05-30", freq="BMS")][:8]
    events = [history_event(day) for day in (*days, "2024-06-17")]
    moves = [outcome_move(e, float(n + 1)) for n, e in enumerate(events[:-1])] + [outcome_move(events[-1], None)]
    conn, repository, snapshot = commit_history_inputs(
        tmp_path, events, moves, days=calendar_sessions("2022-05-25", "2024-06-18").days)
    return conn, repository, snapshot


def test_pinned_build_is_output_identical_cheaper_and_logs_only_to_stderr(tmp_path, monkeypatch, capsys):
    decodes = []
    original = Repository._record

    def counted(self, *args, **kwargs):
        decodes.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Repository, "_record", counted)
    results = {}
    for name, pinned in (("plain", False), ("pinned", True)):
        conn, repository, snapshot = _source(tmp_path / name)
        try:
            repository = Repository(conn, repository._store, pin_snapshots=pinned)
            decodes.clear()
            capsys.readouterr()
            build = entry.build(repository, snapshot, **HISTORY_REQUEST)
            captured = capsys.readouterr()
            stored = entry.read(repository, snapshot,
                                repository.resolve(build["feature_snapshot"]["snapshot_id"]), **HISTORY_REQUEST)
            results[name] = (len(decodes), build["context"], stored["rows"], captured)
        finally:
            conn.close()
    assert results["pinned"][1] == results["plain"][1]
    assert results["pinned"][2] == results["plain"][2]
    assert results["plain"][2] and any(row["n_prior"] > 0 for row in results["plain"][2])
    assert results["pinned"][0] < results["plain"][0] / 2
    for _, _, _, captured in results.values():
        assert captured.out == ""
        for line in ("[history preflight] start", "[history preflight] done", "[history prepare] done",
                     "[history materialize] start", "[history materialize] done"):
            assert line in captured.err
