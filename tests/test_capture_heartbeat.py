"""Focused tests for the capture heartbeat (tools/capture_heartbeat.py) and
its wiring into tools/capture_tier0_corpus.py: start-of-run ETA (fake clock),
heartbeat lines emitted while ONE unit blocks (real short-interval thread),
completed-count -> observed ETA updates, cleanup on failure (no orphan
thread), and the invariant the whole spec rests on -- an instrumented run
writes a byte- and hash-identical corpus. Fully synthetic: no real Scorer,
no panel, no chain table, no full capture.
"""
from __future__ import annotations

import re
import threading
import time

import pandas as pd
import pytest

import tools.capture_heartbeat as heartbeat_mod
import tools.capture_tier0_corpus as capture
from tools.capture_heartbeat import CaptureHeartbeat

from tests.test_capture_tier0_release_scorer import _FakeScorer, _install_empty_pipeline
from tests.test_phase4_capture_writer import _candidate as _traced_candidate


class FakeClock:
    """A monotonic clock a test drives directly."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _heartbeat(**overrides) -> CaptureHeartbeat:
    kwargs = dict(interval_seconds=60.0)  # keep fake-clock tests single-threaded
    kwargs.update(overrides)
    return CaptureHeartbeat(**kwargs)


# ---------------------------------------------------------------------------
# start-of-run ETA: the explicitly labeled prior-run fallback
# ---------------------------------------------------------------------------


def test_start_of_run_eta_is_the_labeled_prior_run_fallback():
    """Before ANY unit completes the ETA is the last full strict capture's
    measured split -- gather ~19m + attach/write ~62m -- printed as an
    estimate and named as prior-run, never as this run's measurement."""
    clock = FakeClock()
    lines: list[str] = []
    hb = _heartbeat(clock=clock, emit=lines.append)
    with hb:
        hb.begin_stage("gather")
        hb.phase("build-scorer")
        line = hb.beat()
        assert line is not None
        assert "phase=build-scorer" in line
        assert "elapsed=0m00s" in line
        assert "done=n/a" in line
        assert "eta=~81m" in line  # 19 + 41 + 21: the measured 81-minute sum
        assert "prior-run estimate" in line
        assert heartbeat_mod.PRIOR_STRICT_CAPTURE in line
        assert "observed" not in line

        # The fallback counts down within the stage but never claims the
        # minute: five minutes into the scorer build, ~76m are projected.
        clock.advance(5 * 60)
        assert "elapsed=5m00s" in hb.beat()
        assert "eta=~76m" in hb.beat()

        # The attach stage boundary shows the combined 62 minutes the prior
        # run actually measured for attach+write (41 + 21).
        hb.begin_stage("attach", units_total=29)
        hb.phase("attach-strict-trace", total=29)
        line = hb.beat()
        assert "eta=~62m" in line
        assert "prior-run" in line

    # A stopped heartbeat emits nothing further.
    assert hb.beat() is None


# ---------------------------------------------------------------------------
# heartbeat during a long unit: the thread, not the work loop, carries it
# ---------------------------------------------------------------------------


def test_heartbeat_emits_while_one_unit_blocks():
    """One blocked unit (the Scorer build, a ChainIndex stream, one long
    candidate, one big pair write) never silences the log: with the main
    thread stuck in an Event.wait and ZERO completion calls made, the
    daemon thread still beats at its interval, each line carrying phase,
    elapsed and the ETA label."""
    lines: list[str] = []
    hb = _heartbeat(interval_seconds=0.05, emit=lines.append)
    hb.start()
    try:
        hb.phase("score-forward", total=40)  # the ONE unit about to run
        blocked = threading.Event()
        timer = threading.Timer(0.4, blocked.set)
        timer.start()
        blocked.wait(5.0)  # main thread does nothing at all meanwhile
        timer.join()
    finally:
        hb.stop()
    # >=1 line per (scaled) minute THROUGH the block: ~8 beats in 0.4 s at
    # 0.05 s; assert 4 to survive a loaded CI box without weakening the
    # real guarantee (interval is what production sets to < 60 s). The
    # FIRST line is the t=0 start beat (phase=starting, before the unit
    # was announced); every beat after it carries the blocked unit's phase.
    assert len(lines) >= 4
    for line in lines[1:]:
        assert "phase=score-forward" in line
        assert "done=0/40" in line  # no unit completed -- nothing invented
        assert "eta=" in line and "remaining (" in line
    assert not hb.is_alive
    assert not any(t.name == "capture-heartbeat" for t in threading.enumerate())


# ---------------------------------------------------------------------------
# completed-count updates: observed ETA takes over where reliable
# ---------------------------------------------------------------------------


def test_completed_units_switch_the_eta_to_observed_progress():
    """The write stage knows its exact total (one unit per pair); after two
    pairs the projection is this run's measurement, labeled observed -- and
    ONLY observed, with no prior-run text left on the line."""
    clock = FakeClock()
    lines: list[str] = []
    hb = _heartbeat(clock=clock, emit=lines.append)
    with hb:
        hb.begin_stage("write", units_total=10)
        hb.phase("write-pairs", total=10)
        clock.advance(60)  # first pair took a minute
        hb.advance()
        hb.advance()       # second done at the same measured 30 s/unit pace
        line = hb.beat()
    assert "done=2/10" in line
    # 8 pairs left at (60 s / 2 units) = 30 s -> 240 s -> "~4m", write is the
    # last stage so no prior-run tail rides along.
    assert "eta=~4m" in line
    assert "observed: 2/10 write unit(s) measured this run" in line
    assert "prior-run" not in line


def test_an_observed_earlier_stage_keeps_the_prior_run_tail_labeled():
    """Attach has a measured rate but stages after it never do: the line
    must say which halves are observed and which are still the fallback."""
    clock = FakeClock()
    lines: list[str] = []
    hb = _heartbeat(clock=clock, emit=lines.append)
    with hb:
        hb.begin_stage("attach", units_total=10)
        hb.phase("attach-strict-trace", total=10)
        clock.advance(60)
        hb.advance()
        hb.advance()
        line = hb.beat()
    # 8 attach units at 30 s = 240 s + the prior-run write tail (21 m)
    # = 1500 s -> 25 m.
    assert "eta=~25m" in line
    assert "observed: 2/10 attach unit(s) measured this run" in line
    assert "+ prior-run for later stages" in line


def test_unstarted_heartbeat_is_inert():
    """Every capture function outside ``main`` may call the singleton
    unconditionally: before ``start`` there is no thread, no line and no
    state -- which is what keeps bare ``write``/pass tests byte-identical."""
    lines: list[str] = []
    hb = _heartbeat(emit=lines.append)
    hb.begin_stage("gather")
    hb.phase("x", total=3)
    hb.advance()
    assert lines == []
    assert hb.beat() is None
    assert not hb.is_alive


# ---------------------------------------------------------------------------
# cleanup on failure
# ---------------------------------------------------------------------------


def test_heartbeat_stops_when_the_capture_raises():
    """A unit dying mid-capture must never leave an orphan thread printing
    lines into a dead run's log."""
    lines: list[str] = []
    hb = _heartbeat(interval_seconds=0.05, emit=lines.append)
    with pytest.raises(RuntimeError):
        with hb:
            hb.begin_stage("gather")
            hb.phase("load-chain-index")
            time.sleep(0.12)  # some beats fire inside the doomed unit
            raise RuntimeError("unit died mid-capture")
    assert not hb.is_alive
    assert not any(t.name == "capture-heartbeat" for t in threading.enumerate())
    quiet = len(lines)
    time.sleep(0.15)
    assert len(lines) == quiet  # and it really is silent now
    hb.stop()  # idempotent


def test_main_stops_the_heartbeat_when_a_unit_fails(monkeypatch, tmp_path):
    """The lifecycle guarantee at the real entry point, not just the class:
    a capture that raises from inside a phase exits ``main`` with the
    thread joined and the singleton inert."""
    _install_empty_pipeline(monkeypatch, [{"fixture_id": "case-0", "kind": "score_result",
                                           "request": {}, "record": {}, "duration": 0.1}])
    monkeypatch.setattr(capture.score_mod, "Scorer", lambda: _FakeScorer())

    def _dies(*args, **kwargs):
        raise RuntimeError("store died mid-capture")

    monkeypatch.setattr(capture, "_events", _dies)
    with pytest.raises(RuntimeError):
        capture.main(["--out", str(tmp_path / "out")])

    hb = capture.heartbeat()
    assert hb.running is False
    assert not hb.is_alive
    assert not any(t.name == "capture-heartbeat" for t in threading.enumerate())


def test_restart_begins_fresh_with_no_stale_phase_count_or_observed_eta():
    """A second run on the SAME instance must not inherit the first run's
    phase, done count or observed-rate projection (Astra finding):
    ``start()`` resets ALL per-run state, not just ``_run_started``."""
    clock = FakeClock()
    lines: list[str] = []
    hb = _heartbeat(clock=clock, emit=lines.append)
    with hb:
        hb.begin_stage("write", units_total=4)
        hb.phase("write-pairs", total=4)
        clock.advance(60)
        hb.advance()
        assert "observed" in hb.beat()
    clock.advance(3600)  # hours later, a fresh capture on the same object
    with hb:
        first = hb.beat()
    assert "phase=starting" in first
    assert "done=n/a" in first
    assert "eta=~81m" in first and "prior-run" in first
    assert "observed" not in first


def test_stop_keeps_reporting_an_emitter_blocked_thread_as_alive():
    """Honest lifecycle (Astra finding): ``stop()`` must not drop the
    thread reference before a timed join -- a beat stuck in a blocked
    emitter still IS alive, and pretending otherwise would hide an
    orphan. It goes inert (no new beats scheduled) immediately."""
    gate = threading.Event()
    calls = {"n": 0}

    def emit(line: str) -> None:
        calls["n"] += 1
        if calls["n"] > 1:  # the t=0 beat is emitted by the caller's start()
            gate.wait(5.0)  # the daemon's first periodic beat blocks in the log

    hb = _heartbeat(interval_seconds=0.05, join_grace_seconds=0.05, emit=emit)
    hb.start()
    try:
        deadline = time.monotonic() + 2.0
        while calls["n"] < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert calls["n"] >= 2, "daemon thread never emitted"
        hb.stop()  # join times out while the emitter is stuck
        assert hb.running is False  # inert at once: no MORE beats scheduled
        assert hb.is_alive, "stop() must not claim a live thread is dead"
    finally:
        gate.set()
        deadline = time.monotonic() + 2.0
        while hb.is_alive and time.monotonic() < deadline:
            time.sleep(0.01)
    assert not hb.is_alive  # once it really ends, the truth is visible
    assert hb.beat() is None


# ---------------------------------------------------------------------------
# blocker 1: a blocked FINAL checkpoint write keeps a nonzero whole ETA
# ---------------------------------------------------------------------------


def test_blocked_final_checkpoint_write_keeps_a_nonzero_whole_capture_eta(
        tmp_path, monkeypatch):
    """Each write unit completes only after the candidate's checkpoint CASE
    is durable, and the finalize/publish unit is explicitly outstanding:
    while the LAST ``write_case`` blocks, the line must still show a
    completed count BEHIND the total and a non-zero whole-capture ETA (never
    done=N/N eta=~0m), because this candidate's unit AND the publication
    unit both remain."""
    clock = FakeClock()
    lines: list[str] = []
    hb = _heartbeat(clock=clock, emit=lines.append)
    original = capture.DiskCheckpointSink.write_case
    calls = {"n": 0}
    mid: dict[str, str] = {}

    def slow_write_case(self, case_id, case_document):
        calls["n"] += 1
        if calls["n"] == 2:  # the LAST candidate's checkpoint write blocks
            clock.advance(120)  # the first unit's durable work took 2 minutes
            mid["line"] = hb.beat()
        return original(self, case_id, case_document)

    monkeypatch.setattr(capture.DiskCheckpointSink, "write_case", slow_write_case)
    previous = capture.set_heartbeat(hb)
    try:
        with hb:
            hb.begin_stage("gather")  # exactly what main() does
            doc = capture.write(
                tmp_path / "corpus",
                [_traced_candidate("case-1"), _traced_candidate("case-2")],
                {"strategy:STR-THRU": ["case-1", "case-2"]},
                pd.Timestamp("2026-01-01"), "snap-1",
            )
        final = mid["line"]
    finally:
        capture.set_heartbeat(previous)

    assert calls["n"] == 2  # the blocked seam really is the final checkpoint
    assert "phase=write-pairs" in final
    assert "done=1/2" in final  # second candidate's case NOT yet durable
    assert "~0m" not in final   # finalize/publish unit still outstanding too
    assert "eta=~4m" in final   # 2 units left (case-2 + finalize) x 120s observed
    assert "observed: 1/3 write unit(s) measured this run" in final
    # The run itself is untouched: both cases and the publication completed.
    assert (tmp_path / "corpus" / "checkpoints" / "cases" / "case-2.json").is_file()
    assert doc["diagnostic_checkpoint_manifest"] == "checkpoints/manifest.json"


# ---------------------------------------------------------------------------
# wiring at main(): visible logging and the ETA exist from the first line
# ---------------------------------------------------------------------------


def test_main_prints_heartbeat_lines_with_phase_elapsed_and_eta(monkeypatch, tmp_path,
                                                                 capsys):
    """A fake-pipeline run of ``main`` (same seams the existing release-scorer
    tests use) still emits a heartbeat line -- the spec's "visible logging
    and an ETA whenever the script runs", at t=0, before any unit."""
    _install_empty_pipeline(monkeypatch, [{"fixture_id": "case-0", "kind": "score_result",
                                           "request": {}, "record": {}, "duration": 0.1}])
    monkeypatch.setattr(capture.score_mod, "Scorer", lambda: _FakeScorer())
    monkeypatch.setattr(capture, "write", lambda *a, **k: {
        "corpus_hash": "sha256:" + "0" * 64, "required_axes": [], "uncovered_axes": [],
    })

    rc = capture.main(["--out", str(tmp_path / "out")])

    assert rc == 0
    out = capsys.readouterr().out
    heartbeat_lines = [line for line in out.splitlines() if "heartbeat" in line]
    assert heartbeat_lines, out
    first = heartbeat_lines[0]
    assert "phase=" in first and "elapsed=" in first and "done=" in first
    assert "eta=~81m" in first and "prior-run estimate" in first
    assert heartbeat_mod.PRIOR_STRICT_CAPTURE in first
    assert capture.heartbeat().running is False  # stopped on the success path too


# ---------------------------------------------------------------------------
# the invariant: an instrumented capture writes the identical corpus
# ---------------------------------------------------------------------------


def _synthetic_candidate(fixture_id: str) -> dict:
    return {
        "fixture_id": fixture_id,
        "covers": ["strategy:STR-THRU"],
        "request": {"strategy": "STR-THRU", "ticker": "ABC", "as_of": "2026-01-01"},
        "record": {"strategy": "STR-THRU", "ticker": "ABC", "flags": []},
        "kind": "score_result",
        "duration": 0.1,
    }


#: ``envelope.captured_at``/``worker_ref`` are deliberately wall-clock and
#: per-process (contracts §2.2, excluded from every hash) -- the only two
#: fields two independent writes of the SAME candidate may differ in.
_ENVELOPE = re.compile(r'"(captured_at|worker_ref)": "[^"]*"')


def _tree(root) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _ENVELOPE.sub(r'"\1": "MASKED"', path.read_text())
        for path in sorted(root.rglob("*.json"))
    }


def test_an_instrumented_run_writes_the_identical_corpus(tmp_path):
    """``write()`` driven through a LIVE, fake-clock heartbeat (the singleton
    the capture functions read) must produce the byte-identical fixture tree
    and the identical corpus hash as the same call with the inert one: the
    heartbeat observes the capture, it never joins it."""
    #: Same version-directory name under different parents: the checkpoint
    #: manifest legitimately carries ``release_id`` (= the directory name),
    #: so only the PARENT may differ between the two runs.
    index = {"strategy:STR-THRU": ["000_ABC"]}
    quiet = capture.write(tmp_path / "quiet" / "corpus", [_synthetic_candidate("000_ABC")],
                          dict(index), pd.Timestamp("2026-01-01"), "snap-1")

    lines: list[str] = []
    hb = _heartbeat(clock=FakeClock(), emit=lines.append)
    previous = capture.set_heartbeat(hb)
    try:
        with hb:
            hb.begin_stage("gather")  # exactly what main() does
            loud = capture.write(tmp_path / "loud" / "corpus",
                                 [_synthetic_candidate("000_ABC")],
                                 dict(index), pd.Timestamp("2026-01-01"), "snap-1")
            last = hb.beat()
    finally:
        capture.set_heartbeat(previous)

    assert loud["corpus_hash"] == quiet["corpus_hash"]
    assert _tree(tmp_path / "loud" / "corpus") == _tree(tmp_path / "quiet" / "corpus")
    # And the instrumented run really did carry the capture all the way
    # through finalize/publish: write's stage counts len(chosen)+1 units
    # (pair + the finalization unit), so the post-write snapshot is the
    # fully measured 2/2 with the publication unit done at ~0m.
    assert "phase=finalize-publish" in last and "done=1/1" in last
    assert "observed: 2/2 write unit(s) measured this run" in last
    assert "eta=~0m" in last
