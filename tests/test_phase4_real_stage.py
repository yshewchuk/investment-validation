"""Unit tests for ``checks.phase4_real._Stage``, the coarse-stage progress
marker added so a ~70-minute phase4_real.py run that is slow, stuck, or
dying can be told apart from one that is silently working (it emitted
nothing at all between start and finish before this change).

Covers: the START/END print format and stage name, the elapsed-seconds
figure on the END line, and -- the property that matters most, since
instrumentation must never hide a real failure -- that an exception raised
inside the ``with`` block still propagates to the caller rather than being
swallowed, with a FAILED marker printed on the way out instead of silence.
Also covers ``_Stage.progress()`` (the attached per-minute case-count
ticker) and ``_counted``, and that ``_native_parity`` reports every
declared fixture exactly once through it.

Does NOT exercise the real ~70-minute ``build_evidence`` run; that is out of
scope for a unit test (see the hand-back for what stays unverified).
"""
from __future__ import annotations

import io
import re
import time
from types import SimpleNamespace

import pytest

from checks import phase4_real
from checks.phase4_real import _counted, _GateEstimate, _Stage
from checks.tier0_corpus import Progress
from tests.test_phase4_population_exclusion import (
    _pair, stub_replay,  # noqa: F401
)
from tests.test_tier0_corpus import _FakeClock


class _Recorder:
    """Duck-typed ``checks.tier0_corpus.Progress`` consumer: records
    begins, counts ticks, prints nothing."""

    def __init__(self):
        self.begins = []
        self.ticks = 0

    def begin(self, phase, total, unit="cases"):
        self.begins.append((phase, total))

    def tick(self, n=1):
        self.ticks += n


def test_stage_prints_start_then_end_with_stage_name(capsys):
    with _Stage("some_stage"):
        pass
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[phase4_real] START some_stage"
    assert re.match(r"^\[phase4_real\] END some_stage \(\d+\.\d+s\)$", out[1])


def test_stage_reports_elapsed_seconds(capsys, monkeypatch):
    ticks = iter([100.0, 102.5])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    with _Stage("timed"):
        pass
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "[phase4_real] END timed (2.5s)"


def test_stage_reraises_instead_of_swallowing_the_exception(capsys):
    with pytest.raises(ValueError, match="boom"):
        with _Stage("dying_stage"):
            raise ValueError("boom")
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "[phase4_real] START dying_stage"
    assert out[1].startswith("[phase4_real] FAILED dying_stage (")


def test_stage_failed_marker_still_reports_elapsed_seconds(capsys, monkeypatch):
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    with pytest.raises(RuntimeError):
        with _Stage("dying_timed"):
            raise RuntimeError("nope")
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "[phase4_real] FAILED dying_timed (1.2s)"


def test_stage_stores_its_name():
    stage = _Stage("named")
    assert stage.name == "named"


# --------------------------------------------------------------------------
# attached per-minute progress (the ~30-minute native_parity stretch and
# the ~107-minute corpus battery must never go silent for a minute)
# --------------------------------------------------------------------------


def test_attached_progress_emits_case_lines_and_closes_with_the_count(capsys):
    with _Stage("slow") as stage:
        ticker = stage.progress(interval=0.0, heartbeat=False)  # every tick prints
        ticker.begin("fixtures", 2)
        ticker.tick()
        ticker.tick()
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "[phase4_real] START slow"
    assert lines[1].startswith(
        "[phase4_real] PROGRESS slow fixtures 1/2 cases elapsed=")
    assert lines[2].startswith(
        "[phase4_real] PROGRESS slow fixtures 2/2 cases elapsed=")
    assert re.match(
        r"^\[phase4_real\] END slow \(\d+\.\d+s, 2/2 cases\)$", lines[3])


def test_failed_stage_line_carries_the_completed_count(capsys):
    with pytest.raises(RuntimeError):
        with _Stage("dying") as stage:
            ticker = stage.progress(interval=0.0, heartbeat=False)
            ticker.begin("fixtures", 3)
            ticker.tick()
            raise RuntimeError("mid case")
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith("[phase4_real] FAILED dying (")
    assert lines[-1].endswith(", 1/3 cases)")


def test_stage_heartbeat_emits_interim_lines_while_one_unit_blocks(capsys):
    """A single native-parity fixture can take longer than the interval
    between completions: with the stage-owned heartbeat running, the
    operator still sees lines -- all of them honestly saying the in-flight
    case is NOT yet done -- and the timer is gone when the block exits."""
    with _Stage("slow_stage") as stage:
        ticker = stage.progress(interval=0.06)
        ticker.begin("fixtures", 1)
        time.sleep(0.25)  # one unit spanning many heartbeat checks
        ticker.tick()
    lines = capsys.readouterr().out.splitlines()
    interim = [line for line in lines if "PROGRESS" in line]
    assert len(interim) >= 2, lines
    assert any("fixtures 0/1 cases" in line for line in interim)
    assert all("fixtures 0/1 cases" in line
               or "fixtures 1/1 cases" in line for line in interim)
    assert lines[-1].startswith("[phase4_real] END slow_stage (")
    assert lines[-1].endswith(", 1/1 cases)")
    assert not ticker.heartbeat_active()


def test_stage_stops_its_heartbeat_when_the_body_raises(capsys):
    with pytest.raises(RuntimeError):
        with _Stage("dying") as stage:
            ticker = stage.progress(interval=5.0)
            assert ticker.heartbeat_active()
            raise RuntimeError("mid case")
    assert not ticker.heartbeat_active()


def test_long_gate_stages_carry_the_fallback_profile_eta(capsys):
    """The last real Phase 4 run (~2h19m) supplies a provisional ETA from
    stage start -- visible even on the very first line, while no observed
    rate exists yet -- labelled ``eta~=`` so it is never mistaken for the
    observed, phase-scoped ``eta=``."""
    assert phase4_real._STAGE_DURATION_PROFILE_S == {
        "corpus_load": 107 * 60.0, "native_parity": 32 * 60.0,
    }
    with _Stage("corpus_load") as stage:
        ticker = stage.progress(interval=0.0, heartbeat=False)
        assert ticker.stage_seconds == 107 * 60.0
        ticker.begin("load", 20, unit="pairs")
        ticker.tick()
    lines = capsys.readouterr().out.splitlines()
    assert "1/20 pairs" in lines[1] and "eta~=" in lines[1]
    with _Stage("registry_load") as stage:
        assert stage.progress(
            interval=0.0, heartbeat=False).stage_seconds is None


def test_gate_eta_covers_remaining_stages_across_the_stage_transition():
    """Astra's blocker: near the end of the battery it must NOT say
    seconds remain -- `native_parity`'s ~32 minutes are still in the
    number -- and the estimate must keep carrying every not-yet-run
    profile until that stage completes."""
    clock = _FakeClock(0.0)
    stream = io.StringIO()
    gate = _GateEstimate({"corpus_load": 6420.0, "native_parity": 1920.0},
                         clock=clock)
    battery = Progress(name="corpus_load", stream=stream, clock=clock,
                       interval=0.0, gate=gate)
    battery.begin("load", 20, unit="pairs")
    battery.tick()
    first = stream.getvalue().splitlines()[0]
    # standing 107m budget + 32m native + short-stage tail
    assert "gate_eta=8460.0s" in first and "phase_eta=" not in first
    # end of early loading: one subprocess case left in this stage
    clock.advance(6300.0)
    battery.begin("fresh_process", 1, unit="subprocess")
    battery.beat()
    line = stream.getvalue().splitlines()[-1]
    assert "gate_eta=2160.0s" in line  # 120s budget + 1920 + 120 tail
    # ...and across the transition into native replay proper
    gate.complete("corpus_load", 6305.0)
    parity_stream = io.StringIO()
    parity = Progress(name="native_parity", stream=parity_stream, clock=clock,
                      interval=0.0, gate=gate)
    parity.begin("fixtures", 512)
    for _ in range(5):
        parity.tick()
        clock.advance(100.0)
    lines = parity_stream.getvalue().splitlines()
    assert "gate_eta=2040.0s" in lines[0]   # 1 sample: budget + tail, no lie
    assert "gate_eta=40680.0s" in lines[4]  # 80s/case observed * 507 + tail
    assert "phase_eta=" not in lines[1] and "phase_eta=" in lines[4]


def test_gate_eta_counts_up_while_a_blocked_unit_overruns_its_profile():
    """Profile exceeded with a case still in flight: the whole-run
    estimate grows with the wait instead of dropping out."""
    clock = _FakeClock(0.0)
    stream = io.StringIO()
    gate = _GateEstimate({"corpus_load": 100.0, "native_parity": 50.0},
                         clock=clock)
    battery = Progress(name="corpus_load", stream=stream, clock=clock,
                       interval=0.0, gate=gate)
    battery.begin("fresh_process", 1, unit="subprocess")
    clock.advance(130.0)
    battery.beat()
    clock.advance(60.0)
    battery.beat()
    lines = stream.getvalue().splitlines()
    assert "0/1 subprocess" in lines[0] and "0/1 subprocess" in lines[1]
    assert "gate_eta=200.0s" in lines[0]  # 30s over-budget + 50 + 120 tail
    assert "gate_eta=260.0s" in lines[1]  # still blocked: grows, never vanishes


def test_stage_shares_one_gate_and_registers_completion():
    gate = _GateEstimate(phase4_real._STAGE_DURATION_PROFILE_S)
    with _Stage("corpus_load") as stage:
        ticker = stage.progress(interval=0.0, heartbeat=False, gate=gate)
        assert ticker.gate is gate and ticker.stage_seconds is None
    assert "corpus_load" in gate._completed


def test_counted_ticks_once_per_completed_case_including_continues():
    rec = _Recorder()
    seen = []
    for fid in _counted(rec, ["a", "b", "c"]):
        seen.append(fid)
        continue
    assert seen == ["a", "b", "c"]
    assert rec.ticks == 3


def test_counted_leaves_the_case_that_raised_unticked():
    rec = _Recorder()
    with pytest.raises(ValueError, match="boom"):
        for fid in _counted(rec, ["a", "b"]):
            if fid == "b":
                raise ValueError("boom")
    assert rec.ticks == 1


def test_native_parity_ticks_every_declared_fixture_once(tmp_path, stub_replay):
    """Every population path -- compared, declared-but-missing, excluded,
    and loaded-but-undeclared -- completes exactly one fixture case, so the
    line counts always agree with ``len(fixture_ids)``."""
    pairs = {"a": _pair("a", "score_result"),
             "r": _pair("r", "research_replay"),
             "x": _pair("x", "score_result")}
    corpus = SimpleNamespace(
        root=tmp_path,
        index={"pairs": {
            "a": {"record_kind": "score_result"},
            "r": {"record_kind": "research_replay"},
            "m": {"record_kind": "score_result"},
        }},
        ordered_ids=sorted(pairs),
        pairs=pairs,
    )
    rec = _Recorder()
    release, _parity = phase4_real._native_parity(corpus, progress=rec)
    assert rec.begins == [("fixtures", 4)]  # a, m, r, x
    assert rec.ticks == 4
    dispositions = {row["fixture_id"]: row["disposition"]
                    for row in release["dispositions"]}
    assert dispositions == {"a": "compared", "m": "incomparable",
                            "r": "excluded", "x": "incomparable"}


def test_native_parity_without_a_ticker_prints_nothing(capsys, tmp_path,
                                                       stub_replay):
    corpus = SimpleNamespace(root=tmp_path, index={"pairs": {}},
                             ordered_ids=[], pairs={})
    release, _parity = phase4_real._native_parity(corpus)
    assert capsys.readouterr().out == ""
    assert release["population"]["expected"] == 0
