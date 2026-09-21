"""Unit tests for ``checks.phase4_real._Stage``, the coarse-stage progress
marker added so a ~70-minute phase4_real.py run that is slow, stuck, or
dying can be told apart from one that is silently working (it emitted
nothing at all between start and finish before this change).

Covers: the START/END print format and stage name, the elapsed-seconds
figure on the END line, and -- the property that matters most, since
instrumentation must never hide a real failure -- that an exception raised
inside the ``with`` block still propagates to the caller rather than being
swallowed, with a FAILED marker printed on the way out instead of silence.

Does NOT exercise the real ~70-minute ``build_evidence`` run; that is out of
scope for a unit test (see the hand-back for what stays unverified).
"""
from __future__ import annotations

import re
import time

import pytest

from checks.phase4_real import _Stage


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
