"""Checkpointing for the computed-moves pull.

The pull is the realized-move SOURCE for every ticker oquants lags on, and the
panel's event universe is bounded by it. Two properties matter enough to pin:
a second run must actually rebuild (the bug these tests exist for), and an
interrupted run must resume rather than restart 2,853 network fetches.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from engine.data.pulls import computed_moves as cm


def _events(max_date="2026-09-10", n=3):
    return pd.DataFrame({
        "ticker": ["A", "B", "C"][:n],
        "event_date": pd.to_datetime([max_date] * n),
    })


class TestBuildFingerprint:
    def test_same_inputs_give_the_same_build(self):
        a = cm._build_fingerprint(True, _events())
        b = cm._build_fingerprint(True, _events())
        assert a == b

    def test_new_prints_make_it_a_different_build(self):
        """The case that matters: events landed since the last pull, so the
        completed set from that pull must not be inherited."""
        old = cm._build_fingerprint(True, _events(max_date="2026-09-03"))
        new = cm._build_fingerprint(True, _events(max_date="2026-09-10"))
        assert old != new

    def test_widening_the_universe_makes_it_a_different_build(self):
        assert cm._build_fingerprint(False, _events()) != cm._build_fingerprint(True, _events())


class TestCheckpoint:
    def test_a_finished_ticker_is_skipped_on_resume(self, tmp_path):
        path = tmp_path / cm.CHECKPOINT_NAME
        cm._start_checkpoint(path, "abc123", total=3)
        cm._record(path, "A", "written")
        cm._record(path, "B", "no_history")

        done = cm._load_checkpoint(path, "abc123")

        assert done == {"A": "written", "B": "no_history"}

    def test_every_terminal_outcome_is_recorded_not_just_writes(self, tmp_path):
        """A name with no history is settled business. Recording only successes
        would make each resume pay the same failing network call again."""
        path = tmp_path / cm.CHECKPOINT_NAME
        cm._start_checkpoint(path, "abc123", total=3)
        for ticker, outcome in (("A", "written"), ("B", "no_history"), ("C", "too_few")):
            cm._record(path, ticker, outcome)

        assert set(cm._load_checkpoint(path, "abc123")) == {"A", "B", "C"}

    def test_a_checkpoint_from_another_build_is_not_inherited(self, tmp_path):
        path = tmp_path / cm.CHECKPOINT_NAME
        cm._start_checkpoint(path, "OLD-BUILD", total=3)
        cm._record(path, "A", "written")

        assert cm._load_checkpoint(path, "NEW-BUILD") == {}

    def test_a_corrupt_checkpoint_is_ignored_rather_than_trusted(self, tmp_path):
        """Repeating work is recoverable; silently skipping it is not."""
        path = tmp_path / cm.CHECKPOINT_NAME
        path.write_text('{"fingerprint": "abc123"}\n{not json at all\n')

        assert cm._load_checkpoint(path, "abc123") == {}

    def test_missing_checkpoint_means_start_from_the_top(self, tmp_path):
        assert cm._load_checkpoint(tmp_path / cm.CHECKPOINT_NAME, "abc123") == {}

    def test_the_checkpoint_cannot_be_read_as_a_moves_file(self, tmp_path):
        """The panel and EXP-119 both glob this directory for moves_*.json."""
        path = tmp_path / cm.CHECKPOINT_NAME
        cm._start_checkpoint(path, "abc123", total=1)

        assert list(tmp_path.glob("moves_*.json")) == []


class TestRerunActuallyRebuilds:
    def test_an_existing_moves_file_does_not_skip_the_ticker(self, tmp_path):
        """The regression this checkpointing replaced.

        The previous rule skipped any ticker whose `moves_<TK>.json` already
        existed. Since a completed build leaves one file per ticker, the NEXT
        run skipped every target and wrote nothing — so the realized moves
        stayed frozen at the date they were last built, and the panel sat at
        2026-09-03 while Tier 2 held prints through 09-10. Resume must key on
        this build's checkpoint, never on a file's mere existence.
        """
        (tmp_path / "moves_A.json").write_text(json.dumps({"ok": True}))
        done = cm._load_checkpoint(tmp_path / cm.CHECKPOINT_NAME, "abc123")

        assert "A" not in done, "a stale file must not count as finished work"
