"""Observed-session read memo: one daily_market pass per pinned repository and snapshot."""
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.research import magnitude_features
from tests.v2.research.magnitude_trial_support import commit_history_inputs, history_event, outcome_move


def _counting(monkeypatch):
    calls = []
    real = magnitude_features.read_table

    def counted(repository, snapshot, table_name, *args, **kwargs):
        calls.append(table_name)
        return real(repository, snapshot, table_name, *args, **kwargs)

    monkeypatch.setattr(magnitude_features, "read_table", counted)
    return calls


def _inputs(tmp_path, **changes):
    target, prior = history_event("2024-06-12"), history_event("2023-01-03")
    return commit_history_inputs(tmp_path, [target, prior], [outcome_move(prior)], **changes)


def test_pinned_repository_reads_daily_market_once_with_identical_output(tmp_path, monkeypatch):
    conn, tracked, snapshot = _inputs(tmp_path)
    try:
        pinned = Repository(conn, tracked._store, pin_snapshots=True)
        assert pinned.pins_snapshots is True and tracked.pins_snapshots is False
        calls = _counting(monkeypatch)
        first = magnitude_features._calendar_input(pinned, snapshot)
        second = magnitude_features._calendar_input(pinned, snapshot)
        assert calls == ["daily_market"]
        assert first == second
        assert first[1] is not second[1]
        unpinned = [magnitude_features._calendar_input(tracked, snapshot) for _ in range(2)]
        assert calls == ["daily_market"] * 3
        assert unpinned[0] == unpinned[1] == first
        magnitude_features._calendar_input(Repository(conn, tracked._store, pin_snapshots=True), snapshot)
        assert calls == ["daily_market"] * 4
    finally:
        conn.close()


def test_failed_read_is_never_cached(tmp_path, monkeypatch):
    conn, tracked, snapshot = _inputs(tmp_path, days=[])
    try:
        pinned = Repository(conn, tracked._store, pin_snapshots=True)
        calls = _counting(monkeypatch)
        for _ in range(2):
            with pytest.raises(DataError, match="CALENDAR_UNAVAILABLE"):
                magnitude_features._calendar_input(pinned, snapshot)
        assert calls == ["daily_market"] * 2
    finally:
        conn.close()
