"""Equivalence tests for event-filtered stored reads and the slim fit payload."""
import json
from types import SimpleNamespace

import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import content_hash
from engine.v2.research import magnitude_history_store as stored
from engine.v2.research.magnitude_history_table import FEATURES
from experiments import magnitude_history_store as entry
from tests.v2.research.magnitude_trial_support import (
    HISTORY_REQUEST,
    build_history,
    calendar_sessions,
    commit_history_inputs,
    history_event,
    outcome_move,
)


def _source(tmp_path, *, mixed_days=(), overrides=None, recent_day="2024-06-12"):
    events = [history_event(day) for day in ("2022-06-02", "2023-01-03", "2023-06-02", recent_day, "2024-06-17")]
    moves = [outcome_move(event, value) for event, value in zip(events, (-2., -4., 6., 10., None))]
    for index, changes in (overrides or {}).items():
        moves[index].update(changes)
    forbidden = [history_event("2023-04-03", protected=True), history_event("2025-12-03")]
    ambiguous = history_event("2023-05-03")
    forbidden.append(ambiguous)
    sibling = {**ambiguous, "event_id": "ambiguous-sibling", "session": "BMO"}
    moves += [outcome_move(event, None) for event in forbidden]
    conn, repository, snapshot = commit_history_inputs(tmp_path, [*events, *forbidden, sibling], moves,
        mixed_days=mixed_days, days=calendar_sessions("2022-05-25", "2024-06-18").days)
    return SimpleNamespace(conn=conn, repository=repository, snapshot=snapshot,
        store=repository._store, events=events, forbidden=forbidden, root=tmp_path)


@pytest.fixture
def source(tmp_path):
    source = _source(tmp_path)
    yield source
    source.conn.close()


def _read(source, output, **changes):
    return entry.read(source.repository, source.snapshot, output, **{**HISTORY_REQUEST, **changes})


def test_event_filtered_read_matches_full_read(source):
    _, output = build_history(source)
    ctx = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    full = stored.read_stored(source.repository, output, ctx)
    ids = sorted({r["event_id"] for r in full["rows"]})[:2]
    slim = stored.read_stored(source.repository, output, ctx, event_ids=ids)
    assert slim["inputs_id"] == full["inputs_id"]
    assert slim["context"] == full["context"]
    assert slim["feature_snapshot"] == full["feature_snapshot"]
    assert slim["rows"] == [{k: r[k] for k in ("event_id", "context_id", "horizon_sessions", *FEATURES)}
                            for r in full["rows"] if r["event_id"] in ids]
    assert "history_json" not in slim["rows"][0]
    empty = stored.read_stored(source.repository, output, ctx, event_ids=[])
    assert empty["rows"] == [] and empty["inputs_id"] == full["inputs_id"]


def test_full_read_inputs_id_is_content_hash_of_identity(source):
    _, output = build_history(source)
    ctx = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    full = stored.read_stored(source.repository, output, ctx)
    assert full["inputs_id"] == content_hash({"context": full["context"],
                                              "feature_snapshot": full["feature_snapshot"],
                                              "rows": full["rows"]})


def test_filtered_read_still_refuses_tampered_metadata(source, monkeypatch):
    original = stored._event_rows

    def malformed(*args):
        rows = original(*args)
        if rows[0]["event_date"] == "2024-06-17":
            rows[-1]["cutoff_id"] = "wrong"
        return rows

    monkeypatch.setattr(stored, "_event_rows", malformed)
    _, output = build_history(source)
    ctx = entry.context(source.repository, source.snapshot, **HISTORY_REQUEST)
    with pytest.raises(DataError, match="CONTRACT_MISMATCH"):
        stored.read_stored(source.repository, output, ctx, event_ids=[source.events[0]["event_id"]])


def test_fit_payload_keeps_only_read_columns():
    from experiments.magnitude_search_execution import fit_payload
    inputs = {"features": [{"event_id": "e", "context_id": "c", "horizon_sessions": 1, "f1": 1.0,
                             "history_json": "x" * 100000, "event_date": "d"}],
              "training": [{"event_id": "e", "absolute_move_pct": 1.0}]}
    payload = fit_payload(inputs, {}, {"horizons": [1]}, {"features": ["f1"]})
    decoded = json.loads(payload)
    assert decoded["inputs"]["features"] == [{"event_id": "e", "context_id": "c", "horizon_sessions": 1, "f1": 1.0}]
    assert len(payload) < 2000